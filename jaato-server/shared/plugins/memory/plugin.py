"""Memory plugin for model self-curated persistent memory across sessions.

Supports the knowledge-curation lifecycle ("The School") where agents store
raw memories during sessions, and an advisor agent later curates them into
validated knowledge or promotes them to reference entries.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.plugins.base import (
    CommandCompletion,
    HelpLines,
    PromptEnrichmentResult,
    UserCommand,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema
from .indexer import MemoryIndexer
from .models import (
    ACTIVE_MATURITIES,
    MATURITY_DISMISSED,
    MATURITY_ESCALATED,
    MATURITY_RAW,
    MATURITY_VALIDATED,
    SCOPE_PROJECT,
    SCOPE_UNIVERSAL,
    VALID_MATURITIES,
    VALID_SCOPES,
    Memory,
)
from .storage import MemoryStorage
from shared.trace import trace as _trace_write


class MemoryPlugin:
    """Plugin for model self-curated persistent memory across sessions.

    This plugin allows the model to:
    1. Store valuable explanations/insights for future reference
    2. Retrieve stored memories when relevant
    3. Build a persistent knowledge base over time

    The plugin participates in the knowledge-curation lifecycle:
    - Working agents store memories with ``maturity="raw"``
    - Prompt enrichment only surfaces *active* memories (raw, validated)
    - The advisor agent uses ``get_pending_curation`` (via storage) to
      review raw memories and transition them to validated/escalated/dismissed

    The plugin uses a two-phase retrieval system:
    - Phase 1: Prompt enrichment adds lightweight hints about active memories
    - Phase 2: Model decides whether to retrieve full content via function calling
    """

    def __init__(self):
        """Initialize the memory plugin.

        Storage is created during initialize() with a relative path template.
        When set_workspace_path() is called (by PluginRegistry broadcast),
        storage is re-created under the correct workspace directory.
        """
        self._name = "memory"
        self._storage: Optional[MemoryStorage] = None
        self._indexer: Optional[MemoryIndexer] = None
        self._agent_name: Optional[str] = None
        self._storage_path_template: str = ".jaato/memories.jsonl"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("MEMORY", msg)

    @property
    def name(self) -> str:
        """Return plugin name."""
        return self._name

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize storage backend and indexer.

        Args:
            config: Optional configuration dict with keys:
                - storage_path: Path to JSONL file (default: .jaato/memories.jsonl)
                - enrichment_limit: Max hints to show in prompt (default: 5)
        """
        config = config or {}
        self._agent_name = config.get("agent_name")
        self._storage_path_template = config.get("storage_path", ".jaato/memories.jsonl")

        self._storage = MemoryStorage(self._storage_path_template)
        self._indexer = MemoryIndexer()

        # Build index from existing memories
        existing_memories = self._storage.load_all()
        self._indexer.build_index(existing_memories)
        self._trace(f"initialize: storage_path={self._storage_path_template}, memories={len(existing_memories)}")

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        self._trace("shutdown")
        if self._indexer:
            self._indexer.clear()
        self._storage = None
        self._indexer = None

    def set_workspace_path(self, path: str) -> None:
        """Re-initialize storage under the correct workspace directory.

        Called by PluginRegistry.set_workspace_path() broadcast after
        plugin initialization. Resolves the relative storage path template
        against the workspace root so that each client's memories are
        isolated to its own workspace.
        """
        resolved = str(Path(path) / self._storage_path_template)
        self._trace(f"set_workspace_path: {path} -> {resolved}")
        self._storage = MemoryStorage(resolved)
        self._indexer = MemoryIndexer()
        existing = self._storage.load_all()
        self._indexer.build_index(existing)

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool declarations for memory operations.

        Returns:
            List of ToolSchema objects for store_memory, retrieve_memories, list_memory_tags
        """
        return [
            ToolSchema(
                name='store_memory',
                description=(
                    'Store information from this conversation for retrieval in future sessions. '
                    'Use this when you provide a comprehensive explanation, architecture overview, '
                    'or useful insight that would help in future conversations about this topic. '
                    'Only store substantial, reusable information - not ephemeral responses. '
                    'Memories are created as "raw" and will later be reviewed by the advisor '
                    'agent for potential promotion to permanent knowledge.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": (
                                "The information to store (explanation, code pattern, "
                                "architecture notes, etc.). Be comprehensive but concise."
                            )
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Brief summary of what this memory contains "
                                "(1-2 sentences max)"
                            )
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 2},
                            "description": (
                                "Specific keywords for retrieval (minimum 2 characters each). "
                                "Tags must be distinctive enough to identify THIS memory "
                                "without matching unrelated ones. "
                                "Good: 'oauth_pkce_flow', 'postgresql_indexing', 'react_hooks'. "
                                "Bad: generic words like 'code', 'error', 'fix', 'config', "
                                "or single letters."
                            )
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "Your confidence in the accuracy of this memory (0.0-1.0). "
                                "Use 0.8-1.0 for well-tested facts, 0.5-0.7 for reasonable "
                                "beliefs, 0.1-0.4 for uncertain observations. Default: 0.5"
                            )
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["project", "universal"],
                            "description": (
                                "How broadly this memory applies. 'project' for codebase-specific "
                                "knowledge, 'universal' for generally applicable insights. "
                                "Default: 'project'"
                            )
                        },
                        "evidence": {
                            "type": "string",
                            "description": (
                                "What triggered this learning — error messages, tool results, "
                                "observations, or other evidence that substantiates this memory. "
                                "Helps the advisor agent assess validity during curation."
                            )
                        }
                    },
                    "required": ["content", "description", "tags"]
                },
                category="memory",
                discoverability="core",
            ),
            ToolSchema(
                name='retrieve_memories',
                description=(
                    'Retrieve previously stored memories by tags. '
                    'Call this when you notice hints about available memories in the prompt, '
                    'or when the user asks about a topic you may have explained before.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to search for (from the hints or user query)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max number of memories to retrieve (default: 3)"
                        }
                    },
                    "required": ["tags"]
                },
                category="memory",
                discoverability="core",
            ),
            ToolSchema(
                name='list_memory_tags',
                description=(
                    'List all available memory tags to discover what has been stored. '
                    'Useful for exploring the knowledge base or finding related topics.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                category="memory",
                discoverability="core",
            )
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return tool executors.

        Returns:
            Dict mapping tool names to executor functions
        """
        return {
            "store_memory": self._execute_store,
            "retrieve_memories": self._execute_retrieve,
            "list_memory_tags": self._execute_list_tags,
            # User command
            "memory": self.execute_memory,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing memory capabilities.

        Includes guidance on the knowledge-curation lifecycle so that
        agents understand their memories will be reviewed and potentially
        promoted to permanent knowledge.

        Returns:
            Instructions for the model about memory usage
        """
        return (
            "# Persistent Memory\n\n"
            "You have access to a persistent memory system that stores information across sessions.\n\n"
            "**When to store memories:**\n"
            "- After providing comprehensive explanations of architecture, patterns, or concepts\n"
            "- When documenting project-specific conventions or decisions\n"
            "- After analyzing complex code structures or workflows\n"
            "- After discovering non-obvious behaviors, gotchas, or workarounds\n"
            "- After a significant debugging session with a hard-to-find root cause\n\n"
            "**How to use:**\n"
            "- Use `store_memory` to save valuable insights for future sessions\n"
            "- When you see memory hints in prompts (💡 **Available Memories**), "
            "use `retrieve_memories` to access stored context\n"
            "- Use `list_memory_tags` to discover what topics have been stored\n\n"
            "**Knowledge curation lifecycle:**\n"
            "Your memories are part of a learning pipeline. When you store a memory:\n"
            "1. It is created as **raw** — awaiting review by the advisor agent\n"
            "2. The advisor may **validate** it (confirmed valuable, kept as memory)\n"
            "3. The advisor may **escalate** it to a permanent reference (becomes knowledge)\n"
            "4. The advisor may **dismiss** it (incorrect, trivial, or superseded)\n\n"
            "To help the advisor assess your memories effectively:\n"
            "- Set `confidence` honestly — how sure are you this is correct?\n"
            "- Set `scope` — is this specific to this project or universally applicable?\n"
            "- Provide `evidence` — what happened that led to this learning?\n\n"
            "**Best practices:**\n"
            "- Only store substantial, reusable information (not ephemeral responses)\n"
            "- Use **specific, distinctive** tags that uniquely identify the topic. "
            "Each tag should narrow retrieval to relevant memories only. "
            "Good: 'oauth_pkce_flow', 'postgresql_indexing', 'celery_retry_policy'. "
            "Bad: generic tags like 'code', 'error', 'fix', 'bug', 'config', 'api' — "
            "these match too many unrelated memories\n"
            "- Write clear descriptions to help future retrieval\n"
            "- Include evidence: error messages, command outputs, or observations\n"
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return list of auto-approved tools.

        All memory tools are safe - read-only or self-directed writes.
        The 'memory' user command is also auto-approved since it's
        invoked directly by the user.

        Returns:
            List of tool names that don't require permission
        """
        return ["store_memory", "retrieve_memories", "list_memory_tags", "memory"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for memory management.

        Returns:
            List of UserCommand objects for the memory command
        """
        return [
            UserCommand(
                name="memory",
                description="Manage persistent memories: list, remove <id>, edit <id>",
                share_with_model=False,
            )
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for memory command arguments.

        Provides autocompletion for:
        - Subcommands: list, remove, edit, help
        - Memory IDs for remove/edit subcommands
        """
        if command != "memory":
            return []

        # Subcommand completions
        subcommands = [
            CommandCompletion("list", "List all stored memories"),
            CommandCompletion("remove", "Remove a memory by ID"),
            CommandCompletion("edit", "Edit a memory in external editor"),
            CommandCompletion("help", "Show detailed help"),
        ]

        if not args:
            return subcommands

        if len(args) == 1:
            # Partial subcommand - filter matching ones
            partial = args[0].lower()
            return [c for c in subcommands if c.value.startswith(partial)]

        if len(args) == 2:
            subcommand = args[0].lower()
            partial = args[1].lower()

            if subcommand in ("remove", "edit"):
                # Provide memory ID completions
                return self._get_memory_id_completions(partial)

        return []

    def get_memory_metadata(self) -> List[Dict[str, Any]]:
        """Return lightweight memory metadata for completion caches.

        Returns:
            List of dicts with id, description, tags, and lifecycle fields
            for each memory.
        """
        if not self._storage:
            return []
        return [
            {
                "id": m.id,
                "description": m.description,
                "tags": m.tags,
                "maturity": m.maturity,
                "confidence": m.confidence,
                "scope": m.scope,
            }
            for m in self._storage.load_all()
        ]

    def _get_memory_id_completions(self, partial: str) -> List[CommandCompletion]:
        """Get memory ID completions matching partial input."""
        if not self._storage:
            return []

        completions = []
        for mem in self._storage.load_all():
            if mem.id.lower().startswith(partial):
                # Truncate description for display
                desc = mem.description[:40] + "..." if len(mem.description) > 40 else mem.description
                completions.append(CommandCompletion(mem.id, desc))

        return completions

    # ===== Prompt Enrichment Protocol =====

    def get_enrichment_priority(self) -> int:
        """Return enrichment priority (lower = earlier).

        Memory runs at priority 80 - late in the pipeline so it can
        analyze the fully enriched prompt for memory matching.
        """
        return 80

    def subscribes_to_prompt_enrichment(self) -> bool:
        """Subscribe to enrich prompts with memory hints.

        Returns:
            True to receive prompts before they're sent to model
        """
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Analyze prompt and inject hints about available memories.

        This is the key method that:
        1. Extracts keywords/concepts from the user prompt
        2. Queries the index for matching memories
        3. Injects lightweight hints (NOT full content)

        Args:
            prompt: User's original prompt text

        Returns:
            PromptEnrichmentResult with enriched prompt and metadata
        """
        if not self._indexer or not self._storage:
            return PromptEnrichmentResult(
                prompt=prompt,
                metadata={"error": "Plugin not initialized"}
            )

        # Extract potential keywords
        keywords = self._indexer.extract_keywords(prompt)

        # Find matching memories (just metadata, not full content)
        matches = self._indexer.find_matches(keywords, limit=5)

        if not matches:
            return PromptEnrichmentResult(
                prompt=prompt,
                metadata={"memory_matches": 0}
            )

        # Build hint section
        hint_lines = [
            "",
            "💡 **Available Memories** (use retrieve_memories to access):"
        ]
        for memory_meta in matches:
            tags_str = ", ".join(memory_meta.tags)
            hint_lines.append(f"  - [{tags_str}]: {memory_meta.description}")

        enriched_prompt = prompt + "\n" + "\n".join(hint_lines)

        # Collect unique tags from matched memories
        matched_tags = []
        seen_tags = set()
        for m in matches:
            for tag in m.tags:
                tag_lower = tag.lower()
                if tag_lower not in seen_tags:
                    seen_tags.add(tag_lower)
                    matched_tags.append(tag)

        # Build notification message with matched tags
        tag_summary = ", ".join(f'"{t}"' for t in matched_tags[:3])
        if len(matched_tags) > 3:
            tag_summary += f" +{len(matched_tags) - 3} more"

        return PromptEnrichmentResult(
            prompt=enriched_prompt,
            metadata={
                "memory_matches": len(matches),
                "matched_ids": [m.id for m in matches],
                "trigger_keywords": matched_tags,
                "notification": {
                    "message": f"added context about {len(matches)} memories (tags: {tag_summary})"
                }
            }
        )

    # ===== Tool Executors =====

    def _execute_store(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute store_memory tool.

        Creates a new memory with ``maturity="raw"``.  The optional
        ``confidence``, ``scope``, and ``evidence`` fields help the
        advisor agent during later curation.

        Args:
            args: Tool arguments (content, description, tags, and optional
                confidence, scope, evidence)

        Returns:
            Result dict with status and memory_id
        """
        description = args.get("description", "")
        tags = args.get("tags", [])
        self._trace(f"store_memory: description={description!r}, tags={tags}")
        if not self._storage or not self._indexer:
            return {
                "status": "error",
                "message": "Memory plugin not initialized"
            }

        # Validate and normalize tags: strip whitespace, reject single-char tags
        raw_tags = args.get("tags", [])
        valid_tags = [
            tag.strip() for tag in raw_tags
            if isinstance(tag, str) and len(tag.strip()) >= 2
        ]
        if not valid_tags:
            return {
                "status": "error",
                "message": (
                    "All tags were rejected — each tag must be a meaningful "
                    "word or phrase (at least 2 characters). "
                    f"Received: {raw_tags!r}"
                )
            }

        # Validate confidence (clamp to 0.0-1.0)
        confidence = args.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5

        # Validate scope
        scope = args.get("scope", SCOPE_PROJECT)
        if scope not in VALID_SCOPES:
            scope = SCOPE_PROJECT

        # Create memory object — always starts as raw
        memory = Memory(
            id=f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]}",
            content=args["content"],
            description=args["description"],
            tags=valid_tags,
            timestamp=datetime.now().isoformat(),
            usage_count=0,
            maturity=MATURITY_RAW,
            confidence=confidence,
            scope=scope,
            evidence=args.get("evidence"),
            source_agent=self._agent_name,
        )

        # Save to storage
        self._storage.save(memory)

        # Update index
        self._indexer.index_memory(memory)

        return {
            "status": "success",
            "memory_id": memory.id,
            "message": f"Stored memory: {memory.description}",
            "tags": memory.tags,
            "maturity": memory.maturity,
            "confidence": memory.confidence,
            "scope": memory.scope,
        }

    def _execute_retrieve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieve_memories tool.

        Only returns active memories (raw, validated) by default.

        Args:
            args: Tool arguments (tags, limit)

        Returns:
            Result dict with memories list including lifecycle metadata
        """
        tags = args.get("tags", [])
        limit = args.get("limit", 3)
        self._trace(f"retrieve_memories: tags={tags}, limit={limit}")
        if not self._storage:
            return {
                "status": "error",
                "message": "Memory plugin not initialized"
            }

        tags = args["tags"]
        limit = args.get("limit", 3)

        # Search storage by tags (active_only=True by default)
        memories = self._storage.search_by_tags(tags, limit=limit)

        if not memories:
            return {
                "status": "no_results",
                "message": f"No memories found for tags: {tags}"
            }

        # Update usage statistics
        for mem in memories:
            mem.usage_count += 1
            mem.last_accessed = datetime.now().isoformat()
            self._storage.update(mem)

        return {
            "status": "success",
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "description": m.description,
                    "content": m.content,
                    "tags": m.tags,
                    "stored": m.timestamp,
                    "usage_count": m.usage_count,
                    "maturity": m.maturity,
                    "confidence": m.confidence,
                    "scope": m.scope,
                }
                for m in memories
            ]
        }

    def _execute_list_tags(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_memory_tags tool.

        Args:
            args: Tool arguments (none)

        Returns:
            Result dict with all tags
        """
        self._trace("list_memory_tags")
        if not self._indexer:
            return {
                "status": "error",
                "message": "Memory plugin not initialized"
            }

        tags = self._indexer.get_all_tags()
        memory_count = self._indexer.get_memory_count()

        return {
            "status": "success",
            "tags": sorted(tags),
            "count": len(tags),
            "memory_count": memory_count,
            "message": f"Found {memory_count} memories with {len(tags)} unique tags"
        }

    # ===== User Command Executor =====

    def execute_memory(self, args: Dict[str, Any]) -> str:
        """Execute the memory user command.

        Subcommands:
            list              - List all stored memories
            remove <id>       - Remove a memory by ID
            edit <id>         - Edit a memory in external editor
            help              - Show detailed help

        Args:
            args: Dict with 'args' key containing list of command arguments

        Returns:
            Formatted string output for display to user
        """
        cmd_args = args.get("args", [])

        if not cmd_args:
            return self._memory_list()

        subcommand = cmd_args[0].lower()

        if subcommand == "list":
            return self._memory_list()
        elif subcommand == "remove":
            if len(cmd_args) < 2:
                return "Usage: memory remove <memory_id>"
            memory_id = cmd_args[1]
            return self._memory_remove(memory_id)
        elif subcommand == "edit":
            if len(cmd_args) < 2:
                return "Usage: memory edit <memory_id>"
            memory_id = cmd_args[1]
            return self._memory_edit(memory_id)
        elif subcommand == "help":
            return self._memory_help()
        else:
            return (
                f"Unknown subcommand: {subcommand}\n"
                "Usage: memory <list|remove|edit|help>\n"
                "  list              - List all stored memories\n"
                "  remove <id>       - Remove a memory by ID\n"
                "  edit <id>         - Edit a memory in external editor\n"
                "  help              - Show detailed help"
            )

    def _memory_list(self) -> HelpLines:
        """List all stored memories with lifecycle metadata.

        Returns HelpLines for pager display (same pattern as session list).
        Shows maturity, confidence, and scope alongside existing metadata.
        """
        if not self._storage:
            return HelpLines(lines=[("Error: Memory plugin not initialized.", "error")])

        memories = self._storage.load_all()

        if not memories:
            return HelpLines(lines=[("No memories stored yet.", "dim")])

        # Group by maturity for summary
        maturity_counts = self._storage.count_by_maturity()

        lines = []
        lines.append(("Stored Memories", "bold"))
        lines.append(("═" * 15, "bold"))

        # Show maturity summary
        summary_parts = []
        for mat in (MATURITY_RAW, MATURITY_VALIDATED, MATURITY_ESCALATED, MATURITY_DISMISSED):
            count = maturity_counts.get(mat, 0)
            if count > 0:
                summary_parts.append(f"{mat}: {count}")
        if summary_parts:
            lines.append((f"  ({', '.join(summary_parts)})", "dim"))
        lines.append(("", ""))

        for mem in memories:
            tags_str = ", ".join(mem.tags[:3])
            if len(mem.tags) > 3:
                tags_str += f" +{len(mem.tags) - 3} more"

            # Maturity indicator
            maturity_icon = {
                MATURITY_RAW: "○",
                MATURITY_VALIDATED: "◑",
                MATURITY_ESCALATED: "●",
                MATURITY_DISMISSED: "✗",
            }.get(mem.maturity, "?")

            lines.append((f"{maturity_icon} ID: {mem.id}", ""))
            lines.append((f"  Description: {mem.description}", "dim"))
            lines.append((f"  Tags: {tags_str}", "dim"))
            lines.append((f"  Created: {mem.timestamp[:10]}  |  Maturity: {mem.maturity}  |  Confidence: {mem.confidence:.0%}  |  Scope: {mem.scope}", "dim"))
            lines.append((f"  Used: {mem.usage_count} times", "dim"))
            if mem.source_agent:
                lines.append((f"  Source: {mem.source_agent}", "dim"))
            lines.append(("", ""))

        lines.append((f"Total: {len(memories)} memories", "bold"))
        return HelpLines(lines=lines)

    def _memory_remove(self, memory_id: str) -> str:
        """Remove a memory by ID."""
        if not self._storage or not self._indexer:
            return "Error: Memory plugin not initialized."

        # Check if memory exists first
        memory = self._storage.get_by_id(memory_id)
        if not memory:
            return f"Error: Memory not found: {memory_id}"

        # Delete from storage
        deleted = self._storage.delete(memory_id)

        if deleted:
            # Rebuild index after deletion
            existing_memories = self._storage.load_all()
            self._indexer.clear()
            self._indexer.build_index(existing_memories)
            return f"Removed memory: {memory_id}\n  Was: {memory.description}"
        else:
            return f"Error: Failed to remove memory: {memory_id}"

    def _memory_edit(self, memory_id: str) -> str:
        """Edit a memory in external editor."""
        if not self._storage or not self._indexer:
            return "Error: Memory plugin not initialized."

        # Get the memory
        memory = self._storage.get_by_id(memory_id)
        if not memory:
            return f"Error: Memory not found: {memory_id}"

        # Get editor
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"

        # Prepare memory as YAML for editing (including lifecycle fields)
        memory_dict = {
            "description": memory.description,
            "content": memory.content,
            "tags": memory.tags,
            "maturity": memory.maturity,
            "confidence": memory.confidence,
            "scope": memory.scope,
            "evidence": memory.evidence,
        }

        # Create temp file with memory content
        try:
            import yaml
            HAS_YAML = True
        except ImportError:
            HAS_YAML = False

        try:
            # Format as YAML or JSON
            if HAS_YAML:
                content = (
                    f"# Edit memory: {memory_id}\n"
                    f"# Modify the fields below and save to update the memory.\n"
                    f"# Close without saving to cancel.\n"
                    f"#\n"
                    f"# Fields:\n"
                    f"#   description: Brief summary (1-2 sentences)\n"
                    f"#   content: Full content/explanation\n"
                    f"#   tags: List of keywords for retrieval\n"
                    f"#   maturity: raw | validated | escalated | dismissed\n"
                    f"#   confidence: 0.0 to 1.0\n"
                    f"#   scope: project | universal\n"
                    f"#   evidence: What triggered this learning (optional)\n"
                    f"\n"
                )
                import yaml
                content += yaml.safe_dump(
                    memory_dict,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
                suffix = ".yaml"
            else:
                content = (
                    f"// Edit memory: {memory_id}\n"
                    f"// Modify the fields below and save to update the memory.\n"
                    f"// Close without saving to cancel.\n"
                    f"\n"
                )
                content += json.dumps(memory_dict, indent=2, ensure_ascii=False)
                suffix = ".json"

            # Write to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=suffix,
                delete=False,
                encoding='utf-8',
            ) as f:
                f.write(content)
                temp_path = f.name

            original_content = content

            # Open in editor
            result = subprocess.run([editor, temp_path], check=False)

            if result.returncode != 0:
                os.unlink(temp_path)
                return f"Editor exited with code {result.returncode}. Edit cancelled."

            # Read back edited content
            with open(temp_path, 'r', encoding='utf-8') as f:
                edited_content = f.read()

            os.unlink(temp_path)

            # Check if content was modified
            if edited_content.strip() == original_content.strip():
                return "No changes made."

            # Parse edited content
            # Strip comment lines
            lines = []
            for line in edited_content.split('\n'):
                stripped = line.strip()
                if not stripped.startswith('#') and not stripped.startswith('//'):
                    lines.append(line)
            clean_content = '\n'.join(lines)

            try:
                if HAS_YAML:
                    parsed = yaml.safe_load(clean_content)
                else:
                    parsed = json.loads(clean_content)
            except Exception as e:
                return f"Error parsing edited content: {e}\nEdit cancelled."

            # Validate schema
            validation_error = self._validate_memory_schema(parsed)
            if validation_error:
                return f"Validation error: {validation_error}\nEdit cancelled."

            # Update memory (core + lifecycle fields)
            memory.description = parsed["description"]
            memory.content = parsed["content"]
            memory.tags = parsed["tags"]
            if "maturity" in parsed:
                memory.maturity = parsed["maturity"]
            if "confidence" in parsed:
                memory.confidence = float(parsed["confidence"])
            if "scope" in parsed:
                memory.scope = parsed["scope"]
            if "evidence" in parsed:
                memory.evidence = parsed["evidence"]

            # Save updated memory
            self._storage.update(memory)

            # Rebuild index
            existing_memories = self._storage.load_all()
            self._indexer.clear()
            self._indexer.build_index(existing_memories)

            return f"Updated memory: {memory_id}\n  Description: {memory.description}"

        except Exception as e:
            # Clean up temp file if it exists
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            return f"Error editing memory: {e}"

    def _validate_memory_schema(self, data: Dict[str, Any]) -> Optional[str]:
        """Validate that edited memory data conforms to schema.

        Validates both the original core fields and the lifecycle fields
        added for the knowledge-curation system.

        Args:
            data: Parsed memory data dict

        Returns:
            Error message if invalid, None if valid
        """
        # Required fields
        required_fields = ["description", "content", "tags"]
        for fld in required_fields:
            if fld not in data:
                return f"Missing required field: {fld}"

        # Type validation — core fields
        if not isinstance(data["description"], str):
            return "description must be a string"
        if not isinstance(data["content"], str):
            return "content must be a string"
        if not isinstance(data["tags"], list):
            return "tags must be a list"
        if not all(isinstance(tag, str) for tag in data["tags"]):
            return "all tags must be strings"

        # Non-empty validation
        if not data["description"].strip():
            return "description cannot be empty"
        if not data["content"].strip():
            return "content cannot be empty"
        if not data["tags"]:
            return "tags cannot be empty"

        # Tag quality: each tag must be at least 2 characters
        short_tags = [tag for tag in data["tags"] if len(tag.strip()) < 2]
        if short_tags:
            return (
                f"tags must be meaningful words (at least 2 characters each), "
                f"got: {short_tags!r}"
            )

        # Lifecycle field validation (optional in schema, validated when present)
        if "maturity" in data:
            if data["maturity"] not in VALID_MATURITIES:
                return (
                    f"maturity must be one of {sorted(VALID_MATURITIES)}, "
                    f"got: {data['maturity']!r}"
                )

        if "confidence" in data:
            try:
                conf = float(data["confidence"])
                if not (0.0 <= conf <= 1.0):
                    return "confidence must be between 0.0 and 1.0"
            except (TypeError, ValueError):
                return f"confidence must be a number, got: {data['confidence']!r}"

        if "scope" in data:
            if data["scope"] not in VALID_SCOPES:
                return (
                    f"scope must be one of {sorted(VALID_SCOPES)}, "
                    f"got: {data['scope']!r}"
                )

        if "evidence" in data:
            if data["evidence"] is not None and not isinstance(data["evidence"], str):
                return "evidence must be a string or null"

        return None

    def _memory_help(self) -> HelpLines:
        """Show detailed help for the memory command."""
        return HelpLines(lines=[
            ("Memory Command", "bold"),
            ("", ""),
            ("Manage persistent memories stored by the AI. Memories persist across", ""),
            ("sessions and help the AI recall context, patterns, and lessons learned.", ""),
            ("", ""),
            ("Memories go through a knowledge-curation lifecycle:", ""),
            ("  raw -> validated -> escalated (promoted to reference)", "dim"),
            ("               \\-> dismissed (rejected by advisor)", "dim"),
            ("", ""),
            ("USAGE", "bold"),
            ("    memory [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list              List all stored memories with metadata", "dim"),
            ("                      Shows ID, description, tags, maturity, confidence", "dim"),
            ("", ""),
            ("    remove <id>       Remove a memory by its ID", "dim"),
            ("                      The memory will be permanently deleted", "dim"),
            ("", ""),
            ("    edit <id>         Edit a memory in your external editor ($EDITOR)", "dim"),
            ("                      Opens the memory in YAML format for editing", "dim"),
            ("                      Validates the schema on save", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    memory                         List all memories (default)", "dim"),
            ("    memory list                    List all memories", "dim"),
            ("    memory remove mem_20240101_... Remove a specific memory", "dim"),
            ("    memory edit mem_20240101_...   Edit a specific memory", "dim"),
            ("", ""),
            ("EDIT FORMAT", "bold"),
            ("    When editing, the memory is presented in YAML format with:", ""),
            ("      description: Brief summary of the memory", "dim"),
            ("      content: Full content/explanation", "dim"),
            ("      tags: List of keywords for retrieval", "dim"),
            ("      maturity: raw | validated | escalated | dismissed", "dim"),
            ("      confidence: 0.0 to 1.0 (accuracy self-assessment)", "dim"),
            ("      scope: project | universal", "dim"),
            ("      evidence: What triggered this learning (optional)", "dim"),
            ("", ""),
            ("    Lines starting with # are comments and will be ignored.", ""),
            ("", ""),
            ("MATURITY LIFECYCLE", "bold"),
            ("    ○ raw          Fresh from agent, awaiting advisor review", "dim"),
            ("    ◑ validated    Advisor confirmed valuable, kept as memory", "dim"),
            ("    ● escalated    Promoted to permanent reference (knowledge)", "dim"),
            ("    ✗ dismissed    Rejected by advisor (incorrect/trivial)", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - Memories are stored in .jaato/memories.jsonl", "dim"),
            ("    - Each memory has a unique ID starting with 'mem_'", "dim"),
            ("    - Use Tab completion for memory IDs in remove/edit", "dim"),
            ("    - Only active memories (raw, validated) appear in prompt hints", "dim"),
        ])


def create_plugin() -> MemoryPlugin:
    """Factory function to create the memory plugin instance.

    Returns:
        MemoryPlugin instance
    """
    return MemoryPlugin()
