"""Memory plugin for model self-curated persistent memory across sessions."""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..base import PromptEnrichmentResult
from ..model_provider.types import ToolSchema
from .indexer import MemoryIndexer
from .models import Memory
from .storage import MemoryStorage


class MemoryPlugin:
    """Plugin for model self-curated persistent memory across sessions.

    This plugin allows the model to:
    1. Store valuable explanations/insights for future reference
    2. Retrieve stored memories when relevant
    3. Build a persistent knowledge base over time

    The plugin uses a two-phase retrieval system:
    - Phase 1: Prompt enrichment adds lightweight hints about available memories
    - Phase 2: Model decides whether to retrieve full content via function calling
    """

    def __init__(self):
        """Initialize the memory plugin."""
        self._name = "memory"
        self._storage: Optional[MemoryStorage] = None
        self._indexer: Optional[MemoryIndexer] = None

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
        storage_path = config.get("storage_path", ".jaato/memories.jsonl")

        self._storage = MemoryStorage(storage_path)
        self._indexer = MemoryIndexer()

        # Build index from existing memories
        existing_memories = self._storage.load_all()
        self._indexer.build_index(existing_memories)

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        if self._indexer:
            self._indexer.clear()
        self._storage = None
        self._indexer = None

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
                    'Only store substantial, reusable information - not ephemeral responses.'
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
                            "items": {"type": "string"},
                            "description": (
                                "Keywords for retrieval. Use specific, searchable terms "
                                "(e.g., ['authentication', 'oauth', 'jwt'] not ['auth stuff'])"
                            )
                        }
                    },
                    "required": ["content", "description", "tags"]
                }
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
                }
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
                }
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
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing memory capabilities.

        Returns:
            Instructions for the model about memory usage
        """
        return (
            "# Persistent Memory\n\n"
            "You have access to a persistent memory system that stores information across sessions.\n\n"
            "**When to store memories:**\n"
            "- After providing comprehensive explanations of architecture, patterns, or concepts\n"
            "- When documenting project-specific conventions or decisions\n"
            "- After analyzing complex code structures or workflows\n\n"
            "**How to use:**\n"
            "- Use `store_memory` to save valuable insights for future sessions\n"
            "- When you see memory hints in prompts (💡 **Available Memories**), "
            "use `retrieve_memories` to access stored context\n"
            "- Use `list_memory_tags` to discover what topics have been stored\n\n"
            "**Best practices:**\n"
            "- Only store substantial, reusable information (not ephemeral responses)\n"
            "- Use specific, searchable tags (e.g., 'database_schema', 'api_auth')\n"
            "- Write clear descriptions to help future retrieval\n"
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return list of auto-approved tools.

        All memory tools are safe - read-only or self-directed writes.

        Returns:
            List of tool names that don't require permission
        """
        return ["store_memory", "retrieve_memories", "list_memory_tags"]

    def get_user_commands(self) -> List:
        """Return user-facing commands.

        Memory plugin is model-driven, no direct user commands needed.

        Returns:
            Empty list
        """
        return []

    # ===== Prompt Enrichment Protocol =====

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

        return PromptEnrichmentResult(
            prompt=enriched_prompt,
            metadata={
                "memory_matches": len(matches),
                "matched_ids": [m.id for m in matches]
            }
        )

    # ===== Tool Executors =====

    def _execute_store(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute store_memory tool.

        Args:
            args: Tool arguments (content, description, tags)

        Returns:
            Result dict with status and memory_id
        """
        if not self._storage or not self._indexer:
            return {
                "status": "error",
                "message": "Memory plugin not initialized"
            }

        # Create memory object
        memory = Memory(
            id=f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]}",
            content=args["content"],
            description=args["description"],
            tags=args["tags"],
            timestamp=datetime.now().isoformat(),
            usage_count=0
        )

        # Save to storage
        self._storage.save(memory)

        # Update index
        self._indexer.index_memory(memory)

        return {
            "status": "success",
            "memory_id": memory.id,
            "message": f"Stored memory: {memory.description}",
            "tags": memory.tags
        }

    def _execute_retrieve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieve_memories tool.

        Args:
            args: Tool arguments (tags, limit)

        Returns:
            Result dict with memories list
        """
        if not self._storage:
            return {
                "status": "error",
                "message": "Memory plugin not initialized"
            }

        tags = args["tags"]
        limit = args.get("limit", 3)

        # Search storage by tags
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
                    "usage_count": m.usage_count
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


def create_plugin() -> MemoryPlugin:
    """Factory function to create the memory plugin instance.

    Returns:
        MemoryPlugin instance
    """
    return MemoryPlugin()
