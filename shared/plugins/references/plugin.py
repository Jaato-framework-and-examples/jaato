"""References plugin for managing documentation source injection.

This plugin maintains a catalog of reference sources (documentation, specs,
guides, etc.) and handles:
- AUTO sources: Included in system instructions, model fetches them at startup
- SELECTABLE sources: User chooses which to include via interactive selection

The model is responsible for fetching content using appropriate tools (CLI, MCP, etc.).
This plugin only manages the catalog and user interaction.
"""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..model_provider.types import ToolSchema
from ..subagent.config import expand_variables

from .models import ReferenceSource, InjectionMode, SourceType
from .channels import SelectionChannel, ConsoleSelectionChannel, QueueSelectionChannel, create_channel
from .config_loader import load_config, ReferencesConfig, resolve_source_paths
from ..base import UserCommand, CommandCompletion


# Maximum depth for transitive reference resolution to prevent runaway recursion
MAX_TRANSITIVE_DEPTH = 10


class ReferencesPlugin:
    """Plugin for managing reference source injection into model context.

    The plugin maintains a catalog of reference sources and:
    - AUTO sources: Included in system instructions for model to fetch
    - SELECTABLE sources: User chooses via channel (console/webhook/file)

    The model uses existing tools (CLI, MCP, URL fetch) to retrieve content.
    This plugin only provides metadata and handles user selection.
    """

    def __init__(self):
        self._name = "references"
        self._config: Optional[ReferencesConfig] = None
        self._sources: List[ReferenceSource] = []
        self._channel: Optional[SelectionChannel] = None
        self._selected_source_ids: List[str] = []  # User-selected during session
        self._exclude_tools: List[str] = []  # Tools to exclude from schema
        self._initialized = False
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Selection lifecycle hooks for UI integration
        self._on_selection_requested: Optional[Callable[[str, List[str]], None]] = None
        self._on_selection_resolved: Optional[Callable[[str, List[str]], None]] = None

    @property
    def name(self) -> str:
        return self._name

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging.

        Uses JAATO_TRACE_LOG env var, or defaults to /tmp/rich_client_trace.log.
        Silently skips if trace file cannot be written.
        """
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [REFERENCES{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass  # Silently skip if trace file cannot be written

    def _get_reference_content(self, source: ReferenceSource) -> Optional[str]:
        """Get the content of a reference source for transitive detection.

        Only LOCAL and INLINE sources are supported for content extraction.
        URL and MCP sources would require external fetching which is deferred
        to the model.

        Args:
            source: The reference source to get content from.

        Returns:
            The content string if available, None otherwise.
        """
        if source.type == SourceType.INLINE:
            return source.content

        if source.type == SourceType.LOCAL:
            # Use resolved_path if available, otherwise original path
            file_path = source.resolved_path or source.path
            if not file_path:
                return None

            path_obj = Path(file_path)

            # Handle directory sources - concatenate all file contents
            if path_obj.is_dir():
                contents: List[str] = []
                # Include common documentation file extensions
                doc_extensions = (
                    '.md', '.txt', '.json', '.yaml', '.yml',
                    '.html', '.htm', '.rst', '.adoc'
                )
                try:
                    for item in sorted(path_obj.rglob("*")):
                        if item.is_file():
                            # Include files with doc extensions or README files (any extension)
                            is_doc_ext = item.suffix.lower() in doc_extensions
                            is_readme = item.stem.upper() == 'README'
                            if is_doc_ext or is_readme:
                                try:
                                    contents.append(item.read_text(encoding='utf-8'))
                                except (IOError, OSError, UnicodeDecodeError):
                                    pass  # Skip unreadable files
                except (PermissionError, OSError):
                    pass
                return "\n".join(contents) if contents else None

            # Handle regular file
            if path_obj.is_file():
                try:
                    return path_obj.read_text(encoding='utf-8')
                except (IOError, OSError, UnicodeDecodeError):
                    return None

        return None

    def _find_referenced_ids(self, content: str, catalog_ids: Set[str]) -> Set[str]:
        """Find reference IDs mentioned in content.

        Searches for catalog IDs appearing as whole words in the content.
        This handles common patterns like:
        - Direct ID mentions: "skill-001-circuit-breaker"
        - Reference syntax: "@ref:skill-001" or "[[skill-001]]"
        - Prose mentions: "see skill-001-circuit-breaker for details"

        Args:
            content: The content to search for reference mentions.
            catalog_ids: Set of valid reference IDs from the catalog.

        Returns:
            Set of reference IDs found in the content.
        """
        found_ids: Set[str] = set()

        for ref_id in catalog_ids:
            # Escape special regex characters in the ID
            escaped_id = re.escape(ref_id)
            # Match as a whole word (with word boundaries or common delimiters)
            # Pattern allows for common reference syntaxes like @ref:id, [[id]], `id`
            pattern = rf'(?:^|[\s\[\]`@:,;()\'"{{}}])({escaped_id})(?:[\s\[\]`@:,;()\'"{{}}]|$)'
            if re.search(pattern, content, re.MULTILINE):
                found_ids.add(ref_id)

        return found_ids

    def _resolve_transitive_references(
        self,
        initial_ids: List[str],
        catalog_by_id: Dict[str, ReferenceSource],
        max_depth: int = MAX_TRANSITIVE_DEPTH
    ) -> List[str]:
        """Resolve transitive references from pre-selected sources.

        Starting from the initially selected reference IDs, reads their content
        and finds mentions of other references in the catalog. Recursively
        resolves those references until no new references are found or max
        depth is reached.

        Args:
            initial_ids: List of initially selected/pre-selected reference IDs.
            catalog_by_id: Mapping of reference ID to ReferenceSource.
            max_depth: Maximum recursion depth to prevent runaway resolution.

        Returns:
            List of all resolved reference IDs (initial + transitively discovered),
            in order of discovery (initial IDs first, then discovered ones).
        """
        if not initial_ids:
            return []

        self._trace(f"transitive: starting from {initial_ids}")

        # Track all resolved IDs and order of discovery
        resolved_ids: List[str] = list(initial_ids)
        resolved_set: Set[str] = set(initial_ids)

        # IDs to process in this iteration
        pending: Set[str] = set(initial_ids)
        catalog_ids = set(catalog_by_id.keys())

        for depth in range(max_depth):
            if not pending:
                break

            newly_found: Set[str] = set()
            self._trace(f"transitive: [depth={depth}] scanning {sorted(pending)}")

            for ref_id in pending:
                source = catalog_by_id.get(ref_id)
                if not source:
                    continue

                # Get content from the source
                content = self._get_reference_content(source)
                if not content:
                    self._trace(f"transitive:   '{ref_id}' -> no content (type={source.type.value})")
                    continue

                self._trace(f"transitive:   '{ref_id}' -> {len(content)} chars")

                # Find references mentioned in content
                mentioned_ids = self._find_referenced_ids(content, catalog_ids)

                # Filter to only newly discovered ones
                new_mentions = mentioned_ids - resolved_set - {ref_id}
                if new_mentions:
                    self._trace(f"transitive:   '{ref_id}' => {sorted(new_mentions)}")
                    for mentioned_id in new_mentions:
                        newly_found.add(mentioned_id)
                        resolved_set.add(mentioned_id)
                        resolved_ids.append(mentioned_id)

            # Next iteration processes newly found IDs
            pending = newly_found

        if pending:
            self._trace(
                f"transitive: max depth {max_depth} reached, {len(pending)} unresolved"
            )

        # Final summary
        transitive_count = len(resolved_ids) - len(initial_ids)
        if transitive_count > 0:
            transitive_ids = resolved_ids[len(initial_ids):]
            self._trace(f"transitive: added {transitive_count}: {transitive_ids}")
        else:
            self._trace("transitive: no additional references found")

        return resolved_ids

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict. If not provided, loads from
                   file specified by REFERENCES_CONFIG_PATH or default locations.

                   Config options:
                   - config_path: Path to references.json file
                   - channel_type: Type of channel ("console", "webhook", "file")
                   - channel_config: Configuration for the channel
                   - sources: Inline sources list (overrides file)
                   - preselected: List of source IDs to pre-select at startup.
                                  Sources are looked up from the master catalog
                                  and automatically added to available sources.
                   - transitive_injection: Enable transitive reference detection (default: True).
                                           When enabled, pre-selected references are scanned for
                                           mentions of other catalog references, which are then
                                           automatically injected.
                   - exclude_tools: List of tool names to exclude (e.g., ["selectReferences"])
        """
        config = config or {}

        # Expand variables in config values (e.g., ${projectPath}, ${workspaceRoot})
        config = expand_variables(config) if config else {}

        # Extract agent name for trace logging
        self._agent_name = config.get("agent_name")

        # Try to load from file first (master catalog)
        config_path = config.get("config_path")
        try:
            self._config = load_config(config_path)
        except FileNotFoundError:
            # Use defaults
            self._config = ReferencesConfig()

        # Build ID -> source lookup from master catalog
        catalog_by_id = {s.id: s for s in self._config.sources}

        # Allow inline sources override
        if "sources" in config:
            resolved_sources = []
            for s in config["sources"]:
                if isinstance(s, dict):
                    resolved_sources.append(ReferenceSource.from_dict(s))
                elif isinstance(s, str):
                    # Look up by ID from master catalog
                    if s in catalog_by_id:
                        resolved_sources.append(catalog_by_id[s])
                    else:
                        print(f"Warning: Source ID '{s}' not found in master catalog")
                else:
                    resolved_sources.append(s)

            # Resolve relative paths for inline sources against provided base or CWD
            # Make paths relative to project root (not CWD which may differ)
            inline_base_path = config.get("base_path", os.getcwd())
            # Detect project root: if base_path contains .jaato, use its parent
            base_path_obj = Path(inline_base_path).resolve()
            if '.jaato' in base_path_obj.parts:
                jaato_idx = base_path_obj.parts.index('.jaato')
                project_root = str(Path(*base_path_obj.parts[:jaato_idx]))
            else:
                project_root = str(base_path_obj)
            resolve_source_paths(resolved_sources, inline_base_path, relative_to=project_root)
            self._sources = resolved_sources
        else:
            self._sources = self._config.sources

        # Handle preselected - look up from catalog and add to sources if needed
        preselected = config.get("preselected", [])
        if preselected:
            current_ids = {s.id for s in self._sources}
            for sid in preselected:
                if sid not in current_ids and sid in catalog_by_id:
                    # Add from master catalog
                    self._sources.append(catalog_by_id[sid])
                    current_ids.add(sid)

        # Initialize channel
        channel_type = config.get("channel_type") or self._config.channel_type
        channel_config = config.get("channel_config", {})

        # Set timeout from config
        if "timeout" not in channel_config:
            channel_config["timeout"] = self._config.channel_timeout

        # Set type-specific config
        if channel_type == "webhook" and "endpoint" not in channel_config:
            if self._config.channel_endpoint:
                channel_config["endpoint"] = self._config.channel_endpoint

        if channel_type == "file" and "base_path" not in channel_config:
            if self._config.channel_base_path:
                channel_config["base_path"] = self._config.channel_base_path

        try:
            self._channel = create_channel(channel_type, channel_config)
        except (ValueError, RuntimeError) as e:
            # Fall back to console channel if configured channel fails
            print(f"Warning: Failed to initialize {channel_type} channel: {e}")
            print("Falling back to console channel")
            self._channel = ConsoleSelectionChannel()
            self._channel.initialize({})

        # Initialize selected sources from preselected config
        # (sources were already added above, now just validate and track IDs)
        if preselected:
            available_ids = {s.id for s in self._sources}
            valid_preselected = [sid for sid in preselected if sid in available_ids]
            invalid = set(preselected) - available_ids - set(catalog_by_id.keys())
            if invalid:
                print(f"Warning: Preselected reference IDs not found: {invalid}")
            self._selected_source_ids = valid_preselected
        else:
            self._selected_source_ids = []

        # Resolve transitive references if enabled (default: True)
        # This scans pre-selected references for mentions of other catalog references
        # and automatically adds them to the selected set
        transitive_enabled = config.get("transitive_injection", True)
        if transitive_enabled and self._selected_source_ids:
            # Build complete catalog including inline sources
            full_catalog = dict(catalog_by_id)
            for source in self._sources:
                if source.id not in full_catalog:
                    full_catalog[source.id] = source

            # Resolve transitive references
            all_resolved = self._resolve_transitive_references(
                self._selected_source_ids,
                full_catalog
            )

            # Add newly discovered sources to self._sources and self._selected_source_ids
            current_source_ids = {s.id for s in self._sources}
            for ref_id in all_resolved:
                if ref_id not in self._selected_source_ids:
                    self._selected_source_ids.append(ref_id)
                # Ensure source is in self._sources
                if ref_id not in current_source_ids and ref_id in full_catalog:
                    self._sources.append(full_catalog[ref_id])
                    current_source_ids.add(ref_id)

            # Log transitive injection results
            transitive_count = len(all_resolved) - len(valid_preselected) if preselected else 0
            if transitive_count > 0:
                transitive_ids = [rid for rid in all_resolved if rid not in valid_preselected]
                self._trace(f"transitive: injected {transitive_count} additional: {transitive_ids}")

        # Capture excluded tools
        self._exclude_tools = config.get("exclude_tools", [])
        self._initialized = True

        # Trace logging for debugging
        channel_type = config.get("channel_type") or self._config.channel_type
        self._trace(f"initialize: sources={len(self._sources)}, channel={channel_type}")

        # Log resolved paths for LOCAL sources (indicate if directory)
        for source in self._sources:
            if source.type == SourceType.LOCAL and source.resolved_path:
                path_obj = Path(source.resolved_path)
                is_dir = path_obj.is_dir() if path_obj.exists() else False
                path_type = "dir" if is_dir else "file"
                self._trace(f"initialize: resolved '{source.id}' ({path_type}): {source.path} -> {source.resolved_path}")

        if self._selected_source_ids:
            self._trace(f"initialize: preselected={self._selected_source_ids}")
        if self._exclude_tools:
            self._trace(f"initialize: exclude_tools={self._exclude_tools}")

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        self._trace("shutdown: cleaning up resources")
        if self._channel:
            self._channel.shutdown()
        self._channel = None
        self._sources = []
        self._selected_source_ids = []
        self._initialized = False

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool declarations for the references plugin.

        Tools can be excluded via the exclude_tools config option.
        """
        all_tools = [
            ToolSchema(
                name="selectReferences",
                description=(
                    "Trigger user selection of additional reference sources to incorporate. "
                    "Call this tool directly - it will inform you if no sources are available. "
                    "All parameters are optional."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": (
                                "Optional: explain why you need references to help user select."
                            )
                        },
                        "filter_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Filter sources by tags. Only sources with at least "
                                "one matching tag will be shown to the user."
                            )
                        }
                    },
                    "required": []
                }
            ),
            ToolSchema(
                name="listReferences",
                description=(
                    "List all available reference sources in the catalog, "
                    "including their access methods, tags, and current selection status. "
                    "Use this to discover what references are available before selecting."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "filter_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["all", "auto", "selectable"],
                            "description": "Filter by injection mode (default: all)"
                        }
                    },
                    "required": []
                }
            )
        ]

        # Filter out excluded tools
        if self._exclude_tools:
            return [t for t in all_tools if t.name not in self._exclude_tools]
        return all_tools

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return tool executors."""
        return {
            "selectReferences": self._execute_select,
            "listReferences": self._execute_list,
        }

    def _execute_select(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reference selection flow.

        Presents available selectable sources to user via the configured channel,
        then returns instructions for the model to fetch selected references.
        """
        context = args.get("context")
        filter_tags = args.get("filter_tags", [])
        self._trace(f"selectReferences: context={context!r}, filter_tags={filter_tags}")

        # Early check: no sources configured at all
        if not self._sources:
            return {
                "status": "no_sources",
                "message": "No reference sources available."
            }

        # Get selectable sources not yet selected
        available = [
            s for s in self._sources
            if s.mode == InjectionMode.SELECTABLE
            and s.id not in self._selected_source_ids
        ]

        # Apply tag filter
        if filter_tags:
            available = [
                s for s in available
                if any(tag in s.tags for tag in filter_tags)
            ]

        if not available:
            self._trace("selectReferences: no sources available for selection")
            return {
                "status": "no_sources",
                "message": "No additional reference sources available for selection."
            }

        available_ids = [s.id for s in available]
        self._trace(f"selectReferences: available={available_ids}")

        # Build prompt lines for UI hooks
        prompt_lines = []
        if context:
            prompt_lines.append(f"Context: {context}")
            prompt_lines.append("")
        prompt_lines.append(f"Available references: {len(available)}")
        for i, source in enumerate(available, 1):
            prompt_lines.append(f"  [{i}] {source.name}")

        # Emit selection requested hook
        if self._on_selection_requested:
            self._on_selection_requested("selectReferences", prompt_lines)

        # Present to user via channel
        selected_ids = self._channel.present_selection(available, context)

        # Emit selection resolved hook
        if self._on_selection_resolved:
            self._on_selection_resolved("selectReferences", selected_ids)

        self._trace(f"selectReferences: selected={selected_ids}")

        if not selected_ids:
            self._channel.notify_result([
                "",
                "─" * 60,
                "No reference sources selected.",
                "─" * 60,
            ])
            return {
                "status": "none_selected",
                "message": "User did not select any reference sources."
            }

        # Track selections
        self._selected_source_ids.extend(selected_ids)

        # Build instructions for the model
        selected_sources = [s for s in available if s.id in selected_ids]
        instructions = []

        for source in selected_sources:
            instructions.append(source.to_instruction())

        # Build formatted result for display
        result_lines = [
            "",
            "=" * 60,
            f"SELECTED {len(selected_sources)} REFERENCE(S)",
            "=" * 60,
            "",
        ]
        for source in selected_sources:
            result_lines.append(f"  ✓ {source.name}")
            result_lines.append(f"    {source.description}")
            result_lines.append("")
        result_lines.append("─" * 60)
        result_lines.append("Instructions provided to model.")
        result_lines.append("─" * 60)

        self._channel.notify_result(result_lines)

        return {
            "status": "success",
            "selected_count": len(selected_sources),
            "message": (
                "The user has selected the following reference sources. "
                "Fetch and incorporate their content as needed:"
            ),
            "sources": "\n\n".join(instructions)
        }

    def _execute_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all available reference sources."""
        filter_tags = args.get("filter_tags", [])
        mode_filter = args.get("mode", "all")
        self._trace(f"listReferences: mode={mode_filter}, filter_tags={filter_tags}")

        # Early check: no sources configured at all
        if not self._sources:
            return {
                "sources": [],
                "total": 0,
                "selected_count": 0,
                "message": "No reference sources available."
            }

        sources = self._sources

        # Filter by mode
        if mode_filter == "auto":
            sources = [s for s in sources if s.mode == InjectionMode.AUTO]
        elif mode_filter == "selectable":
            sources = [s for s in sources if s.mode == InjectionMode.SELECTABLE]

        # Filter by tags
        if filter_tags:
            sources = [
                s for s in sources
                if any(tag in s.tags for tag in filter_tags)
            ]

        # Handle empty case with clear message
        if not sources:
            self._trace("listReferences: no sources match filters")
            return {
                "sources": [],
                "total": 0,
                "selected_count": 0,
                "message": "No reference sources available."
            }

        source_ids = [s.id for s in sources]
        self._trace(f"listReferences: returning {len(sources)} sources={source_ids}")

        return {
            "sources": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "type": s.type.value,
                    "mode": s.mode.value,
                    "tags": s.tags,
                    "selected": s.id in self._selected_source_ids,
                    "access": self._get_access_summary(s),
                }
                for s in sources
            ],
            "total": len(sources),
            "selected_count": sum(
                1 for s in sources if s.id in self._selected_source_ids
            ),
        }

    def _get_access_summary(self, source: ReferenceSource) -> str:
        """Get brief access method description."""
        from .models import SourceType

        if source.type == SourceType.LOCAL:
            return f"File: {source.path}"
        elif source.type == SourceType.URL:
            return f"URL: {source.url}"
        elif source.type == SourceType.MCP:
            return f"MCP: {source.server}/{source.tool}"
        elif source.type == SourceType.INLINE:
            return "Inline content"
        return "Unknown"

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions with AUTO and pre-selected sources.

        AUTO sources and pre-selected sources are included in system instructions
        so the model knows to fetch them at the start of the session.
        """
        auto_sources = [
            s for s in self._sources
            if s.mode == InjectionMode.AUTO
        ]

        # Get pre-selected sources (selectable sources that were pre-selected via config)
        preselected_sources = [
            s for s in self._sources
            if s.mode == InjectionMode.SELECTABLE and s.id in self._selected_source_ids
        ]

        # Sources to fetch immediately = AUTO + pre-selected
        immediate_sources = auto_sources + preselected_sources

        auto_ids = [s.id for s in auto_sources]
        preselected_ids = [s.id for s in preselected_sources]
        self._trace(f"get_system_instructions: auto={auto_ids}, preselected={preselected_ids}")

        if not immediate_sources:
            # Still provide info about selectable sources (if selectReferences is available)
            selectable = [
                s for s in self._sources
                if s.mode == InjectionMode.SELECTABLE
                and s.id not in self._selected_source_ids
            ]
            # If selectReferences is excluded or no selectable sources, nothing to show
            if not selectable or "selectReferences" in self._exclude_tools:
                self._trace("get_system_instructions: no sources to inject")
                return None

            parts = [
                "# Reference Sources",
                "",
                "Additional reference sources are available for this session.",
            ]
            if "listReferences" not in self._exclude_tools:
                parts.append("Use `listReferences` to see available sources and their tags.")
            parts.extend([
                "Use `selectReferences` when you encounter topics matching these tags",
                "to request user selection of relevant documentation.",
                "",
                "When reporting sources from listReferences, always indicate selection status:",
                "- 'available but unselected' for sources not yet selected by the user",
                "- 'selected' for sources the user has chosen to include",
                "",
                "Available tags: " + ", ".join(
                    sorted(set(tag for s in selectable for tag in s.tags))
                ),
            ])
            selectable_ids = [s.id for s in selectable]
            self._trace(f"get_system_instructions: injecting selectable hints={selectable_ids}")
            return "\n".join(parts)

        parts = [
            "# Reference Sources",
            "",
            "The following reference sources should be incorporated into your context.",
            "Fetch their content using the appropriate tools as described.",
            ""
        ]

        for source in immediate_sources:
            parts.append(source.to_instruction())
            parts.append("")

        # Mention remaining selectable sources (not pre-selected) if any
        # Only show if selectReferences tool is available
        if "selectReferences" not in self._exclude_tools:
            selectable = [
                s for s in self._sources
                if s.mode == InjectionMode.SELECTABLE
                and s.id not in self._selected_source_ids
            ]
            if selectable:
                parts.extend([
                    "---",
                    "",
                    "Additional reference sources are available on request.",
                    "Use `selectReferences` when you encounter topics matching these tags:",
                    ", ".join(sorted(set(tag for s in selectable for tag in s.tags))),
                ])

        immediate_ids = [s.id for s in immediate_sources]
        self._trace(f"get_system_instructions: injecting immediate sources={immediate_ids}")
        return "\n".join(parts)

    def get_auto_approved_tools(self) -> List[str]:
        """All tools are auto-approved - this is a user-triggered plugin."""
        return ["selectReferences", "listReferences"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for direct invocation.

        These commands can be typed directly by the user (human or agent)
        to interact with reference sources without model mediation.

        listReferences: share_with_model=True - shows available sources, model should know
        selectReferences: share_with_model=True - selection results should be known by model
        """
        return [
            UserCommand("listReferences", "List available reference sources", share_with_model=True),
            UserCommand("selectReferences", "Select reference sources to include", share_with_model=True),
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for reference command arguments.

        These commands take no arguments, so no completions needed.
        """
        return []

    # Public API for programmatic access

    def get_sources(self) -> List[ReferenceSource]:
        """Get all configured reference sources."""
        return self._sources.copy()

    def get_selected_ids(self) -> List[str]:
        """Get IDs of selected sources (includes pre-selected and user-selected)."""
        return self._selected_source_ids.copy()

    def reset_selections(self) -> None:
        """Clear all session selections."""
        self._selected_source_ids.clear()

    # Interactivity protocol methods

    def supports_interactivity(self) -> bool:
        """References plugin requires user interaction for source selection.

        Returns:
            True - references plugin has interactive selection prompts.
        """
        return True

    def get_supported_channels(self) -> List[str]:
        """Return list of channel types supported by references plugin.

        Returns:
            List of supported channel types: console, webhook, file, queue.
        """
        return ["console", "webhook", "file", "queue"]

    def set_selection_hooks(
        self,
        on_requested: Optional[Callable[[str, List[str]], None]] = None,
        on_resolved: Optional[Callable[[str, List[str]], None]] = None
    ) -> None:
        """Set hooks for selection lifecycle events.

        These hooks enable UI integration by notifying when selection
        requests start and complete.

        Args:
            on_requested: Called when selection session starts.
                Signature: (tool_name, prompt_lines) -> None
            on_resolved: Called when selection is resolved.
                Signature: (tool_name, selected_ids) -> None
        """
        self._on_selection_requested = on_requested
        self._on_selection_resolved = on_resolved

    def set_channel(
        self,
        channel_type: str,
        channel_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the interaction channel for reference selection.

        Args:
            channel_type: One of: console, webhook, file
            channel_config: Optional channel-specific configuration

        Raises:
            ValueError: If channel_type is not supported
        """
        if channel_type not in self.get_supported_channels():
            raise ValueError(
                f"Channel type '{channel_type}' not supported. "
                f"Supported: {self.get_supported_channels()}"
            )

        # Create the channel with config
        from .channels import create_channel
        self._channel = create_channel(channel_type, channel_config)


def create_plugin() -> ReferencesPlugin:
    """Factory function to create the references plugin instance."""
    return ReferencesPlugin()
