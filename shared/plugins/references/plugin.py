"""References plugin for managing documentation source injection.

This plugin maintains a catalog of reference sources (documentation, specs,
guides, etc.) and handles:
- AUTO sources: Included in system instructions, model fetches them at startup
- SELECTABLE sources: User chooses which to include via interactive selection

The model is responsible for fetching content using appropriate tools (CLI, MCP, etc.).
This plugin only manages the catalog and user interaction.
"""

from typing import Any, Callable, Dict, List, Optional

from ..model_provider.types import ToolSchema

from .models import ReferenceSource, InjectionMode
from .channels import SelectionChannel, ConsoleSelectionChannel, QueueSelectionChannel, create_channel
from .config_loader import load_config, ReferencesConfig
from ..base import UserCommand, CommandCompletion


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
        self._initialized = False
        # Selection lifecycle hooks for UI integration
        self._on_selection_requested: Optional[Callable[[str, List[str]], None]] = None
        self._on_selection_resolved: Optional[Callable[[str, List[str]], None]] = None

    @property
    def name(self) -> str:
        return self._name

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
        """
        config = config or {}

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
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        if self._channel:
            self._channel.shutdown()
        self._channel = None
        self._sources = []
        self._selected_source_ids = []
        self._initialized = False

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool declarations for the references plugin."""
        return [
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
        # Early check: no sources configured at all
        if not self._sources:
            return {
                "status": "no_sources",
                "message": "No reference sources available."
            }

        context = args.get("context")
        filter_tags = args.get("filter_tags", [])

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
            return {
                "status": "no_sources",
                "message": "No additional reference sources available for selection."
            }

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
        # Early check: no sources configured at all
        if not self._sources:
            return {
                "sources": [],
                "total": 0,
                "selected_count": 0,
                "message": "No reference sources available."
            }

        filter_tags = args.get("filter_tags", [])
        mode_filter = args.get("mode", "all")

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
            return {
                "sources": [],
                "total": 0,
                "selected_count": 0,
                "message": "No reference sources available."
            }

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

        if not immediate_sources:
            # Still provide info about selectable sources
            selectable = [
                s for s in self._sources
                if s.mode == InjectionMode.SELECTABLE
                and s.id not in self._selected_source_ids
            ]
            if not selectable:
                return None

            parts = [
                "# Reference Sources",
                "",
                "Additional reference sources are available for this session.",
                "Use `listReferences` to see available sources and their tags.",
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
            ]
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
