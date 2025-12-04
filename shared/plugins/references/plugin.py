"""References plugin for managing documentation source injection.

This plugin maintains a catalog of reference sources (documentation, specs,
guides, etc.) and handles:
- AUTO sources: Included in system instructions, model fetches them at startup
- SELECTABLE sources: User chooses which to include via interactive selection

The model is responsible for fetching content using appropriate tools (CLI, MCP, etc.).
This plugin only manages the catalog and user interaction.
"""

from typing import Any, Callable, Dict, List, Optional

from google.genai import types

from .models import ReferenceSource, InjectionMode
from .actors import SelectionActor, ConsoleSelectionActor, create_actor
from .config_loader import load_config, ReferencesConfig


class ReferencesPlugin:
    """Plugin for managing reference source injection into model context.

    The plugin maintains a catalog of reference sources and:
    - AUTO sources: Included in system instructions for model to fetch
    - SELECTABLE sources: User chooses via actor (console/webhook/file)

    The model uses existing tools (CLI, MCP, URL fetch) to retrieve content.
    This plugin only provides metadata and handles user selection.
    """

    def __init__(self):
        self._name = "references"
        self._config: Optional[ReferencesConfig] = None
        self._sources: List[ReferenceSource] = []
        self._actor: Optional[SelectionActor] = None
        self._selected_source_ids: List[str] = []  # User-selected during session
        self._initialized = False

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
                   - actor_type: Type of actor ("console", "webhook", "file")
                   - actor_config: Configuration for the actor
                   - sources: Inline sources list (overrides file)
        """
        config = config or {}

        # Try to load from file first
        config_path = config.get("config_path")
        try:
            self._config = load_config(config_path)
        except FileNotFoundError:
            # Use defaults
            self._config = ReferencesConfig()

        # Allow inline sources override
        if "sources" in config:
            self._sources = [
                ReferenceSource.from_dict(s) if isinstance(s, dict) else s
                for s in config["sources"]
            ]
        else:
            self._sources = self._config.sources

        # Initialize actor
        actor_type = config.get("actor_type") or self._config.actor_type
        actor_config = config.get("actor_config", {})

        # Set timeout from config
        if "timeout" not in actor_config:
            actor_config["timeout"] = self._config.actor_timeout

        # Set type-specific config
        if actor_type == "webhook" and "endpoint" not in actor_config:
            if self._config.actor_endpoint:
                actor_config["endpoint"] = self._config.actor_endpoint

        if actor_type == "file" and "base_path" not in actor_config:
            if self._config.actor_base_path:
                actor_config["base_path"] = self._config.actor_base_path

        try:
            self._actor = create_actor(actor_type, actor_config)
        except (ValueError, RuntimeError) as e:
            # Fall back to console actor if configured actor fails
            print(f"Warning: Failed to initialize {actor_type} actor: {e}")
            print("Falling back to console actor")
            self._actor = ConsoleSelectionActor()
            self._actor.initialize({})

        self._selected_source_ids = []
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        if self._actor:
            self._actor.shutdown()
        self._actor = None
        self._sources = []
        self._selected_source_ids = []
        self._initialized = False

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Return tool declarations for the references plugin."""
        return [
            types.FunctionDeclaration(
                name="selectReferences",
                description=(
                    "Trigger user selection of additional reference sources to incorporate. "
                    "Call this tool directly - it will inform you if no sources are available. "
                    "All parameters are optional."
                ),
                parameters_json_schema={
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
            types.FunctionDeclaration(
                name="listReferences",
                description=(
                    "List all available reference sources in the catalog, "
                    "including their access methods, tags, and current selection status. "
                    "Use this to discover what references are available before selecting."
                ),
                parameters_json_schema={
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

        Presents available selectable sources to user via the configured actor,
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

        # Present to user via actor
        selected_ids = self._actor.present_selection(available, context)

        if not selected_ids:
            self._actor.notify_result("No reference sources selected.")
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

        self._actor.notify_result(
            f"Selected {len(selected_sources)} reference source(s). "
            "Instructions provided to model."
        )

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
        """Return system instructions with AUTO sources.

        AUTO sources are included in system instructions so the model
        knows to fetch them at the start of the session.
        """
        auto_sources = [
            s for s in self._sources
            if s.mode == InjectionMode.AUTO
        ]

        if not auto_sources:
            # Still provide info about selectable sources
            selectable = [
                s for s in self._sources
                if s.mode == InjectionMode.SELECTABLE
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

        for source in auto_sources:
            parts.append(source.to_instruction())
            parts.append("")

        # Mention selectable sources if any
        selectable = [
            s for s in self._sources
            if s.mode == InjectionMode.SELECTABLE
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

    def get_user_commands(self) -> List[tuple[str, str]]:
        """References plugin provides model tools only, no user commands."""
        return []

    # Public API for programmatic access

    def get_sources(self) -> List[ReferenceSource]:
        """Get all configured reference sources."""
        return self._sources.copy()

    def get_selected_ids(self) -> List[str]:
        """Get IDs of sources selected during this session."""
        return self._selected_source_ids.copy()

    def reset_selections(self) -> None:
        """Clear all session selections."""
        self._selected_source_ids.clear()


def create_plugin() -> ReferencesPlugin:
    """Factory function to create the references plugin instance."""
    return ReferencesPlugin()
