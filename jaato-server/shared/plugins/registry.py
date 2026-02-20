"""Plugin registry for discovering, loading, and managing plugins."""

import importlib
import importlib.metadata
import logging
import os
import pkgutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Callable, Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

from jaato_sdk.plugins.base import (
    ToolPlugin,
    UserCommand,
    PromptEnrichmentResult,
    SystemInstructionEnrichmentResult,
    ToolResultEnrichmentResult,
    model_matches_requirements,
    OutputCallback,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema
from .streaming.protocol import StreamingCapable
from .enrichment_formatter import (
    EnrichmentNotification,
    format_enrichment_notifications,
)
from shared.trace import trace as _trace_write

# Entry point group names by plugin kind
PLUGIN_ENTRY_POINT_GROUPS = {
    "tool": "jaato.plugins",
    "gc": "jaato.gc_plugins",
}


def _trace(msg: str, include_traceback: bool = False) -> None:
    """Write trace message to log file for debugging.

    Args:
        msg: The message to log.
        include_traceback: If True, append the current exception traceback.
    """
    _trace_write("PluginRegistry", msg, include_traceback=include_traceback)
    # Also log to standard logger for visibility
    if include_traceback:
        logger.error(msg, exc_info=True)
    else:
        logger.debug(msg)


class PluginRegistry:
    """Manages plugin discovery, lifecycle, and tool exposure state.

    Usage:
        registry = PluginRegistry()
        registry.discover()

        print(registry.list_available())  # ['cli', 'mcp', ...]

        registry.expose_tool('cli', config={'extra_paths': ['/usr/local/bin']})
        registry.expose_tool('mcp')

        # Get tools for exposed plugins
        tool_schemas = registry.get_exposed_tool_schemas()
        executors = registry.get_exposed_executors()

        # Later, unexpose plugins
        registry.unexpose_tool('mcp')
        registry.unexpose_all()

    External Path Authorization:
        Plugins can authorize external paths (outside workspace) for model access.
        This is used by the references plugin to allow readFile to access
        referenced documents outside the workspace.

        # Register an authorized external path
        registry.authorize_external_path('/docs/external/guide.md', 'references')

        # Check if a path is authorized
        if registry.is_path_authorized('/docs/external/guide.md'):
            # Allow access

        # Clear authorizations from a specific plugin
        registry.clear_authorized_paths('references')

    Auto-Wiring:
        Plugins may implement optional methods for automatic dependency injection:

        - set_plugin_registry(registry): Called during expose_tool() to give plugins
          access to the registry for cross-plugin communication.

        - set_workspace_path(path): Called when workspace changes to notify plugins
          that need workspace-relative operations (sandboxing, relative paths, etc.).
          The registry broadcasts to all exposed plugins implementing this method.

        Example plugin implementation:
            class MyPlugin:
                def set_workspace_path(self, path: str) -> None:
                    self._workspace_path = path
                    self._sandbox.set_root(path)
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the plugin registry.

        Args:
            model_name: Optional model name for checking plugin requirements.
                       If provided, plugins with model_requirements that don't
                       match will be skipped during expose_tool().
        """
        self._plugins: Dict[str, ToolPlugin] = {}
        self._exposed: Set[str] = set()
        self._enrichment_only: Set[str] = set()  # Plugins for prompt enrichment only
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._model_name: Optional[str] = model_name
        self._skipped_plugins: Dict[str, List[str]] = {}  # name -> required patterns
        self._disabled_tools: Set[str] = set()  # Individual tools disabled by user
        self._output_callback: Optional[OutputCallback] = None
        self._terminal_width: int = 80
        # Authorized external paths: path -> (source plugin name, access mode)
        # access mode is "readonly" or "readwrite"
        self._authorized_external_paths: Dict[str, tuple] = {}
        # Denied external paths: path -> source plugin name (takes precedence over authorized)
        self._denied_external_paths: Dict[str, str] = {}
        # Core tools: framework-provided tools not from plugins
        self._core_tools: Dict[str, ToolSchema] = {}
        self._core_executors: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._core_auto_approved: Set[str] = set()  # Auto-approved core tools
        # Workspace path for plugins that need it
        self._workspace_path: Optional[str] = None
        # Cache: tool_name -> plugin for get_plugin_for_tool() lookups
        self._tool_plugin_cache: Dict[str, ToolPlugin] = {}

    def set_output_callback(
        self,
        callback: Optional[OutputCallback],
        terminal_width: int = 80
    ) -> None:
        """Set the output callback for enrichment notifications.

        When set, the registry will emit visual notifications about enrichments
        through this callback with source="enrichment".

        Args:
            callback: OutputCallback function (source, text, mode) -> None,
                     or None to disable notifications.
            terminal_width: Terminal width for word wrapping (default 80).
        """
        self._output_callback = callback
        self._terminal_width = terminal_width
        _trace(f"set_output_callback: callback={callback is not None}, terminal_width={terminal_width}")

    def _emit_enrichment_notifications(
        self,
        notifications: List[EnrichmentNotification],
        output_callback: Optional[OutputCallback] = None,
        terminal_width: Optional[int] = None
    ) -> None:
        """Emit enrichment notifications through the output callback.

        Args:
            notifications: List of enrichment notifications to display.
            output_callback: Optional callback to use instead of the registry's
                shared callback. Used for session-specific routing.
            terminal_width: Terminal width for formatting (uses registry default if not provided).
        """
        # Use provided callback or fall back to shared one
        callback = output_callback if output_callback is not None else self._output_callback
        width = terminal_width if terminal_width is not None else self._terminal_width

        _trace(f"_emit_enrichment_notifications: {len(notifications)} notifications, callback={callback is not None}")
        if not callback or not notifications:
            return

        formatted = format_enrichment_notifications(
            notifications,
            terminal_width=width
        )
        _trace(f"_emit_enrichment_notifications: formatted={repr(formatted[:100]) if formatted else 'empty'}")
        if formatted:
            callback("enrichment", formatted, "write")

    def _build_notification(
        self,
        target: str,
        plugin_name: str,
        metadata: Optional[Dict[str, Any]],
        size_bytes: int
    ) -> Optional[EnrichmentNotification]:
        """Build an enrichment notification from plugin metadata.

        Plugins can provide explicit notification info via metadata["notification"],
        or a fallback message is generated from known metadata patterns.

        Args:
            target: What was enriched ("prompt", "result", "system").
            plugin_name: Name of the plugin that performed enrichment.
            metadata: Plugin's returned metadata, may contain notification info.
            size_bytes: Size difference in bytes (can be negative for reductions).

        Returns:
            EnrichmentNotification if a message could be generated, None otherwise.
        """
        if metadata is None:
            metadata = {}

        # Check for explicit notification in metadata
        notif_data = metadata.get("notification")
        if isinstance(notif_data, dict):
            message = notif_data.get("message", "enriched content")
            notif_size = notif_data.get("size_bytes", abs(size_bytes) if size_bytes else None)
            return EnrichmentNotification(
                target=target,
                plugin=plugin_name,
                message=message,
                size_bytes=notif_size
            )

        # Generate fallback message from known metadata patterns
        message = self._generate_fallback_message(plugin_name, metadata)
        if message:
            return EnrichmentNotification(
                target=target,
                plugin=plugin_name,
                message=message,
                size_bytes=abs(size_bytes) if size_bytes else None
            )

        # Last resort: generic message
        return EnrichmentNotification(
            target=target,
            plugin=plugin_name,
            message="enriched content",
            size_bytes=abs(size_bytes) if size_bytes else None
        )

    def _generate_fallback_message(
        self,
        plugin_name: str,
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a notification message from known metadata patterns.

        Args:
            plugin_name: Name of the plugin for pattern matching.
            metadata: Plugin's returned metadata.

        Returns:
            A human-readable message, or None if no pattern matched.
        """
        # Memory plugin pattern
        if "memory_matches" in metadata:
            count = metadata["memory_matches"]
            if count > 0:
                # Include trigger keywords if available
                keywords = metadata.get("trigger_keywords", [])
                if keywords:
                    keyword_summary = ", ".join(f'"{k}"' for k in keywords[:3])
                    if len(keywords) > 3:
                        keyword_summary += f" +{len(keywords) - 3} more"
                    return f"added context about {count} memories (triggered by {keyword_summary})"
                return f"added context about {count} relevant memories"

        # Session plugin pattern
        if metadata.get("description_requested"):
            return "requested session description (turn threshold reached)"

        # LSP plugin pattern - show diagnostic counts
        if "total_errors" in metadata or "total_warnings" in metadata:
            errors = metadata.get("total_errors", 0)
            warnings = metadata.get("total_warnings", 0)
            files = metadata.get("files_checked", [])
            file_str = files[0] if len(files) == 1 else f"{len(files)} files"

            parts = []
            if errors > 0:
                parts.append(f"{errors} error{'s' if errors != 1 else ''}")
            if warnings > 0:
                parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")

            if parts:
                return f"found {', '.join(parts)} in {file_str}"
            else:
                return f"checked {file_str}, no issues found"

        # References plugin pattern — @mention expansion in prompts/tool results
        if "mentioned_references" in metadata:
            refs = metadata["mentioned_references"]
            if refs:
                if len(refs) == 1:
                    return f"expanded @{refs[0]}"
                elif len(refs) <= 3:
                    ref_list = ", ".join(f"@{r}" for r in refs)
                    return f"expanded {ref_list}"
                else:
                    return f"expanded {len(refs)} references"

        # References plugin pattern - tag-matched reference ID hints.
        # Always include the matched tags in the notification so the user
        # can see which tags triggered the match and diagnose false positives.
        if "tag_matched_references" in metadata:
            matched = metadata["tag_matched_references"]  # {source_id: [tags]}
            if matched:
                ids = list(matched.keys())
                # Collect unique matched tags across all hinted sources
                all_tags = sorted(set(t for tags in matched.values() for t in tags))
                tag_summary = ", ".join(all_tags[:5])
                if len(all_tags) > 5:
                    tag_summary += f" +{len(all_tags) - 5}"
                if len(ids) == 1:
                    return f"hinted @{ids[0]} (matched: {tag_summary})"
                elif len(ids) <= 3:
                    ref_list = ", ".join(f"@{r}" for r in ids)
                    return f"hinted {ref_list} (matched: {tag_summary})"
                else:
                    shown = ", ".join(f"@{r}" for r in ids[:3])
                    return f"hinted {shown} +{len(ids) - 3} more (matched: {tag_summary})"

        # References plugin pattern — transitively selected references.
        # Shows which references were auto-included and which parent source
        # triggered the inclusion.
        if "transitive_references" in metadata:
            transitive = metadata["transitive_references"]  # {id: [parent_ids]}
            if transitive:
                ids = list(transitive.keys())
                # Collect unique parent IDs across all transitive sources
                all_parents = sorted(set(p for parents in transitive.values() for p in parents))
                parent_str = ", ".join(f"@{p}" for p in all_parents[:3])
                if len(all_parents) > 3:
                    parent_str += f" +{len(all_parents) - 3}"
                if len(ids) == 1:
                    return f"transitively included @{ids[0]} (from {parent_str})"
                elif len(ids) <= 3:
                    ref_list = ", ".join(f"@{r}" for r in ids)
                    return f"transitively included {ref_list} (from {parent_str})"
                else:
                    shown = ", ".join(f"@{r}" for r in ids[:3])
                    return f"transitively included {shown} +{len(ids) - 3} more (from {parent_str})"

        # Template extraction pattern
        if "extracted_templates" in metadata:
            templates = metadata["extracted_templates"]
            if templates:
                if len(templates) == 1:
                    return f"extracted @{templates[0]} template"
                else:
                    return f"extracted {len(templates)} templates"

        return None

    def discover(
        self,
        plugin_kind: str = "tool",
        include_directory: bool = True
    ) -> List[str]:
        """Discover plugins via entry points and optionally directory scanning.

        Discovery order:
        1. Entry points (group based on plugin_kind) - for installed packages
        2. Directory scanning (optional) - for development/local plugins

        Entry points allow external packages to register plugins:
            [project.entry-points."jaato.plugins"]
            my_plugin = "my_package.plugins:create_plugin"

        Args:
            plugin_kind: Kind of plugin to discover ('tool', 'gc', etc.).
                        Only plugins with matching PLUGIN_KIND are loaded.
            include_directory: Also scan the plugins directory for local plugins.
                             Useful during development when package isn't installed.

        Returns:
            List of discovered plugin names.
        """
        discovered = []

        # First, discover via entry points (installed packages)
        discovered.extend(self._discover_via_entry_points(plugin_kind))

        # Then, optionally scan the plugins directory (development mode)
        if include_directory:
            discovered.extend(self._discover_via_directory(plugin_kind))

        return discovered

    def _discover_via_entry_points(self, plugin_kind: str) -> List[str]:
        """Discover plugins registered via entry points.

        External packages can register plugins by adding to the appropriate
        entry point group in their pyproject.toml.

        Args:
            plugin_kind: Kind of plugin to discover ('tool', 'gc', etc.).

        Returns:
            List of discovered plugin names.
        """
        discovered = []

        entry_point_group = PLUGIN_ENTRY_POINT_GROUPS.get(plugin_kind)
        if not entry_point_group:
            return discovered

        try:
            # Python 3.10+ API
            if sys.version_info >= (3, 10):
                eps = importlib.metadata.entry_points(group=entry_point_group)
            else:
                # Python 3.9 compatibility
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get(entry_point_group, [])

            for ep in eps:
                # Skip if already loaded (avoid duplicates with directory scan)
                if ep.name in self._plugins:
                    continue

                try:
                    create_plugin = ep.load()
                    plugin = create_plugin()

                    # For tool plugins, verify protocol implementation
                    if plugin_kind == "tool" and not isinstance(plugin, ToolPlugin):
                        _trace(f" Entry point '{ep.name}': "
                              f"plugin does not implement ToolPlugin protocol")
                        continue

                    self._plugins[plugin.name] = plugin
                    discovered.append(plugin.name)

                except Exception as exc:
                    _trace(f" Error loading entry point '{ep.name}': {exc}", include_traceback=True)

        except Exception as exc:
            # Entry points not available (package not installed)
            pass

        return discovered

    def _discover_via_directory(
        self,
        plugin_kind: str,
        plugin_dir: Optional[Path] = None
    ) -> List[str]:
        """Discover plugins by scanning the plugins directory.

        This is the fallback/development mode discovery that scans for Python
        modules with a create_plugin() factory function and matching PLUGIN_KIND.

        Args:
            plugin_kind: Kind of plugin to discover ('tool', 'gc', etc.).
            plugin_dir: Directory to scan. Defaults to this package's directory.

        Returns:
            List of discovered plugin names.
        """
        if plugin_dir is None:
            plugin_dir = Path(__file__).parent

        discovered = []

        for finder, name, ispkg in pkgutil.iter_modules([str(plugin_dir)]):
            # Skip internal modules
            if name.startswith('_') or name in ('base', 'registry'):
                continue

            # Skip if already loaded via entry points
            if name in self._plugins:
                continue

            try:
                module = importlib.import_module(f".{name}", package="shared.plugins")

                # Check plugin kind - only load plugins matching requested kind
                module_kind = getattr(module, 'PLUGIN_KIND', None)
                if module_kind != plugin_kind:
                    continue

                if hasattr(module, 'create_plugin'):
                    plugin = module.create_plugin()

                    # For tool plugins, verify protocol implementation
                    if plugin_kind == "tool" and not isinstance(plugin, ToolPlugin):
                        _trace(f" {name}: plugin does not implement ToolPlugin protocol")
                        continue

                    self._plugins[plugin.name] = plugin
                    discovered.append(plugin.name)

            except Exception as exc:
                _trace(f" Error loading plugin '{name}': {exc}", include_traceback=True)

        return discovered

    def set_model_name(self, model_name: str) -> None:
        """Set the model name for checking plugin requirements.

        Args:
            model_name: The model name (e.g., 'gemini-3-pro-preview').
        """
        self._model_name = model_name
        # Clear skipped plugins as model changed
        self._skipped_plugins.clear()

    def get_model_name(self) -> Optional[str]:
        """Get the currently configured model name."""
        return self._model_name

    def list_available(self) -> List[str]:
        """List all discovered plugin names."""
        return list(self._plugins.keys())

    def list_exposed(self) -> List[str]:
        """List currently exposed plugin names."""
        return list(self._exposed)

    def is_exposed(self, name: str) -> bool:
        """Check if a plugin's tools are currently exposed to the model."""
        return name in self._exposed

    def get_plugin(self, name: str) -> Optional[ToolPlugin]:
        """Get a plugin by name, or None if not found."""
        return self._plugins.get(name)

    def register_plugin(
        self,
        plugin: ToolPlugin,
        expose: bool = False,
        enrichment_only: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Manually register a plugin with the registry.

        Use this for plugins that aren't discovered via entry points or directory
        scanning, such as the session plugin which is configured separately.

        This allows the plugin to participate in prompt enrichment and other
        registry-managed features without being discovered.

        Args:
            plugin: The plugin instance to register.
            expose: If True, also expose the plugin's tools (calls initialize).
            enrichment_only: If True, only participate in prompt enrichment
                           (not included in get_exposed_tool_schemas/executors).
            config: Optional configuration dict if exposing.

        Example:
            # Register session plugin for prompt enrichment only
            registry.register_plugin(session_plugin, enrichment_only=True)
        """
        self._plugins[plugin.name] = plugin

        if enrichment_only:
            self._enrichment_only.add(plugin.name)
        elif expose:
            self.expose_tool(plugin.name, config)

    def register_core_tool(
        self,
        schema: ToolSchema,
        executor: Callable[[Dict[str, Any]], Any],
        auto_approved: bool = False
    ) -> None:
        """Register a core framework tool.

        Core tools are provided directly by framework components (like StreamManager)
        rather than through plugins. They are included in get_exposed_tool_schemas()
        and get_exposed_executors() alongside plugin tools.

        Args:
            schema: The tool's schema.
            executor: The tool's executor function.
            auto_approved: If True, tool is auto-approved (no permission required).

        Example:
            # Register dismiss_stream from StreamManager
            for schema in stream_manager.get_tool_schemas():
                executor = stream_manager.get_executors()[schema.name]
                auto_approved = schema.name in stream_manager.get_auto_approved_tools()
                registry.register_core_tool(schema, executor, auto_approved)
        """
        self._core_tools[schema.name] = schema
        self._core_executors[schema.name] = executor
        if auto_approved:
            self._core_auto_approved.add(schema.name)
        _trace(f"Registered core tool: {schema.name} (auto_approved={auto_approved})")

    def expose_tool(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Expose a plugin's tools to the model.

        Calls the plugin's initialize() method if this is the first time
        exposing it, or if a new config is provided.

        If a model_name is set and the plugin has model_requirements that
        don't match, the plugin is skipped with a warning.

        Args:
            name: Plugin name to expose.
            config: Optional configuration dict for the plugin.

        Returns:
            True if the plugin was exposed, False if skipped due to model requirements.

        Raises:
            ValueError: If the plugin is not found.
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found. Available: {self.list_available()}")

        plugin = self._plugins[name]

        # Check model requirements if model_name is set
        if self._model_name and hasattr(plugin, 'get_model_requirements'):
            requirements = plugin.get_model_requirements()
            if requirements and not model_matches_requirements(self._model_name, requirements):
                self._skipped_plugins[name] = requirements
                _trace(f" Plugin '{name}' skipped: "
                      f"model '{self._model_name}' not in {requirements}")
                return False

        # Initialize if not already exposed, or if new config provided
        if name not in self._exposed:
            plugin.initialize(config)
            if config:
                self._configs[name] = config
            self._exposed.add(name)
            self._tool_plugin_cache.clear()
            # Wire up plugin with registry for authorized external paths
            if hasattr(plugin, 'set_plugin_registry'):
                plugin.set_plugin_registry(self)
                _trace(f" Plugin '{name}' wired with registry")
        elif config and config != self._configs.get(name):
            # Re-initialize with new config
            plugin.shutdown()
            plugin.initialize(config)
            self._configs[name] = config
            self._tool_plugin_cache.clear()
            # Re-wire after re-initialization
            if hasattr(plugin, 'set_plugin_registry'):
                plugin.set_plugin_registry(self)
                _trace(f" Plugin '{name}' re-wired with registry")

        return True

    def unexpose_tool(self, name: str) -> None:
        """Stop exposing a plugin's tools to the model.

        Calls the plugin's shutdown() method to clean up resources.

        Args:
            name: Plugin name to unexpose.
        """
        if name in self._exposed:
            self._plugins[name].shutdown()
            self._exposed.discard(name)
            self._configs.pop(name, None)
            self._tool_plugin_cache.clear()

    def expose_all(
        self,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Expose all discovered plugins' tools.

        Args:
            config: Optional dict mapping plugin names to their configs.
            on_progress: Optional callback invoked with each plugin name
                before it is exposed.  Used by the server to emit
                per-plugin init progress events.
        """
        config = config or {}
        for name in self._plugins:
            if on_progress:
                on_progress(name)
            self.expose_tool(name, config.get(name))

    def unexpose_all(self) -> None:
        """Stop exposing all plugins' tools."""
        for name in list(self._exposed):
            self.unexpose_tool(name)

    def collect_prerequisite_policies(self) -> list:
        """Collect prerequisite policies from all exposed plugins.

        Iterates over all exposed (and enrichment-only) plugins, calling
        ``get_prerequisite_policies()`` on those that implement it. Returns
        a flat list of PrerequisitePolicy objects that can be registered
        with the ReliabilityPlugin.

        This follows the same ``hasattr()``-based duck-typing pattern used
        for enrichment subscriptions.

        Returns:
            List of PrerequisitePolicy objects from all plugins.
        """
        policies = []
        all_names = self._exposed | getattr(self, '_enrichment_only', set())
        for name in all_names:
            try:
                plugin = self._plugins[name]
                if hasattr(plugin, 'get_prerequisite_policies'):
                    plugin_policies = plugin.get_prerequisite_policies()
                    if plugin_policies:
                        # Tag each policy with its owner plugin name
                        for p in plugin_policies:
                            if hasattr(p, 'owner_plugin') and not p.owner_plugin:
                                p.owner_plugin = name
                        policies.extend(plugin_policies)
                        _trace(f"Collected {len(plugin_policies)} prerequisite policies from '{name}'")
            except Exception as exc:
                _trace(f"Error collecting prerequisite policies from '{name}': {exc}")
        return policies

    def set_workspace_path(self, path: str) -> None:
        """Set workspace path and broadcast to all plugins that support it.

        Plugins implementing set_workspace_path(path) will be notified.
        This includes: lsp, mcp, file_edit, cli, and any other plugins
        that need to know the workspace root.

        Args:
            path: Absolute path to the workspace root directory.
        """
        self._workspace_path = path
        _trace(f"set_workspace_path: {path}")

        # Broadcast to all exposed plugins that support it
        for name in self._exposed:
            plugin = self._plugins.get(name)
            if plugin and hasattr(plugin, 'set_workspace_path'):
                try:
                    plugin.set_workspace_path(path)
                    _trace(f"  -> {name}.set_workspace_path()")
                except Exception as exc:
                    _trace(f"  -> {name}.set_workspace_path() failed: {exc}")

    def get_workspace_path(self) -> Optional[str]:
        """Get the current workspace path.

        Returns:
            The workspace path, or None if not set.
        """
        return self._workspace_path

    def get_exposed_tool_schemas(self) -> List[ToolSchema]:
        """Get ToolSchemas from all exposed plugins and core tools."""
        schemas = []
        # Add plugin tool schemas
        for name in self._exposed:
            try:
                schemas.extend(self._plugins[name].get_tool_schemas())
            except Exception as exc:
                _trace(f" Error getting tool schemas from '{name}': {exc}", include_traceback=True)
        # Add core tool schemas
        schemas.extend(self._core_tools.values())
        return schemas

    def get_exposed_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Get executor callables from all exposed plugins and core tools."""
        executors = {}
        # Add plugin executors
        for name in self._exposed:
            try:
                executors.update(self._plugins[name].get_executors())
            except Exception as exc:
                _trace(f" Error getting executors from '{name}': {exc}", include_traceback=True)
        # Add core tool executors
        executors.update(self._core_executors)
        return executors

    # ==================== Individual Tool Management ====================

    def get_all_tool_names(self) -> List[str]:
        """Get names of all tools from exposed plugins.

        Returns:
            List of all tool names (regardless of enabled/disabled state).
        """
        names = []
        for name in self._exposed:
            try:
                schemas = self._plugins[name].get_tool_schemas()
                names.extend(schema.name for schema in schemas)
            except Exception as exc:
                _trace(f" Error getting tool names from '{name}': {exc}", include_traceback=True)
        return names

    def get_tool_traits(self, tool_name: str) -> "FrozenSet[str]":
        """Return the trait set declared on a tool's schema.

        Iterates exposed plugins to find the matching tool name and returns
        its ``traits`` frozenset.  Returns an empty frozenset when the tool
        is not found or the schema has no traits.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The tool's declared traits, or ``frozenset()`` if not found.
        """
        for name in self._exposed:
            try:
                schemas = self._plugins[name].get_tool_schemas()
                for schema in schemas:
                    if schema.name == tool_name:
                        return schema.traits
            except Exception:
                continue
        return frozenset()

    def disable_tool(self, tool_name: str) -> bool:
        """Disable a specific tool by name.

        Disabled tools are not exposed to the model but remain available
        for re-enabling later.

        Args:
            tool_name: Name of the tool to disable.

        Returns:
            True if tool was found and disabled, False if not found.
        """
        all_tools = self.get_all_tool_names()
        if tool_name not in all_tools:
            return False
        self._disabled_tools.add(tool_name)
        return True

    def enable_tool(self, tool_name: str) -> bool:
        """Enable a previously disabled tool.

        Args:
            tool_name: Name of the tool to enable.

        Returns:
            True if tool was found and enabled, False if not found.
        """
        all_tools = self.get_all_tool_names()
        if tool_name not in all_tools:
            return False
        self._disabled_tools.discard(tool_name)
        return True

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is currently enabled.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is enabled, False if disabled or not found.
        """
        return tool_name not in self._disabled_tools

    def disable_all_tools(self) -> int:
        """Disable all tools from exposed plugins.

        Returns:
            Number of tools disabled.
        """
        all_tools = self.get_all_tool_names()
        self._disabled_tools = set(all_tools)
        return len(all_tools)

    def enable_all_tools(self) -> int:
        """Enable all tools (clear all disabled).

        Returns:
            Number of tools that were re-enabled.
        """
        count = len(self._disabled_tools)
        self._disabled_tools.clear()
        return count

    def get_tool_status(self) -> List[Dict[str, Any]]:
        """Get status of all tools from exposed plugins.

        Returns:
            List of dicts with 'name', 'description', 'enabled', 'plugin' keys.
        """
        status = []
        for plugin_name in self._exposed:
            try:
                schemas = self._plugins[plugin_name].get_tool_schemas()
                for schema in schemas:
                    status.append({
                        'name': schema.name,
                        'description': schema.description,
                        'enabled': schema.name not in self._disabled_tools,
                        'plugin': plugin_name,
                    })
            except Exception as exc:
                _trace(f" Error getting tool status from '{plugin_name}': {exc}", include_traceback=True)
        return status

    def get_enabled_tool_schemas(self) -> List[ToolSchema]:
        """Get ToolSchemas from exposed plugins and core tools, excluding disabled.

        This is what should be sent to the model - only enabled tools.
        For plugins implementing StreamingCapable, auto-generates :stream
        variants for tools that support streaming.

        Returns:
            List of ToolSchema objects for enabled tools only.
        """
        schemas = []
        for name in self._exposed:
            try:
                plugin = self._plugins[name]
                plugin_schemas = plugin.get_tool_schemas()
                # Filter out disabled tools
                enabled_schemas = [s for s in plugin_schemas if s.name not in self._disabled_tools]
                schemas.extend(enabled_schemas)

                # Auto-generate :stream variants for streaming-capable plugins
                if isinstance(plugin, StreamingCapable):
                    for schema in enabled_schemas:
                        if plugin.supports_streaming(schema.name):
                            stream_schema = self._create_streaming_schema(schema)
                            if stream_schema.name not in self._disabled_tools:
                                schemas.append(stream_schema)
            except Exception as exc:
                _trace(f" Error getting tool schemas from '{name}': {exc}", include_traceback=True)

        # Add core tool schemas (excluding disabled)
        for name, schema in self._core_tools.items():
            if name not in self._disabled_tools:
                schemas.append(schema)

        return schemas

    def _create_streaming_schema(self, base_schema: ToolSchema) -> ToolSchema:
        """Create a :stream variant of a tool schema.

        The streaming variant has the same parameters but different behavior -
        it returns immediately with initial chunks and continues streaming
        in the background.

        Inherits category and discoverability from the base schema.

        Args:
            base_schema: The base tool schema to create a streaming variant for.

        Returns:
            A new ToolSchema for the :stream variant.
        """
        streaming_description = (
            f"{base_schema.description} "
            f"(STREAMING MODE: Returns immediately with initial results and stream_id. "
            f"More results will be automatically injected as they become available. "
            f"Call dismiss_stream(stream_id) when you have enough results.)"
        )
        return ToolSchema(
            name=f"{base_schema.name}:stream",
            description=streaming_description,
            parameters=base_schema.parameters,
            category=base_schema.category,
            discoverability=base_schema.discoverability,
        )

    def get_core_tool_schemas(self) -> List[ToolSchema]:
        """Get ToolSchemas for 'core' tools only (always loaded in context).

        This supports deferred tool loading by returning only tools marked
        with discoverability='core'. Other tools can be discovered via
        introspection (list_tools, get_tool_schemas).

        For plugins implementing StreamingCapable, auto-generates :stream
        variants for core tools that support streaming.

        Returns:
            List of ToolSchema objects for core tools only.
        """
        schemas = []
        for name in self._exposed:
            try:
                plugin = self._plugins[name]
                plugin_schemas = plugin.get_tool_schemas()
                # Filter to core, enabled tools only
                core_schemas = [
                    s for s in plugin_schemas
                    if s.name not in self._disabled_tools
                    and getattr(s, 'discoverability', 'discoverable') == 'core'
                ]
                schemas.extend(core_schemas)

                # Auto-generate :stream variants for streaming-capable plugins (core tools)
                if isinstance(plugin, StreamingCapable):
                    for schema in core_schemas:
                        if plugin.supports_streaming(schema.name):
                            stream_schema = self._create_streaming_schema(schema)
                            if stream_schema.name not in self._disabled_tools:
                                schemas.append(stream_schema)
            except Exception as exc:
                _trace(f" Error getting tool schemas from '{name}': {exc}", include_traceback=True)

        # Add core tool schemas that have discoverability='core' (excluding disabled)
        for name, schema in self._core_tools.items():
            if name not in self._disabled_tools:
                if getattr(schema, 'discoverability', 'discoverable') == 'core':
                    schemas.append(schema)

        return schemas

    def get_enabled_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Get executor callables from exposed plugins and core tools, excluding disabled.

        Returns:
            Dict mapping tool names to executor callables for enabled tools only.
        """
        executors = {}
        for name in self._exposed:
            try:
                plugin_executors = self._plugins[name].get_executors()
                # Filter out disabled tools
                for tool_name, executor in plugin_executors.items():
                    if tool_name not in self._disabled_tools:
                        executors[tool_name] = executor
            except Exception as exc:
                _trace(f" Error getting executors from '{name}': {exc}", include_traceback=True)

        # Add core tool executors (excluding disabled)
        for tool_name, executor in self._core_executors.items():
            if tool_name not in self._disabled_tools:
                executors[tool_name] = executor

        return executors

    def list_disabled_tools(self) -> List[str]:
        """List all currently disabled tool names.

        Returns:
            List of disabled tool names.
        """
        return list(self._disabled_tools)

    # ==================== Streaming Tool Support ====================

    def is_streaming_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a streaming variant.

        Args:
            tool_name: Tool name to check (e.g., "grep_content:stream").

        Returns:
            True if this is a :stream variant.
        """
        return tool_name.endswith(":stream")

    def get_base_tool_name(self, tool_name: str) -> str:
        """Get the base tool name from a streaming variant.

        Args:
            tool_name: Tool name (e.g., "grep_content:stream").

        Returns:
            Base tool name (e.g., "grep_content").
        """
        if self.is_streaming_tool(tool_name):
            return tool_name[:-7]  # Remove ":stream" suffix
        return tool_name

    def get_streaming_plugin(self, tool_name: str) -> Optional[StreamingCapable]:
        """Get the streaming-capable plugin for a tool.

        Args:
            tool_name: Tool name (base or :stream variant).

        Returns:
            The StreamingCapable plugin, or None if not found or not streaming-capable.
        """
        base_name = self.get_base_tool_name(tool_name)

        for plugin_name in self._exposed:
            try:
                plugin = self._plugins[plugin_name]
                if base_name in plugin.get_executors():
                    if isinstance(plugin, StreamingCapable):
                        return plugin
                    return None
            except Exception:
                pass
        return None

    def get_streaming_tools(self) -> List[str]:
        """Get list of all tools that support streaming.

        Returns:
            List of base tool names (without :stream suffix) that support streaming.
        """
        streaming_tools = []
        for plugin_name in self._exposed:
            try:
                plugin = self._plugins[plugin_name]
                if isinstance(plugin, StreamingCapable):
                    for schema in plugin.get_tool_schemas():
                        if plugin.supports_streaming(schema.name):
                            streaming_tools.append(schema.name)
            except Exception as exc:
                _trace(f" Error getting streaming tools from '{plugin_name}': {exc}")
        return streaming_tools

    def plugin_has_core_tools(self, plugin_name: str) -> bool:
        """Check whether a plugin has at least one tool with discoverability='core'.

        Used by the runtime and session to decide whether a plugin's system
        instructions should be included in the initial model context when
        deferred tool loading is enabled.  Plugins whose tools are **all**
        discoverable should have their instructions deferred until the model
        actually discovers one of their tools.

        Args:
            plugin_name: Name of the plugin to check.

        Returns:
            True if the plugin provides at least one core tool, False otherwise.
        """
        plugin = self._plugins.get(plugin_name)
        if not plugin or not hasattr(plugin, 'get_tool_schemas'):
            return False
        try:
            for schema in plugin.get_tool_schemas():
                if getattr(schema, 'discoverability', 'discoverable') == 'core':
                    return True
        except Exception:
            pass
        return False

    def get_system_instructions(
        self,
        run_enrichment: bool = True,
        skip_discoverable_only: bool = False,
    ) -> Optional[str]:
        """Combine system instructions from all exposed plugins.

        Collects raw system instructions from all exposed plugins, then
        optionally runs them through the system instruction enrichment
        pipeline (e.g., template extraction).

        Args:
            run_enrichment: If True (default), run system instruction
                enrichment plugins after combining raw instructions.
            skip_discoverable_only: If True, skip plugins whose tools are
                all discoverable (no core tools).  Used when deferred tool
                loading is enabled so that instructions for undiscovered
                tools do not pollute the initial model context.

        Returns:
            Combined (and optionally enriched) system instructions string,
            or None if no plugins have instructions.
        """
        instructions = []
        for name in self._exposed:
            try:
                if skip_discoverable_only and not self.plugin_has_core_tools(name):
                    continue
                plugin_instructions = self._plugins[name].get_system_instructions()
                if plugin_instructions:
                    instructions.append(plugin_instructions)
            except Exception as exc:
                _trace(f" Error getting system instructions from '{name}': {exc}", include_traceback=True)

        if not instructions:
            return None

        combined = "\n\n".join(instructions)

        # Run through enrichment pipeline if enabled
        if run_enrichment:
            result = self.enrich_system_instructions(combined)
            return result.instructions

        return combined

    def get_auto_approved_tools(self) -> List[str]:
        """Collect auto-approved tool names from all exposed plugins and core tools.

        Returns:
            List of tool names that should be whitelisted for permission checks.
        """
        tools = []
        # Include core auto-approved tools
        tools.extend(self._core_auto_approved)
        # Include plugin auto-approved tools
        for name in self._exposed:
            try:
                if hasattr(self._plugins[name], 'get_auto_approved_tools'):
                    auto_approved = self._plugins[name].get_auto_approved_tools()
                    if auto_approved:
                        tools.extend(auto_approved)
            except Exception as exc:
                _trace(f" Error getting auto-approved tools from '{name}': {exc}", include_traceback=True)
        return tools

    def get_exposed_user_commands(self) -> List[UserCommand]:
        """Collect user-facing commands from all exposed plugins.

        User commands are different from model tools - they are commands
        that users can type directly in the interactive client.

        Each UserCommand includes:
        - name: Command name for invocation and autocompletion
        - description: Brief description shown in autocompletion/help
        - share_with_model: If True, output is added to conversation history

        Returns:
            List of UserCommand objects from all exposed plugins.
        """
        commands: List[UserCommand] = []
        for name in self._exposed:
            try:
                if hasattr(self._plugins[name], 'get_user_commands'):
                    user_commands = self._plugins[name].get_user_commands()
                    if user_commands:
                        commands.extend(user_commands)
            except Exception as exc:
                _trace(f" Error getting user commands from '{name}': {exc}", include_traceback=True)
        return commands

    def get_plugin_for_tool(self, tool_name: str) -> Optional['ToolPlugin']:
        """Get the plugin that provides a specific tool.

        Uses a lazy cache (``_tool_plugin_cache``) that is populated on first
        lookup and invalidated when plugins are exposed/unexposed.  For dynamic
        plugins (e.g. MCP after ``mcp reload``), a cache hit is validated by
        checking the tool still exists in the plugin's executors; a stale entry
        is evicted and the full scan re-runs.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The ToolPlugin instance that provides this tool, or None if not found.
        """
        # Fast path: cached lookup
        cached = self._tool_plugin_cache.get(tool_name)
        if cached is not None:
            # Validate the cache entry — the plugin may have dropped this tool
            # (e.g. MCP reload removed a server).
            if tool_name in cached.get_executors():
                return cached
            # Stale entry — evict and fall through to full scan
            del self._tool_plugin_cache[tool_name]

        # Slow path: scan all exposed plugins
        _trace(f" get_plugin_for_tool: cache miss for '{tool_name}', scanning {len(self._exposed)} plugins")
        for name in self._exposed:
            try:
                plugin = self._plugins[name]
                executors = plugin.get_executors()
                if tool_name in executors:
                    _trace(f" get_plugin_for_tool: FOUND '{tool_name}' in plugin '{name}'")
                    self._tool_plugin_cache[tool_name] = plugin
                    return plugin
            except Exception as exc:
                _trace(f" get_plugin_for_tool: error getting executors from '{name}': {exc}", include_traceback=True)
        _trace(f" get_plugin_for_tool: '{tool_name}' NOT FOUND in any plugin")
        return None

    def invalidate_tool_cache(self) -> None:
        """Clear the tool-to-plugin lookup cache.

        Call this when a plugin's executor set changes outside the normal
        expose/unexpose lifecycle (e.g. after ``mcp reload``).
        """
        self._tool_plugin_cache.clear()

    def list_skipped_plugins(self) -> Dict[str, List[str]]:
        """List plugins that were skipped due to model requirements.

        Returns:
            Dict mapping plugin names to their required model patterns.
        """
        return dict(self._skipped_plugins)

    # ==================== External Path Authorization ====================

    def authorize_external_path(
        self,
        path: str,
        source_plugin: str,
        access: str = "readwrite"
    ) -> None:
        """Authorize an external path for model access.

        Plugins (like references) can use this to allow the model to access
        specific files outside the workspace that are needed for the task.

        Args:
            path: Absolute path to authorize (will be normalized).
            source_plugin: Name of the plugin granting authorization.
            access: Access mode - "readonly" for read-only, "readwrite" for
                   full access (default: "readwrite").
        """
        if access not in ("readonly", "readwrite"):
            raise ValueError(f"Invalid access mode: {access!r}. Must be 'readonly' or 'readwrite'.")
        # Normalize to absolute path
        normalized = os.path.realpath(os.path.abspath(path))
        self._authorized_external_paths[normalized] = (source_plugin, access)
        _trace(f"authorize_external_path: {normalized} (from {source_plugin}, access={access})")

    def authorize_external_paths(
        self,
        paths: List[str],
        source_plugin: str,
        access: str = "readwrite"
    ) -> None:
        """Authorize multiple external paths for model access.

        Args:
            paths: List of absolute paths to authorize.
            source_plugin: Name of the plugin granting authorization.
            access: Access mode - "readonly" for read-only, "readwrite" for
                   full access (default: "readwrite").
        """
        for path in paths:
            self.authorize_external_path(path, source_plugin, access=access)

    def deauthorize_external_path(self, path: str, source_plugin: str) -> bool:
        """Remove authorization for a single external path.

        Only removes the path if it was authorized by the specified plugin.
        This is the single-path counterpart to clear_authorized_paths().

        Args:
            path: Absolute path to deauthorize (will be normalized).
            source_plugin: Name of the plugin that originally authorized it.
                Only removes if the current authorization matches this plugin.

        Returns:
            True if the path was found and removed, False otherwise.
        """
        normalized = os.path.realpath(os.path.abspath(path))
        entry = self._authorized_external_paths.get(normalized)
        if entry and entry[0] == source_plugin:
            del self._authorized_external_paths[normalized]
            _trace(f"deauthorize_external_path: {normalized} (from {source_plugin})")
            return True
        return False

    def is_path_authorized(self, path: str, mode: str = "read") -> bool:
        """Check if a path is authorized for model access.

        This checks both exact path matches and parent directory matches
        (if a directory is authorized, all files within it are authorized).

        Args:
            path: Path to check (will be normalized).
            mode: Required access mode - "read" or "write" (default: "read").
                 A path authorized as "readwrite" satisfies both "read" and
                 "write" mode checks. A path authorized as "readonly" only
                 satisfies "read" mode checks.

        Returns:
            True if the path or a parent directory is authorized with
            sufficient access.
        """
        if not self._authorized_external_paths:
            return False

        # Normalize the path
        normalized = os.path.realpath(os.path.abspath(path))

        def _access_sufficient(entry_access: str) -> bool:
            """Check if the authorized access level is sufficient."""
            if mode == "read":
                return True  # Both "readonly" and "readwrite" allow reads
            # mode == "write": only "readwrite" allows writes
            return entry_access == "readwrite"

        # Check exact match
        if normalized in self._authorized_external_paths:
            _source, entry_access = self._authorized_external_paths[normalized]
            return _access_sufficient(entry_access)

        # Check if any authorized path is a parent directory
        for authorized_path, (_, entry_access) in self._authorized_external_paths.items():
            # Check if normalized path is under an authorized directory
            auth_with_sep = authorized_path.rstrip(os.sep) + os.sep
            if normalized.startswith(auth_with_sep):
                return _access_sufficient(entry_access)

        return False

    def get_path_authorization_source(self, path: str) -> Optional[str]:
        """Get the plugin that authorized a path.

        Args:
            path: Path to check (will be normalized).

        Returns:
            Name of the plugin that authorized this path, or None if not authorized.
        """
        if not self._authorized_external_paths:
            return None

        normalized = os.path.realpath(os.path.abspath(path))

        # Check exact match
        if normalized in self._authorized_external_paths:
            source, _access = self._authorized_external_paths[normalized]
            return source

        # Check parent directories
        for authorized_path, (source, _access) in self._authorized_external_paths.items():
            auth_with_sep = authorized_path.rstrip(os.sep) + os.sep
            if normalized.startswith(auth_with_sep):
                return source

        return None

    def get_path_access_mode(self, path: str) -> Optional[str]:
        """Get the access mode for an authorized path.

        Args:
            path: Path to check (will be normalized).

        Returns:
            Access mode ("readonly" or "readwrite"), or None if not authorized.
        """
        if not self._authorized_external_paths:
            return None

        normalized = os.path.realpath(os.path.abspath(path))

        # Check exact match
        if normalized in self._authorized_external_paths:
            _source, access = self._authorized_external_paths[normalized]
            return access

        # Check parent directories
        for authorized_path, (_source, access) in self._authorized_external_paths.items():
            auth_with_sep = authorized_path.rstrip(os.sep) + os.sep
            if normalized.startswith(auth_with_sep):
                return access

        return None

    def clear_authorized_paths(self, source_plugin: Optional[str] = None) -> int:
        """Clear authorized external paths.

        Args:
            source_plugin: If specified, only clear paths from this plugin.
                          If None, clear all authorized paths.

        Returns:
            Number of paths cleared.
        """
        if source_plugin is None:
            count = len(self._authorized_external_paths)
            self._authorized_external_paths.clear()
            _trace(f"clear_authorized_paths: cleared all {count} paths")
            return count

        # Clear only paths from the specified plugin
        to_remove = [
            path for path, (source, _access) in self._authorized_external_paths.items()
            if source == source_plugin
        ]
        for path in to_remove:
            del self._authorized_external_paths[path]

        _trace(f"clear_authorized_paths: cleared {len(to_remove)} paths from {source_plugin}")
        return len(to_remove)

    def list_authorized_paths(self) -> Dict[str, str]:
        """List all authorized external paths.

        Returns:
            Dict mapping normalized paths to the source plugin that authorized them.

        Note:
            For backward compatibility this returns source plugin names as values.
            Use list_authorized_paths_detailed() to get access mode info.
        """
        return {path: source for path, (source, _access) in self._authorized_external_paths.items()}

    def list_authorized_paths_detailed(self) -> Dict[str, Dict[str, str]]:
        """List all authorized external paths with access mode details.

        Returns:
            Dict mapping normalized paths to {"source": plugin_name, "access": mode}.
        """
        return {
            path: {"source": source, "access": access}
            for path, (source, access) in self._authorized_external_paths.items()
        }

    # ==================== External Path Denial ====================

    def deny_external_path(self, path: str, source_plugin: str) -> None:
        """Deny an external path, blocking model access even if otherwise allowed.

        Denied paths take precedence over authorized paths. This is used by
        the sandbox_manager plugin to implement session-level path blocking.

        Args:
            path: Absolute path to deny (will be normalized).
            source_plugin: Name of the plugin denying access.
        """
        # Normalize to absolute path
        normalized = os.path.realpath(os.path.abspath(path))
        self._denied_external_paths[normalized] = source_plugin
        _trace(f"deny_external_path: {normalized} (from {source_plugin})")

    def deny_external_paths(self, paths: List[str], source_plugin: str) -> None:
        """Deny multiple external paths.

        Args:
            paths: List of absolute paths to deny.
            source_plugin: Name of the plugin denying access.
        """
        for path in paths:
            self.deny_external_path(path, source_plugin)

    def is_path_denied(self, path: str) -> bool:
        """Check if a path is denied for model access.

        This checks both exact path matches and parent directory matches
        (if a directory is denied, all files within it are denied).

        Denial takes precedence over authorization - a path can be both
        authorized and denied, and denial wins.

        Args:
            path: Path to check (will be normalized).

        Returns:
            True if the path or a parent directory is denied.
        """
        if not self._denied_external_paths:
            return False

        # Normalize the path
        normalized = os.path.realpath(os.path.abspath(path))

        # Check exact match
        if normalized in self._denied_external_paths:
            return True

        # Check if any denied path is a parent directory
        for denied_path in self._denied_external_paths:
            # Check if normalized path is under a denied directory
            denied_with_sep = denied_path.rstrip(os.sep) + os.sep
            if normalized.startswith(denied_with_sep):
                return True

        return False

    def get_path_denial_source(self, path: str) -> Optional[str]:
        """Get the plugin that denied a path.

        Args:
            path: Path to check (will be normalized).

        Returns:
            Name of the plugin that denied this path, or None if not denied.
        """
        if not self._denied_external_paths:
            return None

        normalized = os.path.realpath(os.path.abspath(path))

        # Check exact match
        if normalized in self._denied_external_paths:
            return self._denied_external_paths[normalized]

        # Check parent directories
        for denied_path, source in self._denied_external_paths.items():
            denied_with_sep = denied_path.rstrip(os.sep) + os.sep
            if normalized.startswith(denied_with_sep):
                return source

        return None

    def clear_denied_paths(self, source_plugin: Optional[str] = None) -> int:
        """Clear denied external paths.

        Args:
            source_plugin: If specified, only clear paths from this plugin.
                          If None, clear all denied paths.

        Returns:
            Number of paths cleared.
        """
        if source_plugin is None:
            count = len(self._denied_external_paths)
            self._denied_external_paths.clear()
            _trace(f"clear_denied_paths: cleared all {count} paths")
            return count

        # Clear only paths from the specified plugin
        to_remove = [
            path for path, source in self._denied_external_paths.items()
            if source == source_plugin
        ]
        for path in to_remove:
            del self._denied_external_paths[path]

        _trace(f"clear_denied_paths: cleared {len(to_remove)} paths from {source_plugin}")
        return len(to_remove)

    def list_denied_paths(self) -> Dict[str, str]:
        """List all denied external paths.

        Returns:
            Dict mapping normalized paths to the source plugin that denied them.
        """
        return dict(self._denied_external_paths)

    # ==================== Prompt Enrichment ====================

    def _get_enrichment_priority(self, plugin: ToolPlugin) -> int:
        """Get the enrichment priority for a plugin.

        Lower values run first. Default is 50.

        Standard priorities:
        - 20: references (injects content first)
        - 40: template (scans injected content for templates)
        - 60: multimodal (handles @image references)
        - 80: memory (adds memory hints last)

        Args:
            plugin: The plugin to get priority for.

        Returns:
            The enrichment priority (lower = earlier).
        """
        if hasattr(plugin, 'get_enrichment_priority'):
            return plugin.get_enrichment_priority()
        return 50  # Default priority

    def get_prompt_enrichment_subscribers(self) -> List[ToolPlugin]:
        """Get plugins that subscribe to prompt enrichment, sorted by priority.

        Includes both exposed plugins and enrichment-only plugins.
        Plugins are sorted by enrichment priority (lower values run first).

        Returns:
            List of plugins that have subscribed to prompt enrichment,
            sorted by priority.
        """
        subscribers = []
        # Include both exposed and enrichment-only plugins
        all_enrichment_names = self._exposed | self._enrichment_only
        for name in all_enrichment_names:
            try:
                plugin = self._plugins[name]
                if (hasattr(plugin, 'subscribes_to_prompt_enrichment') and
                        plugin.subscribes_to_prompt_enrichment()):
                    subscribers.append(plugin)
            except Exception as exc:
                _trace(f" Error checking enrichment subscription for '{name}': {exc}", include_traceback=True)

        # Sort by priority (lower values first)
        subscribers.sort(key=self._get_enrichment_priority)
        return subscribers

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Run prompt through all subscribed enrichment plugins.

        Each subscribed plugin gets to inspect and optionally modify the prompt.
        Plugins are called in priority order (lower priority values first).

        Args:
            prompt: The user's original prompt text.

        Returns:
            PromptEnrichmentResult with the enriched prompt and combined metadata.
        """
        current_prompt = prompt
        combined_metadata: Dict[str, Any] = {}
        notifications: List[EnrichmentNotification] = []

        for plugin in self.get_prompt_enrichment_subscribers():
            try:
                if hasattr(plugin, 'enrich_prompt'):
                    before = current_prompt
                    result = plugin.enrich_prompt(current_prompt)
                    current_prompt = result.prompt
                    # Merge metadata, using plugin name as namespace
                    if result.metadata:
                        combined_metadata[plugin.name] = result.metadata

                    # Build notification if content changed
                    if current_prompt != before:
                        notif = self._build_notification(
                            target="prompt",
                            plugin_name=plugin.name,
                            metadata=result.metadata,
                            size_bytes=len(current_prompt) - len(before)
                        )
                        if notif:
                            notifications.append(notif)
            except Exception as exc:
                _trace(f" Error in prompt enrichment for '{plugin.name}': {exc}", include_traceback=True)

        # Emit notifications
        self._emit_enrichment_notifications(notifications)

        return PromptEnrichmentResult(prompt=current_prompt, metadata=combined_metadata)

    def _get_system_instruction_enrichment_priority(self, plugin: ToolPlugin) -> int:
        """Get the system instruction enrichment priority for a plugin.

        Lower values run first. Default is 50.

        Args:
            plugin: The plugin to get priority for.

        Returns:
            The enrichment priority (lower = earlier).
        """
        if hasattr(plugin, 'get_system_instruction_enrichment_priority'):
            return plugin.get_system_instruction_enrichment_priority()
        return 50  # Default priority

    def get_system_instruction_enrichment_subscribers(self) -> List[ToolPlugin]:
        """Get plugins that subscribe to system instruction enrichment.

        Includes both exposed plugins and enrichment-only plugins.
        Plugins are sorted by priority (lower values run first).

        Returns:
            List of plugins that subscribe to system instruction enrichment,
            sorted by priority.
        """
        subscribers = []
        all_enrichment_names = self._exposed | self._enrichment_only
        for name in all_enrichment_names:
            try:
                plugin = self._plugins[name]
                if (hasattr(plugin, 'subscribes_to_system_instruction_enrichment') and
                        plugin.subscribes_to_system_instruction_enrichment()):
                    subscribers.append(plugin)
            except Exception as exc:
                _trace(f" Error checking system instruction enrichment for '{name}': {exc}")

        subscribers.sort(key=self._get_system_instruction_enrichment_priority)
        return subscribers

    def enrich_system_instructions(
        self,
        instructions: str
    ) -> SystemInstructionEnrichmentResult:
        """Run system instructions through all subscribed enrichment plugins.

        Each subscribed plugin gets to inspect and optionally modify the
        combined system instructions. This is called after collecting raw
        instructions from all plugins.

        Args:
            instructions: Combined system instructions text.

        Returns:
            SystemInstructionEnrichmentResult with enriched instructions.
        """
        current_instructions = instructions
        combined_metadata: Dict[str, Any] = {}
        notifications: List[EnrichmentNotification] = []

        for plugin in self.get_system_instruction_enrichment_subscribers():
            try:
                if hasattr(plugin, 'enrich_system_instructions'):
                    before = current_instructions
                    result = plugin.enrich_system_instructions(current_instructions)
                    current_instructions = result.instructions
                    if result.metadata:
                        combined_metadata[plugin.name] = result.metadata

                    # Build notification if content changed
                    if current_instructions != before:
                        notif = self._build_notification(
                            target="system",
                            plugin_name=plugin.name,
                            metadata=result.metadata,
                            size_bytes=len(current_instructions) - len(before)
                        )
                        if notif:
                            notifications.append(notif)
            except Exception as exc:
                _trace(f" Error in system instruction enrichment for '{plugin.name}': {exc}")

        # Emit notifications
        self._emit_enrichment_notifications(notifications)

        return SystemInstructionEnrichmentResult(
            instructions=current_instructions,
            metadata=combined_metadata
        )

    def _get_tool_result_enrichment_priority(self, plugin: ToolPlugin) -> int:
        """Get the tool result enrichment priority for a plugin.

        Lower values run first. Default is 50.

        Args:
            plugin: The plugin to get priority for.

        Returns:
            The enrichment priority (lower = earlier).
        """
        if hasattr(plugin, 'get_tool_result_enrichment_priority'):
            return plugin.get_tool_result_enrichment_priority()
        return 50  # Default priority

    def get_tool_result_enrichment_subscribers(self) -> List[ToolPlugin]:
        """Get plugins that subscribe to tool result enrichment.

        Includes both exposed plugins and enrichment-only plugins.
        Plugins are sorted by priority (lower values run first).

        Returns:
            List of plugins that subscribe to tool result enrichment,
            sorted by priority.
        """
        subscribers = []
        all_enrichment_names = self._exposed | self._enrichment_only
        for name in all_enrichment_names:
            try:
                plugin = self._plugins[name]
                if (hasattr(plugin, 'subscribes_to_tool_result_enrichment') and
                        plugin.subscribes_to_tool_result_enrichment()):
                    subscribers.append(plugin)
            except Exception as exc:
                _trace(f" Error checking tool result enrichment for '{name}': {exc}")

        subscribers.sort(key=self._get_tool_result_enrichment_priority)
        return subscribers

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str,
        output_callback: Optional[OutputCallback] = None,
        terminal_width: Optional[int] = None
    ) -> ToolResultEnrichmentResult:
        """Run a tool result through all subscribed enrichment plugins.

        Each subscribed plugin gets to inspect and optionally modify the
        tool result before it is sent back to the model.

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's output as a string.
            output_callback: Optional callback for emitting notifications.
                If provided, this callback is used instead of the registry's
                shared callback. This is important for concurrent sessions
                (e.g., subagents) that need notifications routed to their
                specific output panel.
            terminal_width: Terminal width for formatting (uses registry default if not provided).

        Returns:
            ToolResultEnrichmentResult with enriched result.
        """
        current_result = result
        combined_metadata: Dict[str, Any] = {}
        notifications: List[EnrichmentNotification] = []

        for plugin in self.get_tool_result_enrichment_subscribers():
            try:
                if hasattr(plugin, 'enrich_tool_result'):
                    before = current_result
                    enrichment = plugin.enrich_tool_result(tool_name, current_result)
                    current_result = enrichment.result
                    if enrichment.metadata:
                        combined_metadata[plugin.name] = enrichment.metadata

                    # Build notification if content changed
                    if current_result != before:
                        notif = self._build_notification(
                            target="result",
                            plugin_name=plugin.name,
                            metadata=enrichment.metadata,
                            size_bytes=len(current_result) - len(before)
                        )
                        if notif:
                            notifications.append(notif)
            except Exception as exc:
                _trace(f" Error in tool result enrichment for '{plugin.name}': {exc}")

        # Emit notifications using provided callback or fallback to shared one
        self._emit_enrichment_notifications(
            notifications,
            output_callback=output_callback,
            terminal_width=terminal_width
        )

        return ToolResultEnrichmentResult(
            result=current_result,
            metadata=combined_metadata
        )
