"""Plugin registry for discovering, loading, and managing plugins."""

import importlib
import importlib.metadata
import pkgutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Callable, Any, Optional, Protocol, runtime_checkable

from .base import ToolPlugin, UserCommand, PromptEnrichmentResult, model_matches_requirements
from .model_provider.types import ToolSchema

# Entry point group names by plugin kind
PLUGIN_ENTRY_POINT_GROUPS = {
    "tool": "jaato.plugins",
    "gc": "jaato.gc_plugins",
}


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
                        print(f"[PluginRegistry] Entry point '{ep.name}': "
                              f"plugin does not implement ToolPlugin protocol")
                        continue

                    self._plugins[plugin.name] = plugin
                    discovered.append(plugin.name)

                except Exception as exc:
                    print(f"[PluginRegistry] Error loading entry point '{ep.name}': {exc}")

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
                        print(f"[PluginRegistry] {name}: plugin does not implement ToolPlugin protocol")
                        continue

                    self._plugins[plugin.name] = plugin
                    discovered.append(plugin.name)

            except Exception as exc:
                print(f"[PluginRegistry] Error loading plugin '{name}': {exc}")

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
                print(f"[PluginRegistry] Plugin '{name}' skipped: "
                      f"model '{self._model_name}' not in {requirements}")
                return False

        # Initialize if not already exposed, or if new config provided
        if name not in self._exposed:
            plugin.initialize(config)
            if config:
                self._configs[name] = config
            self._exposed.add(name)
        elif config and config != self._configs.get(name):
            # Re-initialize with new config
            plugin.shutdown()
            plugin.initialize(config)
            self._configs[name] = config

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

    def expose_all(self, config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Expose all discovered plugins' tools.

        Args:
            config: Optional dict mapping plugin names to their configs.
        """
        config = config or {}
        for name in self._plugins:
            self.expose_tool(name, config.get(name))

    def unexpose_all(self) -> None:
        """Stop exposing all plugins' tools."""
        for name in list(self._exposed):
            self.unexpose_tool(name)

    def get_exposed_tool_schemas(self) -> List[ToolSchema]:
        """Get ToolSchemas from all exposed plugins."""
        schemas = []
        for name in self._exposed:
            try:
                schemas.extend(self._plugins[name].get_tool_schemas())
            except Exception as exc:
                print(f"[PluginRegistry] Error getting tool schemas from '{name}': {exc}")
        return schemas

    def get_exposed_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Get executor callables from all exposed plugins."""
        executors = {}
        for name in self._exposed:
            try:
                executors.update(self._plugins[name].get_executors())
            except Exception as exc:
                print(f"[PluginRegistry] Error getting executors from '{name}': {exc}")
        return executors

    def get_system_instructions(self) -> Optional[str]:
        """Combine system instructions from all exposed plugins.

        Returns:
            Combined system instructions string, or None if no plugins
            have instructions.
        """
        instructions = []
        for name in self._exposed:
            try:
                plugin_instructions = self._plugins[name].get_system_instructions()
                if plugin_instructions:
                    instructions.append(plugin_instructions)
            except Exception as exc:
                print(f"[PluginRegistry] Error getting system instructions from '{name}': {exc}")

        if not instructions:
            return None

        return "\n\n".join(instructions)

    def get_auto_approved_tools(self) -> List[str]:
        """Collect auto-approved tool names from all exposed plugins.

        Returns:
            List of tool names that should be whitelisted for permission checks.
        """
        tools = []
        for name in self._exposed:
            try:
                if hasattr(self._plugins[name], 'get_auto_approved_tools'):
                    auto_approved = self._plugins[name].get_auto_approved_tools()
                    if auto_approved:
                        tools.extend(auto_approved)
            except Exception as exc:
                print(f"[PluginRegistry] Error getting auto-approved tools from '{name}': {exc}")
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
                print(f"[PluginRegistry] Error getting user commands from '{name}': {exc}")
        return commands

    def get_plugin_for_tool(self, tool_name: str) -> Optional['ToolPlugin']:
        """Get the plugin that provides a specific tool.

        This is useful for the permission system to call plugin-specific
        formatting methods when displaying tool execution requests.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The ToolPlugin instance that provides this tool, or None if not found.
        """
        for name in self._exposed:
            try:
                plugin = self._plugins[name]
                if tool_name in plugin.get_executors():
                    return plugin
            except Exception:
                pass
        return None

    def list_skipped_plugins(self) -> Dict[str, List[str]]:
        """List plugins that were skipped due to model requirements.

        Returns:
            Dict mapping plugin names to their required model patterns.
        """
        return dict(self._skipped_plugins)

    # ==================== Prompt Enrichment ====================

    def get_prompt_enrichment_subscribers(self) -> List[ToolPlugin]:
        """Get plugins that subscribe to prompt enrichment.

        Includes both exposed plugins and enrichment-only plugins.

        Returns:
            List of plugins that have subscribed to prompt enrichment.
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
                print(f"[PluginRegistry] Error checking enrichment subscription for '{name}': {exc}")
        return subscribers

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Run prompt through all subscribed enrichment plugins.

        Each subscribed plugin gets to inspect and optionally modify the prompt.
        Plugins are called in exposure order.

        Args:
            prompt: The user's original prompt text.

        Returns:
            PromptEnrichmentResult with the enriched prompt and combined metadata.
        """
        current_prompt = prompt
        combined_metadata: Dict[str, Any] = {}

        for plugin in self.get_prompt_enrichment_subscribers():
            try:
                if hasattr(plugin, 'enrich_prompt'):
                    result = plugin.enrich_prompt(current_prompt)
                    current_prompt = result.prompt
                    # Merge metadata, using plugin name as namespace
                    if result.metadata:
                        combined_metadata[plugin.name] = result.metadata
            except Exception as exc:
                print(f"[PluginRegistry] Error in prompt enrichment for '{plugin.name}': {exc}")

        return PromptEnrichmentResult(prompt=current_prompt, metadata=combined_metadata)
