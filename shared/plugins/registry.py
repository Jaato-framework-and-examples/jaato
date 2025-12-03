"""Plugin registry for discovering, loading, and managing tool plugins."""

import importlib
import importlib.metadata
import pkgutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Callable, Any, Optional
from google.genai import types

from .base import ToolPlugin

# Entry point group name for jaato plugins
# External packages can register plugins by adding to this group in their pyproject.toml:
#   [project.entry-points."jaato.plugins"]
#   my_plugin = "my_package.plugins:create_plugin"
PLUGIN_ENTRY_POINT_GROUP = "jaato.plugins"


class PluginRegistry:
    """Manages plugin discovery, lifecycle, and tool exposure state.

    Usage:
        registry = PluginRegistry()
        registry.discover()

        print(registry.list_available())  # ['cli', 'mcp', ...]

        registry.expose_tool('cli', config={'extra_paths': ['/usr/local/bin']})
        registry.expose_tool('mcp')

        # Get tools for exposed plugins
        declarations = registry.get_exposed_declarations()
        executors = registry.get_exposed_executors()

        # Later, unexpose plugins
        registry.unexpose_tool('mcp')
        registry.unexpose_all()
    """

    def __init__(self):
        self._plugins: Dict[str, ToolPlugin] = {}
        self._exposed: Set[str] = set()
        self._configs: Dict[str, Dict[str, Any]] = {}

    def discover(self, include_directory: bool = True) -> List[str]:
        """Discover plugins via entry points and optionally directory scanning.

        Discovery order:
        1. Entry points (jaato.plugins group) - for installed packages
        2. Directory scanning (optional) - for development/local plugins

        Entry points allow external packages to register plugins:
            [project.entry-points."jaato.plugins"]
            my_plugin = "my_package.plugins:create_plugin"

        Args:
            include_directory: Also scan the plugins directory for local plugins.
                             Useful during development when package isn't installed.
                             Default: True

        Returns:
            List of discovered plugin names.
        """
        discovered = []

        # First, discover via entry points (installed packages)
        discovered.extend(self._discover_via_entry_points())

        # Then, optionally scan the plugins directory (development mode)
        if include_directory:
            discovered.extend(self._discover_via_directory())

        return discovered

    def _discover_via_entry_points(self) -> List[str]:
        """Discover plugins registered via entry points.

        External packages can register plugins by adding to the jaato.plugins
        entry point group in their pyproject.toml.

        Returns:
            List of discovered plugin names.
        """
        discovered = []

        try:
            # Python 3.10+ API
            if sys.version_info >= (3, 10):
                eps = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
            else:
                # Python 3.9 compatibility
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get(PLUGIN_ENTRY_POINT_GROUP, [])

            for ep in eps:
                # Skip if already loaded (avoid duplicates with directory scan)
                if ep.name in self._plugins:
                    continue

                try:
                    create_plugin = ep.load()
                    plugin = create_plugin()

                    if isinstance(plugin, ToolPlugin):
                        self._plugins[plugin.name] = plugin
                        discovered.append(plugin.name)
                    else:
                        print(f"[PluginRegistry] Entry point '{ep.name}': "
                              f"plugin does not implement ToolPlugin protocol")

                except Exception as exc:
                    print(f"[PluginRegistry] Error loading entry point '{ep.name}': {exc}")

        except Exception as exc:
            # Entry points not available (package not installed)
            pass

        return discovered

    def _discover_via_directory(self, plugin_dir: Optional[Path] = None) -> List[str]:
        """Discover plugins by scanning the plugins directory.

        This is the fallback/development mode discovery that scans for Python
        modules with a create_plugin() factory function.

        Args:
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

                if hasattr(module, 'create_plugin'):
                    plugin = module.create_plugin()

                    # Verify it implements the protocol
                    if isinstance(plugin, ToolPlugin):
                        self._plugins[plugin.name] = plugin
                        discovered.append(plugin.name)
                    else:
                        print(f"[PluginRegistry] {name}: plugin does not implement ToolPlugin protocol")
                else:
                    print(f"[PluginRegistry] {name}: no create_plugin() function found")

            except Exception as exc:
                print(f"[PluginRegistry] Error loading plugin '{name}': {exc}")

        return discovered

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

    def expose_tool(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Expose a plugin's tools to the model.

        Calls the plugin's initialize() method if this is the first time
        exposing it, or if a new config is provided.

        Args:
            name: Plugin name to expose.
            config: Optional configuration dict for the plugin.

        Raises:
            ValueError: If the plugin is not found.
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found. Available: {self.list_available()}")

        plugin = self._plugins[name]

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

    def get_exposed_declarations(self) -> List[types.FunctionDeclaration]:
        """Get FunctionDeclarations from all exposed plugins."""
        decls = []
        for name in self._exposed:
            try:
                decls.extend(self._plugins[name].get_function_declarations())
            except Exception as exc:
                print(f"[PluginRegistry] Error getting declarations from '{name}': {exc}")
        return decls

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
