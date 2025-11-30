"""Plugin registry for discovering, loading, and managing tool plugins."""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Set, Callable, Any, Optional
from google.genai import types

from .base import ToolPlugin


class PluginRegistry:
    """Manages plugin discovery, lifecycle, and enable/disable state.

    Usage:
        registry = PluginRegistry()
        registry.discover()

        print(registry.list_available())  # ['cli', 'mcp', ...]

        registry.enable('cli', config={'extra_paths': ['/usr/local/bin']})
        registry.enable('mcp')

        # Get tools for enabled plugins
        declarations = registry.get_enabled_declarations()
        executors = registry.get_enabled_executors()

        # Later, disable plugins
        registry.disable('mcp')
        registry.disable_all()
    """

    def __init__(self):
        self._plugins: Dict[str, ToolPlugin] = {}
        self._enabled: Set[str] = set()
        self._configs: Dict[str, Dict[str, Any]] = {}

    def discover(self, plugin_dir: Optional[Path] = None) -> List[str]:
        """Discover all plugins from the plugins directory.

        Scans the plugin directory for Python modules that export a
        `create_plugin()` factory function, and instantiates each plugin.

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

    def list_enabled(self) -> List[str]:
        """List currently enabled plugin names."""
        return list(self._enabled)

    def is_enabled(self, name: str) -> bool:
        """Check if a plugin is currently enabled."""
        return name in self._enabled

    def get_plugin(self, name: str) -> Optional[ToolPlugin]:
        """Get a plugin by name, or None if not found."""
        return self._plugins.get(name)

    def enable(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Enable a plugin.

        Calls the plugin's initialize() method if this is the first time
        enabling it, or if a new config is provided.

        Args:
            name: Plugin name to enable.
            config: Optional configuration dict for the plugin.

        Raises:
            ValueError: If the plugin is not found.
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found. Available: {self.list_available()}")

        plugin = self._plugins[name]

        # Initialize if not already enabled, or if new config provided
        if name not in self._enabled:
            plugin.initialize(config)
            if config:
                self._configs[name] = config
            self._enabled.add(name)
        elif config and config != self._configs.get(name):
            # Re-initialize with new config
            plugin.shutdown()
            plugin.initialize(config)
            self._configs[name] = config

    def disable(self, name: str) -> None:
        """Disable a plugin.

        Calls the plugin's shutdown() method to clean up resources.

        Args:
            name: Plugin name to disable.
        """
        if name in self._enabled:
            self._plugins[name].shutdown()
            self._enabled.discard(name)
            self._configs.pop(name, None)

    def enable_all(self, config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Enable all discovered plugins.

        Args:
            config: Optional dict mapping plugin names to their configs.
        """
        config = config or {}
        for name in self._plugins:
            self.enable(name, config.get(name))

    def disable_all(self) -> None:
        """Disable all enabled plugins."""
        for name in list(self._enabled):
            self.disable(name)

    def get_enabled_declarations(self) -> List[types.FunctionDeclaration]:
        """Get FunctionDeclarations from all enabled plugins."""
        decls = []
        for name in self._enabled:
            try:
                decls.extend(self._plugins[name].get_function_declarations())
            except Exception as exc:
                print(f"[PluginRegistry] Error getting declarations from '{name}': {exc}")
        return decls

    def get_enabled_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Get executor callables from all enabled plugins."""
        executors = {}
        for name in self._enabled:
            try:
                executors.update(self._plugins[name].get_executors())
            except Exception as exc:
                print(f"[PluginRegistry] Error getting executors from '{name}': {exc}")
        return executors
