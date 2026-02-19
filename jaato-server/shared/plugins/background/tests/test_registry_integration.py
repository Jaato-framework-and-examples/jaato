"""Tests for background plugin integration with the plugin registry."""

import pytest
from typing import Any, Dict, Optional

from shared.plugins.registry import PluginRegistry
from shared.plugins.background import (
    BackgroundPlugin,
    BackgroundCapableMixin,
    create_plugin,
    PLUGIN_KIND,
)
from shared.plugins.background.protocol import BackgroundCapable


class TestPluginDiscovery:
    """Tests for plugin discovery via registry."""

    def test_plugin_kind_is_tool(self):
        """Test that the plugin kind is 'tool'."""
        assert PLUGIN_KIND == "tool"

    def test_create_plugin_factory(self):
        """Test the create_plugin factory function."""
        plugin = create_plugin()
        assert isinstance(plugin, BackgroundPlugin)
        assert plugin.name == "background"

    def test_plugin_implements_required_methods(self):
        """Test that plugin has all required ToolPlugin methods."""
        plugin = create_plugin()

        # Required ToolPlugin methods
        assert hasattr(plugin, 'name')
        assert hasattr(plugin, 'get_tool_schemas')
        assert hasattr(plugin, 'get_executors')
        assert hasattr(plugin, 'initialize')
        assert hasattr(plugin, 'shutdown')
        assert hasattr(plugin, 'get_system_instructions')
        assert hasattr(plugin, 'get_auto_approved_tools')
        assert hasattr(plugin, 'get_user_commands')

    def test_registry_discovers_background_plugin(self):
        """Test that registry discovers the background plugin."""
        registry = PluginRegistry()
        discovered = registry.discover(plugin_kind="tool")

        # Background plugin should be discovered
        assert "background" in discovered or "background" in registry.list_available()

    def test_registry_exposes_background_plugin(self):
        """Test exposing background plugin via registry."""
        registry = PluginRegistry()
        registry.discover(plugin_kind="tool")

        # Skip if not discovered (might not be in plugins path)
        if "background" not in registry.list_available():
            pytest.skip("Background plugin not in discovery path")

        registry.expose_tool("background")

        assert "background" in registry.list_exposed()

        # Check schemas are included
        schemas = registry.get_exposed_tool_schemas()
        schema_names = [s.name for s in schemas]

        assert "startBackgroundTask" in schema_names


class TestBackgroundCapableProtocolCheck:
    """Tests for BackgroundCapable protocol checking."""

    def test_mixin_satisfies_protocol(self):
        """Test that BackgroundCapableMixin satisfies BackgroundCapable protocol."""

        class TestPlugin(BackgroundCapableMixin):
            @property
            def name(self) -> str:
                return "test"

            def supports_background(self, tool_name: str) -> bool:
                return True

            def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
                return 10.0

        plugin = TestPlugin()

        # Should pass isinstance check
        assert isinstance(plugin, BackgroundCapable)

    def test_non_capable_plugin_fails_check(self):
        """Test that non-capable plugins don't pass the protocol check."""

        class NotCapablePlugin:
            @property
            def name(self) -> str:
                return "not_capable"

        plugin = NotCapablePlugin()

        # Should NOT pass isinstance check
        assert not isinstance(plugin, BackgroundCapable)


class TestRegistryWithCapablePlugins:
    """Tests for registry with BackgroundCapable plugins."""

    def test_background_plugin_discovers_capable_plugins(self):
        """Test that BackgroundPlugin discovers capable plugins in registry."""

        # Create a capable plugin
        class CapableTestPlugin(BackgroundCapableMixin):
            def __init__(self):
                super().__init__()

            @property
            def name(self) -> str:
                return "capable_test"

            def supports_background(self, tool_name: str) -> bool:
                return tool_name == "bg_tool"

            def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
                return 5.0 if tool_name == "bg_tool" else None

            def get_tool_schemas(self):
                return []

            def get_executors(self) -> Dict[str, Any]:
                return {"bg_tool": lambda args: {"done": True}}

            def initialize(self, config=None):
                pass

            def shutdown(self):
                pass

            def get_system_instructions(self):
                return None

            def get_auto_approved_tools(self):
                return []

            def get_user_commands(self):
                return []

        # Create a mock registry
        class MockRegistry:
            def __init__(self):
                self._plugins = {}

            def add_plugin(self, plugin):
                self._plugins[plugin.name] = plugin

            def list_exposed(self):
                return list(self._plugins.keys())

            def get_plugin(self, name):
                return self._plugins.get(name)

        registry = MockRegistry()
        capable_plugin = CapableTestPlugin()
        registry.add_plugin(capable_plugin)

        # Create and configure background plugin
        bg_plugin = BackgroundPlugin()
        bg_plugin.set_registry(registry)

        # Should have discovered the capable plugin
        assert "capable_test" in bg_plugin._capable_plugins

        # Should be able to list capable tools
        result = bg_plugin._list_capable_tools({})
        assert result["count"] > 0

        tool_info = result["tools"][0]
        assert tool_info["plugin_name"] == "capable_test"
        assert tool_info["tool_name"] == "bg_tool"
        assert tool_info["auto_background_threshold_seconds"] == 5.0
