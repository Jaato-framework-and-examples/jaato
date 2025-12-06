"""Tests for web_search plugin integration with the plugin registry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import WebSearchPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the web_search plugin via the registry."""

    def test_web_search_plugin_discovered(self):
        """Test that web_search plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "web_search" in discovered

    def test_web_search_plugin_available(self):
        """Test that web_search plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "web_search" in registry.list_available()

    def test_get_web_search_plugin(self):
        """Test retrieving the web_search plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("web_search")
        assert plugin is not None
        assert plugin.name == "web_search"


class TestRegistryExposeWebSearchPlugin:
    """Tests for exposing the web_search plugin via the registry."""

    def test_expose_web_search_plugin(self):
        """Test exposing the web_search plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")

        assert registry.is_exposed("web_search")
        assert "web_search" in registry.list_exposed()

        registry.unexpose_tool("web_search")

    def test_expose_web_search_plugin_with_config(self):
        """Test exposing the web_search plugin with configuration."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search", config={
            "max_results": 5,
            "safesearch": "strict"
        })

        assert registry.is_exposed("web_search")
        plugin = registry.get_plugin("web_search")
        assert plugin._max_results == 5
        assert plugin._safesearch == "strict"

        registry.unexpose_tool("web_search")

    def test_unexpose_web_search_plugin(self):
        """Test unexposing the web_search plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")
        assert registry.is_exposed("web_search")

        registry.unexpose_tool("web_search")
        assert not registry.is_exposed("web_search")

    def test_expose_all_includes_web_search(self):
        """Test that expose_all includes the web_search plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("web_search")

        registry.unexpose_all()


class TestRegistryWebSearchToolDeclarations:
    """Tests for web_search tool declarations exposure via registry."""

    def test_web_search_not_exposed_before_expose(self):
        """Test that web_search is not in declarations before expose."""
        registry = PluginRegistry()
        registry.discover()

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "web_search" not in tool_names

    def test_web_search_exposed_after_expose(self):
        """Test that web_search is in declarations after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "web_search" in tool_names

        registry.unexpose_tool("web_search")

    def test_web_search_not_exposed_after_unexpose(self):
        """Test that web_search is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")
        registry.unexpose_tool("web_search")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "web_search" not in tool_names


class TestRegistryWebSearchExecutors:
    """Tests for web_search executors exposure via registry."""

    def test_executor_not_available_before_expose(self):
        """Test that executor is not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "web_search" not in executors

    def test_executor_available_after_expose(self):
        """Test that executor is available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")

        executors = registry.get_exposed_executors()

        assert "web_search" in executors
        assert callable(executors["web_search"])

        registry.unexpose_tool("web_search")

    def test_executor_not_available_after_unexpose(self):
        """Test that executor is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")
        registry.unexpose_tool("web_search")

        executors = registry.get_exposed_executors()

        assert "web_search" not in executors


class TestRegistryWebSearchAutoApproval:
    """Tests for web_search auto-approved tools via registry."""

    def test_web_search_is_auto_approved(self):
        """Test that web_search IS auto-approved (read-only, safe operation)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")

        auto_approved = registry.get_auto_approved_tools()

        # web_search is read-only and should be auto-approved
        assert "web_search" in auto_approved

        registry.unexpose_tool("web_search")


class TestRegistryWebSearchSystemInstructions:
    """Tests for web_search system instructions via registry."""

    def test_system_instructions_included(self):
        """Test that web_search system instructions are included."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")

        instructions = registry.get_system_instructions()

        assert instructions is not None
        assert "web_search" in instructions

        registry.unexpose_tool("web_search")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with web_search plugin."""

    def test_get_plugin_for_web_search_tool(self):
        """Test that get_plugin_for_tool returns web_search plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")

        plugin = registry.get_plugin_for_tool("web_search")
        assert plugin is not None
        assert plugin.name == "web_search"

        registry.unexpose_tool("web_search")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("web_search")
        assert plugin is None


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("web_search")
        plugin = registry.get_plugin("web_search")

        assert plugin._initialized is True

        registry.unexpose_tool("web_search")

        assert plugin._initialized is False
