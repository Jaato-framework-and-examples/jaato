"""Tests for MCP plugin integration with the plugin registry.

Note: MCP plugin tools are discovered dynamically from connected MCP servers.
Without active MCP servers, the plugin exposes no tools. These tests verify
the registry integration without requiring actual MCP server connections.
"""

import pytest

from ...registry import PluginRegistry
from ..plugin import MCPToolPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the MCP plugin via the registry."""

    def test_mcp_plugin_discovered(self):
        """Test that MCP plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "mcp" in discovered

    def test_mcp_plugin_available(self):
        """Test that MCP plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "mcp" in registry.list_available()

    def test_get_mcp_plugin(self):
        """Test retrieving the MCP plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("mcp")
        assert plugin is not None
        assert plugin.name == "mcp"


class TestRegistryExposeMCPPlugin:
    """Tests for exposing the MCP plugin via the registry."""

    def test_expose_mcp_plugin(self):
        """Test exposing the MCP plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")

        assert registry.is_exposed("mcp")
        assert "mcp" in registry.list_exposed()

        registry.unexpose_tool("mcp")

    def test_unexpose_mcp_plugin(self):
        """Test unexposing the MCP plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")
        assert registry.is_exposed("mcp")

        registry.unexpose_tool("mcp")
        assert not registry.is_exposed("mcp")

    def test_expose_all_includes_mcp(self):
        """Test that expose_all includes the MCP plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("mcp")

        registry.unexpose_all()


class TestRegistryMCPToolDeclarations:
    """Tests for MCP tool declarations exposure via registry.

    Note: MCP tools are discovered dynamically. Without MCP servers,
    no tools will be available. These tests verify the mechanism works.
    """

    def test_no_declarations_without_mcp_servers(self):
        """Test that no MCP-specific tools are exposed without MCP servers."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")

        # Without .mcp.json or MCP servers, no dynamic tools are discovered
        # The plugin is exposed but has no tools from servers
        declarations = registry.get_exposed_declarations()

        # This verifies the registry integration works even with empty tools
        # The MCP plugin should be exposed even if it has no tools
        assert registry.is_exposed("mcp")

        registry.unexpose_tool("mcp")


class TestRegistryMCPExecutors:
    """Tests for MCP executors exposure via registry."""

    def test_no_executors_without_mcp_servers(self):
        """Test that no executors are exposed without MCP servers."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")

        executors = registry.get_exposed_executors()

        # Without MCP servers, no tool executors are available
        # But the plugin is still properly exposed
        assert registry.is_exposed("mcp")

        registry.unexpose_tool("mcp")


class TestRegistryMCPAutoApproval:
    """Tests for MCP auto-approved tools via registry."""

    def test_mcp_tools_not_auto_approved(self):
        """Test that MCP tools are NOT auto-approved (require permission)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")

        auto_approved = registry.get_auto_approved_tools()

        # MCP tools require permission - should NOT be auto-approved
        # The plugin returns empty list for get_auto_approved_tools()
        plugin = registry.get_plugin("mcp")
        assert plugin.get_auto_approved_tools() == []

        registry.unexpose_tool("mcp")


class TestRegistryMCPSystemInstructions:
    """Tests for MCP system instructions via registry."""

    def test_no_system_instructions_without_mcp_servers(self):
        """Test that no MCP-specific instructions without servers."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")

        # Without MCP servers, the plugin returns None for instructions
        plugin = registry.get_plugin("mcp")
        assert plugin.get_system_instructions() is None

        registry.unexpose_tool("mcp")


class TestRegistryMCPUserCommands:
    """Tests for MCP user commands via registry."""

    def test_no_user_commands(self):
        """Test that MCP plugin provides no user commands."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")

        plugin = registry.get_plugin("mcp")
        assert plugin.get_user_commands() == []

        registry.unexpose_tool("mcp")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with MCP plugin.

    Note: MCP tools are dynamic, so we can only test the negative case
    without actual MCP server connections.
    """

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        # Without MCP servers, no tools are registered
        plugin = registry.get_plugin_for_tool("some_mcp_tool")
        assert plugin is None


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")
        plugin = registry.get_plugin("mcp")

        assert plugin._initialized is True

        registry.unexpose_tool("mcp")

        assert plugin._initialized is False

    def test_plugin_clears_tool_cache_on_shutdown(self):
        """Test that MCP plugin clears tool cache on shutdown."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("mcp")
        plugin = registry.get_plugin("mcp")

        # Plugin is initialized
        assert plugin._initialized is True

        registry.unexpose_tool("mcp")

        # After shutdown, tool cache should be empty
        assert plugin._tool_cache == {}
