"""Tests for references plugin integration with the plugin registry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import ReferencesPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the references plugin via the registry."""

    def test_references_plugin_discovered(self):
        """Test that references plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "references" in discovered

    def test_references_plugin_available(self):
        """Test that references plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "references" in registry.list_available()

    def test_get_references_plugin(self):
        """Test retrieving the references plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("references")
        assert plugin is not None
        assert plugin.name == "references"


class TestRegistryExposeReferencesPlugin:
    """Tests for exposing the references plugin via the registry."""

    def test_expose_references_plugin(self):
        """Test exposing the references plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        assert registry.is_exposed("references")
        assert "references" in registry.list_exposed()

        registry.unexpose_tool("references")

    def test_unexpose_references_plugin(self):
        """Test unexposing the references plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        assert registry.is_exposed("references")

        registry.unexpose_tool("references")
        assert not registry.is_exposed("references")

    def test_expose_all_includes_references(self):
        """Test that expose_all includes the references plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("references")

        registry.unexpose_all()


class TestRegistryReferencesToolDeclarations:
    """Tests for references tool declarations exposure via registry."""

    def test_references_tools_not_exposed_before_expose(self):
        """Test that references tools are not in declarations before expose."""
        registry = PluginRegistry()
        registry.discover()

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "selectReferences" not in tool_names
        assert "listReferences" not in tool_names

    def test_references_tools_exposed_after_expose(self):
        """Test that references tools are in declarations after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "selectReferences" in tool_names
        assert "listReferences" in tool_names

        registry.unexpose_tool("references")

    def test_references_tools_not_exposed_after_unexpose(self):
        """Test that references tools are not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        registry.unexpose_tool("references")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "selectReferences" not in tool_names
        assert "listReferences" not in tool_names


class TestRegistryReferencesExecutors:
    """Tests for references executors exposure via registry."""

    def test_executor_not_available_before_expose(self):
        """Test that executor is not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "selectReferences" not in executors
        assert "listReferences" not in executors

    def test_executor_available_after_expose(self):
        """Test that executor is available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        executors = registry.get_exposed_executors()

        assert "selectReferences" in executors
        assert "listReferences" in executors
        assert callable(executors["selectReferences"])
        assert callable(executors["listReferences"])

        registry.unexpose_tool("references")

    def test_executor_not_available_after_unexpose(self):
        """Test that executor is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        registry.unexpose_tool("references")

        executors = registry.get_exposed_executors()

        assert "selectReferences" not in executors
        assert "listReferences" not in executors


class TestRegistryReferencesAutoApproval:
    """Tests for references auto-approved tools via registry."""

    def test_references_tools_are_auto_approved(self):
        """Test that references tools ARE auto-approved (user-triggered)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        auto_approved = registry.get_auto_approved_tools()

        # Both tools are auto-approved (user-triggered selection flow)
        assert "selectReferences" in auto_approved
        assert "listReferences" in auto_approved

        registry.unexpose_tool("references")


class TestRegistryReferencesSystemInstructions:
    """Tests for references system instructions via registry."""

    def test_system_instructions_with_no_sources(self):
        """Test system instructions when no sources are configured."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        # Without references.json or sources, may return None
        plugin = registry.get_plugin("references")
        instructions = plugin.get_system_instructions()

        # With no sources, instructions may be None
        # This is expected behavior

        registry.unexpose_tool("references")


class TestRegistryReferencesUserCommands:
    """Tests for references user commands via registry."""

    def test_user_commands_included(self):
        """Test that references user commands are included after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        user_commands = registry.get_exposed_user_commands()
        command_names = [cmd.name for cmd in user_commands]

        assert "listReferences" in command_names
        assert "selectReferences" in command_names

        registry.unexpose_tool("references")

    def test_user_commands_not_included_before_expose(self):
        """Test that user commands are not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        user_commands = registry.get_exposed_user_commands()
        command_names = [cmd.name for cmd in user_commands]

        assert "listReferences" not in command_names
        assert "selectReferences" not in command_names

    def test_user_commands_shared_with_model(self):
        """Test that references user commands have share_with_model=True."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        user_commands = registry.get_exposed_user_commands()
        list_cmd = next((cmd for cmd in user_commands if cmd.name == "listReferences"), None)
        select_cmd = next((cmd for cmd in user_commands if cmd.name == "selectReferences"), None)

        assert list_cmd is not None
        assert list_cmd.share_with_model is True

        assert select_cmd is not None
        assert select_cmd.share_with_model is True

        registry.unexpose_tool("references")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with references plugin."""

    def test_get_plugin_for_select_references(self):
        """Test that get_plugin_for_tool returns references plugin for selectReferences."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        plugin = registry.get_plugin_for_tool("selectReferences")
        assert plugin is not None
        assert plugin.name == "references"

        registry.unexpose_tool("references")

    def test_get_plugin_for_list_references(self):
        """Test that get_plugin_for_tool returns references plugin for listReferences."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        plugin = registry.get_plugin_for_tool("listReferences")
        assert plugin is not None
        assert plugin.name == "references"

        registry.unexpose_tool("references")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("selectReferences")
        assert plugin is None


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        plugin = registry.get_plugin("references")

        assert plugin._initialized is True

        registry.unexpose_tool("references")

        assert plugin._initialized is False

    def test_sources_cleared_on_shutdown(self):
        """Test that sources are cleared on shutdown."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        plugin = registry.get_plugin("references")

        registry.unexpose_tool("references")

        # After shutdown, sources should be cleared
        assert plugin._sources == []

    def test_selections_cleared_on_shutdown(self):
        """Test that selections are cleared on shutdown."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        plugin = registry.get_plugin("references")

        # Simulate a selection
        plugin._selected_source_ids = ["test-source-1"]

        registry.unexpose_tool("references")

        # After shutdown, selections should be cleared
        assert plugin._selected_source_ids == []
