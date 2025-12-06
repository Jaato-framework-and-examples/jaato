"""Tests for subagent plugin integration with the plugin registry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import SubagentPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the subagent plugin via the registry."""

    def test_subagent_plugin_discovered(self):
        """Test that subagent plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "subagent" in discovered

    def test_subagent_plugin_available(self):
        """Test that subagent plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "subagent" in registry.list_available()

    def test_get_subagent_plugin(self):
        """Test retrieving the subagent plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("subagent")
        assert plugin is not None
        assert plugin.name == "subagent"


class TestRegistryExposeSubagentPlugin:
    """Tests for exposing the subagent plugin via the registry."""

    def test_expose_subagent_plugin(self):
        """Test exposing the subagent plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        assert registry.is_exposed("subagent")
        assert "subagent" in registry.list_exposed()

        registry.unexpose_tool("subagent")

    def test_expose_subagent_plugin_with_config(self):
        """Test exposing the subagent plugin with configuration."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent", config={
            "project": "test-project",
            "location": "us-central1",
            "default_model": "gemini-2.5-flash"
        })

        assert registry.is_exposed("subagent")
        plugin = registry.get_plugin("subagent")
        assert plugin._config.project == "test-project"
        assert plugin._config.location == "us-central1"

        registry.unexpose_tool("subagent")

    def test_unexpose_subagent_plugin(self):
        """Test unexposing the subagent plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")
        assert registry.is_exposed("subagent")

        registry.unexpose_tool("subagent")
        assert not registry.is_exposed("subagent")

    def test_expose_all_includes_subagent(self):
        """Test that expose_all includes the subagent plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("subagent")

        registry.unexpose_all()


class TestRegistrySubagentToolDeclarations:
    """Tests for subagent tool declarations exposure via registry."""

    def test_subagent_tools_not_exposed_before_expose(self):
        """Test that subagent tools are not in declarations before expose."""
        registry = PluginRegistry()
        registry.discover()

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "spawn_subagent" not in tool_names
        assert "list_subagent_profiles" not in tool_names

    def test_subagent_tools_exposed_after_expose(self):
        """Test that subagent tools are in declarations after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "spawn_subagent" in tool_names
        assert "list_subagent_profiles" in tool_names

        registry.unexpose_tool("subagent")

    def test_subagent_tools_not_exposed_after_unexpose(self):
        """Test that subagent tools are not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")
        registry.unexpose_tool("subagent")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "spawn_subagent" not in tool_names
        assert "list_subagent_profiles" not in tool_names


class TestRegistrySubagentExecutors:
    """Tests for subagent executors exposure via registry."""

    def test_executor_not_available_before_expose(self):
        """Test that executor is not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "spawn_subagent" not in executors
        assert "list_subagent_profiles" not in executors

    def test_executor_available_after_expose(self):
        """Test that executor is available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        executors = registry.get_exposed_executors()

        assert "spawn_subagent" in executors
        assert "list_subagent_profiles" in executors
        assert callable(executors["spawn_subagent"])
        assert callable(executors["list_subagent_profiles"])

        registry.unexpose_tool("subagent")

    def test_executor_not_available_after_unexpose(self):
        """Test that executor is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")
        registry.unexpose_tool("subagent")

        executors = registry.get_exposed_executors()

        assert "spawn_subagent" not in executors
        assert "list_subagent_profiles" not in executors


class TestRegistrySubagentAutoApproval:
    """Tests for subagent auto-approved tools via registry."""

    def test_list_subagent_profiles_is_auto_approved(self):
        """Test that list_subagent_profiles IS auto-approved (read-only)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        auto_approved = registry.get_auto_approved_tools()

        # list_subagent_profiles is read-only and should be auto-approved
        assert "list_subagent_profiles" in auto_approved

        registry.unexpose_tool("subagent")

    def test_spawn_subagent_not_auto_approved(self):
        """Test that spawn_subagent is NOT auto-approved (requires permission)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        auto_approved = registry.get_auto_approved_tools()

        # spawn_subagent creates subagents - should NOT be auto-approved
        assert "spawn_subagent" not in auto_approved

        registry.unexpose_tool("subagent")


class TestRegistrySubagentSystemInstructions:
    """Tests for subagent system instructions via registry."""

    def test_system_instructions_included(self):
        """Test that subagent system instructions are included."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        instructions = registry.get_system_instructions()

        assert instructions is not None
        assert "subagent" in instructions

        registry.unexpose_tool("subagent")


class TestRegistrySubagentUserCommands:
    """Tests for subagent user commands via registry."""

    def test_user_commands_included(self):
        """Test that subagent user commands are included after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        user_commands = registry.get_exposed_user_commands()
        command_names = [cmd.name for cmd in user_commands]

        assert "profiles" in command_names

        registry.unexpose_tool("subagent")

    def test_user_commands_not_included_before_expose(self):
        """Test that user commands are not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        user_commands = registry.get_exposed_user_commands()
        command_names = [cmd.name for cmd in user_commands]

        assert "profiles" not in command_names


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with subagent plugin."""

    def test_get_plugin_for_spawn_subagent(self):
        """Test that get_plugin_for_tool returns subagent plugin for spawn_subagent."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        plugin = registry.get_plugin_for_tool("spawn_subagent")
        assert plugin is not None
        assert plugin.name == "subagent"

        registry.unexpose_tool("subagent")

    def test_get_plugin_for_list_subagent_profiles(self):
        """Test that get_plugin_for_tool returns subagent plugin for list_subagent_profiles."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")

        plugin = registry.get_plugin_for_tool("list_subagent_profiles")
        assert plugin is not None
        assert plugin.name == "subagent"

        registry.unexpose_tool("subagent")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("spawn_subagent")
        assert plugin is None


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("subagent")
        plugin = registry.get_plugin("subagent")

        assert plugin._initialized is True

        registry.unexpose_tool("subagent")

        assert plugin._initialized is False
