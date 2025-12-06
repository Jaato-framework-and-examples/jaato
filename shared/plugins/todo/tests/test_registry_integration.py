"""Tests for TODO plugin integration with the plugin registry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import TodoPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the TODO plugin via the registry."""

    def test_todo_plugin_discovered(self):
        """Test that TODO plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "todo" in discovered

    def test_todo_plugin_available(self):
        """Test that TODO plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "todo" in registry.list_available()

    def test_get_todo_plugin(self):
        """Test retrieving the TODO plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("todo")
        assert plugin is not None
        assert plugin.name == "todo"


class TestRegistryExposeTodoPlugin:
    """Tests for exposing the TODO plugin via the registry."""

    def test_expose_todo_plugin(self):
        """Test exposing the TODO plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        assert registry.is_exposed("todo")
        assert "todo" in registry.list_exposed()

        registry.unexpose_tool("todo")

    def test_unexpose_todo_plugin(self):
        """Test unexposing the TODO plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")
        assert registry.is_exposed("todo")

        registry.unexpose_tool("todo")
        assert not registry.is_exposed("todo")

    def test_expose_all_includes_todo(self):
        """Test that expose_all includes the TODO plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("todo")

        registry.unexpose_all()


class TestRegistryTodoToolDeclarations:
    """Tests for TODO tool declarations exposure via registry."""

    def test_todo_tools_not_exposed_before_expose(self):
        """Test that TODO tools are not in declarations before expose."""
        registry = PluginRegistry()
        registry.discover()

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "createPlan" not in tool_names
        assert "startPlan" not in tool_names
        assert "updateStep" not in tool_names
        assert "getPlanStatus" not in tool_names
        assert "completePlan" not in tool_names
        assert "addStep" not in tool_names

    def test_todo_tools_exposed_after_expose(self):
        """Test that TODO tools are in declarations after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "createPlan" in tool_names
        assert "startPlan" in tool_names
        assert "updateStep" in tool_names
        assert "getPlanStatus" in tool_names
        assert "completePlan" in tool_names
        assert "addStep" in tool_names

        registry.unexpose_tool("todo")

    def test_todo_tools_not_exposed_after_unexpose(self):
        """Test that TODO tools are not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")
        registry.unexpose_tool("todo")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "createPlan" not in tool_names
        assert "startPlan" not in tool_names


class TestRegistryTodoExecutors:
    """Tests for TODO executors exposure via registry."""

    def test_executor_not_available_before_expose(self):
        """Test that executor is not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "createPlan" not in executors
        assert "startPlan" not in executors
        assert "updateStep" not in executors

    def test_executor_available_after_expose(self):
        """Test that executor is available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        executors = registry.get_exposed_executors()

        assert "createPlan" in executors
        assert "startPlan" in executors
        assert "updateStep" in executors
        assert "getPlanStatus" in executors
        assert "completePlan" in executors
        assert "addStep" in executors
        assert callable(executors["createPlan"])
        assert callable(executors["updateStep"])

        registry.unexpose_tool("todo")

    def test_executor_not_available_after_unexpose(self):
        """Test that executor is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")
        registry.unexpose_tool("todo")

        executors = registry.get_exposed_executors()

        assert "createPlan" not in executors
        assert "updateStep" not in executors


class TestRegistryTodoAutoApproval:
    """Tests for TODO auto-approved tools via registry."""

    def test_most_todo_tools_are_auto_approved(self):
        """Test that most TODO tools ARE auto-approved."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        auto_approved = registry.get_auto_approved_tools()

        # These are auto-approved (no security implications)
        assert "createPlan" in auto_approved
        assert "updateStep" in auto_approved
        assert "getPlanStatus" in auto_approved
        assert "completePlan" in auto_approved
        assert "addStep" in auto_approved
        assert "plan" in auto_approved

        registry.unexpose_tool("todo")

    def test_start_plan_not_auto_approved(self):
        """Test that startPlan is NOT auto-approved (requires user confirmation)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        auto_approved = registry.get_auto_approved_tools()

        # startPlan requires user confirmation to proceed
        assert "startPlan" not in auto_approved

        registry.unexpose_tool("todo")


class TestRegistryTodoSystemInstructions:
    """Tests for TODO system instructions via registry."""

    def test_system_instructions_included(self):
        """Test that TODO system instructions are included."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        instructions = registry.get_system_instructions()

        assert instructions is not None
        assert "createPlan" in instructions

        registry.unexpose_tool("todo")


class TestRegistryTodoUserCommands:
    """Tests for TODO user commands via registry."""

    def test_user_commands_included(self):
        """Test that TODO user commands are included after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        user_commands = registry.get_exposed_user_commands()
        command_names = [cmd.name for cmd in user_commands]

        assert "plan" in command_names

        registry.unexpose_tool("todo")

    def test_user_commands_not_included_before_expose(self):
        """Test that user commands are not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        user_commands = registry.get_exposed_user_commands()
        command_names = [cmd.name for cmd in user_commands]

        assert "plan" not in command_names

    def test_plan_command_not_shared_with_model(self):
        """Test that plan user command has share_with_model=False."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        user_commands = registry.get_exposed_user_commands()
        plan_cmd = next((cmd for cmd in user_commands if cmd.name == "plan"), None)

        assert plan_cmd is not None
        assert plan_cmd.share_with_model is False

        registry.unexpose_tool("todo")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with TODO plugin."""

    def test_get_plugin_for_create_plan(self):
        """Test that get_plugin_for_tool returns TODO plugin for createPlan."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        plugin = registry.get_plugin_for_tool("createPlan")
        assert plugin is not None
        assert plugin.name == "todo"

        registry.unexpose_tool("todo")

    def test_get_plugin_for_update_step(self):
        """Test that get_plugin_for_tool returns TODO plugin for updateStep."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")

        plugin = registry.get_plugin_for_tool("updateStep")
        assert plugin is not None
        assert plugin.name == "todo"

        registry.unexpose_tool("todo")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("createPlan")
        assert plugin is None


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")
        plugin = registry.get_plugin("todo")

        assert plugin._initialized is True

        registry.unexpose_tool("todo")

        assert plugin._initialized is False

    def test_current_plan_cleared_on_shutdown(self):
        """Test that current plan is cleared on shutdown."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("todo")
        plugin = registry.get_plugin("todo")

        # Create a plan
        plugin._execute_create_plan({
            "title": "Test Plan",
            "steps": ["Step 1", "Step 2"]
        })
        assert plugin._current_plan_id is not None

        registry.unexpose_tool("todo")

        # After shutdown, current plan should be cleared
        assert plugin._current_plan_id is None
