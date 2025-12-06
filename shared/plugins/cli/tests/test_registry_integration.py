"""Tests for CLI plugin integration with the plugin registry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import CLIToolPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the CLI plugin via the registry."""

    def test_cli_plugin_discovered(self):
        """Test that CLI plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "cli" in discovered

    def test_cli_plugin_available(self):
        """Test that CLI plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "cli" in registry.list_available()

    def test_get_cli_plugin(self):
        """Test retrieving the CLI plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("cli")
        assert plugin is not None
        assert plugin.name == "cli"


class TestRegistryExposeCLIPlugin:
    """Tests for exposing the CLI plugin via the registry."""

    def test_expose_cli_plugin(self):
        """Test exposing the CLI plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")

        assert registry.is_exposed("cli")
        assert "cli" in registry.list_exposed()

        registry.unexpose_tool("cli")

    def test_expose_cli_plugin_with_config(self):
        """Test exposing the CLI plugin with configuration."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli", config={"extra_paths": ["/usr/local/bin"]})

        assert registry.is_exposed("cli")
        plugin = registry.get_plugin("cli")
        assert "/usr/local/bin" in plugin._extra_paths

        registry.unexpose_tool("cli")

    def test_unexpose_cli_plugin(self):
        """Test unexposing the CLI plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")
        assert registry.is_exposed("cli")

        registry.unexpose_tool("cli")
        assert not registry.is_exposed("cli")

    def test_expose_all_includes_cli(self):
        """Test that expose_all includes the CLI plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("cli")

        registry.unexpose_all()


class TestRegistryCLIToolDeclarations:
    """Tests for CLI tool declarations exposure via registry."""

    def test_cli_based_tool_not_exposed_before_expose(self):
        """Test that cli_based_tool is not in declarations before expose."""
        registry = PluginRegistry()
        registry.discover()

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "cli_based_tool" not in tool_names

    def test_cli_based_tool_exposed_after_expose(self):
        """Test that cli_based_tool is in declarations after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "cli_based_tool" in tool_names

        registry.unexpose_tool("cli")

    def test_cli_based_tool_not_exposed_after_unexpose(self):
        """Test that cli_based_tool is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")
        registry.unexpose_tool("cli")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "cli_based_tool" not in tool_names


class TestRegistryCLIExecutors:
    """Tests for CLI executors exposure via registry."""

    def test_executor_not_available_before_expose(self):
        """Test that executor is not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "cli_based_tool" not in executors

    def test_executor_available_after_expose(self):
        """Test that executor is available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")

        executors = registry.get_exposed_executors()

        assert "cli_based_tool" in executors
        assert callable(executors["cli_based_tool"])

        registry.unexpose_tool("cli")

    def test_executor_not_available_after_unexpose(self):
        """Test that executor is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")
        registry.unexpose_tool("cli")

        executors = registry.get_exposed_executors()

        assert "cli_based_tool" not in executors


class TestRegistryCLIExecution:
    """Tests for executing CLI tool via registry executors."""

    def test_execute_simple_command(self):
        """Test executing a simple command via registry."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")

        executors = registry.get_exposed_executors()
        result = executors["cli_based_tool"]({"command": "echo hello"})

        assert "error" not in result
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

        registry.unexpose_tool("cli")

    def test_execute_command_not_found(self):
        """Test executing a non-existent command via registry."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")

        executors = registry.get_exposed_executors()
        result = executors["cli_based_tool"]({"command": "nonexistent_command_xyz"})

        assert "error" in result

        registry.unexpose_tool("cli")


class TestRegistryCLIAutoApproval:
    """Tests for CLI auto-approved tools via registry."""

    def test_cli_not_auto_approved(self):
        """Test that cli_based_tool is NOT auto-approved (requires permission)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")

        auto_approved = registry.get_auto_approved_tools()

        # CLI tools require permission - should NOT be auto-approved
        assert "cli_based_tool" not in auto_approved

        registry.unexpose_tool("cli")


class TestRegistryCLISystemInstructions:
    """Tests for CLI system instructions via registry."""

    def test_system_instructions_included(self):
        """Test that CLI system instructions are included."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")

        instructions = registry.get_system_instructions()

        assert instructions is not None
        assert "cli_based_tool" in instructions

        registry.unexpose_tool("cli")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with CLI plugin."""

    def test_get_plugin_for_cli_tool(self):
        """Test that get_plugin_for_tool returns CLI plugin for cli_based_tool."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")

        plugin = registry.get_plugin_for_tool("cli_based_tool")
        assert plugin is not None
        assert plugin.name == "cli"

        registry.unexpose_tool("cli")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("cli_based_tool")
        assert plugin is None


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("cli")
        plugin = registry.get_plugin("cli")

        assert plugin._initialized is True

        registry.unexpose_tool("cli")

        assert plugin._initialized is False
