"""Tests for environment plugin integration with the plugin registry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import EnvironmentPlugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the environment plugin via the registry."""

    def test_environment_plugin_discovered(self):
        """Test that environment plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "environment" in discovered

    def test_environment_plugin_available(self):
        """Test that environment plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "environment" in registry.list_available()

    def test_get_environment_plugin(self):
        """Test retrieving the environment plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("environment")
        assert plugin is not None
        assert plugin.name == "environment"


class TestRegistryExposeEnvironmentPlugin:
    """Tests for exposing the environment plugin via the registry."""

    def test_expose_environment_plugin(self):
        """Test exposing the environment plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")

        assert registry.is_exposed("environment")
        assert "environment" in registry.list_exposed()

        registry.unexpose_tool("environment")

    def test_unexpose_environment_plugin(self):
        """Test unexposing the environment plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")
        assert registry.is_exposed("environment")

        registry.unexpose_tool("environment")
        assert not registry.is_exposed("environment")

    def test_expose_all_includes_environment(self):
        """Test that expose_all includes the environment plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("environment")

        registry.unexpose_all()


class TestRegistryEnvironmentToolDeclarations:
    """Tests for environment tool declarations exposure via registry."""

    def test_get_environment_not_exposed_before_expose(self):
        """Test that get_environment is not in declarations before expose."""
        registry = PluginRegistry()
        registry.discover()

        declarations = registry.get_exposed_tool_schemas()
        tool_names = [d.name for d in declarations]

        assert "get_environment" not in tool_names

    def test_get_environment_exposed_after_expose(self):
        """Test that get_environment is in declarations after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")

        declarations = registry.get_exposed_tool_schemas()
        tool_names = [d.name for d in declarations]

        assert "get_environment" in tool_names

        registry.unexpose_tool("environment")

    def test_get_environment_not_exposed_after_unexpose(self):
        """Test that get_environment is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")
        registry.unexpose_tool("environment")

        declarations = registry.get_exposed_tool_schemas()
        tool_names = [d.name for d in declarations]

        assert "get_environment" not in tool_names


class TestRegistryEnvironmentExecutors:
    """Tests for environment executors exposure via registry."""

    def test_executor_not_available_before_expose(self):
        """Test that executor is not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "get_environment" not in executors

    def test_executor_available_after_expose(self):
        """Test that executor is available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")

        executors = registry.get_exposed_executors()

        assert "get_environment" in executors
        assert callable(executors["get_environment"])

        registry.unexpose_tool("environment")

    def test_executor_not_available_after_unexpose(self):
        """Test that executor is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")
        registry.unexpose_tool("environment")

        executors = registry.get_exposed_executors()

        assert "get_environment" not in executors


class TestRegistryEnvironmentExecution:
    """Tests for executing environment tool via registry executors."""

    def test_execute_get_environment(self):
        """Test executing get_environment via registry."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")

        executors = registry.get_exposed_executors()
        result = executors["get_environment"]({})

        # Result should be valid JSON
        import json
        parsed = json.loads(result)

        assert "os" in parsed
        assert "shell" in parsed
        assert "arch" in parsed
        assert "cwd" in parsed

        registry.unexpose_tool("environment")

    def test_execute_get_environment_with_aspect(self):
        """Test executing get_environment with aspect via registry."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")

        executors = registry.get_exposed_executors()
        result = executors["get_environment"]({"aspect": "os"})

        import json
        parsed = json.loads(result)

        assert "type" in parsed
        assert "name" in parsed

        registry.unexpose_tool("environment")


class TestRegistryEnvironmentAutoApproval:
    """Tests for environment auto-approved tools via registry."""

    def test_environment_is_auto_approved(self):
        """Test that get_environment IS auto-approved (read-only, safe)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")

        auto_approved = registry.get_auto_approved_tools()

        # Environment tools are safe and should be auto-approved
        assert "get_environment" in auto_approved

        registry.unexpose_tool("environment")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with environment plugin."""

    def test_get_plugin_for_environment_tool(self):
        """Test that get_plugin_for_tool returns environment plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("environment")

        plugin = registry.get_plugin_for_tool("get_environment")
        assert plugin is not None
        assert plugin.name == "environment"

        registry.unexpose_tool("environment")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("get_environment")
        assert plugin is None
