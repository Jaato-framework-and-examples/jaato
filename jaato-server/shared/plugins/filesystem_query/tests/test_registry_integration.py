"""Tests for filesystem_query plugin integration with the plugin registry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import FilesystemQueryPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the filesystem_query plugin via the registry."""

    def test_filesystem_query_plugin_discovered(self):
        """Test that filesystem_query plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "filesystem_query" in discovered

    def test_filesystem_query_plugin_available(self):
        """Test that filesystem_query plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "filesystem_query" in registry.list_available()

    def test_get_filesystem_query_plugin(self):
        """Test retrieving the filesystem_query plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("filesystem_query")
        assert plugin is not None
        assert plugin.name == "filesystem_query"


class TestRegistryExposeFilesystemQueryPlugin:
    """Tests for exposing the filesystem_query plugin via the registry."""

    def test_expose_filesystem_query_plugin(self):
        """Test exposing the filesystem_query plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")

        assert registry.is_exposed("filesystem_query")
        assert "filesystem_query" in registry.list_exposed()

        registry.unexpose_tool("filesystem_query")

    def test_expose_filesystem_query_plugin_with_config(self):
        """Test exposing the filesystem_query plugin with configuration."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query", config={
            "max_results": 100,
            "timeout_seconds": 60,
        })

        assert registry.is_exposed("filesystem_query")
        plugin = registry.get_plugin("filesystem_query")
        assert plugin._config.max_results == 100
        assert plugin._config.timeout_seconds == 60

        registry.unexpose_tool("filesystem_query")

    def test_unexpose_filesystem_query_plugin(self):
        """Test unexposing the filesystem_query plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")
        assert registry.is_exposed("filesystem_query")

        registry.unexpose_tool("filesystem_query")
        assert not registry.is_exposed("filesystem_query")

    def test_expose_all_includes_filesystem_query(self):
        """Test that expose_all includes the filesystem_query plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("filesystem_query")

        registry.unexpose_all()


class TestRegistryFilesystemQueryToolSchemas:
    """Tests for filesystem_query tool schemas exposure via registry."""

    def test_tools_not_exposed_before_expose(self):
        """Test that tools are not in schemas before expose."""
        registry = PluginRegistry()
        registry.discover()

        schemas = registry.get_exposed_tool_schemas()
        tool_names = [s.name for s in schemas]

        assert "glob_files" not in tool_names
        assert "grep_content" not in tool_names

    def test_tools_exposed_after_expose(self):
        """Test that tools are in schemas after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")

        schemas = registry.get_exposed_tool_schemas()
        tool_names = [s.name for s in schemas]

        assert "glob_files" in tool_names
        assert "grep_content" in tool_names

        registry.unexpose_tool("filesystem_query")

    def test_tools_not_exposed_after_unexpose(self):
        """Test that tools are not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")
        registry.unexpose_tool("filesystem_query")

        schemas = registry.get_exposed_tool_schemas()
        tool_names = [s.name for s in schemas]

        assert "glob_files" not in tool_names
        assert "grep_content" not in tool_names


class TestRegistryFilesystemQueryExecutors:
    """Tests for filesystem_query executors exposure via registry."""

    def test_executors_not_available_before_expose(self):
        """Test that executors are not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "glob_files" not in executors
        assert "grep_content" not in executors

    def test_executors_available_after_expose(self):
        """Test that executors are available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")

        executors = registry.get_exposed_executors()

        assert "glob_files" in executors
        assert "grep_content" in executors
        assert callable(executors["glob_files"])
        assert callable(executors["grep_content"])

        registry.unexpose_tool("filesystem_query")

    def test_executors_not_available_after_unexpose(self):
        """Test that executors are not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")
        registry.unexpose_tool("filesystem_query")

        executors = registry.get_exposed_executors()

        assert "glob_files" not in executors
        assert "grep_content" not in executors


class TestRegistryFilesystemQueryAutoApproval:
    """Tests for filesystem_query auto-approved tools via registry."""

    def test_glob_files_is_auto_approved(self):
        """Test that glob_files IS auto-approved (read-only, safe operation)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")

        auto_approved = registry.get_auto_approved_tools()

        assert "glob_files" in auto_approved

        registry.unexpose_tool("filesystem_query")

    def test_grep_content_is_auto_approved(self):
        """Test that grep_content IS auto-approved (read-only, safe operation)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")

        auto_approved = registry.get_auto_approved_tools()

        assert "grep_content" in auto_approved

        registry.unexpose_tool("filesystem_query")


class TestRegistryFilesystemQuerySystemInstructions:
    """Tests for filesystem_query system instructions via registry."""

    def test_system_instructions_included(self):
        """Test that filesystem_query system instructions are included."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")

        instructions = registry.get_system_instructions()

        assert instructions is not None
        assert "glob_files" in instructions
        assert "grep_content" in instructions

        registry.unexpose_tool("filesystem_query")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with filesystem_query plugin."""

    def test_get_plugin_for_glob_files_tool(self):
        """Test that get_plugin_for_tool returns filesystem_query plugin for glob_files."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")

        plugin = registry.get_plugin_for_tool("glob_files")
        assert plugin is not None
        assert plugin.name == "filesystem_query"

        registry.unexpose_tool("filesystem_query")

    def test_get_plugin_for_grep_content_tool(self):
        """Test that get_plugin_for_tool returns filesystem_query plugin for grep_content."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")

        plugin = registry.get_plugin_for_tool("grep_content")
        assert plugin is not None
        assert plugin.name == "filesystem_query"

        registry.unexpose_tool("filesystem_query")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("glob_files")
        assert plugin is None


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("filesystem_query")
        plugin = registry.get_plugin("filesystem_query")

        assert plugin._initialized is True

        registry.unexpose_tool("filesystem_query")

        assert plugin._initialized is False
