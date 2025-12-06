"""Tests for file_edit plugin integration with the plugin registry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import FileEditPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the file_edit plugin via the registry."""

    def test_file_edit_plugin_discovered(self):
        """Test that file_edit plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "file_edit" in discovered

    def test_file_edit_plugin_available(self):
        """Test that file_edit plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "file_edit" in registry.list_available()

    def test_get_file_edit_plugin(self):
        """Test retrieving the file_edit plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("file_edit")
        assert plugin is not None
        assert plugin.name == "file_edit"


class TestRegistryExposeFileEditPlugin:
    """Tests for exposing the file_edit plugin via the registry."""

    def test_expose_file_edit_plugin(self, tmp_path):
        """Test exposing the file_edit plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})

        assert registry.is_exposed("file_edit")
        assert "file_edit" in registry.list_exposed()

    def test_unexpose_file_edit_plugin(self, tmp_path):
        """Test unexposing the file_edit plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})
        assert registry.is_exposed("file_edit")

        registry.unexpose_tool("file_edit")
        assert not registry.is_exposed("file_edit")

    def test_expose_all_includes_file_edit(self):
        """Test that expose_all includes the file_edit plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("file_edit")

        registry.unexpose_all()


class TestRegistryFileEditToolDeclarations:
    """Tests for file_edit tool declarations exposure via registry."""

    def test_tools_not_exposed_before_expose(self):
        """Test that file_edit tools are not in declarations before expose."""
        registry = PluginRegistry()
        registry.discover()

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "readFile" not in tool_names
        assert "updateFile" not in tool_names
        assert "writeNewFile" not in tool_names
        assert "removeFile" not in tool_names
        assert "undoFileChange" not in tool_names

    def test_all_tools_exposed_after_expose(self, tmp_path):
        """Test that all file_edit tools are in declarations after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "readFile" in tool_names
        assert "updateFile" in tool_names
        assert "writeNewFile" in tool_names
        assert "removeFile" in tool_names
        assert "undoFileChange" in tool_names

        registry.unexpose_tool("file_edit")

    def test_tools_not_exposed_after_unexpose(self, tmp_path):
        """Test that file_edit tools are not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})
        registry.unexpose_tool("file_edit")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        assert "readFile" not in tool_names
        assert "updateFile" not in tool_names


class TestRegistryFileEditExecutors:
    """Tests for file_edit executors exposure via registry."""

    def test_executors_not_available_before_expose(self):
        """Test that executors are not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "readFile" not in executors
        assert "updateFile" not in executors

    def test_executors_available_after_expose(self, tmp_path):
        """Test that executors are available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})

        executors = registry.get_exposed_executors()

        assert "readFile" in executors
        assert "updateFile" in executors
        assert "writeNewFile" in executors
        assert "removeFile" in executors
        assert "undoFileChange" in executors

        assert all(callable(executors[name]) for name in [
            "readFile", "updateFile", "writeNewFile", "removeFile", "undoFileChange"
        ])

        registry.unexpose_tool("file_edit")

    def test_executors_not_available_after_unexpose(self, tmp_path):
        """Test that executors are not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})
        registry.unexpose_tool("file_edit")

        executors = registry.get_exposed_executors()

        assert "readFile" not in executors
        assert "updateFile" not in executors


class TestRegistryFileEditExecution:
    """Tests for executing file_edit tools via registry executors."""

    def test_execute_read_file(self, tmp_path):
        """Test executing readFile via registry."""
        registry = PluginRegistry()
        registry.discover()

        backup_dir = tmp_path / "backups"
        registry.expose_tool("file_edit", config={"backup_dir": str(backup_dir)})

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        executors = registry.get_exposed_executors()
        result = executors["readFile"]({"path": str(test_file)})

        assert "error" not in result
        assert result["content"] == "Hello, World!"

        registry.unexpose_tool("file_edit")

    def test_execute_write_new_file(self, tmp_path):
        """Test executing writeNewFile via registry."""
        registry = PluginRegistry()
        registry.discover()

        backup_dir = tmp_path / "backups"
        registry.expose_tool("file_edit", config={"backup_dir": str(backup_dir)})

        new_file = tmp_path / "new.txt"

        executors = registry.get_exposed_executors()
        result = executors["writeNewFile"]({
            "path": str(new_file),
            "content": "New content"
        })

        assert "error" not in result
        assert result["success"] is True
        assert new_file.exists()
        assert new_file.read_text() == "New content"

        registry.unexpose_tool("file_edit")

    def test_execute_update_file(self, tmp_path):
        """Test executing updateFile via registry."""
        registry = PluginRegistry()
        registry.discover()

        backup_dir = tmp_path / "backups"
        registry.expose_tool("file_edit", config={"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        executors = registry.get_exposed_executors()
        result = executors["updateFile"]({
            "path": str(test_file),
            "new_content": "Updated content"
        })

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "Updated content"
        assert "backup" in result

        registry.unexpose_tool("file_edit")

    def test_execute_remove_file(self, tmp_path):
        """Test executing removeFile via registry."""
        registry = PluginRegistry()
        registry.discover()

        backup_dir = tmp_path / "backups"
        registry.expose_tool("file_edit", config={"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to delete")

        executors = registry.get_exposed_executors()
        result = executors["removeFile"]({"path": str(test_file)})

        assert "error" not in result
        assert result["success"] is True
        assert not test_file.exists()
        assert "backup" in result

        registry.unexpose_tool("file_edit")

    def test_execute_undo_file_change(self, tmp_path):
        """Test executing undoFileChange via registry."""
        registry = PluginRegistry()
        registry.discover()

        backup_dir = tmp_path / "backups"
        registry.expose_tool("file_edit", config={"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        executors = registry.get_exposed_executors()

        # First update the file (creates backup)
        executors["updateFile"]({
            "path": str(test_file),
            "new_content": "Modified content"
        })
        assert test_file.read_text() == "Modified content"

        # Then undo
        result = executors["undoFileChange"]({"path": str(test_file)})

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "Original content"

        registry.unexpose_tool("file_edit")


class TestRegistryFileEditAutoApproval:
    """Tests for auto-approved tools via registry."""

    def test_auto_approved_tools_returned(self, tmp_path):
        """Test that auto-approved tools are returned by registry."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})

        auto_approved = registry.get_auto_approved_tools()

        assert "readFile" in auto_approved
        assert "undoFileChange" in auto_approved
        # These should NOT be auto-approved
        assert "updateFile" not in auto_approved
        assert "writeNewFile" not in auto_approved
        assert "removeFile" not in auto_approved

        registry.unexpose_tool("file_edit")


class TestRegistryFileEditSystemInstructions:
    """Tests for system instructions via registry."""

    def test_system_instructions_included(self, tmp_path):
        """Test that file_edit system instructions are included."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})

        instructions = registry.get_system_instructions()

        assert instructions is not None
        assert "readFile" in instructions
        assert "updateFile" in instructions
        assert "backup" in instructions.lower()

        registry.unexpose_tool("file_edit")


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self, tmp_path):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})
        plugin = registry.get_plugin("file_edit")

        assert plugin._initialized is True

        registry.unexpose_tool("file_edit")

        assert plugin._initialized is False

    def test_unexpose_all_cleans_up(self, tmp_path):
        """Test that unexpose_all properly cleans up."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})
        assert registry.is_exposed("file_edit")

        registry.unexpose_all()

        assert not registry.is_exposed("file_edit")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with file_edit plugin."""

    def test_get_plugin_for_file_edit_tools(self, tmp_path):
        """Test that get_plugin_for_tool returns file_edit plugin for its tools."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("file_edit", config={"backup_dir": str(tmp_path / "backups")})

        for tool_name in ["readFile", "updateFile", "writeNewFile", "removeFile", "undoFileChange"]:
            plugin = registry.get_plugin_for_tool(tool_name)
            assert plugin is not None
            assert plugin.name == "file_edit"

        registry.unexpose_tool("file_edit")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("readFile")
        assert plugin is None
