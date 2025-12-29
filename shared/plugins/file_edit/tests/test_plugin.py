"""Tests for the file_edit plugin."""

import os
import tempfile
from pathlib import Path

import pytest

from ..plugin import FileEditPlugin, create_plugin


class TestFileEditPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, FileEditPlugin)

    def test_plugin_name(self):
        plugin = FileEditPlugin()
        assert plugin.name == "file_edit"

    def test_initialize_without_config(self):
        plugin = FileEditPlugin()
        plugin.initialize()
        assert plugin._initialized is True
        assert plugin._backup_manager is not None

    def test_initialize_with_custom_backup_dir(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "custom_backups"
        plugin.initialize({"backup_dir": str(backup_dir)})
        assert plugin._initialized is True
        assert plugin._backup_manager._base_dir == backup_dir

    def test_shutdown(self):
        plugin = FileEditPlugin()
        plugin.initialize()
        plugin.shutdown()
        assert plugin._initialized is False
        assert plugin._backup_manager is None


class TestFileEditPluginFunctionDeclarations:
    """Tests for function declarations."""

    def test_get_tool_schemas(self):
        plugin = FileEditPlugin()
        declarations = plugin.get_tool_schemas()

        assert len(declarations) == 7
        tool_names = [d.name for d in declarations]
        assert "readFile" in tool_names
        assert "updateFile" in tool_names
        assert "writeNewFile" in tool_names
        assert "removeFile" in tool_names
        assert "moveFile" in tool_names
        assert "renameFile" in tool_names
        assert "undoFileChange" in tool_names

    def test_read_file_schema(self):
        plugin = FileEditPlugin()
        schemas = plugin.get_tool_schemas()
        read_file = [s for s in schemas if s.name == "readFile"][0]
        schema = read_file.parameters

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "path" in schema["required"]

    def test_update_file_schema(self):
        plugin = FileEditPlugin()
        schemas = plugin.get_tool_schemas()
        update_file = [s for s in schemas if s.name == "updateFile"][0]
        schema = update_file.parameters

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "new_content" in schema["properties"]
        assert "path" in schema["required"]
        assert "new_content" in schema["required"]


class TestFileEditPluginExecutors:
    """Tests for executor mapping."""

    def test_get_executors(self):
        plugin = FileEditPlugin()
        executors = plugin.get_executors()

        assert "readFile" in executors
        assert "updateFile" in executors
        assert "writeNewFile" in executors
        assert "removeFile" in executors
        assert "moveFile" in executors
        assert "renameFile" in executors
        assert "undoFileChange" in executors
        assert all(callable(e) for e in executors.values())


class TestFileEditPluginAutoApproval:
    """Tests for auto-approved tools."""

    def test_get_auto_approved_tools(self):
        plugin = FileEditPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "readFile" in auto_approved
        assert "undoFileChange" in auto_approved
        assert "updateFile" not in auto_approved
        assert "writeNewFile" not in auto_approved
        assert "removeFile" not in auto_approved


class TestFileEditPluginSystemInstructions:
    """Tests for system instructions."""

    def test_get_system_instructions(self):
        plugin = FileEditPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "readFile" in instructions
        assert "updateFile" in instructions
        assert "moveFile" in instructions
        assert "renameFile" in instructions
        assert "backup" in instructions.lower()


class TestReadFileExecution:
    """Tests for readFile tool execution."""

    def test_read_existing_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = plugin._execute_read_file({"path": str(test_file)})

        assert "error" not in result
        assert result["content"] == "Hello, World!"
        assert result["path"] == str(test_file)
        assert result["size"] == 13
        assert result["lines"] == 1

    def test_read_nonexistent_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_read_file({"path": str(tmp_path / "nonexistent.txt")})

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_read_file_missing_path(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_read_file({})

        assert "error" in result
        assert "required" in result["error"].lower()


class TestUpdateFileExecution:
    """Tests for updateFile tool execution."""

    def test_update_existing_file(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content"
        })

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "Updated content"
        assert "backup" in result

    def test_update_creates_backup(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content"
        })

        # Check backup was created
        backups = list(backup_dir.glob("*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "Original content"

    def test_update_nonexistent_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_update_file({
            "path": str(tmp_path / "nonexistent.txt"),
            "new_content": "Content"
        })

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_update_file_accepts_content_parameter(self, tmp_path):
        """Test that updateFile accepts 'content' as alias for 'new_content'.

        This ensures consistency with writeNewFile which uses 'content'.
        Models often use 'content' for both tools, so we accept either.
        """
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # Use 'content' instead of 'new_content'
        result = plugin._execute_update_file({
            "path": str(test_file),
            "content": "Updated via content param"
        })

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "Updated via content param"
        assert result["size"] == len("Updated via content param")


class TestWriteNewFileExecution:
    """Tests for writeNewFile tool execution."""

    def test_write_new_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "new.txt"

        result = plugin._execute_write_new_file({
            "path": str(new_file),
            "content": "New file content"
        })

        assert "error" not in result
        assert result["success"] is True
        assert new_file.exists()
        assert new_file.read_text() == "New file content"

    def test_write_new_file_creates_directories(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "subdir" / "nested" / "new.txt"

        result = plugin._execute_write_new_file({
            "path": str(new_file),
            "content": "Content"
        })

        assert "error" not in result
        assert result["success"] is True
        assert new_file.exists()

    def test_write_new_file_fails_if_exists(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        existing_file = tmp_path / "existing.txt"
        existing_file.write_text("Existing content")

        result = plugin._execute_write_new_file({
            "path": str(existing_file),
            "content": "New content"
        })

        assert "error" in result
        assert "already exists" in result["error"].lower()


class TestRemoveFileExecution:
    """Tests for removeFile tool execution."""

    def test_remove_file(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to delete")

        result = plugin._execute_remove_file({"path": str(test_file)})

        assert "error" not in result
        assert result["success"] is True
        assert result["deleted"] is True
        assert not test_file.exists()

    def test_remove_file_creates_backup(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to backup")

        plugin._execute_remove_file({"path": str(test_file)})

        # Check backup was created
        backups = list(backup_dir.glob("*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "Content to backup"

    def test_remove_nonexistent_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_remove_file({"path": str(tmp_path / "nonexistent.txt")})

        assert "error" in result
        assert "not found" in result["error"].lower()


class TestUndoFileChangeExecution:
    """Tests for undoFileChange tool execution."""

    def test_undo_file_change(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # Update the file (creates backup)
        plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content"
        })

        assert test_file.read_text() == "Updated content"

        # Undo the change
        result = plugin._execute_undo_file_change({"path": str(test_file)})

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "Original content"

    def test_undo_restores_deleted_file(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # Delete the file (creates backup)
        plugin._execute_remove_file({"path": str(test_file)})

        assert not test_file.exists()

        # Undo the deletion
        result = plugin._execute_undo_file_change({"path": str(test_file)})

        assert "error" not in result
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == "Original content"

    def test_undo_no_backup_available(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        # Try to undo without any backup
        result = plugin._execute_undo_file_change({"path": str(test_file)})

        assert "error" in result
        assert "no backup" in result["error"].lower()


class TestFormatPermissionRequest:
    """Tests for permission display formatting."""

    def test_format_update_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\n")

        display_info = plugin.format_permission_request(
            "updateFile",
            {"path": str(test_file), "new_content": "Line 1\nLine 2\nLine 3\n"},
            "console"
        )

        assert display_info is not None
        assert "Update" in display_info.summary
        assert display_info.format_hint == "diff"
        assert "+Line 3" in display_info.details

    def test_format_update_file_accepts_content_parameter(self, tmp_path):
        """Test that format_permission_request accepts 'content' for updateFile."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\n")

        # Use 'content' instead of 'new_content'
        display_info = plugin.format_permission_request(
            "updateFile",
            {"path": str(test_file), "content": "Line 1\nLine 2\nLine 3\n"},
            "console"
        )

        assert display_info is not None
        assert "Update" in display_info.summary
        assert display_info.format_hint == "diff"
        assert "+Line 3" in display_info.details

    def test_format_write_new_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "new.txt"

        display_info = plugin.format_permission_request(
            "writeNewFile",
            {"path": str(new_file), "content": "New content"},
            "console"
        )

        assert display_info is not None
        assert "Create" in display_info.summary
        assert display_info.format_hint == "diff"
        assert "+New content" in display_info.details

    def test_format_remove_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to delete")

        display_info = plugin.format_permission_request(
            "removeFile",
            {"path": str(test_file)},
            "console"
        )

        assert display_info is not None
        assert "Delete" in display_info.summary
        assert "backup" in display_info.summary.lower()
        assert display_info.format_hint == "diff"
        assert "-Content to delete" in display_info.details

    def test_format_unknown_tool_returns_none(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        display_info = plugin.format_permission_request(
            "unknownTool",
            {},
            "console"
        )

        assert display_info is None

    def test_format_move_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_file = tmp_path / "source.txt"
        source_file.write_text("File content to move")
        dest_file = tmp_path / "subdir" / "dest.txt"

        display_info = plugin.format_permission_request(
            "moveFile",
            {
                "source_path": str(source_file),
                "destination_path": str(dest_file)
            },
            "console"
        )

        assert display_info is not None
        assert "Move file" in display_info.summary
        assert display_info.format_hint == "diff"
        assert "-File content to move" in display_info.details
        assert "+File content to move" in display_info.details

    def test_format_rename_file(self, tmp_path):
        """Test that renameFile uses the same formatting as moveFile."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_file = tmp_path / "old_name.txt"
        source_file.write_text("Content")
        dest_file = tmp_path / "new_name.txt"

        display_info = plugin.format_permission_request(
            "renameFile",
            {
                "source_path": str(source_file),
                "destination_path": str(dest_file)
            },
            "console"
        )

        assert display_info is not None
        assert "Move file" in display_info.summary


class TestMoveFileExecution:
    """Tests for moveFile/renameFile tool execution."""

    def test_move_file_basic(self, tmp_path):
        """Test basic file move operation."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Content to move")
        dest_file = tmp_path / "dest.txt"

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file)
        })

        assert "error" not in result
        assert result["success"] is True
        assert result["source"] == str(source_file)
        assert result["destination"] == str(dest_file)
        assert not source_file.exists()
        assert dest_file.exists()
        assert dest_file.read_text() == "Content to move"

    def test_move_file_creates_directories(self, tmp_path):
        """Test that move creates destination directories."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Content")
        dest_file = tmp_path / "subdir" / "nested" / "dest.txt"

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file)
        })

        assert "error" not in result
        assert result["success"] is True
        assert dest_file.exists()
        assert dest_file.read_text() == "Content"

    def test_move_file_creates_backup(self, tmp_path):
        """Test that move creates a backup of the source file."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Original content")
        dest_file = tmp_path / "dest.txt"

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file)
        })

        assert "source_backup" in result
        # Check backup was created
        backups = list(backup_dir.glob("*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "Original content"

    def test_move_file_source_not_found(self, tmp_path):
        """Test error when source file doesn't exist."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_move_file({
            "source_path": str(tmp_path / "nonexistent.txt"),
            "destination_path": str(tmp_path / "dest.txt")
        })

        assert "error" in result
        assert "does not exist" in result["error"]
        assert "source" in result

    def test_move_file_destination_exists_no_overwrite(self, tmp_path):
        """Test error when destination exists without overwrite."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Source content")
        dest_file = tmp_path / "dest.txt"
        dest_file.write_text("Existing content")

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file)
        })

        assert "error" in result
        assert "already exists" in result["error"]
        assert "overwrite=True" in result["error"]
        # Source should still exist
        assert source_file.exists()
        # Destination should still have original content
        assert dest_file.read_text() == "Existing content"

    def test_move_file_destination_exists_with_overwrite(self, tmp_path):
        """Test successful overwrite when destination exists."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        source_file = tmp_path / "source.txt"
        source_file.write_text("New content")
        dest_file = tmp_path / "dest.txt"
        dest_file.write_text("Old content")

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file),
            "overwrite": True
        })

        assert "error" not in result
        assert result["success"] is True
        assert not source_file.exists()
        assert dest_file.read_text() == "New content"
        # Should have backups for both source and destination
        assert "source_backup" in result
        assert "destination_backup" in result

    def test_move_file_missing_source_path(self, tmp_path):
        """Test error when source_path is missing."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_move_file({
            "destination_path": str(tmp_path / "dest.txt")
        })

        assert "error" in result
        assert "source_path is required" in result["error"]

    def test_move_file_missing_destination_path(self, tmp_path):
        """Test error when destination_path is missing."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Content")

        result = plugin._execute_move_file({
            "source_path": str(source_file)
        })

        assert "error" in result
        assert "destination_path is required" in result["error"]

    def test_rename_file_uses_same_executor(self, tmp_path):
        """Test that renameFile uses the same executor as moveFile."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        executors = plugin.get_executors()
        assert executors["moveFile"] == executors["renameFile"]

    def test_move_file_source_is_directory(self, tmp_path):
        """Test error when source is a directory."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_dir = tmp_path / "source_dir"
        source_dir.mkdir()

        result = plugin._execute_move_file({
            "source_path": str(source_dir),
            "destination_path": str(tmp_path / "dest")
        })

        assert "error" in result
        assert "not a file" in result["error"]


class TestMoveFileToolSchemas:
    """Tests for moveFile/renameFile tool schemas."""

    def test_move_file_schema(self):
        plugin = FileEditPlugin()
        schemas = plugin.get_tool_schemas()
        move_file = [s for s in schemas if s.name == "moveFile"][0]
        schema = move_file.parameters

        assert schema["type"] == "object"
        assert "source_path" in schema["properties"]
        assert "destination_path" in schema["properties"]
        assert "overwrite" in schema["properties"]
        assert "source_path" in schema["required"]
        assert "destination_path" in schema["required"]
        # overwrite is optional
        assert "overwrite" not in schema["required"]

    def test_rename_file_schema(self):
        plugin = FileEditPlugin()
        schemas = plugin.get_tool_schemas()
        rename_file = [s for s in schemas if s.name == "renameFile"][0]
        schema = rename_file.parameters

        assert schema["type"] == "object"
        assert "source_path" in schema["properties"]
        assert "destination_path" in schema["properties"]
        assert "overwrite" in schema["properties"]

    def test_tool_count_includes_move_and_rename(self):
        plugin = FileEditPlugin()
        declarations = plugin.get_tool_schemas()

        assert len(declarations) == 7
        tool_names = [d.name for d in declarations]
        assert "moveFile" in tool_names
        assert "renameFile" in tool_names
