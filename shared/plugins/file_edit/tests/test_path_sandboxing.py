"""Tests for path sandboxing in the file_edit plugin."""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock

from ..plugin import FileEditPlugin, _detect_workspace_root, create_plugin


class TestDetectWorkspaceRoot:
    """Tests for workspace root detection."""

    def test_detect_jaato_workspace_root(self, monkeypatch):
        """Test detection from JAATO_WORKSPACE_ROOT env var."""
        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", "/test/workspace")
        monkeypatch.delenv("workspaceRoot", raising=False)
        result = _detect_workspace_root()
        assert result is not None
        assert "workspace" in result.lower()

    def test_detect_workspace_root_env(self, monkeypatch):
        """Test detection from workspaceRoot env var."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.setenv("workspaceRoot", "/test/project")
        result = _detect_workspace_root()
        assert result is not None
        assert "project" in result.lower()

    def test_detect_priority_jaato_first(self, monkeypatch):
        """Test that JAATO_WORKSPACE_ROOT takes priority."""
        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", "/priority/workspace")
        monkeypatch.setenv("workspaceRoot", "/fallback/workspace")
        result = _detect_workspace_root()
        assert "priority" in result.lower()

    def test_detect_none_when_unset(self, monkeypatch):
        """Test that None is returned when no env vars are set."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)
        result = _detect_workspace_root()
        assert result is None


class TestFileEditPluginPathSandboxing:
    """Tests for path sandboxing in the file_edit plugin."""

    @pytest.fixture
    def plugin_with_workspace(self, monkeypatch, tmp_path):
        """Create a plugin with a workspace root configured."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        # Create test files
        test_file = Path(workspace) / "test.txt"
        test_file.write_text("test content")

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
        })
        return plugin, workspace

    @pytest.fixture
    def plugin_no_workspace(self, monkeypatch):
        """Create a plugin without workspace root."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = create_plugin()
        plugin.initialize({})
        return plugin

    def test_path_allowed_within_workspace(self, plugin_with_workspace):
        """Test that paths within workspace are allowed."""
        plugin, workspace = plugin_with_workspace

        file_path = os.path.join(workspace, "test.txt")
        assert plugin._is_path_allowed(file_path) is True

    def test_path_allowed_workspace_root(self, plugin_with_workspace):
        """Test that workspace root itself is allowed."""
        plugin, workspace = plugin_with_workspace
        assert plugin._is_path_allowed(workspace) is True

    def test_path_blocked_outside_workspace(self, plugin_with_workspace, tmp_path):
        """Test that paths outside workspace are blocked."""
        plugin, workspace = plugin_with_workspace

        outside_path = str(tmp_path / "other" / "file.txt")
        assert plugin._is_path_allowed(outside_path) is False

    def test_all_paths_allowed_without_workspace(self, plugin_no_workspace, tmp_path):
        """Test that all paths are allowed when workspace is not configured."""
        plugin = plugin_no_workspace

        assert plugin._is_path_allowed("/some/path") is True
        assert plugin._is_path_allowed(str(tmp_path / "any/file.txt")) is True

    def test_authorized_external_path_allowed(self, plugin_with_workspace, tmp_path):
        """Test that externally authorized paths are allowed."""
        plugin, workspace = plugin_with_workspace

        # Create mock registry
        mock_registry = Mock()
        mock_registry.is_path_authorized.return_value = True
        plugin.set_plugin_registry(mock_registry)

        outside_path = str(tmp_path / "external" / "doc.md")
        assert plugin._is_path_allowed(outside_path) is True

        # Verify registry was consulted
        mock_registry.is_path_authorized.assert_called()

    def test_unauthorized_external_path_blocked(self, plugin_with_workspace, tmp_path):
        """Test that unauthorized external paths are blocked."""
        plugin, workspace = plugin_with_workspace

        # Create mock registry that returns False
        mock_registry = Mock()
        mock_registry.is_path_authorized.return_value = False
        plugin.set_plugin_registry(mock_registry)

        outside_path = str(tmp_path / "external" / "secret.txt")
        assert plugin._is_path_allowed(outside_path) is False


class TestReadFileSandboxing:
    """Tests for readFile with path sandboxing."""

    @pytest.fixture
    def plugin_with_workspace(self, monkeypatch, tmp_path):
        """Create a plugin with a workspace root configured."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        # Create test file in workspace
        test_file = Path(workspace) / "test.txt"
        test_file.write_text("workspace file content")

        # Create test file outside workspace
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "external.txt"
        outside_file.write_text("external file content")

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
        })
        return plugin, workspace, str(outside_file)

    def test_read_file_within_workspace(self, plugin_with_workspace):
        """Test reading a file within the workspace succeeds."""
        plugin, workspace, _ = plugin_with_workspace

        result = plugin._execute_read_file({
            "path": os.path.join(workspace, "test.txt"),
        })

        assert "error" not in result
        assert result["content"] == "workspace file content"

    def test_read_file_outside_workspace_blocked(self, plugin_with_workspace):
        """Test reading a file outside workspace returns not found."""
        plugin, workspace, outside_file = plugin_with_workspace

        result = plugin._execute_read_file({
            "path": outside_file,
        })

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_read_file_outside_workspace_authorized(self, plugin_with_workspace):
        """Test reading an authorized external file succeeds."""
        plugin, workspace, outside_file = plugin_with_workspace

        # Create mock registry that authorizes the path
        mock_registry = Mock()
        mock_registry.is_path_authorized.return_value = True
        plugin.set_plugin_registry(mock_registry)

        result = plugin._execute_read_file({
            "path": outside_file,
        })

        assert "error" not in result
        assert result["content"] == "external file content"


class TestAutoDetectWorkspace:
    """Tests for auto-detection of workspace from env vars."""

    def test_auto_detect_from_env(self, monkeypatch, tmp_path):
        """Test that workspace is auto-detected from environment."""
        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", workspace)

        plugin = create_plugin()
        plugin.initialize({})

        assert plugin._workspace_root == os.path.realpath(workspace)

    def test_explicit_config_overrides_env(self, monkeypatch, tmp_path):
        """Test that explicit config overrides environment."""
        env_workspace = str(tmp_path / "env_workspace")
        config_workspace = str(tmp_path / "config_workspace")
        os.makedirs(env_workspace, exist_ok=True)
        os.makedirs(config_workspace, exist_ok=True)

        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", env_workspace)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": config_workspace,
        })

        assert plugin._workspace_root == os.path.realpath(config_workspace)


class TestPathResolution:
    """Tests for path resolution (relative paths resolved against workspace_root)."""

    @pytest.fixture
    def plugin_with_workspace(self, monkeypatch, tmp_path):
        """Create a plugin with a workspace root configured."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        # Create subdirectories and files
        subdir = Path(workspace) / "subdir"
        subdir.mkdir()
        (subdir / "test.txt").write_text("subdir file")
        (Path(workspace) / "root.txt").write_text("root file")

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
        })
        return plugin, workspace

    @pytest.fixture
    def plugin_no_workspace(self, monkeypatch):
        """Create a plugin without workspace root."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = create_plugin()
        plugin.initialize({})
        return plugin

    def test_resolve_absolute_path_unchanged(self, plugin_with_workspace):
        """Test that absolute paths are returned as-is."""
        plugin, workspace = plugin_with_workspace

        abs_path = "/some/absolute/path/file.txt"
        resolved = plugin._resolve_path(abs_path)

        assert str(resolved) == abs_path

    def test_resolve_relative_path_with_workspace(self, plugin_with_workspace):
        """Test that relative paths are resolved against workspace_root."""
        plugin, workspace = plugin_with_workspace

        relative_path = "subdir/test.txt"
        resolved = plugin._resolve_path(relative_path)

        expected = Path(workspace) / relative_path
        assert resolved == expected

    def test_resolve_relative_path_without_workspace(self, plugin_no_workspace):
        """Test that relative paths are returned as-is when no workspace configured."""
        plugin = plugin_no_workspace

        relative_path = "some/relative/path.txt"
        resolved = plugin._resolve_path(relative_path)

        # Without workspace, should return Path(relative_path) unchanged
        assert resolved == Path(relative_path)

    def test_resolve_nested_relative_path(self, plugin_with_workspace):
        """Test resolving deeply nested relative paths."""
        plugin, workspace = plugin_with_workspace

        nested_path = "a/b/c/d/file.txt"
        resolved = plugin._resolve_path(nested_path)

        expected = Path(workspace) / nested_path
        assert resolved == expected

    def test_write_new_file_uses_resolved_path(self, plugin_with_workspace):
        """Test that writeNewFile resolves relative paths correctly."""
        plugin, workspace = plugin_with_workspace

        # Write a file using relative path
        result = plugin._execute_write_new_file({
            "path": "new_file.txt",
            "content": "test content",
        })

        assert "error" not in result
        assert result["success"] is True

        # Verify file was created in workspace, not CWD
        expected_file = Path(workspace) / "new_file.txt"
        assert expected_file.exists()
        assert expected_file.read_text() == "test content"

        # Verify file was NOT created in CWD
        cwd_file = Path.cwd() / "new_file.txt"
        if cwd_file != expected_file:  # Only check if different paths
            assert not cwd_file.exists()

    def test_update_file_uses_resolved_path(self, plugin_with_workspace):
        """Test that updateFile resolves relative paths correctly."""
        plugin, workspace = plugin_with_workspace

        # Update existing file using relative path
        result = plugin._execute_update_file({
            "path": "root.txt",
            "content": "updated content",
        })

        assert "error" not in result
        assert result["success"] is True

        # Verify file was updated in workspace
        expected_file = Path(workspace) / "root.txt"
        assert expected_file.read_text() == "updated content"

    def test_read_file_uses_resolved_path(self, plugin_with_workspace):
        """Test that readFile resolves relative paths correctly."""
        plugin, workspace = plugin_with_workspace

        # Read file using relative path
        result = plugin._execute_read_file({
            "path": "subdir/test.txt",
        })

        assert "error" not in result
        assert result["content"] == "subdir file"

    def test_remove_file_uses_resolved_path(self, plugin_with_workspace):
        """Test that removeFile resolves relative paths correctly."""
        plugin, workspace = plugin_with_workspace

        # Create a file to remove
        file_to_remove = Path(workspace) / "to_remove.txt"
        file_to_remove.write_text("delete me")

        # Remove using relative path
        result = plugin._execute_remove_file({
            "path": "to_remove.txt",
        })

        assert "error" not in result
        assert result["success"] is True
        assert not file_to_remove.exists()

    def test_move_file_uses_resolved_path(self, plugin_with_workspace):
        """Test that moveFile resolves both source and destination paths."""
        plugin, workspace = plugin_with_workspace

        # Create source file
        source_file = Path(workspace) / "source.txt"
        source_file.write_text("move me")

        # Move using relative paths
        result = plugin._execute_move_file({
            "source_path": "source.txt",
            "destination_path": "subdir/moved.txt",
        })

        assert "error" not in result
        assert result["success"] is True

        # Verify file was moved within workspace
        assert not source_file.exists()
        dest_file = Path(workspace) / "subdir" / "moved.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "move me"
