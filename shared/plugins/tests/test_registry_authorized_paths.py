"""Tests for the authorized external paths mechanism in PluginRegistry."""

import os
import pytest
from pathlib import Path

from ..registry import PluginRegistry


class TestAuthorizedExternalPaths:
    """Tests for external path authorization in PluginRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance."""
        return PluginRegistry()

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Create temporary paths for testing."""
        # Create test files and directories
        file1 = tmp_path / "file1.txt"
        file1.write_text("content1")

        file2 = tmp_path / "subdir" / "file2.txt"
        file2.parent.mkdir()
        file2.write_text("content2")

        file3 = tmp_path / "subdir" / "nested" / "file3.txt"
        file3.parent.mkdir()
        file3.write_text("content3")

        return {
            "file1": str(file1),
            "file2": str(file2),
            "file3": str(file3),
            "subdir": str(tmp_path / "subdir"),
            "root": str(tmp_path),
        }

    def test_authorize_single_path(self, registry, temp_paths):
        """Test authorizing a single file path."""
        registry.authorize_external_path(temp_paths["file1"], "test_plugin")

        assert registry.is_path_authorized(temp_paths["file1"]) is True
        assert registry.is_path_authorized(temp_paths["file2"]) is False

    def test_authorize_multiple_paths(self, registry, temp_paths):
        """Test authorizing multiple paths at once."""
        paths = [temp_paths["file1"], temp_paths["file2"]]
        registry.authorize_external_paths(paths, "test_plugin")

        assert registry.is_path_authorized(temp_paths["file1"]) is True
        assert registry.is_path_authorized(temp_paths["file2"]) is True
        assert registry.is_path_authorized(temp_paths["file3"]) is False

    def test_authorize_directory_includes_children(self, registry, temp_paths):
        """Test that authorizing a directory authorizes all files within it."""
        registry.authorize_external_path(temp_paths["subdir"], "test_plugin")

        # Files within the directory should be authorized
        assert registry.is_path_authorized(temp_paths["file2"]) is True
        assert registry.is_path_authorized(temp_paths["file3"]) is True

        # File outside the directory should not be authorized
        assert registry.is_path_authorized(temp_paths["file1"]) is False

    def test_get_authorization_source(self, registry, temp_paths):
        """Test getting the source plugin that authorized a path."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file2"], "plugin_b")

        assert registry.get_path_authorization_source(temp_paths["file1"]) == "plugin_a"
        assert registry.get_path_authorization_source(temp_paths["file2"]) == "plugin_b"
        assert registry.get_path_authorization_source(temp_paths["file3"]) is None

    def test_get_authorization_source_for_directory_child(self, registry, temp_paths):
        """Test getting source for a file within an authorized directory."""
        registry.authorize_external_path(temp_paths["subdir"], "dir_plugin")

        # File within directory should return the directory's source
        assert registry.get_path_authorization_source(temp_paths["file2"]) == "dir_plugin"
        assert registry.get_path_authorization_source(temp_paths["file3"]) == "dir_plugin"

    def test_clear_all_authorized_paths(self, registry, temp_paths):
        """Test clearing all authorized paths."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file2"], "plugin_b")

        count = registry.clear_authorized_paths()

        assert count == 2
        assert registry.is_path_authorized(temp_paths["file1"]) is False
        assert registry.is_path_authorized(temp_paths["file2"]) is False

    def test_clear_paths_from_specific_plugin(self, registry, temp_paths):
        """Test clearing paths from a specific plugin only."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file2"], "plugin_b")
        registry.authorize_external_path(temp_paths["file3"], "plugin_a")

        count = registry.clear_authorized_paths("plugin_a")

        assert count == 2
        # plugin_a paths should be cleared
        assert registry.is_path_authorized(temp_paths["file1"]) is False
        assert registry.is_path_authorized(temp_paths["file3"]) is False
        # plugin_b path should remain
        assert registry.is_path_authorized(temp_paths["file2"]) is True

    def test_list_authorized_paths(self, registry, temp_paths):
        """Test listing all authorized paths."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file2"], "plugin_b")

        paths = registry.list_authorized_paths()

        # Paths should be normalized
        normalized_file1 = os.path.realpath(temp_paths["file1"])
        normalized_file2 = os.path.realpath(temp_paths["file2"])

        assert normalized_file1 in paths
        assert normalized_file2 in paths
        assert paths[normalized_file1] == "plugin_a"
        assert paths[normalized_file2] == "plugin_b"

    def test_path_normalization(self, registry, tmp_path):
        """Test that paths are normalized when authorizing and checking."""
        # Create a file
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("content")

        # Authorize using a path with ..
        rel_path = str(tmp_path / "subdir" / ".." / "test_file.txt")
        registry.authorize_external_path(rel_path, "test_plugin")

        # Check using the canonical path
        canonical = str(file_path.resolve())
        assert registry.is_path_authorized(canonical) is True

    def test_empty_registry_checks(self, registry, temp_paths):
        """Test that checks work correctly on empty registry."""
        assert registry.is_path_authorized(temp_paths["file1"]) is False
        assert registry.get_path_authorization_source(temp_paths["file1"]) is None
        assert registry.list_authorized_paths() == {}
        assert registry.clear_authorized_paths() == 0

    def test_reauthorize_same_path_different_plugin(self, registry, temp_paths):
        """Test that reauthorizing a path updates the source plugin."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file1"], "plugin_b")

        # Should reflect the latest authorization
        assert registry.get_path_authorization_source(temp_paths["file1"]) == "plugin_b"

    def test_relative_path_authorization(self, registry, tmp_path, monkeypatch):
        """Test authorizing with a relative path."""
        # Create a file
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        # Change to tmp_path and authorize with relative path
        monkeypatch.chdir(tmp_path)
        registry.authorize_external_path("test.txt", "test_plugin")

        # Check with absolute path
        assert registry.is_path_authorized(str(file_path)) is True
