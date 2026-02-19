"""Tests for PluginRegistry external path denial functionality.

These tests verify the deny_external_path, is_path_denied, and related
methods added to support the sandbox_manager plugin.
"""

import os
import tempfile
import pytest
from pathlib import Path

from shared.plugins.registry import PluginRegistry


class TestDenyExternalPath:
    """Tests for deny_external_path method."""

    def test_deny_single_path(self, tmp_path):
        """Test denying a single path."""
        registry = PluginRegistry()
        test_path = str(tmp_path / "blocked")

        registry.deny_external_path(test_path, "test_plugin")

        denied = registry.list_denied_paths()
        # Path is normalized, so check the real path
        assert len(denied) == 1
        assert "test_plugin" in denied.values()

    def test_deny_multiple_paths(self, tmp_path):
        """Test denying multiple paths."""
        registry = PluginRegistry()
        paths = [str(tmp_path / f"blocked{i}") for i in range(3)]

        registry.deny_external_paths(paths, "test_plugin")

        denied = registry.list_denied_paths()
        assert len(denied) == 3

    def test_deny_normalizes_path(self, tmp_path):
        """Test that paths are normalized (symlinks resolved)."""
        registry = PluginRegistry()

        # Create a symlink
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir)

        # Deny the symlink path
        registry.deny_external_path(str(link), "test_plugin")

        # The real path should be denied
        denied = registry.list_denied_paths()
        assert str(real_dir.resolve()) in denied


class TestIsPathDenied:
    """Tests for is_path_denied method."""

    def test_exact_path_denied(self, tmp_path):
        """Test exact path match is denied."""
        registry = PluginRegistry()
        blocked = tmp_path / "blocked"
        blocked.mkdir()

        registry.deny_external_path(str(blocked), "test_plugin")

        assert registry.is_path_denied(str(blocked)) is True

    def test_child_path_denied(self, tmp_path):
        """Test that child paths under denied directory are denied."""
        registry = PluginRegistry()
        blocked = tmp_path / "blocked"
        blocked.mkdir()

        registry.deny_external_path(str(blocked), "test_plugin")

        # Child path should also be denied
        child = blocked / "subdir" / "file.txt"
        assert registry.is_path_denied(str(child)) is True

    def test_unrelated_path_not_denied(self, tmp_path):
        """Test that unrelated paths are not denied."""
        registry = PluginRegistry()
        blocked = tmp_path / "blocked"
        allowed = tmp_path / "allowed"

        registry.deny_external_path(str(blocked), "test_plugin")

        assert registry.is_path_denied(str(allowed)) is False

    def test_similar_prefix_not_denied(self, tmp_path):
        """Test that similar prefixes don't cause false positives."""
        registry = PluginRegistry()
        blocked = tmp_path / "blocked"
        similar = tmp_path / "blocked_but_different"

        registry.deny_external_path(str(blocked), "test_plugin")

        # Similar prefix should NOT be denied
        assert registry.is_path_denied(str(similar)) is False

    def test_empty_deny_list_allows_all(self, tmp_path):
        """Test that empty deny list allows all paths."""
        registry = PluginRegistry()

        assert registry.is_path_denied(str(tmp_path / "anything")) is False


class TestGetPathDenialSource:
    """Tests for get_path_denial_source method."""

    def test_returns_source_for_denied_path(self, tmp_path):
        """Test that source plugin is returned for denied path."""
        registry = PluginRegistry()
        blocked = tmp_path / "blocked"
        blocked.mkdir()

        registry.deny_external_path(str(blocked), "sandbox_manager")

        source = registry.get_path_denial_source(str(blocked))
        assert source == "sandbox_manager"

    def test_returns_source_for_child_path(self, tmp_path):
        """Test that source is returned for child of denied directory."""
        registry = PluginRegistry()
        blocked = tmp_path / "blocked"
        blocked.mkdir()

        registry.deny_external_path(str(blocked), "sandbox_manager")

        child = blocked / "subdir" / "file.txt"
        source = registry.get_path_denial_source(str(child))
        assert source == "sandbox_manager"

    def test_returns_none_for_allowed_path(self, tmp_path):
        """Test that None is returned for allowed paths."""
        registry = PluginRegistry()

        source = registry.get_path_denial_source(str(tmp_path / "allowed"))
        assert source is None


class TestClearDeniedPaths:
    """Tests for clear_denied_paths method."""

    def test_clear_all_denied_paths(self, tmp_path):
        """Test clearing all denied paths."""
        registry = PluginRegistry()
        registry.deny_external_path(str(tmp_path / "a"), "plugin_a")
        registry.deny_external_path(str(tmp_path / "b"), "plugin_b")

        count = registry.clear_denied_paths()

        assert count == 2
        assert len(registry.list_denied_paths()) == 0

    def test_clear_paths_from_specific_plugin(self, tmp_path):
        """Test clearing denied paths from specific plugin only."""
        registry = PluginRegistry()
        registry.deny_external_path(str(tmp_path / "a"), "plugin_a")
        registry.deny_external_path(str(tmp_path / "b"), "plugin_b")
        registry.deny_external_path(str(tmp_path / "c"), "plugin_a")

        count = registry.clear_denied_paths("plugin_a")

        assert count == 2
        denied = registry.list_denied_paths()
        assert len(denied) == 1
        assert list(denied.values())[0] == "plugin_b"


class TestDenyTakesPrecedence:
    """Tests verifying denial takes precedence over authorization."""

    def test_path_can_be_both_authorized_and_denied(self, tmp_path):
        """Test that a path can be in both authorized and denied lists."""
        registry = PluginRegistry()
        test_path = tmp_path / "contested"
        test_path.mkdir()

        # Both authorize and deny the same path
        registry.authorize_external_path(str(test_path), "plugin_a")
        registry.deny_external_path(str(test_path), "plugin_b")

        # Path should be both authorized and denied
        assert registry.is_path_authorized(str(test_path)) is True
        assert registry.is_path_denied(str(test_path)) is True

        # The sandbox_utils.check_path_with_jaato_containment checks
        # denial FIRST, so denial effectively takes precedence


class TestListDeniedPaths:
    """Tests for list_denied_paths method."""

    def test_returns_empty_dict_initially(self):
        """Test that empty registry returns empty dict."""
        registry = PluginRegistry()
        assert registry.list_denied_paths() == {}

    def test_returns_copy_not_reference(self, tmp_path):
        """Test that returned dict is a copy."""
        registry = PluginRegistry()
        registry.deny_external_path(str(tmp_path / "test"), "plugin")

        denied1 = registry.list_denied_paths()
        denied2 = registry.list_denied_paths()

        assert denied1 is not denied2
        assert denied1 == denied2


class TestAccessMode:
    """Tests for access mode (readonly/readwrite) support in PluginRegistry."""

    def test_default_access_is_readwrite(self, tmp_path):
        """Test that authorize_external_path defaults to readwrite."""
        registry = PluginRegistry()
        test_path = tmp_path / "data"
        test_path.mkdir()

        registry.authorize_external_path(str(test_path), "test_plugin")

        # Should allow both read and write
        assert registry.is_path_authorized(str(test_path), mode="read") is True
        assert registry.is_path_authorized(str(test_path), mode="write") is True

    def test_readonly_allows_read(self, tmp_path):
        """Test that readonly paths allow read access."""
        registry = PluginRegistry()
        test_path = tmp_path / "docs"
        test_path.mkdir()

        registry.authorize_external_path(str(test_path), "test_plugin", access="readonly")

        assert registry.is_path_authorized(str(test_path), mode="read") is True

    def test_readonly_blocks_write(self, tmp_path):
        """Test that readonly paths block write access."""
        registry = PluginRegistry()
        test_path = tmp_path / "docs"
        test_path.mkdir()

        registry.authorize_external_path(str(test_path), "test_plugin", access="readonly")

        assert registry.is_path_authorized(str(test_path), mode="write") is False

    def test_readwrite_allows_both(self, tmp_path):
        """Test that readwrite paths allow both read and write access."""
        registry = PluginRegistry()
        test_path = tmp_path / "projects"
        test_path.mkdir()

        registry.authorize_external_path(str(test_path), "test_plugin", access="readwrite")

        assert registry.is_path_authorized(str(test_path), mode="read") is True
        assert registry.is_path_authorized(str(test_path), mode="write") is True

    def test_readonly_child_path_blocks_write(self, tmp_path):
        """Test that child paths under readonly directory block write."""
        registry = PluginRegistry()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        registry.authorize_external_path(str(docs_dir), "test_plugin", access="readonly")

        child_file = docs_dir / "guide.md"
        assert registry.is_path_authorized(str(child_file), mode="read") is True
        assert registry.is_path_authorized(str(child_file), mode="write") is False

    def test_readwrite_child_path_allows_write(self, tmp_path):
        """Test that child paths under readwrite directory allow write."""
        registry = PluginRegistry()
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        registry.authorize_external_path(str(projects_dir), "test_plugin", access="readwrite")

        child_file = projects_dir / "main.py"
        assert registry.is_path_authorized(str(child_file), mode="read") is True
        assert registry.is_path_authorized(str(child_file), mode="write") is True

    def test_invalid_access_mode_raises(self, tmp_path):
        """Test that invalid access mode raises ValueError."""
        registry = PluginRegistry()

        with pytest.raises(ValueError, match="Invalid access mode"):
            registry.authorize_external_path(str(tmp_path), "test_plugin", access="invalid")

    def test_get_path_access_mode(self, tmp_path):
        """Test get_path_access_mode returns correct mode."""
        registry = PluginRegistry()
        readonly_path = tmp_path / "readonly"
        readonly_path.mkdir()
        readwrite_path = tmp_path / "readwrite"
        readwrite_path.mkdir()

        registry.authorize_external_path(str(readonly_path), "plugin", access="readonly")
        registry.authorize_external_path(str(readwrite_path), "plugin", access="readwrite")

        assert registry.get_path_access_mode(str(readonly_path)) == "readonly"
        assert registry.get_path_access_mode(str(readwrite_path)) == "readwrite"
        assert registry.get_path_access_mode(str(tmp_path / "unknown")) is None

    def test_get_path_access_mode_child_path(self, tmp_path):
        """Test get_path_access_mode for child paths."""
        registry = PluginRegistry()
        docs = tmp_path / "docs"
        docs.mkdir()

        registry.authorize_external_path(str(docs), "plugin", access="readonly")

        child = docs / "subdir" / "file.txt"
        assert registry.get_path_access_mode(str(child)) == "readonly"

    def test_list_authorized_paths_backward_compatible(self, tmp_path):
        """Test that list_authorized_paths returns source names (backward compatible)."""
        registry = PluginRegistry()
        test_path = tmp_path / "data"
        test_path.mkdir()

        registry.authorize_external_path(str(test_path), "my_plugin", access="readonly")

        paths = registry.list_authorized_paths()
        assert "my_plugin" in paths.values()

    def test_list_authorized_paths_detailed(self, tmp_path):
        """Test that list_authorized_paths_detailed returns access info."""
        registry = PluginRegistry()
        readonly_path = tmp_path / "readonly"
        readonly_path.mkdir()
        readwrite_path = tmp_path / "readwrite"
        readwrite_path.mkdir()

        registry.authorize_external_path(str(readonly_path), "plugin_a", access="readonly")
        registry.authorize_external_path(str(readwrite_path), "plugin_b", access="readwrite")

        detailed = registry.list_authorized_paths_detailed()
        assert len(detailed) == 2

        for path_info in detailed.values():
            assert "source" in path_info
            assert "access" in path_info

        # Check specific entries
        ro_key = str(readonly_path.resolve())
        rw_key = str(readwrite_path.resolve())
        assert detailed[ro_key]["access"] == "readonly"
        assert detailed[rw_key]["access"] == "readwrite"

    def test_authorize_external_paths_with_access(self, tmp_path):
        """Test authorize_external_paths (plural) with access mode."""
        registry = PluginRegistry()
        paths = [str(tmp_path / f"dir{i}") for i in range(3)]
        for p in paths:
            os.makedirs(p, exist_ok=True)

        registry.authorize_external_paths(paths, "plugin", access="readonly")

        for p in paths:
            assert registry.is_path_authorized(p, mode="read") is True
            assert registry.is_path_authorized(p, mode="write") is False

    def test_clear_authorized_paths_by_source(self, tmp_path):
        """Test that clearing paths by source works with access mode tuples."""
        registry = PluginRegistry()
        path_a = tmp_path / "a"
        path_a.mkdir()
        path_b = tmp_path / "b"
        path_b.mkdir()

        registry.authorize_external_path(str(path_a), "plugin_a", access="readonly")
        registry.authorize_external_path(str(path_b), "plugin_b", access="readwrite")

        count = registry.clear_authorized_paths("plugin_a")
        assert count == 1

        # plugin_a's path should be gone
        assert registry.is_path_authorized(str(path_a), mode="read") is False
        # plugin_b's path should still be there
        assert registry.is_path_authorized(str(path_b), mode="read") is True
