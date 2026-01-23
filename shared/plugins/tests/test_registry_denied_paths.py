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
