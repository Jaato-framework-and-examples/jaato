"""Tests for preselected reference read detection.

Tests the _detect_preselected_read method which identifies when the model
reads a file belonging to a preselected reference, including directory
references where the model reads individual files inside the directory.
"""

import os
import tempfile

import pytest

from ..plugin import create_plugin
from ..models import ReferenceSource, SourceType, InjectionMode
from shared.path_utils import normalize_for_comparison


class TestDetectPreselectedRead:
    """Tests for _detect_preselected_read method."""

    def test_exact_file_match(self):
        """Test detection when readFile path matches a preselected file exactly."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/home/user/project/docs/spec.md"):
                ("spec-ref", "Spec Reference")
        }

        result = plugin._detect_preselected_read(
            {"path": "/home/user/project/docs/spec.md"}
        )

        assert result is not None
        assert result == ("spec-ref", "Spec Reference")

    def test_no_match(self):
        """Test no detection when path doesn't match any preselected reference."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/home/user/project/docs/spec.md"):
                ("spec-ref", "Spec Reference")
        }

        result = plugin._detect_preselected_read(
            {"path": "/home/user/project/other/file.md"}
        )

        assert result is None

    def test_directory_containment_match(self):
        """Test detection when readFile path is inside a preselected directory."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/home/user/project/refs/architecture"):
                ("arch-ref", "Architecture Reference")
        }

        result = plugin._detect_preselected_read(
            {"path": "/home/user/project/refs/architecture/module-structure.md"}
        )

        assert result is not None
        assert result == ("arch-ref", "Architecture Reference")

    def test_directory_containment_nested(self):
        """Test detection for deeply nested files inside a preselected directory."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/refs"):
                ("my-ref", "My Reference")
        }

        result = plugin._detect_preselected_read(
            {"path": "/refs/subdir/deep/file.md"}
        )

        assert result is not None
        assert result == ("my-ref", "My Reference")

    def test_directory_containment_no_partial_name_match(self):
        """Test that /refs does not match /refs-old/file (partial dir name)."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/home/user/refs"):
                ("refs", "References")
        }

        result = plugin._detect_preselected_read(
            {"path": "/home/user/refs-old/file.md"}
        )

        assert result is None

    def test_cli_command_substring_match(self):
        """Test detection when the preselected path appears inside a CLI command."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/home/user/docs/spec.md"):
                ("spec-ref", "Spec Reference")
        }

        result = plugin._detect_preselected_read(
            {"command": "cat /home/user/docs/spec.md"}
        )

        assert result is not None
        assert result == ("spec-ref", "Spec Reference")

    def test_cli_command_dir_containment(self):
        """Test detection when a CLI command reads a file inside a preselected dir."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/home/user/refs"):
                ("refs", "References")
        }

        result = plugin._detect_preselected_read(
            {"command": "cat /home/user/refs/design.md"}
        )

        assert result is not None
        assert result == ("refs", "References")

    def test_cli_command_no_partial_dir_match(self):
        """Test that CLI substring check respects path boundaries."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/home/user/refs"):
                ("refs", "References")
        }

        result = plugin._detect_preselected_read(
            {"command": "cat /home/user/refs-old/file.md"}
        )

        assert result is None

    def test_non_string_values_skipped(self):
        """Test that non-string tool arg values are skipped."""
        plugin = create_plugin()
        plugin._preselected_paths = {
            normalize_for_comparison("/some/path"):
                ("ref", "Ref")
        }

        result = plugin._detect_preselected_read(
            {"limit": 100, "offset": 1}
        )

        assert result is None

    def test_empty_preselected_paths(self):
        """Test that empty preselected_paths returns None."""
        plugin = create_plugin()
        plugin._preselected_paths = {}

        result = plugin._detect_preselected_read(
            {"path": "/any/file.md"}
        )

        assert result is None
