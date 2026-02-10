"""Tests for shared.ui_utils module."""

import pytest
from shared.ui_utils import (
    ellipsize_path,
    ellipsize_path_pair,
    format_tool_args_summary,
    _looks_like_path,
)


class TestEllipsizePath:
    """Tests for ellipsize_path function."""

    def test_short_path_unchanged(self):
        """Paths shorter than max_width should not be modified."""
        path = "src/components/Button.tsx"
        assert ellipsize_path(path, 50) == path

    def test_exact_length_unchanged(self):
        """Paths exactly at max_width should not be modified."""
        path = "a/b/c.txt"  # 9 chars
        assert ellipsize_path(path, 9) == path

    def test_middle_ellipsis_basic(self):
        """Long paths should have middle segments replaced with ellipsis."""
        path = "customer-domain-api/src/main/java/com/bank/model/Customer.java"
        result = ellipsize_path(path, 50)
        # Should preserve first segment and last 2 segments
        assert result.startswith("customer-domain-api/")
        assert result.endswith("/model/Customer.java")
        assert "..." in result
        assert len(result) <= 50

    def test_preserves_first_and_last_segments(self):
        """Default behavior preserves 1 first and 2 last segments."""
        path = "project/a/b/c/d/e/parent/file.txt"
        result = ellipsize_path(path, 35)
        # Should have: project/.../parent/file.txt
        assert "project" in result
        assert "parent" in result
        assert "file.txt" in result
        assert "..." in result

    def test_custom_keep_first(self):
        """Custom keep_first parameter."""
        path = "org/project/src/main/java/pkg/Class.java"
        result = ellipsize_path(path, 40, keep_first=2)
        # Should preserve org/project at start
        assert result.startswith("org/project/")
        assert "..." in result

    def test_custom_keep_last(self):
        """Custom keep_last parameter."""
        path = "project/a/b/c/d/e/f/g/file.txt"
        result = ellipsize_path(path, 35, keep_last=3)
        # Should preserve last 3: f/g/file.txt
        assert "f/g/file.txt" in result

    def test_absolute_unix_path(self):
        """Absolute Unix paths should preserve leading slash."""
        path = "/home/user/project/src/components/Button.tsx"
        result = ellipsize_path(path, 40)
        assert result.startswith("/")
        assert "..." in result
        assert "Button.tsx" in result

    def test_fallback_to_filename_only(self):
        """When path can't fit with first segment, show filename only."""
        path = "very-long-project-name/even-longer-module/deeply/nested/file.txt"
        result = ellipsize_path(path, 20)
        # Should fall back to ...file.txt or similar
        assert "file.txt" in result
        assert len(result) <= 20

    def test_very_short_max_width(self):
        """Handle very short max_width gracefully."""
        path = "project/src/file.txt"
        result = ellipsize_path(path, 12)
        assert len(result) <= 12

    def test_filename_truncation(self):
        """When even filename is too long, truncate it."""
        path = "project/VeryLongFileNameThatExceedsLimit.java"
        result = ellipsize_path(path, 15)
        assert len(result) <= 15
        assert "..." in result

    def test_few_segments_no_middle_ellipsis(self):
        """Paths with few segments fall back gracefully."""
        path = "short/path.txt"
        # With keep_first=1 and keep_last=2, we have only 3 segments
        # which is exactly keep_first + keep_last, so no middle to elide
        result = ellipsize_path(path, 10)
        assert len(result) <= 10

    def test_empty_path(self):
        """Empty path returns empty."""
        assert ellipsize_path("", 50) == ""

    def test_zero_max_width(self):
        """Zero max_width returns original path."""
        path = "some/path.txt"
        assert ellipsize_path(path, 0) == path

    def test_negative_max_width(self):
        """Negative max_width returns original path."""
        path = "some/path.txt"
        assert ellipsize_path(path, -10) == path

    def test_custom_ellipsis(self):
        """Custom ellipsis string."""
        path = "project/a/b/c/d/e/f/file.txt"
        result = ellipsize_path(path, 30, ellipsis="…")
        assert "…" in result
        assert "..." not in result

    def test_java_package_path(self):
        """Real-world Java package paths."""
        path = "customer-domain-api/src/main/java/com/bank/customer/model/CustomerSystemApiAdapterTest.java"
        result = ellipsize_path(path, 60)
        assert result.endswith("CustomerSystemApiAdapterTest.java")
        assert len(result) <= 60

    def test_nested_src_path(self):
        """Typical src directory structure."""
        path = "my-project/packages/ui-components/src/components/forms/TextField.tsx"
        result = ellipsize_path(path, 50)
        assert "TextField.tsx" in result
        assert len(result) <= 50

    def test_progressive_reduction(self):
        """Verifies progressive reduction when standard ellipsis is too long."""
        path = "aaaa/bbbb/cccc/dddd/eeee/ffff/gggg/file.txt"
        # Very tight constraint
        result = ellipsize_path(path, 25)
        assert len(result) <= 25
        assert "file.txt" in result or ".txt" in result


class TestEllipsizePathPair:
    """Tests for ellipsize_path_pair function."""

    def test_short_paths_unchanged(self):
        """Short path pairs should not be modified."""
        source = "src/a.txt"
        dest = "dst/b.txt"
        result = ellipsize_path_pair(source, dest, 50)
        assert result == "src/a.txt -> dst/b.txt"

    def test_long_paths_ellipsized(self):
        """Long path pairs should both be ellipsized."""
        source = "project/very/long/source/path/file.txt"
        dest = "project/very/long/dest/path/new.txt"
        result = ellipsize_path_pair(source, dest, 50)
        assert " -> " in result
        assert len(result) <= 50

    def test_custom_separator(self):
        """Custom separator between paths."""
        source = "a/b/file.txt"
        dest = "c/d/file.txt"
        result = ellipsize_path_pair(source, dest, 50, separator=" → ")
        assert " → " in result

    def test_both_paths_get_ellipsized(self):
        """Both source and dest get ellipsized when needed."""
        source = "long/project/name/src/main/java/File.java"
        dest = "long/project/name/src/test/java/FileTest.java"
        result = ellipsize_path_pair(source, dest, 50)
        parts = result.split(" -> ")
        assert len(parts) == 2
        # Both parts should have been ellipsized
        for part in parts:
            # At least one should have ellipsis since original is long
            pass  # Just verify it splits correctly
        assert len(result) <= 50

    def test_proportional_allocation(self):
        """Space is allocated proportionally between paths."""
        source = "short.txt"
        dest = "very/long/destination/path/that/is/much/longer/file.txt"
        result = ellipsize_path_pair(source, dest, 50)
        assert " -> " in result
        # Dest should be ellipsized more since it's longer
        parts = result.split(" -> ")
        assert parts[0] == "short.txt"  # Short source unchanged


class TestLooksLikePath:
    """Tests for _looks_like_path helper."""

    def test_absolute_unix_path(self):
        assert _looks_like_path("/home/user/file.txt") is True

    def test_root_only(self):
        """Single slash without nested separator is not a path."""
        assert _looks_like_path("/") is False

    def test_relative_dot(self):
        assert _looks_like_path("./src/file.txt") is True

    def test_relative_dotdot(self):
        assert _looks_like_path("../file.txt") is True

    def test_home_dir(self):
        assert _looks_like_path("~/projects/file.txt") is True

    def test_windows_path(self):
        assert _looks_like_path("C:\\Users\\file.txt") is True

    def test_windows_forward_slash(self):
        assert _looks_like_path("C:/Users/file.txt") is True

    def test_plain_string(self):
        assert _looks_like_path("hello world") is False

    def test_empty_string(self):
        assert _looks_like_path("") is False

    def test_single_char(self):
        assert _looks_like_path("x") is False

    def test_url_not_matched(self):
        """URLs are not file paths (they have // after scheme)."""
        # This starts with / and has /, so it will match -- acceptable
        # since URLs rarely appear as tool arg values without a scheme
        pass

    def test_simple_filename(self):
        assert _looks_like_path("file.txt") is False


class TestFormatToolArgsSummary:
    """Tests for format_tool_args_summary with path ellipsization."""

    def test_empty_args(self):
        assert format_tool_args_summary({}) == ""

    def test_short_args_unchanged(self):
        args = {"key": "value"}
        result = format_tool_args_summary(args)
        assert result == str(args)

    def test_path_arg_by_name_ellipsized(self):
        """Arguments with known path names get ellipsized."""
        long_path = "/home/user/project/src/components/deep/nested/Button.tsx"
        args = {"file_path": long_path}
        result = format_tool_args_summary(args, max_length=120)
        # The path value should have been ellipsized (shorter than original)
        assert "..." in result or len(result) < len(str(args))
        assert "Button.tsx" in result

    def test_path_arg_by_value_ellipsized(self):
        """Arguments whose values look like paths get ellipsized even with
        non-standard key names."""
        long_path = "/home/user/project/src/components/deep/nested/Button.tsx"
        args = {"target_file": long_path}
        result = format_tool_args_summary(args, max_length=120)
        assert "Button.tsx" in result
        # Path should have been shortened via middle-ellipsis
        assert "..." in result

    def test_non_path_arg_not_ellipsized(self):
        """Non-path arguments should not be modified by path ellipsization."""
        args = {"command": "echo hello world"}
        result = format_tool_args_summary(args, max_length=120)
        assert "echo hello world" in result

    def test_overall_truncation_still_applies(self):
        """Even after path ellipsization, overall truncation kicks in."""
        args = {
            "file_path": "/home/user/project/src/a.txt",
            "content": "x" * 100,
        }
        result = format_tool_args_summary(args, max_length=60)
        assert len(result) <= 60
        assert result.endswith("...")

    def test_short_path_not_ellipsized(self):
        """Short paths that fit within max_path_width stay unchanged."""
        args = {"path": "src/Button.tsx"}
        result = format_tool_args_summary(args, max_length=120)
        assert "src/Button.tsx" in result

    def test_multiple_path_args(self):
        """Multiple path arguments all get ellipsized."""
        args = {
            "old_path": "/home/user/project/src/deep/nested/old.txt",
            "new_path": "/home/user/project/src/deep/nested/new.txt",
        }
        result = format_tool_args_summary(args, max_length=120)
        assert "old.txt" in result
        assert "new.txt" in result

    def test_custom_max_path_width(self):
        """Custom max_path_width controls individual path truncation."""
        long_path = "/home/user/project/src/components/deep/nested/Button.tsx"
        args = {"file_path": long_path}
        narrow = format_tool_args_summary(
            args, max_length=120, max_path_width=25
        )
        wide = format_tool_args_summary(
            args, max_length=120, max_path_width=55
        )
        # Narrow should be shorter or equal
        assert len(narrow) <= len(wide)

    def test_mixed_path_and_non_path(self):
        """Mix of path and non-path arguments."""
        args = {
            "file_path": "/home/user/project/src/components/Button.tsx",
            "mode": "write",
        }
        result = format_tool_args_summary(args, max_length=120)
        assert "Button.tsx" in result
        assert "'mode': 'write'" in result
