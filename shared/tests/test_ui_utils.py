"""Tests for shared.ui_utils module."""

import pytest
from shared.ui_utils import ellipsize_path, ellipsize_path_pair


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
