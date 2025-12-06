"""Tests for diff utilities."""

import pytest

from ..diff_utils import (
    generate_unified_diff,
    generate_new_file_diff,
    generate_delete_file_diff,
    get_diff_stats,
    summarize_diff,
    DEFAULT_MAX_LINES,
)


class TestGenerateUnifiedDiff:
    """Tests for unified diff generation."""

    def test_simple_diff(self):
        old = "line1\nline2\nline3\n"
        new = "line1\nmodified\nline3\n"

        diff, truncated, total = generate_unified_diff(old, new, "test.txt")

        assert truncated is False
        assert "-line2" in diff
        assert "+modified" in diff
        assert "test.txt" in diff

    def test_diff_additions(self):
        old = "line1\n"
        new = "line1\nline2\nline3\n"

        diff, truncated, total = generate_unified_diff(old, new, "test.txt")

        assert "+line2" in diff
        assert "+line3" in diff

    def test_diff_deletions(self):
        old = "line1\nline2\nline3\n"
        new = "line1\n"

        diff, truncated, total = generate_unified_diff(old, new, "test.txt")

        assert "-line2" in diff
        assert "-line3" in diff

    def test_diff_no_changes(self):
        content = "line1\nline2\n"

        diff, truncated, total = generate_unified_diff(content, content, "test.txt")

        # Empty diff (no changes)
        assert diff == ""
        assert truncated is False

    def test_diff_truncation(self):
        # Create content that will generate a large diff
        old = "\n".join([f"line{i}" for i in range(100)])
        new = "\n".join([f"modified{i}" for i in range(100)])

        diff, truncated, total = generate_unified_diff(old, new, "test.txt", max_lines=20)

        assert truncated is True
        assert total > 20
        assert len(diff.split("\n")) <= 20

    def test_diff_unlimited(self):
        old = "\n".join([f"line{i}" for i in range(100)])
        new = "\n".join([f"modified{i}" for i in range(100)])

        diff, truncated, total = generate_unified_diff(old, new, "test.txt", max_lines=None)

        assert truncated is False


class TestGenerateNewFileDiff:
    """Tests for new file diff generation."""

    def test_new_file_diff(self):
        content = "line1\nline2\nline3\n"

        diff, truncated, total = generate_new_file_diff(content, "new.txt")

        assert truncated is False
        assert "+++ b/new.txt" in diff
        assert "--- /dev/null" in diff
        assert "+line1" in diff
        assert "+line2" in diff
        assert "+line3" in diff

    def test_new_file_diff_empty(self):
        diff, truncated, total = generate_new_file_diff("", "empty.txt")

        assert "--- /dev/null" in diff
        assert "+++ b/empty.txt" in diff

    def test_new_file_diff_truncation(self):
        content = "\n".join([f"line{i}" for i in range(100)])

        diff, truncated, total = generate_new_file_diff(content, "large.txt", max_lines=20)

        assert truncated is True
        assert len(diff.split("\n")) <= 20


class TestGenerateDeleteFileDiff:
    """Tests for delete file diff generation."""

    def test_delete_file_diff(self):
        content = "line1\nline2\nline3\n"

        diff, truncated, total = generate_delete_file_diff(content, "delete.txt")

        assert truncated is False
        assert "--- a/delete.txt" in diff
        assert "+++ /dev/null" in diff
        assert "-line1" in diff
        assert "-line2" in diff
        assert "-line3" in diff

    def test_delete_file_diff_truncation(self):
        content = "\n".join([f"line{i}" for i in range(100)])

        diff, truncated, total = generate_delete_file_diff(content, "large.txt", max_lines=20)

        assert truncated is True
        assert len(diff.split("\n")) <= 20


class TestGetDiffStats:
    """Tests for diff statistics."""

    def test_stats_additions_only(self):
        old = "line1\n"
        new = "line1\nline2\nline3\n"

        stats = get_diff_stats(old, new)

        assert stats["lines_added"] == 2
        assert stats["lines_removed"] == 0

    def test_stats_deletions_only(self):
        old = "line1\nline2\nline3\n"
        new = "line1\n"

        stats = get_diff_stats(old, new)

        assert stats["lines_added"] == 0
        assert stats["lines_removed"] == 2

    def test_stats_modifications(self):
        old = "line1\noriginal\nline3\n"
        new = "line1\nmodified\nline3\n"

        stats = get_diff_stats(old, new)

        # Modification = 1 removal + 1 addition
        assert stats["lines_added"] == 1
        assert stats["lines_removed"] == 1

    def test_stats_no_changes(self):
        content = "line1\nline2\n"

        stats = get_diff_stats(content, content)

        assert stats["lines_added"] == 0
        assert stats["lines_removed"] == 0

    def test_stats_totals(self):
        old = "line1\nline2\n"
        new = "line1\nline2\nline3\n"

        stats = get_diff_stats(old, new)

        assert stats["old_total"] == 2
        assert stats["new_total"] == 3


class TestSummarizeDiff:
    """Tests for diff summary generation."""

    def test_summarize_additions(self):
        old = "line1\n"
        new = "line1\nline2\nline3\n"

        summary = summarize_diff(old, new, "test.txt")

        assert "test.txt" in summary
        assert "+2" in summary

    def test_summarize_deletions(self):
        old = "line1\nline2\nline3\n"
        new = "line1\n"

        summary = summarize_diff(old, new, "test.txt")

        assert "test.txt" in summary
        assert "-2" in summary

    def test_summarize_mixed_changes(self):
        old = "line1\nold\nline3\n"
        new = "line1\nnew\nextra\nline3\n"

        summary = summarize_diff(old, new, "test.txt")

        assert "test.txt" in summary
        # Should show both additions and removals
        assert "+" in summary
        assert "-" in summary

    def test_summarize_no_changes(self):
        content = "line1\nline2\n"

        summary = summarize_diff(content, content, "test.txt")

        assert "test.txt" in summary
        assert "no line changes" in summary.lower()
