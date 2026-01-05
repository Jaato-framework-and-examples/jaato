# shared/plugins/diff_formatter/tests/test_parser.py
"""Tests for unified diff parser."""

import pytest

from ..parser import (
    parse_unified_diff,
    get_paired_lines,
    DiffLine,
    DiffHunk,
    ParsedDiff,
    DiffStats,
)


class TestDiffStats:
    """Tests for DiffStats."""

    def test_stats_str_all_types(self):
        stats = DiffStats(added=5, deleted=3, modified=2)
        assert "+5" in str(stats)
        assert "-3" in str(stats)
        assert "~2" in str(stats)

    def test_stats_str_additions_only(self):
        stats = DiffStats(added=5, deleted=0, modified=0)
        assert str(stats) == "+5"

    def test_stats_str_no_changes(self):
        stats = DiffStats()
        assert str(stats) == "no changes"

    def test_total_changes(self):
        stats = DiffStats(added=5, deleted=3, modified=2)
        assert stats.total_changes == 10


class TestParseUnifiedDiff:
    """Tests for unified diff parsing."""

    def test_parse_simple_diff(self):
        diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 line1
-line2
+modified
 line3
"""
        parsed = parse_unified_diff(diff)

        assert parsed.old_path == "a/test.py"
        assert parsed.new_path == "b/test.py"
        assert len(parsed.hunks) == 1
        assert parsed.display_path == "test.py"

    def test_parse_additions(self):
        diff = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,4 @@
 line1
+added1
+added2
 line2
"""
        parsed = parse_unified_diff(diff)

        hunk = parsed.hunks[0]
        added_lines = [l for l in hunk.lines if l.change_type == "added"]
        assert len(added_lines) == 2
        assert added_lines[0].content == "added1"
        assert added_lines[1].content == "added2"

    def test_parse_deletions(self):
        diff = """--- a/test.py
+++ b/test.py
@@ -1,4 +1,2 @@
 line1
-deleted1
-deleted2
 line2
"""
        parsed = parse_unified_diff(diff)

        hunk = parsed.hunks[0]
        deleted_lines = [l for l in hunk.lines if l.change_type == "deleted"]
        assert len(deleted_lines) == 2
        assert deleted_lines[0].content == "deleted1"

    def test_parse_modifications_paired(self):
        diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 line1
-old
+new
 line3
"""
        parsed = parse_unified_diff(diff)

        hunk = parsed.hunks[0]
        modified_lines = [l for l in hunk.lines if l.change_type == "modified"]
        assert len(modified_lines) == 2  # Old and new

        old_line = next(l for l in modified_lines if l.old_line_no is not None)
        new_line = next(l for l in modified_lines if l.new_line_no is not None)

        assert old_line.content == "old"
        assert new_line.content == "new"
        assert old_line.paired_with == new_line
        assert new_line.paired_with == old_line

    def test_parse_multiple_hunks(self):
        diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 line1
-old1
+new1
 line3
@@ -10,3 +10,3 @@
 line10
-old2
+new2
 line12
"""
        parsed = parse_unified_diff(diff)

        assert len(parsed.hunks) == 2
        assert parsed.hunks[0].old_start == 1
        assert parsed.hunks[1].old_start == 10

    def test_parse_new_file(self):
        diff = """--- /dev/null
+++ b/new.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
"""
        parsed = parse_unified_diff(diff)

        assert parsed.is_new_file
        assert not parsed.is_deleted_file
        assert parsed.display_path == "new.py"

    def test_parse_deleted_file(self):
        diff = """--- a/old.py
+++ /dev/null
@@ -1,3 +0,0 @@
-line1
-line2
-line3
"""
        parsed = parse_unified_diff(diff)

        assert not parsed.is_new_file
        assert parsed.is_deleted_file
        assert parsed.display_path == "old.py"

    def test_parse_hunk_header_extra(self):
        diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@ def my_function():
 line1
-old
+new
 line3
"""
        parsed = parse_unified_diff(diff)

        assert parsed.hunks[0].header_extra == "def my_function():"

    def test_line_numbers(self):
        diff = """--- a/test.py
+++ b/test.py
@@ -5,4 +5,5 @@
 line5
-line6
+modified6
+added
 line7
 line8
"""
        parsed = parse_unified_diff(diff)

        hunk = parsed.hunks[0]
        # First line is context at line 5
        assert hunk.lines[0].old_line_no == 5
        assert hunk.lines[0].new_line_no == 5

    def test_stats_calculation(self):
        diff = """--- a/test.py
+++ b/test.py
@@ -1,4 +1,5 @@
 line1
-deleted
-old
+new
+added1
+added2
 line4
"""
        parsed = parse_unified_diff(diff)
        stats = parsed.stats

        # Pairing algorithm: consecutive deletions paired with consecutive additions
        # deletions: [deleted, old], additions: [new, added1, added2]
        # Pairs: deleted<->new, old<->added1 (2 modified pairs)
        # Remaining: added2 (1 pure addition)
        assert stats.modified == 2
        assert stats.deleted == 0
        assert stats.added == 1


class TestGetPairedLines:
    """Tests for line pairing for side-by-side display."""

    def test_pair_unchanged_lines(self):
        hunk = DiffHunk(old_start=1, old_count=2, new_start=1, new_count=2)
        hunk.lines = [
            DiffLine("line1", 1, 1, "unchanged"),
            DiffLine("line2", 2, 2, "unchanged"),
        ]

        pairs = get_paired_lines(hunk)

        assert len(pairs) == 2
        # Each pair should have the same line on both sides
        assert pairs[0][0] == pairs[0][1]
        assert pairs[1][0] == pairs[1][1]

    def test_pair_additions(self):
        hunk = DiffHunk(old_start=1, old_count=1, new_start=1, new_count=3)
        hunk.lines = [
            DiffLine("line1", 1, 1, "unchanged"),
            DiffLine("added1", None, 2, "added"),
            DiffLine("added2", None, 3, "added"),
        ]

        pairs = get_paired_lines(hunk)

        assert len(pairs) == 3
        assert pairs[0][0] is not None and pairs[0][1] is not None  # unchanged
        assert pairs[1][0] is None and pairs[1][1] is not None  # added
        assert pairs[2][0] is None and pairs[2][1] is not None  # added

    def test_pair_deletions(self):
        hunk = DiffHunk(old_start=1, old_count=3, new_start=1, new_count=1)
        hunk.lines = [
            DiffLine("line1", 1, 1, "unchanged"),
            DiffLine("deleted1", 2, None, "deleted"),
            DiffLine("deleted2", 3, None, "deleted"),
        ]

        pairs = get_paired_lines(hunk)

        assert len(pairs) == 3
        assert pairs[0][0] is not None and pairs[0][1] is not None  # unchanged
        assert pairs[1][0] is not None and pairs[1][1] is None  # deleted
        assert pairs[2][0] is not None and pairs[2][1] is None  # deleted

    def test_pair_modifications(self):
        old_line = DiffLine("old", 2, None, "modified")
        new_line = DiffLine("new", None, 2, "modified")
        old_line.paired_with = new_line
        new_line.paired_with = old_line

        hunk = DiffHunk(old_start=1, old_count=3, new_start=1, new_count=3)
        hunk.lines = [
            DiffLine("line1", 1, 1, "unchanged"),
            old_line,
            new_line,
            DiffLine("line3", 3, 3, "unchanged"),
        ]

        pairs = get_paired_lines(hunk)

        assert len(pairs) == 3
        # Modified pair should have old on left, new on right
        assert pairs[1][0].content == "old"
        assert pairs[1][1].content == "new"


class TestDiffLineDisplay:
    """Tests for DiffLine display methods."""

    def test_line_no_display_both(self):
        line = DiffLine("content", 10, 12, "unchanged")
        old, new = line.line_no_display
        assert old == "10"
        assert new == "12"

    def test_line_no_display_addition(self):
        line = DiffLine("content", None, 5, "added")
        old, new = line.line_no_display
        assert old == ""
        assert new == "5"

    def test_line_no_display_deletion(self):
        line = DiffLine("content", 5, None, "deleted")
        old, new = line.line_no_display
        assert old == "5"
        assert new == ""
