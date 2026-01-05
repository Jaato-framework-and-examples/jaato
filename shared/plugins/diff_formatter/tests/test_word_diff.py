# shared/plugins/diff_formatter/tests/test_word_diff.py
"""Tests for word-level diff computation."""

import pytest

from ..word_diff import (
    compute_word_diff,
    compute_word_diff_by_words,
    render_word_diff_old,
    render_word_diff_new,
    WordDiff,
)


class TestComputeWordDiff:
    """Tests for character-level diff computation."""

    def test_identical_lines(self):
        word_diff = compute_word_diff("hello world", "hello world")

        # No changes
        assert not any(changed for _, changed in word_diff.old_segments)
        assert not any(changed for _, changed in word_diff.new_segments)
        assert not word_diff.has_changes

    def test_simple_change(self):
        word_diff = compute_word_diff("hello world", "hello there")

        assert word_diff.has_changes

        # Character-level diff may find partial matches (e.g., 'r' in both)
        # so check that SOME parts are marked as changed
        old_changed = "".join(text for text, changed in word_diff.old_segments if changed)
        new_changed = "".join(text for text, changed in word_diff.new_segments if changed)

        # Both should have some changed content
        assert len(old_changed) > 0
        assert len(new_changed) > 0

        # The unchanged prefix "hello " should be preserved
        old_unchanged = "".join(text for text, changed in word_diff.old_segments if not changed)
        assert "hello " in old_unchanged

    def test_addition_at_end(self):
        word_diff = compute_word_diff("hello", "hello world")

        assert word_diff.has_changes

        # Old should have no changed parts (just missing content)
        # New should have " world" marked as changed
        new_changed = "".join(text for text, changed in word_diff.new_segments if changed)
        assert "world" in new_changed

    def test_deletion_at_end(self):
        word_diff = compute_word_diff("hello world", "hello")

        assert word_diff.has_changes

        # Old should have " world" marked as changed
        old_changed = "".join(text for text, changed in word_diff.old_segments if changed)
        assert "world" in old_changed

    def test_change_in_middle(self):
        word_diff = compute_word_diff("the quick fox", "the slow fox")

        assert word_diff.has_changes

        old_changed = "".join(text for text, changed in word_diff.old_segments if changed)
        new_changed = "".join(text for text, changed in word_diff.new_segments if changed)

        assert "quick" in old_changed
        assert "slow" in new_changed

    def test_preserves_unchanged_portions(self):
        word_diff = compute_word_diff("prefix_old_suffix", "prefix_new_suffix")

        # Check that "prefix_" and "_suffix" are not marked as changed
        old_unchanged = "".join(text for text, changed in word_diff.old_segments if not changed)
        assert "prefix_" in old_unchanged
        assert "_suffix" in old_unchanged


class TestComputeWordDiffByWords:
    """Tests for word-level diff computation."""

    def test_word_change(self):
        word_diff = compute_word_diff_by_words("hello world", "hello there")

        assert word_diff.has_changes

        old_changed = "".join(text for text, changed in word_diff.old_segments if changed)
        new_changed = "".join(text for text, changed in word_diff.new_segments if changed)

        assert "world" in old_changed
        assert "there" in new_changed

    def test_word_addition(self):
        word_diff = compute_word_diff_by_words("one two", "one two three")

        new_changed = "".join(text for text, changed in word_diff.new_segments if changed)
        assert "three" in new_changed

    def test_preserves_whitespace(self):
        word_diff = compute_word_diff_by_words("a  b", "a  c")

        # The double space should be preserved
        full_old = "".join(text for text, _ in word_diff.old_segments)
        assert "  " in full_old


class TestRenderWordDiff:
    """Tests for word diff rendering."""

    def test_render_with_colors(self):
        word_diff = compute_word_diff("hello world", "hello there")

        color_start = "[START]"
        color_end = "[END]"

        old_rendered = render_word_diff_old(word_diff, color_start, color_end)
        new_rendered = render_word_diff_new(word_diff, color_start, color_end)

        # Changed parts should be wrapped in color codes
        assert "[START]" in old_rendered
        assert "[END]" in old_rendered
        assert "[START]" in new_rendered
        assert "[END]" in new_rendered

    def test_render_no_changes(self):
        word_diff = compute_word_diff("same", "same")

        color_start = "[START]"
        color_end = "[END]"

        old_rendered = render_word_diff_old(word_diff, color_start, color_end)
        new_rendered = render_word_diff_new(word_diff, color_start, color_end)

        # No color codes when no changes
        assert "[START]" not in old_rendered
        assert "[START]" not in new_rendered
        assert old_rendered == "same"
        assert new_rendered == "same"

    def test_render_preserves_content(self):
        word_diff = compute_word_diff("the quick fox", "the slow fox")

        # Render without colors to check content preservation
        old_rendered = render_word_diff_old(word_diff, "", "")
        new_rendered = render_word_diff_new(word_diff, "", "")

        assert old_rendered == "the quick fox"
        assert new_rendered == "the slow fox"
