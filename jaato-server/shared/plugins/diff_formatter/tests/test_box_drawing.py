# shared/plugins/diff_formatter/tests/test_box_drawing.py
"""Tests for box drawing utilities."""

import pytest

from ..box_drawing import (
    truncate_text,
    pad_text,
    box_top,
    box_bottom,
    box_separator,
    box_row,
    box_row_colored,
    calculate_column_widths,
    BoxBuilder,
    BOX_TL, BOX_TR, BOX_BL, BOX_BR, BOX_H, BOX_V,
)


class TestTruncateText:
    """Tests for text truncation."""

    def test_no_truncation_needed(self):
        result = truncate_text("hello", 10)
        assert result == "hello"

    def test_exact_width(self):
        result = truncate_text("hello", 5)
        assert result == "hello"

    def test_truncation_with_ellipsis(self):
        result = truncate_text("hello world", 8)
        assert len(result) == 8
        assert result.endswith("…")
        assert result == "hello w…"

    def test_very_short_width(self):
        result = truncate_text("hello", 1)
        assert result == "…"

    def test_custom_ellipsis(self):
        result = truncate_text("hello world", 9, ellipsis="...")
        assert result.endswith("...")


class TestPadText:
    """Tests for text padding."""

    def test_left_align(self):
        result = pad_text("hi", 5, align="left")
        assert result == "hi   "
        assert len(result) == 5

    def test_right_align(self):
        result = pad_text("hi", 5, align="right")
        assert result == "   hi"
        assert len(result) == 5

    def test_center_align(self):
        result = pad_text("hi", 6, align="center")
        assert result == "  hi  "
        assert len(result) == 6

    def test_no_padding_needed(self):
        result = pad_text("hello", 5)
        assert result == "hello"

    def test_truncation_if_too_long(self):
        result = pad_text("hello world", 5)
        assert len(result) == 5


class TestBoxBorders:
    """Tests for box border creation."""

    def test_box_top_simple(self):
        result = box_top(10)
        assert result.startswith(BOX_TL)
        assert result.endswith(BOX_TR)
        assert len(result) == 10

    def test_box_top_with_title(self):
        result = box_top(20, "test")
        assert BOX_TL in result
        assert BOX_TR in result
        assert "test" in result

    def test_box_bottom(self):
        result = box_bottom(10)
        assert result.startswith(BOX_BL)
        assert result.endswith(BOX_BR)
        assert len(result) == 10

    def test_box_separator(self):
        result = box_separator([5, 5])
        assert "├" in result
        assert "┤" in result
        assert "┼" in result


class TestBoxRow:
    """Tests for box row creation."""

    def test_simple_row(self):
        result = box_row(["a", "b"], [3, 3])
        assert BOX_V in result
        assert "a" in result
        assert "b" in result

    def test_row_truncates_content(self):
        result = box_row(["hello world"], [5])
        # Content should be truncated to fit
        assert len([c for c in result if c == BOX_V]) == 2  # Start and end

    def test_colored_row(self):
        result = box_row_colored(
            ["a", "b"],
            [3, 3],
            ["[RED]", "[GREEN]"],
            reset="[RESET]"
        )
        assert "[RED]" in result
        assert "[GREEN]" in result
        assert "[RESET]" in result


class TestCalculateColumnWidths:
    """Tests for column width calculation."""

    def test_equal_distribution(self):
        widths = calculate_column_widths(22, 2)  # 22 - 3 separators = 19, /2 = 9 each + extra
        assert len(widths) == 2
        assert sum(widths) == 22 - 3  # Total minus separators

    def test_with_fixed_columns(self):
        widths = calculate_column_widths(30, 3, fixed_columns=[(0, 5)])
        assert widths[0] == 5
        # Remaining columns split the rest


class TestBoxBuilder:
    """Tests for the BoxBuilder helper class."""

    def test_simple_box(self):
        builder = BoxBuilder(width=20)
        builder.add_row(["a", "b"])
        result = builder.build()

        assert BOX_TL in result
        assert BOX_BR in result
        assert "a" in result
        assert "b" in result

    def test_box_with_title(self):
        builder = BoxBuilder(width=30)
        builder.add_title("My Table")
        builder.add_row(["data"])
        result = builder.build()

        assert "My Table" in result

    def test_box_with_header(self):
        builder = BoxBuilder(width=30)
        builder.add_header(["Col1", "Col2"])
        builder.add_row(["a", "b"])
        result = builder.build()

        assert "Col1" in result
        assert "Col2" in result
        # Should have separator after header
        assert "┼" in result

    def test_box_with_separator(self):
        builder = BoxBuilder(width=30)
        builder.add_row(["a", "b"])
        builder.add_separator()
        builder.add_row(["c", "d"])
        result = builder.build()

        lines = result.split("\n")
        # Should have multiple separator lines
        separator_lines = [l for l in lines if "─" in l and "┼" in l]
        assert len(separator_lines) >= 1
