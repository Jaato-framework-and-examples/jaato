# shared/plugins/table_formatter/tests/test_plugin.py
"""Tests for the table formatter plugin."""

import pytest
from shared.plugins.table_formatter import create_plugin, TableFormatterPlugin
from shared.plugins.table_formatter.plugin import _display_width, _pad_to_width


class TestTableDetection:
    """Tests for table pattern detection."""

    def test_detect_markdown_table(self):
        """Should detect markdown table rows."""
        plugin = create_plugin()

        # Valid markdown table row
        assert plugin._classify_line("| Header 1 | Header 2 |") == "markdown"
        assert plugin._classify_line("|col1|col2|col3|") == "markdown"

        # Separator row
        assert plugin._classify_line("|---|---|") == "markdown"
        assert plugin._classify_line("| :--- | :---: | ---: |") == "markdown"

        # Non-table content
        assert plugin._classify_line("Just some text") is None
        assert plugin._classify_line("| partial row without end") is None

    def test_detect_ascii_grid_table(self):
        """Should detect ASCII grid table patterns."""
        plugin = create_plugin()

        # Grid borders
        assert plugin._classify_line("+---+---+") == "ascii_grid"
        assert plugin._classify_line("+------+------+------+") == "ascii_grid"

        # Grid row detection requires being in a table context
        plugin._in_table = True
        plugin._table_type = "ascii_grid"
        assert plugin._classify_line("| cell | cell |") == "ascii_grid"


class TestMarkdownTableParsing:
    """Tests for markdown table parsing."""

    def test_parse_simple_table(self):
        """Should parse a simple markdown table."""
        plugin = create_plugin()
        text = """| Name | Age |
|------|-----|
| Alice | 30 |
| Bob | 25 |"""

        headers, rows, alignments = plugin._parse_markdown_table(text)

        assert headers == ["Name", "Age"]
        assert rows == [["Alice", "30"], ["Bob", "25"]]
        assert alignments == ["left", "left"]

    def test_parse_aligned_table(self):
        """Should parse alignment markers."""
        plugin = create_plugin()
        text = """| Left | Center | Right |
|:-----|:------:|------:|
| a    | b      | c     |"""

        headers, rows, alignments = plugin._parse_markdown_table(text)

        assert headers == ["Left", "Center", "Right"]
        assert alignments == ["left", "center", "right"]

    def test_is_valid_markdown_table(self):
        """Should validate markdown table structure."""
        plugin = create_plugin()

        # Valid - has separator
        valid = """| H1 | H2 |
|---|---|
| C1 | C2 |"""
        assert plugin._is_valid_markdown_table(valid) is True

        # Invalid - no separator
        invalid = """| H1 | H2 |
| C1 | C2 |"""
        assert plugin._is_valid_markdown_table(invalid) is False


class TestTableRendering:
    """Tests for table rendering with box-drawing characters."""

    def test_render_simple_table(self):
        """Should render table with box-drawing borders."""
        plugin = create_plugin()
        text = """| A | B |
|---|---|
| 1 | 2 |"""

        result = plugin._render_markdown_table(text)

        # Check for box-drawing characters
        assert "┌" in result  # Top-left corner
        assert "┐" in result  # Top-right corner
        assert "└" in result  # Bottom-left corner
        assert "┘" in result  # Bottom-right corner
        assert "│" in result  # Vertical border
        assert "─" in result  # Horizontal border
        assert "┼" in result  # Cross (header separator)

    def test_render_preserves_content(self):
        """Should preserve cell content in rendered output."""
        plugin = create_plugin()
        text = """| Name | Value |
|------|-------|
| foo  | bar   |"""

        result = plugin._render_markdown_table(text)

        assert "Name" in result
        assert "Value" in result
        assert "foo" in result
        assert "bar" in result

    def test_render_alignment(self):
        """Should apply column alignment."""
        plugin = create_plugin()
        text = """| L | C | R |
|:--|:-:|--:|
| x | y | z |"""

        result = plugin._render_markdown_table(text)

        # Content should be present and properly aligned
        lines = result.split("\n")
        # Find the data row (should have x, y, z)
        data_line = [l for l in lines if "x" in l and "y" in l and "z" in l][0]

        # The cells should be present (actual alignment is visual)
        assert "x" in data_line
        assert "y" in data_line
        assert "z" in data_line


class TestStreamingProcessing:
    """Tests for streaming chunk processing."""

    def test_process_complete_table(self):
        """Should process a complete table in one chunk."""
        plugin = create_plugin()
        text = """Here is a table:
| A | B |
|---|---|
| 1 | 2 |
And some text after."""

        result = list(plugin.process_chunk(text))
        result.extend(plugin.flush())

        combined = "".join(result)
        assert "Here is a table:" in combined
        assert "┌" in combined  # Box-drawing in table
        assert "And some text after." in combined

    def test_process_streaming_table(self):
        """Should buffer and process streamed table lines."""
        plugin = create_plugin()

        # Stream line by line
        result = []
        result.extend(plugin.process_chunk("| H1 | H2 |\n"))
        result.extend(plugin.process_chunk("|---|---|\n"))
        result.extend(plugin.process_chunk("| C1 | C2 |\n"))
        result.extend(plugin.process_chunk("End of table\n"))
        result.extend(plugin.flush())

        combined = "".join(result)
        assert "┌" in combined  # Table was rendered with box chars
        assert "End of table" in combined

    def test_flush_incomplete_table(self):
        """Should handle flushing incomplete table gracefully."""
        plugin = create_plugin()

        # Just a single pipe line - not a valid table
        result = list(plugin.process_chunk("| just | one | row |\n"))
        result.extend(plugin.flush())

        combined = "".join(result)
        # Should pass through since no separator line
        assert "just" in combined

    def test_reset_clears_buffer(self):
        """Reset should clear internal state."""
        plugin = create_plugin()

        # Buffer some content
        list(plugin.process_chunk("| A | B |\n"))
        assert len(plugin._buffer) > 0

        plugin.reset()

        assert plugin._buffer == []
        assert plugin._in_table is False
        assert plugin._table_type is None


class TestConfiguration:
    """Tests for plugin configuration."""

    def test_default_priority(self):
        """Should have correct default priority."""
        plugin = create_plugin()
        assert plugin.priority == 25  # Structural formatting range

    def test_initialize_priority(self):
        """Should accept custom priority."""
        plugin = create_plugin()
        plugin.initialize({"priority": 30})
        assert plugin.priority == 30

    def test_set_console_width(self):
        """Should accept console width updates."""
        plugin = create_plugin()
        plugin.set_console_width(80)
        assert plugin._console_width == 80

    def test_name_property(self):
        """Should return correct plugin name."""
        plugin = create_plugin()
        assert plugin.name == "table_formatter"


class TestDisplayWidth:
    """Tests for display width calculation with wide characters."""

    def test_display_width_ascii(self):
        """ASCII characters should have width 1."""
        assert _display_width("hello") == 5
        assert _display_width("test") == 4
        assert _display_width("") == 0

    def test_display_width_emoji(self):
        """Emojis should have width 2."""
        assert _display_width("✅") == 2
        assert _display_width("❌") == 2
        assert _display_width("✅ Pass") == 7  # 2 + 1 + 4

    def test_display_width_mixed(self):
        """Mixed content should sum correctly."""
        assert _display_width("OK ✅") == 5  # 2 + 1 + 2
        assert _display_width("A✅B") == 4  # 1 + 2 + 1

    def test_pad_to_width_left(self):
        """Left padding should add spaces on right."""
        result = _pad_to_width("hi", 5, "left")
        assert result == "hi   "
        assert _display_width(result) == 5

    def test_pad_to_width_right(self):
        """Right padding should add spaces on left."""
        result = _pad_to_width("hi", 5, "right")
        assert result == "   hi"
        assert _display_width(result) == 5

    def test_pad_to_width_center(self):
        """Center padding should add spaces on both sides."""
        result = _pad_to_width("hi", 6, "center")
        assert result == "  hi  "
        assert _display_width(result) == 6

    def test_pad_to_width_emoji(self):
        """Padding should account for emoji width."""
        # ✅ has display width 2, so "✅ Pass" has width 7
        # Padding to 10 should add 3 spaces
        result = _pad_to_width("✅ Pass", 10, "left")
        assert result == "✅ Pass   "
        assert _display_width(result) == 10

    def test_render_table_with_emoji(self):
        """Tables with emojis should have aligned columns."""
        plugin = create_plugin()
        text = """| Test | Result |
|------|--------|
| Basic | ✅ Pass |
| Other | ❌ Fail |"""

        result = plugin._render_markdown_table(text)
        lines = result.strip().split("\n")

        # All lines should have the same display width
        widths = [_display_width(line) for line in lines]
        assert len(set(widths)) == 1, f"Lines have different widths: {widths}"


class TestWidthConstraint:
    """Tests for width-constrained rendering (cell wrapping, shrinking)."""

    def test_wide_table_fits_within_console_width(self):
        """Tables wider than the console should be wrapped to fit."""
        plugin = create_plugin()
        plugin.set_console_width(60)

        long_tags = ", ".join(f"`tag-{i}`" for i in range(20))
        text = (
            "| # | Name | Tags |\n"
            "|---|------|------|\n"
            f"| 1 | alpha | {long_tags} |\n"
        )

        result = plugin._render_markdown_table(text)
        lines = result.strip().split("\n")

        # Every rendered line must fit within the console width.
        widths = [_display_width(line) for line in lines]
        assert max(widths) <= 60, f"Line(s) exceed console width: {widths}"

        # And all lines (borders + row lines) must share a common width
        # so box-drawing characters stay aligned.
        assert len(set(widths)) == 1, f"Lines have different widths: {widths}"

    def test_wide_row_wraps_to_multiple_visual_lines(self):
        """A cell wider than its column should wrap to multiple lines."""
        plugin = create_plugin()
        plugin.set_console_width(40)

        text = (
            "| Name | Description |\n"
            "|------|-------------|\n"
            "| foo | this is a fairly long description that should wrap onto several lines within its column |\n"
        )

        result = plugin._render_markdown_table(text)
        lines = result.strip().split("\n")

        # Header row: top border + header line + mid border = 3 lines minimum.
        # Data row should expand beyond a single line because of wrapping.
        # Expected structure: top, header, middle, >=2 data lines, bottom.
        assert len(lines) >= 6, f"Expected multi-line data row, got: {lines}"

    def test_narrow_console_keeps_minimum_column_width(self):
        """Even in a very narrow terminal we should not collapse columns."""
        plugin = create_plugin()
        plugin.set_console_width(10)  # Smaller than any reasonable 3-col table

        text = (
            "| a | b | c |\n"
            "|---|---|---|\n"
            "| hello | world | !! |\n"
        )

        result = plugin._render_markdown_table(text)
        # Should not raise and should produce some box-drawing output.
        assert "┌" in result
        assert "└" in result

    def test_narrow_rendering_preserves_cell_content(self):
        """Wrapped cells should still contain the full content (possibly across lines)."""
        plugin = create_plugin()
        plugin.set_console_width(50)

        text = (
            "| Name | Tags |\n"
            "|------|------|\n"
            "| x | alpha beta gamma delta epsilon zeta eta theta |\n"
        )

        result = plugin._render_markdown_table(text)
        # Every word should still appear somewhere in the output,
        # even if split across wrapped lines.
        for word in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]:
            assert word in result, f"Lost word '{word}' during wrapping"


class TestWrapCell:
    """Unit tests for the _wrap_cell helper."""

    def test_short_text_not_wrapped(self):
        plugin = create_plugin()
        assert plugin._wrap_cell("hello", 10) == ["hello"]

    def test_breaks_on_whitespace(self):
        plugin = create_plugin()
        lines = plugin._wrap_cell("alpha beta gamma", 6)
        for line in lines:
            assert _display_width(line) <= 6
        assert "alpha" in lines[0]

    def test_breaks_long_token_char_by_char(self):
        plugin = create_plugin()
        long_token = "x" * 25
        lines = plugin._wrap_cell(long_token, 10)
        for line in lines:
            assert _display_width(line) <= 10
        assert "".join(lines) == long_token

    def test_zero_width_returns_original(self):
        plugin = create_plugin()
        assert plugin._wrap_cell("anything", 0) == ["anything"]
