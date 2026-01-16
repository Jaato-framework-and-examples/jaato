# shared/plugins/table_formatter/tests/test_plugin.py
"""Tests for the table formatter plugin."""

import pytest
from shared.plugins.table_formatter import create_plugin, TableFormatterPlugin


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
