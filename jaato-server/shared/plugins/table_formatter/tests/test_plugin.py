# shared/plugins/table_formatter/tests/test_plugin.py
"""Tests for the table formatter plugin.

The server always emits semantic ``<j-table>`` markup — clients render
it natively, so there is no terminal box-drawing path to test here.
"""

from shared.plugins.table_formatter import create_plugin, TableFormatterPlugin
from shared.plugins.table_formatter.plugin import _display_width


class TestTableDetection:
    """Tests for table pattern detection."""

    def test_detect_markdown_table(self):
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
        plugin = create_plugin()

        # Grid borders
        assert plugin._classify_line("+---+---+") == "ascii_grid"
        assert plugin._classify_line("+------+------+------+") == "ascii_grid"


class TestMarkdownTableParsing:
    """Tests for markdown table parsing."""

    def test_parse_simple_table(self):
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
        plugin = create_plugin()
        text = """| Left | Center | Right |
|:-----|:------:|------:|
| a    | b      | c     |"""

        headers, rows, alignments = plugin._parse_markdown_table(text)

        assert headers == ["Left", "Center", "Right"]
        assert alignments == ["left", "center", "right"]

    def test_is_valid_markdown_table(self):
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


class TestSemanticMarkup:
    """Tests for the ``<j-table>`` semantic rendering."""

    def _render(self, text: str) -> str:
        return create_plugin()._render_semantic_table(text)

    def test_no_ansi_ever_emitted(self):
        """The wire format carries no ANSI / box-drawing — clients render locally."""
        out = self._render("| A | B |\n|---|---|\n| 1 | 2 |")
        assert "\x1b[" not in out
        # No Unicode box-drawing leaks out
        for box_char in ("┌", "┐", "└", "┘", "│", "─", "┼"):
            assert box_char not in out

    def test_tags_and_cells(self):
        out = self._render("""| Name | Age |
|------|-----|
| Alice | 30 |
| Bob | 25 |""")

        assert out.startswith("<j-table>")
        assert out.rstrip().endswith("</j-table>")
        assert "<j-thead>" in out
        assert "<j-th>Name</j-th>" in out
        assert "<j-th>Age</j-th>" in out
        assert "<j-tr><j-td>Alice</j-td><j-td>30</j-td></j-tr>" in out
        assert "<j-tr><j-td>Bob</j-td><j-td>25</j-td></j-tr>" in out

    def test_empty_table_passes_through(self):
        """A malformed input with no cells should be returned as-is."""
        # _parse returns empty headers+rows for broken input — the renderer
        # falls back to passing the text through with a trailing newline.
        plugin = create_plugin()
        result = plugin._render_semantic_table("")
        assert "<j-table>" not in result


class TestStreamingProcessing:
    """Tests for streaming chunk processing."""

    def test_process_complete_table(self):
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
        assert "<j-table>" in combined
        assert "And some text after." in combined

    def test_process_streaming_table(self):
        plugin = create_plugin()

        # Stream line by line
        result = []
        result.extend(plugin.process_chunk("| H1 | H2 |\n"))
        result.extend(plugin.process_chunk("|---|---|\n"))
        result.extend(plugin.process_chunk("| C1 | C2 |\n"))
        result.extend(plugin.process_chunk("End of table\n"))
        result.extend(plugin.flush())

        combined = "".join(result)
        assert "<j-table>" in combined
        assert "End of table" in combined

    def test_flush_incomplete_table(self):
        plugin = create_plugin()

        # Just a single pipe line - not a valid table
        result = list(plugin.process_chunk("| just | one | row |\n"))
        result.extend(plugin.flush())

        combined = "".join(result)
        # Should pass through since no separator line
        assert "just" in combined

    def test_reset_clears_buffer(self):
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
        plugin = create_plugin()
        assert plugin.priority == 25  # Structural formatting range

    def test_initialize_priority(self):
        plugin = create_plugin()
        plugin.initialize({"priority": 30})
        assert plugin.priority == 30

    def test_set_console_width(self):
        plugin = create_plugin()
        plugin.set_console_width(80)
        assert plugin._console_width == 80

    def test_name_property(self):
        plugin = create_plugin()
        assert plugin.name == "table_formatter"


class TestDisplayWidth:
    """Tests for the shared display-width helper (still used elsewhere)."""

    def test_display_width_ascii(self):
        assert _display_width("hello") == 5
        assert _display_width("test") == 4
        assert _display_width("") == 0

    def test_display_width_emoji(self):
        assert _display_width("✅") == 2
        assert _display_width("❌") == 2
        assert _display_width("✅ Pass") == 7  # 2 + 1 + 4

    def test_display_width_mixed(self):
        assert _display_width("OK ✅") == 5  # 2 + 1 + 2
        assert _display_width("A✅B") == 4  # 1 + 2 + 1
