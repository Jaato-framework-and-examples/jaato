# shared/plugins/code_block_formatter/tests/test_truncation.py
"""Tests for code block line truncation to terminal width."""

import re

from rich.text import Text

from shared.plugins.code_block_formatter.plugin import (
    CodeBlockFormatterPlugin,
    _get_content_width,
    _trim_line_to_width,
)
from shared.plugins.table_formatter.plugin import _display_width


ANSI_PATTERN = re.compile(r'\033\[[0-9;]*m')


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_PATTERN.sub('', text)


def _visible_width(text: str) -> int:
    """Get visible display width of text with ANSI codes."""
    return _display_width(_strip_ansi(text))


class TestTrimLineToWidth:
    """Tests for _trim_line_to_width with truncation indicator."""

    def test_no_truncation_indicator_when_not_needed(self):
        """Lines within target width should not get an indicator."""
        result = _trim_line_to_width("hello", 10)
        assert "▸" not in _strip_ansi(result)
        assert "hello" in _strip_ansi(result)

    def test_truncation_indicator_appended(self):
        """When truncation_indicator is provided, it should appear in output."""
        long_line = "x" * 100
        result = _trim_line_to_width(long_line, 20, truncation_indicator="▸")
        plain = _strip_ansi(result)
        assert plain.endswith("▸")
        assert _display_width(plain) <= 20

    def test_truncation_preserves_target_width(self):
        """Truncated line (including indicator) should not exceed target_width."""
        long_line = "abcdefghijklmnopqrstuvwxyz" * 5
        result = _trim_line_to_width(long_line, 30, truncation_indicator="▸")
        plain = _strip_ansi(result)
        assert _display_width(plain) <= 30

    def test_empty_line(self):
        result = _trim_line_to_width("", 80)
        assert result == ""

    def test_zero_target_width(self):
        result = _trim_line_to_width("hello", 0)
        assert result == ""

    def test_no_indicator_by_default(self):
        """Default truncation_indicator is empty, so no indicator appears."""
        result = _trim_line_to_width("x" * 100, 10)
        plain = _strip_ansi(result)
        assert "▸" not in plain


class TestCodeBlockTruncation:
    """Tests for code block rendering with terminal width truncation."""

    def _make_formatter(self, console_width=80, line_numbers=False):
        plugin = CodeBlockFormatterPlugin()
        plugin.initialize({
            "line_numbers": line_numbers,
            "console_width": console_width,
        })
        plugin.set_console_width(console_width)
        return plugin

    def _render(self, plugin, code, lang="python"):
        """Render a code block and return the output lines."""
        chunks = list(plugin.process_chunk(f"```{lang}\n{code}\n```"))
        chunks.extend(plugin.flush())
        output = "".join(chunks)
        # Split into non-empty lines
        return [line for line in output.split('\n') if line.strip()]

    def test_short_lines_not_truncated(self):
        """Lines shorter than terminal width should not be truncated."""
        plugin = self._make_formatter(console_width=80)
        lines = self._render(plugin, "x = 1")
        for line in lines:
            assert "▸" not in _strip_ansi(line)

    def test_long_lines_truncated_to_terminal_width(self):
        """Lines longer than terminal width should be truncated with indicator."""
        plugin = self._make_formatter(console_width=40)
        long_code = "x = " + "'" + "a" * 200 + "'"
        lines = self._render(plugin, long_code)
        for line in lines:
            # Each rendered line (including indent) should not exceed console width
            assert _visible_width(line) <= 40, (
                f"Line exceeds terminal width: {_visible_width(line)} > 40"
            )

    def test_truncation_indicator_present_for_long_lines(self):
        """The ▸ indicator should appear when lines are truncated."""
        plugin = self._make_formatter(console_width=40)
        long_code = "variable_name = " + "'" + "a" * 200 + "'"
        lines = self._render(plugin, long_code)
        # At least one line should have the truncation indicator
        has_indicator = any("▸" in _strip_ansi(line) for line in lines)
        assert has_indicator, "Expected truncation indicator ▸ in output"

    def test_line_numbers_accounted_for(self):
        """With line numbers enabled, lines should still fit terminal width."""
        plugin = self._make_formatter(console_width=60, line_numbers=True)
        long_code = "\n".join(["x = " + "'" + "a" * 200 + "'"] * 5)
        lines = self._render(plugin, long_code)
        for line in lines:
            assert _visible_width(line) <= 60, (
                f"Line with line numbers exceeds terminal width: {_visible_width(line)} > 60"
            )

    def test_mixed_short_and_long_lines(self):
        """Short lines stay intact; only long lines get truncated."""
        plugin = self._make_formatter(console_width=50)
        code = "short = 1\n" + "long_var = " + "'" + "b" * 200 + "'"
        lines = self._render(plugin, code)
        for line in lines:
            assert _visible_width(line) <= 50

    def test_width_update_affects_truncation(self):
        """Changing console_width via set_console_width should affect rendering."""
        plugin = self._make_formatter(console_width=200)
        long_code = "x = " + "'" + "a" * 100 + "'"

        # At 200 width, no truncation
        lines = self._render(plugin, long_code)
        has_indicator = any("▸" in _strip_ansi(line) for line in lines)
        assert not has_indicator, "Should not truncate at wide terminal"

        # Shrink terminal
        plugin.reset()
        plugin.set_console_width(40)
        lines = self._render(plugin, long_code)
        has_indicator = any("▸" in _strip_ansi(line) for line in lines)
        assert has_indicator, "Should truncate at narrow terminal"

    def test_fallback_no_crash_on_narrow_terminal(self):
        """Very narrow terminal should not crash."""
        plugin = self._make_formatter(console_width=20)
        code = "x = " + "'" + "a" * 100 + "'"
        # Should not raise
        lines = self._render(plugin, code)
        for line in lines:
            assert _visible_width(line) <= 20
