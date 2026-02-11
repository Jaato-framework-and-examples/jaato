# shared/plugins/inline_markdown_formatter/tests/test_inline_markdown.py
"""Tests for the inline markdown formatter plugin."""

import pytest

from shared.plugins.inline_markdown_formatter.plugin import (
    InlineMarkdownFormatterPlugin,
    create_plugin,
    BOLD_ON,
    BOLD_OFF,
    ITALIC_ON,
    ITALIC_OFF,
    STRIKETHROUGH_ON,
    STRIKETHROUGH_OFF,
    RESET,
    _hex_to_ansi_fg,
    _hex_to_ansi_bg,
    DEFAULT_INLINE_CODE_FG,
    DEFAULT_INLINE_CODE_BG,
    DEFAULT_LINK_FG,
    UNDERLINE_ON,
)


@pytest.fixture
def formatter():
    """Create a fresh formatter for each test."""
    return create_plugin()


# ==================== Basic Pattern Tests ====================


class TestInlineCode:
    """Tests for backtick inline code formatting."""

    def test_basic_inline_code(self, formatter):
        result = _format(formatter, "Use `foo()` here")
        code_start = _hex_to_ansi_fg(DEFAULT_INLINE_CODE_FG) + _hex_to_ansi_bg(DEFAULT_INLINE_CODE_BG)
        assert result == f"Use {code_start}foo(){RESET} here"

    def test_multiple_inline_code(self, formatter):
        result = _format(formatter, "`foo` and `bar`")
        code_start = _hex_to_ansi_fg(DEFAULT_INLINE_CODE_FG) + _hex_to_ansi_bg(DEFAULT_INLINE_CODE_BG)
        assert result == f"{code_start}foo{RESET} and {code_start}bar{RESET}"

    def test_inline_code_with_spaces(self, formatter):
        result = _format(formatter, "`hello world`")
        code_start = _hex_to_ansi_fg(DEFAULT_INLINE_CODE_FG) + _hex_to_ansi_bg(DEFAULT_INLINE_CODE_BG)
        assert result == f"{code_start}hello world{RESET}"

    def test_empty_backticks_not_matched(self, formatter):
        result = _format(formatter, "``")
        assert result == "``"

    def test_inline_code_with_special_chars(self, formatter):
        result = _format(formatter, "Run `ls -la /tmp`")
        code_start = _hex_to_ansi_fg(DEFAULT_INLINE_CODE_FG) + _hex_to_ansi_bg(DEFAULT_INLINE_CODE_BG)
        assert result == f"Run {code_start}ls -la /tmp{RESET}"


class TestBold:
    """Tests for **bold** formatting."""

    def test_basic_bold(self, formatter):
        result = _format(formatter, "This is **bold** text")
        assert result == f"This is {BOLD_ON}bold{BOLD_OFF} text"

    def test_bold_at_start(self, formatter):
        result = _format(formatter, "**bold** at start")
        assert result == f"{BOLD_ON}bold{BOLD_OFF} at start"

    def test_bold_at_end(self, formatter):
        result = _format(formatter, "at end **bold**")
        assert result == f"at end {BOLD_ON}bold{BOLD_OFF}"

    def test_bold_with_spaces_inside(self, formatter):
        result = _format(formatter, "**bold text here**")
        assert result == f"{BOLD_ON}bold text here{BOLD_OFF}"

    def test_double_asterisk_no_content_not_matched(self, formatter):
        """** alone should not match."""
        result = _format(formatter, "just ** here")
        assert result == "just ** here"

    def test_bold_no_leading_space(self, formatter):
        """**  text** (space after opening) should not match."""
        result = _format(formatter, "** nope**")
        assert result == "** nope**"


class TestItalic:
    """Tests for *italic* formatting."""

    def test_basic_italic(self, formatter):
        result = _format(formatter, "This is *italic* text")
        assert result == f"This is {ITALIC_ON}italic{ITALIC_OFF} text"

    def test_italic_not_matched_with_spaces(self, formatter):
        """*  text* (space after opening) should not match."""
        result = _format(formatter, "2 * 3 * 4")
        assert result == "2 * 3 * 4"

    def test_italic_in_sentence(self, formatter):
        result = _format(formatter, "Make it *really* clear")
        assert result == f"Make it {ITALIC_ON}really{ITALIC_OFF} clear"


class TestBoldItalic:
    """Tests for ***bold italic*** formatting."""

    def test_basic_bold_italic(self, formatter):
        result = _format(formatter, "This is ***important*** text")
        assert result == f"This is {BOLD_ON}{ITALIC_ON}important{ITALIC_OFF}{BOLD_OFF} text"


class TestStrikethrough:
    """Tests for ~~strikethrough~~ formatting."""

    def test_basic_strikethrough(self, formatter):
        result = _format(formatter, "This is ~~deleted~~ text")
        assert result == f"This is {STRIKETHROUGH_ON}deleted{STRIKETHROUGH_OFF} text"

    def test_strikethrough_no_match_single_tilde(self, formatter):
        result = _format(formatter, "~not strikethrough~")
        assert result == "~not strikethrough~"


class TestLinks:
    """Tests for [text](url) formatting."""

    def test_basic_link(self, formatter):
        result = _format(formatter, "See [docs](https://example.com)")
        link_start = _hex_to_ansi_fg(DEFAULT_LINK_FG) + UNDERLINE_ON
        assert result == f"See {link_start}docs{RESET}"

    def test_link_with_long_text(self, formatter):
        result = _format(formatter, "[click here for details](https://example.com/path)")
        link_start = _hex_to_ansi_fg(DEFAULT_LINK_FG) + UNDERLINE_ON
        assert result == f"{link_start}click here for details{RESET}"


# ==================== Combined Pattern Tests ====================


class TestCombinedPatterns:
    """Tests for multiple inline elements in the same line."""

    def test_code_and_bold(self, formatter):
        result = _format(formatter, "Use `foo()` for **important** stuff")
        code_start = _hex_to_ansi_fg(DEFAULT_INLINE_CODE_FG) + _hex_to_ansi_bg(DEFAULT_INLINE_CODE_BG)
        assert f"{code_start}foo(){RESET}" in result
        assert f"{BOLD_ON}important{BOLD_OFF}" in result

    def test_all_elements(self, formatter):
        """Line with code, bold, italic, strikethrough."""
        result = _format(formatter, "`code` **bold** *italic* ~~strike~~")
        code_start = _hex_to_ansi_fg(DEFAULT_INLINE_CODE_FG) + _hex_to_ansi_bg(DEFAULT_INLINE_CODE_BG)
        assert f"{code_start}code{RESET}" in result
        assert f"{BOLD_ON}bold{BOLD_OFF}" in result
        assert f"{ITALIC_ON}italic{ITALIC_OFF}" in result
        assert f"{STRIKETHROUGH_ON}strike{STRIKETHROUGH_OFF}" in result


# ==================== Pass-through Tests ====================


class TestPassthrough:
    """Tests for content that should NOT be formatted."""

    def test_plain_text(self, formatter):
        result = _format(formatter, "Just plain text")
        assert result == "Just plain text"

    def test_empty_string(self, formatter):
        result = _format(formatter, "")
        assert result == ""

    def test_ansi_chunks_passed_through(self, formatter):
        """Chunks with ANSI codes should pass through unchanged."""
        ansi_text = "\x1b[36mhello\x1b[0m"
        result = list(formatter.process_chunk(ansi_text))
        assert result == [ansi_text]

    def test_triple_backtick_not_matched(self, formatter):
        """Triple backticks (code fence markers) should not be matched as inline code."""
        result = _format(formatter, "```")
        assert result == "```"

    def test_single_asterisk(self, formatter):
        result = _format(formatter, "a * b")
        assert result == "a * b"

    def test_asterisks_with_no_content(self, formatter):
        result = _format(formatter, "****")
        assert result == "****"

    def test_underscores_not_processed(self, formatter):
        """Underscore emphasis should NOT be processed (to avoid false positives)."""
        result = _format(formatter, "some_variable_name")
        assert result == "some_variable_name"

        result = _format(formatter, "_italic_ or __bold__")
        assert result == "_italic_ or __bold__"


# ==================== Streaming Tests ====================


class TestStreaming:
    """Tests for streaming behavior (process_chunk / flush)."""

    def test_complete_line_yields_immediately(self, formatter):
        """A complete line (with newline) should be formatted and yielded."""
        result = list(formatter.process_chunk("Hello **world**\n"))
        assert len(result) == 1
        assert f"{BOLD_ON}world{BOLD_OFF}" in result[0]
        assert result[0].endswith('\n')

    def test_partial_line_buffered(self, formatter):
        """A partial line (no newline) should be buffered."""
        result = list(formatter.process_chunk("Hello **wor"))
        assert result == []  # Buffered, not yielded

    def test_flush_yields_buffered(self, formatter):
        """flush() should yield any buffered content."""
        list(formatter.process_chunk("Hello **world**"))
        result = list(formatter.flush())
        assert len(result) == 1
        assert f"{BOLD_ON}world{BOLD_OFF}" in result[0]

    def test_multi_chunk_line(self, formatter):
        """Multiple chunks forming a complete line."""
        result1 = list(formatter.process_chunk("Hello "))
        assert result1 == []

        result2 = list(formatter.process_chunk("**world**\n"))
        assert len(result2) == 1
        assert f"{BOLD_ON}world{BOLD_OFF}" in result2[0]

    def test_multiple_lines_in_one_chunk(self, formatter):
        """Multiple lines in a single chunk."""
        result = list(formatter.process_chunk("line **one**\nline *two*\n"))
        assert len(result) == 2
        assert f"{BOLD_ON}one{BOLD_OFF}" in result[0]
        assert f"{ITALIC_ON}two{ITALIC_OFF}" in result[1]

    def test_ansi_chunk_flushes_buffer_first(self, formatter):
        """An ANSI chunk should flush buffered plain text first."""
        r1 = list(formatter.process_chunk("buffered text"))
        assert r1 == []  # Buffered

        ansi = "\x1b[36mcode block\x1b[0m"
        r2 = list(formatter.process_chunk(ansi))
        # Should get: flushed buffer + ANSI passthrough
        assert len(r2) == 2
        assert r2[0] == "buffered text"
        assert r2[1] == ansi

    def test_reset_clears_buffer(self, formatter):
        """reset() should clear the buffer."""
        list(formatter.process_chunk("buffered text"))
        formatter.reset()
        result = list(formatter.flush())
        assert result == []

    def test_inline_code_split_across_chunks(self, formatter):
        """Inline code split across chunks (from code_block_formatter)."""
        # Simulates code_block_formatter yielding "`code" then "` rest\n"
        r1 = list(formatter.process_chunk("`code"))
        assert r1 == []  # Buffered

        r2 = list(formatter.process_chunk("` rest\n"))
        assert len(r2) == 1
        code_start = _hex_to_ansi_fg(DEFAULT_INLINE_CODE_FG) + _hex_to_ansi_bg(DEFAULT_INLINE_CODE_BG)
        assert f"{code_start}code{RESET}" in r2[0]


# ==================== Style Configuration Tests ====================


class TestStyleConfiguration:
    """Tests for style configuration methods."""

    def test_custom_inline_code_colors(self, formatter):
        formatter.set_inline_code_style("#ff0000", "#000000")
        result = _format(formatter, "`test`")
        assert _hex_to_ansi_fg("#ff0000") in result
        assert _hex_to_ansi_bg("#000000") in result

    def test_inline_code_no_background(self, formatter):
        formatter.set_inline_code_style("#ff0000")
        result = _format(formatter, "`test`")
        assert _hex_to_ansi_fg("#ff0000") in result
        # Should not contain any background code
        assert "\x1b[48;" not in result

    def test_custom_link_color(self, formatter):
        formatter.set_link_style("#00ff00")
        result = _format(formatter, "[link](url)")
        assert _hex_to_ansi_fg("#00ff00") in result

    def test_initialize_with_config(self):
        formatter = create_plugin()
        formatter.initialize({
            "inline_code_fg": "#aabbcc",
            "inline_code_bg": "#112233",
            "link_fg": "#445566",
        })
        result = _format(formatter, "`code` and [link](url)")
        assert _hex_to_ansi_fg("#aabbcc") in result
        assert _hex_to_ansi_fg("#445566") in result


# ==================== Protocol Tests ====================


class TestProtocol:
    """Tests for FormatterPlugin protocol compliance."""

    def test_name(self, formatter):
        assert formatter.name == "inline_markdown_formatter"

    def test_priority(self, formatter):
        assert formatter.priority == 45

    def test_reset_preserves_styles(self, formatter):
        formatter.set_inline_code_style("#ff0000", "#000000")
        formatter.reset()
        result = _format(formatter, "`test`")
        assert _hex_to_ansi_fg("#ff0000") in result

    def test_factory_function(self):
        plugin = create_plugin()
        assert isinstance(plugin, InlineMarkdownFormatterPlugin)


# ==================== Helper ====================


def _format(formatter, text: str) -> str:
    """Convenience: process text through formatter in batch mode."""
    parts = []
    for chunk in formatter.process_chunk(text):
        parts.append(chunk)
    for chunk in formatter.flush():
        parts.append(chunk)
    formatter.reset()
    return ''.join(parts)
