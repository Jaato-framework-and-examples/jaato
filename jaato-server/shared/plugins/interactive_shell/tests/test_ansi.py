"""Tests for ANSI escape sequence stripping."""

import pytest

from shared.plugins.interactive_shell.ansi import strip_ansi


class TestStripAnsi:
    """Test ANSI escape code removal."""

    def test_plain_text_unchanged(self):
        assert strip_ansi("hello world") == "hello world"

    def test_empty_string(self):
        assert strip_ansi("") == ""

    def test_strip_color_codes(self):
        # Red text: ESC[31m ... ESC[0m
        colored = "\x1b[31mError\x1b[0m: something failed"
        assert strip_ansi(colored) == "Error: something failed"

    def test_strip_bold_and_colors(self):
        text = "\x1b[1;32mSuccess\x1b[0m"
        assert strip_ansi(text) == "Success"

    def test_strip_256_color(self):
        text = "\x1b[38;5;196mRed text\x1b[0m"
        assert strip_ansi(text) == "Red text"

    def test_strip_24bit_color(self):
        text = "\x1b[38;2;255;0;0mRed\x1b[0m"
        assert strip_ansi(text) == "Red"

    def test_strip_cursor_movement(self):
        # Cursor up 3 lines: ESC[3A
        text = "line1\x1b[3Aline2"
        assert strip_ansi(text) == "line1line2"

    def test_strip_erase_line(self):
        # Erase to end of line: ESC[K
        text = "partial\x1b[Ktext"
        assert strip_ansi(text) == "partialtext"

    def test_strip_osc_title(self):
        # Window title: ESC]0;title BEL
        text = "\x1b]0;My Terminal\x07prompt$ "
        assert strip_ansi(text) == "prompt$ "

    def test_strip_osc_with_st(self):
        # OSC terminated with ST (ESC \)
        text = "\x1b]0;title\x1b\\prompt$ "
        assert strip_ansi(text) == "prompt$ "

    def test_strip_carriage_return(self):
        # Progress bar overwriting
        text = "Progress: 50%\rProgress: 100%"
        assert strip_ansi(text) == "Progress: 50%Progress: 100%"

    def test_strip_backspace_overwriting(self):
        # Bold via overprinting: X backspace X
        text = "H\x08He\x08el\x08ll\x08lo\x08o"
        assert strip_ansi(text) == "Hello"

    def test_strip_character_set_designation(self):
        # ESC(B — set character set to ASCII
        text = "\x1b(Bhello"
        assert strip_ansi(text) == "hello"

    def test_strip_simple_escape(self):
        # ESC M — reverse index
        text = "\x1bMhello"
        assert strip_ansi(text) == "hello"

    def test_multiline_with_colors(self):
        text = (
            "\x1b[1mheader\x1b[0m\n"
            "\x1b[32m  item 1\x1b[0m\n"
            "\x1b[33m  item 2\x1b[0m\n"
        )
        expected = "header\n  item 1\n  item 2\n"
        assert strip_ansi(text) == expected

    def test_prompt_with_escapes(self):
        # Typical bash prompt with colors
        text = "\x1b[01;32muser@host\x1b[00m:\x1b[01;34m~/project\x1b[00m$ "
        assert strip_ansi(text) == "user@host:~/project$ "

    def test_preserves_newlines_and_tabs(self):
        text = "line1\n\tindented\nline3"
        assert strip_ansi(text) == "line1\n\tindented\nline3"

    def test_private_mode_sequences(self):
        # ESC[?25h — show cursor
        text = "\x1b[?25hhello\x1b[?25l"
        assert strip_ansi(text) == "hello"
