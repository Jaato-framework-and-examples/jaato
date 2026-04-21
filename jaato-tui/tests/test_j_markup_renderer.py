"""Tests for the client-side ``<j-*>`` semantic markup renderer.

These tests exercise the public contract the TUI relies on:

- ``contains_j_markup`` is a cheap presence check used to gate the
  full rewrite pass on every streaming chunk.
- ``rewrite_j_markup`` replaces every ``<j-code>`` / ``<j-table>``
  block with ANSI output while leaving every other byte of the input
  untouched (including ``<nb-row>`` notebook markers, which are
  rendered by a separate layer in ``output_buffer``).
- ``ui_theme_to_syntax_theme`` maps UI theme names to Pygments themes
  and falls back to a sane default for unknown names.
"""

import os
import re
import sys

# Make the TUI package importable from the tests directory without a
# setup.py install — mirrors tests/test_output_buffer.py's approach.
_rich_client_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _rich_client_dir not in sys.path:
    sys.path.insert(0, _rich_client_dir)

from j_markup_renderer import (  # noqa: E402
    contains_j_markup,
    rewrite_j_markup,
    ui_theme_to_syntax_theme,
)


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape codes so cell/source content can be asserted
    against plain-text expectations.

    Rich's ``Syntax`` emits per-character colour spans, so a single
    source token like ``"x = 1"`` becomes a run of style-wrapped
    characters in the raw output.  The characters themselves remain
    in-order, so stripping ANSI recovers the original text.
    """
    return _ANSI_RE.sub("", text)


class TestContainsJMarkup:

    def test_plain_text(self):
        assert contains_j_markup("just plain text") is False

    def test_nb_row_is_not_j_markup(self):
        """<nb-row> is a separate marker rendered elsewhere."""
        assert contains_j_markup('<nb-row type="input" label="In [1]:">') is False

    def test_detects_j_code(self):
        assert contains_j_markup('<j-code language="python">...</j-code>') is True

    def test_detects_bare_j_code(self):
        """Opening tag without language attribute still counts."""
        assert contains_j_markup("<j-code>...</j-code>") is True

    def test_detects_j_table(self):
        assert contains_j_markup("<j-table>...</j-table>") is True


class TestRewriteJCode:

    def test_noop_on_plain_text(self):
        text = "hello world\nno markup here"
        assert rewrite_j_markup(text) == text

    def test_strips_j_tags_and_unescapes_entities(self):
        markup = (
            '<j-code language="python">\n'
            '<j-line n="1"><j-tok t="k">def</j-tok> foo():</j-line>\n'
            '<j-line n="2">    <j-tok t="k">return</j-tok> <j-tok t="s2">&lt;ok&gt;</j-tok></j-line>\n'
            '</j-code>\n'
        )
        out = rewrite_j_markup(markup)

        # Every j-* tag must be gone from the rendered output.
        assert "<j-code" not in out
        assert "<j-line" not in out
        assert "<j-tok" not in out
        assert "</j-code>" not in out

        # Rich Syntax emits ANSI colour codes — that is the proof it
        # ran through syntax highlighting, not a plain-text dump.
        assert "\x1b[" in out

        # HTML entities must be restored to their source characters
        # once we strip the per-character ANSI colour spans.
        plain = _strip_ansi(out)
        assert "<ok>" in plain
        assert "&lt;" not in plain
        assert "&gt;" not in plain

    def test_preserves_text_around_markup(self):
        text = (
            "before\n"
            '<j-code language="python">\n'
            '<j-line><j-tok t="mi">42</j-tok></j-line>\n'
            '</j-code>\n'
            "after\n"
        )
        out = rewrite_j_markup(text)
        assert "before" in out
        assert "after" in out
        assert "<j-code" not in out

    def test_multiple_blocks_in_one_pass(self):
        text = (
            '<j-code><j-line>a</j-line></j-code>\n'
            "middle\n"
            '<j-code><j-line>b</j-line></j-code>\n'
        )
        out = rewrite_j_markup(text)
        assert "<j-code" not in out
        assert "middle" in out

    def test_falls_back_to_plain_when_no_lines(self):
        """Malformed body (no <j-line>) should not crash the renderer."""
        markup = '<j-code language="python">x = 1</j-code>\n'
        out = rewrite_j_markup(markup)
        assert "<j-code" not in out
        assert "x = 1" in _strip_ansi(out)


class TestRewriteJTable:

    def test_renders_table_with_box_drawing(self):
        markup = (
            "<j-table>\n"
            "<j-thead>\n"
            "<j-th>Name</j-th><j-th>Age</j-th>\n"
            "</j-thead>\n"
            "<j-tr><j-td>Alice</j-td><j-td>30</j-td></j-tr>\n"
            "<j-tr><j-td>Bob</j-td><j-td>25</j-td></j-tr>\n"
            "</j-table>\n"
        )
        out = rewrite_j_markup(markup)

        # Tags are replaced.
        assert "<j-table>" not in out
        assert "<j-th>" not in out
        assert "<j-td>" not in out

        # Cell contents survive.
        for cell in ("Name", "Age", "Alice", "30", "Bob", "25"):
            assert cell in out

    def test_unescapes_entities_in_cells(self):
        markup = (
            "<j-table>\n"
            "<j-thead><j-th>expr</j-th></j-thead>\n"
            "<j-tr><j-td>a &lt; b &amp;&amp; c &gt; d</j-td></j-tr>\n"
            "</j-table>\n"
        )
        out = rewrite_j_markup(markup)
        assert "a < b && c > d" in out
        assert "&lt;" not in out
        assert "&amp;" not in out


class TestUiThemeMapping:

    def test_known_themes_map(self):
        assert ui_theme_to_syntax_theme("dark") == "monokai"
        assert ui_theme_to_syntax_theme("light") == "solarized-light"
        assert ui_theme_to_syntax_theme("high-contrast") == "native"

    def test_unknown_falls_back_to_monokai(self):
        assert ui_theme_to_syntax_theme("some-custom-theme") == "monokai"
