"""Tests for semantic <j-code> markup emitted for non-terminal clients.

Covers the `set_client_type("web")` path that emits structured markup
instead of ANSI-escaped Rich output.  Terminal default behaviour is
covered by the existing tests in this directory and must remain
unchanged — see `test_default_stays_ansi`.
"""

from shared.plugins.code_block_formatter.plugin import CodeBlockFormatterPlugin


class TestDefaultClientType:

    def test_default_stays_ansi(self):
        """Without set_client_type, terminal ANSI path is used."""
        plugin = CodeBlockFormatterPlugin()
        out = plugin._render_code_block("def foo():\n    pass", "python")
        # ANSI escape sequences present, semantic tags absent.
        assert "\x1b[" in out
        assert "<j-code" not in out


class TestWebSemanticMarkup:

    def _render(self, code: str, lang: str) -> str:
        plugin = CodeBlockFormatterPlugin()
        plugin.set_client_type("web")
        return plugin._render_code_block(code, lang)

    def test_no_ansi_in_web_output(self):
        out = self._render("def foo():\n    pass", "python")
        assert "\x1b[" not in out

    def test_opening_tag_carries_language(self):
        out = self._render("x = 1", "python")
        assert out.startswith('<j-code language="python">\n')
        assert out.rstrip().endswith("</j-code>")

    def test_empty_language_emits_bare_tag(self):
        out = self._render("some text", "")
        assert out.startswith("<j-code>\n")

    def test_line_count_matches_source(self):
        code = "a = 1\nb = 2\nc = 3"
        out = self._render(code, "python")
        assert out.count("<j-line") == 3

    def test_single_line_no_trailing_newline(self):
        out = self._render("x = 1", "python")
        assert out.count("<j-line") == 1

    def test_pygments_short_names_used(self):
        """Keywords must carry t="k", integers t="mi", etc. — the
        canonical Pygments short names from STANDARD_TYPES."""
        out = self._render("def foo():\n    x = 42", "python")
        assert '<j-tok t="k">def</j-tok>' in out
        assert '<j-tok t="mi">42</j-tok>' in out

    def test_unknown_language_falls_back_without_raising(self):
        """Unknown language tokenises as plain text via TextLexer; the
        raw language string is still carried on the opening tag."""
        out = self._render("some content\nmore content", "totally-fake-xyz-999")
        assert '<j-code language="totally-fake-xyz-999">' in out
        assert "<j-line>" in out
        # No crash, no <j-tok> since TextLexer yields only Text tokens
        # which map to empty short name.

    def test_html_escaping(self):
        """<, >, & inside code must be HTML-escaped so they don't
        collide with the semantic tag namespace."""
        out = self._render('x = "<script>&"', "python")
        assert "&lt;script&gt;" in out
        assert "&amp;" in out
        # Ensure raw < > & from code aren't emitted outside escaping
        # (the legitimate ones are inside <j-code>, <j-line>, <j-tok>).
        # Strip all semantic tags and confirm no raw < or > remain.
        import re
        stripped = re.sub(r"</?j-[a-z]+[^>]*>", "", out)
        assert "<" not in stripped
        assert ">" not in stripped

    def test_line_numbers_attribute_only_when_enabled(self):
        """Default (line_numbers=False): no n attribute.
        Enabled: each <j-line> carries n starting at 1."""
        plugin = CodeBlockFormatterPlugin()
        plugin.set_client_type("web")
        plugin._line_numbers = True
        out = plugin._render_code_block("x = 1\ny = 2", "python")
        assert '<j-line n="1">' in out
        assert '<j-line n="2">' in out

    def test_no_inline_styles_or_colours(self):
        """The server emits no colour/style information — only structure."""
        out = self._render("def foo(): return 1", "python")
        assert "style=" not in out
        assert "color:" not in out
        assert "background" not in out

    def test_full_pipeline_streaming(self):
        """Verify the complete streaming path: fenced block in, web
        markup out, no ANSI leaks."""
        plugin = CodeBlockFormatterPlugin()
        plugin.set_client_type("web")
        chunks = list(plugin.process_chunk("Before\n```python\ndef f(): pass\n```\nAfter\n"))
        chunks.extend(plugin.flush())
        output = "".join(chunks)
        assert "<j-code" in output
        assert "\x1b[" not in output
        assert "Before" in output
        assert "After" in output


class TestTokenGrouping:

    def test_whitespace_not_emitted_as_trailing(self):
        """Trailing whitespace runs must be stripped per line."""
        plugin = CodeBlockFormatterPlugin()
        plugin.set_client_type("web")
        out = plugin._render_code_block("x = 1   \ny = 2", "python")
        # The three trailing spaces after "1" should not appear as a
        # bare space run at end of line.  Check that the line ends with
        # the integer token, not with spaces.
        lines = [l for l in out.split("\n") if "<j-line" in l]
        assert lines[0].rstrip().endswith("</j-line>")
        # No trailing whitespace just before </j-line>
        assert not lines[0].endswith(" </j-line>")
