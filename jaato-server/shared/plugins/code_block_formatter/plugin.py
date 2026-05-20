# shared/plugins/code_block_formatter/plugin.py
"""Streaming code block formatter plugin.

This plugin detects markdown fenced code blocks in streaming text and
converts them into semantic ``<j-code>`` markup that clients render
natively.  The server never emits terminal ANSI or colours for code —
that's the client's job.  This lets a TUI, a web dashboard, and a chat
bridge co-attach to the same session without fighting over a single
shared output format.

Usage:
    from shared.plugins.code_block_formatter import create_plugin

    formatter = create_plugin()
    formatter.initialize({"line_numbers": True})

    for chunk in model_output:
        for output in formatter.process_chunk(chunk):
            print(output, end='')
    for output in formatter.flush():
        print(output, end='')
"""

import re
from typing import Any, Dict, Iterator, List, Optional

from shared.trace import trace as _trace_write


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    _trace_write("CODE_BLOCK_FORMATTER", msg)


# Priority for pipeline ordering (40-59 = syntax highlighting)
DEFAULT_PRIORITY = 40


class CodeBlockFormatterPlugin:
    """Streaming plugin that formats code blocks with syntax highlighting.

    Implements the FormatterPlugin protocol. Buffers content inside code
    blocks (```...```) until complete, passes through other text immediately.
    """

    def __init__(self):
        self._line_numbers = False
        self._priority = DEFAULT_PRIORITY

        # Streaming state
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "code_block_formatter"

    @property
    def priority(self) -> int:
        """Execution priority (40 = syntax highlighting range)."""
        return self._priority

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk, buffering code blocks, passing through text.

        Args:
            chunk: Incoming text chunk.

        Yields:
            Output chunks - immediate for regular text, formatted for complete code blocks.
        """
        self._buffer += chunk

        while self._buffer:
            if not self._in_code_block:
                # Look for code block start: ```lang or ```
                match = re.search(r'```(\w*)\n', self._buffer)
                if match:
                    # Yield text before the code block
                    before = self._buffer[:match.start()]
                    if before:
                        yield before

                    # Enter code block mode
                    self._code_block_lang = match.group(1) or "text"
                    self._buffer = self._buffer[match.end():]
                    self._in_code_block = True
                else:
                    # Check if we might have a partial code block start at the end
                    # This includes: `, ``, ```, ```lang (without trailing \n)
                    partial_match = re.search(r'`{1,3}\w*$', self._buffer)
                    if partial_match:
                        # Hold back the potential code block start
                        to_yield = self._buffer[:partial_match.start()]
                        self._buffer = self._buffer[partial_match.start():]
                        if to_yield:
                            yield to_yield
                        return
                    # No code block start, yield everything
                    yield self._buffer
                    self._buffer = ""
            else:
                # In code block, look for closing ```
                # Match ``` at start of buffer OR preceded by newline
                end_match = re.search(r'(?:^|\n)```', self._buffer)
                if end_match:
                    # Extract code block content
                    code = self._buffer[:end_match.start()]

                    # Format and yield the complete code block
                    formatted = self._render_code_block(code, self._code_block_lang)
                    yield formatted

                    # Exit code block mode, continue with remaining text
                    self._buffer = self._buffer[end_match.end():]
                    self._in_code_block = False
                    self._code_block_lang = ""
                else:
                    # Code block not complete yet, keep buffering
                    return

    def flush(self) -> Iterator[str]:
        """Flush any remaining buffered content.

        Yields:
            Any remaining content, formatted if it was a code block.
        """
        if self._buffer:
            if self._in_code_block:
                # Incomplete code block - format what we have
                formatted = self._render_code_block(self._buffer, self._code_block_lang)
                yield formatted
            else:
                # Regular text
                yield self._buffer
            self._buffer = ""
            self._in_code_block = False
            self._code_block_lang = ""

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - line_numbers: Emit ``n="…"`` on ``<j-line>`` (default: False)
                - priority: Pipeline priority (default: 40)
        """
        config = config or {}
        self._line_numbers = config.get("line_numbers", False)
        self._priority = config.get("priority", DEFAULT_PRIORITY)

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        self.reset()

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset — NO-OP for this plugin.

        Phase 1 hotfix (server 0.6.148+): added to satisfy the
        ``ToolPlugin`` / ``EnrichmentPlugin`` protocol's runtime
        ``isinstance`` check.  Per Daniel's litmus test (see
        ``docs/design/runner-cascade-sharing.md`` §4.3), this
        plugin holds no per-session state that the next cascade
        session would benefit from having cleared.  Override in
        future PRs if the litmus test changes.
        """
        pass


    # ==================== Internal Methods ====================

    def _render_code_block(self, code: str, language: str) -> str:
        """Render a fenced code block as semantic ``<j-code>`` markup.

        Emits one ``<j-line>`` per source line and wraps Pygments-classified
        token runs in ``<j-tok t="...">``.  The ``t`` attribute carries the
        Pygments short token name from
        :data:`pygments.token.STANDARD_TYPES` (walking up the token
        hierarchy when the specific subtype has no entry).  Whitespace and
        unclassified text are emitted as bare text inside ``<j-line>``.

        The server emits no colours or inline styles; every attached
        client (TUI, web dashboard, chat bridge) renders this markup
        natively into its own format.  This keeps the wire format
        neutral when heterogeneous clients co-attach to a session.

        Args:
            code: The code content (without ``` markers).
            language: The raw language string from the fenced block.
                The client decides how to normalise it.

        Returns:
            A newline-terminated ``<j-code>…</j-code>`` block.
        """
        from pygments import lex
        from pygments.lexers import get_lexer_by_name
        from pygments.lexers.special import TextLexer
        from pygments.token import STANDARD_TYPES
        from pygments.util import ClassNotFound

        def escape(s: str) -> str:
            return (
                s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
            )

        def short_name(token_type) -> str:
            """Walk up the token hierarchy to find a STANDARD_TYPES entry.

            Mirrors ``pygments.formatters.html.HtmlFormatter._get_css_classes``
            — Pygments' own HTML formatter uses this same walk-up so
            specific subtypes fall back to their parent's class.
            """
            t = token_type
            while t is not None:
                name = STANDARD_TYPES.get(t)
                if name:
                    return name
                t = t.parent
            return ""

        # Strip a leading/trailing newline that fenced blocks commonly
        # carry — they're part of the fence syntax, not the code.
        stripped = code
        if stripped.startswith("\n"):
            stripped = stripped[1:]
        if stripped.endswith("\n"):
            stripped = stripped[:-1]

        # Preserve the raw language string on the <j-code> element.
        lang_attr = escape(language) if language else ""
        open_tag = f'<j-code language="{lang_attr}">' if lang_attr else "<j-code>"

        # Resolve lexer; fall back to plain text (no tokenisation) if
        # Pygments doesn't recognise the language.
        lexer = None
        if language:
            try:
                lexer = get_lexer_by_name(language)
            except ClassNotFound:
                lexer = None
        if lexer is None:
            lexer = TextLexer()

        # Group tokens into per-line runs.  Each entry in `lines` is a
        # list of (short_class, text) pairs.  Empty short_class means
        # plain text (no <j-tok> wrapper).
        source_lines = stripped.split("\n")
        lines: List[List[tuple]] = [[] for _ in source_lines]

        current_line = 0
        for token_type, value in lex(stripped, lexer):
            if not value:
                continue
            cls = short_name(token_type)
            parts = value.split("\n")
            for i, part in enumerate(parts):
                if part:
                    lines[current_line].append((cls, part))
                if i < len(parts) - 1:
                    current_line += 1
                    if current_line >= len(lines):
                        # Safety: shouldn't exceed, but guard anyway.
                        lines.append([])

        # Pygments' lexers typically append a trailing newline, producing
        # one extra empty line entry.  Drop trailing empty lines that
        # exceed the original source line count.
        expected = len(source_lines)
        while len(lines) > expected and not lines[-1]:
            lines.pop()

        # Assemble the markup.  `n` attribute only when line_numbers enabled.
        out_lines = [open_tag]
        for idx, runs in enumerate(lines):
            # Strip trailing whitespace-only runs to avoid trailing spaces
            # in the rendered line.
            while runs and not runs[-1][1].strip():
                runs.pop()
            children = []
            for cls, text in runs:
                esc = escape(text)
                if cls:
                    children.append(f'<j-tok t="{cls}">{esc}</j-tok>')
                else:
                    children.append(esc)
            body = "".join(children)
            if self._line_numbers:
                out_lines.append(f'<j-line n="{idx + 1}">{body}</j-line>')
            else:
                out_lines.append(f'<j-line>{body}</j-line>')
        out_lines.append("</j-code>")
        return "\n".join(out_lines) + "\n"


def create_plugin() -> CodeBlockFormatterPlugin:
    """Factory function to create a CodeBlockFormatterPlugin instance."""
    return CodeBlockFormatterPlugin()
