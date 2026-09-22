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
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from shared.trace import trace as _trace_write


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    _trace_write("CODE_BLOCK_FORMATTER", msg)


# Priority for pipeline ordering (40-59 = syntax highlighting)
DEFAULT_PRIORITY = 40

# What opens and what closes a fenced code block.  Named, because this
# formatter is not the only one that must agree on it: every formatter
# running BEFORE this one (priority < 40) sees the raw fence and must
# leave its contents alone, or it rewrites text that this formatter then
# escapes and shows as code.  ``table_formatter`` did exactly that --
# a markdown table inside a fence became ``<j-table>`` markup and reached
# every client as literal, line-numbered tags (#1191).  It imports
# :func:`open_fence` rather than restating the rule, so the two
# formatters cannot disagree about where a fence starts and ends.
#
# The rule is CommonMark's, and the previous one was not -- which is how
# a model's own prose came to be drawn as code.  The old opener was
# ``\`\`\`(\w*)\n`` and the old closer any line STARTING with three
# backticks, so:
#
#   * an opener the pattern rejected -- ``\`\`\`text `` with a trailing
#     space, ``\`\`\`c++``, ``\`\`\`shell-session``, ``\`\`\`\`markdown`` --
#     was passed through as text, and its CLOSER then opened a block
#     that ran to the end of the reply;
#   * a four-backtick fence quoting a three-backtick one closed on the
#     inner fence, and the parity of every fence after it flipped;
#   * a closer indented inside a list item was never seen at all.
#
# Now an opener at a line start takes any info string (the language is
# its first word), a closer is a line holding only a run of the SAME
# character at least as long as the opener's, and both may be indented.
# The old mid-line opener (``text \`\`\`py``) is kept, with its old
# narrow info string, because models write it and CommonMark's refusal
# would change output nobody reported.
_LINE_START_OPEN = re.compile(r'^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$')
_MID_LINE_OPEN = re.compile(r'(?P<fence>`{3,})(?P<info>[\w+#.-]*)[ \t]*$')
_CLOSE = re.compile(r'^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*$')


@dataclass(frozen=True)
class Fence:
    """An open fenced code block: how it was opened, so it knows its close.

    Attributes:
        char: The fence character, a backtick or a tilde.
        length: How many of it the opener used; a closer needs at least
            as many, which is what lets a four-backtick fence quote a
            three-backtick one.
        indent: Columns of indentation on the opener.  That much is
            removed from each content line, as CommonMark does, so a
            fence inside a list item renders without the list's indent.
        lang: The first word of the info string, ``"text"`` when empty.
    """

    char: str
    length: int
    indent: int
    lang: str

    def closes(self, line: str) -> bool:
        """True when ``line`` (no newline) is this fence's closer."""
        m = _CLOSE.match(line.rstrip("\r"))
        if not m:
            return False
        run = m.group("fence")
        return run[0] == self.char and len(run) >= self.length


def open_fence(line: str, *, at_line_start: bool = True) -> Optional[Tuple[int, Fence]]:
    """Does ``line`` (a complete line, no newline) open a fenced block?

    Args:
        line: The line's text.
        at_line_start: Whether ``line`` really begins a line in the
            stream.  ``False`` when the head of the line has already been
            emitted, so only the mid-line form can apply.

    Returns:
        ``(offset, fence)`` -- where in ``line`` the fence begins, and the
        fence -- or ``None``.
    """
    line = line.rstrip("\r")
    if at_line_start:
        m = _LINE_START_OPEN.match(line)
        # A backtick fence's info string may not contain a backtick:
        # "```x``` is inline code" is not an opener.
        if m and not (m.group("fence")[0] == "`" and "`" in m.group("info")):
            info = m.group("info").strip()
            fence = Fence(char=m.group("fence")[0], length=len(m.group("fence")),
                          indent=len(m.group("indent")), lang=info.split()[0] if info else "text")
            return m.start("fence"), fence
    m = _MID_LINE_OPEN.search(line)
    if m and (m.start() == 0 or line[m.start() - 1] != "`"):
        return m.start(), Fence(char="`", length=len(m.group("fence")), indent=0,
                                lang=m.group("info") or "text")
    return None


def _hold_from(partial: str, at_line_start: bool) -> Optional[int]:
    r"""Where to cut an INCOMPLETE line so a fence opener is not split.

    Text is streamed as it arrives; only the part that could still turn
    out to be an opener waits for its newline.  At a line start that is
    whitespace (the indent of a fence to come) or a leading fence run;
    anywhere else, a trailing run of backticks and a bare word -- the
    mid-line form, and the same hold the old ``\`{1,3}\w*$`` made.
    """
    if at_line_start and re.fullmatch(r'[ \t]*(`{1,2}|`{3,}[^`]*|~{1,2}|~{3,}.*)?', partial):
        return 0
    m = re.search(r'`+[\w+#.-]*[ \t]*$', partial)
    return m.start() if m else None


class CodeBlockFormatterPlugin:
    """Streaming plugin that formats code blocks with syntax highlighting.

    Implements the FormatterPlugin protocol. Buffers content inside code
    blocks until they close, passes through other text immediately.

    State, and what moves it:

    * ``_in_code_block`` / ``_fence`` -- set by an opener line
      (:func:`open_fence`), cleared by that fence's own closer
      (:meth:`Fence.closes`) or by ``flush`` / ``reset``.
    * ``_buffer`` -- outside a block, only the incomplete line that could
      still become an opener (:func:`_hold_from`); inside one, everything
      since the opener.
    * ``_at_line_start`` -- whether ``_buffer`` begins a line in the
      stream.  Text already emitted is gone from the buffer, so without
      it the rest of a line would be judged as if it began one.
    * ``_code_scan_pos`` -- where the closer search resumes, so a long
      block is not rescanned from its start on every chunk.
    """

    def __init__(self):
        self._line_numbers = False
        self._priority = DEFAULT_PRIORITY

        # Streaming state
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""
        self._fence: Optional[Fence] = None
        self._at_line_start = True
        self._code_scan_pos = 0

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
            step = self._scan_code() if self._in_code_block else self._scan_text()
            progressed = yield from step
            if not progressed:
                return

    def _emit(self, text: str) -> Iterator[str]:
        """Yield text outside a block, tracking whether a line is open."""
        if text:
            self._at_line_start = text.endswith("\n")
            yield text

    def _scan_text(self) -> Iterator[str]:
        """Outside a block: emit text up to an opener, or up to what must wait.

        Returns (as the generator's value) True when a block was entered,
        so the caller keeps going; False when the buffer needs more input.
        """
        buf, pos, at_start = self._buffer, 0, self._at_line_start
        while True:
            nl = buf.find("\n", pos)
            if nl == -1:
                hold = _hold_from(buf[pos:], at_start)
                cut = len(buf) if hold is None else pos + hold
                yield from self._emit(buf[:cut])
                self._buffer = buf[cut:]
                return False
            found = open_fence(buf[pos:nl], at_line_start=at_start)
            if found:
                offset, fence = found
                yield from self._emit(buf[:pos + offset])
                self._fence = fence
                self._in_code_block = True
                self._code_block_lang = fence.lang
                self._buffer = buf[nl + 1:]
                self._code_scan_pos = 0
                return True
            pos, at_start = nl + 1, True

    def _scan_code(self) -> Iterator[str]:
        """Inside a block: render it once its closer line is complete.

        The closer is consumed up to its own newline, which stays in the
        buffer and is emitted as text -- the spacing the formatter has
        always produced after a block.
        """
        buf, pos = self._buffer, self._code_scan_pos
        while True:
            nl = buf.find("\n", pos)
            if nl == -1:
                self._code_scan_pos = pos
                return False
            if self._fence.closes(buf[pos:nl]):
                yield self._render_open_block(buf[:pos - 1] if pos else "")
                self._buffer = buf[nl:]
                self._at_line_start = True
                return True
            pos = nl + 1

    def _render_open_block(self, code: str) -> str:
        """Render the open block's content and leave block mode."""
        indent = self._fence.indent if self._fence else 0
        if indent:
            code = "\n".join(
                line[min(indent, len(line) - len(line.lstrip(" "))):]
                for line in code.split("\n")
            )
        formatted = self._render_code_block(code, self._code_block_lang)
        self._in_code_block = False
        self._code_block_lang = ""
        self._fence = None
        self._code_scan_pos = 0
        return formatted

    def flush(self) -> Iterator[str]:
        """Flush any remaining buffered content.

        A block still open is rendered with what it has -- a reply that
        forgets its closer loses nothing -- and a closer that arrived with
        no newline after it is recognised here rather than drawn as code.

        Yields:
            Any remaining content, formatted if it was a code block.
        """
        if self._buffer or self._in_code_block:
            if self._in_code_block:
                code = self._buffer
                last_nl = code.rfind("\n")
                if self._fence.closes(code[last_nl + 1:]):
                    code = code[:last_nl] if last_nl >= 0 else ""
                yield self._render_open_block(code)
            else:
                yield self._buffer
        self._reset_stream()

    def _reset_stream(self) -> None:
        """Back to the state of a stream that has not started."""
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""
        self._fence = None
        self._at_line_start = True
        self._code_scan_pos = 0

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._reset_stream()

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
