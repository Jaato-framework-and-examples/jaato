"""Client-side renderer for semantic ``<j-*>`` markup.

The server's formatter pipeline emits neutral, client-agnostic markup
for content that different clients would present differently:

- ``<j-code language="...">`` wraps each fenced code block, with one
  ``<j-line>`` per source line and ``<j-tok t="…">`` around each
  Pygments-classified token run.  The ``t`` attribute carries the
  Pygments short name from ``pygments.token.STANDARD_TYPES``.
- ``<j-table>`` wraps each markdown table, with ``<j-thead>``,
  ``<j-tr>``, ``<j-th>``, ``<j-td>`` inside.

This module converts both tag families into ANSI strings for display in
the TUI.  Code blocks are re-rendered through ``rich.syntax.Syntax``
using the UI's active Pygments theme; tables are re-rendered through
``rich.table.Table`` with box-drawing characters.

Why client-side rendering?  Multiple heterogeneous clients (TUI, web
dashboard, chat bridge) can co-attach to a single session.  If the
server pre-rendered output to ANSI, the last-attached client would win
and everyone would see the wrong format.  Keeping the wire format
neutral and rendering client-side avoids that shared-state fight.
"""

import re
from typing import Match, Optional

from rich import box
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table


_J_CODE_BLOCK = re.compile(
    r'<j-code(?:\s+language="([^"]*)")?>\n?(.*?)</j-code>\n?',
    re.DOTALL,
)

_J_LINE = re.compile(
    r'<j-line(?:\s+n="\d+")?>(.*?)</j-line>',
    re.DOTALL,
)

_J_TOK_OPEN = re.compile(r'<j-tok[^>]*>')
_J_TOK_CLOSE = re.compile(r'</j-tok>')

_J_TABLE_BLOCK = re.compile(
    r'<j-table>\n?(.*?)</j-table>\n?',
    re.DOTALL,
)

_J_THEAD = re.compile(r'<j-thead>\n?(.*?)</j-thead>', re.DOTALL)
_J_TH = re.compile(r'<j-th>(.*?)</j-th>', re.DOTALL)
_J_TR = re.compile(r'<j-tr>(.*?)</j-tr>', re.DOTALL)
_J_TD = re.compile(r'<j-td>(.*?)</j-td>', re.DOTALL)


# Language aliases the server used to resolve on the terminal path.
# Now that aliasing happens here, the client owns them.  Additions are
# safe — Pygments falls back to ``TextLexer`` on unknown names.
_LANGUAGE_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "yml": "yaml",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "dockerfile": "docker",
    "md": "markdown",
    "cs": "csharp",
    "c++": "cpp",
    "objective-c": "objectivec",
    "ipython3": "ipython",
    "jupyter": "ipython",
    "cbl": "cobol",
    "cob": "cobol",
}


# Map UI theme names → Pygments syntax themes.  Matches the old
# server-side ``CodeBlockFormatterPlugin.set_syntax_theme`` mapping so
# visual output stays consistent after the client-side move.
_UI_THEME_TO_SYNTAX = {
    "dark": "monokai",
    "light": "solarized-light",
    "high-contrast": "native",
}


def ui_theme_to_syntax_theme(ui_theme_name: str) -> str:
    """Map a UI theme name to a Pygments syntax-highlighting theme.

    Unknown names fall back to ``"monokai"`` so the renderer always
    produces something readable.
    """
    return _UI_THEME_TO_SYNTAX.get(ui_theme_name, "monokai")


def _unescape_html(text: str) -> str:
    """Reverse the minimal HTML escaping the server applies.

    The server only escapes the three characters that collide with its
    own tag namespace, so this is the inverse set.
    """
    return (
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
    )


def _extract_raw_code(body: str) -> str:
    """Strip ``<j-line>`` and ``<j-tok>`` wrappers from a ``<j-code>`` body.

    Returns the raw source lines, HTML-unescaped, joined with ``\\n``.
    Handles malformed bodies defensively (missing ``<j-line>`` wrappers
    just get their tags stripped and text returned as-is).
    """
    lines = []
    last_end = 0
    for match in _J_LINE.finditer(body):
        if match.start() > last_end:
            gap = body[last_end:match.start()].strip()
            if gap:
                lines.append(gap)

        inner = match.group(1)
        inner = _J_TOK_OPEN.sub("", inner)
        inner = _J_TOK_CLOSE.sub("", inner)
        lines.append(_unescape_html(inner))
        last_end = match.end()

    if not lines:
        plain = _J_TOK_OPEN.sub("", body)
        plain = _J_TOK_CLOSE.sub("", plain)
        return _unescape_html(plain.strip("\n"))

    return "\n".join(lines)


def _render_code_to_ansi(code: str, language: str, syntax_theme: str) -> str:
    """Render raw source as an ANSI-highlighted, 4-space-indented block.

    The 4-space indent matches the notebook formatter's column layout so
    ``_render_single_notebook_row`` in ``output_buffer.py`` keeps
    alignment right when it strips the leading indent.
    """
    lang = _LANGUAGE_ALIASES.get(
        (language or "").lower(), (language or "text").lower()
    )

    try:
        syntax = Syntax(
            code,
            lang,
            theme=syntax_theme,
            line_numbers=False,
            word_wrap=False,
            background_color="default",
        )

        code_lines = code.split("\n")
        max_width = max((len(line) for line in code_lines), default=0)
        render_width = max(40, max_width + 2)

        console = Console(
            width=render_width,
            force_terminal=True,
            no_color=False,
            highlight=False,
        )
        with console.capture() as capture:
            console.print(syntax, end="")
        rendered = capture.get()

        indent = "    "
        indented = "\n".join(indent + line for line in rendered.split("\n"))
        return "\n" + indented + "\x1b[0m\n"
    except Exception:
        indent = "    "
        return "\n" + "\n".join(indent + line for line in code.split("\n")) + "\n"


def _render_table_to_ansi(body: str) -> str:
    """Render a ``<j-table>`` body as a box-drawing ASCII table."""
    headers = []
    thead_match = _J_THEAD.search(body)
    if thead_match:
        headers = [
            _unescape_html(m.group(1).strip())
            for m in _J_TH.finditer(thead_match.group(1))
        ]

    rows = []
    body_without_thead = _J_THEAD.sub("", body) if thead_match else body
    for tr in _J_TR.finditer(body_without_thead):
        cells = [
            _unescape_html(m.group(1).strip())
            for m in _J_TD.finditer(tr.group(1))
        ]
        if cells:
            rows.append(cells)

    if not headers and not rows:
        return body

    table = Table(box=box.SQUARE, show_header=bool(headers), header_style="bold")
    if headers:
        for header in headers:
            table.add_column(header)
    else:
        col_count = max((len(r) for r in rows), default=1)
        for _ in range(col_count):
            table.add_column("")

    for row in rows:
        padded = row + [""] * (len(table.columns) - len(row))
        table.add_row(*padded[: len(table.columns)])

    try:
        console = Console(force_terminal=True, no_color=False, highlight=False)
        with console.capture() as capture:
            console.print(table, end="")
        return "\n" + capture.get() + "\n"
    except Exception:
        lines = []
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-+-".join("-" * len(h) for h in headers))
        for row in rows:
            lines.append(" | ".join(row))
        return "\n" + "\n".join(lines) + "\n"


def contains_j_markup(text: str) -> bool:
    """Quick check for any supported ``<j-*>`` opening tag in ``text``.

    Cheap: a plain substring search.  Call this before the full rewrite
    to avoid running regex passes on every chunk of output.
    """
    return "<j-code" in text or "<j-table>" in text


def rewrite_j_markup(text: str, syntax_theme: str = "monokai") -> str:
    """Replace every ``<j-code>`` / ``<j-table>`` block with ANSI output.

    Leaves all other text — including ``<nb-row>`` notebook markers,
    ``<security-warning>`` blocks, and plain model output — untouched.
    Safe to call on arbitrary streaming chunks: a no-op when neither
    opening tag is present.

    Args:
        text: Arbitrary text that may embed semantic markup blocks.
        syntax_theme: Pygments theme name for code highlighting.
            Callers thread this through from the active UI theme
            (see :func:`ui_theme_to_syntax_theme`).

    Returns:
        ``text`` with each matched block replaced by its ANSI rendering.
    """
    if not contains_j_markup(text):
        return text

    def replace_code(match: Match[str]) -> str:
        language = match.group(1) or ""
        code = _extract_raw_code(match.group(2))
        return _render_code_to_ansi(code, language, syntax_theme)

    def replace_table(match: Match[str]) -> str:
        return _render_table_to_ansi(match.group(1))

    result = _J_CODE_BLOCK.sub(replace_code, text)
    result = _J_TABLE_BLOCK.sub(replace_table, result)
    return result
