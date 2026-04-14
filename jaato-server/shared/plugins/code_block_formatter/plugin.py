# shared/plugins/code_block_formatter/plugin.py
"""Streaming code block formatter plugin for syntax highlighting.

This plugin transforms model output text containing markdown code blocks
into ANSI-escaped text with syntax highlighting. It buffers content inside
code blocks until they're complete, while passing through regular text
immediately.

Usage:
    from shared.plugins.code_block_formatter import create_plugin

    formatter = create_plugin()
    formatter.initialize({"theme": "monokai", "line_numbers": True})

    # Streaming mode
    for chunk in model_output:
        for output in formatter.process_chunk(chunk):
            print(output, end='')
    for output in formatter.flush():
        print(output, end='')
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from shared.plugins.table_formatter.plugin import _display_width
from shared.trace import trace as _trace_write


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    _trace_write("CODE_BLOCK_FORMATTER", msg)


# Common language aliases mapping
LANGUAGE_ALIASES = {
    'js': 'javascript',
    'ts': 'typescript',
    'py': 'python',
    'rb': 'ruby',
    'yml': 'yaml',
    'sh': 'bash',
    'shell': 'bash',
    'zsh': 'bash',
    'dockerfile': 'docker',
    'md': 'markdown',
    'cs': 'csharp',
    'c++': 'cpp',
    'objective-c': 'objectivec',
    'ipython3': 'ipython',  # IPython with !shell, %magic support
    'jupyter': 'ipython',  # Jupyter notebooks use IPython kernel
    'cbl': 'cobol',  # COBOL source files
    'cob': 'cobol',  # COBOL source files
}

# Priority for pipeline ordering (40-59 = syntax highlighting)
DEFAULT_PRIORITY = 40


def _get_content_width(line: str) -> int:
    """Get the display width of actual content in a line (excluding trailing whitespace).

    Args:
        line: The line (may contain ANSI codes).

    Returns:
        Display width of content (excluding trailing spaces).
    """
    if not line:
        return 0

    # Parse ANSI codes to get plain text
    if '\x1b[' in line:
        text = Text.from_ansi(line)
        plain = text.plain
    else:
        plain = line

    # Strip trailing spaces and measure content width
    content = plain.rstrip()
    if not content:
        return 0

    return _display_width(content)


def _trim_line_to_width(line: str, target_width: int, truncation_indicator: str = "") -> str:
    """Trim a line to target_width, preserving ANSI codes and styling.

    This removes trailing background padding while keeping the actual content
    and its styling intact. If truncation_indicator is provided and the line's
    content exceeds target_width, the indicator is appended after truncation.

    Args:
        line: The line to trim (may contain ANSI codes).
        target_width: The width to trim to.
        truncation_indicator: Character(s) to append when content is truncated
            (e.g. "▸"). Empty string means no indicator.

    Returns:
        The line trimmed to target_width, as an ANSI string.
    """
    if not line:
        return ""

    # Parse ANSI codes to get styled text
    if '\x1b[' in line:
        text = Text.from_ansi(line)
    else:
        text = Text(line)

    plain = text.plain

    if target_width <= 0:
        return ""

    indicator_width = _display_width(truncation_indicator) if truncation_indicator else 0
    trim_target = target_width - indicator_width if truncation_indicator else target_width

    # Find the character index that corresponds to trim_target display columns
    current_width = 0
    char_count = 0
    for char in plain:
        char_width = _display_width(char)
        if current_width + char_width > trim_target:
            break
        current_width += char_width
        char_count += 1

    # Slice the styled text to preserve styling
    if char_count > 0:
        result = text[:char_count]
    else:
        result = Text()

    if truncation_indicator:
        result.append(truncation_indicator, style="dim")

    # Convert back to ANSI string
    # Use a large width to prevent any wrapping during conversion
    console = Console(width=10000, force_terminal=True, no_color=False, highlight=False)
    with console.capture() as capture:
        console.print(result, end="")
    output = capture.get()

    # Always append ANSI reset code to ensure styling doesn't bleed across lines
    # when the text is later split by newlines and stored/rendered separately
    if output and '\x1b[' in output and not output.endswith('\x1b[0m'):
        output += '\x1b[0m'
    return output


class CodeBlockFormatterPlugin:
    """Streaming plugin that formats code blocks with syntax highlighting.

    Implements the FormatterPlugin protocol. Buffers content inside code
    blocks (```...```) until complete, passes through other text immediately.
    """

    def __init__(self):
        self._theme = "monokai"
        self._line_numbers = False
        self._word_wrap = True
        self._background_color: Optional[str] = None
        self._console_width = 80
        self._priority = DEFAULT_PRIORITY

        # Rendering surface.  "terminal" → Rich/ANSI with theme colours;
        # any other value ("web", "chat", …) → semantic <j-code> markup
        # that the client can style however it wants.  See
        # shared/plugins/table_formatter for the mirror implementation
        # that introduced this pattern.
        self._client_type: str = "terminal"

        # Whether to apply width-based line trimming with the ▸ indicator.
        # Defaults to True (no trimming) — clients (TUI, dashboard) handle
        # line width on their own, and the server-side ▸ marker ends up
        # double-clipping with misleading "more content" indicators.
        # The presentation context's client_type can override this if a
        # client explicitly opts in.
        self._disable_truncation = True

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
                - theme: Syntax highlighting theme (default: "monokai")
                - line_numbers: Show line numbers (default: False)
                - word_wrap: Wrap long lines (default: True)
                - background_color: Background color or None (default: None)
                - console_width: Width for rendering (default: 80)
                - priority: Pipeline priority (default: 40)
        """
        config = config or {}
        self._theme = config.get("theme", "monokai")
        self._line_numbers = config.get("line_numbers", False)
        self._word_wrap = config.get("word_wrap", True)
        self._background_color = config.get("background_color", None)
        self._console_width = config.get("console_width", 80)
        self._priority = config.get("priority", DEFAULT_PRIORITY)

    def set_console_width(self, width: int) -> None:
        """Update the console width for rendering."""
        self._console_width = max(20, width)

    def set_client_type(self, client_type: str) -> None:
        """Set the client rendering surface.

        ``"terminal"`` keeps the Rich/ANSI output (unchanged from
        pre-client-awareness behaviour).  Any other value switches to
        semantic ``<j-code>`` markup — no ANSI, no colours, no indent —
        so the client owns the rendering.

        The ``t`` attribute on ``<j-tok>`` elements tracks
        ``pygments.token.STANDARD_TYPES`` short names (e.g. ``k`` for
        keyword, ``mi`` for integer literal).  If Pygments renames any
        of these in a future version, the client CSS must follow.

        Args:
            client_type: ``"terminal"``, ``"web"``, ``"chat"``, or ``"api"``.
        """
        self._client_type = client_type

    def set_disable_truncation(self, disabled: bool) -> None:
        """Enable or disable width-based line truncation.

        When disabled, code blocks are rendered without per-line trimming
        and without the ``▸`` indicator.  Used for non-terminal clients
        (browser dashboards) that re-flow content and don't need fixed-
        width line trimming.

        Args:
            disabled: True to skip truncation, False (default) to apply
                ``console_width``-based trimming.
        """
        self._disable_truncation = disabled

    def set_syntax_theme(self, theme_name: str) -> None:
        """Set the syntax highlighting theme based on UI theme.

        Maps UI theme names to appropriate Pygments syntax themes.

        Args:
            theme_name: UI theme name ("dark", "light", "high-contrast", etc.)
        """
        # Map UI themes to Pygments syntax themes
        theme_mapping = {
            "dark": "monokai",
            "light": "solarized-light",  # Light background, good contrast
            "high-contrast": "native",  # High contrast dark theme
        }
        self._theme = theme_mapping.get(theme_name, "monokai")

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        self.reset()

    # ==================== Internal Methods ====================

    def _render_code_block(self, code: str, language: str) -> str:
        """Render a code block, routed by ``client_type``.

        Terminal clients get Rich/ANSI with syntax highlighting and a
        block indent — unchanged from the original implementation.  Any
        other client gets semantic ``<j-code>`` markup (see
        ``_render_semantic_code_block``).

        Args:
            code: The code content (without ``` markers).
            language: The language identifier from the fenced block.

        Returns:
            ANSI-escaped string for terminal, or semantic markup for web.
        """
        if self._client_type != "terminal":
            return self._render_semantic_code_block(code, language)

        # Map language aliases (terminal path only — web keeps the raw
        # string so the client can normalise it).
        lang = LANGUAGE_ALIASES.get(language.lower(), language.lower())

        # Block indent for visual distinction
        indent = "    "  # 4 spaces

        try:
            # Calculate natural width from raw code to prevent wrapping
            code_lines = code.split('\n')
            max_code_width = max((_display_width(line) for line in code_lines), default=0)

            # Account for line numbers if enabled (Rich format: " NUM │ ")
            if self._line_numbers and code_lines:
                num_lines = len(code_lines)
                line_number_width = len(str(num_lines)) + 4  # padding + separator
                max_code_width += line_number_width

            # Use content width as console width (prevents wrapping)
            # Add buffer of 2 to account for any padding/formatting Rich may add
            # (previous buffer of 1 was insufficient in some cases)
            render_width = max(40, max_code_width + 2)

            # Disable word_wrap during rendering since we've calculated render_width
            # to be large enough for all content. This prevents any wrapping due to
            # width calculation discrepancies between our calculation and Rich's.
            # The self._word_wrap setting is preserved for potential future use.
            syntax = Syntax(
                code,
                lang,
                theme=self._theme,
                line_numbers=self._line_numbers,
                word_wrap=False,
                background_color=self._background_color,
            )

            # Render to ANSI string using a temporary console
            console = Console(
                width=render_width,
                force_terminal=True,
                no_color=False,
                highlight=False,
            )
            with console.capture() as capture:
                console.print(syntax, end="")

            rendered = capture.get()
            lines = rendered.split('\n')

            # Non-terminal clients (browser dashboards) re-flow content,
            # so emit raw rendered lines with indent only — no width-based
            # trimming, no ▸ indicator.
            if self._disable_truncation:
                indented_lines = [indent + line for line in lines]
                return '\n' + '\n'.join(indented_lines) + '\x1b[0m\n'

            # Find the natural width of content (max content width across all lines)
            # This ensures background styling only extends to the widest content line
            natural_width = max((_get_content_width(line) for line in lines), default=0)

            # Calculate maximum allowed width based on terminal width minus indent
            indent_width = _display_width(indent)
            max_allowed_width = self._console_width - indent_width

            # Cap natural width to terminal width so background padding
            # never extends beyond the visible area
            trim_width = min(natural_width, max_allowed_width) if max_allowed_width > 0 else natural_width

            # Trim each line to the capped width, then add indent
            trimmed_lines = []
            for line in lines:
                content_width = _get_content_width(line)
                if max_allowed_width > 0 and content_width > max_allowed_width:
                    # Line content exceeds terminal width: truncate with indicator
                    trimmed = _trim_line_to_width(line, max_allowed_width, truncation_indicator="▸")
                else:
                    # Line fits: trim trailing background padding to capped width
                    trimmed = _trim_line_to_width(line, trim_width)
                trimmed_lines.append(indent + trimmed)

            # Ensure ANSI reset at end to prevent styling from bleeding into subsequent text
            return '\n' + '\n'.join(trimmed_lines) + '\x1b[0m\n'

        except Exception:
            # Fallback: return code as-is with indent if highlighting fails
            indented_lines = [indent + line for line in code.split('\n')]
            return '\n' + '\n'.join(indented_lines) + '\n'

    def _render_semantic_code_block(self, code: str, language: str) -> str:
        """Render a code block as semantic ``<j-code>`` markup.

        Emits one ``<j-line>`` per source line and wraps Pygments-classified
        token runs in ``<j-tok t="...">``.  The ``t`` attribute carries the
        Pygments short token name from
        :data:`pygments.token.STANDARD_TYPES` (walking up the token
        hierarchy when the specific subtype has no entry).  Whitespace and
        unclassified text are emitted as bare text inside ``<j-line>``.

        The server emits no colours or inline styles; the client owns
        presentation.  Trailing whitespace is stripped per line.

        Args:
            code: The code content (without ``` markers).
            language: The raw language string from the fenced block.  Not
                passed through ``LANGUAGE_ALIASES`` — the client decides
                how to normalise.

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
