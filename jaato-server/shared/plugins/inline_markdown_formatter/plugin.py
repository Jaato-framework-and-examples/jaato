# shared/plugins/inline_markdown_formatter/plugin.py
"""Streaming inline markdown formatter plugin for ANSI-styled terminal output.

This plugin transforms inline markdown elements in model output text into
ANSI-styled text. It handles:

- ``code`` → configurable foreground/background color (theme-aware)
- **bold** → ANSI bold attribute
- *italic* → ANSI italic attribute
- ***bold italic*** → ANSI bold + italic attributes
- ~~strikethrough~~ → ANSI strikethrough attribute
- [text](url) → underlined text with configurable color (URL stripped)

The plugin runs at priority 45, after code_block_formatter (40), so code
blocks are already extracted and rendered. Chunks containing ANSI escape
codes from upstream formatters (diff, table, code block) are passed through
unchanged — only plain text is processed for inline markdown.

Underscore-based emphasis (_italic_, __bold__) is intentionally not supported
to avoid false positives with code identifiers, file paths, and URLs.

Streaming strategy: text is buffered line-by-line. Complete lines (terminated
by newline) are formatted and yielded immediately. Partial lines are held
until the next newline arrives or flush() is called. This handles the case
where the upstream code_block_formatter splits inline backtick sequences
across chunk boundaries.

Usage:
    from shared.plugins.inline_markdown_formatter import create_plugin

    formatter = create_plugin()
    formatter.set_inline_code_style("#87d7d7", "#2d2d3d")

    # Streaming mode
    for chunk in model_output:
        for output in formatter.process_chunk(chunk):
            print(output, end='')
    for output in formatter.flush():
        print(output, end='')
"""

import re
from typing import Any, Dict, Iterator, Optional

from shared.trace import trace as _trace_write


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    _trace_write("INLINE_MD_FORMATTER", msg)


def _hex_to_ansi_fg(hex_color: str) -> str:
    """Convert hex color to ANSI 24-bit foreground escape code.

    Args:
        hex_color: Hex color string like "#87d7d7".

    Returns:
        ANSI escape code string.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"\x1b[38;2;{r};{g};{b}m"


def _hex_to_ansi_bg(hex_color: str) -> str:
    """Convert hex color to ANSI 24-bit background escape code.

    Args:
        hex_color: Hex color string like "#2d2d3d".

    Returns:
        ANSI escape code string.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"\x1b[48;2;{r};{g};{b}m"


# ANSI attribute codes
BOLD_ON = "\x1b[1m"
BOLD_OFF = "\x1b[22m"
ITALIC_ON = "\x1b[3m"
ITALIC_OFF = "\x1b[23m"
UNDERLINE_ON = "\x1b[4m"
UNDERLINE_OFF = "\x1b[24m"
STRIKETHROUGH_ON = "\x1b[9m"
STRIKETHROUGH_OFF = "\x1b[29m"
RESET = "\x1b[0m"

# Default inline code colors (teal on slightly blue-tinted dark surface)
DEFAULT_INLINE_CODE_FG = "#87d7d7"
DEFAULT_INLINE_CODE_BG = "#2d2d3d"
DEFAULT_LINK_FG = "#5f87ff"

# Priority: after code_block_formatter (40), before post-processing (60+)
DEFAULT_PRIORITY = 45

# Combined regex for inline markdown elements.
# Order matters: longer/more specific patterns are tried first to prevent
# partial matches (e.g., *** before ** before *).
#
# Each pattern ensures no whitespace immediately inside the markers to
# prevent false positives like "2 * 3 * 4" being treated as italic.
#
# The negative lookahead/lookbehind for * in the italic pattern prevents
# it from matching inside ** or *** sequences.
INLINE_MD_PATTERN = re.compile(
    r'(?P<code>`[^`\n]+`)'                                           # `code`
    r'|(?P<bold_italic>\*\*\*(?!\s)(?:(?!\*\*\*).)+?(?<!\s)\*\*\*)'  # ***bold italic***
    r'|(?P<bold>\*\*(?!\s)(?:(?!\*\*).)+?(?<!\s)\*\*)'               # **bold**
    r'|(?P<italic>(?<!\*)\*(?!\*|\s)(?:(?!\*).)+?(?<!\s)\*(?!\*))'    # *italic*
    r'|(?P<strike>~~(?!\s)(?:(?!~~).)+?(?<!\s)~~)'                   # ~~strikethrough~~
    r'|(?P<link>\[(?P<link_text>[^\]\n]+)\]\((?P<link_url>[^)\n]+)\))'  # [text](url)
)

# Quick check: if a line contains none of these characters, skip regex
_MARKER_CHARS = frozenset('`*~[')


class InlineMarkdownFormatterPlugin:
    """Streaming plugin that formats inline markdown elements with ANSI styling.

    Implements the FormatterPlugin protocol. Buffers text line-by-line and
    applies inline markdown formatting to complete lines. Chunks containing
    ANSI escape codes (from upstream formatters like code_block, diff, table)
    are passed through unchanged.

    Inline elements handled:
    - ``code`` → configurable foreground/background color
    - **bold** → ANSI bold attribute
    - *italic* → ANSI italic attribute
    - ***bold italic*** → ANSI bold + italic
    - ~~strikethrough~~ → ANSI strikethrough attribute
    - [text](url) → underlined text (URL stripped)

    Style lifecycle:
    - Default colors are applied at construction.
    - ``set_inline_code_style()`` / ``set_link_style()`` update colors
      at any time (typically called from pt_display after theme changes).
    - ``reset()`` clears the streaming buffer but preserves styles.
    """

    def __init__(self):
        self._priority = DEFAULT_PRIORITY
        self._buffer = ""

        # Style ANSI codes (configurable via set_inline_code_style / set_link_style)
        self._code_start = (
            _hex_to_ansi_fg(DEFAULT_INLINE_CODE_FG)
            + _hex_to_ansi_bg(DEFAULT_INLINE_CODE_BG)
        )
        self._code_end = RESET
        self._link_start = _hex_to_ansi_fg(DEFAULT_LINK_FG) + UNDERLINE_ON
        self._link_end = RESET

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "inline_markdown_formatter"

    @property
    def priority(self) -> int:
        """Execution priority (45 = after code blocks, before post-processing)."""
        return self._priority

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk, buffering text line-by-line for inline formatting.

        Chunks containing ANSI escape codes are passed through unchanged
        (they come from upstream formatters like diff, table, code block).
        Plain text is buffered until a complete line (newline) is available,
        then formatted and yielded.

        Args:
            chunk: Incoming text chunk.

        Yields:
            Output chunks with inline markdown formatted.
        """
        # Skip chunks that already contain ANSI codes (from upstream formatters)
        if '\x1b[' in chunk:
            # First, flush any buffered plain text
            if self._buffer:
                yield from self._flush_buffer()
            yield chunk
            return

        self._buffer += chunk

        # Process complete lines (text ending with \n)
        while '\n' in self._buffer:
            newline_idx = self._buffer.index('\n')
            line = self._buffer[:newline_idx]
            self._buffer = self._buffer[newline_idx + 1:]
            formatted = self._format_line(line)
            yield formatted + '\n'

    def flush(self) -> Iterator[str]:
        """Flush any remaining buffered content.

        Called at turn end. Processes any remaining text in the buffer
        (which may be an incomplete line without a trailing newline).

        Yields:
            Remaining content with inline markdown formatted.
        """
        if self._buffer:
            yield from self._flush_buffer()

    def reset(self) -> None:
        """Reset streaming state for a new turn. Styles are preserved."""
        self._buffer = ""

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - priority: Pipeline priority (default: 45)
                - inline_code_fg: Hex color for inline code foreground
                - inline_code_bg: Hex color for inline code background
                - link_fg: Hex color for link text
        """
        config = config or {}
        self._priority = config.get("priority", DEFAULT_PRIORITY)

        inline_code_fg = config.get("inline_code_fg")
        inline_code_bg = config.get("inline_code_bg")
        if inline_code_fg:
            self.set_inline_code_style(inline_code_fg, inline_code_bg)

        link_fg = config.get("link_fg")
        if link_fg:
            self.set_link_style(link_fg)

    def set_console_width(self, width: int) -> None:
        """Update console width. Not used by this formatter (inline elements
        don't depend on terminal width), but required by ConfigurableFormatter."""
        pass

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        self.reset()

    # ==================== Style Configuration ====================

    def set_inline_code_style(self, fg_hex: str, bg_hex: Optional[str] = None) -> None:
        """Configure the ANSI style for inline code elements.

        Called by pt_display.py when the theme is loaded or changed.

        Args:
            fg_hex: Foreground hex color (e.g., "#87d7d7").
            bg_hex: Background hex color (e.g., "#2d2d3d"), or None for no background.
        """
        self._code_start = _hex_to_ansi_fg(fg_hex)
        if bg_hex:
            self._code_start += _hex_to_ansi_bg(bg_hex)
        self._code_end = RESET

    def set_link_style(self, fg_hex: str) -> None:
        """Configure the ANSI style for link text.

        Called by pt_display.py when the theme is loaded or changed.

        Args:
            fg_hex: Foreground hex color (e.g., "#5f87ff").
        """
        self._link_start = _hex_to_ansi_fg(fg_hex) + UNDERLINE_ON
        self._link_end = RESET

    # ==================== Internal Methods ====================

    def _flush_buffer(self) -> Iterator[str]:
        """Process and yield all buffered content.

        Handles both complete lines (with newlines) and a trailing partial line.
        """
        if not self._buffer:
            return

        # Process any complete lines still in the buffer
        while '\n' in self._buffer:
            newline_idx = self._buffer.index('\n')
            line = self._buffer[:newline_idx]
            self._buffer = self._buffer[newline_idx + 1:]
            yield self._format_line(line) + '\n'

        # Process remaining partial line
        if self._buffer:
            yield self._format_line(self._buffer)
            self._buffer = ""

    def _format_line(self, line: str) -> str:
        """Apply inline markdown formatting to a single line.

        Args:
            line: Text line (without trailing newline).

        Returns:
            Line with inline markdown elements replaced by ANSI-styled text.
            Returns the line unchanged if no markdown markers are present.
        """
        if not line:
            return line

        # Quick check: skip regex if no markdown marker characters present
        if not _MARKER_CHARS.intersection(line):
            return line

        return self._apply_inline_patterns(line)

    def _apply_inline_patterns(self, text: str) -> str:
        """Apply all inline markdown patterns to text using regex matching.

        Iterates through regex matches left to right, emitting plain text
        between matches and ANSI-styled text for matched patterns. Matches
        are non-overlapping (guaranteed by re.finditer).

        Args:
            text: Text to process.

        Returns:
            Text with inline markdown replaced by ANSI-styled equivalents.
        """
        result_parts = []
        last_end = 0

        for match in INLINE_MD_PATTERN.finditer(text):
            # Emit plain text before this match
            if match.start() > last_end:
                result_parts.append(text[last_end:match.start()])

            # Format the matched element
            if match.group('code'):
                # Strip backticks, apply inline code style
                content = match.group('code')[1:-1]
                result_parts.append(
                    f"{self._code_start}{content}{self._code_end}"
                )
            elif match.group('bold_italic'):
                # Strip ***, apply bold + italic
                content = match.group('bold_italic')[3:-3]
                result_parts.append(
                    f"{BOLD_ON}{ITALIC_ON}{content}{ITALIC_OFF}{BOLD_OFF}"
                )
            elif match.group('bold'):
                # Strip **, apply bold
                content = match.group('bold')[2:-2]
                result_parts.append(f"{BOLD_ON}{content}{BOLD_OFF}")
            elif match.group('italic'):
                # Strip *, apply italic
                content = match.group('italic')[1:-1]
                result_parts.append(f"{ITALIC_ON}{content}{ITALIC_OFF}")
            elif match.group('strike'):
                # Strip ~~, apply strikethrough
                content = match.group('strike')[2:-2]
                result_parts.append(
                    f"{STRIKETHROUGH_ON}{content}{STRIKETHROUGH_OFF}"
                )
            elif match.group('link'):
                # Show link text with link styling, strip URL
                link_text = match.group('link_text')
                result_parts.append(
                    f"{self._link_start}{link_text}{self._link_end}"
                )

            last_end = match.end()

        # Emit remaining plain text after last match
        if last_end < len(text):
            result_parts.append(text[last_end:])

        return ''.join(result_parts)


def create_plugin() -> InlineMarkdownFormatterPlugin:
    """Factory function to create an InlineMarkdownFormatterPlugin instance."""
    return InlineMarkdownFormatterPlugin()
