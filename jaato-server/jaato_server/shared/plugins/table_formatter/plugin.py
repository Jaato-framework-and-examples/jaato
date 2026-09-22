# shared/plugins/table_formatter/plugin.py
"""Table formatter plugin emitting semantic ``<j-table>`` markup.

Detects markdown tables in streaming model output and converts them to
client-agnostic ``<j-table>`` / ``<j-thead>`` / ``<j-tr>`` / ``<j-td>``
markup.  Each attached client (TUI, web dashboard, chat bridge) renders
that markup natively — the server never emits terminal box-drawing or
ANSI.  This keeps the wire format neutral when heterogeneous clients
co-attach to a single session.

Detection patterns:
1. Markdown tables: ``| Header | Header |`` with ``|---|---|`` separator
2. ASCII grid tables (``+---+---+``): passed through unchanged — they
   already carry their own borders.

Usage (pipeline):
    from jaato_server.shared.plugins.formatter_pipeline import create_pipeline
    from jaato_server.shared.plugins.table_formatter import create_plugin

    pipeline = create_pipeline()
    pipeline.register(create_plugin())  # priority 25
"""

import os
import re
import unicodedata
from typing import Any, Dict, Iterator, List, Optional, Tuple

import wcwidth

from jaato_server.shared.plugins.code_block_formatter.plugin import FENCE_CLOSE_RE, FENCE_OPEN_RE


def _get_ambiguous_width() -> int:
    """Get the width to use for East Asian Ambiguous characters.

    Reads from JAATO_AMBIGUOUS_WIDTH environment variable.
    Default is 1 (standard Western terminals).
    Set to 2 for CJK terminals or terminals with ambiguous width = wide.

    Returns:
        1 or 2 depending on configuration.
    """
    try:
        value = os.environ.get("JAATO_AMBIGUOUS_WIDTH", "1")  # env: width of East Asian Ambiguous chars in tables (1 default; 2 for CJK terminals)
        return 2 if value == "2" else 1
    except (ValueError, TypeError):
        return 1

# Priority for pipeline ordering (20-39 = structural formatting)
DEFAULT_PRIORITY = 25

# Patterns for table detection
# Markdown table: | cell | cell | with at least one |---|
MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
MARKDOWN_SEPARATOR = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")

# ASCII grid table: +---+---+ style
ASCII_GRID_BORDER = re.compile(r"^\s*\+[-+]+\+\s*$")
ASCII_GRID_ROW = re.compile(r"^\s*\|.*\|\s*$")


def _display_width(text: str) -> int:
    """Calculate the display width of a string, accounting for wide characters.

    Uses unicodedata.east_asian_width() to properly handle:
    - Fullwidth (F) and Wide (W) characters: 2 columns
    - Ambiguous (A) characters: configurable via JAATO_AMBIGUOUS_WIDTH env var
    - Halfwidth (H), Narrow (Na), Neutral (N): 1 column
    - Zero-width characters (via wcwidth): 0 columns

    This is more accurate than wcwidth alone because it respects the
    terminal's ambiguous width setting for box-drawing characters,
    which are East Asian Ambiguous and may render as 1 or 2 columns
    depending on the terminal configuration.

    Args:
        text: The string to measure.

    Returns:
        The display width in terminal columns.
    """
    ambiguous_width = _get_ambiguous_width()
    width = 0
    for char in text:
        # First check for zero-width characters via wcwidth
        wc = wcwidth.wcwidth(char)
        if wc == 0:
            continue
        if wc == -1:
            # Non-printable, treat as 0
            continue

        # Use East Asian Width for proper handling
        eaw = unicodedata.east_asian_width(char)
        if eaw in ('F', 'W'):
            # Fullwidth and Wide are always 2 columns
            width += 2
        elif eaw == 'A':
            # Ambiguous - depends on terminal settings
            width += ambiguous_width
        else:
            # Halfwidth (H), Narrow (Na), Neutral (N) are 1 column
            width += 1
    return width


class TableFormatterPlugin:
    """Plugin that converts markdown tables into semantic ``<j-table>`` markup.

    Implements the FormatterPlugin protocol for use in a formatter
    pipeline.  Detects markdown tables in streaming input and emits
    client-agnostic semantic tags that attached clients render natively
    (TUI to box-drawing, web dashboards to HTML ``<table>``, chat
    bridges to a card/list layout, …).

    Features:
    - Detects markdown tables (``| col | col |`` with ``|---|---|``)
    - Passes ASCII grid tables (``+---+---+``) through unchanged
    - Handles multi-line streaming input
    - Leaves the inside of a fenced code block alone

    **Fences.**  This formatter runs at priority 25, BEFORE
    ``code_block_formatter`` (40), so it sees fences in their raw form.
    A table inside one is code the author quoted -- a ````markdown``
    example, a notebook cell's fenced stdout -- and rewriting it into
    ``<j-table>`` hands ``code_block_formatter`` a fence full of markup,
    which it escapes into literal, line-numbered ``&lt;j-table&gt;``
    lines that every client then renders faithfully (#1191).  Fence state
    is decided by the SAME two patterns that formatter opens and closes
    on, imported rather than restated, so the two cannot disagree about
    which lines are code.

    Fence state is tracked on COMPLETE lines, and a line can arrive in
    pieces: a head with no ``|`` is passed straight through for streaming
    latency before its newline arrives.  ``_fence_line_prefix`` remembers
    that already-yielded head so the completed line is judged whole --
    otherwise a ````py`` split from its newline would never open a fence.

    State: ``_in_fence`` flips on an opener line outside a fence and on a
    closer line inside one; ``flush()`` and ``reset()`` both clear it,
    because ``code_block_formatter``'s own ``flush()`` closes an
    unterminated block and the two must agree after a flush too.
    """

    def __init__(self):
        self._priority = DEFAULT_PRIORITY
        self._console_width = 120

        # Buffer for accumulating table lines
        self._buffer: List[str] = []
        self._in_table = False
        self._table_type: Optional[str] = None  # "markdown" or "ascii_grid"

        # Buffer for incomplete lines (no trailing newline yet)
        self._line_buffer: str = ""

        # Fence tracking (#1191): inside a fence nothing is rewritten.
        self._in_fence = False
        # The head of the current line already yielded before its newline
        # arrived, so the completed line can be judged whole.
        self._fence_line_prefix: str = ""

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "table_formatter"

    @property
    def priority(self) -> int:
        """Execution priority (25 = structural formatting, after diff)."""
        return self._priority

    def _looks_like_table_content(self, text: str) -> bool:
        """Check if text could potentially be part of a table.

        Used during streaming to determine whether to buffer partial lines.
        Only buffers when content has table-like characteristics to avoid
        blocking regular text output during streaming.

        Args:
            text: Text to check (may be incomplete line).

        Returns:
            True if text contains table-like patterns.
        """
        # Check for pipe character (markdown table cells)
        if "|" in text:
            return True
        # Check for ASCII grid table border start
        if text.lstrip().startswith("+"):
            return True
        return False

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk, buffering table lines for complete rendering.

        Lines that appear to be part of a table are buffered until
        the table is complete (detected by a non-table line or flush).

        Handles partial lines from streaming by buffering until a
        complete line (ending with newline) is received - but ONLY
        when the content looks like potential table content. Regular
        text is passed through immediately to support streaming output.

        Args:
            chunk: Incoming text chunk.

        Yields:
            Formatted output when appropriate.
        """
        # Prepend any buffered partial line
        text = self._line_buffer + chunk
        self._line_buffer = ""

        # Track non-table incomplete content to yield after complete lines
        trailing_non_table: Optional[str] = None

        # Check if the last part is incomplete (no trailing newline)
        if text and not text.endswith("\n"):
            # Find the last newline
            last_newline = text.rfind("\n")
            if last_newline == -1:
                # No complete lines yet
                # Only buffer if it looks like potential table content
                if self._may_hold_table(text):
                    self._line_buffer = text
                    return
                else:
                    # Pass through non-table content immediately for streaming
                    self._fence_line_prefix += text
                    yield text
                    return
            else:
                # We have some complete lines and an incomplete part
                incomplete_part = text[last_newline + 1:]
                text = text[:last_newline + 1]
                # Only buffer incomplete part if it looks like table content
                if self._may_hold_table(incomplete_part):
                    self._line_buffer = incomplete_part
                else:
                    # Yield after processing complete lines
                    trailing_non_table = incomplete_part

        # Process complete lines
        lines = text.split("\n")
        for i, line in enumerate(lines):
            is_last_line = i == len(lines) - 1

            # Skip empty string from trailing newline
            if is_last_line and line == "":
                continue

            whole_line = self._fence_line_prefix + line
            self._fence_line_prefix = ""
            if self._fence_transition(whole_line):
                # A fence boundary, or a line inside a fence: code the
                # author quoted, passed through exactly as written.  Any
                # table buffered before an opener is complete -- flush it
                # (``_flush_buffer`` is a no-op on an empty buffer).
                yield from self._flush_buffer()
                yield line + "\n"
                continue

            table_line_type = self._classify_line(line)

            if table_line_type:
                # Start or continue buffering table content
                if not self._in_table:
                    self._in_table = True
                    self._table_type = table_line_type
                self._buffer.append(line)
            else:
                # Non-table line - flush any buffered table first
                yield from self._flush_buffer()

                # Pass through non-table content with newline
                yield line + "\n"

        # Yield any non-table incomplete content that wasn't buffered
        if trailing_non_table:
            self._fence_line_prefix = trailing_non_table
            yield trailing_non_table

    def _may_hold_table(self, text: str) -> bool:
        """Should this incomplete line be held back as possible table content?

        Never inside a fence: nothing there is rewritten, so holding it
        back would only cost streaming latency.
        """
        if self._in_fence:
            return False
        return self._in_table or self._looks_like_table_content(text)

    def _fence_transition(self, line: str) -> bool:
        """Update fence state for one COMPLETE line; True if it is fence text.

        Returns True for the opener, every line inside the fence, and the
        closer -- the lines this formatter must pass through unchanged.
        The patterns are ``code_block_formatter``'s own (see the class
        docstring): the opener is searched with the line's newline
        restored, because that is how that formatter sees it.
        """
        if self._in_fence:
            if FENCE_CLOSE_RE.search(line):
                self._in_fence = False
            return True
        if FENCE_OPEN_RE.search(line + "\n"):
            self._in_fence = True
            return True
        return False

    def _classify_line(self, line: str) -> Optional[str]:
        """Classify a line as table content or not.

        Returns:
            "markdown" if markdown table line
            "ascii_grid" if ASCII grid table line
            None if not a table line
        """
        # Check for markdown table patterns
        if MARKDOWN_TABLE_ROW.match(line):
            return "markdown"
        if MARKDOWN_SEPARATOR.match(line):
            return "markdown"

        # Check for ASCII grid table patterns
        if ASCII_GRID_BORDER.match(line):
            return "ascii_grid"
        if self._in_table and self._table_type == "ascii_grid":
            if ASCII_GRID_ROW.match(line):
                return "ascii_grid"

        return None

    def _flush_buffer(self) -> Iterator[str]:
        """Flush the table buffer and yield formatted output."""
        if not self._buffer:
            return

        table_text = "\n".join(self._buffer)
        self._buffer = []
        table_type = self._table_type
        self._in_table = False
        self._table_type = None

        # Check if this is actually a valid table
        if table_type == "markdown" and self._is_valid_markdown_table(table_text):
            yield self._render_semantic_table(table_text)
        elif table_type == "ascii_grid":
            yield self._render_ascii_grid_table(table_text)
        else:
            # Not a valid table, pass through as-is
            yield table_text + "\n"

    def _is_valid_markdown_table(self, text: str) -> bool:
        """Check if text is a valid markdown table (has separator row)."""
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return False

        # Must have at least one separator line
        for line in lines:
            if MARKDOWN_SEPARATOR.match(line):
                return True
        return False

    def flush(self) -> Iterator[str]:
        """Flush any remaining buffered content.

        Leaves fence state CLEARED: ``code_block_formatter.flush()``
        renders an unterminated block and forgets it, so after a flush the
        two formatters must again agree that no fence is open.
        """
        self._in_fence = False
        self._fence_line_prefix = ""
        # First, handle any incomplete line in the line buffer
        if self._line_buffer:
            # Try to classify it as a table line
            table_line_type = self._classify_line(self._line_buffer)
            if table_line_type:
                if not self._in_table:
                    self._in_table = True
                    self._table_type = table_line_type
                self._buffer.append(self._line_buffer)
            else:
                # Flush table buffer first, then output the incomplete line
                for output in self._flush_buffer():
                    yield output
                yield self._line_buffer
            self._line_buffer = ""

        # Flush any remaining table content
        for output in self._flush_buffer():
            yield output

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer = []
        self._in_table = False
        self._table_type = None
        self._line_buffer = ""
        self._in_fence = False
        self._fence_line_prefix = ""

    # ==================== Table Parsing ====================

    def _parse_markdown_table(self, text: str) -> Tuple[List[str], List[List[str]], List[str]]:
        """Parse a markdown table into headers, rows, and alignments.

        Returns:
            (headers, rows, alignments) where alignments is list of 'left', 'center', 'right'
        """
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return [], [], []

        # Find the separator line
        separator_idx = -1
        for i, line in enumerate(lines):
            if MARKDOWN_SEPARATOR.match(line):
                separator_idx = i
                break

        if separator_idx == -1:
            return [], [], []

        # Parse header (line before separator)
        header_line = lines[separator_idx - 1] if separator_idx > 0 else ""
        headers = self._parse_row(header_line)

        # Parse alignments from separator
        alignments = self._parse_alignments(lines[separator_idx])

        # Parse data rows (lines after separator)
        rows = []
        for line in lines[separator_idx + 1 :]:
            if MARKDOWN_TABLE_ROW.match(line):
                rows.append(self._parse_row(line))

        return headers, rows, alignments

    def _parse_row(self, line: str) -> List[str]:
        """Parse a markdown table row into cells."""
        # Remove leading/trailing pipes and split
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]

        cells = [cell.strip() for cell in line.split("|")]
        return cells

    def _parse_alignments(self, separator: str) -> List[str]:
        """Parse alignment from separator row (e.g., |:---|:---:|---:|)."""
        cells = self._parse_row(separator)
        alignments = []

        for cell in cells:
            cell = cell.strip()
            if cell.startswith(":") and cell.endswith(":"):
                alignments.append("center")
            elif cell.endswith(":"):
                alignments.append("right")
            else:
                alignments.append("left")

        return alignments

    # ==================== Table Rendering ====================

    def _render_semantic_table(self, text: str) -> str:
        """Render a markdown table as semantic ``<j-table>`` markup.

        Emits format-independent tags that clients render natively:

        - Web clients → HTML ``<table>``
        - Chat clients → card/list layout
        - API clients → structured JSON

        The ``j-`` prefix identifies jaato pipeline semantic tags.
        """
        headers, rows, alignments = self._parse_markdown_table(text)
        if not headers and not rows:
            return text + "\n"

        lines = ["<j-table>"]

        if headers:
            lines.append("<j-thead>")
            cells = "".join(f"<j-th>{cell}</j-th>" for cell in headers)
            lines.append(cells)
            lines.append("</j-thead>")

        for row in rows:
            cells = "".join(f"<j-td>{cell}</j-td>" for cell in row)
            lines.append(f"<j-tr>{cells}</j-tr>")

        lines.append("</j-table>")
        return "\n".join(lines) + "\n"

    def _render_ascii_grid_table(self, text: str) -> str:
        """Render ASCII grid table (already has borders, just pass through)."""
        # ASCII grid tables already have box characters, just ensure proper ending
        if not text.endswith("\n"):
            return text + "\n"
        return text

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - priority: Pipeline priority (default: 25)
                - console_width: Console width in columns (default: 120).
                  Kept for API compatibility; semantic markup rendering
                  ignores terminal width because clients re-flow content
                  to their own display.
        """
        config = config or {}
        self._priority = config.get("priority", DEFAULT_PRIORITY)
        self._console_width = config.get("console_width", 120)

    def set_console_width(self, width: int) -> None:
        """Update console width for rendering.

        Args:
            width: Terminal width in columns.
        """
        self._console_width = width

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        pass

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



def create_plugin() -> TableFormatterPlugin:
    """Factory function to create a TableFormatterPlugin instance."""
    return TableFormatterPlugin()
