# shared/plugins/table_formatter/plugin.py
"""Table formatter plugin with box-drawing rendering.

Detects markdown tables in model output and renders them with Unicode
box-drawing characters for proper fixed-width display.

Detection patterns:
1. Markdown tables: | Header | Header |  with separator line |---|---|
2. ASCII grid tables: +---+---+ style borders

Usage (pipeline):
    from shared.plugins.formatter_pipeline import create_pipeline
    from shared.plugins.table_formatter import create_plugin

    pipeline = create_pipeline()
    pipeline.register(create_plugin())  # priority 25
"""

import os
import re
import unicodedata
from typing import Any, Dict, Iterator, List, Optional, Tuple

import wcwidth


def _get_ambiguous_width() -> int:
    """Get the width to use for East Asian Ambiguous characters.

    Reads from JAATO_AMBIGUOUS_WIDTH environment variable.
    Default is 1 (standard Western terminals).
    Set to 2 for CJK terminals or terminals with ambiguous width = wide.

    Returns:
        1 or 2 depending on configuration.
    """
    try:
        value = os.environ.get("JAATO_AMBIGUOUS_WIDTH", "1")
        return 2 if value == "2" else 1
    except (ValueError, TypeError):
        return 1

# Priority for pipeline ordering (20-39 = structural formatting)
DEFAULT_PRIORITY = 25

# Box-drawing characters for table rendering
BOX_CHARS = {
    "top_left": "┌",
    "top_right": "┐",
    "bottom_left": "└",
    "bottom_right": "┘",
    "horizontal": "─",
    "vertical": "│",
    "t_down": "┬",
    "t_up": "┴",
    "t_right": "├",
    "t_left": "┤",
    "cross": "┼",
}

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


def _pad_to_width(text: str, target_width: int, align: str = "left") -> str:
    """Pad a string to a target display width, accounting for wide characters.

    Args:
        text: The string to pad.
        target_width: The desired display width.
        align: Alignment - 'left', 'right', or 'center'.

    Returns:
        The padded string.
    """
    current_width = _display_width(text)
    padding_needed = max(0, target_width - current_width)

    if align == "right":
        return " " * padding_needed + text
    elif align == "center":
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return " " * left_pad + text + " " * right_pad
    else:  # left
        return text + " " * padding_needed


class TableFormatterPlugin:
    """Plugin that formats tables with box-drawing characters.

    Implements the FormatterPlugin protocol for use in a formatter pipeline.
    Detects markdown tables and renders them with Unicode box-drawing
    characters for proper fixed-width display.

    Features:
    - Detects markdown tables (| col | col | with |---|---|)
    - Detects ASCII grid tables (+---+---+ style)
    - Renders with Unicode box-drawing characters
    - Preserves column alignment
    - Handles multi-line streaming input
    """

    def __init__(self):
        self._priority = DEFAULT_PRIORITY
        self._console_width = 120
        self._client_type: str = "terminal"  # "terminal" → box-drawing, "web"/"chat" → semantic

        # Buffer for accumulating table lines
        self._buffer: List[str] = []
        self._in_table = False
        self._table_type: Optional[str] = None  # "markdown" or "ascii_grid"

        # Buffer for incomplete lines (no trailing newline yet)
        self._line_buffer: str = ""

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
                if self._in_table or self._looks_like_table_content(text):
                    self._line_buffer = text
                    return
                else:
                    # Pass through non-table content immediately for streaming
                    yield text
                    return
            else:
                # We have some complete lines and an incomplete part
                incomplete_part = text[last_newline + 1:]
                text = text[:last_newline + 1]
                # Only buffer incomplete part if it looks like table content
                if self._in_table or self._looks_like_table_content(incomplete_part):
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

            table_line_type = self._classify_line(line)

            if table_line_type:
                # Start or continue buffering table content
                if not self._in_table:
                    self._in_table = True
                    self._table_type = table_line_type
                self._buffer.append(line)
            else:
                # Non-table line - flush any buffered table first
                if self._buffer:
                    for output in self._flush_buffer():
                        yield output

                # Pass through non-table content with newline
                yield line + "\n"

        # Yield any non-table incomplete content that wasn't buffered
        if trailing_non_table:
            yield trailing_non_table

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
            if self._client_type == "terminal":
                yield self._render_markdown_table(table_text)
            else:
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
        """Flush any remaining buffered content."""
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

    # Minimum width a single column may be shrunk to before we stop
    # reducing further. Keeps at least a few characters visible per cell.
    _MIN_COL_WIDTH = 3

    def _render_markdown_table(self, text: str) -> str:
        """Render a markdown table with box-drawing characters.

        If the natural table width exceeds ``self._console_width``, wider
        columns are shrunk and their content is wrapped across multiple
        visual lines so that the rendered table fits within the terminal.
        """
        headers, rows, alignments = self._parse_markdown_table(text)

        if not headers and not rows:
            return text + "\n"

        # Calculate column widths
        all_rows = [headers] + rows if headers else rows
        num_cols = max(len(row) for row in all_rows) if all_rows else 0

        if num_cols == 0:
            return text + "\n"

        # Normalize row lengths
        for row in all_rows:
            while len(row) < num_cols:
                row.append("")

        # Extend alignments if needed
        while len(alignments) < num_cols:
            alignments.append("left")

        # Calculate max width for each column (natural width)
        col_widths = []
        for col_idx in range(num_cols):
            max_width = 0
            for row in all_rows:
                if col_idx < len(row):
                    max_width = max(max_width, _display_width(row[col_idx]))
            col_widths.append(max(max_width, 1))  # Minimum width of 1

        # Shrink columns so the rendered table fits within console_width.
        col_widths = self._shrink_widths(col_widths, self._console_width)

        # Build the table
        lines = []

        # Top border
        lines.append(self._make_border("top", col_widths))

        # Header row (if present)
        if headers:
            lines.extend(self._make_row_lines(headers, col_widths, alignments))
            lines.append(self._make_border("middle", col_widths))

        # Data rows (each row may expand into multiple visual lines)
        for row in rows:
            lines.extend(self._make_row_lines(row, col_widths, alignments))

        # Bottom border
        lines.append(self._make_border("bottom", col_widths))

        return "\n".join(lines) + "\n"

    def _shrink_widths(self, col_widths: List[int], max_total: int) -> List[int]:
        """Reduce column widths so the whole table fits within ``max_total``.

        Overhead per table line is ``3 * num_cols + 1`` characters
        (left+right borders, one separator between each cell pair, and
        two padding spaces per cell). Remaining budget is divided among
        columns; columns wider than their budget are shrunk — widest
        first — down to ``_MIN_COL_WIDTH``. If even the minimum layout
        exceeds ``max_total`` (very narrow terminal, many columns), the
        minimum layout is returned anyway rather than producing a
        degenerate / zero-width column.

        Args:
            col_widths: Natural column widths (from content measurement).
            max_total: Maximum total table width, in terminal columns.

        Returns:
            Adjusted column widths. Same length as ``col_widths``.
        """
        num_cols = len(col_widths)
        if num_cols == 0:
            return col_widths

        overhead = 3 * num_cols + 1
        available = max(max_total - overhead, num_cols * self._MIN_COL_WIDTH)

        widths = list(col_widths)
        total = sum(widths)
        if total <= available:
            return widths

        # Shrink the widest column by 1 each iteration until the table
        # fits or every column has reached the minimum width.
        while total > available:
            max_w = max(widths)
            if max_w <= self._MIN_COL_WIDTH:
                break
            max_idx = widths.index(max_w)
            widths[max_idx] -= 1
            total -= 1
        return widths

    def _wrap_cell(self, text: str, width: int) -> List[str]:
        """Wrap ``text`` to one or more lines of at most ``width`` columns.

        Prefers breaking on whitespace; falls back to character-level
        breaking for tokens that are themselves wider than ``width``
        (long identifiers, URLs, unbroken runs of backticked tags).

        Args:
            text: Cell content. Markdown syntax is preserved verbatim —
                the formatter runs before inline markdown styling, so
                cells are plain strings without ANSI codes at this stage.
            width: Target display width per wrapped line.

        Returns:
            List of lines, each with display width ``<= width``.
            Always contains at least one entry (possibly empty).
        """
        if width <= 0:
            return [text]
        if _display_width(text) <= width:
            return [text]

        # Tokenize into runs of whitespace vs non-whitespace so we can
        # try whitespace-based wrapping first.
        tokens = re.findall(r"\S+|\s+", text)

        lines: List[str] = []
        current = ""
        current_w = 0

        def _flush_current() -> None:
            nonlocal current, current_w
            if current:
                lines.append(current.rstrip())
                current = ""
                current_w = 0

        for tok in tokens:
            tok_w = _display_width(tok)
            if current_w + tok_w <= width:
                current += tok
                current_w += tok_w
                continue

            # Token doesn't fit on the current line.
            if tok.isspace():
                # Drop whitespace at a wrap boundary.
                _flush_current()
                continue

            _flush_current()

            if tok_w <= width:
                current = tok
                current_w = tok_w
                continue

            # Token alone is wider than the column — break char by char.
            chunk = ""
            chunk_w = 0
            for ch in tok:
                cw = _display_width(ch)
                if cw == 0:
                    chunk += ch
                    continue
                if chunk_w + cw > width:
                    if chunk:
                        lines.append(chunk)
                    chunk = ch
                    chunk_w = cw
                else:
                    chunk += ch
                    chunk_w += cw
            current = chunk
            current_w = chunk_w

        _flush_current()
        return lines or [""]

    def _make_border(self, position: str, col_widths: List[int]) -> str:
        """Create a horizontal border line."""
        if position == "top":
            left = BOX_CHARS["top_left"]
            mid = BOX_CHARS["t_down"]
            right = BOX_CHARS["top_right"]
        elif position == "middle":
            left = BOX_CHARS["t_right"]
            mid = BOX_CHARS["cross"]
            right = BOX_CHARS["t_left"]
        else:  # bottom
            left = BOX_CHARS["bottom_left"]
            mid = BOX_CHARS["t_up"]
            right = BOX_CHARS["bottom_right"]

        horiz = BOX_CHARS["horizontal"]
        segments = [horiz * (w + 2) for w in col_widths]
        return left + mid.join(segments) + right

    def _make_row(self, cells: List[str], col_widths: List[int], alignments: List[str]) -> str:
        """Create a single-line data row with proper alignment.

        Assumes every cell fits within its column width. For rendering
        with wrap-aware, multi-line output, use :meth:`_make_row_lines`.
        """
        vert = BOX_CHARS["vertical"]
        formatted_cells = []

        for i, (cell, width, align) in enumerate(zip(cells, col_widths, alignments)):
            formatted = _pad_to_width(cell, width, align)
            formatted_cells.append(f" {formatted} ")

        return vert + vert.join(formatted_cells) + vert

    def _make_row_lines(
        self,
        cells: List[str],
        col_widths: List[int],
        alignments: List[str],
    ) -> List[str]:
        """Create a data row, wrapping cells that exceed their column width.

        A single logical row expands to as many visual lines as the
        tallest (most wrapped) cell requires. Shorter cells are padded
        with blank lines so all columns remain vertically aligned, and
        every emitted line has box-drawing borders at the expected
        positions.

        Args:
            cells: Cell contents for this row (length == ``len(col_widths)``).
            col_widths: Width allotted to each column.
            alignments: Per-column alignment (``"left"``/``"center"``/``"right"``).

        Returns:
            One or more rendered lines. At least one line is always
            returned (possibly an all-blank row).
        """
        vert = BOX_CHARS["vertical"]

        wrapped_cells = [
            self._wrap_cell(cell, width)
            for cell, width in zip(cells, col_widths)
        ]

        row_height = max((len(lines) for lines in wrapped_cells), default=1)

        # Pad each cell's wrapped lines to the row height so all columns
        # occupy the same number of visual rows.
        for cell_lines in wrapped_cells:
            while len(cell_lines) < row_height:
                cell_lines.append("")

        output_lines: List[str] = []
        for line_idx in range(row_height):
            parts: List[str] = []
            for cell_lines, width, align in zip(wrapped_cells, col_widths, alignments):
                line = cell_lines[line_idx]
                formatted = _pad_to_width(line, width, align)
                parts.append(f" {formatted} ")
            output_lines.append(vert + vert.join(parts) + vert)
        return output_lines

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
                - console_width: Terminal width (default: 120)
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

    def set_client_type(self, client_type: str) -> None:
        """Set the client rendering surface.

        When ``client_type`` is ``"terminal"``, tables are rendered with
        Unicode box-drawing characters.  For ``"web"`` or ``"chat"``,
        tables are emitted as semantic ``<nb-table>`` markup so the
        client can render them natively (e.g. HTML ``<table>``).

        Args:
            client_type: ``"terminal"``, ``"web"``, ``"chat"``, or ``"api"``.
        """
        self._client_type = client_type

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        pass


def create_plugin() -> TableFormatterPlugin:
    """Factory function to create a TableFormatterPlugin instance."""
    return TableFormatterPlugin()
