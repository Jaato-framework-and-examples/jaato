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

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import wcwidth

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

    Uses wcwidth to properly handle emojis, CJK characters, and other
    characters that take up more than one terminal column.

    Args:
        text: The string to measure.

    Returns:
        The display width in terminal columns.
    """
    width = 0
    for char in text:
        char_width = wcwidth.wcwidth(char)
        # wcwidth returns -1 for non-printable characters, treat as 0
        if char_width >= 0:
            width += char_width
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

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk, buffering table lines for complete rendering.

        Lines that appear to be part of a table are buffered until
        the table is complete (detected by a non-table line or flush).

        Handles partial lines from streaming by buffering until a
        complete line (ending with newline) is received.

        Args:
            chunk: Incoming text chunk.

        Yields:
            Formatted output when appropriate.
        """
        # Prepend any buffered partial line
        text = self._line_buffer + chunk
        self._line_buffer = ""

        # Check if the last part is incomplete (no trailing newline)
        if text and not text.endswith("\n"):
            # Find the last newline
            last_newline = text.rfind("\n")
            if last_newline == -1:
                # No complete lines yet, buffer everything
                self._line_buffer = text
                return
            else:
                # Buffer the incomplete part, process complete lines
                self._line_buffer = text[last_newline + 1:]
                text = text[:last_newline + 1]

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
            yield self._render_markdown_table(table_text)
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

    def _render_markdown_table(self, text: str) -> str:
        """Render a markdown table with box-drawing characters."""
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

        # Calculate max width for each column
        col_widths = []
        for col_idx in range(num_cols):
            max_width = 0
            for row in all_rows:
                if col_idx < len(row):
                    max_width = max(max_width, _display_width(row[col_idx]))
            col_widths.append(max(max_width, 1))  # Minimum width of 1

        # Build the table
        lines = []

        # Top border
        lines.append(self._make_border("top", col_widths))

        # Header row (if present)
        if headers:
            lines.append(self._make_row(headers, col_widths, alignments))
            lines.append(self._make_border("middle", col_widths))

        # Data rows
        for row in rows:
            lines.append(self._make_row(row, col_widths, alignments))

        # Bottom border
        lines.append(self._make_border("bottom", col_widths))

        return "\n".join(lines) + "\n"

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
        """Create a data row with proper alignment."""
        vert = BOX_CHARS["vertical"]
        formatted_cells = []

        for i, (cell, width, align) in enumerate(zip(cells, col_widths, alignments)):
            formatted = _pad_to_width(cell, width, align)
            formatted_cells.append(f" {formatted} ")

        return vert + vert.join(formatted_cells) + vert

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

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        pass


def create_plugin() -> TableFormatterPlugin:
    """Factory function to create a TableFormatterPlugin instance."""
    return TableFormatterPlugin()
