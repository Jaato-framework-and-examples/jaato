# shared/plugins/diff_formatter/box_drawing.py
"""Unicode box drawing utilities for diff rendering.

Provides functions to create box-drawn tables and frames
using Unicode box-drawing characters.
"""

from typing import List, Optional

# Box drawing characters (single line)
BOX_TL = "┌"  # Top-left corner
BOX_TR = "┐"  # Top-right corner
BOX_BL = "└"  # Bottom-left corner
BOX_BR = "┘"  # Bottom-right corner
BOX_H = "─"   # Horizontal line
BOX_V = "│"   # Vertical line
BOX_CROSS = "┼"  # Cross (all four directions)
BOX_T_DOWN = "┬"  # T pointing down
BOX_T_UP = "┴"    # T pointing up
BOX_T_RIGHT = "├"  # T pointing right
BOX_T_LEFT = "┤"   # T pointing left


def truncate_text(text: str, width: int, ellipsis: str = "…") -> str:
    """Truncate text to fit within width, adding ellipsis if needed.

    Args:
        text: Text to truncate.
        width: Maximum width.
        ellipsis: String to append when truncating.

    Returns:
        Truncated text that fits within width.
    """
    if len(text) <= width:
        return text
    if width <= len(ellipsis):
        return ellipsis[:width]
    return text[:width - len(ellipsis)] + ellipsis


def pad_text(text: str, width: int, align: str = "left") -> str:
    """Pad text to exact width.

    Args:
        text: Text to pad.
        width: Target width.
        align: Alignment - "left", "right", or "center".

    Returns:
        Padded text of exact width.
    """
    if len(text) >= width:
        return text[:width]

    padding = width - len(text)
    if align == "right":
        return " " * padding + text
    elif align == "center":
        left = padding // 2
        right = padding - left
        return " " * left + text + " " * right
    else:  # left
        return text + " " * padding


def box_top(width: int, title: Optional[str] = None, title_color: str = "", reset: str = "") -> str:
    """Create top border of a box, optionally with embedded title.

    Args:
        width: Total width of the box (including corners).
        title: Optional title to embed in the border.
        title_color: ANSI color code for title.
        reset: ANSI reset code.

    Returns:
        Top border string like "┌─ title ─────┐"
    """
    if title:
        # Format: ┌─ title ─────────┐
        title_part = f" {title_color}{title}{reset} "
        # Account for ANSI codes not taking visual space
        visual_title_len = len(f" {title} ")
        remaining = width - 2 - visual_title_len  # -2 for corners
        if remaining < 2:
            # Not enough space, just draw plain border
            return BOX_TL + BOX_H * (width - 2) + BOX_TR
        return BOX_TL + BOX_H + title_part + BOX_H * (remaining - 1) + BOX_TR
    else:
        return BOX_TL + BOX_H * (width - 2) + BOX_TR


def box_bottom(width: int) -> str:
    """Create bottom border of a box.

    Args:
        width: Total width of the box (including corners).

    Returns:
        Bottom border string like "└─────────────┘"
    """
    return BOX_BL + BOX_H * (width - 2) + BOX_BR


def box_separator(widths: List[int]) -> str:
    """Create horizontal separator with column breaks.

    Args:
        widths: List of column widths (not including separators).

    Returns:
        Separator string like "├──────┼──────┼──────┤"
    """
    if not widths:
        return ""

    parts = [BOX_T_RIGHT]
    for i, w in enumerate(widths):
        parts.append(BOX_H * w)
        if i < len(widths) - 1:
            parts.append(BOX_CROSS)
    parts.append(BOX_T_LEFT)

    return "".join(parts)


def box_row(cells: List[str], widths: List[int]) -> str:
    """Create a row with vertical separators.

    Args:
        cells: List of cell contents.
        widths: List of column widths.

    Returns:
        Row string like "│ cell1 │ cell2 │"
    """
    if not cells:
        return BOX_V + BOX_V

    parts = [BOX_V]
    for i, (cell, width) in enumerate(zip(cells, widths)):
        # Truncate if needed, then pad
        content = truncate_text(cell, width)
        content = pad_text(content, width)
        parts.append(content)
        parts.append(BOX_V)

    return "".join(parts)


def box_row_colored(
    cells: List[str],
    widths: List[int],
    colors: List[str],
    reset: str = "",
) -> str:
    """Create a row with vertical separators and per-cell colors.

    Args:
        cells: List of cell contents.
        widths: List of column widths.
        colors: List of ANSI color codes (one per cell, empty string for no color).
        reset: ANSI reset code.

    Returns:
        Row string with colors like "│ [green]cell1[reset] │ [red]cell2[reset] │"
    """
    if not cells:
        return BOX_V + BOX_V

    parts = [BOX_V]
    for cell, width, color in zip(cells, widths, colors):
        # Truncate if needed, then pad
        content = truncate_text(cell, width)
        content = pad_text(content, width)
        if color:
            parts.append(f"{color}{content}{reset}")
        else:
            parts.append(content)
        parts.append(BOX_V)

    return "".join(parts)


def calculate_column_widths(
    total_width: int,
    num_columns: int,
    fixed_columns: Optional[List[int]] = None,
) -> List[int]:
    """Calculate column widths to fill total width.

    Args:
        total_width: Total available width.
        num_columns: Number of columns.
        fixed_columns: Optional list of (index, width) for fixed-width columns.

    Returns:
        List of column widths.
    """
    # Account for separators: │col│col│col│ = num_columns + 1 separators
    available = total_width - (num_columns + 1)

    if fixed_columns is None:
        fixed_columns = []

    # Build width list
    widths = [0] * num_columns
    fixed_total = 0

    for idx, width in fixed_columns:
        if 0 <= idx < num_columns:
            widths[idx] = width
            fixed_total += width

    # Distribute remaining space
    flexible_count = sum(1 for w in widths if w == 0)
    if flexible_count > 0:
        remaining = available - fixed_total
        per_column = remaining // flexible_count
        extra = remaining % flexible_count

        for i in range(num_columns):
            if widths[i] == 0:
                widths[i] = per_column + (1 if extra > 0 else 0)
                extra -= 1

    return widths


class BoxBuilder:
    """Builder for creating box-drawn tables.

    Example:
        builder = BoxBuilder(width=80)
        builder.add_title("My Table")
        builder.add_header(["Col1", "Col2"])
        builder.add_row(["data1", "data2"])
        result = builder.build()
    """

    def __init__(self, width: int, column_widths: Optional[List[int]] = None):
        """Initialize builder.

        Args:
            width: Total width of the box.
            column_widths: Optional explicit column widths.
        """
        self.width = width
        self._column_widths = column_widths
        self._title: Optional[str] = None
        self._title_color = ""
        self._rows: List[tuple] = []  # (type, data)
        self._reset = ""

    def set_reset_code(self, reset: str) -> "BoxBuilder":
        """Set ANSI reset code for colors."""
        self._reset = reset
        return self

    def add_title(self, title: str, color: str = "") -> "BoxBuilder":
        """Set the title for the top border."""
        self._title = title
        self._title_color = color
        return self

    def add_header(self, cells: List[str], colors: Optional[List[str]] = None) -> "BoxBuilder":
        """Add a header row."""
        self._rows.append(("header", (cells, colors)))
        return self

    def add_row(self, cells: List[str], colors: Optional[List[str]] = None) -> "BoxBuilder":
        """Add a content row."""
        self._rows.append(("row", (cells, colors)))
        return self

    def add_separator(self) -> "BoxBuilder":
        """Add a horizontal separator."""
        self._rows.append(("separator", None))
        return self

    def build(self) -> str:
        """Build the complete box."""
        lines = []

        # Determine column widths from first row if not set
        if self._column_widths is None and self._rows:
            for row_type, data in self._rows:
                if row_type in ("header", "row"):
                    cells, _ = data
                    self._column_widths = calculate_column_widths(
                        self.width, len(cells)
                    )
                    break

        if self._column_widths is None:
            self._column_widths = [self.width - 2]

        # Top border
        lines.append(box_top(self.width, self._title, self._title_color, self._reset))

        # Rows
        for i, (row_type, data) in enumerate(self._rows):
            if row_type == "separator":
                lines.append(box_separator(self._column_widths))
            elif row_type in ("header", "row"):
                cells, colors = data
                if colors:
                    lines.append(box_row_colored(
                        cells, self._column_widths, colors, self._reset
                    ))
                else:
                    lines.append(box_row(cells, self._column_widths))

                # Add separator after header
                if row_type == "header" and i < len(self._rows) - 1:
                    lines.append(box_separator(self._column_widths))

        # Bottom border
        lines.append(box_bottom(self.width))

        return "\n".join(lines)
