# shared/plugins/diff_formatter/renderers/side_by_side.py
"""Side-by-side diff renderer for wide terminals.

Displays old and new versions in parallel columns with box drawing,
line numbers on both sides, and word-level highlighting for changes.
"""

from typing import List, Optional, Tuple

from ..parser import ParsedDiff, DiffHunk, DiffLine, get_paired_lines
from ..word_diff import compute_word_diff, render_word_diff_old, render_word_diff_new
from ..box_drawing import (
    BOX_V, BOX_H, BOX_TL, BOX_TR, BOX_BL, BOX_BR,
    BOX_T_DOWN, BOX_T_UP, BOX_T_RIGHT, BOX_T_LEFT, BOX_CROSS,
    truncate_text, pad_text,
)
from .base import ColorScheme


# Minimum width for side-by-side display
MIN_SIDE_BY_SIDE_WIDTH = 120


class SideBySideRenderer:
    """Renders diff in side-by-side format with box drawing.

    Output format:
        ┌─ src/utils/parser.py ─────────────────────────────────────────────┐
        │      │ OLD                         │      │ NEW                   │
        ├──────┼─────────────────────────────┼──────┼───────────────────────┤
        │   10 │ def parse(path):            │   10 │ def parse(path):      │
        │   11 │     data = load(path)       │   11 │     data = load(path) │
        │   12 │     return validate(data)   │   12 │     return validate(d…│
        └──────┴─────────────────────────────┴──────┴───────────────────────┘
         +3 lines, -2 lines
    """

    def min_width(self) -> int:
        """Minimum width for side-by-side display."""
        return MIN_SIDE_BY_SIDE_WIDTH

    def render(self, diff: ParsedDiff, width: int, colors: ColorScheme) -> str:
        """Render diff in side-by-side format.

        Args:
            diff: Parsed diff structure.
            width: Terminal width.
            colors: Color scheme.

        Returns:
            Formatted side-by-side diff with box drawing.
        """
        # Use single-column mode for new/deleted files, two-column for modifications
        if diff.is_new_file or diff.is_deleted_file:
            return self._render_single_column(diff, width, colors)

        return self._render_two_column(diff, width, colors)

    def _render_two_column(
        self, diff: ParsedDiff, width: int, colors: ColorScheme
    ) -> str:
        """Render diff in two-column side-by-side format for modifications."""
        # Calculate column widths
        # Layout: │ ln_old │ content_old │ ln_new │ content_new │
        # That's 5 separators + 2 line number columns + 2 content columns
        line_no_width = 6  # "  123 " - space for up to 4-digit line numbers
        separator_count = 5
        available_for_content = width - (2 * line_no_width) - separator_count
        content_width = available_for_content // 2

        lines = []

        # Top border with file path
        lines.append(self._render_top_border(diff, width, colors))

        # Column headers
        lines.append(self._render_column_headers(
            line_no_width, content_width, colors
        ))

        # Header separator
        lines.append(self._render_separator(
            line_no_width, content_width, colors, is_header=True
        ))

        # Render each hunk
        for hunk in diff.hunks:
            lines.extend(self._render_hunk(
                hunk, line_no_width, content_width, colors
            ))

        # Bottom border
        lines.append(self._render_bottom_border(
            line_no_width, content_width, colors
        ))

        # Stats summary
        lines.append(self._render_stats(diff, colors))

        return "\n".join(lines)

    def _render_single_column(
        self, diff: ParsedDiff, width: int, colors: ColorScheme
    ) -> str:
        """Render diff in single-column format for new/deleted files."""
        # Layout: │ ln │ content │
        # That's 3 separators + 1 line number column + 1 content column
        line_no_width = 6
        separator_count = 3
        content_width = width - line_no_width - separator_count

        is_new = diff.is_new_file
        lines = []

        # Top border with file path
        lines.append(self._render_top_border(diff, width, colors))

        # Single column header (no header row for cleaner look)
        # Just go straight to the separator
        lines.append(self._render_single_separator(
            line_no_width, content_width, colors, is_top=True
        ))

        # Render each hunk
        for hunk in diff.hunks:
            lines.extend(self._render_hunk_single(
                hunk, line_no_width, content_width, colors, is_new
            ))

        # Bottom border
        lines.append(self._render_single_bottom_border(
            line_no_width, content_width, colors
        ))

        # Stats summary
        lines.append(self._render_stats(diff, colors))

        return "\n".join(lines)

    def _render_top_border(
        self, diff: ParsedDiff, width: int, colors: ColorScheme
    ) -> str:
        """Render top border with embedded file path."""
        path = diff.display_path
        title = f" {path} "

        # Calculate available space for dashes
        remaining = width - len(title) - 2  # -2 for corners
        if remaining < 4:
            return f"{colors.box}{BOX_TL}{BOX_H * (width - 2)}{BOX_TR}{colors.reset}"

        return (
            f"{colors.box}{BOX_TL}{BOX_H}"
            f"{colors.reset}{colors.header_path}{title}{colors.reset}"
            f"{colors.box}{BOX_H * (remaining - 1)}{BOX_TR}{colors.reset}"
        )

    def _render_column_headers(
        self, line_no_width: int, content_width: int, colors: ColorScheme
    ) -> str:
        """Render OLD/NEW column headers."""
        # Format line number columns (empty for headers)
        ln_old = pad_text("", line_no_width)
        ln_new = pad_text("", line_no_width)

        # Format content columns with headers
        old_header = pad_text("OLD", content_width)
        new_header = pad_text("NEW", content_width)

        return (
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.line_numbers}{ln_old}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.header_old}{old_header}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.line_numbers}{ln_new}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.header_new}{new_header}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
        )

    def _render_separator(
        self,
        line_no_width: int,
        content_width: int,
        colors: ColorScheme,
        is_header: bool = False,
    ) -> str:
        """Render horizontal separator line."""
        left = BOX_T_RIGHT if is_header else BOX_T_RIGHT
        right = BOX_T_LEFT if is_header else BOX_T_LEFT
        cross = BOX_CROSS

        return (
            f"{colors.box}"
            f"{left}{BOX_H * line_no_width}"
            f"{cross}{BOX_H * content_width}"
            f"{cross}{BOX_H * line_no_width}"
            f"{cross}{BOX_H * content_width}"
            f"{right}"
            f"{colors.reset}"
        )

    def _render_bottom_border(
        self, line_no_width: int, content_width: int, colors: ColorScheme
    ) -> str:
        """Render bottom border."""
        return (
            f"{colors.box}"
            f"{BOX_BL}{BOX_H * line_no_width}"
            f"{BOX_T_UP}{BOX_H * content_width}"
            f"{BOX_T_UP}{BOX_H * line_no_width}"
            f"{BOX_T_UP}{BOX_H * content_width}"
            f"{BOX_BR}"
            f"{colors.reset}"
        )

    def _render_stats(self, diff: ParsedDiff, colors: ColorScheme) -> str:
        """Render statistics summary below the box."""
        stats = diff.stats
        parts = []
        if stats.added:
            parts.append(f"+{stats.added} line{'s' if stats.added != 1 else ''}")
        if stats.deleted:
            parts.append(f"-{stats.deleted} line{'s' if stats.deleted != 1 else ''}")
        if stats.modified:
            parts.append(f"~{stats.modified} modified")

        if parts:
            return f"{colors.stats} {', '.join(parts)}{colors.reset}"
        return f"{colors.stats} no changes{colors.reset}"

    def _render_hunk(
        self,
        hunk: DiffHunk,
        line_no_width: int,
        content_width: int,
        colors: ColorScheme,
    ) -> List[str]:
        """Render a single hunk as side-by-side rows."""
        lines = []
        pairs = get_paired_lines(hunk)

        for old_line, new_line in pairs:
            lines.append(self._render_pair(
                old_line, new_line,
                line_no_width, content_width,
                colors
            ))

        return lines

    def _render_pair(
        self,
        old_line: Optional[DiffLine],
        new_line: Optional[DiffLine],
        line_no_width: int,
        content_width: int,
        colors: ColorScheme,
    ) -> str:
        """Render a pair of old/new lines as a single row."""
        # Determine line numbers
        old_ln = ""
        new_ln = ""
        old_content = ""
        new_content = ""
        old_color = ""
        new_color = ""

        if old_line:
            if old_line.old_line_no is not None:
                old_ln = str(old_line.old_line_no)
            old_content = old_line.content

        if new_line:
            if new_line.new_line_no is not None:
                new_ln = str(new_line.new_line_no)
            new_content = new_line.content

        # Determine colors and apply word-level diff for modified lines
        if old_line and new_line:
            if old_line.change_type == "modified" and new_line.change_type == "modified":
                # Modified - compute word-level diff
                word_diff = compute_word_diff(old_content, new_content)

                # Render with word-level highlighting
                old_content = render_word_diff_old(
                    word_diff, colors.deleted_bold, colors.reset
                )
                new_content = render_word_diff_new(
                    word_diff, colors.added_bold, colors.reset
                )
                old_color = colors.deleted
                new_color = colors.added
            elif old_line.change_type == "unchanged":
                # Unchanged - no special coloring
                pass
        elif old_line and old_line.change_type in ("deleted", "modified"):
            old_color = colors.deleted
        elif new_line and new_line.change_type in ("added", "modified"):
            new_color = colors.added

        # Format line numbers (right-aligned)
        old_ln_fmt = pad_text(old_ln, line_no_width, align="right")
        new_ln_fmt = pad_text(new_ln, line_no_width, align="right")

        # Truncate and pad content
        # Need to handle ANSI codes in content for word-level diff
        old_content_fmt = self._format_content(old_content, content_width)
        new_content_fmt = self._format_content(new_content, content_width)

        # Build the row
        parts = [
            f"{colors.box}{BOX_V}{colors.reset}",
            f"{colors.line_numbers}{old_ln_fmt}{colors.reset}",
            f"{colors.box}{BOX_V}{colors.reset}",
        ]

        # Old content with color
        if old_color:
            parts.append(f"{old_color}{old_content_fmt}{colors.reset}")
        else:
            parts.append(old_content_fmt)

        parts.extend([
            f"{colors.box}{BOX_V}{colors.reset}",
            f"{colors.line_numbers}{new_ln_fmt}{colors.reset}",
            f"{colors.box}{BOX_V}{colors.reset}",
        ])

        # New content with color
        if new_color:
            parts.append(f"{new_color}{new_content_fmt}{colors.reset}")
        else:
            parts.append(new_content_fmt)

        parts.append(f"{colors.box}{BOX_V}{colors.reset}")

        return "".join(parts)

    def _format_content(self, content: str, width: int) -> str:
        """Format content to fit width, handling ANSI codes.

        Args:
            content: Content (may contain ANSI codes).
            width: Target width.

        Returns:
            Content truncated/padded to exact width.
        """
        # Calculate visible length (excluding ANSI codes)
        visible_len = self._visible_length(content)

        if visible_len <= width:
            # Pad to width
            padding = width - visible_len
            return content + " " * padding
        else:
            # Truncate with ellipsis
            return self._truncate_with_ansi(content, width)

    def _visible_length(self, text: str) -> int:
        """Calculate visible length excluding ANSI escape codes."""
        import re
        ansi_pattern = re.compile(r'\033\[[0-9;]*m')
        return len(ansi_pattern.sub('', text))

    def _truncate_with_ansi(self, text: str, width: int) -> str:
        """Truncate text to width, preserving ANSI codes and adding ellipsis."""
        import re
        ansi_pattern = re.compile(r'(\033\[[0-9;]*m)')

        if width <= 1:
            return "…"

        result = []
        visible_count = 0
        target = width - 1  # Leave room for ellipsis

        parts = ansi_pattern.split(text)
        for part in parts:
            if ansi_pattern.match(part):
                # ANSI code - always include
                result.append(part)
            else:
                # Regular text - count visible chars
                remaining = target - visible_count
                if remaining <= 0:
                    break
                if len(part) <= remaining:
                    result.append(part)
                    visible_count += len(part)
                else:
                    result.append(part[:remaining])
                    visible_count += remaining
                    break

        result.append("…")
        # Pad if needed
        padding = width - visible_count - 1
        if padding > 0:
            result.append(" " * padding)

        return "".join(result)

    # -------------------------------------------------------------------------
    # Single-column rendering methods (for new/deleted files)
    # -------------------------------------------------------------------------

    def _render_single_separator(
        self,
        line_no_width: int,
        content_width: int,
        colors: ColorScheme,
        is_top: bool = False,
    ) -> str:
        """Render horizontal separator for single-column layout."""
        left = BOX_T_RIGHT
        right = BOX_T_LEFT
        cross = BOX_CROSS if not is_top else BOX_T_DOWN

        return (
            f"{colors.box}"
            f"{left}{BOX_H * line_no_width}"
            f"{cross}{BOX_H * content_width}"
            f"{right}"
            f"{colors.reset}"
        )

    def _render_single_bottom_border(
        self, line_no_width: int, content_width: int, colors: ColorScheme
    ) -> str:
        """Render bottom border for single-column layout."""
        return (
            f"{colors.box}"
            f"{BOX_BL}{BOX_H * line_no_width}"
            f"{BOX_T_UP}{BOX_H * content_width}"
            f"{BOX_BR}"
            f"{colors.reset}"
        )

    def _render_hunk_single(
        self,
        hunk: DiffHunk,
        line_no_width: int,
        content_width: int,
        colors: ColorScheme,
        is_new: bool,
    ) -> List[str]:
        """Render a single hunk in single-column format."""
        lines = []

        for line in hunk.lines:
            lines.append(self._render_line_single(
                line, line_no_width, content_width, colors, is_new
            ))

        return lines

    def _render_line_single(
        self,
        line: DiffLine,
        line_no_width: int,
        content_width: int,
        colors: ColorScheme,
        is_new: bool,
    ) -> str:
        """Render a single line in single-column format."""
        # Get appropriate line number
        if is_new:
            ln = str(line.new_line_no) if line.new_line_no is not None else ""
            color = colors.added if line.change_type == "added" else ""
        else:
            ln = str(line.old_line_no) if line.old_line_no is not None else ""
            color = colors.deleted if line.change_type == "deleted" else ""

        # Format line number (right-aligned)
        ln_fmt = pad_text(ln, line_no_width, align="right")

        # Format content
        content_fmt = self._format_content(line.content, content_width)

        # Build the row
        if color:
            return (
                f"{colors.box}{BOX_V}{colors.reset}"
                f"{colors.line_numbers}{ln_fmt}{colors.reset}"
                f"{colors.box}{BOX_V}{colors.reset}"
                f"{color}{content_fmt}{colors.reset}"
                f"{colors.box}{BOX_V}{colors.reset}"
            )
        else:
            return (
                f"{colors.box}{BOX_V}{colors.reset}"
                f"{colors.line_numbers}{ln_fmt}{colors.reset}"
                f"{colors.box}{BOX_V}{colors.reset}"
                f"{content_fmt}"
                f"{colors.box}{BOX_V}{colors.reset}"
            )
