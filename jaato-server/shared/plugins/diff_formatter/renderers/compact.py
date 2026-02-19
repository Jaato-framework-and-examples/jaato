# shared/plugins/diff_formatter/renderers/compact.py
"""Compact diff renderer for medium-width terminals.

Uses a single-column layout with box drawing, showing changes
with arrow notation (→) for modifications and +/- for pure adds/deletes.
"""

from typing import List

from ..parser import ParsedDiff, DiffHunk, DiffLine
from ..word_diff import compute_word_diff
from ..box_drawing import (
    BOX_V, BOX_H, BOX_TL, BOX_TR, BOX_BL, BOX_BR,
    BOX_T_DOWN, BOX_T_UP, BOX_T_RIGHT, BOX_T_LEFT,
    pad_text,
)
from .base import ColorScheme


# Minimum width for compact display
MIN_COMPACT_WIDTH = 80


class CompactRenderer:
    """Renders diff in compact single-column format with box drawing.

    Output format:
        ┌─ src/utils/parser.py ─────────────────────────────────────────────┐
        │  12 │     result = validate(data) → validate(data, strict=True)   │
        │  13 │ -   return result                                           │
        │  13 │ +   if not result.ok:                                       │
        │  14 │ +       raise ConfigError(result)                           │
        │  15 │ +   return result                                           │
        └───────────────────────────────────────────────────────────────────┘
         +3 lines, -2 lines

    Modified lines use → to show inline changes.
    Pure additions use + prefix.
    Pure deletions use - prefix.
    """

    def min_width(self) -> int:
        """Minimum width for compact display."""
        return MIN_COMPACT_WIDTH

    def render(self, diff: ParsedDiff, width: int, colors: ColorScheme) -> str:
        """Render diff in compact format.

        Args:
            diff: Parsed diff structure.
            width: Terminal width.
            colors: Color scheme.

        Returns:
            Formatted compact diff with box drawing.
        """
        # Layout: │ ln │ marker content │
        # marker is 2 chars: "  ", "+ ", "- ", "→ "
        line_no_width = 5  # " 123 "
        marker_width = 2   # "+ " or "- " or "→ " or "  "
        # Box chars: 3 (│ at start, │ after ln, │ at end)
        content_width = width - line_no_width - marker_width - 3

        lines = []

        # Top border with file path
        lines.append(self._render_top_border(diff, width, colors))

        # Render each hunk
        for hunk in diff.hunks:
            lines.extend(self._render_hunk(
                hunk, line_no_width, marker_width, content_width, colors
            ))

        # Bottom border
        lines.append(self._render_bottom_border(width, colors))

        # Stats summary
        lines.append(self._render_stats(diff, colors))

        return "\n".join(lines)

    def _render_top_border(
        self, diff: ParsedDiff, width: int, colors: ColorScheme
    ) -> str:
        """Render top border with embedded file path."""
        path = diff.display_path
        title = f" {path} "

        remaining = width - len(title) - 2
        if remaining < 4:
            return f"{colors.box}{BOX_TL}{BOX_H * (width - 2)}{BOX_TR}{colors.reset}"

        return (
            f"{colors.box}{BOX_TL}{BOX_H}"
            f"{colors.reset}{colors.header_path}{title}{colors.reset}"
            f"{colors.box}{BOX_H * (remaining - 1)}{BOX_TR}{colors.reset}"
        )

    def _render_bottom_border(self, width: int, colors: ColorScheme) -> str:
        """Render bottom border."""
        return f"{colors.box}{BOX_BL}{BOX_H * (width - 2)}{BOX_BR}{colors.reset}"

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
        marker_width: int,
        content_width: int,
        colors: ColorScheme,
    ) -> List[str]:
        """Render a single hunk."""
        lines = []

        i = 0
        while i < len(hunk.lines):
            line = hunk.lines[i]

            if line.change_type == "unchanged":
                # Context line - skip in compact view to save space
                # Uncomment below to show context:
                # lines.append(self._render_context_line(
                #     line, line_no_width, marker_width, content_width, colors
                # ))
                i += 1
            elif line.change_type == "modified":
                # Modified line - show with arrow notation
                if line.old_line_no is not None and line.paired_with:
                    lines.append(self._render_modified_line(
                        line, line.paired_with,
                        line_no_width, marker_width, content_width, colors
                    ))
                    # Skip the paired new line
                    i += 1
                    if i < len(hunk.lines) and hunk.lines[i] == line.paired_with:
                        i += 1
                else:
                    # Orphaned modified line, show as add/delete
                    if line.old_line_no is not None:
                        lines.append(self._render_deleted_line(
                            line, line_no_width, marker_width, content_width, colors
                        ))
                    else:
                        lines.append(self._render_added_line(
                            line, line_no_width, marker_width, content_width, colors
                        ))
                    i += 1
            elif line.change_type == "deleted":
                lines.append(self._render_deleted_line(
                    line, line_no_width, marker_width, content_width, colors
                ))
                i += 1
            elif line.change_type == "added":
                lines.append(self._render_added_line(
                    line, line_no_width, marker_width, content_width, colors
                ))
                i += 1
            else:
                i += 1

        return lines

    def _render_modified_line(
        self,
        old_line: DiffLine,
        new_line: DiffLine,
        line_no_width: int,
        marker_width: int,
        content_width: int,
        colors: ColorScheme,
    ) -> str:
        """Render a modified line with arrow notation: old → new."""
        line_no = str(old_line.old_line_no) if old_line.old_line_no else ""
        ln_fmt = pad_text(line_no, line_no_width, align="right")

        # Compute word-level diff to show only changed parts
        word_diff = compute_word_diff(old_line.content, new_line.content)

        # For compact view, show condensed format: old_part → new_part
        # Extract the changed portions
        old_changed = "".join(
            text for text, is_changed in word_diff.old_segments if is_changed
        )
        new_changed = "".join(
            text for text, is_changed in word_diff.new_segments if is_changed
        )

        # If changes are simple enough, show inline
        # Otherwise fall back to showing full lines
        arrow = " → "
        if old_changed and new_changed and len(old_changed) + len(new_changed) + len(arrow) < content_width // 2:
            # Show context + highlighted change
            content = self._build_inline_change(
                old_line.content, new_line.content, word_diff,
                content_width - marker_width, colors
            )
        else:
            # Show old → new (truncated)
            available = content_width - marker_width - len(arrow)
            half = available // 2
            old_truncated = self._truncate(old_line.content.strip(), half)
            new_truncated = self._truncate(new_line.content.strip(), half)
            content = (
                f"{colors.deleted}{old_truncated}{colors.reset}"
                f"{arrow}"
                f"{colors.added}{new_truncated}{colors.reset}"
            )

        marker = "→ "
        row_content = f"{marker}{content}"

        # Pad to content width (accounting for ANSI)
        visible_len = self._visible_length(row_content)
        padding = content_width - visible_len
        if padding > 0:
            row_content += " " * padding

        return (
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.line_numbers}{ln_fmt}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{row_content}"
            f"{colors.box}{BOX_V}{colors.reset}"
        )

    def _render_added_line(
        self,
        line: DiffLine,
        line_no_width: int,
        marker_width: int,
        content_width: int,
        colors: ColorScheme,
    ) -> str:
        """Render an added line with + prefix."""
        line_no = str(line.new_line_no) if line.new_line_no else ""
        ln_fmt = pad_text(line_no, line_no_width, align="right")

        content = self._truncate(line.content, content_width - marker_width)
        padded = pad_text(content, content_width - marker_width)

        return (
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.line_numbers}{ln_fmt}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.added}+ {padded}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
        )

    def _render_deleted_line(
        self,
        line: DiffLine,
        line_no_width: int,
        marker_width: int,
        content_width: int,
        colors: ColorScheme,
    ) -> str:
        """Render a deleted line with - prefix."""
        line_no = str(line.old_line_no) if line.old_line_no else ""
        ln_fmt = pad_text(line_no, line_no_width, align="right")

        content = self._truncate(line.content, content_width - marker_width)
        padded = pad_text(content, content_width - marker_width)

        return (
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.line_numbers}{ln_fmt}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.deleted}- {padded}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
        )

    def _render_context_line(
        self,
        line: DiffLine,
        line_no_width: int,
        marker_width: int,
        content_width: int,
        colors: ColorScheme,
    ) -> str:
        """Render a context (unchanged) line."""
        line_no = str(line.old_line_no) if line.old_line_no else ""
        ln_fmt = pad_text(line_no, line_no_width, align="right")

        content = self._truncate(line.content, content_width - marker_width)
        padded = pad_text(content, content_width - marker_width)

        return (
            f"{colors.box}{BOX_V}{colors.reset}"
            f"{colors.line_numbers}{ln_fmt}{colors.reset}"
            f"{colors.box}{BOX_V}{colors.reset}"
            f"  {padded}"
            f"{colors.box}{BOX_V}{colors.reset}"
        )

    def _build_inline_change(
        self,
        old_content: str,
        new_content: str,
        word_diff,
        max_width: int,
        colors: ColorScheme,
    ) -> str:
        """Build inline change display showing context with highlighted changes."""
        # Find common prefix and suffix for context
        # Then show: prefix [old→new] suffix
        parts = []

        # Build new content with changes highlighted
        for text, is_changed in word_diff.new_segments:
            if is_changed:
                parts.append(f"{colors.added_bold}{text}{colors.reset}")
            else:
                parts.append(text)

        result = "".join(parts)

        # Truncate if needed
        visible_len = self._visible_length(result)
        if visible_len > max_width:
            return self._truncate_with_ansi(result, max_width)

        return result

    def _truncate(self, text: str, width: int) -> str:
        """Truncate text with ellipsis if needed."""
        if len(text) <= width:
            return text
        if width <= 1:
            return "…"
        return text[:width - 1] + "…"

    def _visible_length(self, text: str) -> int:
        """Calculate visible length excluding ANSI codes."""
        import re
        ansi_pattern = re.compile(r'\033\[[0-9;]*m')
        return len(ansi_pattern.sub('', text))

    def _truncate_with_ansi(self, text: str, width: int) -> str:
        """Truncate text with ANSI codes, preserving codes."""
        import re
        ansi_pattern = re.compile(r'(\033\[[0-9;]*m)')

        if width <= 1:
            return "…"

        result = []
        visible_count = 0
        target = width - 1

        parts = ansi_pattern.split(text)
        for part in parts:
            if ansi_pattern.match(part):
                result.append(part)
            else:
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
        return "".join(result)
