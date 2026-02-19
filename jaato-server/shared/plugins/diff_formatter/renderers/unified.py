# shared/plugins/diff_formatter/renderers/unified.py
"""Unified diff renderer - classic +/- format with colors.

This is the fallback renderer for narrow terminals, preserving
the traditional unified diff format with ANSI color highlighting.
"""

from ..parser import ParsedDiff, DiffHunk, DiffLine
from .base import ColorScheme


class UnifiedRenderer:
    """Renders diff in traditional unified format with colors.

    Output format:
        ── path/to/file.py (+5, -3) ──
        @@ -10,6 +10,8 @@
         context line
        -deleted line
        +added line
         context line
    """

    def min_width(self) -> int:
        """No minimum width - this is the fallback renderer."""
        return 0

    def render(self, diff: ParsedDiff, width: int, colors: ColorScheme) -> str:
        """Render diff in unified format with colors.

        Args:
            diff: Parsed diff structure.
            width: Terminal width (used for header decoration).
            colors: Color scheme.

        Returns:
            Colored unified diff string.
        """
        lines = []

        # Header with file path and stats
        lines.append(self._render_header(diff, width, colors))

        # Render each hunk
        for hunk in diff.hunks:
            lines.append(self._render_hunk_header(hunk, colors))
            for line in hunk.lines:
                lines.append(self._render_line(line, colors))

        return "\n".join(lines)

    def _render_header(self, diff: ParsedDiff, width: int, colors: ColorScheme) -> str:
        """Render file header with stats."""
        path = diff.display_path
        stats = diff.stats

        # Format: ── path/to/file.py (+5, -3) ──
        stat_str = f" ({stats})" if stats.total_changes > 0 else ""
        header_text = f" {path}{stat_str} "

        # Calculate decorations
        available = width - len(header_text)
        if available > 4:
            left = available // 2
            right = available - left
            decorated = f"{'─' * left}{header_text}{'─' * right}"
        else:
            decorated = header_text

        return f"{colors.header_path}{decorated}{colors.reset}"

    def _render_hunk_header(self, hunk: DiffHunk, colors: ColorScheme) -> str:
        """Render @@ hunk header."""
        header = f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
        if hunk.header_extra:
            header = f"{header} {hunk.header_extra}"
        return f"{colors.hunk_header}{header}{colors.reset}"

    def _render_line(self, line: DiffLine, colors: ColorScheme) -> str:
        """Render a single diff line with appropriate prefix and color."""
        if line.change_type == "unchanged":
            return f" {line.content}"
        elif line.change_type == "added":
            return f"{colors.added}+{line.content}{colors.reset}"
        elif line.change_type == "deleted":
            return f"{colors.deleted}-{line.content}{colors.reset}"
        elif line.change_type == "modified":
            # Modified lines show as +/- based on which version
            if line.old_line_no is not None:
                return f"{colors.deleted}-{line.content}{colors.reset}"
            else:
                return f"{colors.added}+{line.content}{colors.reset}"
        else:
            return f" {line.content}"


def render_raw_unified(diff_text: str, colors: ColorScheme) -> str:
    """Render raw unified diff text with colors (without parsing).

    This is a simpler approach that just colorizes lines based on prefix,
    used when we receive raw diff text and don't need structured data.

    Args:
        diff_text: Raw unified diff text.
        colors: Color scheme.

    Returns:
        Colorized diff text.
    """
    lines = diff_text.split("\n")
    colored = []

    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            colored.append(f"{colors.dim}{line}{colors.reset}")
        elif line.startswith("@@"):
            colored.append(f"{colors.hunk_header}{line}{colors.reset}")
        elif line.startswith("+"):
            colored.append(f"{colors.added}{line}{colors.reset}")
        elif line.startswith("-"):
            colored.append(f"{colors.deleted}{line}{colors.reset}")
        else:
            colored.append(line)

    return "\n".join(colored)
