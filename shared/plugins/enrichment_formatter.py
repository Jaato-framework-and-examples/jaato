"""Formatter for enrichment notification visual output.

Produces dimmed, aligned multi-line notifications showing what enrichments
were applied to prompts or tool results.

Visual format:
  ╭ prompt ← memory: recalled discussion about JWT token
  │                  expiration bug from yesterday's session
  │                  (1.2kb)
  ╰ prompt ← references: expanded @api-spec (4.2kb)

Single enrichment:
  ── result ← references: extracted @error-codes (892b)
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EnrichmentNotification:
    """A single enrichment notification to display.

    Attributes:
        target: What was enriched - "prompt", "result", or "system"
        plugin: Name of the plugin that performed enrichment
        message: Human-readable description of what was done
        size_bytes: Size of the enrichment in bytes (for display)
    """
    target: str  # "prompt", "result", "system"
    plugin: str
    message: str
    size_bytes: Optional[int] = None


def format_size(size_bytes: Optional[int]) -> str:
    """Format byte size for display.

    Args:
        size_bytes: Size in bytes, or None.

    Returns:
        Formatted string like "(1.2kb)", "(892b)", or empty string.
    """
    if size_bytes is None:
        return ""

    if size_bytes < 1024:
        return f"({size_bytes}b)"
    elif size_bytes < 1024 * 1024:
        kb = size_bytes / 1024
        if kb >= 10:
            return f"({int(kb)}kb)"
        return f"({kb:.1f}kb)"
    else:
        mb = size_bytes / (1024 * 1024)
        if mb >= 10:
            return f"({int(mb)}mb)"
        return f"({mb:.1f}mb)"


def format_enrichment_notifications(
    notifications: List[EnrichmentNotification],
    terminal_width: int = 80
) -> str:
    """Format a list of enrichment notifications for display.

    Args:
        notifications: List of notifications to format.
        terminal_width: Terminal width for word wrapping.

    Returns:
        Formatted string with proper prefixes and alignment.
    """
    if not notifications:
        return ""

    lines = []
    count = len(notifications)

    for i, notif in enumerate(notifications):
        # Determine prefix character
        if count == 1:
            prefix = "──"
            cont_prefix = "  "
        elif i == 0:
            prefix = "╭"
            cont_prefix = "│"
        elif i == count - 1:
            prefix = "╰"
            cont_prefix = " "
        else:
            prefix = "├"
            cont_prefix = "│"

        # Build the content: "target ← plugin: message (size)"
        size_str = format_size(notif.size_bytes)
        if size_str:
            content = f"{notif.message} {size_str}"
        else:
            content = notif.message

        # Format: "  {prefix} {target} ← {plugin}: {content}"
        header = f"{notif.target} ← {notif.plugin}: "
        first_line_prefix = f"  {prefix} {header}"

        # Calculate continuation indent (align to first word after colon)
        # "  ╭ prompt ← memory: " -> continuation starts at same position as message
        cont_indent = len(f"  {cont_prefix} {header}")

        # Word wrap the content
        wrapped = _word_wrap(content, terminal_width, first_line_prefix, cont_indent, cont_prefix)
        lines.extend(wrapped)

    return "\n".join(lines)


def _word_wrap(
    content: str,
    terminal_width: int,
    first_line_prefix: str,
    cont_indent: int,
    cont_prefix: str
) -> List[str]:
    """Word wrap content with proper prefixes and alignment.

    Args:
        content: The text content to wrap.
        terminal_width: Maximum line width.
        first_line_prefix: Prefix for the first line (e.g., "  ╭ prompt ← memory: ").
        cont_indent: Number of spaces to indent continuation lines.
        cont_prefix: The prefix char for continuation lines ("│" or " ").

    Returns:
        List of formatted lines.
    """
    words = content.split()
    if not words:
        return [first_line_prefix]

    lines = []
    current_line = first_line_prefix
    first_line_max = terminal_width
    cont_line_max = terminal_width

    # Build continuation line prefix: "  │" + spaces to align
    # cont_indent includes the "  {prefix} {header}" length
    # We need: "  {cont_prefix}" + spaces to reach cont_indent
    base_cont_prefix = f"  {cont_prefix} "
    spaces_needed = cont_indent - len(base_cont_prefix)
    cont_line_prefix = base_cont_prefix + " " * max(0, spaces_needed)

    is_first_line = True

    for word in words:
        max_width = first_line_max if is_first_line else cont_line_max

        # Check if adding this word exceeds the width
        test_line = current_line + (" " if current_line.rstrip() != current_line.rstrip().rstrip() or
                                     (is_first_line and current_line != first_line_prefix) or
                                     (not is_first_line and current_line != cont_line_prefix)
                                     else "")
        if current_line.endswith(": ") or current_line == first_line_prefix or current_line == cont_line_prefix:
            test_line = current_line + word
        else:
            test_line = current_line + " " + word

        if len(test_line) <= max_width:
            current_line = test_line
        else:
            # Start a new line
            if current_line.strip():
                lines.append(current_line)
            is_first_line = False
            current_line = cont_line_prefix + word

    # Add the last line
    if current_line.strip():
        lines.append(current_line)

    return lines if lines else [first_line_prefix]
