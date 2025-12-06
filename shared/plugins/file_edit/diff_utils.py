"""Diff generation utilities for file editing operations.

Uses Python's difflib to generate unified diffs for display in the
permission approval UI.
"""

import difflib
from pathlib import Path
from typing import Optional, Tuple


# Default maximum lines to show in diff preview
DEFAULT_MAX_LINES = 50


def generate_unified_diff(
    old_content: str,
    new_content: str,
    file_path: str,
    max_lines: Optional[int] = DEFAULT_MAX_LINES
) -> Tuple[str, bool, int]:
    """Generate a unified diff between old and new content.

    Args:
        old_content: Original file content
        new_content: New file content
        file_path: Path to the file (used in diff header)
        max_lines: Maximum lines to include in diff. None for unlimited.

    Returns:
        Tuple of (diff_string, was_truncated, total_lines)
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    # Generate unified diff
    diff_lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=""
    ))

    total_lines = len(diff_lines)
    truncated = False

    if max_lines is not None and total_lines > max_lines:
        diff_lines = diff_lines[:max_lines]
        truncated = True

    # Join lines, handling potential trailing newlines
    diff_text = "\n".join(line.rstrip('\n\r') for line in diff_lines)

    return diff_text, truncated, total_lines


def generate_new_file_diff(
    content: str,
    file_path: str,
    max_lines: Optional[int] = DEFAULT_MAX_LINES
) -> Tuple[str, bool, int]:
    """Generate a diff showing a new file creation (all additions).

    Args:
        content: Content of the new file
        file_path: Path to the new file
        max_lines: Maximum lines to include. None for unlimited.

    Returns:
        Tuple of (diff_string, was_truncated, total_lines)
    """
    lines = content.splitlines()
    total_lines = len(lines) + 4  # +4 for diff header lines

    # Build diff header
    diff_parts = [
        f"--- /dev/null",
        f"+++ b/{file_path}",
        f"@@ -0,0 +1,{len(lines)} @@"
    ]

    # Add content lines with + prefix
    for line in lines:
        diff_parts.append(f"+{line}")

    truncated = False
    if max_lines is not None and len(diff_parts) > max_lines:
        diff_parts = diff_parts[:max_lines]
        truncated = True

    diff_text = "\n".join(diff_parts)
    return diff_text, truncated, total_lines


def generate_delete_file_diff(
    content: str,
    file_path: str,
    max_lines: Optional[int] = DEFAULT_MAX_LINES
) -> Tuple[str, bool, int]:
    """Generate a diff showing a file deletion (all removals).

    Args:
        content: Current content of the file to be deleted
        file_path: Path to the file
        max_lines: Maximum lines to include. None for unlimited.

    Returns:
        Tuple of (diff_string, was_truncated, total_lines)
    """
    lines = content.splitlines()
    total_lines = len(lines) + 4  # +4 for diff header lines

    # Build diff header
    diff_parts = [
        f"--- a/{file_path}",
        f"+++ /dev/null",
        f"@@ -1,{len(lines)} +0,0 @@"
    ]

    # Add content lines with - prefix
    for line in lines:
        diff_parts.append(f"-{line}")

    truncated = False
    if max_lines is not None and len(diff_parts) > max_lines:
        diff_parts = diff_parts[:max_lines]
        truncated = True

    diff_text = "\n".join(diff_parts)
    return diff_text, truncated, total_lines


def get_diff_stats(old_content: str, new_content: str) -> dict:
    """Get statistics about the diff between two contents.

    Args:
        old_content: Original content
        new_content: New content

    Returns:
        Dict with stats: lines_added, lines_removed, lines_changed
    """
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    # Use SequenceMatcher for more detailed stats
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    lines_added = 0
    lines_removed = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'insert':
            lines_added += j2 - j1
        elif tag == 'delete':
            lines_removed += i2 - i1
        elif tag == 'replace':
            lines_removed += i2 - i1
            lines_added += j2 - j1

    return {
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "old_total": len(old_lines),
        "new_total": len(new_lines),
    }


def summarize_diff(old_content: str, new_content: str, file_path: str) -> str:
    """Generate a brief summary of changes.

    Args:
        old_content: Original content
        new_content: New content
        file_path: Path to the file

    Returns:
        Human-readable summary string
    """
    stats = get_diff_stats(old_content, new_content)

    parts = []
    if stats["lines_added"] > 0:
        parts.append(f"+{stats['lines_added']}")
    if stats["lines_removed"] > 0:
        parts.append(f"-{stats['lines_removed']}")

    if parts:
        change_str = ", ".join(parts)
        return f"Update {file_path} ({change_str} lines)"
    else:
        return f"Update {file_path} (no line changes)"
