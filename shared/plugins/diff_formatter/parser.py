# shared/plugins/diff_formatter/parser.py
"""Parser for unified diff format to structured representation.

Converts standard unified diff output into a structured ParsedDiff
object that can be rendered in different formats (side-by-side,
compact, unified).
"""

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


@dataclass
class DiffStats:
    """Statistics about a diff."""
    added: int = 0
    deleted: int = 0
    modified: int = 0  # Lines that have both old and new versions

    @property
    def total_changes(self) -> int:
        return self.added + self.deleted + self.modified

    def __str__(self) -> str:
        parts = []
        if self.added:
            parts.append(f"+{self.added}")
        if self.deleted:
            parts.append(f"-{self.deleted}")
        if self.modified:
            parts.append(f"~{self.modified}")
        return ", ".join(parts) if parts else "no changes"


ChangeType = Literal["unchanged", "added", "deleted", "modified"]


@dataclass
class DiffLine:
    """A single line from the diff with metadata."""
    content: str
    old_line_no: Optional[int]  # None if addition
    new_line_no: Optional[int]  # None if deletion
    change_type: ChangeType
    paired_with: Optional["DiffLine"] = None  # For modified lines, the other version

    @property
    def line_no_display(self) -> Tuple[str, str]:
        """Return (old_line_no, new_line_no) as strings for display."""
        old = str(self.old_line_no) if self.old_line_no is not None else ""
        new = str(self.new_line_no) if self.new_line_no is not None else ""
        return old, new


@dataclass
class DiffHunk:
    """A hunk of changes with context."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[DiffLine] = field(default_factory=list)
    header_extra: str = ""  # Optional text after @@ markers (e.g., function name)


@dataclass
class ParsedDiff:
    """Complete parsed diff for a file."""
    old_path: str
    new_path: str
    hunks: List[DiffHunk] = field(default_factory=list)

    @property
    def stats(self) -> DiffStats:
        """Calculate statistics from hunks."""
        stats = DiffStats()
        for hunk in self.hunks:
            for line in hunk.lines:
                if line.change_type == "added":
                    stats.added += 1
                elif line.change_type == "deleted":
                    stats.deleted += 1
                elif line.change_type == "modified" and line.old_line_no is not None:
                    # Only count modified once (from the old line)
                    stats.modified += 1
        return stats

    @property
    def is_new_file(self) -> bool:
        """Check if this diff represents a new file."""
        return self.old_path == "/dev/null"

    @property
    def is_deleted_file(self) -> bool:
        """Check if this diff represents a deleted file."""
        return self.new_path == "/dev/null"

    @property
    def display_path(self) -> str:
        """Get the most relevant path for display."""
        if self.is_new_file:
            return self.new_path.lstrip("b/")
        return self.old_path.lstrip("a/")


# Regex patterns for parsing unified diff
DIFF_HEADER_OLD = re.compile(r"^--- (.+?)(?:\t.*)?$")
DIFF_HEADER_NEW = re.compile(r"^\+\+\+ (.+?)(?:\t.*)?$")
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """Parse unified diff text into structured form.

    Args:
        diff_text: Standard unified diff output.

    Returns:
        ParsedDiff with hunks and line information.
    """
    lines = diff_text.split("\n")

    old_path = ""
    new_path = ""
    hunks: List[DiffHunk] = []
    current_hunk: Optional[DiffHunk] = None

    old_line_no = 0
    new_line_no = 0

    for line in lines:
        # Check for file headers
        old_match = DIFF_HEADER_OLD.match(line)
        if old_match:
            old_path = old_match.group(1)
            continue

        new_match = DIFF_HEADER_NEW.match(line)
        if new_match:
            new_path = new_match.group(1)
            continue

        # Check for hunk header
        hunk_match = HUNK_HEADER.match(line)
        if hunk_match:
            # Save previous hunk
            if current_hunk:
                _pair_modified_lines(current_hunk)
                hunks.append(current_hunk)

            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2) or 1)
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4) or 1)
            header_extra = hunk_match.group(5).strip()

            current_hunk = DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                header_extra=header_extra,
            )
            old_line_no = old_start
            new_line_no = new_start
            continue

        # Skip if we're not in a hunk
        if current_hunk is None:
            continue

        # Parse content lines
        if line.startswith("+"):
            current_hunk.lines.append(DiffLine(
                content=line[1:],
                old_line_no=None,
                new_line_no=new_line_no,
                change_type="added",
            ))
            new_line_no += 1
        elif line.startswith("-"):
            current_hunk.lines.append(DiffLine(
                content=line[1:],
                old_line_no=old_line_no,
                new_line_no=None,
                change_type="deleted",
            ))
            old_line_no += 1
        elif line.startswith(" ") or line == "":
            # Context line (or empty line which is context)
            content = line[1:] if line.startswith(" ") else ""
            current_hunk.lines.append(DiffLine(
                content=content,
                old_line_no=old_line_no,
                new_line_no=new_line_no,
                change_type="unchanged",
            ))
            old_line_no += 1
            new_line_no += 1

    # Don't forget the last hunk
    if current_hunk:
        _pair_modified_lines(current_hunk)
        hunks.append(current_hunk)

    return ParsedDiff(
        old_path=old_path,
        new_path=new_path,
        hunks=hunks,
    )


def _pair_modified_lines(hunk: DiffHunk) -> None:
    """Identify and pair modified lines (consecutive delete+add).

    When a line is modified (not purely added or deleted), it appears
    as a deletion followed by an addition. This function pairs them
    and marks them as "modified" for better rendering.
    """
    i = 0
    while i < len(hunk.lines):
        line = hunk.lines[i]

        # Look for deletion followed by addition(s)
        if line.change_type == "deleted":
            # Collect consecutive deletions
            deletions = []
            j = i
            while j < len(hunk.lines) and hunk.lines[j].change_type == "deleted":
                deletions.append(j)
                j += 1

            # Collect consecutive additions after deletions
            additions = []
            while j < len(hunk.lines) and hunk.lines[j].change_type == "added":
                additions.append(j)
                j += 1

            # Pair them up (min of both counts)
            pairs = min(len(deletions), len(additions))
            for k in range(pairs):
                del_idx = deletions[k]
                add_idx = additions[k]

                hunk.lines[del_idx].change_type = "modified"
                hunk.lines[add_idx].change_type = "modified"
                hunk.lines[del_idx].paired_with = hunk.lines[add_idx]
                hunk.lines[add_idx].paired_with = hunk.lines[del_idx]

            i = j
        else:
            i += 1


def get_paired_lines(hunk: DiffHunk) -> List[Tuple[Optional[DiffLine], Optional[DiffLine]]]:
    """Convert hunk lines to paired format for side-by-side display.

    Returns a list of (old_line, new_line) tuples where:
    - Unchanged: (line, line) - same line on both sides
    - Added: (None, line) - only on new side
    - Deleted: (line, None) - only on old side
    - Modified: (old_line, new_line) - different content on each side

    Args:
        hunk: A DiffHunk with lines.

    Returns:
        List of (old_line, new_line) pairs for rendering.
    """
    pairs: List[Tuple[Optional[DiffLine], Optional[DiffLine]]] = []

    i = 0
    while i < len(hunk.lines):
        line = hunk.lines[i]

        if line.change_type == "unchanged":
            pairs.append((line, line))
            i += 1
        elif line.change_type == "added":
            pairs.append((None, line))
            i += 1
        elif line.change_type == "deleted":
            pairs.append((line, None))
            i += 1
        elif line.change_type == "modified":
            if line.old_line_no is not None:
                # This is the old version, pair with its partner
                pairs.append((line, line.paired_with))
                i += 1
                # Skip the paired line (it will be the next one with new_line_no)
                if i < len(hunk.lines) and hunk.lines[i] == line.paired_with:
                    i += 1
            else:
                # This is the new version without a pair (shouldn't happen normally)
                pairs.append((None, line))
                i += 1
        else:
            i += 1

    return pairs
