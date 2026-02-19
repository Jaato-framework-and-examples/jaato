# shared/plugins/diff_formatter/word_diff.py
"""Word-level (character-level) diff computation.

Finds the specific characters that changed between two lines,
enabling more precise highlighting of modifications.
"""

import difflib
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WordDiff:
    """Character-level changes within a line.

    Each segment is a tuple of (text, is_changed) where is_changed
    indicates whether this segment was modified.
    """
    old_segments: List[Tuple[str, bool]]  # (text, is_changed)
    new_segments: List[Tuple[str, bool]]  # (text, is_changed)

    @property
    def has_changes(self) -> bool:
        """Check if there are any highlighted changes."""
        return any(changed for _, changed in self.old_segments) or \
               any(changed for _, changed in self.new_segments)


def compute_word_diff(old_line: str, new_line: str) -> WordDiff:
    """Find character-level differences between two lines.

    Uses difflib.SequenceMatcher to find matching blocks,
    then marks non-matching regions as changed.

    Args:
        old_line: Original line content.
        new_line: New line content.

    Returns:
        WordDiff with segments marked as changed or unchanged.
    """
    matcher = difflib.SequenceMatcher(None, old_line, new_line, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()

    old_segments: List[Tuple[str, bool]] = []
    new_segments: List[Tuple[str, bool]] = []

    old_pos = 0
    new_pos = 0

    for match in matching_blocks:
        old_start, new_start, size = match.a, match.b, match.size

        # Add changed segment before this match (if any)
        if old_pos < old_start:
            old_segments.append((old_line[old_pos:old_start], True))
        if new_pos < new_start:
            new_segments.append((new_line[new_pos:new_start], True))

        # Add the matching (unchanged) segment
        if size > 0:
            old_segments.append((old_line[old_start:old_start + size], False))
            new_segments.append((new_line[new_start:new_start + size], False))

        old_pos = old_start + size
        new_pos = new_start + size

    return WordDiff(old_segments=old_segments, new_segments=new_segments)


def compute_word_diff_by_words(old_line: str, new_line: str) -> WordDiff:
    """Find word-level differences (splits on whitespace first).

    This provides coarser granularity than character-level,
    which can be less noisy for lines with many small changes.

    Args:
        old_line: Original line content.
        new_line: New line content.

    Returns:
        WordDiff with segments marked as changed or unchanged.
    """
    old_words = _tokenize_with_whitespace(old_line)
    new_words = _tokenize_with_whitespace(new_line)

    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()

    old_segments: List[Tuple[str, bool]] = []
    new_segments: List[Tuple[str, bool]] = []

    old_pos = 0
    new_pos = 0

    for match in matching_blocks:
        old_start, new_start, size = match.a, match.b, match.size

        # Add changed words before this match
        if old_pos < old_start:
            changed_text = "".join(old_words[old_pos:old_start])
            if changed_text:
                old_segments.append((changed_text, True))
        if new_pos < new_start:
            changed_text = "".join(new_words[new_pos:new_start])
            if changed_text:
                new_segments.append((changed_text, True))

        # Add matching words
        if size > 0:
            matched_text = "".join(old_words[old_start:old_start + size])
            old_segments.append((matched_text, False))
            new_segments.append((matched_text, False))

        old_pos = old_start + size
        new_pos = new_start + size

    return WordDiff(old_segments=old_segments, new_segments=new_segments)


def _tokenize_with_whitespace(text: str) -> List[str]:
    """Split text into words while preserving whitespace as separate tokens.

    Example: "hello  world" -> ["hello", "  ", "world"]
    """
    tokens = []
    current = []
    in_whitespace = False

    for char in text:
        is_ws = char.isspace()
        if is_ws != in_whitespace:
            if current:
                tokens.append("".join(current))
                current = []
            in_whitespace = is_ws
        current.append(char)

    if current:
        tokens.append("".join(current))

    return tokens


def render_word_diff_old(
    word_diff: WordDiff,
    color_changed: str,
    color_reset: str,
) -> str:
    """Render the old line with changed portions highlighted.

    Args:
        word_diff: Computed word diff.
        color_changed: ANSI code for changed text.
        color_reset: ANSI reset code.

    Returns:
        String with ANSI codes for highlighting.
    """
    parts = []
    for text, is_changed in word_diff.old_segments:
        if is_changed:
            parts.append(f"{color_changed}{text}{color_reset}")
        else:
            parts.append(text)
    return "".join(parts)


def render_word_diff_new(
    word_diff: WordDiff,
    color_changed: str,
    color_reset: str,
) -> str:
    """Render the new line with changed portions highlighted.

    Args:
        word_diff: Computed word diff.
        color_changed: ANSI code for changed text.
        color_reset: ANSI reset code.

    Returns:
        String with ANSI codes for highlighting.
    """
    parts = []
    for text, is_changed in word_diff.new_segments:
        if is_changed:
            parts.append(f"{color_changed}{text}{color_reset}")
        else:
            parts.append(text)
    return "".join(parts)
