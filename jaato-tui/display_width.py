"""Display width utilities for terminal rendering.

Provides accurate display width measurement for strings containing
wide characters (CJK), ambiguous-width characters (box-drawing), and
zero-width characters.
"""

import os
import unicodedata

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
