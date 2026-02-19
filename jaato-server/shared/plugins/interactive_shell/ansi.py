"""ANSI escape sequence stripping for clean terminal output.

Strips ANSI control sequences (colors, cursor movement, line clearing, etc.)
from PTY output so the model receives clean, readable text.
"""

import re

# Comprehensive ANSI escape sequence pattern:
# - CSI sequences: ESC[ ... final_byte (colors, cursor, erase, scroll, etc.)
# - OSC sequences: ESC] ... ST (title, hyperlinks, etc.)
# - Simple escapes: ESC followed by single char (e.g., ESC M for reverse index)
# - C1 control codes: 0x80-0x9F range (8-bit equivalents)
_ANSI_ESCAPE_RE = re.compile(
    r'(?:'
    r'\x1b'             # ESC character
    r'(?:'
    r'\[[0-9;?]*[A-Za-z~]'  # CSI: ESC [ params final_byte
    r'|\][^\x07\x1b]*(?:\x07|\x1b\\)'  # OSC: ESC ] ... BEL/ST
    r'|\([0-9A-Za-z]'   # Character set designation: ESC ( X
    r'|[A-Za-z=<>]'     # Simple escape: ESC + single char
    r')'
    r'|\r'              # Carriage return (overwrite lines)
    r')'
)

# Backspace-based overwriting: char + BS + char (bold/overprint on old terminals)
_BACKSPACE_OVERWRITE_RE = re.compile(r'[^\x08]\x08')


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences and terminal control codes from text.

    Removes:
    - CSI sequences (colors, cursor movement, erase, etc.)
    - OSC sequences (window title, hyperlinks)
    - Simple escape sequences
    - Carriage returns (line overwriting)
    - Backspace-based overprinting

    Args:
        text: Raw terminal output potentially containing ANSI codes.

    Returns:
        Clean text with all escape sequences removed.
    """
    # Strip ANSI escapes
    result = _ANSI_ESCAPE_RE.sub('', text)

    # Handle backspace overprinting (e.g., man page bold: X\bX)
    while _BACKSPACE_OVERWRITE_RE.search(result):
        result = _BACKSPACE_OVERWRITE_RE.sub('', result)

    return result
