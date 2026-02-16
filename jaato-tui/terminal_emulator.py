"""Terminal emulator for tool output processing.

Uses pyte to interpret terminal control sequences (cursor movement,
line erase, carriage return, etc.) and produce clean visual output
with ANSI SGR color/style codes preserved.
"""

import pyte
from typing import Dict, List, Optional, Tuple


# Map pyte named colors to SGR codes
_FG_CODES: Dict[str, str] = {
    "black": "30", "red": "31", "green": "32", "brown": "33",
    "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
    "brightblack": "90", "brightred": "91", "brightgreen": "92",
    "brightbrown": "93", "brightblue": "94", "brightmagenta": "95",
    "brightcyan": "96", "brightwhite": "97",
}
_BG_CODES: Dict[str, str] = {
    "black": "40", "red": "41", "green": "42", "brown": "43",
    "blue": "44", "magenta": "45", "cyan": "46", "white": "47",
    "brightblack": "100", "brightred": "101", "brightgreen": "102",
    "brightbrown": "103", "brightblue": "104", "brightmagenta": "105",
    "brightcyan": "106", "brightwhite": "107",
}


def _char_style_codes(char: pyte.screens.Char) -> List[str]:
    """Extract SGR parameter codes from a pyte Char's attributes."""
    codes: List[str] = []

    if char.bold:
        codes.append("1")
    if char.italics:
        codes.append("3")
    if char.underscore:
        codes.append("4")
    if char.blink:
        codes.append("5")
    if char.reverse:
        codes.append("7")
    if char.strikethrough:
        codes.append("9")

    # Foreground color
    fg = char.fg
    if fg and fg != "default":
        fg_code = _FG_CODES.get(fg)
        if fg_code:
            codes.append(fg_code)
        elif len(fg) == 6:
            # Hex color (from 256-color conversion by pyte) → 24-bit true color
            r, g, b = int(fg[0:2], 16), int(fg[2:4], 16), int(fg[4:6], 16)
            codes.append(f"38;2;{r};{g};{b}")

    # Background color
    bg = char.bg
    if bg and bg != "default":
        bg_code = _BG_CODES.get(bg)
        if bg_code:
            codes.append(bg_code)
        elif len(bg) == 6:
            r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
            codes.append(f"48;2;{r};{g};{b}")

    return codes


def _render_buffer_row(row_data: Dict[int, pyte.screens.Char], columns: int) -> str:
    """Render a pyte screen buffer row to a string with ANSI SGR codes.

    Tracks style changes between adjacent characters to emit minimal
    SGR escape sequences.

    Args:
        row_data: Dict mapping column index to Char (from screen.buffer[row]).
        columns: Number of columns in the screen.

    Returns:
        String with ANSI SGR codes for colors/styles, trailing spaces stripped.
    """
    default_char = pyte.screens.Char(" ")
    parts: List[str] = []
    prev_codes: List[str] = []

    # Find last non-space column to avoid trailing spaces
    last_content_col = -1
    for col in range(columns - 1, -1, -1):
        char = row_data.get(col, default_char)
        if char.data != " " or _char_style_codes(char):
            last_content_col = col
            break

    if last_content_col < 0:
        return ""

    for col in range(last_content_col + 1):
        char = row_data.get(col, default_char)
        codes = _char_style_codes(char)

        if codes != prev_codes:
            # Style changed — emit reset + new codes
            if prev_codes:
                parts.append("\x1b[0m")
            if codes:
                parts.append(f"\x1b[{';'.join(codes)}m")
            prev_codes = codes

        parts.append(char.data)

    # Reset at end if we had active styling
    if prev_codes:
        parts.append("\x1b[0m")

    result = "".join(parts)
    # Strip trailing uncolored spaces
    return result.rstrip() if not prev_codes else result


def _render_history_line(line_data: Dict[int, pyte.screens.Char]) -> str:
    """Render a pyte history line to a string with ANSI SGR codes.

    History lines are StaticDefaultDict with same structure as buffer rows.

    Args:
        line_data: Dict mapping column index to Char (from history.top entry).

    Returns:
        String with ANSI SGR codes for colors/styles.
    """
    if not line_data:
        return ""

    max_col = max(line_data.keys()) if line_data else -1
    return _render_buffer_row(line_data, max_col + 1)


class TerminalEmulator:
    """Wraps a pyte HistoryScreen to emulate terminal behavior.

    Interprets all terminal control sequences (cursor movement, line erase,
    carriage return, SGR colors, etc.) and produces clean visual output lines
    with ANSI color codes preserved.

    Usage:
        emulator = TerminalEmulator()
        emulator.feed("hello\\n")
        emulator.feed("progress 50%\\rprogress 100%\\n")
        lines = emulator.get_lines()
        # → ["hello", "progress 100%"]
    """

    def __init__(self, columns: int = 300, lines: int = 100, history: int = 5000):
        """Initialize terminal emulator.

        Args:
            columns: Virtual screen width. Use a large value to avoid
                     premature wrapping (visual truncation happens in the renderer).
            lines: Virtual screen height.
            history: Maximum number of scrollback history lines to retain.
        """
        self._screen = pyte.HistoryScreen(columns, lines, history=history)
        self._screen.set_mode(pyte.modes.LNM)  # \n means \r\n
        self._stream = pyte.Stream(self._screen)
        self._columns = columns
        self._dirty = True  # Whether get_lines() needs to rebuild
        self._cached_lines: List[str] = []

    def feed(self, data: str) -> None:
        """Feed raw output data through the terminal emulator.

        Args:
            data: Raw output string, may contain ANSI escape sequences,
                  carriage returns, cursor movement codes, etc.
        """
        self._stream.feed(data)
        self._dirty = True

    def get_lines(self) -> List[str]:
        """Get all output lines (history + current screen content).

        Returns lines with ANSI SGR codes for colors/styles preserved.
        Cursor movement and erase sequences have been interpreted and
        applied to produce the correct visual result.

        Returns:
            List of output lines with ANSI SGR codes.
        """
        if not self._dirty:
            return self._cached_lines

        result: List[str] = []

        # History lines (scrolled off the top of the screen)
        for line_data in self._screen.history.top:
            result.append(_render_history_line(line_data))

        # Current screen lines (strip trailing empty lines)
        screen_lines: List[str] = []
        for row in range(self._screen.lines):
            screen_lines.append(
                _render_buffer_row(self._screen.buffer[row], self._columns)
            )

        # Strip trailing empty lines from screen
        while screen_lines and not screen_lines[-1]:
            screen_lines.pop()

        result.extend(screen_lines)
        self._cached_lines = result
        self._dirty = False
        return result
