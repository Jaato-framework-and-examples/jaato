"""OSC 52 clipboard provider."""

import base64
import os
import sys

# OSC 52 size limit (base64 encoded) - some terminals cap at ~74KB
OSC52_MAX_BYTES = 74994

# Conservative limit for tmux passthrough to avoid terminal parsing failures.
# tmux's OSC52 passthrough is fragile with large payloads - the outer terminal
# may fail to parse the sequence and render raw base64 as visible text.
# 16KB base64 (~12KB text) is a safe practical limit.
OSC52_TMUX_MAX_BYTES = 16384

# Screen has similar issues
OSC52_SCREEN_MAX_BYTES = 16384


def _truncate_utf8_safe(text: str, max_bytes: int) -> str:
    """Truncate text to fit within max_bytes when UTF-8 encoded.

    Ensures truncation doesn't corrupt UTF-8 by splitting multi-byte characters.
    Uses a safe approach that backs up past any incomplete sequences.

    Args:
        text: The text to truncate.
        max_bytes: Maximum bytes for the UTF-8 encoded result.

    Returns:
        Truncated text that fits within max_bytes when UTF-8 encoded.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    # Truncate at byte boundary
    truncated = encoded[:max_bytes]

    # Back up past any UTF-8 continuation bytes (10xxxxxx pattern)
    # This handles the case where we cut in the middle of a multi-byte char
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]

    # If last byte is a multi-byte start (11xxxxxx), check if sequence is complete
    if truncated:
        last = truncated[-1]
        if last >= 0xF0:  # 4-byte sequence start
            # Would need 3 continuation bytes, but we backed up past them
            truncated = truncated[:-1]
        elif last >= 0xE0:  # 3-byte sequence start
            truncated = truncated[:-1]
        elif last >= 0xC0:  # 2-byte sequence start
            truncated = truncated[:-1]

    return truncated.decode("utf-8") if truncated else ""


class OSC52Provider:
    """Clipboard provider using OSC 52 escape sequence.

    OSC 52 allows terminal applications to write directly to the system
    clipboard. Works over SSH, requires no external dependencies.

    Supported terminals: iTerm2, Alacritty, kitty, Windows Terminal,
    tmux (with set-clipboard on), and others.

    Note: macOS Terminal.app does not support OSC 52.
    """

    def __init__(self):
        self._in_tmux = bool(os.environ.get("TMUX"))
        self._in_screen = "screen" in os.environ.get("TERM", "")

    @property
    def name(self) -> str:
        if self._in_tmux:
            return "OSC 52 (tmux)"
        elif self._in_screen:
            return "OSC 52 (screen)"
        return "OSC 52"

    def copy(self, text: str) -> bool:
        """Copy text via OSC 52.

        Args:
            text: The text to copy.

        Returns:
            True (fire-and-forget, cannot detect actual success).
        """
        if not text:
            return False

        # Calculate effective limit - use conservative limits for tmux/screen
        # to avoid terminal parsing failures with large sequences
        if self._in_tmux:
            max_encoded = OSC52_TMUX_MAX_BYTES
        elif self._in_screen:
            max_encoded = OSC52_SCREEN_MAX_BYTES
        else:
            max_encoded = OSC52_MAX_BYTES

        # Formula: (max_encoded // 4) * 3 ensures base64 output never exceeds limit
        max_text_bytes = (max_encoded // 4) * 3

        # Truncate if needed, preserving UTF-8 character boundaries
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > max_text_bytes:
            text = _truncate_utf8_safe(text, max_text_bytes)

        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")

        # OSC 52: ESC ] 52 ; c ; <base64> ST
        # c = clipboard, ST = string terminator (ESC \ or BEL)
        # Use ESC \ as terminator for better compatibility
        osc52_seq = f"\x1b]52;c;{encoded}\x1b\\"

        # Wrap for tmux/screen passthrough if needed
        if self._in_tmux:
            # tmux passthrough: ESC Ptmux; ESC <seq> ESC \
            # Double any ESC in the sequence for tmux
            inner = osc52_seq.replace("\x1b", "\x1b\x1b")
            osc52_seq = f"\x1bPtmux;{inner}\x1b\\"
        elif self._in_screen:
            # screen passthrough: ESC P <seq> ESC \
            osc52_seq = f"\x1bP{osc52_seq}\x1b\\"

        # Write directly to TTY to bypass prompt_toolkit's stdout capture
        try:
            with open("/dev/tty", "w") as tty:
                tty.write(osc52_seq)
                tty.flush()
        except (OSError, IOError):
            # Fallback to stdout if /dev/tty unavailable (e.g., Windows)
            sys.stdout.write(osc52_seq)
            sys.stdout.flush()

        return True
