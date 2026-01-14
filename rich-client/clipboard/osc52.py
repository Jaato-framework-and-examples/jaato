"""OSC 52 clipboard provider."""

import base64
import os
import sys

# OSC 52 size limit (base64 encoded) - some terminals cap at ~74KB
OSC52_MAX_BYTES = 74994


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

        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")

        # Truncate if too large
        if len(encoded) > OSC52_MAX_BYTES:
            # Re-encode truncated text
            max_text_bytes = (OSC52_MAX_BYTES * 3) // 4  # Approximate
            truncated = text.encode("utf-8")[:max_text_bytes].decode("utf-8", errors="ignore")
            encoded = base64.b64encode(truncated.encode("utf-8")).decode("ascii")

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
