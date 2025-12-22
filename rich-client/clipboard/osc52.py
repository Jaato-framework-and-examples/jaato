"""OSC 52 clipboard provider."""

import base64
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

    @property
    def name(self) -> str:
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

        # OSC 52: ESC ] 52 ; c ; <base64> BEL
        # c = clipboard, BEL (\x07) = string terminator
        sys.stdout.write(f"\x1b]52;c;{encoded}\x07")
        sys.stdout.flush()

        return True
