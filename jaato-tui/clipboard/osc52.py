"""OSC 52 clipboard provider."""

import base64
import os
import sys

# OSC 52 size limit (base64 encoded) - some terminals cap at ~74KB
OSC52_MAX_BYTES = 74994

# Screen DCS passthrough is fragile with large payloads - the outer terminal
# may fail to parse the sequence and render raw base64 as visible text.
# 16KB base64 (~12KB text) is a safe practical limit.
OSC52_SCREEN_MAX_BYTES = 16384


def _truncate_utf8_safe(text: str, max_bytes: int) -> str:
    """Truncate text to fit within max_bytes when UTF-8 encoded.

    Ensures truncation doesn't corrupt UTF-8 by splitting multi-byte
    characters, while spending every byte of the budget that a WHOLE
    character fits into.  A cut that lands exactly on a character
    boundary keeps that character; only a genuinely split sequence is
    dropped.

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

    # Walk back to the lead byte of the FINAL sequence.  Continuation
    # bytes are 10xxxxxx; a lead byte is not.
    i = len(truncated) - 1
    while i >= 0 and (truncated[i] & 0xC0) == 0x80:
        i -= 1

    # Drop that final sequence ONLY if the cut actually split it.  The
    # pre-fix code stripped the continuation bytes and then the lead
    # byte unconditionally, so a cut landing exactly on a character
    # boundary still lost the whole last character: `"日本語"` at 6
    # bytes returned `"日"`, spending 3 of its 6 permitted bytes on
    # nothing.  Silently lossy, at a size boundary, for non-ASCII users
    # only -- which is why it shipped green (#736).
    if i >= 0:
        lead = truncated[i]
        need = (
            4 if lead >= 0xF0
            else 3 if lead >= 0xE0
            else 2 if lead >= 0xC0
            else 1
        )
        if len(truncated) - i < need:
            truncated = truncated[:i]

    return truncated.decode("utf-8") if truncated else ""


class OSC52Provider:
    """Clipboard provider using OSC 52 escape sequence.

    OSC 52 allows terminal applications to write directly to the system
    clipboard. Works over SSH, requires no external dependencies.

    Supported terminals: iTerm2, Alacritty, kitty, Windows Terminal,
    tmux (with set-clipboard on), and others.

    Note: macOS Terminal.app does not support OSC 52.

    tmux handling:
        When running inside tmux, we send the raw OSC 52 sequence without
        DCS passthrough wrapping. tmux natively intercepts OSC 52 via its
        ``set-clipboard`` option (defaults to ``on`` or ``external`` since
        tmux 1.8/3.2). tmux stores the content in its paste buffer and
        re-emits its own OSC 52 to the outer terminal. This avoids the
        ~12 KB size limit imposed by DCS passthrough fragility and allows
        copying the same amount of text as a standard terminal (~56 KB).

    screen handling:
        GNU screen still requires DCS passthrough wrapping with a
        conservative 16 KB limit, as it lacks native OSC 52 interception.
    """

    def __init__(self):
        self._in_tmux = bool(os.environ.get("TMUX"))
        # Detect screen, but not when TMUX is also set (tmux sets TERM=screen*)
        self._in_screen = (
            not self._in_tmux and "screen" in os.environ.get("TERM", "")
        )

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

        # Calculate effective limit.
        # tmux: standard limit — tmux handles OSC 52 natively via set-clipboard,
        #   no DCS passthrough needed, so no fragility-related size constraint.
        # screen: conservative limit — requires DCS passthrough which is fragile.
        if self._in_screen:
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

        # Wrap for screen passthrough if needed.
        # tmux: no wrapping — tmux natively intercepts OSC 52 when
        #   set-clipboard is on/external and forwards to the outer terminal.
        if self._in_screen:
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
