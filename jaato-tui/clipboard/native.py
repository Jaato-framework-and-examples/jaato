"""Native clipboard provider using system tools."""

import os
import shutil
import subprocess
import sys


class NativeProvider:
    """Clipboard provider using native system clipboard tools.

    Supported tools (in order of preference):
    - Linux/BSD: wl-copy (Wayland), xclip, xsel (X11)
    - macOS: pbcopy
    - Windows: clip.exe
    """

    def __init__(self):
        self._tool = self._detect_tool()

    @property
    def name(self) -> str:
        return f"Native ({self._tool[0] if self._tool else 'unavailable'})"

    @property
    def available(self) -> bool:
        """Check if a native clipboard tool is available."""
        return self._tool is not None

    def _detect_tool(self) -> tuple[str, list[str]] | None:
        """Detect available clipboard tool.

        Returns:
            Tuple of (tool_name, command_args) or None if unavailable.
        """
        if sys.platform == "darwin":
            if shutil.which("pbcopy"):
                return ("pbcopy", ["pbcopy"])
        elif sys.platform == "win32":
            if shutil.which("clip"):
                return ("clip", ["clip"])
        else:
            # Linux/BSD - detect session type to pick the right tool
            session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

            if session_type == "wayland":
                if shutil.which("wl-copy"):
                    return ("wl-copy", ["wl-copy"])

            # X11 or unknown session - prefer X11 tools
            if session_type == "x11" or os.environ.get("DISPLAY"):
                if shutil.which("xclip"):
                    return ("xclip", ["xclip", "-selection", "clipboard"])
                if shutil.which("xsel"):
                    return ("xsel", ["xsel", "--clipboard", "--input"])

            # Fallback: try any available tool
            if shutil.which("wl-copy"):
                return ("wl-copy", ["wl-copy"])
            if shutil.which("xclip"):
                return ("xclip", ["xclip", "-selection", "clipboard"])
            if shutil.which("xsel"):
                return ("xsel", ["xsel", "--clipboard", "--input"])
        return None

    def copy(self, text: str) -> bool:
        """Copy text using native clipboard tool.

        Args:
            text: The text to copy.

        Returns:
            True if successful, False otherwise.
        """
        if not text or not self._tool:
            return False

        try:
            proc = subprocess.Popen(
                self._tool[1],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = proc.communicate(input=text.encode("utf-8"), timeout=5)
            if proc.returncode != 0 and stderr:
                # Store error for debugging
                self._last_error = stderr.decode("utf-8", errors="replace").strip()
            else:
                self._last_error = None
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError, IOError) as e:
            self._last_error = str(e)
            return False
