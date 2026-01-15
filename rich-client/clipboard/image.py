"""Image clipboard support using platform-specific tools."""

import platform
import subprocess
from pathlib import Path


def copy_image_to_clipboard(path: str) -> bool:
    """Copy an image file to the system clipboard.

    Uses platform-specific tools:
    - Linux: xclip (falls back to xsel)
    - macOS: osascript
    - Windows: PowerShell with System.Windows.Forms

    Args:
        path: Path to the image file (PNG recommended).

    Returns:
        True if copy succeeded, False otherwise.
    """
    path = str(Path(path).resolve())
    system = platform.system()

    if system == "Linux":
        return _copy_linux(path)
    elif system == "Darwin":
        return _copy_macos(path)
    elif system == "Windows":
        return _copy_windows(path)

    return False


def _copy_linux(path: str) -> bool:
    """Copy image to clipboard on Linux using xclip or xsel."""
    # Try xclip first (more common)
    try:
        with open(path, "rb") as f:
            subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png"],
                stdin=f,
                check=True,
                capture_output=True,
            )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Fall back to xsel
    try:
        with open(path, "rb") as f:
            subprocess.run(
                ["xsel", "--clipboard", "--input"],
                stdin=f,
                check=True,
                capture_output=True,
            )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return False


def _copy_macos(path: str) -> bool:
    """Copy image to clipboard on macOS using osascript."""
    # Escape path for AppleScript
    escaped_path = path.replace('"', '\\"')
    script = f'set the clipboard to (read (POSIX file "{escaped_path}") as «class PNGf»)'

    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return False


def _copy_windows(path: str) -> bool:
    """Copy image to clipboard on Windows using PowerShell."""
    # Escape path for PowerShell
    escaped_path = path.replace("'", "''")
    ps_cmd = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        f"[System.Windows.Forms.Clipboard]::SetImage("
        f"[System.Drawing.Image]::FromFile('{escaped_path}'))"
    )

    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return False
