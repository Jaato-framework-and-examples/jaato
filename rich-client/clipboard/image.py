"""Image clipboard support using platform-specific tools."""

import os
import platform
import subprocess
from pathlib import Path


def is_ssh_session() -> bool:
    """Check if running in an SSH session."""
    return bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"))


def copy_image_to_clipboard(path: str) -> tuple[bool, str]:
    """Copy an image file to the system clipboard.

    Uses platform-specific tools:
    - Linux: wl-copy (Wayland), xclip, or xsel (X11)
    - macOS: osascript
    - Windows: PowerShell with System.Windows.Forms

    Args:
        path: Path to the image file (PNG recommended).

    Returns:
        Tuple of (success, error_message). Error message is empty on success.
    """
    # Check for SSH session - clipboard won't work without X11 forwarding
    if is_ssh_session():
        has_display = bool(os.environ.get("DISPLAY"))
        if not has_display:
            return False, "SSH session without X11 forwarding (use 'ssh -X' for clipboard)"

    path = str(Path(path).resolve())
    system = platform.system()

    if system == "Linux":
        if _copy_linux(path):
            return True, ""
        return False, "wl-copy/xclip/xsel not available"
    elif system == "Darwin":
        if _copy_macos(path):
            return True, ""
        return False, "osascript failed"
    elif system == "Windows":
        if _copy_windows(path):
            return True, ""
        return False, "PowerShell clipboard failed"

    return False, f"Unsupported platform: {system}"


def _copy_linux(path: str) -> bool:
    """Copy image to clipboard on Linux using wl-copy (Wayland) or xclip/xsel (X11)."""
    # Try wl-copy first (Wayland) - most modern Linux desktops
    try:
        with open(path, "rb") as f:
            subprocess.run(
                ["wl-copy", "--type", "image/png"],
                stdin=f,
                check=True,
                capture_output=True,
            )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Try xclip (X11)
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

    # Fall back to xsel (X11)
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
