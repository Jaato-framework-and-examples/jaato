"""Console encoding utilities for cross-platform Unicode support.

Windows consoles (cmd, PowerShell) use code pages like cp1252 by default,
which cannot encode certain Unicode characters (emojis, checkmarks, etc.).
This module provides utilities to configure UTF-8 encoding for console output.

Git Bash (mintty/MSYS2) on Windows requires additional handling as it operates
in a POSIX-like environment that may not respect Windows console encoding settings.
"""

import os
import sys


def _is_git_bash() -> bool:
    """Detect if running under Git Bash (mintty/MSYS2) on Windows.

    Git Bash sets MSYSTEM to indicate the MSYS2 environment type:
    - MINGW64: 64-bit MinGW
    - MINGW32: 32-bit MinGW
    - MSYS: MSYS2 environment

    Additionally, mintty (Git Bash's terminal) sets TERM_PROGRAM.
    """
    if sys.platform != 'win32':
        return False

    # MSYSTEM is set by Git Bash/MSYS2 environments
    msystem = os.environ.get('MSYSTEM', '')
    if msystem in ('MINGW64', 'MINGW32', 'MSYS', 'UCRT64', 'CLANG64', 'CLANGARM64'):
        return True

    # Also check for mintty terminal (Git Bash default terminal)
    if os.environ.get('TERM_PROGRAM') == 'mintty':
        return True

    return False


def configure_utf8_output() -> None:
    """Configure stdout and stderr to use UTF-8 encoding with error handling.

    This is primarily needed on Windows where the console encoding defaults
    to a code page (e.g., cp1252) that cannot handle all Unicode characters.

    Uses 'replace' error handling to substitute unencodable characters with
    a replacement character (?) rather than raising UnicodeEncodeError.

    Also sets environment variables to ensure subprocesses use UTF-8 encoding.

    Safe to call on any platform - does nothing if already configured or
    if reconfiguration is not supported.
    """
    if sys.platform == 'win32':
        # Set environment variables early to affect subprocesses
        # PYTHONUTF8=1 enables Python UTF-8 mode (PEP 540) for child processes
        # PYTHONIOENCODING sets encoding for stdin/stdout/stderr
        os.environ.setdefault('PYTHONUTF8', '1')
        os.environ.setdefault('PYTHONIOENCODING', 'utf-8:replace')

        # Git Bash (mintty/MSYS2) requires additional locale settings
        # to properly handle UTF-8 encoding in its POSIX-like environment
        if _is_git_bash():
            # Set POSIX locale variables for UTF-8
            # These are respected by mintty and MSYS2 applications
            os.environ.setdefault('LANG', 'en_US.UTF-8')
            os.environ.setdefault('LC_ALL', 'en_US.UTF-8')
            # Ensure consistent character width calculation
            os.environ.setdefault('LC_CTYPE', 'en_US.UTF-8')

        try:
            # Python 3.7+ supports reconfigure() on text streams
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            # If reconfiguration fails, the environment variables above
            # provide a fallback for subprocess operations
            pass
