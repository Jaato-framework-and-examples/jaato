"""Cross-platform path utilities for MSYS2/Git Bash compatibility.

When running native Windows Python (MinGW) under an MSYS2 or Git Bash shell,
Python's path functions produce Windows-style paths (backslashes, drive letters)
but the shell environment expects Unix-style paths (forward slashes).

MSYS2 path mapping:
    Windows:  C:\\Users\\foo\\project   or  C:/Users/foo/project
    MSYS2:    /c/Users/foo/project

This module provides:
- MSYS2 environment detection
- Path normalization for display and string comparison
- Drive letter conversion: C:/foo <-> /c/foo
- Input normalization: /c/foo -> C:/foo (so Python can open MSYS2-style paths)
- Output normalization: C:/foo -> /c/foo (for display in MSYS2 shell)

Reference: https://www.msys2.org/docs/python/
           https://www.msys2.org/docs/filesystem-paths/
"""

import functools
import os
import re
import sys
from typing import Optional


@functools.lru_cache(maxsize=1)
def is_msys2_environment() -> bool:
    """Detect if running under MSYS2 or Git Bash on Windows.

    Checks for MSYS2 environment indicators:
    - MSYSTEM env var set to MINGW64, MINGW32, MSYS, UCRT64, CLANG64, CLANGARM64
    - TERM_PROGRAM set to mintty (Git Bash default terminal)

    This only returns True when running native Windows Python (sys.platform == 'win32')
    in an MSYS2 shell. MSYS2's own Python (Cygwin-derived) already uses Unix paths
    natively.

    The result is cached since the environment doesn't change during a process's
    lifetime.

    Returns:
        True if running Windows Python under MSYS2/Git Bash.
    """
    if sys.platform != 'win32':
        return False

    # MSYSTEM is set by all MSYS2 environments
    msystem = os.environ.get('MSYSTEM', '')
    if msystem in ('MINGW64', 'MINGW32', 'MSYS', 'UCRT64', 'CLANG64', 'CLANGARM64'):
        return True

    # mintty is Git Bash's default terminal
    if os.environ.get('TERM_PROGRAM') == 'mintty':
        return True

    return False


# Regex matching MSYS2-style drive paths: /c/... or /C/...
# The drive letter must be followed by / or end of string to avoid false positives
# like /config or /cache
_MSYS2_DRIVE_RE = re.compile(r'^/([a-zA-Z])(?:/|$)')

# Regex matching Windows-style drive paths: C:/ or C:\
_WINDOWS_DRIVE_RE = re.compile(r'^([a-zA-Z]):[\\/]')


def msys2_to_windows_path(path: str) -> str:
    """Convert an MSYS2-style path to a Windows path.

    Converts /c/Users/foo -> C:/Users/foo so that Python's file operations
    can find the file. Only converts paths that start with a single drive
    letter (e.g., /c/, /d/) to avoid false positives on paths like /config/.

    This should be applied to user-provided paths that may come from
    the MSYS2 shell, so that Python (which uses Windows APIs) can resolve them.

    Args:
        path: Path string, possibly in MSYS2 format.

    Returns:
        Windows-style path if input was MSYS2 drive path, unchanged otherwise.
    """
    if not path:
        return path

    m = _MSYS2_DRIVE_RE.match(path)
    if m:
        drive = m.group(1).upper()
        rest = path[2:]  # Everything after /c
        return f"{drive}:{rest}" if rest else f"{drive}:/"

    return path


def windows_to_msys2_path(path: str) -> str:
    """Convert a Windows-style path to MSYS2 format for display.

    Converts C:/Users/foo -> /c/Users/foo and also replaces backslashes
    with forward slashes. This makes paths copy-pasteable in the MSYS2 shell.

    Args:
        path: Path string, possibly with Windows drive letter.

    Returns:
        MSYS2-style path if input had a drive letter, unchanged otherwise.
    """
    if not path:
        return path

    # First normalize separators
    path = path.replace('\\', '/')

    m = _WINDOWS_DRIVE_RE.match(path)
    if m:
        drive = m.group(1).lower()
        rest = path[2:]  # Everything after C:
        return f"/{drive}{rest}"

    return path


def normalize_path(path: str) -> str:
    """Normalize a Windows path for display under MSYS2.

    Under MSYS2, performs two conversions:
    1. Backslashes to forward slashes: C:\\foo -> C:/foo
    2. Drive letter to MSYS2 mount: C:/foo -> /c/foo

    On other platforms, returns the path unchanged.

    This is primarily for paths that will be:
    - Displayed to the user in tool output
    - Returned as part of structured tool results

    The resulting /c/foo format is the native MSYS2 representation and can
    be directly copy-pasted into the MSYS2 shell.

    Note: For string comparison, use normalize_for_comparison() instead,
    which only normalizes separators without drive letter conversion.

    Args:
        path: Path string to normalize.

    Returns:
        MSYS2-style path (/c/foo) under MSYS2, unchanged otherwise.
    """
    if not path:
        return path

    if is_msys2_environment():
        return windows_to_msys2_path(path)

    return path


def normalize_for_comparison(path: str) -> str:
    """Normalize path for consistent string comparison.

    On Windows (including MSYS2), converts backslashes to forward slashes
    to ensure that prefix-matching and equality checks work correctly
    regardless of which separator was used to construct the path.

    This is more aggressive than normalize_path() - it normalizes on ALL
    Windows systems (not just MSYS2), because mixed separators can cause
    comparison failures even on standard Windows.

    Args:
        path: Path string to normalize for comparison.

    Returns:
        Path with consistent forward-slash separators on Windows,
        unchanged on Unix.
    """
    if not path:
        return path

    if sys.platform == 'win32':
        return path.replace('\\', '/')

    return path


def normalized_startswith(path: str, prefix: str) -> bool:
    """Check if path starts with prefix, handling mixed separators on Windows.

    On Windows, normalizes both path and prefix to forward slashes before
    comparing. On Unix, performs a simple startswith check.

    Args:
        path: The path to check.
        prefix: The prefix to match against.

    Returns:
        True if the normalized path starts with the normalized prefix.
    """
    return normalize_for_comparison(path).startswith(normalize_for_comparison(prefix))


def normalized_equals(path1: str, path2: str) -> bool:
    """Check if two paths are equal, handling mixed separators on Windows.

    On Windows, normalizes both paths to forward slashes before comparing.
    On Unix, performs a simple equality check.

    Args:
        path1: First path.
        path2: Second path.

    Returns:
        True if the normalized paths are equal.
    """
    return normalize_for_comparison(path1) == normalize_for_comparison(path2)


def normalize_result_path(path: str) -> str:
    """Normalize a path for inclusion in tool results.

    Under MSYS2, converts Windows paths to MSYS2 format:
      C:\\Users\\foo\\file.py -> /c/Users/foo/file.py

    On other platforms, returns unchanged.

    This should be applied to any path that appears in structured tool
    output (JSON results, error messages, etc.) that the model or user
    might reference.

    Args:
        path: Path to include in tool output.

    Returns:
        Normalized path string.
    """
    return normalize_path(path)


def get_display_separator() -> str:
    """Get the path separator to use for display purposes.

    Under MSYS2, returns '/' even though os.sep is '\\', because the user's
    shell expects Unix-style paths.

    Returns:
        '/' under MSYS2, os.sep otherwise.
    """
    if is_msys2_environment():
        return '/'
    return os.sep
