"""Cross-platform path utilities for MSYS2/Git Bash compatibility.

When running native Windows Python (MinGW) under an MSYS2 or Git Bash shell,
Python's path functions produce Windows-style paths (backslashes, drive letters)
but the shell environment expects Unix-style paths (forward slashes).

This module provides:
- MSYS2 environment detection
- Path normalization for display and string comparison
- Consistent path separator handling across platforms

The normalization converts backslashes to forward slashes when running under
MSYS2, which:
- Makes paths copy-pasteable in the MSYS2 shell
- Ensures string-based prefix matching works with mixed-separator paths
- Keeps internal Python path operations working (Python handles both separators)

Reference: https://www.msys2.org/docs/python/
           https://www.msys2.org/docs/filesystem-paths/
"""

import functools
import os
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


def normalize_path(path: str) -> str:
    """Normalize path separators for the current environment.

    Under MSYS2 on Windows, converts backslashes to forward slashes so that
    paths are compatible with the Unix-like shell environment. On other
    platforms, returns the path unchanged.

    This is primarily for paths that will be:
    - Displayed to the user in tool output
    - Used in string-based prefix matching/comparisons
    - Returned as part of structured tool results

    Internal Python file operations (open, read, write, os.path, pathlib)
    handle both separators on Windows, so normalization is only needed at
    display/comparison boundaries.

    Args:
        path: Path string to normalize.

    Returns:
        Path with forward slashes under MSYS2, unchanged otherwise.
    """
    if not path:
        return path

    if is_msys2_environment():
        return path.replace('\\', '/')

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

    Under MSYS2, converts to forward slashes so paths in tool responses
    are usable in the shell. On other platforms, returns unchanged.

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
