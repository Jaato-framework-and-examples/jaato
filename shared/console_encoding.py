"""Console encoding utilities for cross-platform Unicode support.

Windows consoles (cmd, PowerShell) use code pages like cp1252 by default,
which cannot encode certain Unicode characters (emojis, checkmarks, etc.).
This module provides utilities to configure UTF-8 encoding for console output.
"""

import sys


def configure_utf8_output() -> None:
    """Configure stdout and stderr to use UTF-8 encoding with error handling.

    This is primarily needed on Windows where the console encoding defaults
    to a code page (e.g., cp1252) that cannot handle all Unicode characters.

    Uses 'replace' error handling to substitute unencodable characters with
    a replacement character (?) rather than raising UnicodeEncodeError.

    Safe to call on any platform - does nothing if already configured or
    if reconfiguration is not supported.
    """
    if sys.platform == 'win32':
        try:
            # Python 3.7+ supports reconfigure() on text streams
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            # If reconfiguration fails, try setting via environment
            # This is a fallback that may help in some scenarios
            import os
            os.environ.setdefault('PYTHONIOENCODING', 'utf-8:replace')
