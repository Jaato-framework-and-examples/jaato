# shared/plugins/diff_formatter/syntax_highlight.py
"""Syntax highlighting support for diff content.

Uses Pygments to apply syntax highlighting to code lines within diffs.
Highlights context (unchanged) lines while preserving diff colors for changes.
"""

import os
from functools import lru_cache
from typing import Optional

# Try to import Pygments - gracefully degrade if not available
try:
    from pygments import highlight
    from pygments.lexers import get_lexer_for_filename, get_lexer_by_name
    from pygments.formatters import TerminalTrueColorFormatter
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False


# File extension to Pygments lexer name mapping for common cases
EXTENSION_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.jsx': 'jsx',
    '.rb': 'ruby',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
    '.cs': 'csharp',
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.sh': 'bash',
    '.bash': 'bash',
    '.zsh': 'zsh',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.xml': 'xml',
    '.html': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sql': 'sql',
    '.md': 'markdown',
    '.toml': 'toml',
    '.ini': 'ini',
    '.dockerfile': 'docker',
    '.lua': 'lua',
    '.r': 'r',
    '.R': 'r',
    '.pl': 'perl',
    '.pm': 'perl',
    '.cob': 'cobol',
    '.COB': 'cobol',
    '.cbl': 'cobol',
    '.CBL': 'cobol',
    '.cpy': 'cobol',
    '.CPY': 'cobol',
}


@lru_cache(maxsize=32)
def _get_lexer(filename: str):
    """Get Pygments lexer for a filename (cached).

    Args:
        filename: File path or name.

    Returns:
        Pygments lexer or None if not found.
    """
    if not PYGMENTS_AVAILABLE:
        return None

    # Try extension mapping first
    _, ext = os.path.splitext(filename)
    if ext.lower() in EXTENSION_MAP:
        try:
            return get_lexer_by_name(EXTENSION_MAP[ext.lower()])
        except ClassNotFound:
            pass

    # Fall back to Pygments filename detection
    try:
        return get_lexer_for_filename(filename)
    except ClassNotFound:
        return None


@lru_cache(maxsize=1)
def _get_formatter():
    """Get Pygments terminal formatter (cached)."""
    if not PYGMENTS_AVAILABLE:
        return None
    return TerminalTrueColorFormatter(style='monokai')


def highlight_line(line: str, filename: str) -> str:
    """Apply syntax highlighting to a single line of code.

    Args:
        line: The code line to highlight.
        filename: Filename for language detection.

    Returns:
        Line with ANSI escape codes for syntax highlighting,
        or original line if highlighting fails/unavailable.
    """
    if not PYGMENTS_AVAILABLE:
        return line

    lexer = _get_lexer(filename)
    if not lexer:
        return line

    formatter = _get_formatter()
    if not formatter:
        return line

    try:
        # Highlight the line (Pygments adds a trailing newline, strip it)
        highlighted = highlight(line, lexer, formatter)
        # Remove trailing newline added by Pygments
        if highlighted.endswith('\n'):
            highlighted = highlighted[:-1]
        # Also remove any trailing reset codes for cleaner concatenation
        # (we'll handle resets ourselves)
        return highlighted
    except Exception:
        return line


def can_highlight(filename: str) -> bool:
    """Check if syntax highlighting is available for a file.

    Args:
        filename: Filename to check.

    Returns:
        True if highlighting is available.
    """
    return _get_lexer(filename) is not None
