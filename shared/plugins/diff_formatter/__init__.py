# shared/plugins/diff_formatter/__init__.py
"""Diff formatter plugin with adaptive rendering based on terminal width.

This plugin supports three rendering modes:
- Side-by-side (≥120 cols): Two-column view with box drawing
- Compact (80-119 cols): Single-column with arrow notation
- Unified (<80 cols): Traditional +/- format

The mode is automatically selected based on terminal width.

Example:
    from shared.plugins.diff_formatter import create_plugin

    formatter = create_plugin()
    formatter.set_console_width(140)  # Will use side-by-side mode
    output = formatter.format_output(diff_text)
"""

from .plugin import DiffFormatterPlugin, create_plugin
from .parser import (
    ParsedDiff,
    DiffHunk,
    DiffLine,
    DiffStats,
    parse_unified_diff,
    get_paired_lines,
)
from .renderers.base import ColorScheme, DEFAULT_COLOR_SCHEME, NO_COLOR_SCHEME
from .word_diff import WordDiff, compute_word_diff

__all__ = [
    # Main plugin
    "DiffFormatterPlugin",
    "create_plugin",
    # Parser types
    "ParsedDiff",
    "DiffHunk",
    "DiffLine",
    "DiffStats",
    "parse_unified_diff",
    "get_paired_lines",
    # Color scheme
    "ColorScheme",
    "DEFAULT_COLOR_SCHEME",
    "NO_COLOR_SCHEME",
    # Word diff
    "WordDiff",
    "compute_word_diff",
]
