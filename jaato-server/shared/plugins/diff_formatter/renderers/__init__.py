# shared/plugins/diff_formatter/renderers/__init__.py
"""Diff renderers for different terminal widths and formats."""

from .base import DiffRenderer, ColorScheme, DEFAULT_COLOR_SCHEME
from .unified import UnifiedRenderer
from .side_by_side import SideBySideRenderer
from .compact import CompactRenderer

__all__ = [
    "DiffRenderer",
    "ColorScheme",
    "DEFAULT_COLOR_SCHEME",
    "UnifiedRenderer",
    "SideBySideRenderer",
    "CompactRenderer",
]
