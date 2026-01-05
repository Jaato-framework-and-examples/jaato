# shared/plugins/diff_formatter/plugin.py
"""Diff formatter plugin with adaptive rendering based on terminal width.

Supports three rendering modes:
- Side-by-side (≥120 cols): Two-column view with box drawing
- Compact (80-119 cols): Single-column with arrow notation
- Unified (<80 cols): Traditional +/- format

The mode is automatically selected based on terminal width.

Usage (standalone):
    from shared.plugins.diff_formatter import create_plugin

    formatter = create_plugin()
    formatter.set_console_width(140)  # Side-by-side mode
    colored = formatter.format_output(diff_text)

Usage (pipeline):
    from shared.plugins.formatter_pipeline import create_pipeline
    from shared.plugins.diff_formatter import create_plugin

    pipeline = create_pipeline()
    pipeline.register(create_plugin())  # priority 20
    formatted = pipeline.format(text, format_hint="diff")
"""

import re
from typing import Any, Dict, List, Optional

from .parser import parse_unified_diff, ParsedDiff
from .renderers.base import ColorScheme, DEFAULT_COLOR_SCHEME, NO_COLOR_SCHEME
from .renderers.unified import UnifiedRenderer, render_raw_unified
from .renderers.side_by_side import SideBySideRenderer
from .renderers.compact import CompactRenderer

# Pattern to detect unified diff content
DIFF_LINE_PATTERN = re.compile(r'^(\+\+\+|---|\+|-|@@)', re.MULTILINE)
UNIFIED_DIFF_HEADER = re.compile(r'^(---|\+\+\+) ', re.MULTILINE)
HUNK_HEADER = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@', re.MULTILINE)

# Priority for pipeline ordering (20-39 = structural formatting)
DEFAULT_PRIORITY = 20

# Width thresholds for mode selection
SIDE_BY_SIDE_MIN_WIDTH = 120
COMPACT_MIN_WIDTH = 80


class DiffFormatterPlugin:
    """Plugin that formats unified diff output with adaptive rendering.

    Implements the FormatterPlugin protocol for use in a formatter pipeline.
    Automatically selects the best rendering mode based on terminal width:

    - Side-by-side (≥120 cols): Two-column box-drawn table with:
        - OLD and NEW columns
        - Line numbers for both versions
        - Word-level highlighting for modifications

    - Compact (80-119 cols): Single-column box-drawn format with:
        - Arrow notation (→) for modifications
        - +/- prefix for pure adds/deletes
        - Condensed display without context lines

    - Unified (<80 cols): Traditional format with:
        - Standard +/- line prefixes
        - Colored output (green/red)
        - Full context preserved
    """

    def __init__(self):
        self._priority = DEFAULT_PRIORITY
        self._console_width = 120
        self._colors = DEFAULT_COLOR_SCHEME

        # Renderers in priority order (highest min_width first)
        self._renderers = [
            SideBySideRenderer(),
            CompactRenderer(),
            UnifiedRenderer(),
        ]

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "diff_formatter"

    @property
    def priority(self) -> int:
        """Execution priority (20 = structural formatting range)."""
        return self._priority

    def should_format(self, text: str, format_hint: Optional[str] = None) -> bool:
        """Check if this formatter should process the text.

        Uses format_hint="diff" as fast path, otherwise detects diff patterns.

        Args:
            text: Text to check.
            format_hint: Optional hint - "diff" triggers formatting.

        Returns:
            True if text appears to be a unified diff.
        """
        # Fast path: explicit hint
        if format_hint == "diff":
            return True

        # Detection: look for unified diff markers
        if HUNK_HEADER.search(text):
            return True
        if UNIFIED_DIFF_HEADER.search(text) and DIFF_LINE_PATTERN.search(text):
            return True

        return False

    def format_output(self, text: str) -> str:
        """Format unified diff output with adaptive rendering.

        Parses the diff and selects the appropriate renderer
        based on current terminal width.

        Args:
            text: Text containing unified diff.

        Returns:
            Formatted diff with appropriate rendering style.
        """
        # Parse the unified diff
        try:
            parsed = parse_unified_diff(text)
        except Exception:
            # If parsing fails, fall back to simple colorization
            return render_raw_unified(text, self._colors)

        # If no hunks were parsed, fall back to raw colorization
        if not parsed.hunks:
            return render_raw_unified(text, self._colors)

        # Select renderer based on width
        renderer = self._select_renderer()

        # Render the diff
        return renderer.render(parsed, self._console_width, self._colors)

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - priority: Pipeline priority (default: 20)
                - console_width: Terminal width (default: 120)
                - colors: Custom ColorScheme or False for no colors
        """
        config = config or {}
        self._priority = config.get("priority", DEFAULT_PRIORITY)
        self._console_width = config.get("console_width", 120)

        colors_config = config.get("colors")
        if colors_config is False:
            self._colors = NO_COLOR_SCHEME
        elif isinstance(colors_config, ColorScheme):
            self._colors = colors_config
        else:
            self._colors = DEFAULT_COLOR_SCHEME

    def set_console_width(self, width: int) -> None:
        """Update console width for adaptive rendering.

        Args:
            width: Terminal width in columns.
        """
        self._console_width = width

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        pass

    # ==================== Mode Selection ====================

    def _select_renderer(self):
        """Select the appropriate renderer based on terminal width.

        Returns:
            Renderer instance appropriate for current width.
        """
        for renderer in self._renderers:
            if self._console_width >= renderer.min_width():
                return renderer

        # Fallback to last (unified) renderer
        return self._renderers[-1]

    def get_current_mode(self) -> str:
        """Get the current rendering mode name.

        Returns:
            One of: "side_by_side", "compact", "unified"
        """
        renderer = self._select_renderer()
        if isinstance(renderer, SideBySideRenderer):
            return "side_by_side"
        elif isinstance(renderer, CompactRenderer):
            return "compact"
        else:
            return "unified"

    # ==================== Convenience Methods ====================

    def colorize_diff(self, diff_text: str) -> str:
        """Colorize a diff string (alias for format_output).

        Args:
            diff_text: Unified diff text.

        Returns:
            Formatted diff.
        """
        return self.format_output(diff_text)

    def is_diff(self, text: str) -> bool:
        """Check if text appears to be a unified diff.

        Args:
            text: Text to check.

        Returns:
            True if text looks like a diff.
        """
        return self.should_format(text)

    def set_colors(self, colors: ColorScheme) -> None:
        """Set custom color scheme.

        Args:
            colors: ColorScheme instance.
        """
        self._colors = colors

    def disable_colors(self) -> None:
        """Disable color output."""
        self._colors = NO_COLOR_SCHEME

    def enable_colors(self) -> None:
        """Enable default color output."""
        self._colors = DEFAULT_COLOR_SCHEME

    # ==================== Legacy Compatibility ====================

    # These properties maintain compatibility with code that accessed
    # the old color configuration directly

    @property
    def _color_additions(self) -> str:
        return self._colors.added

    @property
    def _color_deletions(self) -> str:
        return self._colors.deleted

    @property
    def _color_hunks(self) -> str:
        return self._colors.hunk_header

    @property
    def _color_headers(self) -> str:
        return self._colors.dim


def create_plugin() -> DiffFormatterPlugin:
    """Factory function to create a DiffFormatterPlugin instance."""
    return DiffFormatterPlugin()
