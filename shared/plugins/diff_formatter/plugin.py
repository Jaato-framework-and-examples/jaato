# shared/plugins/diff_formatter/plugin.py
"""Diff formatter plugin for colorizing unified diff output.

This plugin detects unified diff format and applies ANSI colors:
- Green for additions (+)
- Red for deletions (-)
- Cyan for hunk headers (@@)
- Dim for file headers (+++, ---)

Can be used standalone or registered with a FormatterPipeline.

Usage (standalone):
    from shared.plugins.diff_formatter import create_plugin

    formatter = create_plugin()
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

# ANSI color codes
ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"

# Pattern to detect unified diff content
# Matches lines starting with diff markers
DIFF_LINE_PATTERN = re.compile(r'^(\+\+\+|---|\+|-|@@)', re.MULTILINE)

# More strict pattern for unified diff headers
UNIFIED_DIFF_HEADER = re.compile(r'^(---|\+\+\+) ', re.MULTILINE)
HUNK_HEADER = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@', re.MULTILINE)

# Priority for pipeline ordering (20-39 = structural formatting)
DEFAULT_PRIORITY = 20


class DiffFormatterPlugin:
    """Plugin that colorizes unified diff output.

    Implements the FormatterPlugin protocol for use in a formatter pipeline.
    Detects unified diff format and applies ANSI colors to diff lines.
    """

    def __init__(self):
        self._priority = DEFAULT_PRIORITY
        self._color_additions = ANSI_GREEN
        self._color_deletions = ANSI_RED
        self._color_hunks = ANSI_CYAN
        self._color_headers = ANSI_DIM

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
        # Must have at least a hunk header or file headers to be a diff
        if HUNK_HEADER.search(text):
            return True
        if UNIFIED_DIFF_HEADER.search(text) and DIFF_LINE_PATTERN.search(text):
            return True

        return False

    def format_output(self, text: str) -> str:
        """Colorize unified diff output.

        Args:
            text: Text containing unified diff.

        Returns:
            Text with ANSI color codes for diff highlighting.
        """
        lines = text.split('\n')
        colored_lines = [self._colorize_line(line) for line in lines]
        return '\n'.join(colored_lines)

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - priority: Pipeline priority (default: 20)
                - color_additions: ANSI code for additions (default: green)
                - color_deletions: ANSI code for deletions (default: red)
                - color_hunks: ANSI code for hunk headers (default: cyan)
                - color_headers: ANSI code for file headers (default: dim)
        """
        config = config or {}
        self._priority = config.get("priority", DEFAULT_PRIORITY)
        self._color_additions = config.get("color_additions", ANSI_GREEN)
        self._color_deletions = config.get("color_deletions", ANSI_RED)
        self._color_hunks = config.get("color_hunks", ANSI_CYAN)
        self._color_headers = config.get("color_headers", ANSI_DIM)

    def set_console_width(self, width: int) -> None:
        """Update console width (not used by diff formatter)."""
        pass

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        pass

    # ==================== Internal Methods ====================

    def _colorize_line(self, line: str) -> str:
        """Apply color to a single diff line based on its prefix.

        Args:
            line: A single line from the diff.

        Returns:
            Line with ANSI color codes applied.
        """
        if line.startswith('+++') or line.startswith('---'):
            # File headers - dim
            return f"{self._color_headers}{line}{ANSI_RESET}"
        elif line.startswith('@@'):
            # Hunk headers - cyan
            return f"{self._color_hunks}{line}{ANSI_RESET}"
        elif line.startswith('+'):
            # Additions - green
            return f"{self._color_additions}{line}{ANSI_RESET}"
        elif line.startswith('-'):
            # Deletions - red
            return f"{self._color_deletions}{line}{ANSI_RESET}"
        else:
            # Context lines - no color
            return line

    # ==================== Convenience Methods ====================

    def colorize_diff(self, diff_text: str) -> str:
        """Colorize a diff string (alias for format_output).

        Args:
            diff_text: Unified diff text.

        Returns:
            Colorized diff with ANSI codes.
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


def create_plugin() -> DiffFormatterPlugin:
    """Factory function to create a DiffFormatterPlugin instance."""
    return DiffFormatterPlugin()
