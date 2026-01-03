# shared/plugins/formatter_pipeline/protocol.py
"""Protocol definition for output formatter plugins.

Formatter plugins subscribe to the output pipeline and transform text
based on detection rules. Each formatter has a priority that determines
execution order (lower priority runs first).

Usage:
    class MyFormatter:
        name = "my_formatter"
        priority = 50  # Lower = runs earlier

        def should_format(self, text: str, format_hint: Optional[str] = None) -> bool:
            return "my_pattern" in text

        def format_output(self, text: str) -> str:
            return transform(text)
"""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class FormatterPlugin(Protocol):
    """Protocol for output formatter plugins.

    Formatters transform text in the output pipeline. Each formatter:
    - Has a unique name for identification
    - Has a priority for ordering (lower = runs first)
    - Decides if it should format given text (via should_format)
    - Transforms text if applicable (via format_output)
    """

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority. Lower values run first.

        Suggested ranges:
        - 0-19: Pre-processing (normalization, encoding fixes)
        - 20-39: Structural formatting (diffs, tables)
        - 40-59: Syntax highlighting (code blocks, JSON)
        - 60-79: Post-processing (line wrapping, truncation)
        - 80-99: Final touches (emoji, special characters)
        """
        ...

    def should_format(self, text: str, format_hint: Optional[str] = None) -> bool:
        """Determine if this formatter should process the text.

        Args:
            text: The text to potentially format.
            format_hint: Optional hint about content type (e.g., "diff", "json").
                        Formatters can use this for fast-path detection.

        Returns:
            True if this formatter should process the text.
        """
        ...

    def format_output(self, text: str) -> str:
        """Transform the text.

        Args:
            text: The input text (may contain ANSI codes from prior formatters).

        Returns:
            Transformed text with ANSI codes for styling.
        """
        ...


@runtime_checkable
class ConfigurableFormatter(FormatterPlugin, Protocol):
    """Extended protocol for formatters that support configuration."""

    def initialize(self, config: dict) -> None:
        """Initialize with configuration.

        Args:
            config: Configuration dictionary with formatter-specific settings.
        """
        ...

    def set_console_width(self, width: int) -> None:
        """Update console width for rendering.

        Args:
            width: Console width in characters.
        """
        ...
