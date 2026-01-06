# shared/plugins/formatter_pipeline/pipeline.py
"""Formatter pipeline for processing output through registered formatters.

The pipeline manages a collection of formatter plugins, executing them
in priority order on output text. Each formatter decides whether to
process the text based on its detection rules.

Usage:
    from shared.plugins.formatter_pipeline import FormatterPipeline
    from shared.plugins.code_block_formatter import create_plugin as create_code_formatter
    from shared.plugins.diff_formatter import create_plugin as create_diff_formatter

    pipeline = FormatterPipeline()
    pipeline.register(create_diff_formatter())      # priority 20
    pipeline.register(create_code_formatter())      # priority 40

    # Format output (formatters run in priority order)
    formatted = pipeline.format("Here is a diff:\\n- old\\n+ new")
"""

from typing import Any, Dict, List, Optional
import os
import tempfile
from datetime import datetime

from .protocol import FormatterPlugin, ConfigurableFormatter


def _trace(msg: str) -> None:
    """Write debug trace to file."""
    try:
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        with open(trace_path, 'a') as f:
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            f.write(f"[{ts}] FormatterPipeline: {msg}\n")
    except Exception:
        pass


class FormatterPipeline:
    """Pipeline that routes output through registered formatters.

    Formatters are executed in priority order (lowest first). Each formatter
    decides whether to process the text via should_format(). If a formatter
    processes text, subsequent formatters see the transformed output.
    """

    def __init__(self):
        """Initialize an empty pipeline."""
        self._formatters: List[FormatterPlugin] = []
        self._console_width: int = 80

    def register(self, formatter: FormatterPlugin) -> None:
        """Register a formatter plugin.

        Args:
            formatter: Formatter implementing FormatterPlugin protocol.

        Raises:
            TypeError: If formatter doesn't implement required protocol.
        """
        _trace(f"register: attempting to register {type(formatter).__name__}")
        if not isinstance(formatter, FormatterPlugin):
            _trace(f"register: FAILED protocol check for {type(formatter).__name__}")
            raise TypeError(
                f"Formatter must implement FormatterPlugin protocol, "
                f"got {type(formatter).__name__}"
            )

        # Insert in priority order (lower priority first)
        inserted = False
        for i, existing in enumerate(self._formatters):
            if formatter.priority < existing.priority:
                self._formatters.insert(i, formatter)
                inserted = True
                break

        if not inserted:
            self._formatters.append(formatter)

        # Sync console width if formatter supports it
        if isinstance(formatter, ConfigurableFormatter):
            formatter.set_console_width(self._console_width)

        _trace(f"register: SUCCESS - {formatter.name} at priority {formatter.priority}, total formatters: {len(self._formatters)}")

    def unregister(self, name: str) -> bool:
        """Remove a formatter by name.

        Args:
            name: Name of the formatter to remove.

        Returns:
            True if formatter was found and removed.
        """
        for i, formatter in enumerate(self._formatters):
            if formatter.name == name:
                self._formatters.pop(i)
                return True
        return False

    def get_formatter(self, name: str) -> Optional[FormatterPlugin]:
        """Get a registered formatter by name.

        Args:
            name: Name of the formatter.

        Returns:
            The formatter or None if not found.
        """
        for formatter in self._formatters:
            if formatter.name == name:
                return formatter
        return None

    def list_formatters(self) -> List[str]:
        """List registered formatter names in priority order.

        Returns:
            List of formatter names.
        """
        return [f.name for f in self._formatters]

    def set_console_width(self, width: int) -> None:
        """Update console width for all formatters.

        Args:
            width: Console width in characters.
        """
        self._console_width = width
        for formatter in self._formatters:
            if isinstance(formatter, ConfigurableFormatter):
                formatter.set_console_width(width)

    def format(
        self,
        text: str,
        format_hint: Optional[str] = None,
        source: Optional[str] = None
    ) -> str:
        """Process text through the formatter pipeline.

        Args:
            text: The text to format.
            format_hint: Optional hint about content type (e.g., "diff", "json").
            source: Optional source identifier (e.g., "model", "tool").
                   Some formatters may only apply to specific sources.

        Returns:
            Formatted text with ANSI codes for styling.
        """
        result = text
        has_code_block = '```' in text
        if has_code_block:
            _trace(f"format: processing text with code blocks ({len(text)} chars), {len(self._formatters)} formatters registered")

        for formatter in self._formatters:
            should = formatter.should_format(result, format_hint)
            if has_code_block:
                _trace(f"format: {formatter.name} (priority {formatter.priority}) should_format={should}")
            if should:
                result = formatter.format_output(result)

        return result

    def format_if_needed(
        self,
        text: str,
        format_hint: Optional[str] = None,
        source: Optional[str] = None
    ) -> tuple[str, bool]:
        """Process text and indicate if any formatting was applied.

        Args:
            text: The text to format.
            format_hint: Optional hint about content type.
            source: Optional source identifier.

        Returns:
            Tuple of (formatted_text, was_formatted).
        """
        result = text
        was_formatted = False

        for formatter in self._formatters:
            if formatter.should_format(result, format_hint):
                result = formatter.format_output(result)
                was_formatted = True

        return result, was_formatted


    def get_accumulated_validation_issues(self) -> List[Dict[str, Any]]:
        """Get accumulated validation issues from all formatters.

        Returns:
            List of validation issue dicts from formatters that support validation.
        """
        all_issues: List[Dict[str, Any]] = []

        for formatter in self._formatters:
            if hasattr(formatter, 'get_accumulated_issues'):
                issues = formatter.get_accumulated_issues()
                if issues:
                    all_issues.extend(issues)

        return all_issues

    def clear_accumulated_validation_issues(self) -> None:
        """Clear accumulated validation issues from all formatters."""
        for formatter in self._formatters:
            if hasattr(formatter, 'clear_accumulated_issues'):
                formatter.clear_accumulated_issues()


def create_pipeline() -> FormatterPipeline:
    """Factory function to create a FormatterPipeline instance."""
    return FormatterPipeline()
