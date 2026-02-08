# shared/plugins/formatter_pipeline/pipeline.py
"""Streaming formatter pipeline for processing output through registered formatters.

The pipeline manages a collection of formatter plugins, processing chunks
through them in priority order. Each formatter can either pass chunks through
immediately or buffer them until ready (e.g., for code blocks).

Usage:
    from shared.plugins.formatter_pipeline import FormatterPipeline
    from shared.plugins.code_block_formatter import create_plugin as create_code_formatter

    pipeline = FormatterPipeline()
    pipeline.register(create_code_formatter())

    # Process streaming chunks
    for chunk in model_output_stream:
        for output in pipeline.process_chunk(chunk):
            display(output)  # Display immediately

    # At turn end, flush any remaining buffered content
    for output in pipeline.flush():
        display(output)

    # Reset for next turn
    pipeline.reset()
"""

from typing import Any, Dict, Iterator, List, Optional
import os
from datetime import datetime

from .protocol import FormatterPlugin, ConfigurableFormatter
from shared.trace import trace as _trace_write


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    _trace_write("FormatterPipeline", msg)


class FormatterPipeline:
    """Streaming pipeline that routes chunks through registered formatters.

    Formatters are executed in priority order (lowest first). Each formatter
    receives chunks and decides whether to pass through or buffer. Output
    from one formatter becomes input to the next.
    """

    def __init__(self):
        """Initialize an empty pipeline."""
        self._formatters: List[FormatterPlugin] = []
        self._console_width: int = 80
        self._pending_feedback: Optional[str] = None

    def register(self, formatter: FormatterPlugin) -> None:
        """Register a formatter plugin.

        Args:
            formatter: Formatter implementing FormatterPlugin protocol.
        """
        _trace(f"register: {formatter.name} at priority {formatter.priority}")

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

        _trace(f"register: total formatters: {len(self._formatters)}")

    def unregister(self, name: str) -> bool:
        """Remove a formatter by name."""
        for i, formatter in enumerate(self._formatters):
            if formatter.name == name:
                self._formatters.pop(i)
                return True
        return False

    def get_formatter(self, name: str) -> Optional[FormatterPlugin]:
        """Get a registered formatter by name."""
        for formatter in self._formatters:
            if formatter.name == name:
                return formatter
        return None

    def list_formatters(self) -> List[str]:
        """List registered formatter names in priority order."""
        return [f.name for f in self._formatters]

    def set_console_width(self, width: int) -> None:
        """Update console width for all formatters."""
        self._console_width = width
        for formatter in self._formatters:
            if isinstance(formatter, ConfigurableFormatter):
                formatter.set_console_width(width)

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk through all formatters.

        The chunk flows through formatters in priority order. Each formatter
        may yield zero or more output chunks, which become input to the next.

        Args:
            chunk: Incoming text chunk.

        Yields:
            Output chunks after processing through all formatters.
        """
        if not self._formatters:
            yield chunk
            return

        # Chain through formatters: output of one becomes input to next
        chunks = [chunk]
        for formatter in self._formatters:
            next_chunks = []
            for c in chunks:
                for output in formatter.process_chunk(c):
                    next_chunks.append(output)
            chunks = next_chunks

        for c in chunks:
            yield c

    def flush(self) -> Iterator[str]:
        """Flush all formatters and yield remaining content.

        Called at turn end. Flushes each formatter in order, with output
        from earlier formatters flowing through later ones.

        Yields:
            Any remaining buffered content from all formatters.
        """
        if not self._formatters:
            return

        # Flush each formatter, passing output through remaining formatters
        for i, formatter in enumerate(self._formatters):
            # Get flushed content from this formatter
            flushed = list(formatter.flush())

            # Pass through remaining formatters
            for chunk in flushed:
                remaining_formatters = self._formatters[i + 1:]
                outputs = [chunk]
                for remaining in remaining_formatters:
                    next_outputs = []
                    for c in outputs:
                        for output in remaining.process_chunk(c):
                            next_outputs.append(output)
                    outputs = next_outputs

                for output in outputs:
                    yield output

        # Final flush of all remaining formatters (in case flush added new content)
        for formatter in self._formatters:
            for output in formatter.flush():
                yield output

    def reset(self) -> None:
        """Reset all formatters for a new turn."""
        for formatter in self._formatters:
            formatter.reset()

    # ==================== Turn Feedback ====================

    def collect_turn_feedback(self) -> Optional[str]:
        """Collect turn feedback from registered formatters and store it.

        Called after flush() at turn end. Iterates all formatters and collects
        feedback from those that implement the optional get_turn_feedback()
        method. Stores the result for retrieval via get_pending_feedback().

        Returns:
            Combined feedback string, or None if no formatter has feedback.
        """
        parts = []
        for formatter in self._formatters:
            if hasattr(formatter, "get_turn_feedback"):
                feedback = formatter.get_turn_feedback()
                if feedback:
                    parts.append(feedback)
        combined = "\n\n".join(parts) if parts else None
        if combined:
            self._pending_feedback = combined
        return combined

    def get_pending_feedback(self) -> Optional[str]:
        """Return and clear any pending turn feedback.

        Returns:
            The feedback string stored by collect_turn_feedback(), or None.
        """
        feedback = self._pending_feedback
        self._pending_feedback = None
        return feedback

    # ==================== System Instructions ====================

    def get_system_instructions(self) -> Optional[str]:
        """Collect system instructions from registered formatters.

        Iterates all formatters and collects instructions from those that
        implement the optional get_system_instructions() method. This allows
        output formatters to inform the model about rendering capabilities
        it can take advantage of (e.g., mermaid diagram rendering).

        Returns:
            Combined instruction string, or None if no formatter contributes.
        """
        parts = []
        for formatter in self._formatters:
            if hasattr(formatter, "get_system_instructions"):
                instr = formatter.get_system_instructions()
                if instr:
                    parts.append(instr)
        return "\n\n".join(parts) if parts else None

    # ==================== Convenience Methods ====================

    def format(self, text: str) -> str:
        """Process complete text through pipeline (batch mode).

        Convenience method for non-streaming use. Processes text as a single
        chunk and flushes.

        Args:
            text: Complete text to format.

        Returns:
            Formatted text.
        """
        self.reset()
        result_parts = []
        for output in self.process_chunk(text):
            result_parts.append(output)
        for output in self.flush():
            result_parts.append(output)
        return ''.join(result_parts)

    def get_accumulated_validation_issues(self) -> List[Dict[str, Any]]:
        """Get accumulated validation issues from all formatters."""
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
