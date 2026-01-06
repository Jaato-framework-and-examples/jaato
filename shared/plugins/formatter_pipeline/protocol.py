# shared/plugins/formatter_pipeline/protocol.py
"""Protocol definition for streaming formatter plugins.

Formatter plugins process output text in a streaming pipeline. Each formatter
receives chunks incrementally and decides whether to:
- Pass through immediately (yield chunk as-is)
- Buffer internally (yield nothing, accumulate)
- Emit processed content (yield transformed text when ready)

This enables efficient streaming where regular text flows through immediately
while special content (code blocks, diffs) is buffered until complete.

Usage:
    class MyFormatter:
        name = "my_formatter"
        priority = 50  # Lower = runs earlier

        def process_chunk(self, chunk: str) -> Iterator[str]:
            # For simple pass-through with transform:
            yield self._transform(chunk)

            # For buffering until pattern complete:
            # ... accumulate, yield when ready ...

        def flush(self) -> Iterator[str]:
            if self._buffer:
                yield self._process(self._buffer)

        def reset(self) -> None:
            self._buffer = ""
"""

from typing import Iterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class FormatterPlugin(Protocol):
    """Protocol for streaming formatter plugins.

    Formatters transform text in the output pipeline. Each formatter:
    - Has a unique name for identification
    - Has a priority for ordering (lower = runs first)
    - Processes chunks incrementally via process_chunk()
    - Flushes remaining content at turn end via flush()
    - Resets state for new turns via reset()
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

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process an incoming chunk, yielding output chunks.

        The formatter may:
        - Yield the chunk immediately (pass-through)
        - Yield nothing and buffer internally (accumulating)
        - Yield processed content (when accumulation is complete)
        - Yield multiple chunks (if chunk completes a buffer and has more)

        Args:
            chunk: Incoming text chunk from the stream.

        Yields:
            Zero or more output chunks to pass to the next stage.
        """
        ...

    def flush(self) -> Iterator[str]:
        """Flush any buffered content.

        Called at turn end to ensure all buffered content is emitted,
        even if accumulation patterns weren't completed.

        Yields:
            Any remaining buffered content, possibly processed.
        """
        ...

    def reset(self) -> None:
        """Reset internal state for a new turn.

        Called at the start of a new turn to clear any lingering state.
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
