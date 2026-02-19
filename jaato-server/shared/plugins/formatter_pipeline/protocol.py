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


# Optional auto-wiring methods for formatters
#
# Formatters may implement these methods to receive automatic dependency injection
# from the FormatterRegistry during pipeline creation. These are not part of the
# protocol but are recognized by the registry's auto-wiring mechanism.
#
# wire_dependencies(tool_registry: Any) -> bool:
#     """Wire formatter with tool plugins it depends on.
#
#     Called by FormatterRegistry during create_pipeline() before initialize().
#     Formatters that need access to tool plugins (e.g., code_validation_formatter
#     needs the LSP plugin) should implement this method.
#
#     The FormatterRegistry provides access to the PluginRegistry containing
#     all tool plugins, allowing formatters to retrieve plugins they depend on.
#
#     Args:
#         tool_registry: The PluginRegistry instance containing tool plugins.
#                       Use tool_registry.get_plugin("name") to retrieve plugins.
#
#     Returns:
#         True if wiring succeeded and formatter should be included in pipeline.
#         False if required dependencies are unavailable - formatter will be skipped.
#
#     Example:
#         def wire_dependencies(self, tool_registry: Any) -> bool:
#             lsp = tool_registry.get_plugin("lsp") if tool_registry else None
#             if lsp:
#                 self._lsp_plugin = lsp
#                 return True
#             return False  # Skip this formatter - LSP not available
#     """
#
# get_system_instructions() -> Optional[str]:
#     """Return system instructions describing this formatter's capabilities.
#
#     Called by FormatterPipeline.get_system_instructions() during session
#     configuration. Allows output formatters to inform the model about
#     rendering capabilities it can take advantage of.
#
#     Unlike tool plugins (which always contribute instructions), formatter
#     instructions are optional - most formatters silently transform output
#     without the model needing to know. Only implement this when the model
#     can actively benefit from knowing about the formatter (e.g., using
#     mermaid diagrams because the pipeline renders them graphically).
#
#     Returns:
#         System instruction string, or None if no instructions needed.
#
#     Example:
#         def get_system_instructions(self) -> Optional[str]:
#             return (
#                 "The output pipeline renders ```mermaid code blocks as "
#                 "graphical diagrams. Feel free to use mermaid syntax."
#             )
#     """
#
# get_turn_feedback() -> Optional[str]:
#     """Return feedback from this turn for injection into the next user prompt.
#
#     Called by FormatterPipeline.collect_turn_feedback() after flush() at turn
#     end. Allows formatters that detect issues (syntax errors, validation
#     failures) to report them back to the model so it can self-correct on
#     the next turn.
#
#     The returned text is prepended to the next user message as a <hidden>
#     block — the model sees it, but the user doesn't (they already saw the
#     diagnostic in the terminal output).
#
#     Implementations should return and clear any accumulated feedback
#     (one-shot pattern: each call drains the feedback).
#
#     Returns:
#         Feedback string, or None if no feedback to report.
#
#     Example:
#         def get_turn_feedback(self) -> Optional[str]:
#             if self._turn_feedback:
#                 fb = self._turn_feedback
#                 self._turn_feedback = None
#                 return fb
#             return None
#     """
