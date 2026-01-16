# shared/plugins/formatter_pipeline/__init__.py
"""Streaming formatter pipeline for processing output through registered formatters.

This module provides a streaming pipeline where formatters process chunks
incrementally. Each formatter can either pass chunks through immediately
or buffer them until ready (e.g., for code blocks).

Example (streaming):
    from shared.plugins.formatter_pipeline import FormatterPipeline, create_pipeline
    from shared.plugins.code_block_formatter import create_plugin as create_code_formatter

    pipeline = create_pipeline()
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

Example (batch):
    formatted = pipeline.format(complete_text)  # Convenience method
"""

from .protocol import FormatterPlugin, ConfigurableFormatter
from .pipeline import FormatterPipeline, create_pipeline
from .registry import FormatterRegistry, create_registry, create_default_pipeline

__all__ = [
    "FormatterPlugin",
    "ConfigurableFormatter",
    "FormatterPipeline",
    "create_pipeline",
    "FormatterRegistry",
    "create_registry",
    "create_default_pipeline",
]
