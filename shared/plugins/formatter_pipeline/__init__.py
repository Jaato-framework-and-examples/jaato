# shared/plugins/formatter_pipeline/__init__.py
"""Formatter pipeline for processing output through registered formatters.

This module provides the infrastructure for a priority-based formatter pipeline
where multiple formatter plugins can subscribe to process output text.

Example:
    from shared.plugins.formatter_pipeline import FormatterPipeline, create_pipeline
    from shared.plugins.diff_formatter import create_plugin as create_diff_formatter
    from shared.plugins.code_block_formatter import create_plugin as create_code_formatter

    # Create pipeline and register formatters
    pipeline = create_pipeline()
    pipeline.register(create_diff_formatter())      # priority 20
    pipeline.register(create_code_formatter())      # priority 40

    # Process output
    formatted = pipeline.format(text, format_hint="diff")
"""

from .protocol import FormatterPlugin, ConfigurableFormatter
from .pipeline import FormatterPipeline, create_pipeline

__all__ = [
    "FormatterPlugin",
    "ConfigurableFormatter",
    "FormatterPipeline",
    "create_pipeline",
]
