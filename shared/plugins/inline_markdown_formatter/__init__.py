# shared/plugins/inline_markdown_formatter/__init__.py
"""Inline markdown formatter plugin for styling inline text elements in model output.

Handles: `code`, **bold**, *italic*, ***bold italic***, ~~strikethrough~~,
and [links](url). Converts markdown syntax to ANSI-styled terminal text.
"""

from .plugin import InlineMarkdownFormatterPlugin, create_plugin

__all__ = ["InlineMarkdownFormatterPlugin", "create_plugin"]
