# shared/plugins/code_block_formatter/__init__.py
"""Code block formatter plugin for syntax highlighting code blocks in model output."""

from .plugin import CodeBlockFormatterPlugin, create_plugin

__all__ = ["CodeBlockFormatterPlugin", "create_plugin"]
