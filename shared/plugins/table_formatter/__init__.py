# shared/plugins/table_formatter/__init__.py
"""Table formatter plugin for detecting and rendering tables with box-drawing characters.

Detects markdown tables and renders them with proper box-drawing borders
to ensure fixed-width alignment in the terminal.
"""

from .plugin import TableFormatterPlugin, create_plugin

__all__ = ["TableFormatterPlugin", "create_plugin"]
