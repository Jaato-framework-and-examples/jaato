# shared/plugins/diff_formatter/__init__.py
"""Diff formatter plugin for colorizing unified diff output."""

from .plugin import DiffFormatterPlugin, create_plugin

__all__ = ["DiffFormatterPlugin", "create_plugin"]
