"""LSP tool plugin for code intelligence via Language Server Protocol."""

from .plugin import LSPToolPlugin, create_plugin

PLUGIN_KIND = "tool"

__all__ = ['LSPToolPlugin', 'create_plugin']
