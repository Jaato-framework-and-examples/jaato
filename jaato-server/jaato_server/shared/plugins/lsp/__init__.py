"""LSP tool plugin for code intelligence via Language Server Protocol."""

from .plugin import LSPToolPlugin, create_plugin

PLUGIN_KIND = "tool"

PLUGIN_TIER = "runner"
__all__ = ['LSPToolPlugin', 'create_plugin']
