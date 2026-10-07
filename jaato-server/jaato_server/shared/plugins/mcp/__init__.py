"""MCP tool plugin for executing Model Context Protocol tools.

This plugin connects to MCP servers defined in .mcp.json and exposes
their tools to the AI model.
"""

from .plugin import MCPToolPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

PLUGIN_TIER = "runner"

#: Modules this package imports lazily (imported on the plugin's own thread at initialize()).
#: The pool template imports them before it forks, so a session's
#: slot inherits them shared instead of importing them privately.
#: A literal tuple of strings, read without importing the package;
#: nothing listed may read session-scoped state at import time.
#: See jaato_server/server/runner/template_preload.py.
PLUGIN_PRELOAD = (
    "mcp",
    "mcp.client.stdio",
    "jaato_server.shared.mcp_context_manager",
)
__all__ = [
    'MCPToolPlugin',
    'create_plugin',
]
