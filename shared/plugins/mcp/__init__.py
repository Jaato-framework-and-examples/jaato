"""MCP tool plugin for executing Model Context Protocol tools.

This plugin connects to MCP servers defined in .mcp.json and exposes
their tools to the AI model.
"""

from .plugin import MCPToolPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'MCPToolPlugin',
    'create_plugin',
]
