"""Tool Discovery & Introspection Plugin.

This plugin enables agents to discover and query available tools at runtime.
It provides two tools:

- list_tools: Returns all registered tools with optional category filtering
- get_tool_schema: Returns full documentation for a specific tool

Example usage:

    from shared.plugins.introspection import create_plugin

    plugin = create_plugin()
    plugin.initialize({})

    # List all tools
    executors = plugin.get_executors()
    result = executors["list_tools"]({})

    # Filter by category
    result = executors["list_tools"]({"category": "filesystem"})

    # Get detailed schema
    result = executors["get_tool_schema"]({"tool_name": "readFile"})
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

from .plugin import IntrospectionPlugin, create_plugin

__all__ = [
    'IntrospectionPlugin',
    'create_plugin',
]
