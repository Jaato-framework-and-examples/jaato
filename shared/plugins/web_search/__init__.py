"""Web search plugin for performing internet searches.

This plugin provides the `web_search` function that performs web searches
using DuckDuckGo and returns relevant results.
"""

from .plugin import WebSearchPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'WebSearchPlugin',
    'create_plugin',
]
