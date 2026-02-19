"""Web fetch plugin for retrieving and parsing web page content.

This plugin provides the `web_fetch` function that fetches URLs and returns
content in model-friendly formats (markdown, structured data, or raw HTML).
"""

from .plugin import WebFetchPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'WebFetchPlugin',
    'create_plugin',
]
