"""Service connector plugin for webservice discovery and consumption.

This plugin provides tools for:
- Discovering APIs via OpenAPI/Swagger specifications
- Managing request/response schemas stored on the filesystem
- Making HTTP requests with validation
- Importing API collections from Bruno
"""

from .plugin import ServiceConnectorPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'ServiceConnectorPlugin',
    'create_plugin',
]
