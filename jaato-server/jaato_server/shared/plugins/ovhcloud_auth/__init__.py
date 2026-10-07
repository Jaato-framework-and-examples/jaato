"""OVHcloud AI Endpoints authentication plugin.

Provides user commands for API key authentication with OVHcloud AI
Endpoints.
"""

# Plugin kind for registry discovery
PLUGIN_KIND = "tool"

PLUGIN_TIER = "daemon"
# Auth plugins work without an active session/provider connection.
# The daemon loads them at startup so commands are available immediately.
SESSION_INDEPENDENT = True

from .plugin import OVHcloudAuthPlugin, create_plugin

__all__ = ["OVHcloudAuthPlugin", "create_plugin", "PLUGIN_KIND", "PLUGIN_TIER", "SESSION_INDEPENDENT"]
