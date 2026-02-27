"""NVIDIA NIM authentication plugin.

Provides user commands for API key authentication with NVIDIA NIM.
"""

# Plugin kind for registry discovery
PLUGIN_KIND = "tool"

# Auth plugins work without an active session/provider connection.
# The daemon loads them at startup so commands are available immediately.
SESSION_INDEPENDENT = True

from .plugin import NIMAuthPlugin, create_plugin

__all__ = ["NIMAuthPlugin", "create_plugin", "PLUGIN_KIND", "SESSION_INDEPENDENT"]
