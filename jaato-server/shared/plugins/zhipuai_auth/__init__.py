"""Zhipu AI authentication plugin.

Provides user commands for API key authentication with Z.AI Coding Plan.
"""

# Plugin kind for registry discovery
PLUGIN_KIND = "tool"

# Auth plugins work without an active session/provider connection.
# The daemon loads them at startup so commands are available immediately.
SESSION_INDEPENDENT = True

from .plugin import ZhipuAIAuthPlugin, create_plugin

__all__ = ["ZhipuAIAuthPlugin", "create_plugin", "PLUGIN_KIND", "SESSION_INDEPENDENT"]
