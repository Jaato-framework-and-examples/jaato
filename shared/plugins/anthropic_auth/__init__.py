"""Anthropic authentication plugin.

Provides user commands for OAuth authentication with Claude Pro/Max subscriptions.
"""

PLUGIN_KIND = "tool"

# Auth plugins work without an active session/provider connection.
# The daemon loads them at startup so commands are available immediately.
SESSION_INDEPENDENT = True

from .plugin import create_plugin, AnthropicAuthPlugin

__all__ = ["create_plugin", "AnthropicAuthPlugin", "PLUGIN_KIND", "SESSION_INDEPENDENT"]
