"""Anthropic authentication plugin.

Provides user commands for OAuth authentication with Claude Pro/Max subscriptions.
"""

PLUGIN_KIND = "tool"

from .plugin import create_plugin, AnthropicAuthPlugin

__all__ = ["create_plugin", "AnthropicAuthPlugin", "PLUGIN_KIND"]
