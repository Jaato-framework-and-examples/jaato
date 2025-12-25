"""Anthropic authentication plugin.

Provides user commands for OAuth authentication with Claude Pro/Max subscriptions.
"""

from .plugin import create_plugin, AnthropicAuthPlugin

PLUGIN_KIND = "tool"

__all__ = ["create_plugin", "AnthropicAuthPlugin", "PLUGIN_KIND"]
