"""Zhipu AI authentication plugin.

Provides user commands for API key authentication with Z.AI Coding Plan.
"""

# Plugin kind for registry discovery
PLUGIN_KIND = "tool"

from .plugin import ZhipuAIAuthPlugin, create_plugin

__all__ = ["ZhipuAIAuthPlugin", "create_plugin", "PLUGIN_KIND"]
