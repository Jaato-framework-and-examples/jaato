"""Anthropic Cache Plugin — explicit breakpoint caching for Claude models.

Decouples cache strategy from provider internals, enabling budget-aware
breakpoint placement and extended TTL support (5m default or 1h).
"""

PLUGIN_KIND = "cache"

PLUGIN_TIER = "daemon"
from .plugin import AnthropicCachePlugin, create_plugin

__all__ = ["PLUGIN_KIND", "PLUGIN_TIER", "AnthropicCachePlugin", "create_plugin"]
