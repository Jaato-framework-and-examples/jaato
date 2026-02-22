"""Anthropic Cache Plugin — explicit breakpoint caching for Claude models.

Extracted from AnthropicProvider to decouple cache strategy from provider
inheritance, enabling budget-aware breakpoint placement and clean
ZhipuAI/Ollama inheritance without ``_enable_caching = False`` hacks.
"""

PLUGIN_KIND = "cache"

from .plugin import AnthropicCachePlugin, create_plugin

__all__ = ["PLUGIN_KIND", "AnthropicCachePlugin", "create_plugin"]
