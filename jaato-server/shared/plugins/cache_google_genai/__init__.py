"""Google GenAI Cache Plugin — explicit context caching via CachedContent.

When ``enable_caching`` is set in provider config, this plugin creates
a server-side ``CachedContent`` object containing the session's system
instruction and tool definitions.  Subsequent requests reference the
cache by name, avoiding re-transmission of the (potentially large)
prefix.

When caching is disabled or the content falls below the minimum token
threshold (~32K tokens), the plugin operates in monitoring-only mode
and tracks cache metrics without modifying requests.
"""

PLUGIN_KIND = "cache"

from .plugin import GoogleGenAICachePlugin, create_plugin

__all__ = ["PLUGIN_KIND", "GoogleGenAICachePlugin", "create_plugin"]
