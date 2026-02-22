"""Google GenAI Cache Plugin — monitoring for implicit context caching.

Google GenAI (Gemini) uses automatic context caching where the system
detects repeated prefixes across requests.  This plugin tracks cache
hit metrics and GC invalidations without modifying requests.

Future enhancement: Active caching via the ``google.genai.caching``
API (create/manage ``CachedContent`` objects) for explicit prefix
pinning.
"""

PLUGIN_KIND = "cache"

from .plugin import GoogleGenAICachePlugin, create_plugin

__all__ = ["PLUGIN_KIND", "GoogleGenAICachePlugin", "create_plugin"]
