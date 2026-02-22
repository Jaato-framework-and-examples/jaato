"""ZhipuAI Cache Plugin — monitoring-only for implicit caching.

ZhipuAI uses fully automatic/implicit caching that requires no
annotations.  This plugin tracks cache hit rates and GC
invalidations without modifying requests.
"""

PLUGIN_KIND = "cache"

from .plugin import ZhipuAICachePlugin, create_plugin

__all__ = ["PLUGIN_KIND", "ZhipuAICachePlugin", "create_plugin"]
