"""Truncate GC Plugin - Simple turn-based garbage collection.

This plugin implements the simplest GC strategy: remove oldest turns
while preserving the most recent N turns. No summarization, minimal
overhead, fast execution.

Usage:
    from shared.plugins.gc_truncate import create_plugin

    plugin = create_plugin()
    plugin.initialize({"preserve_recent_turns": 10})
    client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "gc"

from .plugin import TruncateGCPlugin, create_plugin

__all__ = ["TruncateGCPlugin", "create_plugin"]
