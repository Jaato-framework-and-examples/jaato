"""Hybrid GC Plugin - Generational garbage collection.

This plugin implements a hybrid GC strategy inspired by Java's generational GC:
- Young generation (recent turns): Always preserved intact
- Old generation (middle-aged turns): Summarized for compression
- Ancient (very old turns): Truncated/removed entirely

Usage:
    from shared.plugins.gc_hybrid import create_plugin

    def my_summarizer(conversation: str) -> str:
        return model.generate(f"Summarize: {conversation}")

    plugin = create_plugin()
    plugin.initialize({
        "preserve_recent_turns": 5,
        "summarize_middle_turns": 15,
        "summarizer": my_summarizer
    })
    client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))
"""

from .plugin import HybridGCPlugin, create_plugin

__all__ = ["HybridGCPlugin", "create_plugin"]
