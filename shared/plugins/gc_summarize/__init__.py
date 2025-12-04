"""Summarize GC Plugin - Compression-based garbage collection.

This plugin implements a summarization GC strategy: compress old turns
into a summary rather than removing them entirely. Preserves context
information while freeing token space.

Usage:
    from shared.plugins.gc_summarize import create_plugin

    def my_summarizer(conversation: str) -> str:
        # Use your model to generate summary
        return model.generate(f"Summarize: {conversation}")

    plugin = create_plugin()
    plugin.initialize({
        "preserve_recent_turns": 10,
        "summarizer": my_summarizer
    })
    client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))
"""

from .plugin import SummarizeGCPlugin, create_plugin, DEFAULT_SUMMARIZE_PROMPT

__all__ = ["SummarizeGCPlugin", "create_plugin", "DEFAULT_SUMMARIZE_PROMPT"]
