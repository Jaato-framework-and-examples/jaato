"""Memory plugin for model self-curated persistent memory.

This plugin enables the model to:
- Store valuable explanations and insights for future reference
- Retrieve stored memories when relevant to current conversations
- Build a persistent knowledge base over time

The plugin uses a two-phase retrieval system:
1. Prompt enrichment: Lightweight hints about available memories are injected
2. Model-driven retrieval: Model decides whether to fetch full content

Usage:
    # Plugin is auto-discovered by PluginRegistry
    # Configure in your code or config file:
    registry.expose_plugin("memory", config={
        "storage_path": ".jaato/memories.jsonl"
    })
"""

from .plugin import MemoryPlugin, create_plugin

__all__ = ["MemoryPlugin", "create_plugin"]
