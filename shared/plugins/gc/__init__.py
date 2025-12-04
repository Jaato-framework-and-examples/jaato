"""Context Garbage Collection plugin infrastructure.

This module provides the base types and protocol for implementing
GC strategy plugins that manage conversation history to prevent
context window overflow.

GC plugins implement different strategies:
- Truncation: Remove oldest turns
- Summarization: Compress old turns into summaries
- Hybrid: Combine truncation and summarization

Usage:
    from shared.plugins.gc import GCPlugin, GCConfig, GCResult, discover_gc_plugins

    # Discover available GC plugins
    plugins = discover_gc_plugins()
    print(plugins)  # {'gc_truncate': <factory>, 'gc_summarize': <factory>, ...}

    # Load and configure a GC plugin
    gc_plugin = load_gc_plugin('gc_truncate')
    gc_plugin.initialize({"preserve_recent_turns": 10})

    # Set on JaatoClient
    client.set_gc_plugin(gc_plugin, GCConfig(threshold_percent=75.0))
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "gc"

import sys
from typing import Callable, Dict, Optional

from .base import (
    GCConfig,
    GCPlugin,
    GCResult,
    GCTriggerReason,
)
from .utils import (
    Turn,
    create_gc_notification_content,
    create_summary_content,
    estimate_content_tokens,
    estimate_history_tokens,
    estimate_turn_tokens,
    flatten_turns,
    get_preserved_indices,
    split_into_turns,
)


# Entry point group for GC plugins
GC_PLUGIN_ENTRY_POINT = "jaato.gc_plugins"


def discover_gc_plugins() -> Dict[str, Callable[[], GCPlugin]]:
    """Discover all available GC plugins via entry points.

    Returns:
        Dict mapping plugin names to their factory functions.

    Example:
        plugins = discover_gc_plugins()
        # {'gc_truncate': <function>, 'gc_summarize': <function>, ...}
    """
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points
        eps = entry_points(group=GC_PLUGIN_ENTRY_POINT)
    else:
        from importlib.metadata import entry_points
        all_eps = entry_points()
        eps = all_eps.get(GC_PLUGIN_ENTRY_POINT, [])

    plugins: Dict[str, Callable[[], GCPlugin]] = {}
    for ep in eps:
        try:
            factory = ep.load()
            plugins[ep.name] = factory
        except Exception:
            # Skip plugins that fail to load
            pass

    return plugins


def load_gc_plugin(name: str, config: Optional[Dict] = None) -> GCPlugin:
    """Load a GC plugin by name and optionally initialize it.

    Args:
        name: The plugin name (e.g., 'gc_truncate', 'gc_summarize').
        config: Optional configuration to pass to initialize().

    Returns:
        An initialized GCPlugin instance.

    Raises:
        ValueError: If the plugin is not found.
    """
    plugins = discover_gc_plugins()

    if name not in plugins:
        available = list(plugins.keys())
        raise ValueError(
            f"GC plugin '{name}' not found. Available: {available}"
        )

    plugin = plugins[name]()
    plugin.initialize(config)
    return plugin


__all__ = [
    # Core types
    "GCPlugin",
    "GCConfig",
    "GCResult",
    "GCTriggerReason",
    # Discovery
    "discover_gc_plugins",
    "load_gc_plugin",
    "GC_PLUGIN_ENTRY_POINT",
    # Utilities
    "Turn",
    "split_into_turns",
    "flatten_turns",
    "estimate_content_tokens",
    "estimate_turn_tokens",
    "estimate_history_tokens",
    "create_summary_content",
    "create_gc_notification_content",
    "get_preserved_indices",
]
