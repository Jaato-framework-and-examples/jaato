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
    create_gc_notification_message,
    create_summary_message,
    estimate_message_tokens,
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


def load_gc_from_file(
    file_path: str = ".jaato/gc.json"
) -> Optional[tuple["GCPlugin", "GCConfig"]]:
    """Load GC configuration from a JSON file.

    Loads GC configuration from a JSON file (default: .jaato/gc.json)
    and returns an initialized GC plugin with its config.

    The JSON file should have this structure:
        {
            "type": "hybrid",          // "truncate", "summarize", or "hybrid"
            "threshold_percent": 80.0,
            "preserve_recent_turns": 5,
            "notify_on_gc": true,
            "summarize_middle_turns": 10,  // For hybrid strategy
            "max_turns": null,
            "plugin_config": {}
        }

    Args:
        file_path: Path to the JSON config file (default: .jaato/gc.json).

    Returns:
        Tuple of (GCPlugin, GCConfig) if file exists and is valid,
        None if file doesn't exist or is invalid.

    Example:
        result = load_gc_from_file()
        if result:
            gc_plugin, gc_config = result
            client.set_gc_plugin(gc_plugin, gc_config)
    """
    import json
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    config_path = Path(file_path)
    if not config_path.exists():
        return None

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        gc_type = data.get('type', 'truncate')
        # Map gc type names (e.g., "truncate" -> "gc_truncate")
        gc_plugin_name = gc_type if gc_type.startswith('gc_') else f'gc_{gc_type}'

        # Build plugin init config
        gc_init_config = {
            'preserve_recent_turns': data.get('preserve_recent_turns', 5),
            'notify_on_gc': data.get('notify_on_gc', True),
        }
        if data.get('summarize_middle_turns') is not None:
            gc_init_config['summarize_middle_turns'] = data['summarize_middle_turns']
        # Merge plugin-specific config
        gc_init_config.update(data.get('plugin_config') or {})

        gc_plugin = load_gc_plugin(gc_plugin_name, gc_init_config)

        # Create GCConfig for the client
        gc_config = GCConfig(
            threshold_percent=data.get('threshold_percent', 80.0),
            max_turns=data.get('max_turns'),
            preserve_recent_turns=data.get('preserve_recent_turns', 5),
            plugin_config=data.get('plugin_config') or {},
        )

        logger.info("Loaded GC config from %s: type=%s", file_path, gc_type)
        return gc_plugin, gc_config

    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in GC config file %s: %s", file_path, e)
        return None
    except ValueError as e:
        logger.warning("Failed to load GC plugin from %s: %s", file_path, e)
        return None
    except Exception as e:
        logger.warning("Error reading GC config file %s: %s", file_path, e)
        return None


__all__ = [
    # Core types
    "GCPlugin",
    "GCConfig",
    "GCResult",
    "GCTriggerReason",
    # Discovery
    "discover_gc_plugins",
    "load_gc_plugin",
    "load_gc_from_file",
    "GC_PLUGIN_ENTRY_POINT",
    # Utilities
    "Turn",
    "split_into_turns",
    "flatten_turns",
    "estimate_message_tokens",
    "estimate_turn_tokens",
    "estimate_history_tokens",
    "create_summary_message",
    "create_gc_notification_message",
    "get_preserved_indices",
]
