"""Cache Control plugin infrastructure.

This module provides the base types and protocol for implementing
provider-specific cache control plugins that manage prompt caching
strategies (e.g., Anthropic explicit breakpoints, ZhipuAI monitoring).

Cache plugins are the fifth plugin kind alongside tool, gc,
model_provider, and session plugins.

Usage:
    from shared.plugins.cache import CachePlugin, discover_cache_plugins

    # Discover available cache plugins
    plugins = discover_cache_plugins()
    # {'cache_anthropic': <factory>, 'cache_zhipuai': <factory>}

    # Load by provider name
    plugin = load_cache_plugin_for_provider('anthropic')
    plugin.initialize({'enable_caching': True, 'cache_ttl': '5m'})
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "cache"

import sys
from typing import Callable, Dict, Optional

from .base import CachePlugin

# Entry point group for cache plugins
CACHE_PLUGIN_ENTRY_POINT = "jaato.cache_plugins"


def discover_cache_plugins() -> Dict[str, Callable[[], CachePlugin]]:
    """Discover all available cache plugins via entry points.

    Returns:
        Dict mapping plugin names to their factory functions.

    Example:
        plugins = discover_cache_plugins()
        # {'cache_anthropic': <factory>, 'cache_zhipuai': <factory>}
    """
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points
        eps = entry_points(group=CACHE_PLUGIN_ENTRY_POINT)
    else:
        from importlib.metadata import entry_points
        all_eps = entry_points()
        eps = all_eps.get(CACHE_PLUGIN_ENTRY_POINT, [])

    plugins: Dict[str, Callable[[], CachePlugin]] = {}
    for ep in eps:
        try:
            factory = ep.load()
            plugins[ep.name] = factory
        except Exception:
            # Skip plugins that fail to load
            pass

    return plugins


def load_cache_plugin(name: str, config: Optional[Dict] = None) -> CachePlugin:
    """Load a cache plugin by name and optionally initialize it.

    Args:
        name: The plugin name (e.g., 'cache_anthropic', 'cache_zhipuai').
        config: Optional configuration to pass to initialize().

    Returns:
        An initialized CachePlugin instance.

    Raises:
        ValueError: If the plugin is not found.
    """
    plugins = discover_cache_plugins()

    if name not in plugins:
        available = list(plugins.keys())
        raise ValueError(
            f"Cache plugin '{name}' not found. Available: {available}"
        )

    plugin = plugins[name]()
    plugin.initialize(config)
    return plugin


def load_cache_plugin_for_provider(
    provider_name: str,
    config: Optional[Dict] = None
) -> Optional[CachePlugin]:
    """Find and load the cache plugin matching a provider name.

    Iterates through discovered cache plugins and returns the one whose
    ``provider_name`` property matches.

    Args:
        provider_name: Provider identifier (e.g., 'anthropic', 'zhipuai').
        config: Optional configuration to pass to initialize().

    Returns:
        An initialized CachePlugin, or None if no match found.
    """
    plugins = discover_cache_plugins()

    for _name, factory in plugins.items():
        try:
            plugin = factory()
            if plugin.provider_name == provider_name:
                plugin.initialize(config)
                return plugin
        except Exception:
            # Skip plugins that fail to instantiate
            continue

    return None


__all__ = [
    # Core types
    "CachePlugin",
    # Discovery
    "discover_cache_plugins",
    "load_cache_plugin",
    "load_cache_plugin_for_provider",
    "CACHE_PLUGIN_ENTRY_POINT",
]
