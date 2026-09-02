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

PLUGIN_TIER = "daemon"
import sys
from typing import Callable, Dict, List, Optional

from .base import (
    GCConfig,
    GCPlugin,
    GCRemovalItem,
    GCResult,
    GCTriggerReason,
)
from .utils import (
    Turn,
    create_gc_notification_message,
    create_summary_message,
    ensure_tool_call_integrity,
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
    file_path: Optional[str] = None,
    agent_name: Optional[str] = None,
    workspace_root: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[tuple["GCPlugin", "GCConfig"]]:
    """Load GC configuration from a JSON file.

    Loads GC configuration from a JSON file and returns an initialized
    GC plugin with its config.

    Resolution order (when ``file_path`` is not provided):
        1. ``<config_root>/gc.json`` if ``config_root`` is set,
           else ``<workspace_root>/.jaato/gc.json`` if ``workspace_root``
           is set, else ``./.jaato/gc.json`` (relative to cwd, legacy
           behavior).
        2. ``~/.jaato/gc.json`` (user-level fallback).

    When ``file_path`` is given:
        - Absolute paths are used as-is.
        - Relative paths are resolved against ``workspace_root`` when set,
          otherwise against the current working directory.

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
        file_path: Optional explicit path to the JSON config file. When None,
            the workspace + user fallback search order described above is used.
        agent_name: Optional agent name for trace logging identification.
        workspace_root: Absolute path to the session's workspace. Required for
            correct resolution in daemon mode where ``os.getcwd()`` reflects
            the daemon's startup directory, not the client's workspace.

    Returns:
        Tuple of (GCPlugin, GCConfig) if a config file exists and is valid,
        None if no candidate file exists or the file is invalid.

    Example:
        result = load_gc_from_file(agent_name="main", workspace_root="/path/to/ws")
        if result:
            gc_plugin, gc_config = result
            client.set_gc_plugin(gc_plugin, gc_config)
    """
    import json
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    candidates: list[Path] = []
    if file_path is not None:
        explicit = Path(file_path)
        if not explicit.is_absolute() and workspace_root:
            explicit = Path(workspace_root) / explicit
        candidates.append(explicit)
    else:
        if config_root:
            candidates.append(Path(config_root).expanduser().resolve() / "gc.json")
        elif workspace_root:
            candidates.append(Path(workspace_root) / ".jaato" / "gc.json")
        else:
            candidates.append(Path(".jaato") / "gc.json")
        candidates.append(Path.home() / ".jaato" / "gc.json")

    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
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
        if agent_name:
            gc_init_config['agent_name'] = agent_name
        if data.get('summarize_middle_turns') is not None:
            gc_init_config['summarize_middle_turns'] = data['summarize_middle_turns']
        if data.get('target_percent') is not None:
            gc_init_config['target_percent'] = data['target_percent']
        if 'pressure_percent' in data:
            gc_init_config['pressure_percent'] = data['pressure_percent']
        # Merge plugin-specific config
        gc_init_config.update(data.get('plugin_config') or {})

        gc_plugin = load_gc_plugin(gc_plugin_name, gc_init_config)

        # Create GCConfig for the client
        # Handle pressure_percent specially: 0 means continuous mode, use None
        pressure_percent = data.get('pressure_percent')
        if pressure_percent == 0:
            pressure_percent = None

        gc_config = GCConfig(
            threshold_percent=data.get('threshold_percent', 80.0),
            target_percent=data.get('target_percent', 60.0),
            pressure_percent=pressure_percent,
            max_turns=data.get('max_turns'),
            preserve_recent_turns=data.get('preserve_recent_turns', 5),
            plugin_config=data.get('plugin_config') or {},
        )

        logger.info("Loaded GC config from %s: type=%s", config_path, gc_type)
        return gc_plugin, gc_config

    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in GC config file %s: %s", config_path, e)
        return None
    except ValueError as e:
        logger.warning("Failed to load GC plugin from %s: %s", config_path, e)
        return None
    except Exception as e:
        logger.warning("Error reading GC config file %s: %s", config_path, e)
        return None


def get_gc_apparmor_rules() -> List[str]:
    """Return AppArmor rules required by any gc plugin.

    Phase 3b of the plugin-apparmor-contribution refactor
    (template v25, 2026-05-16).  The ``~/.jaato/gc.json`` user-level
    fallback was previously hardcoded in ``apparmor.py:PROFILE_TEMPLATE``;
    sessions whose profile has no ``gc`` field no longer carry the
    grant (least-privilege).

    Surfaced as a module-level function rather than a per-plugin
    classmethod because:
    - The rule is shared by all 4 gc strategies (``load_gc_from_file``
      reads the same path regardless of the active strategy).
    - ``GCPlugin`` is a structural ``Protocol`` with no concrete base
      class to hang a shared classmethod on.

    The daemon-side resolver
    (``server.apparmor.resolve_plugin_apparmor_rules``) calls this
    function when ``profile.gc`` is set.
    """
    return ["@{HOME}/.jaato/gc.json  r,"]


__all__ = [
    # Core types
    "GCPlugin",
    "GCConfig",
    "GCRemovalItem",
    "GCResult",
    "GCTriggerReason",
    # Discovery
    "discover_gc_plugins",
    "load_gc_plugin",
    "load_gc_from_file",
    "get_gc_apparmor_rules",
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
    "ensure_tool_call_integrity",
    "get_preserved_indices",
]
