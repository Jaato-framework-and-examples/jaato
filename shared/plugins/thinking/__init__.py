"""Thinking mode plugin for controlling extended reasoning.

This plugin provides user-facing commands to control thinking/reasoning
modes in AI providers (Anthropic extended thinking, Gemini thinking mode).

The thinking command is explicitly user-only - the model cannot see or
modify its own thinking configuration.

Example usage:

    from shared.plugins.thinking import create_plugin, ThinkingConfig

    # Create and initialize plugin
    plugin = create_plugin()
    plugin.initialize()

    # Connect to session
    plugin.set_session(session)

    # User can now run:
    # /thinking          - Show status
    # /thinking off      - Disable
    # /thinking deep     - Enable deep preset
    # /thinking 50000    - Custom budget

Configuration file (.jaato/thinking.json):

    {
      "default": "off",
      "presets": {
        "off": { "enabled": false, "budget": 0 },
        "on": { "enabled": true, "budget": 10000 },
        "deep": { "enabled": true, "budget": 25000 },
        "ultra": { "enabled": true, "budget": 100000 }
      }
    }
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

# Re-export ThinkingConfig from provider types (single source of truth)
from ..model_provider.types import ThinkingConfig
from .config import (
    ThinkingPluginConfig,
    ThinkingPreset,
    load_config,
    save_config,
)
from .plugin import (
    ThinkingPlugin,
    create_plugin,
)

__all__ = [
    # Config
    'ThinkingConfig',
    'ThinkingPluginConfig',
    'ThinkingPreset',
    'load_config',
    'save_config',
    # Plugin
    'ThinkingPlugin',
    'create_plugin',
]
