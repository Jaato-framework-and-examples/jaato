"""Configuration for thinking mode plugin.

Thinking mode controls extended reasoning capabilities in AI providers:
- Anthropic: Extended thinking with budget_tokens
- Google Gemini: Thinking mode (Gemini 2.0+)

Configuration is loaded from (in priority order):
1. .jaato/thinking.json (project-level)
2. ~/.jaato/thinking.json (user-level)
3. Built-in defaults
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

# Import ThinkingConfig from provider types (single source of truth)
from jaato_sdk.plugins.model_provider.types import ThinkingConfig


@dataclass
class ThinkingPreset:
    """A named thinking mode preset.

    Attributes:
        enabled: Whether thinking is enabled.
        budget: Token budget for thinking (provider-specific interpretation).
    """
    enabled: bool
    budget: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {"enabled": self.enabled, "budget": self.budget}

    @classmethod
    def from_dict(cls, data: Dict) -> 'ThinkingPreset':
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            budget=data.get("budget", 0)
        )

    def to_config(self) -> ThinkingConfig:
        """Convert preset to ThinkingConfig."""
        return ThinkingConfig(enabled=self.enabled, budget=self.budget)


@dataclass
class ThinkingPluginConfig:
    """Plugin configuration loaded from config file.

    Attributes:
        default: Name of the default preset to use on startup.
        presets: Named presets mapping name -> ThinkingPreset.
    """
    default: str = "off"
    presets: Dict[str, ThinkingPreset] = field(default_factory=dict)

    def get_preset(self, name: str) -> Optional[ThinkingPreset]:
        """Get a preset by name."""
        return self.presets.get(name)

    def get_default_preset(self) -> Optional[ThinkingPreset]:
        """Get the default preset."""
        return self.presets.get(self.default)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "default": self.default,
            "presets": {
                name: preset.to_dict()
                for name, preset in self.presets.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ThinkingPluginConfig':
        """Create from dictionary."""
        presets = {}
        for name, preset_data in data.get("presets", {}).items():
            presets[name] = ThinkingPreset.from_dict(preset_data)

        return cls(
            default=data.get("default", "off"),
            presets=presets or cls._default_presets()
        )

    @classmethod
    def _default_presets(cls) -> Dict[str, ThinkingPreset]:
        """Built-in default presets."""
        return {
            "off": ThinkingPreset(enabled=False, budget=0),
            "on": ThinkingPreset(enabled=True, budget=10_000),
            "deep": ThinkingPreset(enabled=True, budget=25_000),
            "ultra": ThinkingPreset(enabled=True, budget=100_000),
        }

    @classmethod
    def default(cls) -> 'ThinkingPluginConfig':
        """Create default configuration."""
        return cls(
            default="off",
            presets=cls._default_presets()
        )


def load_config(config_path: Optional[str] = None) -> ThinkingPluginConfig:
    """Load thinking configuration from file or defaults.

    Search order:
    1. Explicit config_path argument
    2. .jaato/thinking.json (project-level)
    3. ~/.jaato/thinking.json (user-level)
    4. Built-in defaults

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        ThinkingPluginConfig with loaded or default settings.
    """
    search_paths = [
        config_path,
        ".jaato/thinking.json",
        os.path.expanduser("~/.jaato/thinking.json"),
    ]

    for path in search_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return ThinkingPluginConfig.from_dict(data)
            except (json.JSONDecodeError, IOError):
                # Invalid config, continue to next
                continue

    return ThinkingPluginConfig.default()


def save_config(config: ThinkingPluginConfig, path: str) -> None:
    """Save configuration to file.

    Args:
        config: Configuration to save.
        path: Path to write config file.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
