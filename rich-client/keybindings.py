"""Keybinding configuration for the rich client.

Allows users to customize keybindings via:
1. JSON config file: .jaato/keybindings.json (project-level)
                     ~/.jaato/keybindings.json (user-level fallback)
2. Environment variables: JAATO_KEY_<ACTION>=<key>

Key syntax follows prompt_toolkit conventions:
- Simple keys: "enter", "space", "tab", "q", "v"
- Control: "c-c", "c-d", "c-p" (Ctrl+C, Ctrl+D, Ctrl+P)
- Function keys: "f1", "f2", "f12"
- Special: "pageup", "pagedown", "home", "end", "up", "down"
- Multi-key sequences: ["escape", "enter"] for Escape then Enter
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Type alias for keybinding: either a single key or a sequence
KeyBinding = Union[str, List[str]]


def normalize_key(key: KeyBinding) -> KeyBinding:
    """Normalize a keybinding to consistent format.

    Handles string parsing like "escape enter" -> ["escape", "enter"]
    """
    if isinstance(key, list):
        return key
    # Check for space-separated multi-key sequence
    if " " in key and not key.startswith(" ") and not key.endswith(" "):
        parts = key.split()
        if len(parts) > 1:
            return parts
    return key


def key_to_args(key: KeyBinding) -> tuple:
    """Convert a KeyBinding to args suitable for kb.add()."""
    if isinstance(key, list):
        return tuple(key)
    return (key,)


# Default keybindings matching current pt_display.py behavior
DEFAULT_KEYBINDINGS = {
    # Input submission
    "submit": "enter",
    "newline": ["escape", "enter"],
    "clear_input": ["escape", "escape"],

    # Exit/cancel
    "cancel": "c-c",
    "exit": "c-d",

    # Scrolling
    "scroll_up": "pageup",
    "scroll_down": "pagedown",
    "scroll_top": "home",
    "scroll_bottom": "end",

    # Navigation (also used for history when not in popup mode)
    "nav_up": "up",
    "nav_down": "down",

    # Pager
    "pager_quit": "q",
    "pager_next": "space",

    # Features
    "toggle_plan": "c-p",
    "toggle_tools": "c-t",
    "cycle_agents": "f2",
    "yank": "c-y",
    "view_full": "v",
}


@dataclass
class KeybindingConfig:
    """Configuration for keybindings in the rich client.

    All keybindings can be customized. Values can be:
    - A single key string: "c-c", "enter", "f2"
    - A list for multi-key sequences: ["escape", "enter"]

    When loading from JSON or environment, space-separated strings
    are automatically converted to lists: "escape enter" -> ["escape", "enter"]
    """
    # Input submission
    submit: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["submit"])
    newline: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["newline"].copy() if isinstance(DEFAULT_KEYBINDINGS["newline"], list) else DEFAULT_KEYBINDINGS["newline"])
    clear_input: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["clear_input"].copy() if isinstance(DEFAULT_KEYBINDINGS["clear_input"], list) else DEFAULT_KEYBINDINGS["clear_input"])

    # Exit/cancel
    cancel: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["cancel"])
    exit: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["exit"])

    # Scrolling
    scroll_up: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["scroll_up"])
    scroll_down: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["scroll_down"])
    scroll_top: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["scroll_top"])
    scroll_bottom: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["scroll_bottom"])

    # Navigation
    nav_up: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["nav_up"])
    nav_down: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["nav_down"])

    # Pager
    pager_quit: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["pager_quit"])
    pager_next: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["pager_next"])

    # Features
    toggle_plan: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["toggle_plan"])
    toggle_tools: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["toggle_tools"])
    cycle_agents: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["cycle_agents"])
    yank: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["yank"])
    view_full: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["view_full"])

    def get_key_args(self, action: str) -> tuple:
        """Get the key arguments for kb.add() for a given action.

        Args:
            action: The action name (e.g., "submit", "cancel")

        Returns:
            Tuple of key strings suitable for kb.add(*args)
        """
        key = getattr(self, action, None)
        if key is None:
            raise ValueError(f"Unknown keybinding action: {action}")
        return key_to_args(key)

    @classmethod
    def from_dict(cls, data: Dict[str, KeyBinding]) -> "KeybindingConfig":
        """Create config from a dictionary.

        Unknown keys are ignored. Values are normalized.
        """
        # Get valid field names
        valid_fields = {f for f in DEFAULT_KEYBINDINGS.keys()}

        # Filter and normalize
        kwargs = {}
        for key, value in data.items():
            if key in valid_fields:
                kwargs[key] = normalize_key(value)
            else:
                logger.warning(f"Unknown keybinding action '{key}' - ignoring")

        return cls(**kwargs)

    @classmethod
    def from_env(cls) -> "KeybindingConfig":
        """Create config from environment variables.

        Environment variables use the format JAATO_KEY_<ACTION>=<key>
        Examples:
            JAATO_KEY_SUBMIT=enter
            JAATO_KEY_CANCEL=c-c
            JAATO_KEY_NEWLINE=escape enter
        """
        prefix = "JAATO_KEY_"
        data = {}

        for env_key, value in os.environ.items():
            if env_key.startswith(prefix):
                action = env_key[len(prefix):].lower()
                data[action] = normalize_key(value)

        if not data:
            return cls()

        return cls.from_dict(data)

    @classmethod
    def from_file(
        cls,
        project_path: str = ".jaato/keybindings.json",
        user_path: Optional[str] = None
    ) -> Optional["KeybindingConfig"]:
        """Load keybinding config from a JSON file.

        Tries project-level config first, then falls back to user-level.

        Args:
            project_path: Project-level config path (default: .jaato/keybindings.json)
            user_path: User-level config path (default: ~/.jaato/keybindings.json)

        Returns:
            KeybindingConfig if a config file was found and loaded, None otherwise.
        """
        if user_path is None:
            user_path = str(Path.home() / ".jaato" / "keybindings.json")

        # Try project path first
        for path in [project_path, user_path]:
            config_path = Path(path)
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)

                    logger.info(f"Loaded keybindings from {path}")
                    return cls.from_dict(data)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in keybindings file {path}: {e}")
                except Exception as e:
                    logger.warning(f"Error reading keybindings file {path}: {e}")

        return None

    def to_dict(self) -> Dict[str, KeyBinding]:
        """Export configuration to a dictionary."""
        return {
            "submit": self.submit,
            "newline": self.newline,
            "clear_input": self.clear_input,
            "cancel": self.cancel,
            "exit": self.exit,
            "scroll_up": self.scroll_up,
            "scroll_down": self.scroll_down,
            "scroll_top": self.scroll_top,
            "scroll_bottom": self.scroll_bottom,
            "nav_up": self.nav_up,
            "nav_down": self.nav_down,
            "pager_quit": self.pager_quit,
            "pager_next": self.pager_next,
            "toggle_plan": self.toggle_plan,
            "toggle_tools": self.toggle_tools,
            "cycle_agents": self.cycle_agents,
            "yank": self.yank,
            "view_full": self.view_full,
        }


def load_keybindings(
    project_path: str = ".jaato/keybindings.json",
    user_path: Optional[str] = None
) -> KeybindingConfig:
    """Load keybindings with fallback chain.

    Priority order:
    1. Environment variables (JAATO_KEY_*)
    2. Project-level file (.jaato/keybindings.json)
    3. User-level file (~/.jaato/keybindings.json)
    4. Default values

    Environment variables override file-based settings.

    Args:
        project_path: Project-level config path
        user_path: User-level config path

    Returns:
        Merged KeybindingConfig with all sources applied.
    """
    # Start with defaults
    config = KeybindingConfig()

    # Try to load from file (project or user level)
    file_config = KeybindingConfig.from_file(project_path, user_path)
    if file_config:
        config = file_config

    # Apply environment variable overrides
    env_config = KeybindingConfig.from_env()
    env_data = {}
    for key in DEFAULT_KEYBINDINGS.keys():
        env_val = getattr(env_config, key)
        default_val = DEFAULT_KEYBINDINGS[key]
        # Check if env actually set this (not just default)
        if env_val != default_val or os.environ.get(f"JAATO_KEY_{key.upper()}"):
            env_key = f"JAATO_KEY_{key.upper()}"
            if env_key in os.environ:
                env_data[key] = env_val

    # Merge env overrides
    if env_data:
        merged = config.to_dict()
        merged.update(env_data)
        config = KeybindingConfig.from_dict(merged)
        logger.info(f"Applied environment keybinding overrides: {list(env_data.keys())}")

    return config


def generate_example_config() -> str:
    """Generate an example keybindings.json with comments.

    Returns JSON string suitable for creating a template file.
    """
    example = {
        "_comment": "Keybinding configuration for jaato rich client",
        "_syntax": "Keys use prompt_toolkit syntax: 'c-c' for Ctrl+C, 'f2' for F2, etc.",
        "_multi_key": "Multi-key sequences: ['escape', 'enter'] or 'escape enter'",

        "submit": "enter",
        "newline": ["escape", "enter"],
        "clear_input": ["escape", "escape"],

        "cancel": "c-c",
        "exit": "c-d",

        "scroll_up": "pageup",
        "scroll_down": "pagedown",
        "scroll_top": "home",
        "scroll_bottom": "end",

        "nav_up": "up",
        "nav_down": "down",

        "pager_quit": "q",
        "pager_next": "space",

        "toggle_plan": "c-p",
        "toggle_tools": "c-t",
        "cycle_agents": "f2",
        "yank": "c-y",
        "view_full": "v",
    }
    return json.dumps(example, indent=2)
