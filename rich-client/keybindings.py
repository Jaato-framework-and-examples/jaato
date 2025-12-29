"""Keybinding configuration for the rich client.

Allows users to customize keybindings via:
1. JSON config file: .jaato/keybindings.json (project-level)
                     ~/.jaato/keybindings.json (user-level fallback)
2. Terminal-specific profiles: .jaato/keybindings.<terminal>.json
3. Environment variables: JAATO_KEY_<ACTION>=<key>
4. Profile override: JAATO_KEYBINDING_PROFILE=<profile>

Key syntax follows prompt_toolkit conventions:
- Simple keys: "enter", "space", "tab", "q", "v"
- Control: "c-c", "c-d", "c-p" (Ctrl+C, Ctrl+D, Ctrl+P)
- Function keys: "f1", "f2", "f12"
- Special: "pageup", "pagedown", "home", "end", "up", "down"
- Multi-key sequences: ["escape", "enter"] for Escape then Enter

Terminal profiles allow different keybindings for different terminals:
- Auto-detected from $TERM_PROGRAM, $TERMINAL, $TERM, etc.
- Can be overridden with JAATO_KEYBINDING_PROFILE environment variable
- Profile-specific files: .jaato/keybindings.tmux.json, etc.
- Or embedded in main config under "_profiles" key
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


def format_key_for_display(key: KeyBinding) -> str:
    """Format a keybinding for human-readable display.

    Converts prompt_toolkit key syntax to user-friendly format:
    - "c-a" -> "Ctrl+A"
    - "f2" -> "F2"
    - "escape" -> "Esc"
    - ["escape", "enter"] -> "Esc Enter"

    Args:
        key: The keybinding in prompt_toolkit format.

    Returns:
        Human-readable string representation.
    """
    if isinstance(key, list):
        return " ".join(format_key_for_display(k) for k in key)

    key_str = str(key).lower()

    # Control keys: c-x -> Ctrl+X
    if key_str.startswith("c-"):
        char = key_str[2:].upper()
        return f"Ctrl+{char}"

    # Function keys: f1 -> F1
    if key_str.startswith("f") and key_str[1:].isdigit():
        return key_str.upper()

    # Special keys
    special_keys = {
        "escape": "Esc",
        "enter": "Enter",
        "space": "Space",
        "tab": "Tab",
        "pageup": "PgUp",
        "pagedown": "PgDn",
        "home": "Home",
        "end": "End",
        "up": "Up",
        "down": "Down",
        "left": "Left",
        "right": "Right",
    }

    if key_str in special_keys:
        return special_keys[key_str]

    # Single character keys
    if len(key_str) == 1:
        return key_str.upper()

    # Default: capitalize
    return key_str.capitalize()


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
    "cycle_agents": "c-a",
    "yank": "c-y",
    "view_full": "v",

    # Tool navigation
    "tool_nav_enter": "c-n",  # Enter tool navigation mode
    "tool_expand": "right",   # Expand selected tool's output
    "tool_collapse": "left",  # Collapse selected tool's output
    "tool_exit": "escape",    # Exit tool navigation mode
    "tool_output_up": "up",   # Scroll up within expanded tool output (same as nav_up)
    "tool_output_down": "down",  # Scroll down within expanded tool output (same as nav_down)
}


def detect_terminal() -> str:
    """Detect the current terminal type from environment variables.

    Returns:
        Terminal identifier string (lowercase), e.g., "tmux", "iterm2", "vscode", etc.
        Returns "default" if no specific terminal is detected.
    """
    # Check for explicit profile override first
    override = os.environ.get("JAATO_KEYBINDING_PROFILE", "").strip().lower()
    if override:
        return override

    # Check common terminal environment variables
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    terminal = os.environ.get("TERMINAL", "").lower()
    term = os.environ.get("TERM", "").lower()

    # tmux detection (check TMUX env var exists)
    if os.environ.get("TMUX"):
        return "tmux"

    # screen detection
    if term.startswith("screen"):
        return "screen"

    # iTerm2
    if "iterm" in term_program:
        return "iterm2"

    # VS Code integrated terminal
    if term_program == "vscode" or os.environ.get("VSCODE_INJECTION"):
        return "vscode"

    # Alacritty
    if "alacritty" in term_program or "alacritty" in terminal:
        return "alacritty"

    # Kitty
    if "kitty" in term_program or os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"

    # WezTerm
    if "wezterm" in term_program or os.environ.get("WEZTERM_PANE"):
        return "wezterm"

    # Hyper
    if "hyper" in term_program:
        return "hyper"

    # Windows Terminal
    if os.environ.get("WT_SESSION"):
        return "windows-terminal"

    # Apple Terminal
    if term_program == "apple_terminal":
        return "apple-terminal"

    # GNOME Terminal
    if "gnome-terminal" in terminal or os.environ.get("GNOME_TERMINAL_SCREEN"):
        return "gnome-terminal"

    # Konsole
    if os.environ.get("KONSOLE_VERSION"):
        return "konsole"

    # xterm
    if term.startswith("xterm"):
        return "xterm"

    return "default"


def list_available_profiles(base_dir: str = ".jaato") -> List[str]:
    """List available keybinding profiles.

    Scans for profile-specific config files like keybindings.tmux.json.

    Args:
        base_dir: Directory to scan for profile files.

    Returns:
        List of profile names (e.g., ["default", "tmux", "iterm2"]).
    """
    profiles = ["default"]
    base_path = Path(base_dir)

    if base_path.exists():
        for f in base_path.glob("keybindings.*.json"):
            # Extract profile name from keybindings.<profile>.json
            parts = f.stem.split(".")
            if len(parts) == 2 and parts[0] == "keybindings":
                profile = parts[1]
                if profile not in profiles:
                    profiles.append(profile)

    # Also check user directory
    user_path = Path.home() / ".jaato"
    if user_path.exists():
        for f in user_path.glob("keybindings.*.json"):
            parts = f.stem.split(".")
            if len(parts) == 2 and parts[0] == "keybindings":
                profile = parts[1]
                if profile not in profiles:
                    profiles.append(profile)

    return sorted(profiles)


@dataclass
class KeybindingConfig:
    """Configuration for keybindings in the rich client.

    All keybindings can be customized. Values can be:
    - A single key string: "c-c", "enter", "f2"
    - A list for multi-key sequences: ["escape", "enter"]

    When loading from JSON or environment, space-separated strings
    are automatically converted to lists: "escape enter" -> ["escape", "enter"]

    Supports terminal-specific profiles:
    - Auto-detected from environment (tmux, iterm2, vscode, etc.)
    - Profile-specific files: .jaato/keybindings.tmux.json
    - Embedded profiles: "_profiles": {"tmux": {...}} in main config
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

    # Tool navigation
    tool_nav_enter: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["tool_nav_enter"])
    tool_expand: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["tool_expand"])
    tool_collapse: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["tool_collapse"])
    tool_exit: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["tool_exit"])
    tool_output_up: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["tool_output_up"])
    tool_output_down: KeyBinding = field(default_factory=lambda: DEFAULT_KEYBINDINGS["tool_output_down"])

    # Profile metadata (not a keybinding)
    _profile: str = field(default="default")
    _profile_source: str = field(default="default")  # Where profile was loaded from

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
        user_path: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Optional["KeybindingConfig"]:
        """Load keybinding config from a JSON file with profile support.

        Supports terminal-specific profiles:
        1. Profile-specific file: .jaato/keybindings.<profile>.json
        2. Embedded profile: "_profiles": {"<profile>": {...}} in main config
        3. Falls back to base config if profile not found

        Args:
            project_path: Project-level config path (default: .jaato/keybindings.json)
            user_path: User-level config path (default: ~/.jaato/keybindings.json)
            profile: Profile name to load (default: auto-detect from terminal)

        Returns:
            KeybindingConfig if a config file was found and loaded, None otherwise.
        """
        if user_path is None:
            user_path = str(Path.home() / ".jaato" / "keybindings.json")

        # Auto-detect profile if not specified
        if profile is None:
            profile = detect_terminal()

        project_dir = str(Path(project_path).parent)
        user_dir = str(Path(user_path).parent)

        # Try profile-specific files first (if not default profile)
        if profile != "default":
            profile_paths = [
                f"{project_dir}/keybindings.{profile}.json",
                f"{user_dir}/keybindings.{profile}.json",
            ]
            for path in profile_paths:
                config_path = Path(path)
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            data = json.load(f)

                        logger.info(f"Loaded keybindings profile '{profile}' from {path}")
                        config = cls.from_dict(data)
                        config._profile = profile
                        config._profile_source = path
                        return config

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in keybindings file {path}: {e}")
                    except Exception as e:
                        logger.warning(f"Error reading keybindings file {path}: {e}")

        # Try base config files (with embedded profile support)
        for path in [project_path, user_path]:
            config_path = Path(path)
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)

                    # Check for embedded profile
                    profiles = data.get("_profiles", {})
                    if profile != "default" and profile in profiles:
                        # Merge base config with profile overrides
                        base_data = {k: v for k, v in data.items() if not k.startswith("_")}
                        base_data.update(profiles[profile])
                        logger.info(f"Loaded keybindings profile '{profile}' (embedded) from {path}")
                        config = cls.from_dict(base_data)
                        config._profile = profile
                        config._profile_source = f"{path} [embedded]"
                        return config

                    # Use base config
                    logger.info(f"Loaded keybindings from {path}")
                    config = cls.from_dict(data)
                    config._profile = "default"
                    config._profile_source = path
                    return config

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in keybindings file {path}: {e}")
                except Exception as e:
                    logger.warning(f"Error reading keybindings file {path}: {e}")

        return None

    @property
    def profile(self) -> str:
        """Get the current profile name."""
        return self._profile

    @property
    def profile_source(self) -> str:
        """Get the source file/location of the current profile."""
        return self._profile_source

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
            "tool_nav_enter": self.tool_nav_enter,
            "tool_expand": self.tool_expand,
            "tool_collapse": self.tool_collapse,
            "tool_exit": self.tool_exit,
            "tool_output_up": self.tool_output_up,
            "tool_output_down": self.tool_output_down,
        }

    def set_binding(self, action: str, key: KeyBinding) -> bool:
        """Set a keybinding for a specific action.

        Args:
            action: The action name (e.g., "submit", "cancel")
            key: The key binding (string or list for multi-key)

        Returns:
            True if the action was valid and binding was set.
        """
        if action not in DEFAULT_KEYBINDINGS:
            return False

        normalized = normalize_key(key)
        setattr(self, action, normalized)
        return True

    def save_to_file(
        self,
        path: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> bool:
        """Save current keybindings to a JSON file.

        Args:
            path: Path to save the config file. If None, uses default based on profile.
            profile: Profile name for profile-specific file. If None, uses current profile.
                     Use "default" to save to base keybindings.json.

        Returns:
            True if saved successfully, False otherwise.
        """
        # Determine profile and path
        if profile is None:
            profile = self._profile

        if path is None:
            if profile == "default":
                path = ".jaato/keybindings.json"
            else:
                path = f".jaato/keybindings.{profile}.json"

        config_path = Path(path)

        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load existing file to preserve comments/structure if it exists
            existing_data = {}
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, Exception):
                    pass

            # Update with current bindings
            current = self.to_dict()
            existing_data.update(current)

            # Remove internal keys that start with _
            save_data = {k: v for k, v in existing_data.items() if not k.startswith('_')}

            with open(config_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"Saved keybindings to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save keybindings to {path}: {e}")
            return False


def load_keybindings(
    project_path: str = ".jaato/keybindings.json",
    user_path: Optional[str] = None,
    profile: Optional[str] = None,
) -> KeybindingConfig:
    """Load keybindings with fallback chain and profile support.

    Priority order:
    1. Environment variables (JAATO_KEY_*)
    2. Profile-specific file (.jaato/keybindings.<profile>.json)
    3. Embedded profile in base config
    4. Base config file (.jaato/keybindings.json)
    5. User-level file (~/.jaato/keybindings.json)
    6. Default values

    Profile is auto-detected from terminal environment if not specified.
    Environment variables override file-based settings.

    Args:
        project_path: Project-level config path
        user_path: User-level config path
        profile: Profile name (default: auto-detect from terminal)

    Returns:
        Merged KeybindingConfig with all sources applied.
    """
    # Start with defaults
    config = KeybindingConfig()

    # Try to load from file with profile support
    file_config = KeybindingConfig.from_file(project_path, user_path, profile)
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
        profile_backup = config._profile
        source_backup = config._profile_source
        config = KeybindingConfig.from_dict(merged)
        config._profile = profile_backup
        config._profile_source = source_backup
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
        "cycle_agents": "c-a",
        "yank": "c-y",
        "view_full": "v",

        "tool_nav_enter": "c-n",
        "tool_expand": "right",
        "tool_collapse": "left",
        "tool_exit": "escape",
        "tool_output_up": "up",
        "tool_output_down": "down",
    }
    return json.dumps(example, indent=2)
