"""Configuration loader for session persistence plugin.

Loads session configuration from .jaato/.sessions.json if it exists,
otherwise uses default configuration.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .base import SessionConfig


# Default location for session config file
DEFAULT_CONFIG_PATH = ".jaato/.sessions.json"


def load_session_config(
    config_path: Optional[str] = None,
    base_path: Optional[str] = None
) -> SessionConfig:
    """Load session configuration from a JSON file.

    Searches for config file in this order:
    1. Explicit config_path if provided
    2. JAATO_SESSION_CONFIG environment variable
    3. .jaato/.sessions.json in base_path (or cwd)

    If no config file is found, returns default configuration.

    Args:
        config_path: Optional explicit path to config file.
        base_path: Base directory for relative paths (default: cwd).

    Returns:
        SessionConfig with loaded or default values.

    Example config file (.jaato/.sessions.json):
    ```json
    {
        "storage_path": ".jaato/sessions",
        "auto_save_on_exit": true,
        "auto_save_interval": null,
        "checkpoint_after_turns": 10,
        "auto_resume_last": false,
        "request_description_after_turns": 3,
        "max_sessions": 20
    }
    ```
    """
    base_path = base_path or os.getcwd()

    # Determine config file path
    if config_path:
        file_path = Path(config_path)
        if not file_path.is_absolute():
            file_path = Path(base_path) / file_path
    elif os.environ.get("JAATO_SESSION_CONFIG"):
        file_path = Path(os.environ["JAATO_SESSION_CONFIG"])
        if not file_path.is_absolute():
            file_path = Path(base_path) / file_path
    else:
        file_path = Path(base_path) / DEFAULT_CONFIG_PATH

    # Try to load the config file
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return _parse_config(data, base_path)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[SessionConfig] Warning: Failed to load {file_path}: {e}")
            return SessionConfig()

    # No config file found, use defaults
    return SessionConfig()


def _parse_config(data: Dict[str, Any], base_path: str) -> SessionConfig:
    """Parse configuration dictionary into SessionConfig.

    Args:
        data: Dictionary from JSON config file.
        base_path: Base directory for relative paths.

    Returns:
        SessionConfig with parsed values.
    """
    # Handle storage_path relative to base_path
    storage_path = data.get("storage_path", ".jaato/sessions")
    if not Path(storage_path).is_absolute():
        storage_path = str(Path(base_path) / storage_path)

    return SessionConfig(
        storage_path=storage_path,
        auto_save_on_exit=data.get("auto_save_on_exit", True),
        auto_save_interval=data.get("auto_save_interval"),
        checkpoint_after_turns=data.get("checkpoint_after_turns"),
        auto_resume_last=data.get("auto_resume_last", False),
        request_description_after_turns=data.get("request_description_after_turns", 3),
        max_sessions=data.get("max_sessions", 20),
        plugin_config=data.get("plugin_config", {}),
    )


def save_session_config(
    config: SessionConfig,
    config_path: Optional[str] = None,
    base_path: Optional[str] = None
) -> None:
    """Save session configuration to a JSON file.

    Args:
        config: SessionConfig to save.
        config_path: Optional path for config file.
        base_path: Base directory for relative paths (default: cwd).
    """
    base_path = base_path or os.getcwd()

    if config_path:
        file_path = Path(config_path)
        if not file_path.is_absolute():
            file_path = Path(base_path) / file_path
    else:
        file_path = Path(base_path) / DEFAULT_CONFIG_PATH

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Build config dict
    data = {
        "storage_path": config.storage_path,
        "auto_save_on_exit": config.auto_save_on_exit,
        "auto_save_interval": config.auto_save_interval,
        "checkpoint_after_turns": config.checkpoint_after_turns,
        "auto_resume_last": config.auto_resume_last,
        "request_description_after_turns": config.request_description_after_turns,
        "max_sessions": config.max_sessions,
    }

    if config.plugin_config:
        data["plugin_config"] = config.plugin_config

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
