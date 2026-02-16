"""User preferences persistence for the rich client.

Stores user preferences to ~/.jaato/preferences.json.
Used by various components (theme, vision capture, etc.) to persist settings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_preferences_path() -> Path:
    """Get the path to the user preferences file."""
    return Path.home() / ".jaato" / "preferences.json"


def _load_all_preferences() -> dict:
    """Load all preferences from file.

    Returns:
        Dictionary of all preferences, empty dict if file doesn't exist.
    """
    prefs_path = _get_preferences_path()
    try:
        if prefs_path.exists():
            with open(prefs_path, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load preferences: {e}")
    return {}


def _save_all_preferences(prefs: dict) -> bool:
    """Save all preferences to file.

    Args:
        prefs: Dictionary of preferences to save.

    Returns:
        True if saved successfully, False otherwise.
    """
    prefs_path = _get_preferences_path()
    try:
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prefs_path, "w") as f:
            json.dump(prefs, f, indent=2)
        return True
    except Exception as e:
        logger.warning(f"Failed to save preferences: {e}")
        return False


def save_preference(key: str, value: Any) -> bool:
    """Save a single preference value.

    Args:
        key: Preference key (e.g., "theme", "vision_format").
        value: Value to save (must be JSON-serializable).

    Returns:
        True if saved successfully, False otherwise.
    """
    prefs = _load_all_preferences()
    prefs[key] = value
    success = _save_all_preferences(prefs)
    if success:
        logger.info(f"Saved preference: {key}={value}")
    return success


def load_preference(key: str, default: Any = None) -> Any:
    """Load a single preference value.

    Args:
        key: Preference key to load.
        default: Default value if key not found.

    Returns:
        The preference value, or default if not found.
    """
    prefs = _load_all_preferences()
    value = prefs.get(key, default)
    if value != default:
        logger.debug(f"Loaded preference: {key}={value}")
    return value
