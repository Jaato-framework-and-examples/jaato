"""Configuration loading and validation for the TODO plugin.

This module handles loading todo.json files and validating their structure.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TodoConfig:
    """Structured representation of a TODO plugin configuration file."""

    version: str = "1.0"

    # Reporter configuration
    reporter_type: str = "console"  # console, webhook, file
    reporter_endpoint: Optional[str] = None
    reporter_timeout: int = 10  # seconds
    reporter_headers: Dict[str, str] = field(default_factory=dict)
    reporter_base_path: Optional[str] = None

    # Storage configuration
    storage_type: str = "memory"  # memory, file, hybrid
    storage_path: Optional[str] = None
    storage_use_directory: bool = False

    # Display configuration (for console reporter)
    display_timestamps: bool = True
    display_progress_bar: bool = True
    display_colors: bool = True

    def to_reporter_config(self) -> Dict[str, Any]:
        """Convert to dict format expected by reporter initialization."""
        config: Dict[str, Any] = {
            "show_timestamps": self.display_timestamps,
            "progress_bar": self.display_progress_bar,
            "colors": self.display_colors,
        }

        if self.reporter_type == "webhook":
            config["endpoint"] = self.reporter_endpoint
            config["timeout"] = self.reporter_timeout
            config["headers"] = self.reporter_headers

        elif self.reporter_type == "file":
            config["base_path"] = self.reporter_base_path

        return config

    def to_storage_config(self) -> Dict[str, Any]:
        """Convert to dict format expected by storage initialization."""
        return {
            "type": self.storage_type,
            "path": self.storage_path,
            "use_directory": self.storage_use_directory,
        }


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a TODO configuration dict.

    Args:
        config: Raw configuration dict loaded from JSON

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Check version
    version = config.get("version")
    if version and version not in ("1.0", "1"):
        errors.append(f"Unsupported config version: {version}")

    # Validate reporter configuration
    reporter = config.get("reporter", {})
    if reporter:
        reporter_type = reporter.get("type", "console")
        if reporter_type not in ("console", "webhook", "file"):
            errors.append(f"Invalid reporter type: {reporter_type}")

        if reporter_type == "webhook":
            endpoint = reporter.get("endpoint")
            if not endpoint or not isinstance(endpoint, str):
                errors.append("Webhook reporter requires 'endpoint' URL")

        if reporter_type == "file":
            base_path = reporter.get("base_path")
            if not base_path or not isinstance(base_path, str):
                errors.append("File reporter requires 'base_path'")

        timeout = reporter.get("timeout")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            errors.append("Reporter timeout must be a positive number")

        headers = reporter.get("headers", {})
        if not isinstance(headers, dict):
            errors.append("Reporter 'headers' must be an object")

    # Validate storage configuration
    storage = config.get("storage", {})
    if storage:
        storage_type = storage.get("type", "memory")
        if storage_type not in ("memory", "file", "hybrid"):
            errors.append(f"Invalid storage type: {storage_type}")

        if storage_type in ("file", "hybrid"):
            storage_path = storage.get("path")
            if storage_type == "file" and not storage_path:
                errors.append("File storage requires 'path'")

        use_directory = storage.get("use_directory")
        if use_directory is not None and not isinstance(use_directory, bool):
            errors.append("Storage 'use_directory' must be a boolean")

    # Validate display configuration
    display = config.get("display", {})
    if display:
        for key in ("show_timestamps", "progress_bar", "colors"):
            value = display.get(key)
            if value is not None and not isinstance(value, bool):
                errors.append(f"Display '{key}' must be a boolean")

    return len(errors) == 0, errors


def load_config(
    path: Optional[str] = None,
    env_var: str = "TODO_CONFIG_PATH"
) -> TodoConfig:
    """Load and validate a TODO configuration file.

    Args:
        path: Direct path to config file. If None, uses env_var or defaults.
        env_var: Environment variable name for config path

    Returns:
        TodoConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigValidationError: If config validation fails
        json.JSONDecodeError: If config file is not valid JSON
    """
    # Resolve path
    if path is None:
        path = os.environ.get(env_var)

    if path is None:
        # Try default locations
        default_paths = [
            Path.cwd() / "todo.json",
            Path.cwd() / ".todo.json",
            Path.home() / ".config" / "jaato" / "todo.json",
        ]
        for default_path in default_paths:
            if default_path.exists():
                path = str(default_path)
                break

    if path is None:
        # Return default config if no file found
        return TodoConfig()

    # Load and parse
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"TODO config file not found: {path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)

    # Validate
    is_valid, errors = validate_config(raw_config)
    if errors:
        raise ConfigValidationError(errors)

    # Parse into structured config
    reporter = raw_config.get("reporter", {})
    storage = raw_config.get("storage", {})
    display = raw_config.get("display", {})

    return TodoConfig(
        version=str(raw_config.get("version", "1.0")),
        reporter_type=reporter.get("type", "console"),
        reporter_endpoint=reporter.get("endpoint"),
        reporter_timeout=reporter.get("timeout", 10),
        reporter_headers=reporter.get("headers", {}),
        reporter_base_path=reporter.get("base_path"),
        storage_type=storage.get("type", "memory"),
        storage_path=storage.get("path"),
        storage_use_directory=storage.get("use_directory", False),
        display_timestamps=display.get("show_timestamps", True),
        display_progress_bar=display.get("progress_bar", True),
        display_colors=display.get("colors", True),
    )


def create_default_config(path: str) -> None:
    """Create a default todo.json file at the given path.

    Args:
        path: Path where to create the config file
    """
    default_config = {
        "version": "1.0",
        "reporter": {
            "type": "console",
            "timeout": 10
        },
        "storage": {
            "type": "memory"
        },
        "display": {
            "show_timestamps": True,
            "progress_bar": True,
            "colors": True
        }
    }

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2)
