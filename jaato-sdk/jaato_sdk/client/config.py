"""Client configuration loading with layered precedence.

Provides configuration loading for the rich client with support for:
1. JSON configuration files (project-level and user-level)
2. Environment variable overrides
3. Built-in defaults

Configuration precedence (highest wins):
1. Environment variables (JAATO_IPC_*)
2. Project config (.jaato/client.json)
3. User config (~/.jaato/client.json)
4. Built-in defaults

Usage:
    from jaato_sdk.client.config import load_client_config, get_recovery_config

    # Load full client config
    config = load_client_config(workspace_path=Path.cwd())

    # Get just recovery config
    recovery = get_recovery_config(workspace_path=Path.cwd())

Environment Variables:
    JAATO_IPC_AUTO_RECONNECT: Enable automatic reconnection (default: true)
    JAATO_IPC_RETRY_MAX_ATTEMPTS: Maximum reconnection attempts (default: 10)
    JAATO_IPC_RETRY_BASE_DELAY: Initial backoff delay seconds (default: 1.0)
    JAATO_IPC_RETRY_MAX_DELAY: Maximum backoff delay seconds (default: 60.0)
    JAATO_IPC_RETRY_JITTER: Random jitter factor (default: 0.3)
    JAATO_IPC_CONNECTION_TIMEOUT: Per-attempt timeout seconds (default: 5.0)
    JAATO_IPC_REATTACH_SESSION: Auto-reattach to previous session (default: true)
"""

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, get_type_hints

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_env_value(value: str, target_type: Type) -> Any:
    """Parse environment variable value to target type."""
    # Handle Optional types
    origin = getattr(target_type, '__origin__', None)
    if origin is type(None):
        return None

    # Extract inner type from Optional
    args = getattr(target_type, '__args__', ())
    if args and type(None) in args:
        # It's Optional[X], get X
        inner_types = [a for a in args if a is not type(None)]
        if inner_types:
            target_type = inner_types[0]

    if target_type == bool:
        return _parse_bool(value)
    elif target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == str:
        return value
    return value


@dataclass
class RecoveryConfig:
    """IPC connection recovery settings.

    Controls how the client handles server disconnections and reconnection
    attempts. Uses exponential backoff with jitter for retry delays.

    Attributes:
        enabled: Whether automatic reconnection is enabled.
        max_attempts: Maximum number of reconnection attempts before giving up.
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds (caps exponential growth).
        jitter_factor: Random jitter range (0.3 = ±30% variation).
        connection_timeout: Timeout for each connection attempt in seconds.
        reattach_session: Whether to automatically reattach to the previous
            session after reconnecting.
    """
    enabled: bool = True
    max_attempts: int = 10
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.3
    connection_timeout: float = 5.0
    reattach_session: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0.1:
            raise ValueError("base_delay must be at least 0.1 seconds")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if not (0.0 <= self.jitter_factor <= 1.0):
            raise ValueError("jitter_factor must be between 0.0 and 1.0")
        if self.connection_timeout < 1.0:
            raise ValueError("connection_timeout must be at least 1.0 seconds")


@dataclass
class ClientConfig:
    """Root client configuration.

    Contains all configuration sections for the rich client.
    Currently includes recovery settings, but can be extended with
    other configuration sections in the future.

    Attributes:
        recovery: IPC connection recovery settings.
    """
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)


# Environment variable mapping for recovery config
# Maps "section.field" paths to environment variable names
ENV_VAR_MAPPING: Dict[str, str] = {
    "recovery.enabled": "JAATO_IPC_AUTO_RECONNECT",
    "recovery.max_attempts": "JAATO_IPC_RETRY_MAX_ATTEMPTS",
    "recovery.base_delay": "JAATO_IPC_RETRY_BASE_DELAY",
    "recovery.max_delay": "JAATO_IPC_RETRY_MAX_DELAY",
    "recovery.jitter_factor": "JAATO_IPC_RETRY_JITTER",
    "recovery.connection_timeout": "JAATO_IPC_CONNECTION_TIMEOUT",
    "recovery.reattach_session": "JAATO_IPC_REATTACH_SESSION",
}


def _find_config_files(workspace_path: Optional[Path] = None) -> List[Path]:
    """Find configuration files in order of precedence (lowest first).

    Searches for client.json in:
    1. ~/.jaato/client.json (user-level, lowest precedence)
    2. <workspace>/.jaato/client.json (project-level, higher precedence)

    Args:
        workspace_path: Path to project workspace. If None, only user config
            is searched.

    Returns:
        List of existing config file paths, ordered from lowest to highest
        precedence.
    """
    files = []

    # User-level config (lowest precedence)
    user_config = Path.home() / ".jaato" / "client.json"
    if user_config.exists():
        files.append(user_config)

    # Project-level config (higher precedence)
    if workspace_path:
        project_config = workspace_path / ".jaato" / "client.json"
        if project_config.exists():
            files.append(project_config)

    return files


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge overlay into base, returning new dict.

    Recursively merges nested dictionaries. Overlay values override base
    values. Lists and other types are replaced, not merged.

    Args:
        base: Base dictionary to merge into.
        overlay: Dictionary with values to overlay.

    Returns:
        New dictionary with merged values.
    """
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_field_type(dataclass_type: Type, field_name: str) -> Type:
    """Get the type of a field in a dataclass.

    Args:
        dataclass_type: The dataclass type to inspect.
        field_name: Name of the field.

    Returns:
        The field's type, or str as fallback.
    """
    try:
        hints = get_type_hints(dataclass_type)
        return hints.get(field_name, str)
    except Exception:
        return str


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config dict.

    Checks each mapped environment variable and applies its value
    to the corresponding config path if set.

    Args:
        config_dict: Configuration dictionary to modify.

    Returns:
        New dictionary with environment overrides applied.
    """
    result = config_dict.copy()

    for path, env_var in ENV_VAR_MAPPING.items():
        env_value = os.environ.get(env_var)
        if env_value is None:
            continue

        # Navigate/create nested path
        parts = path.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]

        # Determine target type from RecoveryConfig
        field_name = parts[-1]
        target_type = _get_field_type(RecoveryConfig, field_name)

        try:
            current[parts[-1]] = _parse_env_value(env_value, target_type)
            logger.debug(f"Applied env override: {env_var}={env_value}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid value for {env_var}: {env_value} ({e})")

    return result


def _dict_to_recovery_config(data: Dict[str, Any]) -> RecoveryConfig:
    """Convert dict to RecoveryConfig dataclass.

    Filters out unknown keys and handles type conversion.

    Args:
        data: Dictionary with recovery config values.

    Returns:
        RecoveryConfig instance.
    """
    valid_fields = {f.name for f in fields(RecoveryConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    # Log any unknown keys
    unknown = set(data.keys()) - valid_fields
    if unknown:
        logger.warning(f"Unknown recovery config keys (ignored): {unknown}")

    try:
        return RecoveryConfig(**filtered)
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid recovery config values, using defaults: {e}")
        return RecoveryConfig()


def _dict_to_config(data: Dict[str, Any]) -> ClientConfig:
    """Convert dict to ClientConfig dataclass.

    Args:
        data: Dictionary with config values.

    Returns:
        ClientConfig instance.
    """
    recovery_data = data.get("recovery", {})
    if not isinstance(recovery_data, dict):
        logger.warning("Invalid 'recovery' config (expected dict), using defaults")
        recovery_data = {}

    recovery_config = _dict_to_recovery_config(recovery_data)
    return ClientConfig(recovery=recovery_config)


def load_client_config(workspace_path: Optional[Path] = None) -> ClientConfig:
    """Load client configuration with layered precedence.

    Configuration is loaded and merged from multiple sources:
    1. Built-in defaults (lowest precedence)
    2. User config (~/.jaato/client.json)
    3. Project config (<workspace>/.jaato/client.json)
    4. Environment variables (highest precedence)

    Args:
        workspace_path: Path to project workspace for project-level config.
            If None, only user config and environment variables are used.

    Returns:
        Merged ClientConfig instance.

    Example:
        # Load with workspace
        config = load_client_config(Path.cwd())

        # Access recovery settings
        if config.recovery.enabled:
            print(f"Max retry attempts: {config.recovery.max_attempts}")
    """
    # Start with empty dict (defaults come from dataclass)
    merged: Dict[str, Any] = {}

    # Load and merge config files (lowest precedence first)
    for config_file in _find_config_files(workspace_path):
        try:
            with open(config_file) as f:
                file_config = json.load(f)

            if not isinstance(file_config, dict):
                logger.warning(f"Invalid config format in {config_file} (expected object)")
                continue

            merged = _deep_merge(merged, file_config)
            logger.debug(f"Loaded config from {config_file}")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {config_file}: {e}")
        except PermissionError:
            logger.warning(f"Permission denied reading {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load {config_file}: {e}")

    # Apply environment variable overrides (highest precedence)
    merged = _apply_env_overrides(merged)

    # Convert to typed config
    return _dict_to_config(merged)


def get_recovery_config(workspace_path: Optional[Path] = None) -> RecoveryConfig:
    """Convenience function to get just the recovery config.

    Args:
        workspace_path: Path to project workspace for project-level config.

    Returns:
        RecoveryConfig instance with all sources merged.
    """
    return load_client_config(workspace_path).recovery


def generate_example_config() -> str:
    """Generate an example client.json configuration file.

    Returns:
        JSON string with example configuration and comments.
    """
    example = {
        "_comment": "Jaato rich client configuration",
        "_docs": "https://github.com/your-org/jaato/blob/main/docs/configuration.md",
        "recovery": {
            "_comment": "IPC connection recovery settings",
            "enabled": True,
            "max_attempts": 10,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "jitter_factor": 0.3,
            "connection_timeout": 5.0,
            "reattach_session": True,
        },
    }
    return json.dumps(example, indent=2)


def get_config_paths(workspace_path: Optional[Path] = None) -> Dict[str, Path]:
    """Get the paths where config files are searched.

    Useful for displaying configuration locations to users.

    Args:
        workspace_path: Path to project workspace.

    Returns:
        Dict with 'user' and optionally 'project' paths.
    """
    paths = {
        "user": Path.home() / ".jaato" / "client.json",
    }
    if workspace_path:
        paths["project"] = workspace_path / ".jaato" / "client.json"
    return paths


__all__ = [
    "ClientConfig",
    "RecoveryConfig",
    "generate_example_config",
    "get_config_paths",
    "get_recovery_config",
    "load_client_config",
]
