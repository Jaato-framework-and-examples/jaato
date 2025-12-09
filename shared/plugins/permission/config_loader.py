"""Configuration loading and validation for permission policies.

This module handles loading permissions.json files and validating their structure.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PermissionConfig:
    """Structured representation of a permissions configuration file."""

    version: str = "1.0"
    default_policy: str = "deny"

    # Blacklist configuration
    blacklist_tools: List[str] = field(default_factory=list)
    blacklist_patterns: List[str] = field(default_factory=list)
    blacklist_arguments: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # Whitelist configuration
    whitelist_tools: List[str] = field(default_factory=list)
    whitelist_patterns: List[str] = field(default_factory=list)
    whitelist_arguments: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # Channel configuration
    channel_type: str = "console"  # console, webhook, queue, file
    channel_endpoint: Optional[str] = None
    channel_timeout: int = 30  # seconds

    def to_policy_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by PermissionPolicy.from_config()."""
        return {
            "defaultPolicy": self.default_policy,
            "blacklist": {
                "tools": self.blacklist_tools,
                "patterns": self.blacklist_patterns,
                "arguments": self.blacklist_arguments,
            },
            "whitelist": {
                "tools": self.whitelist_tools,
                "patterns": self.whitelist_patterns,
                "arguments": self.whitelist_arguments,
            },
        }


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a permissions configuration dict.

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

    # Validate defaultPolicy
    default_policy = config.get("defaultPolicy", "deny")
    if default_policy not in ("allow", "deny", "ask"):
        errors.append(f"Invalid defaultPolicy: {default_policy}. Must be 'allow', 'deny', or 'ask'")

    # Validate blacklist structure
    blacklist = config.get("blacklist", {})
    if not isinstance(blacklist, dict):
        errors.append("'blacklist' must be an object")
    else:
        _validate_list_rules(blacklist, "blacklist", errors)

    # Validate whitelist structure
    whitelist = config.get("whitelist", {})
    if not isinstance(whitelist, dict):
        errors.append("'whitelist' must be an object")
    else:
        _validate_list_rules(whitelist, "whitelist", errors)

    # Validate channel configuration
    channel = config.get("channel", {})
    if channel:
        channel_type = channel.get("type", "console")
        if channel_type not in ("console", "webhook", "queue", "file"):
            errors.append(f"Invalid channel type: {channel_type}")

        if channel_type == "webhook":
            endpoint = channel.get("endpoint")
            if not endpoint or not isinstance(endpoint, str):
                errors.append("Webhook channel requires 'endpoint' URL")

        timeout = channel.get("timeout")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            errors.append("Channel timeout must be a positive number")

    # Check for blacklist/whitelist conflicts (warning, not error)
    _check_conflicts(config, errors)

    return len(errors) == 0, errors


def _validate_list_rules(rules: Dict[str, Any], list_type: str, errors: List[str]) -> None:
    """Validate blacklist or whitelist rules structure."""

    # Validate tools list
    tools = rules.get("tools", [])
    if not isinstance(tools, list):
        errors.append(f"'{list_type}.tools' must be an array")
    elif not all(isinstance(t, str) for t in tools):
        errors.append(f"'{list_type}.tools' must contain only strings")

    # Validate patterns list
    patterns = rules.get("patterns", [])
    if not isinstance(patterns, list):
        errors.append(f"'{list_type}.patterns' must be an array")
    elif not all(isinstance(p, str) for p in patterns):
        errors.append(f"'{list_type}.patterns' must contain only strings")

    # Validate arguments structure
    arguments = rules.get("arguments", {})
    if not isinstance(arguments, dict):
        errors.append(f"'{list_type}.arguments' must be an object")
    else:
        for tool_name, arg_rules in arguments.items():
            if not isinstance(arg_rules, dict):
                errors.append(f"'{list_type}.arguments.{tool_name}' must be an object")
                continue
            for arg_name, values in arg_rules.items():
                if not isinstance(values, list):
                    errors.append(
                        f"'{list_type}.arguments.{tool_name}.{arg_name}' must be an array"
                    )
                elif not all(isinstance(v, str) for v in values):
                    errors.append(
                        f"'{list_type}.arguments.{tool_name}.{arg_name}' must contain only strings"
                    )


def _check_conflicts(config: Dict[str, Any], errors: List[str]) -> None:
    """Check for potential conflicts between blacklist and whitelist.

    This adds warnings (not errors) when the same item appears in both lists,
    since blacklist always wins but it may indicate a configuration mistake.
    """
    blacklist = config.get("blacklist", {})
    whitelist = config.get("whitelist", {})

    # Guard against non-dict types (already reported as errors in validation)
    if not isinstance(blacklist, dict) or not isinstance(whitelist, dict):
        return

    # Get tools lists, guarding against non-list types
    bl_tools_raw = blacklist.get("tools", [])
    wl_tools_raw = whitelist.get("tools", [])

    if not isinstance(bl_tools_raw, list) or not isinstance(wl_tools_raw, list):
        return

    # Check tool conflicts
    bl_tools = set(bl_tools_raw)
    wl_tools = set(wl_tools_raw)
    conflicts = bl_tools & wl_tools
    if conflicts:
        errors.append(
            f"Warning: Tools in both blacklist and whitelist (blacklist wins): {conflicts}"
        )


def load_config(
    path: Optional[str] = None,
    env_var: str = "PERMISSION_CONFIG_PATH"
) -> PermissionConfig:
    """Load and validate a permissions configuration file.

    Args:
        path: Direct path to config file. If None, uses env_var or defaults.
        env_var: Environment variable name for config path

    Returns:
        PermissionConfig instance

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
            Path.cwd() / "permissions.json",
            Path.cwd() / ".permissions.json",
            Path.home() / ".config" / "jaato" / "permissions.json",
        ]
        for default_path in default_paths:
            if default_path.exists():
                path = str(default_path)
                break

    if path is None:
        # Return default config if no file found
        return PermissionConfig()

    # Load and parse
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Permission config file not found: {path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)

    # Validate
    is_valid, errors = validate_config(raw_config)
    # Filter out warnings for validation check
    actual_errors = [e for e in errors if not e.startswith("Warning:")]
    if actual_errors:
        raise ConfigValidationError(actual_errors)

    # Parse into structured config
    blacklist = raw_config.get("blacklist", {})
    whitelist = raw_config.get("whitelist", {})
    channel = raw_config.get("channel", {})

    return PermissionConfig(
        version=str(raw_config.get("version", "1.0")),
        default_policy=raw_config.get("defaultPolicy", "deny"),
        blacklist_tools=blacklist.get("tools", []),
        blacklist_patterns=blacklist.get("patterns", []),
        blacklist_arguments=blacklist.get("arguments", {}),
        whitelist_tools=whitelist.get("tools", []),
        whitelist_patterns=whitelist.get("patterns", []),
        whitelist_arguments=whitelist.get("arguments", {}),
        channel_type=channel.get("type", "console"),
        channel_endpoint=channel.get("endpoint"),
        channel_timeout=channel.get("timeout", 30),
    )


def create_default_config(path: str) -> None:
    """Create a default permissions.json file at the given path.

    Args:
        path: Path where to create the config file
    """
    default_config = {
        "version": "1.0",
        "defaultPolicy": "ask",
        "blacklist": {
            "tools": [],
            "patterns": [
                "rm -rf *",
                "sudo *",
                "chmod 777 *",
            ],
            "arguments": {
                "cli_based_tool": {
                    "command": ["rm -rf", "sudo", "shutdown", "reboot"]
                }
            }
        },
        "whitelist": {
            "tools": [],
            "patterns": [
                "git *",
                "npm *",
                "python *",
                "pytest *",
            ],
            "arguments": {}
        },
        "channel": {
            "type": "console",
            "timeout": 30
        }
    }

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2)
