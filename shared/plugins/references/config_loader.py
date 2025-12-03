"""Configuration loading and validation for the references plugin.

This module handles loading references.json files and validating their structure.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import ReferenceSource, SourceType, InjectionMode


@dataclass
class ReferencesConfig:
    """Structured representation of a references configuration file."""

    version: str = "1.0"
    sources: List[ReferenceSource] = field(default_factory=list)

    # Actor configuration
    actor_type: str = "console"
    actor_timeout: int = 60
    actor_endpoint: Optional[str] = None  # For webhook
    actor_base_path: Optional[str] = None  # For file


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


def validate_source(source: Dict[str, Any], index: int, errors: List[str]) -> None:
    """Validate a single source definition."""
    prefix = f"sources[{index}]"

    # Required fields
    if not source.get("id"):
        errors.append(f"{prefix}: 'id' is required")
    if not source.get("name"):
        errors.append(f"{prefix}: 'name' is required")

    # Validate type
    source_type = source.get("type", "local")
    if source_type not in ("local", "url", "mcp", "inline"):
        errors.append(f"{prefix}: Invalid type '{source_type}'. Must be one of: local, url, mcp, inline")

    # Validate mode
    mode = source.get("mode", "selectable")
    if mode not in ("auto", "selectable"):
        errors.append(f"{prefix}: Invalid mode '{mode}'. Must be 'auto' or 'selectable'")

    # Type-specific validation
    if source_type == "local" and not source.get("path"):
        errors.append(f"{prefix}: 'path' is required for local type")
    elif source_type == "url" and not source.get("url"):
        errors.append(f"{prefix}: 'url' is required for url type")
    elif source_type == "mcp":
        if not source.get("server"):
            errors.append(f"{prefix}: 'server' is required for mcp type")
        if not source.get("tool"):
            errors.append(f"{prefix}: 'tool' is required for mcp type")
    elif source_type == "inline" and not source.get("content"):
        errors.append(f"{prefix}: 'content' is required for inline type")

    # Validate tags is a list of strings
    tags = source.get("tags", [])
    if not isinstance(tags, list):
        errors.append(f"{prefix}: 'tags' must be an array")
    elif not all(isinstance(t, str) for t in tags):
        errors.append(f"{prefix}: 'tags' must contain only strings")


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a references configuration dict.

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

    # Validate sources
    sources = config.get("sources", [])
    if not isinstance(sources, list):
        errors.append("'sources' must be an array")
    else:
        # Check for duplicate IDs
        seen_ids = set()
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                errors.append(f"sources[{i}]: must be an object")
                continue

            validate_source(source, i, errors)

            source_id = source.get("id")
            if source_id:
                if source_id in seen_ids:
                    errors.append(f"sources[{i}]: Duplicate id '{source_id}'")
                seen_ids.add(source_id)

    # Validate actor configuration
    actor = config.get("actor", {})
    if actor:
        actor_type = actor.get("type", "console")
        if actor_type not in ("console", "webhook", "file"):
            errors.append(f"Invalid actor type: {actor_type}. Must be 'console', 'webhook', or 'file'")

        if actor_type == "webhook":
            endpoint = actor.get("endpoint")
            if not endpoint or not isinstance(endpoint, str):
                errors.append("Webhook actor requires 'endpoint' URL")

        if actor_type == "file":
            base_path = actor.get("base_path")
            if not base_path or not isinstance(base_path, str):
                errors.append("File actor requires 'base_path'")

        timeout = actor.get("timeout")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            errors.append("Actor timeout must be a positive number")

    return len(errors) == 0, errors


def load_config(
    path: Optional[str] = None,
    env_var: str = "REFERENCES_CONFIG_PATH"
) -> ReferencesConfig:
    """Load and validate a references configuration file.

    Args:
        path: Direct path to config file. If None, uses env_var or defaults.
        env_var: Environment variable name for config path

    Returns:
        ReferencesConfig instance

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
            Path.cwd() / "references.json",
            Path.cwd() / ".references.json",
            Path.home() / ".config" / "jaato" / "references.json",
        ]
        for default_path in default_paths:
            if default_path.exists():
                path = str(default_path)
                break

    if path is None:
        # Return default config if no file found
        return ReferencesConfig()

    # Load and parse
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"References config file not found: {path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)

    # Validate
    is_valid, errors = validate_config(raw_config)
    if not is_valid:
        raise ConfigValidationError(errors)

    # Parse sources
    sources = [
        ReferenceSource.from_dict(s)
        for s in raw_config.get("sources", [])
    ]

    # Parse actor config
    actor = raw_config.get("actor", {})

    return ReferencesConfig(
        version=str(raw_config.get("version", "1.0")),
        sources=sources,
        actor_type=actor.get("type", "console"),
        actor_timeout=actor.get("timeout", 60),
        actor_endpoint=actor.get("endpoint"),
        actor_base_path=actor.get("base_path"),
    )


def create_default_config(path: str) -> None:
    """Create a default references.json file at the given path.

    Args:
        path: Path where to create the config file
    """
    default_config = {
        "version": "1.0",
        "sources": [
            {
                "id": "readme",
                "name": "Project README",
                "description": "Main project documentation",
                "type": "local",
                "path": "./README.md",
                "mode": "auto",
                "tags": ["overview", "getting-started"]
            },
            {
                "id": "api-docs",
                "name": "API Documentation",
                "description": "REST API reference",
                "type": "local",
                "path": "./docs/api.md",
                "mode": "selectable",
                "tags": ["api", "reference"]
            }
        ],
        "actor": {
            "type": "console",
            "timeout": 60
        }
    }

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2)
