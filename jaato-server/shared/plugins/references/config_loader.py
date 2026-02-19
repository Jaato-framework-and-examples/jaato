"""Configuration loading and validation for the references plugin.

This module handles loading references.json files and validating their structure.
It also supports auto-discovery of individual reference files from a directory.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.path_utils import normalize_path
from .models import ReferenceSource, SourceType, InjectionMode

logger = logging.getLogger(__name__)


def discover_references(
    references_dir: str,
    base_path: Optional[str] = None,
    project_root: Optional[str] = None
) -> List[ReferenceSource]:
    """Discover reference sources from individual JSON files in a directory.

    Each .json file in the directory should contain a single reference source
    definition with fields: id, name, description, type, path/url/content, mode, tags.

    Args:
        references_dir: Directory path to scan (relative or absolute).
        base_path: Base path for resolving relative references_dir.
                   Returns empty list when None (cannot resolve without workspace).
        project_root: Root directory for making resolved paths relative.
                      This should be where the agent operates from.
                      Defaults to parent of references_dir if it's under .jaato/.

    Returns:
        List of ReferenceSource instances discovered from the directory.
    """
    if base_path is None:
        return []  # No base path — cannot discover references

    # Resolve the references directory path
    refs_path = Path(references_dir)
    if not refs_path.is_absolute():
        refs_path = Path(base_path) / refs_path

    # Determine project root for relative path computation
    if project_root is None:
        # If refs_path is under .jaato/, project root is parent of .jaato
        resolved_refs = refs_path.resolve()
        if '.jaato' in resolved_refs.parts:
            jaato_idx = resolved_refs.parts.index('.jaato')
            project_root = str(Path(*resolved_refs.parts[:jaato_idx]))
        else:
            project_root = str(resolved_refs.parent)

    if not refs_path.exists():
        logger.debug("References directory does not exist: %s", refs_path)
        return []

    if not refs_path.is_dir():
        logger.warning("References path is not a directory: %s", refs_path)
        return []

    sources: List[ReferenceSource] = []

    # Scan for reference files
    for file_path in sorted(refs_path.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix != '.json':
            continue

        try:
            content = file_path.read_text(encoding='utf-8')
            data = json.loads(content)

            if not isinstance(data, dict):
                logger.warning("Reference file must contain a dict: %s", file_path)
                continue

            # Validate required fields
            if not data.get("id") or not data.get("name"):
                logger.warning(
                    "Reference file missing required fields (id, name): %s",
                    file_path
                )
                continue

            source = ReferenceSource.from_dict(data)

            # Resolve relative paths for LOCAL sources
            # Make paths relative to project root (where agent operates)
            if source.type == SourceType.LOCAL and source.path:
                source_path = Path(source.path)
                if source_path.is_absolute():
                    absolute_path = source_path.resolve()
                else:
                    # Check if path is project-relative (starts with .jaato/)
                    # These should be resolved against project_root, not file_path.parent
                    # Use original string to avoid Windows path normalization issues
                    # (Path() on Windows converts ./.jaato/foo to .jaato\foo, breaking prefix check)
                    original_path = source.path
                    if original_path.startswith('.jaato/') or original_path.startswith('./.jaato/'):
                        # Strip leading ./ if present for clean join
                        clean_path = original_path[2:] if original_path.startswith('./') else original_path
                        absolute_path = (Path(project_root) / clean_path).resolve()
                    else:
                        # Regular relative path - resolve against reference file's directory
                        absolute_path = (file_path.parent / source_path).resolve()
                # Convert to project-root-relative path
                # Normalize separators for MSYS2 compatibility (forward slashes)
                try:
                    source.resolved_path = normalize_path(os.path.relpath(absolute_path, project_root))
                except ValueError:
                    # On Windows, relpath fails for paths on different drives
                    source.resolved_path = normalize_path(str(absolute_path))

            sources.append(source)
            logger.debug("Discovered reference '%s' from %s", source.id, file_path)

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in reference file %s: %s", file_path, e)
        except Exception as e:
            logger.warning("Error reading reference file %s: %s", file_path, e)

    if sources:
        logger.info(
            "Discovered %d reference(s) from %s: %s",
            len(sources),
            refs_path,
            ", ".join(s.id for s in sources)
        )

    return sources


@dataclass
class ReferencesConfig:
    """Structured representation of a references configuration file.

    Attributes:
        version: Config format version.
        sources: List of reference sources.
        channel_type: Type of channel for user interaction.
        channel_timeout: Timeout for channel operations.
        channel_endpoint: Webhook endpoint URL (for webhook channel).
        channel_base_path: Base path (for file channel).
        auto_discover_references: Whether to auto-discover references from references_dir.
        references_dir: Directory to scan for reference files (default: .jaato/references).
        config_base_path: Directory where config file was loaded from (for resolving relative paths).
    """

    version: str = "1.0"
    sources: List[ReferenceSource] = field(default_factory=list)

    # Channel configuration
    channel_type: str = "console"
    channel_timeout: int = 60
    channel_endpoint: Optional[str] = None  # For webhook
    channel_base_path: Optional[str] = None  # For file

    # Auto-discovery configuration
    auto_discover_references: bool = True
    references_dir: str = ".jaato/references"

    # Path resolution
    config_base_path: Optional[str] = None  # Directory of config file


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


def resolve_source_paths(
    sources: List[ReferenceSource],
    base_path: str,
    relative_to: Optional[str] = None
) -> None:
    """Resolve relative paths in LOCAL sources against a base directory.

    For LOCAL type sources with relative paths, computes the resolved path
    and stores it in resolved_path. This helps the model find files when
    the config was loaded from a different directory.

    Args:
        sources: List of ReferenceSource instances to process (modified in-place).
        base_path: Base directory to resolve relative paths against.
        relative_to: If provided, make resolved paths relative to this directory
                     (typically CWD). If None, stores absolute paths.
    """
    base = Path(base_path)
    cwd = Path(relative_to) if relative_to else None

    for source in sources:
        if source.type != SourceType.LOCAL:
            continue
        if not source.path:
            continue

        source_path = Path(source.path)

        # Resolve to absolute first (handles .. and other path components)
        if source_path.is_absolute():
            absolute_path = source_path.resolve()
        else:
            # Check if path is project-relative (starts with .jaato/)
            # These should be resolved against cwd (project root), not base_path
            # Use original string to avoid Windows path normalization issues
            # (Path() on Windows converts ./.jaato/foo to .jaato\foo, breaking prefix check)
            original_path = source.path
            if cwd and (original_path.startswith('.jaato/') or original_path.startswith('./.jaato/')):
                # Strip leading ./ if present for clean join
                clean_path = original_path[2:] if original_path.startswith('./') else original_path
                absolute_path = (cwd / clean_path).resolve()
            else:
                absolute_path = (base / source_path).resolve()

        # Make relative to CWD if requested
        # Normalize separators for MSYS2 compatibility (forward slashes)
        if cwd:
            try:
                resolved = normalize_path(os.path.relpath(absolute_path, cwd))
            except ValueError:
                # On Windows, relpath fails for paths on different drives
                resolved = normalize_path(str(absolute_path))
        else:
            resolved = normalize_path(str(absolute_path))

        source.resolved_path = resolved
        logger.debug(
            "Resolved path for '%s': %s -> %s",
            source.id, source.path, source.resolved_path
        )


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


def validate_reference_file(data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Validate a single standalone reference JSON file.

    Unlike validate_config() which validates a full references.json with a
    sources[] array, this validates a single reference definition (the format
    produced by gen-references and stored in .jaato/references/*.json).

    Args:
        data: Raw reference dict loaded from a single-reference JSON file.

    Returns:
        Tuple of (is_valid, errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(data, dict):
        return False, ["File must contain a JSON object"], []

    # Required fields
    if not data.get("id"):
        errors.append("'id' is required")
    if not data.get("name"):
        errors.append("'name' is required")

    # Validate type
    source_type = data.get("type", "local")
    valid_types = ("local", "url", "mcp", "inline")
    if source_type not in valid_types:
        errors.append(f"Invalid type '{source_type}'. Must be one of: {', '.join(valid_types)}")

    # Validate mode
    mode = data.get("mode", "selectable")
    valid_modes = ("auto", "selectable")
    if mode not in valid_modes:
        errors.append(f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}")

    # Type-specific validation
    if source_type == "local":
        if not data.get("path"):
            errors.append("'path' is required for local type")
        else:
            # Warn if path doesn't exist on disk
            path_val = data["path"]
            if os.path.isabs(path_val) and not os.path.exists(path_val):
                warnings.append(f"path does not exist on disk: {path_val}")
    elif source_type == "url":
        if not data.get("url"):
            errors.append("'url' is required for url type")
    elif source_type == "mcp":
        if not data.get("server"):
            errors.append("'server' is required for mcp type")
        if not data.get("tool"):
            errors.append("'tool' is required for mcp type")
    elif source_type == "inline":
        if not data.get("content"):
            errors.append("'content' is required for inline type")

    # Validate tags
    tags = data.get("tags")
    if tags is not None:
        if not isinstance(tags, list):
            errors.append("'tags' must be an array")
        elif not all(isinstance(t, str) for t in tags):
            errors.append("'tags' must contain only strings")

    return len(errors) == 0, errors, warnings


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

    # Validate channel configuration
    channel = config.get("channel", {})
    if channel:
        channel_type = channel.get("type", "console")
        if channel_type not in ("console", "webhook", "file"):
            errors.append(f"Invalid channel type: {channel_type}. Must be 'console', 'webhook', or 'file'")

        if channel_type == "webhook":
            endpoint = channel.get("endpoint")
            if not endpoint or not isinstance(endpoint, str):
                errors.append("Webhook channel requires 'endpoint' URL")

        if channel_type == "file":
            base_path = channel.get("base_path")
            if not base_path or not isinstance(base_path, str):
                errors.append("File channel requires 'base_path'")

        timeout = channel.get("timeout")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            errors.append("Channel timeout must be a positive number")

    return len(errors) == 0, errors


def load_config(
    path: Optional[str] = None,
    env_var: str = "REFERENCES_CONFIG_PATH",
    auto_discover: bool = True,
    references_dir: str = ".jaato/references",
    workspace_path: Optional[str] = None,
) -> ReferencesConfig:
    """Load and validate a references configuration file.

    Args:
        path: Direct path to config file. If None, uses env_var or defaults.
        env_var: Environment variable name for config path
        auto_discover: Whether to auto-discover references from references_dir.
        references_dir: Directory to scan for individual reference files.
        workspace_path: Workspace root for resolving workspace-relative paths.
            When None, workspace-relative default paths are skipped.

    Returns:
        ReferencesConfig instance with merged sources from config file
        and auto-discovered references.

    Raises:
        FileNotFoundError: If config file doesn't exist (when path is explicit)
        ConfigValidationError: If config validation fails
        json.JSONDecodeError: If config file is not valid JSON
    """
    # Resolve path
    if path is None:
        path = os.environ.get(env_var)

    if path is None:
        # Try default locations — workspace-relative paths only if workspace is set
        default_paths = []
        if workspace_path:
            ws = Path(workspace_path)
            default_paths.append(ws / "references.json")
            default_paths.append(ws / ".references.json")
        default_paths.append(Path.home() / ".config" / "jaato" / "references.json")
        for default_path in default_paths:
            if default_path.exists():
                path = str(default_path)
                break

    # Start with defaults
    config = ReferencesConfig(
        auto_discover_references=auto_discover,
        references_dir=references_dir
    )

    # Load from config file if found
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"References config file not found: {path}")

        # Capture the config file's directory for resolving relative paths
        config_base_path = str(config_path.parent.resolve())
        # Project root is typically the parent of .jaato/ or where config resides
        project_root = str(config_path.parent.parent.resolve()) if config_path.parent.name == '.jaato' else config_base_path

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

        # Resolve relative paths for LOCAL sources against config file directory
        # Make paths relative to project root (not CWD which may differ from where agent operates)
        resolve_source_paths(sources, config_base_path, relative_to=project_root)

        # Parse channel config
        channel = raw_config.get("channel", {})

        # Parse auto-discover settings from config (can override function args)
        config = ReferencesConfig(
            version=str(raw_config.get("version", "1.0")),
            sources=sources,
            channel_type=channel.get("type", "console"),
            channel_timeout=channel.get("timeout", 60),
            channel_endpoint=channel.get("endpoint"),
            channel_base_path=channel.get("base_path"),
            auto_discover_references=raw_config.get("auto_discover_references", auto_discover),
            references_dir=raw_config.get("references_dir", references_dir),
            config_base_path=config_base_path,
        )

    # Auto-discover references if enabled
    if config.auto_discover_references:
        discovered = discover_references(config.references_dir, base_path=workspace_path)
        if discovered:
            # Build set of existing IDs to avoid duplicates
            existing_ids = {s.id for s in config.sources}
            # Merge discovered sources, explicit sources take precedence
            for source in discovered:
                if source.id not in existing_ids:
                    config.sources.append(source)
                else:
                    logger.debug(
                        "Skipping discovered reference '%s' - explicit source exists",
                        source.id
                    )

    return config


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
        "channel": {
            "type": "console",
            "timeout": 60
        }
    }

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2)
