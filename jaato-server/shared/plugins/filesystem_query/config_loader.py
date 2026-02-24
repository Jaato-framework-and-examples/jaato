"""Configuration loading and validation for the filesystem_query plugin.

This module handles loading filesystem_query.json config files and merging
with hardcoded defaults. Supports three configuration sources (in priority order):
1. Runtime config passed to initialize()
2. Config file (.jaato/filesystem_query.json or env var path)
3. Hardcoded defaults

Additionally, when a workspace root is available, the plugin integrates with
``GitignoreParser`` to honour the project's ``.gitignore`` file.  Gitignore-based
exclusions are checked *after* the hardcoded/config-file patterns and are still
subject to force-include overrides.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from shared.utils.gitignore import GitignoreParser

logger = logging.getLogger(__name__)


# Hardcoded default exclusions - common directories to skip during searches
DEFAULT_EXCLUDE_PATTERNS: List[str] = [
    # Version control
    ".git",
    ".svn",
    ".hg",
    ".bzr",
    # JavaScript/Node
    "node_modules",
    "bower_components",
    ".npm",
    ".yarn",
    # Python
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
    ".eggs",
    # Build artifacts
    "dist",
    "build",
    "out",
    "target",
    "_build",
    # IDE/Editor
    ".idea",
    ".vscode",
    ".eclipse",
    "*.swp",
    "*.swo",
    # Coverage/docs
    "htmlcov",
    ".coverage",
    "coverage",
    # Minified files (for grep)
    "*.min.js",
    "*.min.css",
    # Lock files (usually not useful to search)
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    # OS files
    ".DS_Store",
    "Thumbs.db",
]

# Default limits
DEFAULT_MAX_RESULTS = 500
DEFAULT_MAX_FILE_SIZE_KB = 1024  # 1MB
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CONTEXT_LINES = 2


@dataclass
class FilesystemQueryConfig:
    """Structured configuration for the filesystem_query plugin.

    Exclusion is checked in this order:
    1. Force-include patterns (``include_patterns``) — if matched, the path is
       never excluded regardless of other rules.
    2. Hardcoded/config-file exclude patterns (``exclude_patterns`` merged with
       ``DEFAULT_EXCLUDE_PATTERNS`` according to ``exclude_mode``).
    3. ``.gitignore`` patterns via an optional ``GitignoreParser`` instance
       injected at runtime with :meth:`set_gitignore`.

    Attributes:
        version: Config format version.
        exclude_patterns: Glob patterns to exclude from searches.
        exclude_mode: How to handle exclusions - "extend" adds to defaults,
                      "replace" uses only the configured patterns.
        include_patterns: Force-include patterns (override exclusions).
        max_results: Maximum number of files/matches to return.
        max_file_size_kb: Maximum file size in KB to grep (skip larger files).
        timeout_seconds: Timeout threshold for auto-backgrounding.
        context_lines: Default number of context lines for grep matches.
    """

    version: str = "1.0"
    exclude_patterns: List[str] = field(default_factory=list)
    exclude_mode: str = "extend"  # "extend" or "replace"
    include_patterns: List[str] = field(default_factory=list)
    max_results: int = DEFAULT_MAX_RESULTS
    max_file_size_kb: int = DEFAULT_MAX_FILE_SIZE_KB
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    context_lines: int = DEFAULT_CONTEXT_LINES

    # Runtime-only: GitignoreParser instance for .gitignore-based exclusions.
    # Not part of serialized config; injected via set_gitignore().
    _gitignore: Optional["GitignoreParser"] = field(
        default=None, init=False, repr=False, compare=False
    )

    def get_effective_excludes(self) -> List[str]:
        """Compute the final list of exclude patterns.

        Returns:
            List of patterns to exclude, merging defaults based on mode.
        """
        if self.exclude_mode == "replace":
            return list(self.exclude_patterns)
        else:
            # extend mode: combine defaults with configured patterns
            combined = set(DEFAULT_EXCLUDE_PATTERNS)
            combined.update(self.exclude_patterns)
            return sorted(combined)

    def should_include(self, path: str) -> bool:
        """Check if a path should be force-included despite exclusions.

        Args:
            path: The file or directory path to check.

        Returns:
            True if the path matches any include_pattern.
        """
        if not self.include_patterns:
            return False

        from fnmatch import fnmatch
        path_str = str(path)

        for pattern in self.include_patterns:
            if fnmatch(path_str, pattern) or pattern in path_str:
                return True
        return False

    def set_gitignore(self, parser: "GitignoreParser") -> None:
        """Inject a ``GitignoreParser`` for ``.gitignore``-based exclusions.

        When set, :meth:`should_exclude` will also consult the parser after
        checking the hardcoded/config-file patterns.  Force-include patterns
        still take priority.

        Args:
            parser: A ``GitignoreParser`` instance initialised with the
                workspace root.
        """
        self._gitignore = parser

    def should_exclude(self, path: str) -> bool:
        """Check if a path should be excluded from search.

        The check proceeds in order:
        1. Force-include patterns — if matched, return ``False`` immediately.
        2. Hardcoded/config-file exclude patterns — if matched, return ``True``.
        3. ``.gitignore`` patterns (via ``GitignoreParser``) — if matched,
           return ``True``.

        Args:
            path: The file or directory path to check (relative to workspace).

        Returns:
            True if the path should be excluded (and not force-included).
        """
        # Force-includes override exclusions
        if self.should_include(path):
            return False

        from fnmatch import fnmatch
        path_str = str(path)
        path_parts = Path(path_str).parts

        for pattern in self.get_effective_excludes():
            # Check if pattern matches any path component
            for part in path_parts:
                if fnmatch(part, pattern):
                    return True
            # Also check full path match
            if fnmatch(path_str, pattern):
                return True

        # Check .gitignore patterns if a parser is available
        if self._gitignore is not None:
            if self._gitignore.is_ignored(Path(path_str)):
                return True

        return False


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a filesystem_query configuration dict.

    Args:
        config: Raw configuration dict loaded from JSON.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: List[str] = []

    # Check version
    version = config.get("version")
    if version and version not in ("1.0", "1"):
        errors.append(f"Unsupported config version: {version}")

    # Validate exclude_patterns
    exclude_patterns = config.get("exclude_patterns", [])
    if not isinstance(exclude_patterns, list):
        errors.append("'exclude_patterns' must be an array")
    elif not all(isinstance(p, str) for p in exclude_patterns):
        errors.append("'exclude_patterns' must contain only strings")

    # Validate exclude_mode
    exclude_mode = config.get("exclude_mode", "extend")
    if exclude_mode not in ("extend", "replace"):
        errors.append(
            f"Invalid exclude_mode: {exclude_mode}. Must be 'extend' or 'replace'"
        )

    # Validate include_patterns
    include_patterns = config.get("include_patterns", [])
    if not isinstance(include_patterns, list):
        errors.append("'include_patterns' must be an array")
    elif not all(isinstance(p, str) for p in include_patterns):
        errors.append("'include_patterns' must contain only strings")

    # Validate numeric fields
    max_results = config.get("max_results")
    if max_results is not None:
        if not isinstance(max_results, int) or max_results <= 0:
            errors.append("'max_results' must be a positive integer")

    max_file_size_kb = config.get("max_file_size_kb")
    if max_file_size_kb is not None:
        if not isinstance(max_file_size_kb, int) or max_file_size_kb <= 0:
            errors.append("'max_file_size_kb' must be a positive integer")

    timeout_seconds = config.get("timeout_seconds")
    if timeout_seconds is not None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            errors.append("'timeout_seconds' must be a positive number")

    context_lines = config.get("context_lines")
    if context_lines is not None:
        if not isinstance(context_lines, int) or context_lines < 0:
            errors.append("'context_lines' must be a non-negative integer")

    return len(errors) == 0, errors


def load_config(
    path: Optional[str] = None,
    env_var: str = "FILESYSTEM_QUERY_CONFIG_PATH",
    runtime_config: Optional[Dict[str, Any]] = None,
    base_path: Optional[str] = None
) -> FilesystemQueryConfig:
    """Load and validate a filesystem_query configuration.

    Configuration sources are merged in this priority order:
    1. runtime_config (highest priority)
    2. Config file (from path, env_var, or default locations)
    3. Hardcoded defaults (lowest priority)

    Args:
        path: Direct path to config file. If None, uses env_var or defaults.
        env_var: Environment variable name for config path.
        runtime_config: Runtime configuration dict passed to initialize().
        base_path: Base directory for resolving default config locations.
                   Used instead of CWD in daemon mode where CWD is the
                   server's directory, not the client's workspace.

    Returns:
        FilesystemQueryConfig instance with merged configuration.

    Raises:
        FileNotFoundError: If explicit path doesn't exist.
        ConfigValidationError: If config validation fails.
        json.JSONDecodeError: If config file is not valid JSON.
    """
    # Start with defaults
    config_dict: Dict[str, Any] = {
        "version": "1.0",
        "exclude_patterns": [],
        "exclude_mode": "extend",
        "include_patterns": [],
        "max_results": DEFAULT_MAX_RESULTS,
        "max_file_size_kb": DEFAULT_MAX_FILE_SIZE_KB,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "context_lines": DEFAULT_CONTEXT_LINES,
    }

    # Resolve config file path
    config_path = None
    if path is not None:
        config_path = Path(path)
    else:
        env_path = os.environ.get(env_var)
        if env_path:
            config_path = Path(env_path)
        else:
            # Try default locations
            cwd = Path(base_path) if base_path else Path(os.environ.get('JAATO_WORKSPACE_ROOT') or Path.cwd())
            default_paths = [
                cwd / ".jaato" / "filesystem_query.json",
                cwd / "filesystem_query.json",
                Path.home() / ".config" / "jaato" / "filesystem_query.json",
            ]
            for default_path in default_paths:
                if default_path.exists():
                    config_path = default_path
                    logger.debug("Found config file at: %s", config_path)
                    break

    # Load from config file if found
    if config_path is not None:
        if path is not None and not config_path.exists():
            raise FileNotFoundError(
                f"Filesystem query config file not found: {config_path}"
            )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)

            # Validate file config
            is_valid, errors = validate_config(file_config)
            if not is_valid:
                raise ConfigValidationError(errors)

            # Merge file config into defaults
            for key, value in file_config.items():
                if key in config_dict:
                    config_dict[key] = value

            logger.info("Loaded filesystem_query config from: %s", config_path)

    # Apply runtime config (highest priority)
    if runtime_config:
        # Validate runtime config
        is_valid, errors = validate_config(runtime_config)
        if not is_valid:
            raise ConfigValidationError(errors)

        # Merge runtime config
        for key, value in runtime_config.items():
            if key in config_dict:
                config_dict[key] = value

        logger.debug("Applied runtime configuration overrides")

    # Create structured config
    return FilesystemQueryConfig(
        version=str(config_dict.get("version", "1.0")),
        exclude_patterns=config_dict.get("exclude_patterns", []),
        exclude_mode=config_dict.get("exclude_mode", "extend"),
        include_patterns=config_dict.get("include_patterns", []),
        max_results=config_dict.get("max_results", DEFAULT_MAX_RESULTS),
        max_file_size_kb=config_dict.get("max_file_size_kb", DEFAULT_MAX_FILE_SIZE_KB),
        timeout_seconds=config_dict.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        context_lines=config_dict.get("context_lines", DEFAULT_CONTEXT_LINES),
    )


def create_default_config(path: str) -> None:
    """Create a default filesystem_query.json file at the given path.

    Args:
        path: Path where to create the config file.
    """
    default_config = {
        "version": "1.0",
        "exclude_patterns": [],
        "exclude_mode": "extend",
        "include_patterns": [],
        "max_results": DEFAULT_MAX_RESULTS,
        "max_file_size_kb": DEFAULT_MAX_FILE_SIZE_KB,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "context_lines": DEFAULT_CONTEXT_LINES
    }

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)

    logger.info("Created default filesystem_query config at: %s", path)


def get_default_excludes() -> List[str]:
    """Get the list of default exclude patterns.

    Returns:
        List of default exclusion patterns.
    """
    return list(DEFAULT_EXCLUDE_PATTERNS)


__all__ = [
    "FilesystemQueryConfig",
    "ConfigValidationError",
    "load_config",
    "validate_config",
    "create_default_config",
    "get_default_excludes",
    "DEFAULT_EXCLUDE_PATTERNS",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_MAX_FILE_SIZE_KB",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_CONTEXT_LINES",
]
