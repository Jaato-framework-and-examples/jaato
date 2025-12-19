"""Filesystem Query Plugin - Read-only tools for exploring the filesystem.

This plugin provides fast, safe, auto-approved tools for:
- Finding files with glob patterns (glob_files)
- Searching file contents with regex (grep_content)

Both tools return structured JSON output and support background execution
for large searches.
"""

from .plugin import FilesystemQueryPlugin, create_plugin
from .config_loader import (
    FilesystemQueryConfig,
    ConfigValidationError,
    load_config,
    create_default_config,
    get_default_excludes,
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_MAX_RESULTS,
    DEFAULT_MAX_FILE_SIZE_KB,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_CONTEXT_LINES,
)

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    # Plugin
    "FilesystemQueryPlugin",
    "create_plugin",
    # Configuration
    "FilesystemQueryConfig",
    "ConfigValidationError",
    "load_config",
    "create_default_config",
    "get_default_excludes",
    # Constants
    "DEFAULT_EXCLUDE_PATTERNS",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_MAX_FILE_SIZE_KB",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_CONTEXT_LINES",
]
