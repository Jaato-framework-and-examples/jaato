"""AST Search Plugin - Structural code search using AST patterns.

This plugin provides semantic code search capabilities using ast-grep-py,
allowing searches based on code structure rather than just text patterns.

Supports multi-language codebases including Python, JavaScript, TypeScript,
Go, Rust, Java, C/C++, and more.
"""

from .plugin import (
    ASTSearchPlugin,
    create_plugin,
    LANGUAGE_EXTENSIONS,
    EXTENSION_TO_LANGUAGE,
    DEFAULT_MAX_RESULTS,
    DEFAULT_CONTEXT_LINES,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_EXCLUDE_DIRS,
)

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    # Plugin
    "ASTSearchPlugin",
    "create_plugin",
    # Constants
    "LANGUAGE_EXTENSIONS",
    "EXTENSION_TO_LANGUAGE",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_CONTEXT_LINES",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_EXCLUDE_DIRS",
]
