"""Plugin system for tool discovery and management.

This package provides a plugin architecture for managing tool implementations
that can be discovered, enabled/disabled, and used by the AI tool runner.

Usage:
    from shared.plugins import PluginRegistry

    registry = PluginRegistry()
    registry.discover()

    # List available plugins
    print(registry.list_available())  # ['cli', 'mcp', ...]

    # Enable specific plugins
    registry.enable('cli', config={'extra_paths': ['/usr/local/bin']})

    # Get tools for enabled plugins
    declarations = registry.get_enabled_declarations()
    executors = registry.get_enabled_executors()

    # Disable when done
    registry.disable_all()
"""

from .base import ToolPlugin
from .registry import PluginRegistry

__all__ = ['ToolPlugin', 'PluginRegistry']
