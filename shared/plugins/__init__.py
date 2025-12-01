"""Plugin system for tool discovery and management.

This package provides a plugin architecture for managing tool implementations
that can be discovered, exposed/unexposed, and used by the AI tool runner.

Usage:
    from shared.plugins import PluginRegistry

    registry = PluginRegistry()
    registry.discover()

    # List available plugins
    print(registry.list_available())  # ['cli', 'mcp', ...]

    # Expose specific plugins' tools to the model
    registry.expose_tool('cli', config={'extra_paths': ['/usr/local/bin']})

    # Get tools for exposed plugins
    declarations = registry.get_exposed_declarations()
    executors = registry.get_exposed_executors()

    # Unexpose when done
    registry.unexpose_all()
"""

from .base import ToolPlugin
from .registry import PluginRegistry

__all__ = ['ToolPlugin', 'PluginRegistry']
