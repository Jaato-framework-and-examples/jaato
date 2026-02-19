"""Background task processing plugin.

This package provides infrastructure for background task execution:

- BackgroundCapable protocol: Interface for plugins to declare background support
- BackgroundCapableMixin: Default implementation for easy plugin integration
- BackgroundPlugin: Orchestrator that exposes tools to the model

Usage:
    # Make a plugin background-capable using the mixin
    from shared.plugins.background import BackgroundCapableMixin

    class MyPlugin(BackgroundCapableMixin):
        def __init__(self):
            super().__init__(max_workers=4)
            # ... plugin init ...

        def supports_background(self, tool_name: str) -> bool:
            return tool_name == 'slow_operation'

        def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
            if tool_name == 'slow_operation':
                return 10.0  # Auto-background after 10 seconds
            return None

    # Or use the protocol directly for custom implementations
    from shared.plugins.background import BackgroundCapable, TaskHandle, TaskResult
"""

from .protocol import (
    BackgroundCapable,
    TaskHandle,
    TaskInfo,
    TaskOutput,
    TaskResult,
    TaskStatus,
)
from .mixin import BackgroundCapableMixin
from .plugin import BackgroundPlugin, create_plugin

PLUGIN_KIND = "tool"

__all__ = [
    # Protocol and data structures
    'BackgroundCapable',
    'TaskHandle',
    'TaskInfo',
    'TaskOutput',
    'TaskResult',
    'TaskStatus',
    # Mixin for easy implementation
    'BackgroundCapableMixin',
    # Orchestrator plugin
    'BackgroundPlugin',
    'create_plugin',
    # Plugin kind marker
    'PLUGIN_KIND',
]
