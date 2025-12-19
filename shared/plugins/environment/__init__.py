# shared/plugins/environment/__init__.py

"""Environment awareness plugin for querying execution environment details.

This plugin provides the `get_environment` tool that returns OS, shell,
architecture, and working directory information.
"""

from .plugin import EnvironmentPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    "EnvironmentPlugin",
    "create_plugin",
]
