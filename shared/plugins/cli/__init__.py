"""CLI tool plugin for executing local shell commands.

This plugin provides the `cli_based_tool` function that executes shell commands
on the local machine via subprocess.
"""

from .plugin import CLIToolPlugin, create_plugin

__all__ = [
    'CLIToolPlugin',
    'create_plugin',
]
