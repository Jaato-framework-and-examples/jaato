"""Slash command plugin for processing /command references.

This plugin enables users to reference command files in .jaato/commands/ directory
using /command_name syntax. The model can then call processCommand to read and
execute the referenced command file contents.
"""

from .plugin import SlashCommandPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'SlashCommandPlugin',
    'create_plugin',
]
