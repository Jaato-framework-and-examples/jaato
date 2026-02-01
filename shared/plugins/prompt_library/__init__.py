"""Prompt library plugin for managing reusable prompts.

This plugin enables users to create, manage, and use reusable prompts from
a library stored in .jaato/prompts/ (project) and ~/.jaato/prompts/ (global).

It also provides read-only interop with Claude Code skills (.claude/skills/)
and legacy commands (.claude/commands/).
"""

from .plugin import PromptLibraryPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'PromptLibraryPlugin',
    'create_plugin',
]
