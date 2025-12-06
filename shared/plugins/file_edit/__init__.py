"""File editing plugin for reading, modifying, and managing files.

This plugin provides tools for:
- Reading file contents
- Updating existing files (with diff preview and backup)
- Creating new files (with content preview)
- Removing files (with backup)
- Undoing file changes (restore from backup)

Integrates with the permission system to show diffs for approval before
making file modifications.
"""

PLUGIN_KIND = "tool"

from .plugin import FileEditPlugin, create_plugin

__all__ = ['FileEditPlugin', 'create_plugin']
