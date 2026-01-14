"""File editing plugin for reading, modifying, and managing files.

This plugin provides tools for:
- Reading file contents
- Updating existing files (with diff preview and backup)
- Creating new files (with content preview)
- Removing files (with backup)
- Moving/renaming files (with backup)
- Undoing file changes (restore from backup)
- Atomic multi-file operations (all-or-nothing with rollback)
- Find and replace across files (regex-based with glob patterns)
- Backup management (list, restore)

Integrates with the permission system to show diffs for approval before
making file modifications.
"""

PLUGIN_KIND = "tool"

from .plugin import FileEditPlugin, create_plugin
from .backup import BackupManager, BackupInfo
from .multi_file import (
    MultiFileExecutor,
    MultiFileResult,
    FileOperation,
    OperationType,
    generate_multi_file_diff_preview,
)
from .find_replace import (
    FindReplaceExecutor,
    FindReplaceResult,
    GitignoreParser,
    generate_find_replace_preview,
)

__all__ = [
    # Plugin
    'FileEditPlugin',
    'create_plugin',
    # Backup
    'BackupManager',
    'BackupInfo',
    # Multi-file
    'MultiFileExecutor',
    'MultiFileResult',
    'FileOperation',
    'OperationType',
    'generate_multi_file_diff_preview',
    # Find/replace
    'FindReplaceExecutor',
    'FindReplaceResult',
    'GitignoreParser',
    'generate_find_replace_preview',
]
