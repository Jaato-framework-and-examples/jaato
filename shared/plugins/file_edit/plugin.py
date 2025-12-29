"""File editing plugin implementation.

Provides tools for reading, modifying, and managing files with
integrated permission approval (showing diffs) and automatic backups.
"""

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..base import UserCommand, PermissionDisplayInfo
from ..model_provider.types import ToolSchema
from .backup import BackupManager
from .diff_utils import (
    generate_unified_diff,
    generate_new_file_diff,
    generate_delete_file_diff,
    generate_move_file_diff,
    summarize_diff,
    DEFAULT_MAX_LINES,
)


class FileEditPlugin:
    """Plugin for file reading and editing operations.

    Tools provided:
    - readFile: Read file contents (auto-approved, low risk)
    - updateFile: Modify existing file (shows diff for approval, creates backup)
    - writeNewFile: Create new file (shows content for approval)
    - removeFile: Delete file (shows confirmation, creates backup)
    - moveFile: Move/rename file (shows confirmation, creates backup)
    - renameFile: Alias for moveFile (for discoverability)
    - undoFileChange: Restore from most recent backup (auto-approved)

    Integrates with the permission system to show formatted diffs
    when requesting approval for file modifications.
    """

    def __init__(self):
        self._backup_manager: Optional[BackupManager] = None
        self._initialized = False
        # Agent context for trace logging
        self._agent_name: Optional[str] = None

    @property
    def name(self) -> str:
        return "file_edit"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [FILE_EDIT{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the file edit plugin.

        Creates backup directory and ensures .jaato is in .gitignore.

        Args:
            config: Optional configuration dict with:
                - backup_dir: Custom backup directory (default: .jaato/backups)
        """
        config = config or {}

        # Extract agent name for trace logging
        self._agent_name = config.get("agent_name")

        # Initialize backup manager
        backup_dir = config.get("backup_dir")
        if backup_dir:
            self._backup_manager = BackupManager(Path(backup_dir))
        else:
            self._backup_manager = BackupManager()

        # Ensure .jaato is in .gitignore
        self._ensure_gitignore()

        self._initialized = True
        backup_dir_str = str(self._backup_manager._base_dir) if self._backup_manager else "none"
        self._trace(f"initialize: backup_dir={backup_dir_str}")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._trace("shutdown: cleaning up")
        self._backup_manager = None
        self._initialized = False

    def _ensure_gitignore(self) -> None:
        """Add .jaato to .gitignore if it exists and entry is missing."""
        gitignore = Path(".gitignore")
        if not gitignore.exists():
            return

        try:
            content = gitignore.read_text()
            lines = content.splitlines()

            # Check if .jaato is already present
            if ".jaato" in lines or ".jaato/" in lines:
                return

            # Add .jaato to gitignore
            with gitignore.open("a") as f:
                # Add newline if file doesn't end with one
                if content and not content.endswith("\n"):
                    f.write("\n")
                f.write(".jaato\n")
        except OSError:
            # If we can't read/write gitignore, just skip
            pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for file editing tools."""
        return [
            ToolSchema(
                name="readFile",
                description="Read the contents of a file. ALWAYS use this instead of `cat`, `head`, "
                           "`tail`, or `less` CLI commands. Returns file content as text with proper "
                           "encoding handling and structured metadata.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            ),
            ToolSchema(
                name="updateFile",
                description="Update an existing file with new content. Shows a diff for approval "
                           "and creates a backup before modifying. Use this for modifying existing files.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to update"
                        },
                        "new_content": {
                            "type": "string",
                            "description": (
                                "The complete new content to write to the file. "
                                "Provide the raw file content directly - do NOT wrap in quotes, "
                                "triple-quotes, or treat as a string literal. The content is "
                                "written verbatim to the file."
                            )
                        }
                    },
                    "required": ["path", "new_content"]
                }
            ),
            ToolSchema(
                name="writeNewFile",
                description="Create a new file with the specified content. Shows the content for "
                           "approval before creating. Fails if the file already exists.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path where the new file should be created"
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "The content to write to the new file. "
                                "Provide the raw file content directly - do NOT wrap in quotes, "
                                "triple-quotes, or treat as a string literal. For example, for a "
                                "Python file, start directly with 'import ...' or code, not with "
                                "'''...''' or quotes around the code. The content is written "
                                "verbatim to the file."
                            )
                        }
                    },
                    "required": ["path", "content"]
                }
            ),
            ToolSchema(
                name="removeFile",
                description="Delete a file. Creates a backup before deletion so it can be restored "
                           "with undoFileChange if needed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to delete"
                        }
                    },
                    "required": ["path"]
                }
            ),
            ToolSchema(
                name="moveFile",
                description="Move or rename a file. ALWAYS use this instead of `mv` CLI command. "
                           "Creates destination directories if needed and creates a backup before "
                           "moving. Fails if destination already exists unless overwrite=True.",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_path": {
                            "type": "string",
                            "description": "Path to the source file to move"
                        },
                        "destination_path": {
                            "type": "string",
                            "description": "Path where the file should be moved to"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "If True, overwrite destination if it exists. Default is False."
                        }
                    },
                    "required": ["source_path", "destination_path"]
                }
            ),
            ToolSchema(
                name="renameFile",
                description="Rename a file (alias for moveFile). ALWAYS use this instead of `mv` CLI "
                           "command. Creates destination directories if needed and creates a backup "
                           "before renaming. Fails if destination already exists unless overwrite=True.",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_path": {
                            "type": "string",
                            "description": "Path to the source file to rename"
                        },
                        "destination_path": {
                            "type": "string",
                            "description": "New path/name for the file"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "If True, overwrite destination if it exists. Default is False."
                        }
                    },
                    "required": ["source_path", "destination_path"]
                }
            ),
            ToolSchema(
                name="undoFileChange",
                description="Restore a file from its most recent backup. Use this to undo a previous "
                           "updateFile or removeFile operation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to restore"
                        }
                    },
                    "required": ["path"]
                }
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor functions for each tool."""
        return {
            "readFile": self._execute_read_file,
            "updateFile": self._execute_update_file,
            "writeNewFile": self._execute_write_new_file,
            "removeFile": self._execute_remove_file,
            "moveFile": self._execute_move_file,
            "renameFile": self._execute_move_file,  # Alias for moveFile
            "undoFileChange": self._execute_undo_file_change,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for file editing tools."""
        return """You have access to file editing tools.

IMPORTANT: Always prefer these tools over CLI commands:
- Use `readFile` instead of `cat`, `head`, `tail`, or `less`
- Use `updateFile`/`writeNewFile` instead of `echo >`, `sed`, or heredocs
- Use `removeFile` instead of `rm`
- Use `moveFile`/`renameFile` instead of `mv`

These tools provide structured output, automatic backups, and proper encoding handling.

Tools available:
- `readFile(path)`: Read file contents. Safe operation, no approval needed.
- `updateFile(path, new_content)`: Update an existing file. Shows diff for approval and creates backup.
- `writeNewFile(path, content)`: Create a new file. Shows content for approval. Fails if file exists.
- `removeFile(path)`: Delete a file. Creates backup before deletion.
- `moveFile(source_path, destination_path, overwrite=False)`: Move or rename a file. Creates destination directories if needed. Creates backup before moving. Fails if destination exists unless overwrite=True.
- `renameFile(source_path, destination_path, overwrite=False)`: Alias for moveFile. Use for renaming files.
- `undoFileChange(path)`: Restore a file from its most recent backup.

IMPORTANT: When using updateFile or writeNewFile, provide the raw file content directly.
Do NOT wrap the content in quotes, triple-quotes (''' or \"\"\"), or treat it as a string literal.
For example, to create a Python file, the content should start with 'import ...' or actual code,
NOT with ''' or quotes around the code. The content parameter value is written verbatim to the file.

File modifications (updateFile, writeNewFile, removeFile, moveFile) will show you a preview
and require approval before execution. Backups are automatically created for
updateFile, removeFile, and moveFile operations."""

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved.

        readFile and undoFileChange are low-risk operations.
        """
        return ["readFile", "undoFileChange"]

    def get_user_commands(self) -> List[UserCommand]:
        """File edit plugin provides model tools only."""
        return []

    def format_permission_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        channel_type: str
    ) -> Optional[PermissionDisplayInfo]:
        """Format permission request with diff display for file operations.

        This method is called by the permission system to get a custom
        display format for file editing tools.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            channel_type: Type of channel (console, webhook, file)

        Returns:
            PermissionDisplayInfo with formatted diff, or None for default display
        """
        if tool_name == "updateFile":
            return self._format_update_file(arguments)
        elif tool_name == "writeNewFile":
            return self._format_write_new_file(arguments)
        elif tool_name == "removeFile":
            return self._format_remove_file(arguments)
        elif tool_name in ("moveFile", "renameFile"):
            return self._format_move_file(arguments)

        return None

    def _format_update_file(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format updateFile for permission display."""
        path = arguments.get("path", "")
        # Accept both 'new_content' (canonical) and 'content' (for consistency with writeNewFile)
        new_content = arguments.get("new_content") or arguments.get("content", "")

        file_path = Path(path)
        if not file_path.exists():
            # File doesn't exist - skip permission, let executor return the error
            return None

        try:
            old_content = file_path.read_text()
        except OSError as e:
            return PermissionDisplayInfo(
                summary=f"Update file: {path} (read error)",
                details=f"Error reading file: {e}",
                format_hint="text"
            )

        diff_text, truncated, total_lines = generate_unified_diff(
            old_content, new_content, path, max_lines=DEFAULT_MAX_LINES
        )

        summary = summarize_diff(old_content, new_content, path)

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated,
            original_lines=total_lines if truncated else None
        )

    def _format_write_new_file(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format writeNewFile for permission display."""
        path = arguments.get("path", "")
        content = arguments.get("content", "")

        file_path = Path(path)
        if file_path.exists():
            # File already exists - skip permission, let executor return the error
            return None

        diff_text, truncated, total_lines = generate_new_file_diff(
            content, path, max_lines=DEFAULT_MAX_LINES
        )

        lines = content.splitlines()
        summary = f"Create new file: {path} ({len(lines)} lines)"

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated,
            original_lines=total_lines if truncated else None
        )

    def _format_remove_file(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format removeFile for permission display."""
        path = arguments.get("path", "")

        file_path = Path(path)
        if not file_path.exists():
            # File doesn't exist - skip permission, let executor return the error
            return None

        try:
            content = file_path.read_text()
        except OSError as e:
            return PermissionDisplayInfo(
                summary=f"Delete file: {path} (read error)",
                details=f"Error reading file: {e}",
                format_hint="text"
            )

        diff_text, truncated, total_lines = generate_delete_file_diff(
            content, path, max_lines=DEFAULT_MAX_LINES
        )

        lines = content.splitlines()
        summary = f"Delete file: {path} ({len(lines)} lines, backup will be created)"

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated,
            original_lines=total_lines if truncated else None
        )

    def _format_move_file(self, arguments: Dict[str, Any]) -> Optional[PermissionDisplayInfo]:
        """Format moveFile/renameFile for permission display."""
        source_path = arguments.get("source_path", "")
        destination_path = arguments.get("destination_path", "")
        overwrite = arguments.get("overwrite", False)

        source = Path(source_path)
        destination = Path(destination_path)

        if not source.exists():
            # Source doesn't exist - skip permission, let executor return the error
            return None

        if destination.exists() and not overwrite:
            # Destination exists and overwrite not set - skip permission, let executor return the error
            return None

        try:
            content = source.read_text()
        except OSError as e:
            return PermissionDisplayInfo(
                summary=f"Move file: {source_path} (read error)",
                details=f"Error reading source file: {e}",
                format_hint="text"
            )

        diff_text, truncated, total_lines = generate_move_file_diff(
            source_path, destination_path, content, max_lines=DEFAULT_MAX_LINES
        )

        lines = content.splitlines()
        if overwrite and destination.exists():
            summary = f"Move file: {source_path} -> {destination_path} ({len(lines)} lines, overwrite enabled)"
        else:
            summary = f"Move file: {source_path} -> {destination_path} ({len(lines)} lines)"

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated,
            original_lines=total_lines if truncated else None
        )

    # Tool executors

    def _execute_read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute readFile tool."""
        path = args.get("path", "")
        self._trace(f"readFile: path={path}")

        if not path:
            return {"error": "path is required"}

        file_path = Path(path)
        if not file_path.exists():
            return {"error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}"}

        try:
            content = file_path.read_text()
            return {
                "path": path,
                "content": content,
                "size": len(content),
                "lines": len(content.splitlines())
            }
        except OSError as e:
            return {"error": f"Failed to read file: {e}"}

    def _execute_update_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute updateFile tool."""
        path = args.get("path", "")
        # Accept both 'new_content' (canonical) and 'content' (for consistency with writeNewFile)
        new_content = args.get("new_content") or args.get("content", "")
        self._trace(f"updateFile: path={path}, content_len={len(new_content)}")

        if not path:
            return {"error": "path is required"}

        file_path = Path(path)
        if not file_path.exists():
            return {"error": f"File not found: {path}. Use writeNewFile for new files."}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}"}

        # Create backup before modification
        backup_path = None
        if self._backup_manager:
            backup_path = self._backup_manager.create_backup(file_path)

        try:
            file_path.write_text(new_content)
            result = {
                "success": True,
                "path": path,
                "size": len(new_content),
                "lines": len(new_content.splitlines())
            }
            if backup_path:
                result["backup"] = str(backup_path)
            return result
        except OSError as e:
            return {"error": f"Failed to write file: {e}"}

    def _execute_write_new_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute writeNewFile tool."""
        path = args.get("path", "")
        content = args.get("content", "")
        self._trace(f"writeNewFile: path={path}, content_len={len(content)}")

        if not path:
            return {"error": "path is required"}

        file_path = Path(path)
        if file_path.exists():
            return {"error": f"File already exists: {path}. Use updateFile to modify existing files."}

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return {
                "success": True,
                "path": path,
                "size": len(content),
                "lines": len(content.splitlines())
            }
        except OSError as e:
            return {"error": f"Failed to create file: {e}"}

    def _execute_remove_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute removeFile tool."""
        path = args.get("path", "")
        self._trace(f"removeFile: path={path}")

        if not path:
            return {"error": "path is required"}

        file_path = Path(path)
        if not file_path.exists():
            return {"error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}"}

        # Create backup before deletion
        backup_path = None
        if self._backup_manager:
            backup_path = self._backup_manager.create_backup(file_path)

        try:
            file_path.unlink()
            result = {
                "success": True,
                "path": path,
                "deleted": True
            }
            if backup_path:
                result["backup"] = str(backup_path)
            return result
        except OSError as e:
            return {"error": f"Failed to delete file: {e}"}

    def _execute_move_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute moveFile/renameFile tool."""
        source_path = args.get("source_path", "")
        destination_path = args.get("destination_path", "")
        overwrite = args.get("overwrite", False)
        self._trace(f"moveFile: source={source_path}, dest={destination_path}, overwrite={overwrite}")

        if not source_path:
            return {"error": "source_path is required", "source": source_path}

        if not destination_path:
            return {"error": "destination_path is required", "source": source_path}

        source = Path(source_path)
        destination = Path(destination_path)

        if not source.exists():
            return {"error": "Source file does not exist", "source": source_path}

        if not source.is_file():
            return {"error": f"Source is not a file: {source_path}", "source": source_path}

        if destination.exists() and not overwrite:
            return {
                "error": "Destination file already exists. Use overwrite=True to replace it.",
                "source": source_path,
                "destination": destination_path
            }

        # Create backup of source file before moving
        backup_path = None
        if self._backup_manager:
            backup_path = self._backup_manager.create_backup(source)

        # If overwriting, also backup the destination
        dest_backup_path = None
        if destination.exists() and overwrite and self._backup_manager:
            dest_backup_path = self._backup_manager.create_backup(destination)

        try:
            # Create destination parent directories if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Move the file (this handles overwrite if destination exists)
            shutil.move(str(source), str(destination))

            result = {
                "success": True,
                "source": source_path,
                "destination": destination_path
            }
            if backup_path:
                result["source_backup"] = str(backup_path)
            if dest_backup_path:
                result["destination_backup"] = str(dest_backup_path)
            return result
        except OSError as e:
            return {
                "error": f"Failed to move file: {e}",
                "source": source_path,
                "destination": destination_path
            }

    def _execute_undo_file_change(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute undoFileChange tool."""
        path = args.get("path", "")
        self._trace(f"undoFileChange: path={path}")

        if not path:
            return {"error": "path is required"}

        if not self._backup_manager:
            return {"error": "Backup manager not initialized"}

        file_path = Path(path)

        # Check if backup exists
        if not self._backup_manager.has_backup(file_path):
            return {"error": f"No backup found for: {path}"}

        # Get the backup path for reporting
        backup_path = self._backup_manager.get_latest_backup(file_path)

        # Restore from backup
        if self._backup_manager.restore_from_backup(file_path):
            return {
                "success": True,
                "path": path,
                "restored_from": str(backup_path) if backup_path else "unknown",
                "message": f"File restored from backup"
            }
        else:
            return {"error": f"Failed to restore file from backup"}


def create_plugin() -> FileEditPlugin:
    """Factory function to create the file edit plugin instance."""
    return FileEditPlugin()
