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


def _detect_workspace_root() -> Optional[str]:
    """Auto-detect workspace root from environment variables.

    Checks JAATO_WORKSPACE_ROOT first, then workspaceRoot.

    Returns:
        Absolute path to workspace root, or None if not configured.
    """
    workspace = os.environ.get('JAATO_WORKSPACE_ROOT')
    if workspace:
        return os.path.realpath(os.path.abspath(workspace))
    workspace = os.environ.get('workspaceRoot')
    if workspace:
        return os.path.realpath(os.path.abspath(workspace))
    return None


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

    Path Sandboxing:
        When workspace_root is configured, file operations are restricted to
        paths within the workspace. Paths outside the workspace are only
        allowed if they've been authorized via the plugin registry (e.g.,
        by the references plugin for external documentation).
    """

    def __init__(self):
        self._backup_manager: Optional[BackupManager] = None
        self._initialized = False
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Workspace root for path sandboxing
        self._workspace_root: Optional[str] = None
        # Plugin registry for checking external path authorization
        self._plugin_registry = None

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
                - workspace_root: Path to workspace root for sandboxing.
                                  Auto-detected from JAATO_WORKSPACE_ROOT or
                                  workspaceRoot env vars if not specified.
        """
        config = config or {}

        # Extract agent name for trace logging
        self._agent_name = config.get("agent_name")

        # Configure workspace root for path sandboxing
        workspace_root = config.get("workspace_root")
        if workspace_root:
            self._workspace_root = os.path.realpath(os.path.abspath(workspace_root))
        else:
            self._workspace_root = _detect_workspace_root()

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
        workspace_str = self._workspace_root or "none"
        self._trace(f"initialize: backup_dir={backup_dir_str}, workspace_root={workspace_str}")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._trace("shutdown: cleaning up")
        self._backup_manager = None
        self._initialized = False
        self._workspace_root = None
        self._plugin_registry = None

    def set_plugin_registry(self, registry) -> None:
        """Set the plugin registry for checking external path authorization.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry
        self._trace("set_plugin_registry: registry set")

    def set_workspace_path(self, path: Optional[str]) -> None:
        """Update the workspace root path.

        Called when a client connects with a different working directory.

        Args:
            path: The new workspace root path, or None to disable sandboxing.
        """
        if path:
            self._workspace_root = os.path.realpath(os.path.abspath(path))
        else:
            self._workspace_root = None
        self._trace(f"set_workspace_path: workspace_root={self._workspace_root}")

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is allowed for access.

        A path is allowed if:
        1. No workspace_root is configured (sandboxing disabled)
        2. The path is within the workspace_root
        3. The path is authorized via the plugin registry

        Args:
            path: Path to check.

        Returns:
            True if access is allowed, False otherwise.
        """
        # If no workspace_root, allow all paths
        if not self._workspace_root:
            return True

        # Normalize the path
        abs_path = os.path.realpath(os.path.abspath(path))

        # Check if within workspace
        workspace_prefix = self._workspace_root.rstrip(os.sep) + os.sep
        if abs_path == self._workspace_root or abs_path.startswith(workspace_prefix):
            return True

        # Check if authorized via plugin registry
        if self._plugin_registry and self._plugin_registry.is_path_authorized(abs_path):
            self._trace(f"_is_path_allowed: {path} authorized via registry")
            return True

        self._trace(f"_is_path_allowed: {path} blocked (outside workspace)")
        return False

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, making relative paths relative to workspace_root.

        This ensures that when a client specifies a relative path like
        'customer-domain-api/pom.xml', it resolves to the client's workspace
        directory rather than the server's current working directory.

        Args:
            path: Path string (absolute or relative).

        Returns:
            Resolved Path object. Relative paths are resolved against
            workspace_root if configured, otherwise against CWD.
        """
        p = Path(path)
        if p.is_absolute():
            return p
        if self._workspace_root:
            resolved = Path(self._workspace_root) / p
            self._trace(f"_resolve_path: {path} -> {resolved} (relative to workspace)")
            return resolved
        return p

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
                           "encoding handling and structured metadata. Supports chunked reading with "
                           "offset and limit parameters for large files.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start reading from (1-indexed). Default is 1 (start of file)."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read. If not specified, reads the entire file."
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
- `readFile(path, offset=None, limit=None)`: Read file contents. Safe operation, no approval needed.
  - For large files, use `offset` (1-indexed line number) and `limit` (max lines) for chunked reading.
  - Example: `readFile(path="large.txt", offset=1, limit=100)` reads lines 1-100.
  - Example: `readFile(path="large.txt", offset=101, limit=100)` reads lines 101-200.
  - Chunked responses include: `total_lines`, `start_line`, `end_line`, and `has_more` (boolean).
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

        file_path = self._resolve_path(path)
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

        file_path = self._resolve_path(path)
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

        file_path = self._resolve_path(path)
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

        source = self._resolve_path(source_path)
        destination = self._resolve_path(destination_path)

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
        """Execute readFile tool.

        Supports chunked reading via offset and limit parameters:
        - offset: Line number to start from (1-indexed, default: 1)
        - limit: Maximum lines to read (default: entire file)

        Returns metadata including has_more flag when limit is used.
        """
        path = args.get("path", "")
        offset = args.get("offset")  # 1-indexed line number
        limit = args.get("limit")    # Max lines to read
        self._trace(f"readFile: path={path}, offset={offset}, limit={limit}")

        if not path:
            return {"error": "path is required"}

        # Resolve path first, then check if allowed
        file_path = self._resolve_path(path)

        # Check if resolved path is allowed (within workspace or authorized)
        if not self._is_path_allowed(str(file_path)):
            return {"error": f"File not found: {path}"}

        if not file_path.exists():
            return {"error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}"}

        # Validate offset and limit if provided
        if offset is not None:
            if not isinstance(offset, int) or offset < 1:
                return {"error": "offset must be a positive integer (1-indexed)"}

        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                return {"error": "limit must be a positive integer"}

        try:
            content = file_path.read_text()
            all_lines = content.splitlines(keepends=True)
            total_lines = len(all_lines)

            # Apply chunking if offset or limit specified
            if offset is not None or limit is not None:
                start_idx = (offset - 1) if offset else 0  # Convert to 0-indexed

                if limit is not None:
                    end_idx = start_idx + limit
                    selected_lines = all_lines[start_idx:end_idx]
                    has_more = end_idx < total_lines
                else:
                    selected_lines = all_lines[start_idx:]
                    has_more = False

                # Reconstruct content from selected lines
                chunk_content = "".join(selected_lines)

                # Calculate actual start/end line numbers (1-indexed)
                actual_start = start_idx + 1
                actual_end = min(start_idx + len(selected_lines), total_lines)

                return {
                    "path": path,
                    "content": chunk_content,
                    "size": len(chunk_content),
                    "lines": len(selected_lines),
                    "total_lines": total_lines,
                    "start_line": actual_start,
                    "end_line": actual_end,
                    "has_more": has_more
                }
            else:
                # No chunking - return entire file
                return {
                    "path": path,
                    "content": content,
                    "size": len(content),
                    "lines": total_lines
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

        file_path = self._resolve_path(path)
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

        file_path = self._resolve_path(path)
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

        file_path = self._resolve_path(path)
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

        source = self._resolve_path(source_path)
        destination = self._resolve_path(destination_path)

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

        file_path = self._resolve_path(path)

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
