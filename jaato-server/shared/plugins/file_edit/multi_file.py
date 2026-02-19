"""Multi-file atomic operations with transaction/rollback support.

Provides atomic execution of multiple file operations - either ALL changes
succeed or NONE are applied (rollback on failure).

Supported operations:
- edit: Modify existing file content
- create: Create a new file
- delete: Remove a file
- rename: Move/rename a file

Example usage:
    operations = [
        {"action": "edit", "path": "src/foo.py", "old": "...", "new": "..."},
        {"action": "create", "path": "src/new.py", "content": "..."},
        {"action": "rename", "from": "src/old.py", "to": "src/new_name.py"},
        {"action": "delete", "path": "src/deprecated.py"}
    ]

    executor = MultiFileExecutor(resolve_path_fn, is_path_allowed_fn)
    result = executor.execute(operations)
"""

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.ui_utils import ellipsize_path, ellipsize_path_pair
from .edit_core import apply_edit, EditNotFoundError, AmbiguousEditError

# Default maximum width for file paths in multi-file previews
DEFAULT_MAX_PATH_WIDTH = 50


class OperationType(Enum):
    """Types of file operations supported."""
    EDIT = "edit"
    CREATE = "create"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class FileOperation:
    """Represents a single file operation in a transaction.

    For edit operations, ``old_content`` and ``new_content`` are treated as
    targeted search-and-replace fragments (not full-file content).  Optional
    ``prologue`` and ``epilogue`` fields provide disambiguation context
    around the search text.
    """
    action: OperationType
    path: Optional[str] = None
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    source_path: Optional[str] = None
    dest_path: Optional[str] = None
    content: Optional[str] = None
    prologue: Optional[str] = None
    epilogue: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileOperation":
        """Create a FileOperation from a dictionary.

        Args:
            data: Dictionary with operation details

        Returns:
            FileOperation instance

        Raises:
            ValueError: If action is invalid or required fields are missing
        """
        action_str = data.get("action", "").lower()
        try:
            action = OperationType(action_str)
        except ValueError:
            raise ValueError(f"Invalid action: {action_str}. "
                           f"Valid actions: {[t.value for t in OperationType]}")

        if action == OperationType.EDIT:
            path = data.get("path")
            if not path:
                raise ValueError("'path' is required for edit operation")
            # 'old' and 'new' are targeted search-and-replace fragments
            old_content = data.get("old")
            new_content = data.get("new")
            if old_content is None:
                raise ValueError("'old' content is required for edit operation")
            if new_content is None:
                raise ValueError("'new' content is required for edit operation")
            prologue = data.get("prologue")
            epilogue = data.get("epilogue")
            return cls(
                action=action, path=path,
                old_content=old_content, new_content=new_content,
                prologue=prologue, epilogue=epilogue,
            )

        elif action == OperationType.CREATE:
            path = data.get("path")
            content = data.get("content")
            if not path:
                raise ValueError("'path' is required for create operation")
            if content is None:
                raise ValueError("'content' is required for create operation")
            return cls(action=action, path=path, content=content)

        elif action == OperationType.DELETE:
            path = data.get("path")
            if not path:
                raise ValueError("'path' is required for delete operation")
            return cls(action=action, path=path)

        elif action == OperationType.RENAME:
            source = data.get("from") or data.get("source_path")
            dest = data.get("to") or data.get("destination_path")
            if not source:
                raise ValueError("'from' (or 'source_path') is required for rename operation")
            if not dest:
                raise ValueError("'to' (or 'destination_path') is required for rename operation")
            return cls(action=action, source_path=source, dest_path=dest)

        # Should not reach here due to enum validation
        raise ValueError(f"Unhandled action: {action_str}")


@dataclass
class RollbackState:
    """Tracks state needed to rollback a single operation."""
    operation_index: int
    operation_type: OperationType
    # For edit/delete: original file content
    original_content: Optional[bytes] = None
    original_path: Optional[Path] = None
    # For create: path of created file to remove
    created_path: Optional[Path] = None
    # For rename: source and dest paths to swap back
    rename_source: Optional[Path] = None
    rename_dest: Optional[Path] = None
    # For rename with overwrite: original dest content
    original_dest_content: Optional[bytes] = None


@dataclass
class OperationResult:
    """Result of a single operation within the batch."""
    success: bool
    operation_index: int
    action: str
    path: Optional[str] = None
    source_path: Optional[str] = None
    dest_path: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiFileResult:
    """Result of executing a batch of file operations."""
    success: bool
    operations_completed: int
    operations_total: int
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    files_renamed: List[Tuple[str, str]] = field(default_factory=list)
    failed_operation: Optional[int] = None
    error: Optional[str] = None
    rollback_completed: bool = False
    operation_results: List[OperationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for tool response."""
        result = {
            "success": self.success,
            "operations_completed": self.operations_completed,
            "operations_total": self.operations_total,
        }

        if self.files_modified:
            result["files_modified"] = self.files_modified
        if self.files_created:
            result["files_created"] = self.files_created
        if self.files_deleted:
            result["files_deleted"] = self.files_deleted
        if self.files_renamed:
            result["files_renamed"] = [
                {"from": src, "to": dst} for src, dst in self.files_renamed
            ]

        if not self.success:
            result["failed_operation_index"] = self.failed_operation
            result["error"] = self.error
            result["rollback_completed"] = self.rollback_completed

        return result


class MultiFileExecutor:
    """Executes multiple file operations atomically.

    All operations either succeed together or are rolled back on failure.
    Uses in-memory copies for rollback rather than filesystem backups.
    """

    def __init__(
        self,
        resolve_path_fn: Callable[[str], Path],
        is_path_allowed_fn: Callable,
        trace_fn: Optional[Callable[[str], None]] = None
    ):
        """Initialize the multi-file executor.

        Args:
            resolve_path_fn: Function to resolve relative paths to absolute Path objects
            is_path_allowed_fn: Function to check if a path is within sandbox.
                Signature: (path: str, mode: str = "read") -> bool
            trace_fn: Optional function for debug tracing
        """
        self._resolve_path = resolve_path_fn
        self._is_path_allowed = is_path_allowed_fn
        self._trace = trace_fn or (lambda msg: None)

    def validate_operations(
        self,
        operations: List[FileOperation]
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Validate all operations before execution.

        Checks:
        - All paths are within sandbox
        - Files exist (for edit/delete)
        - Files don't exist (for create)
        - Source files exist (for rename)
        - Targeted edit anchor found exactly once (for edit)
        - No conflicting operations (e.g., edit + delete same file)
        - Circular renames are detected and handled

        Multiple targeted edits on the same file are allowed and validated
        sequentially (each edit is checked against the result of applying
        all previous edits to that file).

        Args:
            operations: List of operations to validate

        Returns:
            Tuple of (valid, error_message, failed_operation_index)
        """
        # Track what files will be affected
        files_to_edit: Dict[str, List[int]] = {}  # path -> list of operation indices
        files_to_create: Dict[str, int] = {}
        files_to_delete: Dict[str, int] = {}
        files_to_rename_from: Dict[str, int] = {}  # source -> operation index
        files_to_rename_to: Dict[str, int] = {}  # dest -> operation index
        # Cache of "working" file content for sequential validation of
        # multiple edits on the same file.
        working_content: Dict[str, str] = {}

        for i, op in enumerate(operations):
            if op.action == OperationType.EDIT:
                resolved = self._resolve_path(op.path)
                resolved_str = str(resolved)

                # Check sandbox
                if not self._is_path_allowed(resolved_str, mode="write"):
                    return False, f"Path outside workspace: {op.path}", i

                # Check file exists (unless being created in same batch)
                if not resolved.exists() and resolved_str not in files_to_create:
                    # Check if it's being renamed to this path
                    if resolved_str not in files_to_rename_to:
                        return False, f"File not found: {op.path}", i

                # Check for conflict with delete
                if resolved_str in files_to_delete:
                    return False, f"Cannot edit file that is being deleted: {op.path}", i

                # Validate that the targeted edit can be applied.
                # For multiple edits on the same file, validate sequentially
                # against the running "working content".
                if resolved.exists():
                    if resolved_str not in working_content:
                        try:
                            working_content[resolved_str] = resolved.read_text(encoding="utf-8")
                        except OSError as e:
                            return False, f"Cannot read file {op.path}: {e}", i

                    try:
                        working_content[resolved_str] = apply_edit(
                            working_content[resolved_str],
                            op.old_content, op.new_content,
                            op.prologue, op.epilogue,
                        )
                    except EditNotFoundError:
                        return False, f"Edit target not found in {op.path}: {_truncate_for_msg(op.old_content, 80)}", i
                    except AmbiguousEditError:
                        return False, f"Edit target is ambiguous in {op.path} (matched multiple times). Use prologue/epilogue to disambiguate.", i

                files_to_edit.setdefault(resolved_str, []).append(i)

            elif op.action == OperationType.CREATE:
                resolved = self._resolve_path(op.path)
                resolved_str = str(resolved)

                # Check sandbox
                if not self._is_path_allowed(resolved_str, mode="write"):
                    return False, f"Path outside workspace: {op.path}", i

                # Check file doesn't exist (unless being deleted in same batch)
                if resolved.exists() and resolved_str not in files_to_delete:
                    return False, f"File already exists: {op.path}", i

                # Check for duplicate create
                if resolved_str in files_to_create:
                    return False, f"Duplicate create for: {op.path}", i

                files_to_create[resolved_str] = i

            elif op.action == OperationType.DELETE:
                resolved = self._resolve_path(op.path)
                resolved_str = str(resolved)

                # Check sandbox
                if not self._is_path_allowed(resolved_str, mode="write"):
                    return False, f"Path outside workspace: {op.path}", i

                # Check file exists (unless being created in same batch)
                if not resolved.exists() and resolved_str not in files_to_create:
                    return False, f"File not found: {op.path}", i

                # Check for conflict with edit (edit then delete is allowed if in order)
                # Actually, let's disallow this to keep semantics simple
                if resolved_str in files_to_edit:
                    return False, f"Cannot delete file that is being edited: {op.path}", i

                # Check for duplicate delete
                if resolved_str in files_to_delete:
                    return False, f"Duplicate delete for: {op.path}", i

                files_to_delete[resolved_str] = i

            elif op.action == OperationType.RENAME:
                source_resolved = self._resolve_path(op.source_path)
                dest_resolved = self._resolve_path(op.dest_path)
                source_str = str(source_resolved)
                dest_str = str(dest_resolved)

                # Check sandbox for both paths
                if not self._is_path_allowed(source_str, mode="write"):
                    return False, f"Source path outside workspace: {op.source_path}", i
                if not self._is_path_allowed(dest_str, mode="write"):
                    return False, f"Destination path outside workspace: {op.dest_path}", i

                # Check source exists (unless being created in same batch)
                if not source_resolved.exists() and source_str not in files_to_create:
                    # Check if source is being renamed from another file
                    if source_str not in files_to_rename_to:
                        return False, f"Source file not found: {op.source_path}", i

                # Check dest doesn't exist (unless being deleted or renamed away)
                if dest_resolved.exists():
                    if dest_str not in files_to_delete and dest_str not in files_to_rename_from:
                        return False, f"Destination already exists: {op.dest_path}", i

                # Check for duplicate rename from same source
                if source_str in files_to_rename_from:
                    return False, f"Duplicate rename from: {op.source_path}", i

                # Check for duplicate rename to same dest
                if dest_str in files_to_rename_to:
                    return False, f"Duplicate rename to: {op.dest_path}", i

                # Check if source is being edited (rename after edit is OK but complex)
                # For simplicity, disallow editing a file that's being renamed
                if source_str in files_to_edit:
                    return False, f"Cannot rename file that is being edited: {op.source_path}", i

                files_to_rename_from[source_str] = i
                files_to_rename_to[dest_str] = i

        return True, None, None

    def execute(self, operations: List[Dict[str, Any]]) -> MultiFileResult:
        """Execute a batch of file operations atomically.

        All operations either succeed together or are rolled back.

        Args:
            operations: List of operation dictionaries

        Returns:
            MultiFileResult with success/failure details
        """
        self._trace(f"multi_file_edit: starting with {len(operations)} operations")

        # Parse operations
        try:
            parsed_ops = [FileOperation.from_dict(op) for op in operations]
        except ValueError as e:
            return MultiFileResult(
                success=False,
                operations_completed=0,
                operations_total=len(operations),
                error=str(e),
                failed_operation=0
            )

        # Validate all operations
        valid, error, failed_idx = self.validate_operations(parsed_ops)
        if not valid:
            return MultiFileResult(
                success=False,
                operations_completed=0,
                operations_total=len(operations),
                error=error,
                failed_operation=failed_idx
            )

        # Execute operations with rollback tracking
        rollback_stack: List[RollbackState] = []
        result = MultiFileResult(
            success=True,
            operations_completed=0,
            operations_total=len(operations)
        )

        for i, op in enumerate(parsed_ops):
            try:
                op_result = self._execute_single(op, i, rollback_stack)
                result.operation_results.append(op_result)

                if not op_result.success:
                    result.success = False
                    result.failed_operation = i
                    result.error = op_result.error
                    break

                # Track successful operation
                result.operations_completed += 1
                if op.action == OperationType.EDIT:
                    result.files_modified.append(op.path)
                elif op.action == OperationType.CREATE:
                    result.files_created.append(op.path)
                elif op.action == OperationType.DELETE:
                    result.files_deleted.append(op.path)
                elif op.action == OperationType.RENAME:
                    result.files_renamed.append((op.source_path, op.dest_path))

            except Exception as e:
                result.success = False
                result.failed_operation = i
                result.error = f"Unexpected error: {e}"
                result.operation_results.append(OperationResult(
                    success=False,
                    operation_index=i,
                    action=op.action.value,
                    error=str(e)
                ))
                break

        # Rollback if failed
        if not result.success and rollback_stack:
            self._trace(f"multi_file_edit: rolling back {len(rollback_stack)} operations")
            result.rollback_completed = self._rollback(rollback_stack)
            # Clear partial results since we rolled back
            result.files_modified = []
            result.files_created = []
            result.files_deleted = []
            result.files_renamed = []

        self._trace(f"multi_file_edit: completed, success={result.success}")
        return result

    def _execute_single(
        self,
        op: FileOperation,
        index: int,
        rollback_stack: List[RollbackState]
    ) -> OperationResult:
        """Execute a single operation and track rollback state.

        Args:
            op: Operation to execute
            index: Operation index in batch
            rollback_stack: Stack to push rollback state onto

        Returns:
            OperationResult for this operation
        """
        if op.action == OperationType.EDIT:
            return self._execute_edit(op, index, rollback_stack)
        elif op.action == OperationType.CREATE:
            return self._execute_create(op, index, rollback_stack)
        elif op.action == OperationType.DELETE:
            return self._execute_delete(op, index, rollback_stack)
        elif op.action == OperationType.RENAME:
            return self._execute_rename(op, index, rollback_stack)

        return OperationResult(
            success=False,
            operation_index=index,
            action=op.action.value,
            error=f"Unknown action: {op.action}"
        )

    def _execute_edit(
        self,
        op: FileOperation,
        index: int,
        rollback_stack: List[RollbackState]
    ) -> OperationResult:
        """Execute an edit operation using targeted search-and-replace.

        Reads the current file content, applies ``apply_edit()`` to find and
        replace the ``old`` fragment with ``new``, then writes the result.
        """
        resolved = self._resolve_path(op.path)

        # Read original content for rollback
        try:
            original_bytes = resolved.read_bytes()
            current_content = original_bytes.decode()
        except OSError as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="edit",
                path=op.path,
                error=f"Failed to read file: {e}"
            )

        # Apply targeted edit
        try:
            new_content = apply_edit(
                current_content, op.old_content, op.new_content,
                op.prologue, op.epilogue,
            )
        except (EditNotFoundError, AmbiguousEditError) as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="edit",
                path=op.path,
                error=f"Targeted edit failed: {e}"
            )

        # Write result
        try:
            resolved.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="edit",
                path=op.path,
                error=f"Failed to write file: {e}"
            )

        # Track for rollback
        rollback_stack.append(RollbackState(
            operation_index=index,
            operation_type=OperationType.EDIT,
            original_content=original_bytes,
            original_path=resolved
        ))

        return OperationResult(
            success=True,
            operation_index=index,
            action="edit",
            path=op.path,
            details={"size": len(new_content)}
        )

    def _execute_create(
        self,
        op: FileOperation,
        index: int,
        rollback_stack: List[RollbackState]
    ) -> OperationResult:
        """Execute a create operation."""
        resolved = self._resolve_path(op.path)

        # Create parent directories
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="create",
                path=op.path,
                error=f"Failed to create parent directories: {e}"
            )

        # Write content
        try:
            resolved.write_text(op.content, encoding="utf-8")
        except OSError as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="create",
                path=op.path,
                error=f"Failed to create file: {e}"
            )

        # Track for rollback
        rollback_stack.append(RollbackState(
            operation_index=index,
            operation_type=OperationType.CREATE,
            created_path=resolved
        ))

        return OperationResult(
            success=True,
            operation_index=index,
            action="create",
            path=op.path,
            details={"size": len(op.content)}
        )

    def _execute_delete(
        self,
        op: FileOperation,
        index: int,
        rollback_stack: List[RollbackState]
    ) -> OperationResult:
        """Execute a delete operation."""
        resolved = self._resolve_path(op.path)

        # Read original content for rollback
        try:
            original_content = resolved.read_bytes()
        except OSError as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="delete",
                path=op.path,
                error=f"Failed to read file for backup: {e}"
            )

        # Delete file
        try:
            resolved.unlink()
        except OSError as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="delete",
                path=op.path,
                error=f"Failed to delete file: {e}"
            )

        # Track for rollback
        rollback_stack.append(RollbackState(
            operation_index=index,
            operation_type=OperationType.DELETE,
            original_content=original_content,
            original_path=resolved
        ))

        return OperationResult(
            success=True,
            operation_index=index,
            action="delete",
            path=op.path
        )

    def _execute_rename(
        self,
        op: FileOperation,
        index: int,
        rollback_stack: List[RollbackState]
    ) -> OperationResult:
        """Execute a rename operation."""
        source = self._resolve_path(op.source_path)
        dest = self._resolve_path(op.dest_path)

        # Save dest content if it exists (for rollback)
        original_dest_content = None
        if dest.exists():
            try:
                original_dest_content = dest.read_bytes()
            except OSError as e:
                return OperationResult(
                    success=False,
                    operation_index=index,
                    action="rename",
                    source_path=op.source_path,
                    dest_path=op.dest_path,
                    error=f"Failed to read destination file for backup: {e}"
                )

        # Create parent directories for destination
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="rename",
                source_path=op.source_path,
                dest_path=op.dest_path,
                error=f"Failed to create destination directory: {e}"
            )

        # Move file
        try:
            shutil.move(str(source), str(dest))
        except OSError as e:
            return OperationResult(
                success=False,
                operation_index=index,
                action="rename",
                source_path=op.source_path,
                dest_path=op.dest_path,
                error=f"Failed to move file: {e}"
            )

        # Track for rollback
        rollback_stack.append(RollbackState(
            operation_index=index,
            operation_type=OperationType.RENAME,
            rename_source=source,
            rename_dest=dest,
            original_dest_content=original_dest_content
        ))

        return OperationResult(
            success=True,
            operation_index=index,
            action="rename",
            source_path=op.source_path,
            dest_path=op.dest_path
        )

    def _rollback(self, rollback_stack: List[RollbackState]) -> bool:
        """Rollback all completed operations in reverse order.

        Args:
            rollback_stack: Stack of rollback states (most recent last)

        Returns:
            True if all rollbacks succeeded, False if any failed
        """
        all_succeeded = True

        # Process in reverse order (LIFO)
        while rollback_stack:
            state = rollback_stack.pop()

            try:
                if state.operation_type == OperationType.EDIT:
                    # Restore original content
                    if state.original_path and state.original_content is not None:
                        state.original_path.write_bytes(state.original_content)
                        self._trace(f"rollback: restored {state.original_path}")

                elif state.operation_type == OperationType.CREATE:
                    # Remove created file
                    if state.created_path and state.created_path.exists():
                        state.created_path.unlink()
                        self._trace(f"rollback: removed {state.created_path}")

                elif state.operation_type == OperationType.DELETE:
                    # Restore deleted file
                    if state.original_path and state.original_content is not None:
                        state.original_path.parent.mkdir(parents=True, exist_ok=True)
                        state.original_path.write_bytes(state.original_content)
                        self._trace(f"rollback: restored deleted {state.original_path}")

                elif state.operation_type == OperationType.RENAME:
                    # Move file back
                    if state.rename_source and state.rename_dest:
                        if state.rename_dest.exists():
                            shutil.move(str(state.rename_dest), str(state.rename_source))
                            self._trace(f"rollback: moved {state.rename_dest} back to {state.rename_source}")
                        # Restore original dest content if there was one
                        if state.original_dest_content is not None:
                            state.rename_dest.parent.mkdir(parents=True, exist_ok=True)
                            state.rename_dest.write_bytes(state.original_dest_content)
                            self._trace(f"rollback: restored original {state.rename_dest}")

            except OSError as e:
                self._trace(f"rollback: failed for operation {state.operation_index}: {e}")
                all_succeeded = False
                # Continue trying to rollback remaining operations

        return all_succeeded


def _truncate_for_msg(text: str, max_len: int) -> str:
    """Truncate *text* for use in error/log messages."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def generate_multi_file_diff_preview(
    operations: List[Dict[str, Any]],
    resolve_path_fn: Callable[[str], Path],
    max_lines_per_file: int = 30
) -> Tuple[str, bool]:
    """Generate a preview diff for multiple file operations.

    For edit operations the actual file is read from disk and the targeted
    edit is applied so the diff shows the real before/after content.

    Args:
        operations: List of operation dictionaries
        resolve_path_fn: Function to resolve paths
        max_lines_per_file: Maximum diff lines per file

    Returns:
        Tuple of (preview_text, truncated)
    """
    import difflib

    lines = []
    truncated = False
    # Track working content for sequential edits on the same file
    working_content: Dict[str, str] = {}

    for i, op in enumerate(operations):
        action = op.get("action", "").lower()

        if action == "edit":
            path = op.get("path", "unknown")
            old_text = op.get("old", "")
            new_text = op.get("new", "")
            prologue = op.get("prologue")
            epilogue = op.get("epilogue")

            # Read file content and apply targeted edit for accurate diff
            resolved = resolve_path_fn(path)
            resolved_str = str(resolved)

            if resolved_str not in working_content and resolved.exists():
                try:
                    working_content[resolved_str] = resolved.read_text(encoding="utf-8")
                except OSError:
                    pass

            if resolved_str in working_content:
                before = working_content[resolved_str]
                try:
                    after = apply_edit(before, old_text, new_text, prologue, epilogue)
                    working_content[resolved_str] = after
                except (EditNotFoundError, AmbiguousEditError):
                    # Fall through to simple fragment diff below
                    before = old_text
                    after = new_text
            else:
                # File not readable; show fragment diff
                before = old_text
                after = new_text

            diff = list(difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}"
            ))

            if len(diff) > max_lines_per_file:
                diff = diff[:max_lines_per_file]
                diff.append(f"... ({len(diff)} more lines)\n")
                truncated = True

            display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
            lines.append(f"[{i+1}] EDIT: {display_path}\n")
            lines.extend(diff)
            lines.append("\n")

        elif action == "create":
            path = op.get("path", "unknown")
            content = op.get("content", "")
            content_lines = content.splitlines()

            display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
            lines.append(f"[{i+1}] CREATE: {display_path} ({len(content_lines)} lines)\n")
            preview_lines = content_lines[:max_lines_per_file]
            for line in preview_lines:
                lines.append(f"+{line}\n")
            if len(content_lines) > max_lines_per_file:
                lines.append(f"... ({len(content_lines) - max_lines_per_file} more lines)\n")
                truncated = True
            lines.append("\n")

        elif action == "delete":
            path = op.get("path", "unknown")
            display_path = ellipsize_path(path, DEFAULT_MAX_PATH_WIDTH)
            lines.append(f"[{i+1}] DELETE: {display_path}\n\n")

        elif action == "rename":
            source = op.get("from") or op.get("source_path", "unknown")
            dest = op.get("to") or op.get("destination_path", "unknown")
            display_pair = ellipsize_path_pair(source, dest, DEFAULT_MAX_PATH_WIDTH)
            lines.append(f"[{i+1}] RENAME: {display_pair}\n\n")

    return "".join(lines), truncated
