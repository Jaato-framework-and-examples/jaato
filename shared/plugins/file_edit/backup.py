"""Backup management for file editing operations.

Provides automatic backup creation before file modifications and
restoration capabilities for undoing changes.

Features:
- Automatic backup creation before file modifications
- Configurable backup retention per file
- Session-based backup tracking for cleanup
- List all backups across all files
- Restore from any backup
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default number of backups to keep per file
DEFAULT_BACKUP_COUNT = 5

# Environment variable for configuring backup count
BACKUP_COUNT_ENV_VAR = "JAATO_FILE_BACKUP_COUNT"

# Default max session operations before auto-cleanup
DEFAULT_SESSION_MAX_OPS = 100


@dataclass
class BackupInfo:
    """Information about a single backup file."""
    backup_path: Path
    original_path: str  # The original file path this is a backup of
    timestamp: datetime
    size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "backup_path": str(self.backup_path),
            "original_path": self.original_path,
            "timestamp": self.timestamp.isoformat(),
            "size": self.size
        }


class BackupManager:
    """Manages file backups for the file_edit plugin.

    Backups are stored in .jaato/backups/ with the naming convention:
    {path_with_underscores}_{ISO_timestamp}.bak

    The number of backups kept per file is controlled by the
    JAATO_FILE_BACKUP_COUNT environment variable (default: 5).

    Session tracking:
    - Tracks backups created in current session
    - Auto-cleanup when session operation count exceeds threshold
    - Manual cleanup via cleanup_session()
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        session_max_ops: Optional[int] = None
    ):
        """Initialize the backup manager.

        Args:
            base_dir: Directory for storing backups. Defaults to .jaato/backups
                     resolved as an absolute path from the current working directory.
            session_max_ops: Max operations before auto-cleanup. Defaults to 100.
                            Set to 0 to disable auto-cleanup.
        """
        # Always resolve to absolute path to avoid issues with CWD changes
        if base_dir is not None:
            self._base_dir = Path(base_dir).resolve()
        else:
            self._base_dir = Path(".jaato/backups").resolve()
        self._max_backups = int(
            os.environ.get(BACKUP_COUNT_ENV_VAR, DEFAULT_BACKUP_COUNT)
        )

        # Session tracking
        self._session_max_ops = session_max_ops if session_max_ops is not None else DEFAULT_SESSION_MAX_OPS
        self._session_operation_count = 0
        self._session_backups: List[Path] = []  # Backups created in this session

    @property
    def base_dir(self) -> Path:
        """Get the backup directory path."""
        return self._base_dir

    @property
    def max_backups(self) -> int:
        """Get the maximum number of backups to keep per file."""
        return self._max_backups

    def _sanitize_path(self, file_path: Path) -> str:
        """Convert a file path to a safe backup filename prefix.

        Replaces path separators with underscores to create a flat
        backup namespace while preserving uniqueness.

        Args:
            file_path: Original file path

        Returns:
            Safe string for use in backup filename
        """
        # Resolve to absolute path for consistency
        resolved = file_path.resolve()
        # Replace path separators with underscores
        safe_name = str(resolved).replace("/", "_").replace("\\", "_")
        # Remove leading underscore if present (from absolute paths)
        if safe_name.startswith("_"):
            safe_name = safe_name[1:]
        return safe_name

    def _backup_filename(self, file_path: Path) -> str:
        """Generate a backup filename for the given file.

        Args:
            file_path: Original file path

        Returns:
            Backup filename with timestamp
        """
        safe_name = self._sanitize_path(file_path)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        return f"{safe_name}_{timestamp}.bak"

    def _get_backups_for_file(self, file_path: Path) -> List[Path]:
        """Get all backup files for a given file, sorted oldest to newest.

        Args:
            file_path: Original file path

        Returns:
            List of backup file paths, sorted by modification time (oldest first)
        """
        if not self._base_dir.exists():
            return []

        safe_prefix = self._sanitize_path(file_path) + "_"
        backups = [
            f for f in self._base_dir.glob("*.bak")
            if f.name.startswith(safe_prefix)
        ]
        return sorted(backups, key=lambda p: p.stat().st_mtime)

    def create_backup(self, file_path: Path) -> Optional[Path]:
        """Create a backup of the specified file.

        Also prunes old backups if the count exceeds max_backups.
        Tracks backup in session for potential cleanup.

        Args:
            file_path: Path to the file to backup

        Returns:
            Path to the created backup, or None if the file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        # Ensure backup directory exists
        self._base_dir.mkdir(parents=True, exist_ok=True)

        # Create backup
        backup_path = self._base_dir / self._backup_filename(file_path)
        backup_path.write_bytes(file_path.read_bytes())

        # Track in session
        self._session_backups.append(backup_path)
        self._session_operation_count += 1

        # Auto-cleanup if threshold exceeded
        if self._session_max_ops > 0 and self._session_operation_count >= self._session_max_ops:
            self._auto_cleanup_session()

        # Prune old backups
        self._prune_old_backups(file_path)

        return backup_path

    def _prune_old_backups(self, file_path: Path) -> int:
        """Remove old backups exceeding the maximum count.

        Args:
            file_path: Original file path

        Returns:
            Number of backups removed
        """
        backups = self._get_backups_for_file(file_path)
        removed = 0

        while len(backups) > self._max_backups:
            oldest = backups.pop(0)
            try:
                oldest.unlink()
                removed += 1
            except OSError:
                pass

        return removed

    def get_latest_backup(self, file_path: Path) -> Optional[Path]:
        """Get the most recent backup for a file.

        Args:
            file_path: Original file path

        Returns:
            Path to the most recent backup, or None if no backups exist
        """
        backups = self._get_backups_for_file(Path(file_path))
        return backups[-1] if backups else None

    def list_backups(self, file_path: Path) -> List[Path]:
        """List all backups for a file.

        Args:
            file_path: Original file path

        Returns:
            List of backup paths, sorted oldest to newest
        """
        return self._get_backups_for_file(Path(file_path))

    def restore_from_backup(
        self,
        file_path: Path,
        backup_path: Optional[Path] = None
    ) -> bool:
        """Restore a file from backup.

        If backup_path is not specified, restores from the most recent backup.

        Args:
            file_path: Original file path to restore to
            backup_path: Specific backup to restore from (optional)

        Returns:
            True if restoration was successful, False otherwise
        """
        file_path = Path(file_path)

        if backup_path is None:
            backup_path = self.get_latest_backup(file_path)

        if backup_path is None or not backup_path.exists():
            return False

        try:
            # Ensure parent directory exists (in case file was deleted)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(backup_path.read_bytes())
            return True
        except OSError:
            return False

    def has_backup(self, file_path: Path) -> bool:
        """Check if a file has any backups.

        Args:
            file_path: Original file path

        Returns:
            True if at least one backup exists
        """
        return len(self._get_backups_for_file(Path(file_path))) > 0

    def cleanup_all(self) -> int:
        """Remove all backup files.

        Returns:
            Number of backups removed
        """
        if not self._base_dir.exists():
            return 0

        removed = 0
        for backup in self._base_dir.glob("*.bak"):
            try:
                backup.unlink()
                removed += 1
            except OSError:
                pass

        # Reset session tracking
        self._session_backups = []
        self._session_operation_count = 0

        return removed

    def list_all_backups(self) -> List[BackupInfo]:
        """List all backup files across all original files.

        Returns:
            List of BackupInfo objects, sorted by timestamp (newest first)
        """
        if not self._base_dir.exists():
            return []

        backups: List[BackupInfo] = []

        for backup_path in self._base_dir.glob("*.bak"):
            try:
                stat = backup_path.stat()

                # Parse original path from backup filename
                # Format: {sanitized_path}_{timestamp}.bak
                name = backup_path.name[:-4]  # Remove .bak
                # Split at last underscore followed by timestamp pattern
                parts = name.rsplit("_", 1)
                if len(parts) == 2:
                    sanitized_path = parts[0]
                    timestamp_str = parts[1]

                    # Reconstruct approximate original path
                    # Note: This is an approximation since we can't fully reverse sanitization
                    original_path = "/" + sanitized_path.replace("_", "/")

                    # Parse timestamp
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
                    except ValueError:
                        timestamp = datetime.fromtimestamp(stat.st_mtime)
                else:
                    original_path = name
                    timestamp = datetime.fromtimestamp(stat.st_mtime)

                backups.append(BackupInfo(
                    backup_path=backup_path,
                    original_path=original_path,
                    timestamp=timestamp,
                    size=stat.st_size
                ))

            except OSError:
                continue

        # Sort by timestamp, newest first
        return sorted(backups, key=lambda b: b.timestamp, reverse=True)

    def get_backup_info(self, backup_path: Path) -> Optional[BackupInfo]:
        """Get information about a specific backup file.

        Args:
            backup_path: Path to the backup file

        Returns:
            BackupInfo if backup exists, None otherwise
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            return None

        try:
            stat = backup_path.stat()

            # Parse original path from backup filename
            name = backup_path.name[:-4] if backup_path.name.endswith(".bak") else backup_path.name
            parts = name.rsplit("_", 1)

            if len(parts) == 2:
                sanitized_path = parts[0]
                timestamp_str = parts[1]
                original_path = "/" + sanitized_path.replace("_", "/")

                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
                except ValueError:
                    timestamp = datetime.fromtimestamp(stat.st_mtime)
            else:
                original_path = name
                timestamp = datetime.fromtimestamp(stat.st_mtime)

            return BackupInfo(
                backup_path=backup_path,
                original_path=original_path,
                timestamp=timestamp,
                size=stat.st_size
            )

        except OSError:
            return None

    def cleanup_session(self) -> int:
        """Clean up backups created in the current session.

        This removes all backups created since the session started,
        but respects the max_backups setting (won't remove if doing
        so would leave fewer than max_backups for a file).

        Returns:
            Number of backups removed
        """
        removed = 0

        for backup_path in self._session_backups:
            if backup_path.exists():
                try:
                    backup_path.unlink()
                    removed += 1
                except OSError:
                    pass

        # Reset session tracking
        self._session_backups = []
        self._session_operation_count = 0

        return removed

    def _auto_cleanup_session(self) -> int:
        """Internal auto-cleanup when session threshold is exceeded.

        Removes the oldest half of session backups to make room for more.

        Returns:
            Number of backups removed
        """
        if not self._session_backups:
            return 0

        # Remove oldest half of session backups
        remove_count = len(self._session_backups) // 2
        to_remove = self._session_backups[:remove_count]
        self._session_backups = self._session_backups[remove_count:]

        removed = 0
        for backup_path in to_remove:
            if backup_path.exists():
                try:
                    backup_path.unlink()
                    removed += 1
                except OSError:
                    pass

        return removed

    def reset_session(self) -> None:
        """Reset session tracking without removing backups.

        Call this when starting a new logical session where you want
        to keep existing backups but start fresh tracking.
        """
        self._session_backups = []
        self._session_operation_count = 0

    @property
    def session_operation_count(self) -> int:
        """Get the number of operations in the current session."""
        return self._session_operation_count

    @property
    def session_backup_count(self) -> int:
        """Get the number of backups created in the current session."""
        return len(self._session_backups)
