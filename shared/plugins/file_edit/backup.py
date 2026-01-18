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

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default waypoint ID (session start)
DEFAULT_WAYPOINT_ID = "w0"


# Default number of backups to keep per file
DEFAULT_BACKUP_COUNT = 5

# Environment variable for configuring backup count
BACKUP_COUNT_ENV_VAR = "JAATO_FILE_BACKUP_COUNT"

# Default max session operations before auto-cleanup
DEFAULT_SESSION_MAX_OPS = 100


@dataclass
class BackupInfo:
    """Information about a single backup file.

    Backups are tagged with waypoint information for tree-based navigation:
    - diverged_from: The waypoint that was current when this backup was created.
      The backup contains the file state AT this waypoint (before the edit).
    - next_waypoint: The waypoint created after this backup (set retroactively
      when a waypoint is created). None if no waypoint has been created yet.
    """
    backup_path: Path
    original_path: str  # The original file path this is a backup of
    timestamp: datetime
    size: int
    diverged_from: str = "w0"  # Waypoint ID this backup diverged from
    next_waypoint: Optional[str] = None  # Waypoint created after this backup (set lazily)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "backup_path": str(self.backup_path),
            "original_path": self.original_path,
            "timestamp": self.timestamp.isoformat(),
            "size": self.size,
            "diverged_from": self.diverged_from,
            "next_waypoint": self.next_waypoint,
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

        # Waypoint tracking - backups are tagged with the waypoint they diverged from
        self._current_waypoint: str = DEFAULT_WAYPOINT_ID
        self._backup_metadata: Dict[str, BackupInfo] = {}  # backup_path -> BackupInfo
        self._load_metadata()

    @property
    def base_dir(self) -> Path:
        """Get the backup directory path."""
        return self._base_dir

    @property
    def max_backups(self) -> int:
        """Get the maximum number of backups to keep per file."""
        return self._max_backups

    @property
    def current_waypoint(self) -> str:
        """Get the current waypoint ID that new backups will be tagged with."""
        return self._current_waypoint

    @property
    def _metadata_path(self) -> Path:
        """Get the path to the metadata JSON file."""
        return self._base_dir / "_backup_metadata.json"

    def _load_metadata(self) -> None:
        """Load backup metadata from disk."""
        if not self._metadata_path.exists():
            return

        try:
            with open(self._metadata_path, 'r') as f:
                data = json.load(f)
                for backup_path_str, info_dict in data.items():
                    self._backup_metadata[backup_path_str] = BackupInfo(
                        backup_path=Path(info_dict["backup_path"]),
                        original_path=info_dict["original_path"],
                        timestamp=datetime.fromisoformat(info_dict["timestamp"]),
                        size=info_dict["size"],
                        diverged_from=info_dict.get("diverged_from", DEFAULT_WAYPOINT_ID),
                        next_waypoint=info_dict.get("next_waypoint"),  # None for pending
                    )
        except (json.JSONDecodeError, IOError, KeyError):
            # If metadata is corrupted, start fresh
            self._backup_metadata = {}

    def _save_metadata(self) -> None:
        """Save backup metadata to disk."""
        self._base_dir.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                str(bp): info.to_dict()
                for bp, info in self._backup_metadata.items()
            }
            with open(self._metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass

    def set_current_waypoint(self, waypoint_id: str) -> None:
        """Set the current waypoint ID for tagging new backups.

        All backups created after this call will be tagged as having
        diverged from this waypoint. This is called by WaypointManager
        when a new waypoint is created.

        Args:
            waypoint_id: The waypoint ID (e.g., "w0", "w1", "w2").
        """
        self._current_waypoint = waypoint_id

    def get_backups_by_waypoint(self, waypoint_id: str) -> List[BackupInfo]:
        """Get all backups that diverged from a specific waypoint.

        These backups contain the file state that existed AT the waypoint,
        before edits were made that diverged from it.

        Args:
            waypoint_id: The waypoint ID to query (e.g., "w0", "w1").

        Returns:
            List of BackupInfo objects for backups that diverged from this waypoint.
        """
        return [
            info for info in self._backup_metadata.values()
            if info.diverged_from == waypoint_id
        ]

    def get_first_backup_per_file_by_waypoint(
        self,
        waypoint_id: str
    ) -> Dict[str, BackupInfo]:
        """Get the first backup for each file that diverged from a waypoint.

        When restoring to a waypoint, we need the FIRST backup of each file
        that was created after the waypoint. This backup contains the file's
        state AT the waypoint (before any divergent edits).

        Args:
            waypoint_id: The waypoint ID to query.

        Returns:
            Dict mapping original file paths to their first backup after the waypoint.
        """
        backups = self.get_backups_by_waypoint(waypoint_id)

        # Sort by timestamp to get earliest first
        backups.sort(key=lambda b: b.timestamp)

        # Keep only the first backup per file
        first_per_file: Dict[str, BackupInfo] = {}
        for backup in backups:
            if backup.original_path not in first_per_file:
                first_per_file[backup.original_path] = backup

        return first_per_file

    def get_pending_backups(self) -> List[BackupInfo]:
        """Get backups that haven't been assigned to a waypoint yet.

        These are backups where next_waypoint is None, meaning no waypoint
        has been created since these edits were made. They diverge from
        the current waypoint.

        Returns:
            List of BackupInfo objects with next_waypoint=None.
        """
        return [
            info for info in self._backup_metadata.values()
            if info.next_waypoint is None and info.diverged_from == self._current_waypoint
        ]

    def tag_pending_backups(self, waypoint_id: str) -> int:
        """Tag all pending backups with the given waypoint ID.

        Called when a new waypoint is created to "close" the pending backups,
        associating them with the waypoint they lead to.

        Args:
            waypoint_id: The new waypoint ID to tag backups with.

        Returns:
            Number of backups tagged.
        """
        tagged = 0
        for info in self._backup_metadata.values():
            if info.next_waypoint is None and info.diverged_from == self._current_waypoint:
                info.next_waypoint = waypoint_id
                tagged += 1

        if tagged > 0:
            self._save_metadata()

        return tagged

    def get_backups_by_next_waypoint(self, waypoint_id: str) -> List[BackupInfo]:
        """Get all backups that lead to a specific waypoint.

        These are backups whose edits were "closed" by the creation of the
        given waypoint. Used for tree-based restoration.

        Args:
            waypoint_id: The waypoint ID to query.

        Returns:
            List of BackupInfo objects where next_waypoint matches.
        """
        return [
            info for info in self._backup_metadata.values()
            if info.next_waypoint == waypoint_id
        ]

    def get_first_backup_per_file_by_next_waypoint(
        self,
        waypoint_id: str
    ) -> Dict[str, BackupInfo]:
        """Get the first backup for each file that leads to a waypoint.

        For tree-based restoration, we need backups that were created as part
        of the journey TO a waypoint. The first backup per file contains the
        file state before the edits that led to that waypoint.

        Args:
            waypoint_id: The waypoint ID to query.

        Returns:
            Dict mapping original file paths to their first backup leading to waypoint.
        """
        backups = self.get_backups_by_next_waypoint(waypoint_id)

        # Sort by timestamp to get earliest first
        backups.sort(key=lambda b: b.timestamp)

        # Keep only the first backup per file
        first_per_file: Dict[str, BackupInfo] = {}
        for backup in backups:
            if backup.original_path not in first_per_file:
                first_per_file[backup.original_path] = backup

        return first_per_file

    def has_pending_backups(self) -> bool:
        """Check if there are uncommitted edits (pending backups).

        Returns:
            True if there are backups with next_waypoint=None for current waypoint.
        """
        return len(self.get_pending_backups()) > 0

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

        # Record metadata with waypoint tagging
        backup_info = BackupInfo(
            backup_path=backup_path,
            original_path=str(file_path.resolve()),
            timestamp=datetime.now(),
            size=backup_path.stat().st_size,
            diverged_from=self._current_waypoint,
        )
        self._backup_metadata[str(backup_path)] = backup_info
        self._save_metadata()

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
                # Also remove from metadata
                self._backup_metadata.pop(str(oldest), None)
                removed += 1
            except OSError:
                pass

        if removed > 0:
            self._save_metadata()

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

        # Clear all metadata
        self._backup_metadata = {}
        self._save_metadata()

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
                    # Also remove from metadata
                    self._backup_metadata.pop(str(backup_path), None)
                    removed += 1
                except OSError:
                    pass

        # Reset session tracking
        self._session_backups = []
        self._session_operation_count = 0

        if removed > 0:
            self._save_metadata()

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
                    # Also remove from metadata
                    self._backup_metadata.pop(str(backup_path), None)
                    removed += 1
                except OSError:
                    pass

        if removed > 0:
            self._save_metadata()

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
