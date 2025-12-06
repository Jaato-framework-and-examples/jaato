"""Backup management for file editing operations.

Provides automatic backup creation before file modifications and
restoration capabilities for undoing changes.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# Default number of backups to keep per file
DEFAULT_BACKUP_COUNT = 5

# Environment variable for configuring backup count
BACKUP_COUNT_ENV_VAR = "JAATO_FILE_BACKUP_COUNT"


class BackupManager:
    """Manages file backups for the file_edit plugin.

    Backups are stored in .jaato/backups/ with the naming convention:
    {path_with_underscores}_{ISO_timestamp}.bak

    The number of backups kept per file is controlled by the
    JAATO_FILE_BACKUP_COUNT environment variable (default: 5).
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the backup manager.

        Args:
            base_dir: Directory for storing backups. Defaults to .jaato/backups
        """
        self._base_dir = base_dir or Path(".jaato/backups")
        self._max_backups = int(
            os.environ.get(BACKUP_COUNT_ENV_VAR, DEFAULT_BACKUP_COUNT)
        )

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

        return removed
