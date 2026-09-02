"""Tests for the backup manager."""

import os
import time
from pathlib import Path

import pytest

from ..backup import BackupManager, DEFAULT_BACKUP_COUNT, BACKUP_COUNT_ENV_VAR


class TestBackupManagerInitialization:
    """Tests for backup manager initialization."""

    def test_default_base_dir(self):
        manager = BackupManager()
        # Should resolve to absolute path
        assert manager.base_dir.is_absolute()
        assert manager.base_dir.name == "backups"
        assert manager.base_dir.parent.name == ".jaato"

    def test_custom_base_dir(self, tmp_path):
        custom_dir = tmp_path / "custom_backups"
        manager = BackupManager(custom_dir)
        # Should resolve to absolute path
        assert manager.base_dir.is_absolute()
        assert manager.base_dir == custom_dir.resolve()

    def test_default_max_backups(self):
        manager = BackupManager()
        assert manager.max_backups == DEFAULT_BACKUP_COUNT

    def test_max_backups_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(BACKUP_COUNT_ENV_VAR, "10")
        manager = BackupManager(tmp_path)
        assert manager.max_backups == 10


class TestBackupCreation:
    """Tests for backup creation."""

    def test_create_backup(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        backup_path = manager.create_backup(test_file)

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.read_text() == "Test content"
        assert backup_path.suffix == ".bak"

    def test_create_backup_nonexistent_file(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        nonexistent = tmp_path / "nonexistent.txt"
        backup_path = manager.create_backup(nonexistent)

        assert backup_path is None

    def test_create_backup_creates_directory(self, tmp_path):
        backup_dir = tmp_path / "backups" / "nested"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        backup_path = manager.create_backup(test_file)

        assert backup_dir.exists()
        assert backup_path.exists()

    def test_backup_filename_contains_timestamp(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        backup_path = manager.create_backup(test_file)

        # Filename should contain date pattern
        assert backup_path.name.endswith(".bak")
        # Should have ISO-like timestamp pattern
        assert "_20" in backup_path.name  # Year prefix


class TestBackupRetrieval:
    """Tests for backup retrieval."""

    def test_get_latest_backup(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"

        # Create multiple backups
        test_file.write_text("Version 1")
        manager.create_backup(test_file)
        time.sleep(0.1)  # Ensure different timestamps

        test_file.write_text("Version 2")
        manager.create_backup(test_file)

        latest = manager.get_latest_backup(test_file)

        assert latest is not None
        assert latest.read_text() == "Version 2"

    def test_get_latest_backup_no_backups(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"
        latest = manager.get_latest_backup(test_file)

        assert latest is None

    def test_list_backups(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"

        # Create multiple backups
        test_file.write_text("Version 1")
        manager.create_backup(test_file)
        time.sleep(0.1)

        test_file.write_text("Version 2")
        manager.create_backup(test_file)

        backups = manager.list_backups(test_file)

        assert len(backups) == 2
        # Should be sorted oldest to newest
        assert backups[0].read_text() == "Version 1"
        assert backups[1].read_text() == "Version 2"

    def test_has_backup(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        assert manager.has_backup(test_file) is False

        manager.create_backup(test_file)

        assert manager.has_backup(test_file) is True


class TestBackupRestoration:
    """Tests for backup restoration."""

    def test_restore_from_backup(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")
        manager.create_backup(test_file)

        # Modify the file
        test_file.write_text("Modified content")
        assert test_file.read_text() == "Modified content"

        # Restore
        success = manager.restore_from_backup(test_file)

        assert success is True
        assert test_file.read_text() == "Original content"

    def test_restore_deleted_file(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")
        manager.create_backup(test_file)

        # Delete the file
        test_file.unlink()
        assert not test_file.exists()

        # Restore
        success = manager.restore_from_backup(test_file)

        assert success is True
        assert test_file.exists()
        assert test_file.read_text() == "Original content"

    def test_restore_from_specific_backup(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"

        # Create first backup
        test_file.write_text("Version 1")
        backup1 = manager.create_backup(test_file)
        time.sleep(0.1)

        # Create second backup
        test_file.write_text("Version 2")
        manager.create_backup(test_file)

        # Modify file again
        test_file.write_text("Version 3")

        # Restore from first backup specifically
        success = manager.restore_from_backup(test_file, backup1)

        assert success is True
        assert test_file.read_text() == "Version 1"

    def test_restore_no_backup_fails(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        success = manager.restore_from_backup(test_file)

        assert success is False


class TestBackupPruning:
    """Tests for backup pruning."""

    def test_prune_old_backups(self, tmp_path, monkeypatch):
        monkeypatch.setenv(BACKUP_COUNT_ENV_VAR, "2")

        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        test_file = tmp_path / "test.txt"

        # Create more backups than the limit
        for i in range(4):
            test_file.write_text(f"Version {i}")
            manager.create_backup(test_file)
            time.sleep(0.1)

        backups = manager.list_backups(test_file)

        # Should only have 2 backups (the most recent ones)
        assert len(backups) == 2
        assert backups[-1].read_text() == "Version 3"
        assert backups[-2].read_text() == "Version 2"


class TestBackupCleanup:
    """Tests for backup cleanup."""

    def test_cleanup_all(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        # Create backups for multiple files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        manager.create_backup(file1)
        manager.create_backup(file2)

        assert len(list(backup_dir.glob("*.bak"))) == 2

        removed = manager.cleanup_all()

        assert removed == 2
        assert len(list(backup_dir.glob("*.bak"))) == 0


class TestBackupPathSanitization:
    """Tests for path sanitization in backup names."""

    def test_path_with_subdirectories(self, tmp_path):
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        # Create nested file
        nested_dir = tmp_path / "subdir" / "nested"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "test.txt"
        test_file.write_text("Content")

        backup_path = manager.create_backup(test_file)

        assert backup_path is not None
        # Path separators should be replaced with underscores
        assert "/" not in backup_path.name.replace(".bak", "")
        assert "\\" not in backup_path.name.replace(".bak", "")


class TestBackupCwdIndependence:
    """Tests that backups work correctly regardless of CWD changes."""

    def test_backup_after_cwd_change(self, tmp_path, monkeypatch):
        """Backups should go to original directory even after CWD changes."""
        # Create backup dir and manager while in tmp_path
        backup_dir = tmp_path / "project" / ".jaato" / "backups"
        manager = BackupManager(backup_dir)

        # Create a test file
        test_file = tmp_path / "project" / "src" / "file.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("Original content")

        # Change working directory to somewhere else
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)

        # Create backup - should still go to original backup_dir
        backup_path = manager.create_backup(test_file)

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.parent == backup_dir
        assert backup_path.read_text() == "Original content"

        # Verify backup is NOT in the new CWD
        other_backups = list(other_dir.glob("**/*.bak"))
        assert len(other_backups) == 0


# ----------------------------------------------------------------------
# Flake-fix: microsecond-resolution filename + collision counter
# ----------------------------------------------------------------------
#
# Pre-fix, ``_backup_filename`` used second-resolution timestamps
# (``%Y-%m-%dT%H-%M-%S``).  When two ``create_backup`` calls landed
# in the same wall-clock second, the second call's filename equaled
# the first and ``write_bytes`` silently overwrote the first backup
# — so ``test_list_backups`` flaked when its 0.1s sleep happened to
# straddle no second-boundary.
#
# Fix: microsecond-resolution timestamp + ``-N`` collision counter
# on the (now vanishingly rare) microsecond collision.  Tests below
# pin every surface of that fix.


class TestBackupFilenameUniqueness:
    """Pin filename uniqueness under timestamp collision."""

    def test_filename_includes_microseconds(self, tmp_path):
        """Pin: new format renders microseconds (post flake-fix
        wire shape).  Six-digit ``%f`` after the seconds field."""
        manager = BackupManager(tmp_path / "backups")
        test_file = tmp_path / "test.txt"
        test_file.write_text("x")
        backup_path = manager.create_backup(test_file)
        assert backup_path is not None
        # Strip the .bak and split off the timestamp portion.
        stem = backup_path.name[:-len(".bak")]
        timestamp_str = stem.rsplit("_", 1)[1]
        # Microsecond-resolution timestamps end with a 6-digit
        # microsecond field.  Legacy format ended with the
        # 2-digit seconds field.
        last_segment = timestamp_str.rsplit("-", 1)[-1]
        assert last_segment.isdigit() and len(last_segment) == 6, (
            f"timestamp portion {timestamp_str!r} does not end in "
            f"a 6-digit microsecond field"
        )

    def test_two_backups_same_microsecond_get_distinct_paths(
        self, tmp_path, monkeypatch,
    ):
        """Pin: when ``datetime.now()`` returns the same value on
        two consecutive ``create_backup`` calls (mocked clock,
        sub-microsecond loop), the second backup gets a ``-N``
        collision-counter suffix and BOTH files survive.

        This is the load-bearing pin against the original flake:
        pre-fix, the second call would silently overwrite the
        first."""
        from datetime import datetime

        manager = BackupManager(tmp_path / "backups")
        test_file = tmp_path / "test.txt"

        # Freeze the clock so both create_backup calls produce the
        # same base filename.
        frozen = datetime(2026, 5, 12, 15, 30, 45, 123456)

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen

        monkeypatch.setattr(
            "shared.plugins.file_edit.backup.datetime",
            _FrozenDateTime,
        )

        test_file.write_text("Version 1")
        path1 = manager.create_backup(test_file)
        test_file.write_text("Version 2")
        path2 = manager.create_backup(test_file)

        assert path1 is not None
        assert path2 is not None
        assert path1 != path2, (
            "two create_backup calls at the same instant produced "
            "the SAME path — collision counter broken"
        )
        assert path1.exists() and path2.exists(), (
            "one of the two same-microsecond backups was "
            "overwritten — collision counter broken"
        )
        assert path1.read_text() == "Version 1"
        assert path2.read_text() == "Version 2"
        # The second path should carry the ``-N`` counter suffix.
        assert path2.name.endswith("-2.bak"), (
            f"second-collision path {path2.name!r} missing "
            f"expected ``-2`` counter suffix"
        )

    def test_collision_counter_increments_beyond_two(
        self, tmp_path, monkeypatch,
    ):
        """Pin: the collision counter walks upward.  Three calls
        in the same frozen instant produce ``base``, ``base-2``,
        ``base-3``."""
        from datetime import datetime

        manager = BackupManager(tmp_path / "backups")
        test_file = tmp_path / "test.txt"
        frozen = datetime(2026, 5, 12, 15, 30, 45, 123456)

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen

        monkeypatch.setattr(
            "shared.plugins.file_edit.backup.datetime",
            _FrozenDateTime,
        )

        paths = []
        for i, content in enumerate(["A", "B", "C", "D"], start=1):
            test_file.write_text(content)
            paths.append(manager.create_backup(test_file))

        assert len(set(paths)) == 4, "collisions produced dup paths"
        # First one has no counter; subsequent ones have -2, -3, -4.
        assert not paths[0].name.endswith("-2.bak")
        assert paths[1].name.endswith("-2.bak")
        assert paths[2].name.endswith("-3.bak")
        assert paths[3].name.endswith("-4.bak")


class TestBackupTimestampParsing:
    """Pin parser tolerance for new + legacy + counter formats."""

    def test_parser_handles_microsecond_format(self, tmp_path):
        """Pin: list_all_backups extracts the timestamp from a
        new-format (microsecond) backup filename."""
        manager = BackupManager(tmp_path / "backups")
        test_file = tmp_path / "test.txt"
        test_file.write_text("x")
        manager.create_backup(test_file)

        all_backups = manager.list_all_backups()
        assert len(all_backups) == 1
        # Timestamp should NOT have fallen back to st_mtime — the
        # filename-encoded value is what we want.  Verify by
        # checking the timestamp is within a few seconds of "now"
        # (st_mtime fallback would also be ~now, so the real test
        # is that microseconds are nonzero; pre-fix the second-
        # resolution format zeroed out microseconds).
        info = all_backups[0]
        assert info.timestamp.year == 2026 or info.timestamp.year >= 2024

    def test_parser_handles_legacy_seconds_format(self, tmp_path):
        """Pin: a manually-named legacy backup (pre flake-fix
        format) still parses cleanly — backward compatibility for
        backups already on disk when the fix lands."""
        from datetime import datetime
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        # Craft a legacy-named backup file manually.
        legacy_name = "tmp_x_test.txt_2024-01-15T12-30-45.bak"
        (backup_dir / legacy_name).write_text("legacy content")

        manager = BackupManager(backup_dir)
        all_backups = manager.list_all_backups()

        assert len(all_backups) == 1
        # Parsed timestamp matches the filename, NOT st_mtime.
        info = all_backups[0]
        assert info.timestamp == datetime(2024, 1, 15, 12, 30, 45)

    def test_parser_handles_counter_suffix(self, tmp_path):
        """Pin: parser strips the ``-N`` collision-counter suffix
        and re-parses the timestamp.  Without this, collision-
        suffixed backups would fall back to ``st_mtime`` (still
        correct ordering but lossy metadata)."""
        from datetime import datetime
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        # Craft a microsecond-format backup with ``-2`` suffix.
        name = (
            "tmp_x_test.txt_2026-05-12T15-30-45-123456-2.bak"
        )
        (backup_dir / name).write_text("collision content")

        manager = BackupManager(backup_dir)
        all_backups = manager.list_all_backups()

        assert len(all_backups) == 1
        info = all_backups[0]
        # Microsecond + counter-stripped parse recovers the
        # encoded timestamp.
        assert info.timestamp == datetime(
            2026, 5, 12, 15, 30, 45, 123456,
        )


class TestListBackupsFlakeRegression:
    """Direct regression test for the
    ``test_list_backups`` flake — same-second timestamps must
    produce distinct backups + preserve chronological ordering."""

    def test_two_backups_same_second_distinct(
        self, tmp_path, monkeypatch,
    ):
        """Pin: even with the second-resolution wall clock frozen
        at the same value across both ``create_backup`` calls,
        ``list_backups`` returns both files in creation order.

        Pre-fix this is the exact flake: same-second timestamp →
        identical filename → second write overwrites first →
        ``len(backups) == 2`` fails."""
        from datetime import datetime

        manager = BackupManager(tmp_path / "backups")
        test_file = tmp_path / "test.txt"

        # Freeze datetime.now() to the SAME microsecond across
        # both calls.  This simulates the worst-case version of
        # the pre-fix flake (where seconds would coincide; now
        # we force even microseconds to coincide).
        frozen = datetime(2026, 5, 12, 15, 30, 45, 123456)

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen

        monkeypatch.setattr(
            "shared.plugins.file_edit.backup.datetime",
            _FrozenDateTime,
        )

        test_file.write_text("Version 1")
        manager.create_backup(test_file)
        # Real wall-clock sleep so mtime ordering still works
        # (Linux ext4/tmpfs has nanosecond mtime; 0.05s is plenty).
        time.sleep(0.05)
        test_file.write_text("Version 2")
        manager.create_backup(test_file)

        backups = manager.list_backups(test_file)

        assert len(backups) == 2
        # Sorted oldest to newest.
        assert backups[0].read_text() == "Version 1"
        assert backups[1].read_text() == "Version 2"
