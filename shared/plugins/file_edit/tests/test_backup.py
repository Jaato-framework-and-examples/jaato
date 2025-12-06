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
        assert manager.base_dir == Path(".jaato/backups")

    def test_custom_base_dir(self, tmp_path):
        custom_dir = tmp_path / "custom_backups"
        manager = BackupManager(custom_dir)
        assert manager.base_dir == custom_dir

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
