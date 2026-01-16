"""Tests for WaypointManager."""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from ..models import Waypoint, RestoreResult, INITIAL_WAYPOINT_ID
from ..manager import WaypointManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_backup_manager():
    """Create a mock BackupManager."""
    mock = MagicMock()
    mock.current_waypoint = INITIAL_WAYPOINT_ID
    mock.get_backups_by_waypoint.return_value = []
    mock.get_first_backup_per_file_by_waypoint.return_value = {}
    mock.restore_from_backup.return_value = True
    return mock


@pytest.fixture
def manager(temp_dir, mock_backup_manager):
    """Create a WaypointManager for testing."""
    storage_path = temp_dir / "waypoints.json"
    mgr = WaypointManager(
        backup_manager=mock_backup_manager,
        storage_path=storage_path,
    )
    return mgr


class TestWaypointManagerInitialization:
    """Test WaypointManager initialization."""

    def test_initial_waypoint_created(self, manager):
        """Test that implicit w0 waypoint is created on init."""
        waypoints = manager.list()
        assert len(waypoints) == 1
        assert waypoints[0].id == INITIAL_WAYPOINT_ID
        assert waypoints[0].is_implicit is True

    def test_current_waypoint_is_w0(self, manager):
        """Test that current waypoint starts at w0."""
        assert manager.current_waypoint == INITIAL_WAYPOINT_ID

    def test_storage_path_created(self, temp_dir, mock_backup_manager):
        """Test that storage directory is created."""
        storage_path = temp_dir / "subdir" / "waypoints.json"
        mgr = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )
        mgr.create("test waypoint")
        assert storage_path.exists()


class TestWaypointCreate:
    """Test waypoint creation."""

    def test_create_waypoint(self, manager, mock_backup_manager):
        """Test creating a new waypoint."""
        wp = manager.create("first checkpoint")

        assert wp.id == "w1"
        assert wp.description == "first checkpoint"
        assert wp.is_implicit is False
        mock_backup_manager.set_current_waypoint.assert_called_with("w1")

    def test_create_multiple_waypoints(self, manager):
        """Test creating multiple waypoints with sequential IDs."""
        wp1 = manager.create("first")
        wp2 = manager.create("second")
        wp3 = manager.create("third")

        assert wp1.id == "w1"
        assert wp2.id == "w2"
        assert wp3.id == "w3"

    def test_create_with_turn_index(self, manager):
        """Test creating waypoint with turn index."""
        wp = manager.create("test", turn_index=5)
        assert wp.turn_index == 5

    def test_create_with_user_message_preview(self, manager):
        """Test creating waypoint with user message preview."""
        wp = manager.create("test", user_message_preview="Fix the bug...")
        assert wp.user_message_preview == "Fix the bug..."


class TestWaypointList:
    """Test waypoint listing."""

    def test_list_includes_implicit(self, manager):
        """Test listing includes implicit w0."""
        manager.create("first")
        waypoints = manager.list(include_implicit=True)

        assert len(waypoints) == 2
        assert waypoints[0].id == INITIAL_WAYPOINT_ID

    def test_list_excludes_implicit(self, manager):
        """Test listing can exclude implicit w0."""
        manager.create("first")
        waypoints = manager.list(include_implicit=False)

        assert len(waypoints) == 1
        assert waypoints[0].id == "w1"

    def test_list_sorted_by_time(self, manager):
        """Test waypoints are sorted by creation time."""
        manager.create("first")
        manager.create("second")
        manager.create("third")

        waypoints = manager.list()
        ids = [wp.id for wp in waypoints]
        assert ids == [INITIAL_WAYPOINT_ID, "w1", "w2", "w3"]


class TestWaypointDelete:
    """Test waypoint deletion."""

    def test_delete_user_waypoint(self, manager):
        """Test deleting a user-created waypoint."""
        manager.create("to delete")
        assert manager.delete("w1") is True
        assert manager.get("w1") is None

    def test_cannot_delete_implicit(self, manager):
        """Test cannot delete implicit w0."""
        assert manager.delete(INITIAL_WAYPOINT_ID) is False
        assert manager.get(INITIAL_WAYPOINT_ID) is not None

    def test_delete_nonexistent(self, manager):
        """Test deleting nonexistent waypoint returns False."""
        assert manager.delete("w999") is False

    def test_delete_all(self, manager, mock_backup_manager):
        """Test deleting all user waypoints."""
        manager.create("first")
        manager.create("second")
        manager.create("third")

        count = manager.delete_all()

        assert count == 3
        waypoints = manager.list()
        assert len(waypoints) == 1
        assert waypoints[0].id == INITIAL_WAYPOINT_ID
        mock_backup_manager.set_current_waypoint.assert_called_with(INITIAL_WAYPOINT_ID)

    def test_id_reused_after_delete(self, manager):
        """Test that IDs are reused after deletion."""
        # Create w1
        wp1 = manager.create("first")
        assert wp1.id == "w1"

        # Delete w1
        manager.delete("w1")

        # Next create should reuse w1
        wp1_again = manager.create("first again")
        assert wp1_again.id == "w1"

    def test_lowest_available_id_used(self, manager):
        """Test that the lowest available ID is always used."""
        # Create w1, w2, w3
        manager.create("first")
        manager.create("second")
        manager.create("third")

        # Delete w2 (middle one)
        manager.delete("w2")

        # Next create should be w2 (lowest available)
        wp = manager.create("new")
        assert wp.id == "w2"


class TestWaypointRestore:
    """Test waypoint restoration."""

    def test_restore_nonexistent(self, manager):
        """Test restoring nonexistent waypoint fails."""
        result = manager.restore("w999")
        assert result.success is False
        assert "not found" in result.error

    def test_restore_files(self, manager, mock_backup_manager):
        """Test restoring files to waypoint state."""
        manager.create("checkpoint")

        # Setup mock backups
        mock_backup_manager.get_first_backup_per_file_by_waypoint.return_value = {
            "/path/to/file.py": MagicMock(backup_path=Path("/backup/file.py"))
        }

        result = manager.restore("w1")

        assert result.success is True
        assert len(result.files_restored) == 1

    def test_restore_updates_current_waypoint(self, manager, mock_backup_manager):
        """Test that restore updates current waypoint."""
        manager.create("checkpoint")
        manager.restore("w1")

        mock_backup_manager.set_current_waypoint.assert_called_with("w1")


class TestWaypointPersistence:
    """Test waypoint persistence."""

    def test_waypoints_persisted(self, temp_dir, mock_backup_manager):
        """Test waypoints are saved to disk."""
        storage_path = temp_dir / "waypoints.json"

        mgr1 = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )
        mgr1.create("test waypoint")

        # Create new manager from same storage
        mgr2 = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )

        waypoints = mgr2.list()
        assert len(waypoints) == 2  # w0 + w1
        assert waypoints[1].description == "test waypoint"

    def test_next_id_persisted(self, temp_dir, mock_backup_manager):
        """Test next ID counter is persisted."""
        storage_path = temp_dir / "waypoints.json"

        mgr1 = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )
        mgr1.create("first")
        mgr1.create("second")

        mgr2 = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )
        wp = mgr2.create("third")

        assert wp.id == "w3"


class TestWaypointInfo:
    """Test waypoint info retrieval."""

    def test_get_info(self, manager, mock_backup_manager):
        """Test getting waypoint info."""
        manager.create("test checkpoint")

        mock_backup_manager.get_backups_by_waypoint.return_value = [
            MagicMock(original_path="/path/to/file1.py"),
            MagicMock(original_path="/path/to/file2.py"),
            MagicMock(original_path="/path/to/file1.py"),  # Same file, multiple backups
        ]

        info = manager.get_info("w1")

        assert info is not None
        assert info["id"] == "w1"
        assert info["description"] == "test checkpoint"
        assert len(info["files_changed_since"]) == 2  # Unique files
        assert info["total_backups_since"] == 3

    def test_get_info_nonexistent(self, manager):
        """Test getting info for nonexistent waypoint."""
        info = manager.get_info("w999")
        assert info is None


class TestSessionScoping:
    """Test session-scoped waypoint storage."""

    def test_session_id_creates_isolated_storage(self, temp_dir, mock_backup_manager):
        """Test that different session_ids use different storage paths."""
        # Create manager for session A
        mgr_a = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=temp_dir / "sessions" / "session-a" / "waypoints.json",
            session_id="session-a",
        )
        mgr_a.create("session A waypoint")

        # Create manager for session B
        mgr_b = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=temp_dir / "sessions" / "session-b" / "waypoints.json",
            session_id="session-b",
        )
        mgr_b.create("session B waypoint")

        # Each session should only see its own waypoints (plus w0)
        assert len(mgr_a.list()) == 2  # w0 + w1
        assert len(mgr_b.list()) == 2  # w0 + w1

        # Verify content is different
        assert mgr_a.get("w1").description == "session A waypoint"
        assert mgr_b.get("w1").description == "session B waypoint"

    def test_default_session_storage_path(self, temp_dir, mock_backup_manager):
        """Test that session_id generates correct default storage path."""
        # When session_id is provided without explicit storage_path,
        # the manager should use .jaato/sessions/{session_id}/waypoints.json
        mgr = WaypointManager(
            backup_manager=mock_backup_manager,
            session_id="test-session-123",
        )

        # The storage path should include the session ID
        assert "sessions" in str(mgr._storage_path)
        assert "test-session-123" in str(mgr._storage_path)
