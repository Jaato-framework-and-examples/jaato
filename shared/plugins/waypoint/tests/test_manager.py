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
    """Create a mock BackupManager with state tracking."""
    mock = MagicMock()
    # Use a mutable container to track current_waypoint state
    state = {"current": INITIAL_WAYPOINT_ID, "has_pending": False}
    type(mock).current_waypoint = property(lambda self: state["current"])
    mock.set_current_waypoint.side_effect = lambda wp_id: state.update({"current": wp_id})
    mock.get_backups_by_waypoint.return_value = []
    mock.get_first_backup_per_file_by_waypoint.return_value = {}
    mock.restore_from_backup.return_value = True
    mock.has_pending_backups.return_value = state["has_pending"]
    mock.tag_pending_backups.return_value = 0
    # Store state for test access
    mock._test_state = state
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


class TestWaypointOwnership:
    """Test waypoint ownership model."""

    def test_default_owner_is_user(self, manager):
        """Test that default owner is 'user'."""
        wp = manager.create("test")
        assert wp.owner == "user"

    def test_create_model_owned(self, manager):
        """Test creating model-owned waypoint."""
        wp = manager.create("model checkpoint", owner="model")
        assert wp.owner == "model"

    def test_all_waypoints_get_w_prefix(self, manager):
        """Test all waypoints use w-prefix IDs regardless of owner."""
        user_wp = manager.create("user checkpoint", owner="user")
        model_wp = manager.create("model checkpoint", owner="model")
        assert user_wp.id == "w1"
        assert model_wp.id == "w2"

    def test_sequential_ids_regardless_of_owner(self, manager):
        """Test IDs are sequential regardless of owner."""
        user_wp1 = manager.create("user 1", owner="user")
        model_wp1 = manager.create("model 1", owner="model")
        user_wp2 = manager.create("user 2", owner="user")
        model_wp2 = manager.create("model 2", owner="model")

        assert user_wp1.id == "w1"
        assert model_wp1.id == "w2"
        assert user_wp2.id == "w3"
        assert model_wp2.id == "w4"

    def test_ownership_persisted(self, temp_dir, mock_backup_manager):
        """Test that ownership is persisted to storage."""
        storage_path = temp_dir / "waypoints.json"

        mgr1 = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )
        mgr1.create("user wp", owner="user")
        mgr1.create("model wp", owner="model")

        # Create new manager from same storage
        mgr2 = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )

        user_wp = mgr2.get("w1")
        model_wp = mgr2.get("w2")

        assert user_wp.owner == "user"
        assert model_wp.owner == "model"

    def test_get_info_includes_ownership(self, manager, mock_backup_manager):
        """Test get_info includes ownership field."""
        manager.create("model checkpoint", owner="model")

        mock_backup_manager.get_backups_by_waypoint.return_value = []

        info = manager.get_info("w1")

        assert info["owner"] == "model"

    def test_initial_waypoint_is_user_owned(self, manager):
        """Test that implicit w0 is user-owned."""
        wp = manager.get(INITIAL_WAYPOINT_ID)
        assert wp.owner == "user"

    def test_delete_model_waypoint(self, manager):
        """Test deleting model-owned waypoint."""
        manager.create("model wp", owner="model")
        assert manager.delete("w1") is True
        assert manager.get("w1") is None

    def test_id_reused_after_delete(self, manager):
        """Test that IDs are reused after deletion regardless of owner."""
        manager.create("first", owner="model")
        manager.delete("w1")
        wp = manager.create("second", owner="user")  # Different owner
        assert wp.id == "w1"

    def test_lowest_id_used_after_delete(self, manager):
        """Test that the lowest available ID is used."""
        manager.create("first", owner="model")
        manager.create("second", owner="user")
        manager.create("third", owner="model")

        manager.delete("w2")

        wp = manager.create("new", owner="user")
        assert wp.id == "w2"


class TestWaypointTree:
    """Test waypoint tree structure and navigation."""

    def test_parent_id_set_on_create(self, manager):
        """Test that parent_id is set to current waypoint on creation."""
        # First waypoint's parent should be w0
        wp1 = manager.create("first")
        assert wp1.parent_id == INITIAL_WAYPOINT_ID

        # Second waypoint's parent should be w1 (now current)
        wp2 = manager.create("second")
        assert wp2.parent_id == "w1"

    def test_w0_has_no_parent(self, manager):
        """Test that initial waypoint w0 has no parent."""
        w0 = manager.get(INITIAL_WAYPOINT_ID)
        assert w0.parent_id is None

    def test_get_ancestors(self, manager):
        """Test getting ancestors from waypoint to root."""
        manager.create("first")   # w1, parent=w0
        manager.create("second")  # w2, parent=w1
        manager.create("third")   # w3, parent=w2

        ancestors = manager.get_ancestors("w3")
        assert ancestors == ["w3", "w2", "w1", "w0"]

    def test_get_ancestors_of_root(self, manager):
        """Test getting ancestors of root returns just root."""
        ancestors = manager.get_ancestors(INITIAL_WAYPOINT_ID)
        assert ancestors == ["w0"]

    def test_get_children(self, manager):
        """Test getting direct children of a waypoint."""
        manager.create("first")   # w1, parent=w0
        manager.create("second")  # w2, parent=w1

        # w0 should have w1 as child
        children_w0 = manager.get_children(INITIAL_WAYPOINT_ID)
        assert children_w0 == ["w1"]

        # w1 should have w2 as child
        children_w1 = manager.get_children("w1")
        assert children_w1 == ["w2"]

        # w2 should have no children
        children_w2 = manager.get_children("w2")
        assert children_w2 == []

    def test_find_path_same_waypoint(self, manager):
        """Test finding path from waypoint to itself."""
        manager.create("first")
        path = manager.find_path("w1", "w1")
        assert path == ["w1"]

    def test_find_path_parent_to_child(self, manager):
        """Test finding path from parent to child."""
        manager.create("first")   # w1
        manager.create("second")  # w2

        path = manager.find_path("w0", "w2")
        assert path == ["w0", "w1", "w2"]

    def test_find_path_child_to_parent(self, manager):
        """Test finding path from child to parent."""
        manager.create("first")   # w1
        manager.create("second")  # w2

        path = manager.find_path("w2", "w0")
        assert path == ["w2", "w1", "w0"]

    def test_get_tree_structure(self, manager):
        """Test getting full tree structure."""
        manager.create("first")   # w1
        manager.create("second")  # w2

        tree = manager.get_tree_structure()

        assert tree["root"] == INITIAL_WAYPOINT_ID
        assert tree["current"] == "w2"
        assert "w0" in tree["nodes"]
        assert "w1" in tree["nodes"]
        assert "w2" in tree["nodes"]

        # Check node structure
        assert tree["nodes"]["w0"]["children"] == ["w1"]
        assert tree["nodes"]["w1"]["children"] == ["w2"]
        assert tree["nodes"]["w2"]["children"] == []

    def test_get_info_includes_tree_data(self, manager, mock_backup_manager):
        """Test that get_info includes parent_id and children."""
        manager.create("first")   # w1
        manager.create("second")  # w2

        mock_backup_manager.get_backups_by_waypoint.return_value = []

        info = manager.get_info("w1")

        assert info["parent_id"] == "w0"
        assert info["children"] == ["w2"]

    def test_parent_id_persisted(self, temp_dir, mock_backup_manager):
        """Test that parent_id is persisted to storage."""
        storage_path = temp_dir / "waypoints.json"

        mgr1 = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )
        mgr1.create("first")   # w1
        mgr1.create("second")  # w2

        # Create new manager from same storage
        mgr2 = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=storage_path,
        )

        w1 = mgr2.get("w1")
        w2 = mgr2.get("w2")

        assert w1.parent_id == "w0"
        assert w2.parent_id == "w1"


class TestWaypointRestoreTree:
    """Test restore behavior with tree structure."""

    def test_restore_updates_current_to_target(self, manager, mock_backup_manager):
        """Test that restore sets current waypoint to target."""
        manager.create("first")   # w1, current
        manager.create("second")  # w2, current

        # Restore to w1
        mock_backup_manager.get_first_backup_per_file_by_waypoint.return_value = {}
        manager.restore("w1")

        assert manager.current_waypoint == "w1"

    def test_restore_with_pending_creates_ceiling(self, temp_dir):
        """Test that restore auto-creates ceiling waypoint with pending edits."""
        # Create mock that reports pending backups
        mock = MagicMock()
        state = {"current": INITIAL_WAYPOINT_ID}
        type(mock).current_waypoint = property(lambda self: state["current"])
        mock.set_current_waypoint.side_effect = lambda wp_id: state.update({"current": wp_id})
        mock.get_backups_by_waypoint.return_value = []
        mock.get_first_backup_per_file_by_waypoint.return_value = {}
        mock.restore_from_backup.return_value = True
        mock.has_pending_backups.return_value = True  # Has uncommitted edits
        mock.tag_pending_backups.return_value = 1

        storage_path = temp_dir / "waypoints.json"
        manager = WaypointManager(
            backup_manager=mock,
            storage_path=storage_path,
        )

        manager.create("first")   # w1
        # Now current is w1, and there are "pending edits"

        # Restore to w0 - should auto-create ceiling at w2
        result = manager.restore("w0")

        assert result.success is True
        # Ceiling waypoint should have been created
        assert manager.get("w2") is not None
        ceiling = manager.get("w2")
        assert "auto-saved" in ceiling.description
        assert ceiling.parent_id == "w1"  # Parent was w1 (where we were before restore)

    def test_restore_without_pending_no_ceiling(self, manager, mock_backup_manager):
        """Test that restore without pending edits doesn't create ceiling."""
        manager.create("first")   # w1

        # Ensure no pending backups
        mock_backup_manager.has_pending_backups.return_value = False
        mock_backup_manager.get_first_backup_per_file_by_waypoint.return_value = {}

        # Restore to w0
        manager.restore("w0")

        # No ceiling should be created (still just w0 and w1)
        assert manager.get("w2") is None
        waypoints = manager.list()
        assert len(waypoints) == 2  # w0 and w1 only
