"""Tests for waypoint capture of session-attached state.

Waypoints carry a JSON snapshot of session-attached state so a
fork-from-waypoint primitive can replay extension-owned state
(premium pseudonymization lookup table, audit chain head, etc.)
across the fork.

The snapshot is captured at waypoint create time via the
``set_session_state_callback`` registered on
:class:`WaypointManager`, which delegates to
:meth:`JaatoSession.get_all_session_state` (so registered providers
are invoked and the snapshot is live).

See ``project_backlog_session_attached_state.md`` for the full
design.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ..manager import WaypointManager
from ..models import INITIAL_WAYPOINT_ID, Waypoint


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_backup_manager():
    mock = MagicMock()
    state = {"current": INITIAL_WAYPOINT_ID}
    type(mock).current_waypoint = property(lambda self: state["current"])
    mock.set_current_waypoint.side_effect = lambda wp_id: state.update({"current": wp_id})
    mock.tag_pending_backups.return_value = 0
    return mock


@pytest.fixture
def manager(temp_dir, mock_backup_manager):
    return WaypointManager(
        backup_manager=mock_backup_manager,
        storage_path=temp_dir / "waypoints.json",
    )


class TestSessionStateSnapshot:

    def test_no_callback_means_no_snapshot(self, manager):
        """Without set_session_state_callback wiring, waypoints are
        created with session_state_snapshot=None — backward-compatible
        for callers that haven't been updated."""
        wp = manager.create("checkpoint")
        assert wp.session_state_snapshot is None

    def test_callback_invoked_at_create_time(self, manager):
        """Callback fires once per create — confirming that the
        manager calls into the live session at waypoint time, not at
        callback-registration time."""
        invocations = []

        def get_state():
            invocations.append("called")
            return {"key": "value"}

        manager.set_session_state_callback(get_state)
        manager.create("checkpoint-1")
        assert invocations == ["called"]
        manager.create("checkpoint-2")
        assert invocations == ["called", "called"]

    def test_snapshot_serialised_to_json(self, manager):
        manager.set_session_state_callback(lambda: {"audit": "abc", "n": 7})
        wp = manager.create("checkpoint")
        assert wp.session_state_snapshot is not None
        assert json.loads(wp.session_state_snapshot) == {"audit": "abc", "n": 7}

    def test_empty_snapshot_collapses_to_none(self, manager):
        """When the callback returns an empty dict (no extension has
        attached anything), the waypoint stores None rather than a
        ``"{}"`` blob — keeps legacy waypoints round-tripping
        unchanged."""
        manager.set_session_state_callback(lambda: {})
        wp = manager.create("checkpoint")
        assert wp.session_state_snapshot is None

    def test_snapshot_reflects_live_value_at_create_time(self, manager):
        """Pattern C contract: the callback delegates to
        ``JaatoSession.get_all_session_state``, which invokes
        registered providers — so the snapshot reflects the live
        value at *create time*, not whatever was last pushed."""
        live = {"counter": 1}

        manager.set_session_state_callback(lambda: dict(live))
        wp1 = manager.create("checkpoint-1")
        assert json.loads(wp1.session_state_snapshot) == {"counter": 1}

        live["counter"] = 5
        live["new_key"] = "added later"
        wp2 = manager.create("checkpoint-2")
        assert json.loads(wp2.session_state_snapshot) == {
            "counter": 5,
            "new_key": "added later",
        }


class TestWaypointPersistence:

    def test_session_state_snapshot_round_trips_through_disk(
        self, manager, temp_dir, mock_backup_manager
    ):
        manager.set_session_state_callback(lambda: {"k": "v"})
        wp = manager.create("checkpoint")
        wp_id = wp.id

        # Reload manager from disk.
        reloaded = WaypointManager(
            backup_manager=mock_backup_manager,
            storage_path=temp_dir / "waypoints.json",
        )
        recovered = reloaded.get(wp_id)
        assert recovered is not None
        assert json.loads(recovered.session_state_snapshot) == {"k": "v"}

    def test_legacy_waypoint_dict_without_snapshot_field(self):
        """Waypoint files predating this facility have no
        ``session_state_snapshot`` key — Waypoint.from_dict must
        default to None without raising."""
        legacy = {
            "id": "w1",
            "description": "legacy",
            "created_at": "2026-01-01T00:00:00",
            "turn_index": 0,
            "is_implicit": False,
            "history_snapshot": None,
            "message_count": 0,
            "user_message_preview": None,
            "owner": "user",
            "parent_id": "w0",
            # session_state_snapshot intentionally absent
        }
        wp = Waypoint.from_dict(legacy)
        assert wp.session_state_snapshot is None
