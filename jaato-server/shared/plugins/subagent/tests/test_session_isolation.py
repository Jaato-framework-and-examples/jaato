"""Tests for subagent session isolation.

Verifies that subagents are scoped to their owning parent session:
- Each parent session only sees its own subagents
- Subagent IDs are numbered independently per owner
- Tool executors (list, close, cancel, send) enforce ownership
- Persistence exports only the owning session's subagents
- shutdown() cleans up all state
"""

import threading
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ..plugin import SubagentPlugin


def _make_mock_session(agent_id="main"):
    """Create a lightweight mock JaatoSession for testing.

    The mock has the minimum attributes the SubagentPlugin inspects
    when filtering / enumerating subagents.
    """
    session = MagicMock()
    session._agent_id = agent_id
    session.is_running = False
    session.supports_stop = True
    session.activity_phase = MagicMock()
    session.activity_phase.value = "idle"
    session.phase_duration_seconds = 0.0
    session.phase_started_at = None
    return session


def _make_profile(name="test_profile"):
    """Create a minimal SubagentProfile mock."""
    profile = MagicMock()
    profile.name = name
    profile.max_turns = 10
    return profile


def _register_subagent(plugin, agent_id, owner_session, profile=None):
    """Directly register a subagent in the plugin's _active_sessions.

    This bypasses the full spawn flow so we can test filtering logic
    in isolation.
    """
    if profile is None:
        profile = _make_profile()
    child_session = _make_mock_session(agent_id)
    owner_id = id(owner_session)
    with plugin._sessions_lock:
        plugin._active_sessions[agent_id] = {
            'session': child_session,
            'profile': profile,
            'agent_id': agent_id,
            'owner_id': owner_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'turn_count': 0,
            'max_turns': profile.max_turns,
        }
    return child_session


# =========================================================================
# Owner helpers
# =========================================================================

class TestGetOwnerId:
    """Tests for _get_owner_id()."""

    def test_returns_zero_when_no_parent(self):
        plugin = SubagentPlugin()
        assert plugin._get_owner_id() == 0

    def test_returns_session_id_when_parent_set(self):
        plugin = SubagentPlugin()
        session = _make_mock_session()
        plugin.set_parent_session(session)
        assert plugin._get_owner_id() == id(session)

    def test_different_sessions_produce_different_ids(self):
        plugin = SubagentPlugin()
        s1 = _make_mock_session("session_a")
        s2 = _make_mock_session("session_b")

        plugin.set_parent_session(s1)
        id1 = plugin._get_owner_id()

        plugin.set_parent_session(s2)
        id2 = plugin._get_owner_id()

        assert id1 != id2


# =========================================================================
# Per-owner agent ID generation
# =========================================================================

class TestNextAgentId:
    """Tests for _next_agent_id()."""

    def test_ids_start_from_one(self):
        plugin = SubagentPlugin()
        plugin.initialize()
        with plugin._sessions_lock:
            aid = plugin._next_agent_id(owner_id=1)
        assert aid == "subagent_1"

    def test_ids_increment_per_owner(self):
        plugin = SubagentPlugin()
        plugin.initialize()
        with plugin._sessions_lock:
            a1 = plugin._next_agent_id(owner_id=1)
            a2 = plugin._next_agent_id(owner_id=1)
        assert a1 == "subagent_1"
        assert a2 == "subagent_2"

    def test_independent_counters_per_owner(self):
        """Two different owners should each start numbering from 1."""
        plugin = SubagentPlugin()
        plugin.initialize()
        with plugin._sessions_lock:
            o1_a1 = plugin._next_agent_id(owner_id=100)
            o2_a1 = plugin._next_agent_id(owner_id=200)
            o1_a2 = plugin._next_agent_id(owner_id=100)
        assert o1_a1 == "subagent_1"
        assert o2_a1 == "subagent_1"
        assert o1_a2 == "subagent_2"


# =========================================================================
# Filtered session access
# =========================================================================

class TestGetOwnedSessions:
    """Tests for _get_owned_sessions()."""

    def test_empty_when_no_sessions(self):
        plugin = SubagentPlugin()
        with plugin._sessions_lock:
            assert plugin._get_owned_sessions(owner_id=1) == {}

    def test_returns_only_matching_owner(self):
        plugin = SubagentPlugin()
        s_owner_a = _make_mock_session("owner_a")
        s_owner_b = _make_mock_session("owner_b")

        _register_subagent(plugin, "subagent_1", s_owner_a)
        _register_subagent(plugin, "subagent_2", s_owner_b)
        _register_subagent(plugin, "subagent_3", s_owner_a)

        with plugin._sessions_lock:
            owned_a = plugin._get_owned_sessions(id(s_owner_a))
            owned_b = plugin._get_owned_sessions(id(s_owner_b))

        assert set(owned_a.keys()) == {"subagent_1", "subagent_3"}
        assert set(owned_b.keys()) == {"subagent_2"}


# =========================================================================
# list_active_subagents isolation
# =========================================================================

class TestListActiveSubagentsIsolation:
    """list_active_subagents should only return subagents owned by
    the current parent session."""

    def test_empty_when_no_subagents(self):
        plugin = SubagentPlugin()
        plugin.initialize()
        session = _make_mock_session()
        plugin.set_parent_session(session)

        result = plugin._execute_list_active_subagents({})
        assert result['active_sessions'] == []

    def test_only_shows_own_subagents(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner_a = _make_mock_session("owner_a")
        owner_b = _make_mock_session("owner_b")

        _register_subagent(plugin, "subagent_1", owner_a)
        _register_subagent(plugin, "subagent_2", owner_b)
        _register_subagent(plugin, "subagent_3", owner_a)

        # Set parent to owner_a — should only see subagent_1 and subagent_3
        plugin.set_parent_session(owner_a)
        result = plugin._execute_list_active_subagents({})
        ids = {s['subagent_id'] for s in result['active_sessions']}
        assert ids == {"subagent_1", "subagent_3"}

        # Switch to owner_b — should only see subagent_2
        plugin.set_parent_session(owner_b)
        result = plugin._execute_list_active_subagents({})
        ids = {s['subagent_id'] for s in result['active_sessions']}
        assert ids == {"subagent_2"}


# =========================================================================
# close_subagent isolation
# =========================================================================

class TestCloseSubagentIsolation:
    """close_subagent should refuse to close subagents not owned by
    the current parent session."""

    def test_close_own_subagent_succeeds(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner = _make_mock_session("owner")
        _register_subagent(plugin, "subagent_1", owner)

        plugin.set_parent_session(owner)
        result = plugin._execute_close_subagent({'subagent_id': 'subagent_1'})
        assert result['success'] is True

    def test_close_other_owners_subagent_fails(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner_a = _make_mock_session("owner_a")
        owner_b = _make_mock_session("owner_b")

        _register_subagent(plugin, "subagent_1", owner_a)

        # owner_b tries to close owner_a's subagent
        plugin.set_parent_session(owner_b)
        result = plugin._execute_close_subagent({'subagent_id': 'subagent_1'})
        assert result['success'] is False
        assert 'No active session found' in result['message']

        # Verify the subagent is still there
        assert 'subagent_1' in plugin._active_sessions


# =========================================================================
# cancel_subagent isolation
# =========================================================================

class TestCancelSubagentIsolation:
    """cancel_subagent should refuse to cancel subagents not owned by
    the current parent session."""

    def test_cancel_other_owners_subagent_fails(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner_a = _make_mock_session("owner_a")
        owner_b = _make_mock_session("owner_b")

        child = _register_subagent(plugin, "subagent_1", owner_a)
        child.is_running = True

        # owner_b tries to cancel owner_a's subagent
        plugin.set_parent_session(owner_b)
        result = plugin._execute_cancel_subagent({'subagent_id': 'subagent_1'})
        assert result['success'] is False
        assert 'No active session found' in result['message']


# =========================================================================
# send_to_subagent isolation
# =========================================================================

class TestSendToSubagentIsolation:
    """send_to_subagent should refuse to send to subagents not owned by
    the current parent session."""

    def test_send_to_other_owners_subagent_fails(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner_a = _make_mock_session("owner_a")
        owner_b = _make_mock_session("owner_b")

        _register_subagent(plugin, "subagent_1", owner_a)

        # owner_b tries to send to owner_a's subagent
        plugin.set_parent_session(owner_b)
        result = plugin._execute_send_to_subagent({
            'subagent_id': 'subagent_1',
            'message': 'hello',
        })
        assert result['success'] is False
        assert 'No active session found' in result['error']


# =========================================================================
# cancel_all_running isolation
# =========================================================================

class TestCancelAllRunningIsolation:
    """cancel_all_running with owner_only=True should only cancel the
    current owner's subagents."""

    def test_cancel_all_owner_only(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner_a = _make_mock_session("owner_a")
        owner_b = _make_mock_session("owner_b")

        child_a = _register_subagent(plugin, "subagent_1", owner_a)
        child_a.is_running = True
        child_a.supports_stop = True
        child_a.request_stop.return_value = True

        child_b = _register_subagent(plugin, "subagent_2", owner_b)
        child_b.is_running = True
        child_b.supports_stop = True
        child_b.request_stop.return_value = True

        plugin.set_parent_session(owner_a)
        count = plugin.cancel_all_running(owner_only=True)
        assert count == 1
        child_a.request_stop.assert_called_once()
        child_b.request_stop.assert_not_called()

    def test_cancel_all_global(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner_a = _make_mock_session("owner_a")
        owner_b = _make_mock_session("owner_b")

        child_a = _register_subagent(plugin, "subagent_1", owner_a)
        child_a.is_running = True
        child_a.supports_stop = True
        child_a.request_stop.return_value = True

        child_b = _register_subagent(plugin, "subagent_2", owner_b)
        child_b.is_running = True
        child_b.supports_stop = True
        child_b.request_stop.return_value = True

        count = plugin.cancel_all_running(owner_only=False)
        assert count == 2


# =========================================================================
# Persistence isolation
# =========================================================================

class TestPersistenceIsolation:
    """get_persistence_state should only export the current owner's
    subagents."""

    def test_persistence_filtered_by_owner(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner_a = _make_mock_session("owner_a")
        owner_b = _make_mock_session("owner_b")

        _register_subagent(plugin, "subagent_1", owner_a)
        _register_subagent(plugin, "subagent_2", owner_b)

        plugin.set_parent_session(owner_a)
        state = plugin.get_persistence_state()
        agent_ids = [a['agent_id'] for a in state.get('agents', [])]
        assert 'subagent_1' in agent_ids
        assert 'subagent_2' not in agent_ids


# =========================================================================
# Shutdown cleanup
# =========================================================================

class TestShutdownCleanup:
    """shutdown() should clear all state."""

    def test_shutdown_clears_sessions(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        owner = _make_mock_session("owner")
        _register_subagent(plugin, "subagent_1", owner)

        assert len(plugin._active_sessions) == 1
        plugin.shutdown()
        assert len(plugin._active_sessions) == 0

    def test_shutdown_clears_counters(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        with plugin._sessions_lock:
            plugin._next_agent_id(owner_id=42)

        assert plugin._owner_counters.get(42) == 1
        plugin.shutdown()
        assert len(plugin._owner_counters) == 0

    def test_shutdown_clears_parent_session(self):
        plugin = SubagentPlugin()
        plugin.initialize()

        plugin.set_parent_session(_make_mock_session())
        assert plugin._parent_session is not None

        plugin.shutdown()
        assert plugin._parent_session is None
