"""Tests for ``SessionManager.ensure_session_loaded`` (PR-212).

Programmatic trigger for the existing disk-load path.  The non-attach
counterpart to ``attach_session`` — same load semantics, none of the
client-attach side effects (no ``attached_clients`` mutation, no
``_client_to_session``, no ``SessionRestoredEvent`` emit).

Pins the documented contract:

- In-memory hit: returns the existing Session without touching disk.
- In-memory miss + disk hit: loads via ``_load_session`` and registers
  in ``_sessions``.
- In-memory miss + disk miss: returns None.
- Race: concurrent ensure_session_loaded for the same session_id where
  one wins the race to ``_sessions`` registration; the loser returns
  the winner's entry.
- Side-effect-free: no ``attached_clients`` mutation, no
  ``_client_to_session`` write, no SessionRestoredEvent emission.
- Client_id passed to ``_load_session`` is the synthetic
  ``"_session_ops"`` (won't collide with real client_ids that use
  IPC/WS connection formats).

Background: the rehydration path already exists via
``attach_session → _load_session``.  This method exposes it
programmatically for daemon-resident plugins doing cross-session work
(canonical caller: ``jaato_premium.session_ops._resolve_target_session``).
Per ``feedback-audit-existing-machinery-before-designing-parallel-path``
— don't reinvent SessionPlugin/persistence; expose the existing trigger.
"""

from __future__ import annotations

import threading
from typing import Optional
from unittest.mock import MagicMock

import pytest

from server.session_manager import SessionManager


def _make_sm() -> SessionManager:
    """Build a SessionManager skeleton sufficient for
    ensure_session_loaded tests.

    Mirrors the pattern in test_cascade_as_client_phase1.py:_make_sm
    + test_cascade_cancel.py:_make_sm.  Patches ``_load_session`` per
    test rather than instantiating the full plugin / runner stack.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._client_to_session = {}
    # _load_session is the workhorse; tests patch it per-case.
    sm._load_session = MagicMock(return_value=None)
    return sm


def _make_session(session_id: str, workspace_path: Optional[str] = None):
    """Build a Session-like mock with the fields ensure_session_loaded
    reads (none — it just stores the returned object) + the fields
    real consumers read after retrieval."""
    session = MagicMock()
    session.session_id = session_id
    session.workspace_path = workspace_path
    session.attached_clients = set()
    return session


# ======================================================================
# In-memory hit path
# ======================================================================


class TestInMemoryHit:
    def test_returns_existing_session_without_calling_load(self):
        sm = _make_sm()
        existing = _make_session("sid1")
        sm._sessions["sid1"] = existing

        result = sm.ensure_session_loaded("sid1")

        assert result is existing
        sm._load_session.assert_not_called()

    def test_in_memory_hit_ignores_workspace_path_arg(self):
        """Once the session is in-memory, workspace_path is irrelevant
        — we've already located it."""
        sm = _make_sm()
        existing = _make_session("sid1", workspace_path="/orig/ws")
        sm._sessions["sid1"] = existing

        result = sm.ensure_session_loaded("sid1", workspace_path="/different/ws")
        assert result is existing
        # workspace_path is NOT passed to _load_session because we
        # never called it.
        sm._load_session.assert_not_called()


# ======================================================================
# In-memory miss → disk load path
# ======================================================================


class TestDiskLoadPath:
    def test_miss_calls_load_with_session_ops_client_id(self):
        sm = _make_sm()
        loaded = _make_session("sid1")
        sm._load_session.return_value = loaded

        result = sm.ensure_session_loaded("sid1", workspace_path="/ws/a")

        assert result is loaded
        sm._load_session.assert_called_once_with(
            "sid1",
            client_id="_session_ops",
            workspace_path="/ws/a",
        )

    def test_loaded_session_registered_in_memory(self):
        sm = _make_sm()
        loaded = _make_session("sid1")
        sm._load_session.return_value = loaded

        sm.ensure_session_loaded("sid1", workspace_path="/ws/a")

        assert sm._sessions["sid1"] is loaded

    def test_subsequent_call_hits_memory(self):
        """After the first load, the second call hits in-memory and
        does NOT re-load."""
        sm = _make_sm()
        loaded = _make_session("sid1")
        sm._load_session.return_value = loaded

        first = sm.ensure_session_loaded("sid1", workspace_path="/ws/a")
        second = sm.ensure_session_loaded("sid1", workspace_path="/ws/a")

        assert first is second
        sm._load_session.assert_called_once()  # NOT twice

    def test_workspace_path_passed_through_to_load(self):
        sm = _make_sm()
        sm._load_session.return_value = _make_session("sid1")

        sm.ensure_session_loaded("sid1", workspace_path="/some/specific/ws")

        call_args = sm._load_session.call_args
        assert call_args[1]["workspace_path"] == "/some/specific/ws"

    def test_no_workspace_path_passes_none(self):
        """When caller doesn't know the workspace, None is passed
        through to _load_session.  Falls back to the daemon's default
        storage path."""
        sm = _make_sm()
        sm._load_session.return_value = _make_session("sid1")

        sm.ensure_session_loaded("sid1")
        assert sm._load_session.call_args[1]["workspace_path"] is None


# ======================================================================
# Disk miss path
# ======================================================================


class TestDiskMiss:
    def test_load_returning_none_propagates_to_caller(self):
        sm = _make_sm()
        sm._load_session.return_value = None

        result = sm.ensure_session_loaded("ghost", workspace_path="/ws/a")
        assert result is None

    def test_disk_miss_does_not_pollute_sessions(self):
        """A disk miss must NOT register anything in _sessions."""
        sm = _make_sm()
        sm._load_session.return_value = None

        sm.ensure_session_loaded("ghost", workspace_path="/ws/a")
        assert "ghost" not in sm._sessions


# ======================================================================
# Side-effect contract — no attach behaviour
# ======================================================================


class TestNoAttachSideEffects:
    def test_does_not_mutate_attached_clients(self):
        """The loaded Session must NOT have any attached_clients added
        by this path — session_ops asks, doesn't join."""
        sm = _make_sm()
        loaded = _make_session("sid1")
        sm._load_session.return_value = loaded

        sm.ensure_session_loaded("sid1", workspace_path="/ws/a")

        # attached_clients on the Session itself remains untouched.
        assert loaded.attached_clients == set(), (
            "ensure_session_loaded must NOT add anything to "
            "attached_clients — session_ops is querying, not attaching."
        )

    def test_does_not_write_client_to_session_map(self):
        """The _client_to_session map must NOT acquire a mapping for
        '_session_ops' (it's a synthetic id, not a real client)."""
        sm = _make_sm()
        sm._load_session.return_value = _make_session("sid1")

        sm.ensure_session_loaded("sid1", workspace_path="/ws/a")

        assert "_session_ops" not in sm._client_to_session, (
            "ensure_session_loaded must NOT write _session_ops to "
            "_client_to_session — that map is for real attached clients."
        )

    def test_does_not_emit_session_restored_event(self):
        """SessionRestoredEvent is the client-attach-on-disk-restored
        signal; ensure_session_loaded must NOT fire it (there's no
        client to receive it)."""
        sm = _make_sm()
        sm._emit_to_session = MagicMock()
        sm._load_session.return_value = _make_session("sid1")

        sm.ensure_session_loaded("sid1", workspace_path="/ws/a")
        sm._emit_to_session.assert_not_called()


# ======================================================================
# Race: concurrent loads for the same session_id
# ======================================================================


class TestConcurrentLoadRace:
    def test_second_caller_returns_winning_entry(self):
        """When two ensure_session_loaded calls race on the same
        session_id (both miss in-memory and call _load_session), the
        second caller's post-load registration finds the winner's
        entry and returns IT rather than overwriting.

        Race tolerance: the load runs OUTSIDE the lock (it's slow);
        the post-load registration re-checks under lock.  Both
        loaders successfully load (each spawns a runner — one runner
        is wasted; reclaimed by subreaper); the second registration
        is a no-op overwrite-prevention.
        """
        sm = _make_sm()

        # Simulate: first caller's load returns sessA; we sneak sessB
        # into _sessions BEFORE the first caller's post-load
        # registration runs.  This models the race where another
        # ensure_session_loaded got there first.
        sess_a = _make_session("sid1")
        sess_b = _make_session("sid1")

        def _load_side_effect(*args, **kwargs):
            # Mid-load, simulate concurrent registration of sess_b.
            sm._sessions["sid1"] = sess_b
            return sess_a

        sm._load_session.side_effect = _load_side_effect

        result = sm.ensure_session_loaded("sid1", workspace_path="/ws/a")

        # The result must be the winning entry (sess_b), not the
        # second caller's loaded copy (sess_a).
        assert result is sess_b
        # _sessions still holds sess_b — we did NOT overwrite.
        assert sm._sessions["sid1"] is sess_b
