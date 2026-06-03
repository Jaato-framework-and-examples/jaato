"""Tests for the ``cascade.cancel cid`` IPC verb (PR-211).

Three test classes corresponding to the design surface:

1. ``TestIsCidCancelled`` — the reactor-consult predicate
   (``SessionManager.is_cid_cancelled``).  Shape A from the brainstorm:
   framework owns the cancelled-cid set; reactor polls.

2. ``TestCancelCascade`` — the cancel-iteration mechanism
   (``SessionManager.cancel_cascade``).  Pins: iterates filtered by
   cascade_driver_id, calls server.stop() per session, emits one
   SessionTerminatedEvent(reason="cascade_cancelled") per session,
   exception in one session doesn't block others.

3. ``TestCommandRouterCascadeCancel`` — the IPC verb integration
   (``CommandRouter._handle_cascade_cancel``).  Pins: args validation,
   error envelope on missing cid, system-message confirmation, success
   path forwards to SessionManager.

Test fixtures mirror the production caller shape per
``feedback_test_fixtures_must_mirror_production_caller_shape``: each
Session is constructed with a ``cascade_driver_id`` field exactly
where production sets it (Session dataclass line 126), not bypassed.
"""

from __future__ import annotations

import threading
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from server.session_manager import SessionManager


# ----------------------------------------------------------------------
# Lightweight SessionManager construction
# ----------------------------------------------------------------------


def _make_sm() -> SessionManager:
    """Build a SessionManager skeleton sufficient for cascade.cancel
    tests without invoking the full __init__.

    Mirrors the pattern in test_cascade_as_client_phase1.py:_make_sm.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._cascade_clients = {}
    sm._cascade_clients_lock = threading.Lock()
    sm._cancelled_cids = set()
    sm._cancelled_cids_lock = threading.Lock()
    # _emit_to_session is invoked once per cancelled session.  Mock it
    # so tests can assert on the SessionTerminatedEvent payloads.
    sm._emit_to_session = MagicMock()
    return sm


def _make_session(
    session_id: str,
    cid: Optional[str] = None,
    main_agent_id: Optional[str] = None,
    server_stop_return: bool = False,
    server_stop_raises: Optional[Exception] = None,
):
    """Build a Session mock with the attrs cancel_cascade reads.

    Production shape (Session dataclass at session_manager.py:126):
      - ``cascade_driver_id``: Optional[str], set at create-time from
        envelope.cascade_driver_id.
      - ``server``: the JaatoServer instance; has ``_main_agent_id``
        + ``stop()``.
    """
    session = MagicMock()
    session.session_id = session_id
    session.cascade_driver_id = cid

    if server_stop_raises is not None:
        session.server.stop.side_effect = server_stop_raises
    else:
        session.server.stop.return_value = server_stop_return
    session.server._main_agent_id = main_agent_id
    return session


# ======================================================================
# TestIsCidCancelled — reactor-consult predicate
# ======================================================================


class TestIsCidCancelled:
    def test_empty_set_returns_false(self):
        sm = _make_sm()
        assert sm.is_cid_cancelled("abc") is False

    def test_after_cancel_returns_true(self):
        sm = _make_sm()
        sm.cancel_cascade("abc")
        assert sm.is_cid_cancelled("abc") is True

    def test_unrelated_cid_returns_false(self):
        sm = _make_sm()
        sm.cancel_cascade("abc")
        assert sm.is_cid_cancelled("xyz") is False

    def test_empty_string_cid_returns_false(self):
        """Empty cid → cannot be cancelled (defensive but cheap)."""
        sm = _make_sm()
        # Even if we somehow added "" to the set (we don't — cancel_cascade
        # short-circuits), the predicate returns False on empty input.
        assert sm.is_cid_cancelled("") is False

    def test_idempotent_repeated_marks(self):
        """Marking the same cid twice stays True; set semantics."""
        sm = _make_sm()
        sm.cancel_cascade("abc")
        sm.cancel_cascade("abc")
        assert sm.is_cid_cancelled("abc") is True
        # Set has exactly one entry.
        with sm._cancelled_cids_lock:
            assert sm._cancelled_cids == {"abc"}


# ======================================================================
# TestCancelCascade — iteration + stop + event emission
# ======================================================================


class TestCancelCascade:
    def test_no_matching_sessions_marks_cid_returns_zero(self):
        """Cancelling a cid with no loaded sessions still marks the cid
        (so reactor suppression engages even if all sessions of that
        cascade already completed)."""
        sm = _make_sm()
        sm._sessions["unrelated"] = _make_session("unrelated", cid="other")

        result = sm.cancel_cascade("abc")

        assert result["cid"] == "abc"
        assert result["cancelled_session_ids"] == []
        assert result["stopped_count"] == 0
        # The unrelated session was NOT stopped.
        sm._sessions["unrelated"].server.stop.assert_not_called()
        # The cid IS marked cancelled for the reactor.
        assert sm.is_cid_cancelled("abc") is True

    def test_single_matching_session_stopped_and_emitted(self):
        sm = _make_sm()
        sess = _make_session("sid1", cid="abc", main_agent_id="agent_42")
        sm._sessions["sid1"] = sess

        result = sm.cancel_cascade("abc")

        sess.server.stop.assert_called_once()
        assert result["cancelled_session_ids"] == ["sid1"]
        assert result["stopped_count"] == 1

        # One SessionTerminatedEvent emitted with reason=cascade_cancelled
        # + agent_id from server._main_agent_id.
        sm._emit_to_session.assert_called_once()
        call_args = sm._emit_to_session.call_args
        assert call_args[0][0] == "sid1"
        event = call_args[0][1]
        assert event.session_id == "sid1"
        assert event.agent_id == "agent_42"
        assert event.reason == "cascade_cancelled"

    def test_multiple_matching_sessions_all_stopped(self):
        sm = _make_sm()
        sm._sessions["sid1"] = _make_session("sid1", cid="abc")
        sm._sessions["sid2"] = _make_session("sid2", cid="abc")
        sm._sessions["sid3"] = _make_session("sid3", cid="abc")
        # An unrelated session that must NOT be touched.
        sm._sessions["other"] = _make_session("other", cid="xyz")

        result = sm.cancel_cascade("abc")

        for sid in ("sid1", "sid2", "sid3"):
            sm._sessions[sid].server.stop.assert_called_once()
        sm._sessions["other"].server.stop.assert_not_called()

        assert result["stopped_count"] == 3
        assert set(result["cancelled_session_ids"]) == {"sid1", "sid2", "sid3"}

    def test_main_agent_id_defaults_to_main_when_missing(self):
        """If server._main_agent_id is None, the emitted event uses
        agent_id='main' (mirrors the session.end pattern at
        command_router.py:653)."""
        sm = _make_sm()
        sm._sessions["sid1"] = _make_session("sid1", cid="abc", main_agent_id=None)

        sm.cancel_cascade("abc")
        event = sm._emit_to_session.call_args[0][1]
        assert event.agent_id == "main"

    def test_server_stop_exception_doesnt_block_other_sessions(self):
        """One session's stop raising must not prevent other sessions
        from being cancelled.  Same robustness contract as session.end's
        best-effort stop pattern."""
        sm = _make_sm()
        sm._sessions["sid_bad"] = _make_session(
            "sid_bad", cid="abc",
            server_stop_raises=RuntimeError("boom"),
        )
        sm._sessions["sid_ok"] = _make_session("sid_ok", cid="abc")

        result = sm.cancel_cascade("abc")

        # Both stops attempted.
        sm._sessions["sid_bad"].server.stop.assert_called_once()
        sm._sessions["sid_ok"].server.stop.assert_called_once()
        # Both sessions reported as cancelled (best-effort).
        assert result["stopped_count"] == 2

    def test_session_without_server_skipped(self):
        """A session whose .server is None (bootstrap-in-progress / shutdown
        race) is skipped without crashing — defensive against the dataclass
        field being Optional."""
        sm = _make_sm()
        sess = MagicMock()
        sess.session_id = "sid1"
        sess.cascade_driver_id = "abc"
        sess.server = None
        sm._sessions["sid1"] = sess

        result = sm.cancel_cascade("abc")
        assert result["stopped_count"] == 0
        sm._emit_to_session.assert_not_called()

    def test_empty_cid_is_noop(self):
        """Empty cid → no-op returning zero counts.  Doesn't mark the
        empty string in the cancelled set."""
        sm = _make_sm()
        sm._sessions["sid1"] = _make_session("sid1", cid=None)
        result = sm.cancel_cascade("")
        assert result["stopped_count"] == 0
        assert sm.is_cid_cancelled("") is False
        with sm._cancelled_cids_lock:
            assert "" not in sm._cancelled_cids

    def test_reactor_marker_set_before_emit(self):
        """Race-safety contract: by the time any SessionTerminatedEvent
        is emitted (which can trigger downstream reactors), the cid is
        already marked cancelled — so reactors that see the event
        immediately know to suppress."""
        sm = _make_sm()
        sess = _make_session("sid1", cid="abc")
        sm._sessions["sid1"] = sess

        # Patch _emit_to_session to assert the marker is already set
        # AT the moment of emission.
        observed_marker = []
        def _capture(sid, event):
            observed_marker.append(sm.is_cid_cancelled("abc"))
        sm._emit_to_session = _capture

        sm.cancel_cascade("abc")
        assert observed_marker == [True], (
            f"Marker must be set BEFORE the event emits — race window "
            f"with the reactor.  Observed: {observed_marker}"
        )


# ======================================================================
# TestCommandRouterCascadeCancel — IPC verb integration
# ======================================================================


class TestCommandRouterCascadeCancel:
    def _make_router(self):
        """Build a CommandRouter skeleton sufficient for the cascade.cancel
        handler.  Imports the class lazily because CommandRouter has
        heavy module-level dependencies."""
        from server.command_router import CommandRouter

        router = CommandRouter.__new__(CommandRouter)
        router._session_manager = MagicMock()
        router._event_sink = MagicMock()
        return router

    def test_missing_args_sends_error_event(self):
        router = self._make_router()
        router._handle_cascade_cancel("client_1", [])

        # ErrorEvent sent; SessionManager.cancel_cascade NOT called.
        router._event_sink.send_event.assert_called_once()
        call = router._event_sink.send_event.call_args
        assert call[0][0] == "client_1"
        event = call[0][1]
        assert event.error_type == "UsageError"
        assert "cascade.cancel requires args" in event.error
        router._session_manager.cancel_cascade.assert_not_called()

    def test_success_path_forwards_to_session_manager(self):
        router = self._make_router()
        router._session_manager.cancel_cascade.return_value = {
            "cid": "abc",
            "cancelled_session_ids": ["sid1", "sid2"],
            "stopped_count": 2,
        }
        router._handle_cascade_cancel("client_1", ["abc"])

        router._session_manager.cancel_cascade.assert_called_once_with("abc")
        # One SystemMessageEvent confirmation back to caller.
        router._event_sink.send_event.assert_called_once()
        call = router._event_sink.send_event.call_args
        assert call[0][0] == "client_1"
        msg = call[0][1].message
        assert "abc" in msg
        assert "2 session" in msg
        assert "sid1" in msg and "sid2" in msg

    def test_zero_count_still_emits_confirmation(self):
        """If no sessions matched, the operator should still get a
        message confirming the cid was marked cancelled (so reactor
        suppression is now in effect)."""
        router = self._make_router()
        router._session_manager.cancel_cascade.return_value = {
            "cid": "abc",
            "cancelled_session_ids": [],
            "stopped_count": 0,
        }
        router._handle_cascade_cancel("client_1", ["abc"])

        router._event_sink.send_event.assert_called_once()
        msg = router._event_sink.send_event.call_args[0][1].message
        assert "no loaded sessions matched" in msg
        assert "reactor suppression engaged" in msg
