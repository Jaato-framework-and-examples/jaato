"""Tests for ``SessionManager._forward_subagent_event_to_parent``
(Phase 4 §4.3.6b).

Daemon-side helper that dispatches sub-runner events to the parent
runner via the ``subagent.forward_event`` RPC.  These tests pin the
wire shape + failure modes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from server.session_manager import (
    Session,
    SessionManager,
    SubRunnerHandle,
)


def _make_session_manager():
    sm = SessionManager.__new__(SessionManager)
    import threading
    sm._lock = threading.RLock()
    sm._sessions = {}
    sm._isolated_sub_runners = {}
    return sm


def _make_handle(parent_session_id="sess-A", subagent_id="agent-1"):
    return SubRunnerHandle(
        parent_session_id=parent_session_id,
        subagent_id=subagent_id,
        isolated_session_id=f"{parent_session_id}__sub_{subagent_id}",
        rpc=MagicMock(),
        spawned=MagicMock(),
        sub_apparmor_profile=f"jaato-ws-{parent_session_id}//{subagent_id}",
        cgroup_path="",
    )


def _make_parent_session(session_id, runner_rpc):
    """Build a minimal Session with a server that exposes
    ``runner_rpc``."""
    server = MagicMock()
    server.runner_rpc = runner_rpc
    return Session(
        session_id=session_id,
        name="parent",
        server=server,
        created_at="2026-05-11T00:00:00Z",
    )


class TestForwardSubagentEventToParent:
    def test_happy_path_dispatches_via_call_threadsafe(self):
        sm = _make_session_manager()
        rpc = MagicMock()
        rpc.call_threadsafe.return_value = MagicMock(ok=True, error=None)
        sm._sessions["sess-A"] = _make_parent_session("sess-A", rpc)
        handle = _make_handle()

        result = sm._forward_subagent_event_to_parent(
            handle, "output", {"text": "hello", "source": "assistant"},
        )

        assert result is True
        rpc.call_threadsafe.assert_called_once()
        call = rpc.call_threadsafe.call_args
        assert call[0][0] == "subagent.forward_event"
        args = call[0][1]
        assert args["subagent_id"] == "agent-1"
        assert args["event_kind"] == "output"
        assert args["event_payload"] == {
            "text": "hello", "source": "assistant",
        }

    def test_parent_session_not_found_returns_false(self):
        sm = _make_session_manager()  # no parent registered
        handle = _make_handle()
        result = sm._forward_subagent_event_to_parent(
            handle, "output", {"text": "hello"},
        )
        assert result is False

    def test_parent_session_has_no_runner_rpc_returns_false(self):
        sm = _make_session_manager()
        sm._sessions["sess-A"] = _make_parent_session("sess-A", None)
        handle = _make_handle()
        result = sm._forward_subagent_event_to_parent(
            handle, "output", {"text": "hello"},
        )
        assert result is False

    def test_rpc_exception_returns_false(self):
        sm = _make_session_manager()
        rpc = MagicMock()
        rpc.call_threadsafe.side_effect = RuntimeError("channel closed")
        sm._sessions["sess-A"] = _make_parent_session("sess-A", rpc)
        handle = _make_handle()
        result = sm._forward_subagent_event_to_parent(
            handle, "output", {"text": "hello"},
        )
        assert result is False

    def test_handler_rejection_returns_false(self):
        """Runner-side handler returned ok=False (e.g., plugin not
        loaded).  Daemon helper returns False so caller can log."""
        sm = _make_session_manager()
        rpc = MagicMock()
        err = MagicMock()
        err.message = "SubagentPlugin not loaded"
        rpc.call_threadsafe.return_value = MagicMock(ok=False, error=err)
        sm._sessions["sess-A"] = _make_parent_session("sess-A", rpc)
        handle = _make_handle()
        result = sm._forward_subagent_event_to_parent(
            handle, "output", {"text": "hello"},
        )
        assert result is False

    def test_timeout_propagates_to_call_threadsafe(self):
        sm = _make_session_manager()
        rpc = MagicMock()
        rpc.call_threadsafe.return_value = MagicMock(ok=True, error=None)
        sm._sessions["sess-A"] = _make_parent_session("sess-A", rpc)
        handle = _make_handle()
        sm._forward_subagent_event_to_parent(
            handle, "status", {"status": "done"}, timeout=10.0,
        )
        kwargs = rpc.call_threadsafe.call_args.kwargs
        assert kwargs["timeout"] == 10.0
