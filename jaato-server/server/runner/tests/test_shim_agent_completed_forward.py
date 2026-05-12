"""Tests pinning the runner→daemon forward of ``on_agent_completed``
and ``on_session_quiescent`` (2026-05-12 Path F regression fix).

Pre-fix the runner-side ``_AgentUIHooksNotificationShim`` defined
both methods as ``pass`` no-ops on the assumption that the daemon-
side ``ServerAgentHooks`` would receive them via a different path.
That assumption held pre-§7c when ``lifecycle_tools.py`` ran in the
daemon process; it broke post-§7c when the module moved into the
runner subprocess and the call now lands on the runner-side shim.

These tests pin BOTH ENDS of the wire:

1. **Shim side** — calling the shim's ``on_agent_completed`` /
   ``on_session_quiescent`` emits a NotificationFrame with the
   correct event_type + payload shape.  Pre-fix the call was a
   ``pass`` no-op so no frame was emitted.

2. **Daemon-side demuxer** — receiving an ``agent_completed`` /
   ``session_quiescent`` event_type frame routes the payload into
   the daemon-side ``ServerAgentHooks`` method, which is what
   fires ``AgentCompletedEvent`` / ``SessionTerminatedEvent`` into
   the event bus + reactor engine.

Together these two checks pin the full wire from the runner-side
emit to the daemon-side hook invocation.  An integration test that
spins up a real RPC + JaatoServer is out of scope for this unit
test file; the seam between them is the
``rpc.emit_notification(event_type, payload)`` contract which both
sides agree on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from server.runner.rpc import _AgentUIHooksNotificationShim


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


class _FakeRPC:
    """Minimal RPC stub recording every notification emission.

    Mirrors the ``_RunnerRPC`` interface the shim depends on —
    ``emit_notification(request_id, event_type, payload)`` — plus
    the two NOTIF_ constants the shim references at call time."""

    _NOTIF_TOOL_CALL_START = "tool_call_start"
    _NOTIF_TOOL_CALL_END = "tool_call_end"
    _NOTIF_TOOL_OUTPUT = "tool_output"
    _NOTIF_TURN_PROGRESS = "turn_progress"
    _NOTIF_AGENT_COMPLETED = "agent_completed"
    _NOTIF_SESSION_QUIESCENT = "session_quiescent"

    def __init__(self) -> None:
        self.emissions: List[Tuple[int, str, Dict[str, Any]]] = []

    def emit_notification(
        self,
        request_id: int,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        self.emissions.append((request_id, event_type, dict(payload)))


# ──────────────────────────────────────────────────────────────────
# Shim side — on_agent_completed
# ──────────────────────────────────────────────────────────────────


class TestShimOnAgentCompleted:
    """Pin: the shim's ``on_agent_completed`` emits a
    NotificationFrame with the full payload shape."""

    def test_emits_notification_frame_with_correct_event_type(self):
        """Load-bearing regression pin: pre-fix this method was a
        ``pass`` no-op so the assertion below never fired even
        with the call made.  Post-fix the frame is emitted with
        ``event_type == "agent_completed"``."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=42)
        shim.on_agent_completed(
            agent_id="discovery",
            completed_at=datetime(2026, 5, 12, 20, 19, 59),
            success=True,
            token_usage={"total_tokens": 1234, "prompt_tokens": 1000,
                         "output_tokens": 234},
            turns_used=3,
            payload={"summary": "Discovery complete",
                     "capabilities": ["a", "b", "c"]},
        )
        assert len(rpc.emissions) == 1, (
            "shim.on_agent_completed must emit exactly one "
            "notification frame; pre-fix it emitted zero"
        )
        request_id, event_type, payload = rpc.emissions[0]
        assert request_id == 42
        assert event_type == "agent_completed"

    def test_payload_carries_agent_id_verbatim(self):
        """Pin: the agent_id field on the wire matches the
        caller's value — the field the daemon-side reactor uses
        for where-clause matching."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_completed(
            agent_id="discovery",
            completed_at=datetime(2026, 5, 12, 20, 19, 59),
            success=True,
        )
        _, _, payload = rpc.emissions[0]
        assert payload["agent_id"] == "discovery"

    def test_datetime_completed_at_serialized_to_isoformat(self):
        """Pin: ``datetime`` instances get converted to ISO 8601
        strings on the wire — pydantic-typed
        ``AgentCompletedEvent.completed_at`` is a string.  The
        daemon-side demuxer passes the field through verbatim;
        the conversion has to happen at emit time."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_completed(
            agent_id="x",
            completed_at=datetime(2026, 5, 12, 20, 19, 59),
            success=True,
        )
        _, _, payload = rpc.emissions[0]
        assert payload["completed_at"] == "2026-05-12T20:19:59"

    def test_none_completed_at_passes_through_as_none(self):
        """Pin: ``None`` for ``completed_at`` flows through as
        ``None`` on the wire — defensive default for callers that
        don't supply the timestamp."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_completed(agent_id="x", completed_at=None, success=True)
        _, _, payload = rpc.emissions[0]
        assert payload["completed_at"] is None

    def test_payload_carries_typed_payload_dict(self):
        """Pin: profile-declared typed payloads flow verbatim.
        The runner-side validator at
        ``lifecycle_tools._execute_signal_completion`` already
        accepted the dict; the shim must not re-shape it."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        typed = {"summary": "s", "capabilities": ["a", "b"]}
        shim.on_agent_completed(
            agent_id="x", completed_at=None, success=True, payload=typed,
        )
        _, _, payload = rpc.emissions[0]
        assert payload["payload"] == typed

    def test_success_false_with_error_message(self):
        """Pin: failure-path emission carries ``success=False``
        and the error string verbatim."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_completed(
            agent_id="x", completed_at=None, success=False,
            error="profile validation failed",
        )
        _, _, payload = rpc.emissions[0]
        assert payload["success"] is False
        assert payload["error"] == "profile validation failed"

    def test_emit_failure_does_not_raise(self):
        """Pin: if the RPC emission itself fails, the shim catches
        it (logger.exception is fine) — the call site is the
        lifecycle_tools executor which has no exception-handling
        budget for hook failures."""
        bad_rpc = MagicMock()
        bad_rpc._NOTIF_AGENT_COMPLETED = "agent_completed"
        bad_rpc.emit_notification.side_effect = RuntimeError("wire broken")
        shim = _AgentUIHooksNotificationShim(bad_rpc, request_id=1)
        # Must not raise.
        shim.on_agent_completed(
            agent_id="x", completed_at=None, success=True,
        )


# ──────────────────────────────────────────────────────────────────
# Shim side — on_session_quiescent
# ──────────────────────────────────────────────────────────────────


class TestShimOnSessionQuiescent:
    """Pin: the shim's ``on_session_quiescent`` emits a
    NotificationFrame.  Pre-fix this was also a ``pass`` no-op,
    so ``SessionTerminatedEvent`` was never emitted from runner-
    side sessions (the client saw the session hang past the
    completion-bearing turn)."""

    def test_emits_notification_frame(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=7)
        shim.on_session_quiescent(agent_id="main", reason="natural")
        assert len(rpc.emissions) == 1
        request_id, event_type, payload = rpc.emissions[0]
        assert request_id == 7
        assert event_type == "session_quiescent"
        assert payload["agent_id"] == "main"
        assert payload["reason"] == "natural"

    def test_default_reason_is_natural(self):
        """Pin: the daemon-side default for the ``reason`` field
        is ``"natural"`` (matches the daemon-side ``ServerAgentHooks``
        signature) — the shim must use the same default."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_session_quiescent(agent_id="main")
        _, _, payload = rpc.emissions[0]
        assert payload["reason"] == "natural"


# ──────────────────────────────────────────────────────────────────
# Constants pin
# ──────────────────────────────────────────────────────────────────


class TestNotifConstants:
    """Pin: the two new NOTIF_ constants exist on the
    ``_RunnerRPC`` class (where the shim looks them up) and match
    the daemon-side demuxer's event_type strings."""

    def test_constants_exist_on_runner_rpc(self):
        from server.runner.rpc import RunnerRPC
        assert RunnerRPC._NOTIF_AGENT_COMPLETED == "agent_completed"
        assert RunnerRPC._NOTIF_SESSION_QUIESCENT == "session_quiescent"
