"""Tests pinning the runner→daemon forward for the six remaining
``ServerAgentHooks`` methods unwired in the 2026-05-12 Path F audit
sweep (follow-up to ``test_shim_agent_completed_forward.py``).

The original Path F audit at ``rpc.py:4087-4094`` listed eight
methods as "covered daemon-side":

  ✓ on_agent_completed       — fixed in PR #81
  ✓ on_session_quiescent     — fixed in PR #81
  → on_agent_created         — this PR
  → on_agent_status_changed  — this PR
  → on_agent_turn_completed  — this PR
  → on_agent_context_updated — this PR
  → on_agent_gc_config       — this PR
  → on_agent_history_updated — this PR

The audit assumption held pre-§7c when the callers (``subagent``
plugin, ``JaatoClient`` wrapper) all ran in the daemon process.
Post-§7c the subagent plugin moved to ``PLUGIN_TIER="runner"`` and
``JaatoClient`` lives runner-side, so every
``self._ui_hooks.on_X`` call hits the runner-side shim — which was
a ``pass`` no-op for these 6 methods pre-fix, silently dropping
the event before it could cross the wire to the daemon-side
``ServerAgentHooks`` + event-bus + reactor engine.

Same shape as PR #81's tests: each test verifies the shim's
notification-emit (event_type + payload shape).  The daemon-side
demuxer's payload-unpacking is exercised separately by the
existing tool_call_* demuxer tests; adding 6 more demuxer-side
tests would be duplicative scaffolding.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

from server.runner.rpc import _AgentUIHooksNotificationShim


class _FakeRPC:
    """Records emit_notification calls.  Stand-in for ``RunnerRPC``
    in unit tests — same shape as the PR #81 test helper but with
    every NOTIF_ constant the sweep references."""

    _NOTIF_AGENT_CREATED = "agent_created"
    _NOTIF_AGENT_STATUS_CHANGED = "agent_status_changed"
    _NOTIF_AGENT_TURN_COMPLETED = "agent_turn_completed"
    _NOTIF_AGENT_CONTEXT_UPDATED = "agent_context_updated"
    _NOTIF_AGENT_GC_CONFIG = "agent_gc_config"
    _NOTIF_AGENT_HISTORY_UPDATED = "agent_history_updated"

    def __init__(self) -> None:
        self.emissions: List[Tuple[int, str, Dict[str, Any]]] = []

    def emit_notification(
        self, request_id: int, event_type: str, payload: Dict[str, Any],
    ) -> None:
        self.emissions.append((request_id, event_type, dict(payload)))


# ──────────────────────────────────────────────────────────────────
# on_agent_created
# ──────────────────────────────────────────────────────────────────


class TestOnAgentCreated:
    def test_emits_with_full_payload(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_created(
            agent_id="subagent_1",
            agent_name="researcher",
            agent_type="subagent",
            profile_name="researcher",
            parent_agent_id="main",
            created_at=datetime(2026, 5, 12, 20, 19, 59),
        )
        assert len(rpc.emissions) == 1
        _, event_type, payload = rpc.emissions[0]
        assert event_type == "agent_created"
        assert payload["agent_id"] == "subagent_1"
        assert payload["agent_name"] == "researcher"
        assert payload["agent_type"] == "subagent"
        assert payload["profile_name"] == "researcher"
        assert payload["parent_agent_id"] == "main"
        assert payload["created_at"] == "2026-05-12T20:19:59"

    def test_datetime_created_at_isoformat(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_created(
            agent_id="x", created_at=datetime(2026, 1, 1, 0, 0, 0),
        )
        _, _, payload = rpc.emissions[0]
        assert payload["created_at"] == "2026-01-01T00:00:00"


# ──────────────────────────────────────────────────────────────────
# on_agent_status_changed
# ──────────────────────────────────────────────────────────────────


class TestOnAgentStatusChanged:
    def test_emits_active_status(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_status_changed(
            agent_id="subagent_1", status="active",
        )
        _, event_type, payload = rpc.emissions[0]
        assert event_type == "agent_status_changed"
        assert payload["agent_id"] == "subagent_1"
        assert payload["status"] == "active"
        assert payload["error"] is None

    def test_emits_error_status_with_error_string(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_status_changed(
            agent_id="x", status="errored", error="provider auth failed",
        )
        _, _, payload = rpc.emissions[0]
        assert payload["status"] == "errored"
        assert payload["error"] == "provider auth failed"


# ──────────────────────────────────────────────────────────────────
# on_agent_turn_completed
# ──────────────────────────────────────────────────────────────────


class TestOnAgentTurnCompleted:
    def test_emits_with_full_token_breakdown(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        function_calls = [
            {"name": "list_tools", "duration_seconds": 0.01},
            {"name": "get_tool_schemas", "duration_seconds": 0.02},
        ]
        shim.on_agent_turn_completed(
            agent_id="main",
            turn_number=3,
            prompt_tokens=2000,
            output_tokens=500,
            total_tokens=2500,
            duration_seconds=4.25,
            function_calls=function_calls,
            cache_read_tokens=1800,
            cache_creation_tokens=200,
        )
        _, event_type, payload = rpc.emissions[0]
        assert event_type == "agent_turn_completed"
        assert payload["turn_number"] == 3
        assert payload["prompt_tokens"] == 2000
        assert payload["output_tokens"] == 500
        assert payload["total_tokens"] == 2500
        assert payload["duration_seconds"] == 4.25
        # ``TurnCompletedEvent.function_calls`` is typed
        # ``List[Dict[str, Any]]`` — the shim must pass the list through
        # verbatim (pre-2026-06-07 this coerced to ``int`` and crashed
        # for any turn that actually contained tool calls).
        assert payload["function_calls"] == function_calls
        assert payload["cache_read_tokens"] == 1800
        assert payload["cache_creation_tokens"] == 200

    def test_optional_cache_fields_none_pass_through(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_turn_completed(
            agent_id="main",
            turn_number=1,
            prompt_tokens=100,
            output_tokens=50,
            total_tokens=150,
            duration_seconds=1.0,
            function_calls=[],
        )
        _, _, payload = rpc.emissions[0]
        assert payload["function_calls"] == []
        assert payload["cache_read_tokens"] is None
        assert payload["cache_creation_tokens"] is None


# ──────────────────────────────────────────────────────────────────
# on_agent_context_updated
# ──────────────────────────────────────────────────────────────────


class TestOnAgentContextUpdated:
    def test_emits_with_usage_breakdown(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_context_updated(
            agent_id="main",
            total_tokens=50000,
            prompt_tokens=48000,
            output_tokens=2000,
            turns=12,
            percent_used=25.0,
        )
        _, event_type, payload = rpc.emissions[0]
        assert event_type == "agent_context_updated"
        assert payload["total_tokens"] == 50000
        assert payload["prompt_tokens"] == 48000
        assert payload["output_tokens"] == 2000
        assert payload["turns"] == 12
        assert payload["percent_used"] == 25.0


# ──────────────────────────────────────────────────────────────────
# on_agent_gc_config
# ──────────────────────────────────────────────────────────────────


class TestOnAgentGCConfig:
    def test_emits_with_full_config(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_gc_config(
            agent_id="main",
            threshold=80.0,
            strategy="hybrid",
            target_percent=60.0,
            continuous_mode=True,
        )
        _, event_type, payload = rpc.emissions[0]
        assert event_type == "agent_gc_config"
        assert payload["threshold"] == 80.0
        assert payload["strategy"] == "hybrid"
        assert payload["target_percent"] == 60.0
        assert payload["continuous_mode"] is True

    def test_default_target_percent_continuous_mode(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_gc_config(
            agent_id="main", threshold=80.0, strategy="truncate",
        )
        _, _, payload = rpc.emissions[0]
        assert payload["target_percent"] is None
        assert payload["continuous_mode"] is False


# ──────────────────────────────────────────────────────────────────
# on_agent_history_updated
# ──────────────────────────────────────────────────────────────────


class TestOnAgentHistoryUpdated:
    def test_emits_with_history_snapshot(self):
        """Pin: history payload flows verbatim across the wire.

        The history snapshot is an opaque list of message dicts;
        the daemon-side hook just stores it on agent state.  The
        shim must not re-shape or filter the data."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        history = [
            {"role": "user", "content": "hello"},
            {"role": "model", "content": "hi"},
        ]
        shim.on_agent_history_updated(agent_id="main", history=history)
        _, event_type, payload = rpc.emissions[0]
        assert event_type == "agent_history_updated"
        assert payload["agent_id"] == "main"
        assert payload["history"] == history

    def test_empty_history(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_history_updated(agent_id="main", history=[])
        _, _, payload = rpc.emissions[0]
        assert payload["history"] == []


# ──────────────────────────────────────────────────────────────────
# NOTIF constants
# ──────────────────────────────────────────────────────────────────


class TestNotifConstants:
    """Pin: all 6 new NOTIF_ strings exist on RunnerRPC and match
    the daemon-side demuxer's event_type literals."""

    def test_all_six_constants_exist(self):
        from server.runner.rpc import RunnerRPC
        assert RunnerRPC._NOTIF_AGENT_CREATED == "agent_created"
        assert RunnerRPC._NOTIF_AGENT_STATUS_CHANGED == "agent_status_changed"
        assert RunnerRPC._NOTIF_AGENT_TURN_COMPLETED == "agent_turn_completed"
        assert RunnerRPC._NOTIF_AGENT_CONTEXT_UPDATED == "agent_context_updated"
        assert RunnerRPC._NOTIF_AGENT_GC_CONFIG == "agent_gc_config"
        assert RunnerRPC._NOTIF_AGENT_HISTORY_UPDATED == "agent_history_updated"
