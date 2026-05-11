"""Tests for ``RunnerRPCClient.spawn_isolated_runner`` wrapper.

Phase 4 §4.3.2.

The wrapper is the runner-side surface for the
``subagent.spawn_isolated_runner`` RPC.  Pins four contracts:

1. **Method-name routing** — calls ``outgoing_call`` with the
   canonical method name ``subagent.spawn_isolated_runner``.
2. **Args-dict construction** — required kwargs are always
   present; optional kwargs are included only when non-None.
3. **Envelope parsing** — success envelopes (``ok=True`` with
   dict result) flow through unchanged; non-dict results raise
   ``RunnerRPCError``.
4. **Failure translation** — transport / handler failures raise
   ``RunnerRPCError``; domain failures (``stage`` envelopes) flow
   through as the success-path dict so callers can branch on the
   ``ok`` key inside the result.

These pins close the wire contract that §4.3.7 will consume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from server.runner.envelope import ErrorPayload, ResponseEnvelope
from server.runner.rpc_client import RunnerRPCClient, RunnerRPCError


# ──────────────────────────────────────────────────────────────────────
# Test fixture: stub RunnerRPC capturing outgoing_call args
# ──────────────────────────────────────────────────────────────────────


class _StubRPC:
    """Records every outgoing_call invocation and returns a queued
    envelope.  Tests configure the queue per-case."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any], Optional[float]]] = []
        self._queue: List[ResponseEnvelope] = []

    def queue_response(self, env: ResponseEnvelope) -> None:
        self._queue.append(env)

    def outgoing_call(
        self,
        method: str,
        args: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> ResponseEnvelope:
        self.calls.append((method, args, timeout))
        if not self._queue:
            raise RuntimeError(
                "stub outgoing_call: no queued response for this test"
            )
        return self._queue.pop(0)


def _ok_envelope(result: Any) -> ResponseEnvelope:
    """Success envelope (ok=True with the given result)."""
    return ResponseEnvelope(id=1, ok=True, result=result, error=None)


def _err_envelope(err_type: str, err_msg: str) -> ResponseEnvelope:
    """Transport / handler-level error envelope (ok=False)."""
    return ResponseEnvelope(
        id=1, ok=False, result=None,
        error=ErrorPayload(type=err_type, message=err_msg),
    )


# ──────────────────────────────────────────────────────────────────────
# Method-name routing
# ──────────────────────────────────────────────────────────────────────


class TestMethodNameRouting:
    """The wrapper MUST send the canonical method name on the wire.
    Pinning the string here so refactors can't silently change the
    identifier the daemon-side handler is registered under."""

    def test_routes_through_canonical_method_name(self):
        rpc = _StubRPC()
        rpc.queue_response(_ok_envelope({"ok": True, "session_id": "S-1"}))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        client.spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload={"name": "researcher"},
            task="do thing",
            workspace_path="/work",
        )

        assert len(rpc.calls) == 1
        method, _args, _timeout = rpc.calls[0]
        assert method == "subagent.spawn_isolated_runner"


# ──────────────────────────────────────────────────────────────────────
# Args-dict construction
# ──────────────────────────────────────────────────────────────────────


class TestArgsConstruction:
    """The wrapper builds the request args dict from kwargs.  Pins:
    - Required kwargs are always present.
    - Optional kwargs are included only when non-None (so daemon
      sees ``args.get('agent_params')`` is ``None`` only when caller
      explicitly omitted it, not because the wrapper added a
      placeholder)."""

    def test_required_args_always_present(self):
        rpc = _StubRPC()
        rpc.queue_response(_ok_envelope({"ok": True}))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        client.spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload={"name": "researcher"},
            task="do thing",
            workspace_path="/work",
        )

        _method, args, _timeout = rpc.calls[0]
        assert args["parent_session_id"] == "sess-A"
        assert args["subagent_id"] == "agent-1"
        assert args["profile_payload"] == {"name": "researcher"}
        assert args["task"] == "do thing"
        assert args["workspace_path"] == "/work"

    def test_optional_args_omitted_when_none(self):
        rpc = _StubRPC()
        rpc.queue_response(_ok_envelope({"ok": True}))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        client.spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload={"name": "researcher"},
            task="do thing",
            workspace_path="/work",
            # All optional kwargs left at default None.
        )

        _method, args, _timeout = rpc.calls[0]
        assert "agent_params" not in args
        assert "display_name" not in args
        assert "parent_agent_id" not in args

    def test_optional_args_included_when_set(self):
        rpc = _StubRPC()
        rpc.queue_response(_ok_envelope({"ok": True}))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        client.spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload={"name": "researcher"},
            task="do thing",
            workspace_path="/work",
            agent_params={"case_id": "42"},
            display_name="my-agent",
            parent_agent_id="parent-7",
        )

        _method, args, _timeout = rpc.calls[0]
        assert args["agent_params"] == {"case_id": "42"}
        assert args["display_name"] == "my-agent"
        assert args["parent_agent_id"] == "parent-7"

    def test_timeout_propagates_to_outgoing_call(self):
        rpc = _StubRPC()
        rpc.queue_response(_ok_envelope({"ok": True}))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        client.spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload={"name": "researcher"},
            task="do thing",
            workspace_path="/work",
            timeout=60.0,
        )

        _method, _args, timeout = rpc.calls[0]
        assert timeout == 60.0


# ──────────────────────────────────────────────────────────────────────
# Envelope parsing
# ──────────────────────────────────────────────────────────────────────


class TestEnvelopeParsing:
    """Success-path envelopes (handler returned a dict) flow through
    unchanged so the caller can read fields like ``session_id`` and
    ``stage``."""

    def test_success_envelope_returns_result_dict(self):
        rpc = _StubRPC()
        rpc.queue_response(_ok_envelope({
            "ok": True,
            "session_id": "S-new",
            "subagent_id": "agent-1",
            "runner_pid": 12345,
            "apparmor_profile": "jaato-ws-S-A//agent-1",
            "cgroup_path": "/sys/fs/cgroup/jaato/S-A/agent-1",
        }))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        result = client.spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload={"name": "researcher"},
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is True
        assert result["session_id"] == "S-new"
        assert result["runner_pid"] == 12345

    def test_domain_failure_envelope_returns_result_dict(self):
        """Domain failure (handler returned ``ok=False`` with a
        stage) is NOT a transport failure — it flows through as the
        success-path dict so callers can branch on ``ok`` and
        surface ``stage`` for diagnostics."""
        rpc = _StubRPC()
        rpc.queue_response(_ok_envelope({
            "ok": False,
            "error": "isolated-runner spawn machinery not yet implemented",
            "stage": "spawn",
        }))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        result = client.spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload={"name": "researcher"},
            task="do thing",
            workspace_path="/work",
        )

        # Domain failure surfaces in the result dict, NOT as an exception.
        assert result["ok"] is False
        assert result["stage"] == "spawn"
        assert "not yet implemented" in result["error"].lower()

    def test_non_dict_result_raises(self):
        """Result must be a dict; the wrapper raises if the daemon
        returns anything else (would indicate a wire-protocol
        regression)."""
        rpc = _StubRPC()
        rpc.queue_response(_ok_envelope("not-a-dict"))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        with pytest.raises(RunnerRPCError, match="unexpected result type"):
            client.spawn_isolated_runner(
                parent_session_id="sess-A",
                subagent_id="agent-1",
                profile_payload={"name": "researcher"},
                task="do thing",
                workspace_path="/work",
            )


# ──────────────────────────────────────────────────────────────────────
# Transport-level failure translation
# ──────────────────────────────────────────────────────────────────────


class TestTransportFailureTranslation:
    """Transport-level failures (channel closed, handler crashed
    mid-call, malformed envelope) MUST translate to ``RunnerRPCError``
    so plugin code can distinguish them from domain failures
    (which flow through as the dict)."""

    def test_envelope_ok_false_raises_runner_rpc_error(self):
        rpc = _StubRPC()
        rpc.queue_response(_err_envelope("ValueError", "bad args"))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        with pytest.raises(RunnerRPCError, match="ValueError: bad args"):
            client.spawn_isolated_runner(
                parent_session_id="sess-A",
                subagent_id="agent-1",
                profile_payload={"name": "researcher"},
                task="do thing",
                workspace_path="/work",
            )

    def test_error_message_includes_method_name(self):
        """The error message must include the method name so callers
        can identify which RPC failed in a stack."""
        rpc = _StubRPC()
        rpc.queue_response(_err_envelope("RuntimeError", "channel closed"))
        client = RunnerRPCClient(rpc)  # type: ignore[arg-type]

        with pytest.raises(RunnerRPCError) as exc_info:
            client.spawn_isolated_runner(
                parent_session_id="sess-A",
                subagent_id="agent-1",
                profile_payload={"name": "researcher"},
                task="do thing",
                workspace_path="/work",
            )

        assert "subagent.spawn_isolated_runner" in str(exc_info.value)
