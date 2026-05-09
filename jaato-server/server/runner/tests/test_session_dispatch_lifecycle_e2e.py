"""End-to-end integration test: full session lifecycle through
the §3.3c dispatch surface.

Drives a realistic sequence of RPC calls through a real
:func:`socketpair` connecting a :class:`RunnerRPC` (runner side)
to a :class:`RunnerRPCClient` (daemon side):

1. Daemon checks runner status before bootstrap → empty.
2. Daemon installs a session host (simulating bootstrap).
3. Daemon checks status → has host.
4. Daemon sets initial session state via ``session_set_state``.
5. Daemon reads it back via ``session_get_state``.
6. Daemon writes more state, then bulk-snapshots via
   ``session_get_all_state`` to verify consistency.
7. Daemon pushes terminal_width + streaming_enabled config.
8. Daemon reads context-usage stats.
9. Daemon reads conversation history (empty for this test).
10. Daemon checks ``session_is_running`` → False.
11. Daemon attempts ``session_request_stop`` → False (no message
    running) — proves the wire contract for the no-op case.
12. Daemon calls ``session_shutdown`` to tear down.
13. Daemon checks status post-shutdown → no host.

This test verifies the dispatch surface composes correctly —
each handler interacts with the same runner-side state and the
ordering / consistency invariants hold across a realistic
session arc.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List

import pytest

from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient
from shared.session_envelope import SessionInitEnvelope


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-lifecycle-e2e",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Comprehensive stand-in for JaatoSession exercising every
    method the §3.3c dispatch surface invokes."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.terminal_width: int = 0
        self.streaming_enabled: bool = True
        self.running: bool = False
        self.stop_calls: List[str] = []
        self.history: List[Any] = []
        self.usage: Dict[str, Any] = {
            "model": "test-model",
            "context_limit": 100000,
            "total_tokens": 5,
            "prompt_tokens": 5,
            "output_tokens": 0,
            "turns": 1,
            "percent_used": 0.005,
            "tokens_remaining": 99995,
        }
        self.close_calls: List[None] = []

    def get_session_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set_session_state(self, key: str, value: Any) -> None:
        import json
        json.dumps(value)
        self.state[key] = value

    def get_all_session_state(self) -> Dict[str, Any]:
        return dict(self.state)

    def is_running(self) -> bool:
        return self.running

    def request_stop(self, reason: str = "") -> bool:
        self.stop_calls.append(reason)
        return self.running  # only "cancelled" if a message was running

    def get_history(self) -> List[Any]:
        return list(self.history)

    def get_history_raw(self) -> List[Any]:
        return list(self.history)

    def get_context_usage(self) -> Dict[str, Any]:
        return dict(self.usage)

    def set_terminal_width(self, width: int) -> None:
        self.terminal_width = width

    def set_streaming_enabled(self, enabled: bool) -> None:
        self.streaming_enabled = enabled

    def set_presentation_context(self, ctx: Any) -> None:
        # Implicit: also touches terminal_width, but the test
        # doesn't verify that side effect.
        pass

    def close_session(self) -> None:
        self.close_calls.append(None)


class _Pair:
    def __init__(
        self,
        runner_rpc: RunnerRPC,
        runner_thread: threading.Thread,
        daemon_client: RunnerRPCClient,
    ) -> None:
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair() -> _Pair:
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    runner_rpc = RunnerRPC(runner_sock, _no_executor)
    thread = threading.Thread(
        target=runner_rpc.serve, name="rpc-serve-test", daemon=True,
    )
    thread.start()

    daemon_client = RunnerRPCClient(daemon_sock, runner_pid=0)
    await daemon_client.start()
    return _Pair(runner_rpc, thread, daemon_client)


async def _teardown(pair: _Pair) -> None:
    pair.runner_rpc.shutdown()
    pair.runner_thread.join(timeout=2)
    pair.daemon_client._closed = True
    if pair.daemon_client._writer is not None:
        try:
            pair.daemon_client._writer.close()
            await pair.daemon_client._writer.wait_closed()
        except Exception:
            pass
    if pair.daemon_client._read_task is not None:
        pair.daemon_client._read_task.cancel()
        try:
            await pair.daemon_client._read_task
        except (asyncio.CancelledError, Exception):
            pass


# ----------------------------------------------------------------------
# Full lifecycle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_session_lifecycle_through_dispatch_surface() -> None:
    pair = await _make_pair()
    try:
        client = pair.daemon_client

        # ---- 1. Pre-bootstrap status check ----
        status = await client.session_health_check()
        assert status["has_host"] is False
        assert status["session_id"] == ""

        # ---- 2. Install session host (simulates bootstrap) ----
        session = _FakeSession()
        pair.runner_rpc._session_host = RunnerSessionHost(
            envelope=_good_envelope(),
            runtime=None,
            session=session,
        )

        # ---- 3. Post-bootstrap status check ----
        status = await client.session_health_check()
        assert status["has_host"] is True
        assert status["ready"] is True
        assert status["session_id"] == "sess-lifecycle-e2e"

        # ---- 4-6. Set state, read it back, bulk-snapshot ----
        await client.session_set_state("audit_chain", "h0")
        await client.session_set_state("counters", {"calls": 0})
        await client.session_set_state("counters", {"calls": 7})

        v = await client.session_get_state("audit_chain")
        assert v == "h0"
        v = await client.session_get_state("counters")
        assert v == {"calls": 7}

        snapshot = await client.session_get_all_state()
        assert snapshot == {
            "audit_chain": "h0",
            "counters": {"calls": 7},
        }

        # ---- 7. Push config ----
        await client.session_set_terminal_width(120)
        await client.session_set_streaming_enabled(False)
        assert session.terminal_width == 120
        assert session.streaming_enabled is False

        # ---- 8. Read context usage ----
        usage = await client.session_get_context_usage()
        assert usage["model"] == "test-model"
        assert usage["turns"] == 1

        # ---- 9. Read history (empty for this test) ----
        history = await client.session_get_history()
        assert history == []

        # ---- 10. Status: not running ----
        running = await client.session_is_running()
        assert running is False

        # ---- 11. request_stop with no message running → False ----
        cancelled = await client.session_request_stop("operator-cancel")
        assert cancelled is False
        assert session.stop_calls == ["operator-cancel"]

        # ---- 12. Shutdown ----
        sid = await client.session_shutdown()
        assert sid == "sess-lifecycle-e2e"
        assert session.close_calls == [None]

        # ---- 13. Post-shutdown: no host ----
        status = await client.session_health_check()
        assert status["has_host"] is False
        assert status["session_id"] == ""

    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_running_session_request_stop_returns_true() -> None:
    """Variant: when the session is "running", request_stop
    actually issues a cancellation (returns True)."""
    pair = await _make_pair()
    try:
        client = pair.daemon_client
        session = _FakeSession()
        session.running = True
        pair.runner_rpc._session_host = RunnerSessionHost(
            envelope=_good_envelope(),
            runtime=None,
            session=session,
        )

        running = await client.session_is_running()
        assert running is True

        cancelled = await client.session_request_stop("user-cancel")
        assert cancelled is True
        assert session.stop_calls == ["user-cancel"]
    finally:
        await _teardown(pair)
