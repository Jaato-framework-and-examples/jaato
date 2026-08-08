"""Tests for the session.get_all_session_state bulk-snapshot
RPC handler (Phase 3 §3.3c precursor).

Daemon-side journal-save / waypoint-snapshot / fork-snapshot
paths will dispatch here once the seat-flip migrates them.

Tests pin:

- happy path: returns the session's full state dict
- empty state → empty dict (not None / error)
- returned dict is a copy (mutation isolation)
- no_host / no_session error paths
- session without get_all_session_state → missing_method error
- session-side read failure → stage=read error
- non-dict return → stage=read error (defensive)
- daemon-side wrapper round-trips correctly
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-all-state",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(self) -> None:
        self.snapshot: Dict[str, Any] = {}

    def get_all_session_state(self) -> Dict[str, Any]:
        return dict(self.snapshot)


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_get_all_state_returns_session_dict() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.snapshot = {
        "audit_chain": "abc123",
        "version": 7,
        "counters": {"requests": 42},
    }
    _install(rpc, session)

    ok, result = rpc._handle_session_get_all_state()
    assert ok is True
    assert result == {"state": session.snapshot}


def test_get_all_state_empty_returns_empty_dict() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_get_all_state()
    assert ok is True
    assert result == {"state": {}}


def test_get_all_state_returns_copy_not_reference() -> None:
    """Mutation of the returned dict must not propagate back into
    session state."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.snapshot = {"key": "value"}
    _install(rpc, session)

    ok, result = rpc._handle_session_get_all_state()
    assert ok is True
    result["state"]["new_key"] = "added"
    # Mutate again to verify isolation.
    ok2, result2 = rpc._handle_session_get_all_state()
    assert "new_key" not in result2["state"]


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_get_all_state_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_all_state()
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_all_state_no_session_returns_error() -> None:
    rpc = _make_lone_runner()
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=None,
    )
    ok, result = rpc._handle_session_get_all_state()
    assert ok is False
    assert result["stage"] == "no_session"


def test_get_all_state_missing_method_returns_error() -> None:
    """Defensive: session lacking get_all_session_state surfaces a
    clean error rather than AttributeError."""
    rpc = _make_lone_runner()
    _install(rpc, object())  # no method

    ok, result = rpc._handle_session_get_all_state()
    assert ok is False
    assert result["stage"] == "missing_method"


def test_get_all_state_session_raises_returns_read_error() -> None:
    rpc = _make_lone_runner()

    class _BoomSession:
        def get_all_session_state(self):
            raise RuntimeError("snapshot boom")

    _install(rpc, _BoomSession())

    ok, result = rpc._handle_session_get_all_state()
    assert ok is False
    assert result["stage"] == "read"
    assert "snapshot boom" in result["error"]


def test_get_all_state_non_dict_return_returns_read_error() -> None:
    rpc = _make_lone_runner()

    class _BadSession:
        def get_all_session_state(self):
            return ["not", "a", "dict"]

    _install(rpc, _BadSession())

    ok, result = rpc._handle_session_get_all_state()
    assert ok is False
    assert result["stage"] == "read"
    assert "expected dict" in result["error"]


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_get_all_session_state() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.snapshot = {"x": 1}
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.get_all_session_state", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"state": {"x": 1}}


# ----------------------------------------------------------------------
# E2E daemon-side wrapper
# ----------------------------------------------------------------------


class _Pair:
    def __init__(
        self, runner_rpc: RunnerRPC, runner_thread: threading.Thread,
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


@pytest.mark.asyncio
async def test_e2e_session_get_all_state_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.snapshot = {
            "audit": {"chain": "h0"},
            "counters": {"calls": 12},
        }
        _install(pair.runner_rpc, session)

        result = await pair.daemon_client.session_get_all_state()
        assert result == session.snapshot
    finally:
        await _teardown(pair)
