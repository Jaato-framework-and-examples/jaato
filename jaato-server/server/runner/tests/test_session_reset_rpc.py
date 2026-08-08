"""Tests for the session.reset RPC handler + wrapper
(Phase 3 §3.3c precursor).

Currently supports the no-history "fresh reset" path only.
Restoring a saved history requires Message round-trip
serialization (Message lacks ``from_dict`` today) and lands as a
separate handler when that design completes.

Tests pin:

- happy path: reset_session called once with no args
- no_host / no_session error paths
- reset_session crash → stage=reset error
- dispatch routing
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, List

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
        session_id="sess-reset-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(self) -> None:
        self.reset_calls: List[None] = []

    def reset_session(self, *args: Any, **kwargs: Any) -> None:
        self.reset_calls.append(None)


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Direct handler
# ----------------------------------------------------------------------


def test_reset_happy_path() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_reset()
    assert ok is True
    assert result == {"ok": True}
    assert session.reset_calls == [None]


def test_reset_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_reset()
    assert ok is False
    assert result["stage"] == "no_host"


def test_reset_session_crash_returns_reset_error() -> None:
    rpc = _make_lone_runner()

    class _BoomSession:
        def reset_session(self):
            raise RuntimeError("reset boom")

    _install(rpc, _BoomSession())

    ok, result = rpc._handle_session_reset()
    assert ok is False
    assert result["stage"] == "reset"
    assert "reset boom" in result["error"]


def test_dispatch_routes_session_reset() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    env = RequestEnvelope(id=1, method="session.reset", args={})
    ok, _ = rpc._dispatch_method(env)
    assert ok is True
    assert session.reset_calls == [None]


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
async def test_e2e_session_reset_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)
        await pair.daemon_client.session_reset()
        assert session.reset_calls == [None]
    finally:
        await _teardown(pair)
