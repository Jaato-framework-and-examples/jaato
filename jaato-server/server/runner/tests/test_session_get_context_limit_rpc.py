"""Tests for the session.get_context_limit RPC handler + wrapper
(Phase 3 §7b.1 precursor).

Read-only handler returning the context-window size in tokens.
Daemon-side fallback path uses this when ``get_context_usage``
returns 0 / missing context_limit.

Tests pin: happy path, no_host error, read-failure error,
non-int return rejection, negative int rejection, dispatch
routing, e2e wrapper.
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
        session_id="sess-cl-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(self, limit: int = 200_000) -> None:
        self._limit = limit
        self._raises: Any = None

    def get_context_limit(self) -> int:
        if self._raises is not None:
            raise self._raises
        return self._limit


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_get_context_limit_returns_int() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(limit=128_000))

    ok, result = rpc._handle_session_get_context_limit()
    assert ok is True
    assert result == {"context_limit": 128_000}


def test_get_context_limit_zero_is_valid() -> None:
    """Zero is a legitimate result (provider not yet initialized);
    caller does its own fallback logic."""
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(limit=0))

    ok, result = rpc._handle_session_get_context_limit()
    assert ok is True
    assert result == {"context_limit": 0}


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_get_context_limit_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_context_limit()
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_context_limit_session_raises_returns_read_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session._raises = RuntimeError("limit boom")
    _install(rpc, session)

    ok, result = rpc._handle_session_get_context_limit()
    assert ok is False
    assert result["stage"] == "read"
    assert "limit boom" in result["error"]


def test_get_context_limit_non_int_return_rejected() -> None:
    rpc = _make_lone_runner()

    class _BadSession:
        def get_context_limit(self):
            return "not_an_int"

    _install(rpc, _BadSession())

    ok, result = rpc._handle_session_get_context_limit()
    assert ok is False
    assert result["stage"] == "read"
    assert "non-negative int" in result["error"]


def test_get_context_limit_negative_int_rejected() -> None:
    """Defensive: a -1 sentinel from a buggy provider would
    confuse daemon-side fallback logic.  Reject upfront."""
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(limit=-1))

    ok, result = rpc._handle_session_get_context_limit()
    assert ok is False
    assert result["stage"] == "read"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_get_context_limit() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(limit=64_000))

    env = RequestEnvelope(
        id=1, method="session.get_context_limit", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"context_limit": 64_000}


# ----------------------------------------------------------------------
# E2E daemon-side wrapper
# ----------------------------------------------------------------------


class _Pair:
    def __init__(self, runner_rpc, runner_thread, daemon_client) -> None:
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair() -> _Pair:
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _no_executor(name, args):
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
async def test_e2e_session_get_context_limit_round_trip() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _FakeSession(limit=200_000))
        limit = await pair.daemon_client.session_get_context_limit()
        assert limit == 200_000
    finally:
        await _teardown(pair)
