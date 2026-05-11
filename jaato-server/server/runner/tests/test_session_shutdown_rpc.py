"""Tests for the session.shutdown RPC handler + wrapper
(Phase 3 §3.3c precursor).

Graceful runner-side session teardown — drops the bootstrapped
host reference and calls ``close_session`` on the underlying
JaatoSession (firing on_session_end hooks) BEFORE the
RunnerRPCClient transport-level close ladder kicks in.

Tests pin:

- shutdown with no host bootstrapped is a no-op (returns empty
  session_id, no error)
- shutdown with host but no session (test-stub mode) drops the
  host without calling close_session
- shutdown with full host invokes close_session and drops the
  ref so subsequent dispatches see "no host"
- close_session crash → clean stage=close error
- shutdown is idempotent (second call returns no-op success)
- end-to-end through the daemon-side wrapper round-trips the
  session_id correctly
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional

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
        session_id="sess-shutdown-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


# ----------------------------------------------------------------------
# Direct handler tests (no socket)
# ----------------------------------------------------------------------


def test_shutdown_no_host_returns_empty_session_id() -> None:
    """No bootstrap yet → shutdown is a no-op, returns empty
    session_id.  Daemon callers can rely on this for unconditional
    cleanup paths (try shutdown; ignore empty result)."""
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_shutdown()
    assert ok is True
    assert result == {"shutdown_session_id": ""}


def test_shutdown_host_no_session_drops_ref_without_close() -> None:
    """Test-stub bootstrap mode (session=None): the handler still
    drops the host reference.  No close_session attempt since
    there's no session object."""
    rpc = _make_lone_runner()
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=None,
    )
    ok, result = rpc._handle_session_shutdown()
    assert ok is True
    assert result == {"shutdown_session_id": "sess-shutdown-1"}
    # Host reference dropped.
    assert rpc._session_host is None


def test_shutdown_invokes_close_session_on_full_host() -> None:
    """Happy path: host has a session, close_session called once,
    host reference dropped."""
    rpc = _make_lone_runner()
    close_calls: List[None] = []

    class _Session:
        def close_session(self) -> None:
            close_calls.append(None)

    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=_Session(),
    )
    ok, result = rpc._handle_session_shutdown()
    assert ok is True
    assert result == {"shutdown_session_id": "sess-shutdown-1"}
    assert close_calls == [None]
    assert rpc._session_host is None


def test_shutdown_session_without_close_session_method_is_safe() -> None:
    """Sessions that don't define close_session (some test doubles,
    bare object instances) just get the host dropped without a
    close attempt.  Defensive: no AttributeError."""
    rpc = _make_lone_runner()
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=object(),  # has no close_session
    )
    ok, result = rpc._handle_session_shutdown()
    assert ok is True
    assert result["shutdown_session_id"] == "sess-shutdown-1"
    assert rpc._session_host is None


def test_shutdown_close_session_crash_returns_error_but_drops_host() -> None:
    """If close_session raises, the handler still drops the host
    (so the runner-side state is consistent — partial cleanup is
    still cleanup) but surfaces the error to the daemon for
    logging.  Daemon-side caller branches on success vs error."""
    rpc = _make_lone_runner()

    class _BoomSession:
        def close_session(self):
            raise RuntimeError("close boom")

    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=_BoomSession(),
    )
    ok, result = rpc._handle_session_shutdown()
    assert ok is False
    assert result["stage"] == "close"
    assert "close boom" in result["error"]
    # Host reference STILL dropped — partial cleanup is correct.
    assert rpc._session_host is None


def test_shutdown_idempotent_second_call_is_noop() -> None:
    """Re-calling shutdown after the host is dropped returns the
    no-op success (matches the no-host case at the top)."""
    rpc = _make_lone_runner()

    class _Session:
        close_count = 0

        def close_session(self):
            type(self).close_count += 1

    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=_Session(),
    )

    ok1, r1 = rpc._handle_session_shutdown()
    ok2, r2 = rpc._handle_session_shutdown()

    assert ok1 is True
    assert r1["shutdown_session_id"] == "sess-shutdown-1"
    assert ok2 is True
    assert r2["shutdown_session_id"] == ""  # no-op second time
    assert _Session.close_count == 1  # not double-called


def test_dispatch_routes_session_shutdown() -> None:
    """The new dispatch entry routes correctly via _dispatch_method."""
    rpc = _make_lone_runner()
    env = RequestEnvelope(id=1, method="session.shutdown", args={})
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result["shutdown_session_id"] == ""


# ----------------------------------------------------------------------
# End-to-end via daemon-side wrapper
# ----------------------------------------------------------------------


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


@pytest.mark.asyncio
async def test_e2e_session_shutdown_no_host_returns_empty_string() -> None:
    pair = await _make_pair()
    try:
        sid = await pair.daemon_client.session_shutdown()
        assert sid == ""
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_session_shutdown_drops_bootstrapped_host() -> None:
    pair = await _make_pair()
    try:
        close_calls: List[None] = []

        class _Session:
            def close_session(self):
                close_calls.append(None)

        pair.runner_rpc._session_host = RunnerSessionHost(
            envelope=_good_envelope(),
            runtime=None,
            session=_Session(),
        )

        sid = await pair.daemon_client.session_shutdown()
        assert sid == "sess-shutdown-1"
        assert close_calls == [None]
        # Health-check now reports no host.
        status = await pair.daemon_client.session_health_check()
        assert status["has_host"] is False
    finally:
        await _teardown(pair)
