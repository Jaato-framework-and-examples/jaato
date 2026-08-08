"""Tests for the session.set_parallel_tools_override RPC handler
+ daemon-side wrapper (Phase 3 §7c step 6.6.3.3, 3/5 of the
6.6.3 prerequisite handler series).

Stashes a per-turn parallel-tools override on the runner-side
session.  Replaces the pre-§7c daemon-side private-attr write
at ``server/session_manager.py:4096``:

    jaato_session._parallel_tools_override = event.parallel_tools

Now wraps the public method
``JaatoSession.set_parallel_tools_override`` added in §7c
step 6.6.3.0.

Tests pin:

- happy path: True / False round-trip
- bool coercion: truthy / falsy non-bool values coerce
- arg validation: missing 'enabled' key → stage="decode"
- no_host / no_session error paths
- setter raises → stage="set"
- e2e via daemon-side wrapper round-trip
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, List, Optional

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
        session_id="sess-parallel-tools",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(self) -> None:
        self.calls: List[bool] = []
        self.raise_on_set: Optional[Exception] = None

    def set_parallel_tools_override(self, enabled: bool) -> None:
        if self.raise_on_set is not None:
            raise self.raise_on_set
        self.calls.append(bool(enabled))


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Direct handler tests
# ----------------------------------------------------------------------


def test_set_parallel_tools_override_true() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_parallel_tools_override({
        "enabled": True,
    })
    assert ok is True
    assert result == {"ok": True}
    assert session.calls == [True]


def test_set_parallel_tools_override_false() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_parallel_tools_override({
        "enabled": False,
    })
    assert ok is True
    assert result == {"ok": True}
    assert session.calls == [False]


def test_set_parallel_tools_override_coerces_truthy_non_bool() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for truthy in [1, "yes", [1]]:
        ok, _ = rpc._handle_session_set_parallel_tools_override({
            "enabled": truthy,
        })
        assert ok is True
    assert all(v is True for v in session.calls)
    assert len(session.calls) == 3


def test_set_parallel_tools_override_coerces_falsy_non_bool() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for falsy in [0, "", []]:
        ok, _ = rpc._handle_session_set_parallel_tools_override({
            "enabled": falsy,
        })
        assert ok is True
    assert all(v is False for v in session.calls)
    assert len(session.calls) == 3


def test_set_parallel_tools_override_rejects_missing_key() -> None:
    """Absent 'enabled' surfaces as stage='decode' rather than
    silently treating as False — protects against a daemon-side
    wrapper typo silently disabling parallel-tool execution."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_parallel_tools_override({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "enabled" in result["error"]
    assert session.calls == []


def test_set_parallel_tools_override_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_set_parallel_tools_override({
        "enabled": True,
    })
    assert ok is False
    assert result["stage"] == "no_host"


def test_set_parallel_tools_override_setter_raises_returns_set_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_set = RuntimeError("setter boom")
    _install(rpc, session)

    ok, result = rpc._handle_session_set_parallel_tools_override({
        "enabled": True,
    })
    assert ok is False
    assert result["stage"] == "set"
    assert "setter boom" in result["error"]


def test_set_parallel_tools_override_session_missing_method() -> None:
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits set_parallel_tools_override.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_set_parallel_tools_override({
        "enabled": True,
    })
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_set_parallel_tools_override() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.set_parallel_tools_override",
        args={"enabled": True},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"ok": True}
    assert session.calls == [True]


# ----------------------------------------------------------------------
# E2E daemon-side wrapper round-trip
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
        target=runner_rpc.serve, name="rpc-parallel-tools-test", daemon=True,
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
async def test_e2e_set_parallel_tools_override_true() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_set_parallel_tools_override(True)
        assert session.calls == [True]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_set_parallel_tools_override_false() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_set_parallel_tools_override(False)
        assert session.calls == [False]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_set_parallel_tools_override_toggle_sequence() -> None:
    """Per-turn override pattern: each turn pre-send_message gets
    its own override.  Pin that consecutive calls all land in
    order."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_set_parallel_tools_override(True)
        await pair.daemon_client.session_set_parallel_tools_override(False)
        await pair.daemon_client.session_set_parallel_tools_override(True)

        assert session.calls == [True, False, True]
    finally:
        await _teardown(pair)
