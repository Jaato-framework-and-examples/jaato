"""Tests for the session.set_reference_authorizer RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.1, 1/3).

Forwards the daemon's ``ReferenceAuthorizer`` state to the
runner-side ``JaatoSession`` as a bool flag — the Python object
can't cross the RPC boundary (holds a daemon-side
``AppArmorManager`` reference).  When the references plugin
migrates runner-side, it reads the flag via
:meth:`JaatoSession.is_reference_authorization_enabled` and uses
the existing ``apparmor.add_reference_fragment`` runner→daemon
RPC to authorize paths.

Tests pin:

- happy path: handler stores the bool on the session via
  ``set_reference_authorization_enabled``
- True / False values both round-trip
- bool coercion: truthy non-bool ``1`` becomes ``True`` (avoids
  spurious decode failures from the wire)
- arg validation: missing ``enabled`` key → stage="decode"
- no_host / no_session error paths
- e2e via daemon-side wrapper: bool flag round-trip
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
        session_id="sess-refauth",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Minimal stand-in exercising the
    ``set_reference_authorization_enabled`` setter."""

    def __init__(self) -> None:
        self.enabled_calls: List[bool] = []
        self.raise_on_set: Exception = None

    def set_reference_authorization_enabled(self, enabled: bool) -> None:
        if self.raise_on_set is not None:
            raise self.raise_on_set
        self.enabled_calls.append(bool(enabled))


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Direct handler tests
# ----------------------------------------------------------------------


def test_set_reference_authorizer_enabled_true() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_reference_authorizer(
        {"enabled": True},
    )
    assert ok is True
    assert result == {"ok": True}
    assert session.enabled_calls == [True]


def test_set_reference_authorizer_enabled_false() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_reference_authorizer(
        {"enabled": False},
    )
    assert ok is True
    assert result == {"ok": True}
    assert session.enabled_calls == [False]


def test_set_reference_authorizer_coerces_truthy_non_bool() -> None:
    """Defensive: ``1`` / ``"yes"`` / non-empty values from the
    JSON wire coerce to True rather than raise.  Daemon callers
    should pass real bools but coercion avoids spurious failures."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for truthy in [1, "yes", [1], {"a": 1}]:
        ok, result = rpc._handle_session_set_reference_authorizer(
            {"enabled": truthy},
        )
        assert ok is True, f"truthy={truthy!r} should coerce to True"
        assert result == {"ok": True}

    assert all(v is True for v in session.enabled_calls)
    assert len(session.enabled_calls) == 4


def test_set_reference_authorizer_coerces_falsy_non_bool() -> None:
    """Symmetric to truthy coercion: ``0`` / empty values coerce
    to False."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for falsy in [0, "", [], {}]:
        ok, result = rpc._handle_session_set_reference_authorizer(
            {"enabled": falsy},
        )
        assert ok is True, f"falsy={falsy!r} should coerce to False"

    assert all(v is False for v in session.enabled_calls)
    assert len(session.enabled_calls) == 4


def test_set_reference_authorizer_rejects_missing_key() -> None:
    """Absent ``enabled`` key surfaces as stage='decode' rather
    than silently being treated as ``False`` — protects against a
    daemon-side wrapper typo silently disabling authorization."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_reference_authorizer({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "enabled" in result["error"]
    assert session.enabled_calls == []


def test_set_reference_authorizer_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_set_reference_authorizer(
        {"enabled": True},
    )
    assert ok is False
    assert result["stage"] == "no_host"


def test_set_reference_authorizer_setter_raises_returns_set_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_set = RuntimeError("setter boom")
    _install(rpc, session)

    ok, result = rpc._handle_session_set_reference_authorizer(
        {"enabled": True},
    )
    assert ok is False
    assert result["stage"] == "set"
    assert "setter boom" in result["error"]


def test_set_reference_authorizer_session_missing_method() -> None:
    """Defensive: a session without
    ``set_reference_authorization_enabled`` (rolling-upgrade gap)
    surfaces as stage='missing_method'.  Forward-compat with
    sessions that pre-date §7c step 6.1."""
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits set_reference_authorization_enabled.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_set_reference_authorizer(
        {"enabled": True},
    )
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_set_reference_authorizer() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    env = RequestEnvelope(
        id=99, method="session.set_reference_authorizer",
        args={"enabled": True},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"ok": True}
    assert session.enabled_calls == [True]


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
        target=runner_rpc.serve, name="rpc-refauth-test", daemon=True,
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
async def test_e2e_set_reference_authorizer_round_trip_true() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_set_reference_authorizer(True)

        # Wait for the runner to process (RPC is one-way; we don't
        # get the call result back through some mechanism, but the
        # async wrapper awaited the response so it's done).
        assert session.enabled_calls == [True]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_set_reference_authorizer_round_trip_false() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_set_reference_authorizer(False)

        assert session.enabled_calls == [False]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_set_reference_authorizer_toggle_sequence() -> None:
    """Realistic toggle sequence: enable → disable → enable.
    Each call lands in order; runner-side session sees the
    transitions."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_set_reference_authorizer(True)
        await pair.daemon_client.session_set_reference_authorizer(False)
        await pair.daemon_client.session_set_reference_authorizer(True)

        assert session.enabled_calls == [True, False, True]
    finally:
        await _teardown(pair)
