"""Tests for the session-config write RPC handlers
(Phase 3 §3.3c precursors).

Two write-only setters daemon-side will dispatch to once the
seat-flip migrates ``client.set_terminal_width`` /
``client.set_streaming_enabled`` through the runner-RPC surface:

- ``session.set_terminal_width(width: int)``
- ``session.set_streaming_enabled(enabled: bool)``

Tests pin:

- happy path: setter called with the right value, returns
  ``{"ok": True}``
- arg validation: width must be positive int; enabled must be
  present (None rejected); enabled is bool-coerced for non-bool
  truthy values
- no_host / no_session error paths
- setter crash → clean stage=set error
- daemon-side wrapper validates width client-side too (raises
  ValueError before any RPC call)
"""

from __future__ import annotations

import socket
from typing import Any, List

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-cfg-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(self) -> None:
        self.terminal_width: int = 0
        self.streaming_enabled: bool = True
        self.set_terminal_width_calls: List[int] = []
        self.set_streaming_calls: List[bool] = []

    def set_terminal_width(self, width: int) -> None:
        self.set_terminal_width_calls.append(width)
        self.terminal_width = width

    def set_streaming_enabled(self, enabled: bool) -> None:
        self.set_streaming_calls.append(enabled)
        self.streaming_enabled = enabled


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# set_terminal_width
# ----------------------------------------------------------------------


def test_set_terminal_width_happy_path() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_terminal_width({"width": 120})
    assert ok is True
    assert result == {"ok": True}
    assert session.set_terminal_width_calls == [120]


def test_set_terminal_width_rejects_zero() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_set_terminal_width({"width": 0})
    assert ok is False
    assert result["stage"] == "decode"


def test_set_terminal_width_rejects_negative() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_set_terminal_width({"width": -5})
    assert ok is False
    assert result["stage"] == "decode"


def test_set_terminal_width_rejects_non_int() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    for bad in ["80", 80.5, None, [], {}]:
        ok, result = rpc._handle_session_set_terminal_width({"width": bad})
        assert ok is False, f"should reject width={bad!r}"
        assert result["stage"] == "decode"


def test_set_terminal_width_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_set_terminal_width({"width": 80})
    assert ok is False
    assert result["stage"] == "no_host"


def test_set_terminal_width_setter_crash_returns_set_error() -> None:
    rpc = _make_lone_runner()

    class _BoomSession:
        def set_terminal_width(self, width: int) -> None:
            raise RuntimeError("setter boom")

    _install(rpc, _BoomSession())

    ok, result = rpc._handle_session_set_terminal_width({"width": 80})
    assert ok is False
    assert result["stage"] == "set"
    assert "setter boom" in result["error"]


# ----------------------------------------------------------------------
# set_streaming_enabled
# ----------------------------------------------------------------------


def test_set_streaming_enabled_true() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_streaming_enabled({"enabled": True})
    assert ok is True
    assert session.set_streaming_calls == [True]


def test_set_streaming_enabled_false() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, _ = rpc._handle_session_set_streaming_enabled({"enabled": False})
    assert ok is True
    assert session.set_streaming_calls == [False]


def test_set_streaming_enabled_truthy_int_coerced_to_bool() -> None:
    """Defensive: daemon-side callers passing ``enabled: 1`` (e.g.
    from a JSON config that didn't use Python's ``True``) get
    coerced rather than crashing."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, _ = rpc._handle_session_set_streaming_enabled({"enabled": 1})
    assert ok is True
    assert session.set_streaming_calls == [True]


def test_set_streaming_enabled_rejects_missing_arg() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_set_streaming_enabled({})
    assert ok is False
    assert result["stage"] == "decode"


def test_set_streaming_enabled_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_set_streaming_enabled({"enabled": True})
    assert ok is False
    assert result["stage"] == "no_host"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_set_terminal_width() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.set_terminal_width", args={"width": 100},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert session.set_terminal_width_calls == [100]


def test_dispatch_routes_set_streaming_enabled() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    env = RequestEnvelope(
        id=2, method="session.set_streaming_enabled",
        args={"enabled": False},
    )
    ok, _ = rpc._dispatch_method(env)
    assert ok is True
    assert session.set_streaming_calls == [False]


# ----------------------------------------------------------------------
# Daemon-side wrapper client-side validation
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrapper_set_terminal_width_validates_client_side() -> None:
    """Daemon-side wrapper raises ValueError on bad width before
    making any RPC call — fail-fast for typos."""
    from server.runner_rpc_client import RunnerRPCClient

    # No need for a real socket — the validation runs before the
    # call.
    client = RunnerRPCClient.__new__(RunnerRPCClient)
    with pytest.raises(ValueError, match="positive int"):
        await client.session_set_terminal_width(0)
    with pytest.raises(ValueError, match="positive int"):
        await client.session_set_terminal_width(-1)
    with pytest.raises(ValueError, match="positive int"):
        await client.session_set_terminal_width("80")  # type: ignore[arg-type]
