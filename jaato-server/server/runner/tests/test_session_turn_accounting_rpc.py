"""Tests for the session.get_turn_accounting RPC handler
(Phase 3 §3.3c precursor).

Read-only handler returning the per-turn token-usage / timing
list daemon-side telemetry + persistence paths use.

Tests pin: happy path, empty list, no_host error, read failure
→ stage=read error, non-list return rejection, dispatch routing.
"""

from __future__ import annotations

import socket
from typing import Any, Dict, List

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
        session_id="sess-turns-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(self) -> None:
        self.turns: List[Dict[str, Any]] = []

    def get_turn_accounting(self) -> List[Dict[str, Any]]:
        return list(self.turns)


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_get_turn_accounting_returns_list() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.turns = [
        {"turn": 1, "tokens": 100, "duration_seconds": 1.5},
        {"turn": 2, "tokens": 250, "duration_seconds": 2.1},
    ]
    _install(rpc, session)

    ok, result = rpc._handle_session_get_turn_accounting()
    assert ok is True
    assert result == {"turns": session.turns}


def test_get_turn_accounting_empty() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_get_turn_accounting()
    assert ok is True
    assert result == {"turns": []}


def test_get_turn_accounting_returns_copies_not_references() -> None:
    """Each turn dict is copied — daemon-side mutation doesn't
    propagate back into session state."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.turns = [{"turn": 1, "tokens": 100}]
    _install(rpc, session)

    ok, result = rpc._handle_session_get_turn_accounting()
    assert ok is True
    result["turns"][0]["tokens"] = 99999
    assert session.turns[0]["tokens"] == 100  # unchanged


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_get_turn_accounting_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_turn_accounting()
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_turn_accounting_session_raises_returns_read_error() -> None:
    rpc = _make_lone_runner()

    class _BoomSession:
        def get_turn_accounting(self):
            raise RuntimeError("turns boom")

    _install(rpc, _BoomSession())

    ok, result = rpc._handle_session_get_turn_accounting()
    assert ok is False
    assert result["stage"] == "read"
    assert "turns boom" in result["error"]


def test_get_turn_accounting_non_list_return_rejected() -> None:
    rpc = _make_lone_runner()

    class _BadSession:
        def get_turn_accounting(self):
            return {"not": "a list"}

    _install(rpc, _BadSession())

    ok, result = rpc._handle_session_get_turn_accounting()
    assert ok is False
    assert result["stage"] == "read"
    assert "expected list" in result["error"]


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_get_turn_accounting() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.turns = [{"turn": 5}]
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.get_turn_accounting", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"turns": [{"turn": 5}]}
