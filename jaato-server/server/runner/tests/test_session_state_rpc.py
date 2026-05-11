"""Tests for the session-state + lifecycle RPC handlers
(Phase 3 §3.3c precursors).

This batch adds four runner-side handlers that the daemon will
dispatch to once the seat-flip lands:

- ``session.get_session_state(key, default)`` — read a single key
- ``session.set_session_state(key, value)`` — write a single key
- ``session.is_running`` — probe whether send_message is in flight
- ``session.request_stop(reason)`` — signal cancellation

All four share a precondition: the runner must be bootstrapped and
the JaatoSession must be configured.  The shared
``_require_ready_session`` helper enforces this with a clean error
envelope when not ready.

Tests pin the contract:

- get/set round-trip a value
- get returns the supplied default for missing keys
- set rejects missing key/value
- get rejects missing key
- both error cleanly when no session bootstrapped (``no_host``)
  and when the session is None (``no_session``)
- set surfaces TypeError from non-JSON-serialisable values as a
  clean ``stage="validate"`` error
- is_running mirrors the session's flag
- request_stop returns whether cancellation was actually issued
"""

from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional

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


def _good_envelope(**overrides: Any) -> SessionInitEnvelope:
    base = dict(
        session_id="sess-state-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )
    base.update(overrides)
    return SessionInitEnvelope(**base)


class _FakeSession:
    """Stand-in for :class:`JaatoSession` exposing only the methods
    these handlers need.  Tests configure the state dict / running
    flag / stop_outcome directly."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.running: bool = False
        self.stop_calls: List[str] = []
        self.stop_outcome: bool = False

    def get_session_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set_session_state(self, key: str, value: Any) -> None:
        # Mirror JaatoSession's TypeError on non-JSON values.
        import json
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"set_session_state({key!r}): value must be JSON-serialisable: {exc}"
            )
        self.state[key] = value

    def is_running(self) -> bool:
        return self.running

    def request_stop(self, reason: str = "") -> bool:
        self.stop_calls.append(reason)
        return self.stop_outcome


def _install_session(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# get_session_state
# ----------------------------------------------------------------------


def test_get_session_state_returns_stored_value() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.state["key1"] = "value1"
    _install_session(rpc, session)

    ok, result = rpc._handle_session_get_state({"key": "key1"})
    assert ok is True
    assert result == {"value": "value1"}


def test_get_session_state_returns_default_for_missing_key() -> None:
    rpc = _make_lone_runner()
    _install_session(rpc, _FakeSession())

    ok, result = rpc._handle_session_get_state({
        "key": "absent",
        "default": "fallback",
    })
    assert ok is True
    assert result == {"value": "fallback"}


def test_get_session_state_default_is_none_when_not_supplied() -> None:
    rpc = _make_lone_runner()
    _install_session(rpc, _FakeSession())

    ok, result = rpc._handle_session_get_state({"key": "absent"})
    assert ok is True
    assert result == {"value": None}


def test_get_session_state_rejects_missing_key() -> None:
    rpc = _make_lone_runner()
    _install_session(rpc, _FakeSession())

    ok, result = rpc._handle_session_get_state({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "missing 'key'" in result["error"]


def test_get_session_state_no_host_returns_error() -> None:
    rpc = _make_lone_runner()  # no _session_host
    ok, result = rpc._handle_session_get_state({"key": "k"})
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_session_state_no_session_returns_error() -> None:
    """Host bootstrapped but session=None (test-stub mode)."""
    rpc = _make_lone_runner()
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=None,
    )
    ok, result = rpc._handle_session_get_state({"key": "k"})
    assert ok is False
    assert result["stage"] == "no_session"


def test_get_session_state_provider_failure_surfaces_clean_error() -> None:
    """If the underlying get_session_state raises, the handler
    must return a clean stage=provider error rather than
    propagating."""
    rpc = _make_lone_runner()

    class _BoomSession:
        def get_session_state(self, key, default=None):
            raise RuntimeError("provider boom")

    _install_session(rpc, _BoomSession())

    ok, result = rpc._handle_session_get_state({"key": "k"})
    assert ok is False
    assert result["stage"] == "provider"
    assert "provider boom" in result["error"]


# ----------------------------------------------------------------------
# set_session_state
# ----------------------------------------------------------------------


def test_set_session_state_writes_value() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install_session(rpc, session)

    ok, result = rpc._handle_session_set_state({
        "key": "k1",
        "value": {"nested": [1, 2, 3]},
    })
    assert ok is True
    assert result == {"ok": True}
    assert session.state["k1"] == {"nested": [1, 2, 3]}


def test_set_session_state_rejects_missing_key() -> None:
    rpc = _make_lone_runner()
    _install_session(rpc, _FakeSession())

    ok, result = rpc._handle_session_set_state({"value": "v"})
    assert ok is False
    assert result["stage"] == "decode"


def test_set_session_state_rejects_missing_value() -> None:
    rpc = _make_lone_runner()
    _install_session(rpc, _FakeSession())

    ok, result = rpc._handle_session_set_state({"key": "k"})
    assert ok is False
    assert result["stage"] == "decode"


def test_set_session_state_accepts_none_value() -> None:
    """``value=None`` is a legitimate write (``"key" present,
    value is null``); the missing-value check must distinguish
    "absent" from "explicitly null"."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install_session(rpc, session)

    ok, result = rpc._handle_session_set_state({"key": "k", "value": None})
    assert ok is True
    assert session.state["k"] is None


def test_set_session_state_non_json_value_returns_validate_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install_session(rpc, session)

    # A set-of-objects is not JSON-serialisable.
    ok, result = rpc._handle_session_set_state({
        "key": "bad",
        "value": {object()},  # set with non-serialisable element
    })
    assert ok is False
    assert result["stage"] == "validate"


def test_set_session_state_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_set_state({"key": "k", "value": 1})
    assert ok is False
    assert result["stage"] == "no_host"


# ----------------------------------------------------------------------
# is_running
# ----------------------------------------------------------------------


def test_is_running_returns_true_when_session_running() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.running = True
    _install_session(rpc, session)

    ok, result = rpc._handle_session_is_running()
    assert ok is True
    assert result == {"running": True}


def test_is_running_returns_false_when_session_idle() -> None:
    rpc = _make_lone_runner()
    _install_session(rpc, _FakeSession())

    ok, result = rpc._handle_session_is_running()
    assert ok is True
    assert result == {"running": False}


def test_is_running_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_is_running()
    assert ok is False
    assert result["stage"] == "no_host"


# ----------------------------------------------------------------------
# request_stop
# ----------------------------------------------------------------------


def test_request_stop_returns_true_when_cancellation_issued() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.stop_outcome = True
    _install_session(rpc, session)

    ok, result = rpc._handle_session_request_stop({"reason": "operator-cancel"})
    assert ok is True
    assert result == {"cancelled": True}
    assert session.stop_calls == ["operator-cancel"]


def test_request_stop_returns_false_when_no_message_running() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.stop_outcome = False
    _install_session(rpc, session)

    ok, result = rpc._handle_session_request_stop({})
    assert ok is True
    assert result == {"cancelled": False}
    # Empty reason passed through.
    assert session.stop_calls == [""]


def test_request_stop_normalizes_non_string_reason() -> None:
    """Defensive: a buggy daemon-side caller passing a non-string
    reason gets normalized to empty rather than crashing."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install_session(rpc, session)

    ok, _result = rpc._handle_session_request_stop({"reason": 42})
    assert ok is True
    assert session.stop_calls == [""]


def test_request_stop_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_request_stop({})
    assert ok is False
    assert result["stage"] == "no_host"


# ----------------------------------------------------------------------
# Dispatch routing (sanity that the new methods are reachable via
# _dispatch_method, not just the direct handler call)
# ----------------------------------------------------------------------


def test_dispatch_routes_get_session_state() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.state["x"] = "y"
    _install_session(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.get_session_state", args={"key": "x"},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"value": "y"}


def test_dispatch_routes_set_session_state() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install_session(rpc, session)

    env = RequestEnvelope(
        id=2, method="session.set_session_state",
        args={"key": "x", "value": 7},
    )
    ok, _ = rpc._dispatch_method(env)
    assert ok is True
    assert session.state["x"] == 7


def test_dispatch_routes_is_running_and_request_stop() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.running = True
    session.stop_outcome = True
    _install_session(rpc, session)

    env_running = RequestEnvelope(
        id=3, method="session.is_running", args={},
    )
    ok, result = rpc._dispatch_method(env_running)
    assert ok is True
    assert result == {"running": True}

    env_stop = RequestEnvelope(
        id=4, method="session.request_stop", args={"reason": "x"},
    )
    ok, result = rpc._dispatch_method(env_stop)
    assert ok is True
    assert result == {"cancelled": True}
