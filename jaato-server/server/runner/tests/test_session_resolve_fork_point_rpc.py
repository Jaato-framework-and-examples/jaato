"""Tests for the session.resolve_fork_point RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.3.5, FINAL of the
6.6.3 prerequisite handler series).

Resolves a fork-point specifier (after_message /
after_tool_call / after_timestamp) to a message index in the
session's history.  Replaces the pre-§7c daemon-side call at
``server/session_manager.py:4362`` (ResolveForkPointRequest
SDK handler).

Wraps the existing public method
``JaatoSession.resolve_fork_point`` (no missing-method gap —
this is the second of the 5 prerequisites without an
encapsulation cleanup).

Tests pin:

- happy paths for each specifier (after_message /
  after_tool_call / after_timestamp)
- history defaults to session.get_history() when omitted
- explicit history (caller-provided) round-trips
- arg validation: non-int / non-str / non-list types →
  stage="decode"
- no_host / no_session error paths
- resolver raises → stage="resolve"
- non-int return rejected → stage="resolve"
- e2e via daemon-side wrapper round-trip
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient
from shared.plugins.session.serializer import serialize_history
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-fork-point",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in capturing every ``resolve_fork_point`` call +
    optionally serving a default history."""

    def __init__(
        self,
        *,
        default_history: Optional[List[Message]] = None,
        return_value: int = 0,
        raise_on_resolve: Optional[Exception] = None,
        non_int_return: Any = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._default_history = default_history or []
        self._return_value = return_value
        self._raise = raise_on_resolve
        self._non_int = non_int_return

    def get_history(self) -> List[Message]:
        # Returns a copy (matches JaatoSession's defensive shape).
        return list(self._default_history)

    def resolve_fork_point(
        self,
        history: List[Message],
        after_message: Optional[int] = None,
        after_tool_call: Optional[str] = None,
        after_timestamp: Optional[str] = None,
    ) -> Any:
        self.calls.append({
            "history": history,
            "after_message": after_message,
            "after_tool_call": after_tool_call,
            "after_timestamp": after_timestamp,
        })
        if self._raise is not None:
            raise self._raise
        if self._non_int is not None:
            return self._non_int
        return self._return_value


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _user_msg(text: str) -> Message:
    return Message(role=Role.USER, parts=[Part(text=text)])


# ----------------------------------------------------------------------
# Direct handler tests — happy paths per specifier
# ----------------------------------------------------------------------


def test_resolve_fork_point_after_message_specifier() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(return_value=2)
    _install(rpc, session)

    ok, result = rpc._handle_session_resolve_fork_point({
        "after_message": 5,
    })
    assert ok is True
    assert result == {"fork_index": 2}
    assert session.calls[0]["after_message"] == 5
    assert session.calls[0]["after_tool_call"] is None
    assert session.calls[0]["after_timestamp"] is None


def test_resolve_fork_point_after_tool_call_specifier() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(return_value=4)
    _install(rpc, session)

    ok, result = rpc._handle_session_resolve_fork_point({
        "after_tool_call": "tool-call-abc123",
    })
    assert ok is True
    assert result == {"fork_index": 4}
    assert session.calls[0]["after_tool_call"] == "tool-call-abc123"


def test_resolve_fork_point_after_timestamp_specifier() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(return_value=3)
    _install(rpc, session)

    ok, result = rpc._handle_session_resolve_fork_point({
        "after_timestamp": "2024-01-15T10:30:00",
    })
    assert ok is True
    assert result == {"fork_index": 3}
    assert session.calls[0]["after_timestamp"] == "2024-01-15T10:30:00"


def test_resolve_fork_point_no_specifier_returns_last() -> None:
    """Underlying method returns last index when no specifier
    given.  Handler preserves that semantic."""
    rpc = _make_lone_runner()
    session = _FakeSession(return_value=10)
    _install(rpc, session)

    ok, result = rpc._handle_session_resolve_fork_point({})
    assert ok is True
    assert result == {"fork_index": 10}


# ----------------------------------------------------------------------
# History argument: defaults to session.get_history(); explicit override
# ----------------------------------------------------------------------


def test_resolve_fork_point_defaults_to_session_get_history() -> None:
    """When 'history' is omitted, runner uses session.get_history()."""
    rpc = _make_lone_runner()
    default = [_user_msg("turn 1"), _user_msg("turn 2")]
    session = _FakeSession(default_history=default, return_value=1)
    _install(rpc, session)

    ok, _ = rpc._handle_session_resolve_fork_point({"after_message": 1})
    assert ok is True
    # Resolver received the session's default history.
    received_history = session.calls[0]["history"]
    assert len(received_history) == 2
    assert received_history[0].parts[0].text == "turn 1"
    assert received_history[1].parts[0].text == "turn 2"


def test_resolve_fork_point_explicit_history_override() -> None:
    """When 'history' is provided, runner uses it instead of
    session.get_history()."""
    rpc = _make_lone_runner()
    default = [_user_msg("default 1")]
    session = _FakeSession(default_history=default, return_value=0)
    _install(rpc, session)

    explicit = [_user_msg("explicit a"), _user_msg("explicit b")]
    ok, _ = rpc._handle_session_resolve_fork_point({
        "after_message": 0,
        "history": serialize_history(explicit),
    })
    assert ok is True
    received_history = session.calls[0]["history"]
    assert len(received_history) == 2
    assert received_history[0].parts[0].text == "explicit a"
    assert received_history[1].parts[0].text == "explicit b"


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------


def test_resolve_fork_point_rejects_non_int_after_message() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in ["string", 1.5, [1]]:
        ok, result = rpc._handle_session_resolve_fork_point({
            "after_message": bad,
        })
        assert ok is False, f"should reject after_message={bad!r}"
        assert result["stage"] == "decode"


def test_resolve_fork_point_rejects_non_str_after_tool_call() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, [1], {"d": 1}]:
        ok, result = rpc._handle_session_resolve_fork_point({
            "after_tool_call": bad,
        })
        assert ok is False, f"should reject after_tool_call={bad!r}"
        assert result["stage"] == "decode"


def test_resolve_fork_point_rejects_non_str_after_timestamp() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, [1], {"d": 1}]:
        ok, result = rpc._handle_session_resolve_fork_point({
            "after_timestamp": bad,
        })
        assert ok is False, f"should reject after_timestamp={bad!r}"
        assert result["stage"] == "decode"


def test_resolve_fork_point_rejects_non_list_history() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, "string", {"d": 1}]:
        ok, result = rpc._handle_session_resolve_fork_point({
            "history": bad,
        })
        assert ok is False, f"should reject history={bad!r}"
        assert result["stage"] == "decode"


def test_resolve_fork_point_rejects_malformed_history_dict() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_resolve_fork_point({
        "history": [{"parts": [{"type": "text", "text": "hi"}]}],
    })
    assert ok is False
    assert result["stage"] == "decode"
    assert "deserialize failed" in result["error"]


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_resolve_fork_point_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_resolve_fork_point({})
    assert ok is False
    assert result["stage"] == "no_host"


def test_resolve_fork_point_resolver_raises_returns_resolve_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(raise_on_resolve=ValueError("invalid timestamp"))
    _install(rpc, session)

    ok, result = rpc._handle_session_resolve_fork_point({
        "after_timestamp": "garbage",
    })
    assert ok is False
    assert result["stage"] == "resolve"
    assert "invalid timestamp" in result["error"]


def test_resolve_fork_point_non_int_return_rejected() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(non_int_return="not_an_int")
    _install(rpc, session)

    ok, result = rpc._handle_session_resolve_fork_point({})
    assert ok is False
    assert result["stage"] == "resolve"
    assert "expected int" in result["error"]


def test_resolve_fork_point_session_missing_method() -> None:
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits resolve_fork_point.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_resolve_fork_point({})
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_resolve_fork_point() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(return_value=7)
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.resolve_fork_point",
        args={"after_message": 5},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"fork_index": 7}


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
        target=runner_rpc.serve, name="rpc-fork-point-test", daemon=True,
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
async def test_e2e_resolve_fork_point_after_message() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession(return_value=3)
        _install(pair.runner_rpc, session)

        result = await pair.daemon_client.session_resolve_fork_point(
            after_message=5,
        )
        assert result == 3
        assert session.calls[0]["after_message"] == 5
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_resolve_fork_point_with_explicit_history() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession(default_history=[_user_msg("default")], return_value=1)
        _install(pair.runner_rpc, session)

        explicit = [_user_msg("a"), _user_msg("b"), _user_msg("c")]
        result = await pair.daemon_client.session_resolve_fork_point(
            after_message=2,
            history=explicit,
        )
        assert result == 1
        # Runner-side session received the explicit history (not its default).
        received = session.calls[0]["history"]
        assert len(received) == 3
        assert received[0].parts[0].text == "a"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_resolve_fork_point_history_defaults_to_session() -> None:
    pair = await _make_pair()
    try:
        default = [_user_msg("session msg 1"), _user_msg("session msg 2")]
        session = _FakeSession(default_history=default, return_value=1)
        _install(pair.runner_rpc, session)

        # Don't pass history — runner should use session.get_history().
        result = await pair.daemon_client.session_resolve_fork_point(
            after_message=1,
        )
        assert result == 1
        # Resolver received the session's default history.
        received = session.calls[0]["history"]
        assert len(received) == 2
        assert received[0].parts[0].text == "session msg 1"
    finally:
        await _teardown(pair)
