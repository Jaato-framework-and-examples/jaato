"""Tests for the session.replay_messages RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.3.4, 4/5 of the
6.6.3 prerequisite handler series).

Runs a one-shot completion against an arbitrary message list
and returns the model's text response.  Replaces the pre-§7c
daemon-side call at ``server/session_manager.py:4338``.

Wraps the existing public method
``JaatoSession.replay_messages`` (already at
jaato_session.py:8252; no missing-method gap — this is the
ONLY one of the 5 prerequisite handlers without an
encapsulation cleanup needed).

Wire shape: ``{"messages": [<dict>, ...], "timeout": float}``
reusing the existing ``serialize_history`` /
``deserialize_history`` round-trip.  Returns
``{"response_text": str}``.

Tests pin:

- happy path: messages round-trip + response_text returned
- timeout argument round-trips
- empty message list accepted
- arg validation: missing/non-list messages, non-positive
  timeout, malformed message dicts → stage="decode"
- no_host / no_session error paths
- replay raises (provider error) → stage="replay"
- non-str response coerced (defensive against custom sessions)
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
        session_id="sess-replay",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in capturing every ``replay_messages`` call."""

    def __init__(
        self,
        *,
        response: str = "model response",
        raise_on_replay: Optional[Exception] = None,
        non_str_response: Any = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._response = response
        self._raise = raise_on_replay
        self._non_str = non_str_response

    def replay_messages(
        self,
        messages: List[Message],
        *,
        timeout: float = 120.0,
    ) -> Any:
        self.calls.append({"messages": messages, "timeout": timeout})
        if self._raise is not None:
            raise self._raise
        if self._non_str is not None:
            return self._non_str
        return self._response


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _user_msg(text: str) -> Message:
    return Message(role=Role.USER, parts=[Part(text=text)])


# ----------------------------------------------------------------------
# Direct handler tests — happy path
# ----------------------------------------------------------------------


def test_replay_messages_happy_path() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(response="replay response")
    _install(rpc, session)

    history = [_user_msg("question 1")]
    ok, result = rpc._handle_session_replay_messages({
        "messages": serialize_history(history),
    })
    assert ok is True
    assert result == {"response_text": "replay response"}
    assert len(session.calls) == 1
    received = session.calls[0]["messages"]
    assert len(received) == 1
    assert received[0].parts[0].text == "question 1"


def test_replay_messages_threads_timeout() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    history = [_user_msg("hi")]
    ok, _ = rpc._handle_session_replay_messages({
        "messages": serialize_history(history),
        "timeout": 30.0,
    })
    assert ok is True
    assert session.calls[0]["timeout"] == 30.0


def test_replay_messages_default_timeout() -> None:
    """Omitted timeout uses the handler's default (120.0)."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, _ = rpc._handle_session_replay_messages({
        "messages": serialize_history([_user_msg("hi")]),
    })
    assert ok is True
    assert session.calls[0]["timeout"] == 120.0


def test_replay_messages_empty_list_accepted() -> None:
    """Empty messages list is technically valid (provider may
    reject; that's a domain concern not a handler concern)."""
    rpc = _make_lone_runner()
    session = _FakeSession(response="ack")
    _install(rpc, session)

    ok, result = rpc._handle_session_replay_messages({"messages": []})
    assert ok is True
    assert result == {"response_text": "ack"}


def test_replay_messages_non_str_response_coerced() -> None:
    """Defensive: if a custom session returns non-str, the
    handler coerces to str rather than crashing JSON encoding."""
    rpc = _make_lone_runner()
    session = _FakeSession(non_str_response=42)
    _install(rpc, session)

    ok, result = rpc._handle_session_replay_messages({
        "messages": serialize_history([_user_msg("hi")]),
    })
    assert ok is True
    assert result == {"response_text": "42"}


def test_replay_messages_none_response_coerced_to_empty() -> None:
    """None response coerces to empty string."""
    rpc = _make_lone_runner()
    session = _FakeSession(non_str_response=None, response=None)
    _install(rpc, session)

    ok, result = rpc._handle_session_replay_messages({
        "messages": serialize_history([_user_msg("hi")]),
    })
    assert ok is True
    assert result == {"response_text": ""}


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------


def test_replay_messages_rejects_missing_messages_key() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_replay_messages({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "messages" in result["error"]


def test_replay_messages_rejects_non_list_messages() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, None, "string", {"d": 1}]:
        ok, result = rpc._handle_session_replay_messages({"messages": bad})
        assert ok is False, f"should reject messages={bad!r}"
        assert result["stage"] == "decode"


def test_replay_messages_rejects_non_positive_timeout() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [0, -1, -1.5, "string"]:
        ok, result = rpc._handle_session_replay_messages({
            "messages": [],
            "timeout": bad,
        })
        assert ok is False, f"should reject timeout={bad!r}"
        assert result["stage"] == "decode"
        assert "timeout" in result["error"]


def test_replay_messages_rejects_malformed_message() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    # Missing 'role' will fail deserialize.
    ok, result = rpc._handle_session_replay_messages({
        "messages": [{"parts": [{"type": "text", "text": "hi"}]}],
    })
    assert ok is False
    assert result["stage"] == "decode"
    assert "deserialize failed" in result["error"]


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_replay_messages_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_replay_messages({"messages": []})
    assert ok is False
    assert result["stage"] == "no_host"


def test_replay_messages_replay_raises_returns_replay_error() -> None:
    """Provider errors during replay surface as stage='replay'
    (distinct from stage='set' since this is an active
    completion, not a pure setter)."""
    rpc = _make_lone_runner()
    session = _FakeSession(raise_on_replay=RuntimeError("provider boom"))
    _install(rpc, session)

    ok, result = rpc._handle_session_replay_messages({
        "messages": serialize_history([_user_msg("hi")]),
    })
    assert ok is False
    assert result["stage"] == "replay"
    assert "provider boom" in result["error"]


def test_replay_messages_session_missing_method() -> None:
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits replay_messages.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_replay_messages({"messages": []})
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_replay_messages() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(response="routed!")
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.replay_messages",
        args={"messages": serialize_history([_user_msg("hi")])},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"response_text": "routed!"}


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
        target=runner_rpc.serve, name="rpc-replay-test", daemon=True,
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
async def test_e2e_replay_messages_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession(response="end-to-end response")
        _install(pair.runner_rpc, session)

        history = [
            _user_msg("first"),
            Message(role=Role.MODEL, parts=[Part(text="acknowledged")]),
            _user_msg("second"),
        ]
        result = await pair.daemon_client.session_replay_messages(history)

        assert result == "end-to-end response"
        # All three messages reached the runner-side method.
        received = session.calls[0]["messages"]
        assert len(received) == 3
        assert received[0].parts[0].text == "first"
        assert received[1].role is Role.MODEL
        assert received[2].parts[0].text == "second"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_replay_messages_threads_replay_timeout() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_replay_messages(
            [_user_msg("hi")], replay_timeout=45.0,
        )
        assert session.calls[0]["timeout"] == 45.0
    finally:
        await _teardown(pair)
