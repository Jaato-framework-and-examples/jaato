"""Tests for the session.append_history_message RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.3.1, 1/5 of the
6.6.3 prerequisite handler trio).

Appends a single message to the runner-side session's history.
Replaces the pre-§7c daemon-side get-modify-reset dance at
``server/session_manager.py:2855`` (interrupted-tool-call
recovery path):

    current_history = jaato_session.get_history()
    current_history.append(synthetic_message)
    jaato_session.reset_session(current_history)

Now wraps the public method
``JaatoSession.append_history_message`` added in §7c step
6.6.3.0.

Wire shape: ``{"message": <dict>}`` reusing the existing
``shared.plugins.session.serializer.serialize_message`` /
``deserialize_message`` round-trip.

Tests pin:

- happy path: single message round-trip
- provenance fields preserved (model + provider on Message)
- arg validation: missing key, non-dict, malformed inner dict
  → stage="decode"
- no_host / no_session error paths
- appender raises → stage="set"
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
from shared.plugins.session.serializer import serialize_message
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-append-history",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in capturing every ``append_history_message`` call."""

    def __init__(self) -> None:
        self.calls: List[Message] = []
        self.raise_on_set: Optional[Exception] = None

    def append_history_message(self, message: Message) -> None:
        if self.raise_on_set is not None:
            raise self.raise_on_set
        self.calls.append(message)


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


def test_append_history_message_happy_path() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    msg = _user_msg("synthetic tool message")
    ok, result = rpc._handle_session_append_history_message({
        "message": serialize_message(msg),
    })
    assert ok is True
    assert result == {"ok": True}
    assert len(session.calls) == 1
    assert session.calls[0].role is Role.USER
    assert session.calls[0].parts[0].text == "synthetic tool message"


def test_append_history_message_preserves_provenance() -> None:
    """Cross-provider provenance fields (model + provider) on
    the appended message round-trip via the wire."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    msg = Message(
        role=Role.MODEL,
        parts=[Part(text="recovered tool result")],
        model="claude-sonnet-4-6",
        provider="anthropic",
    )
    ok, _ = rpc._handle_session_append_history_message({
        "message": serialize_message(msg),
    })
    assert ok is True
    received = session.calls[0]
    assert received.model == "claude-sonnet-4-6"
    assert received.provider == "anthropic"


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------


def test_append_history_message_rejects_missing_key() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_append_history_message({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "message" in result["error"]
    assert session.calls == []


def test_append_history_message_rejects_non_dict_message() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, "string", ["list"]]:
        ok, result = rpc._handle_session_append_history_message({
            "message": bad,
        })
        assert ok is False, f"should reject message={bad!r}"
        assert result["stage"] == "decode"
    assert session.calls == []


def test_append_history_message_rejects_malformed_dict() -> None:
    """Per-element decode failure surfaces as stage='decode'
    rather than partial-append."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    # Missing 'role' field will raise during deserialize.
    ok, result = rpc._handle_session_append_history_message({
        "message": {"parts": [{"type": "text", "text": "hi"}]},
    })
    assert ok is False
    assert result["stage"] == "decode"
    assert "deserialize failed" in result["error"]
    assert session.calls == []


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_append_history_message_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    msg = serialize_message(_user_msg("hi"))
    ok, result = rpc._handle_session_append_history_message({"message": msg})
    assert ok is False
    assert result["stage"] == "no_host"


def test_append_history_message_appender_raises_returns_set_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_set = RuntimeError("history full")
    _install(rpc, session)

    msg = serialize_message(_user_msg("hi"))
    ok, result = rpc._handle_session_append_history_message({"message": msg})
    assert ok is False
    assert result["stage"] == "set"
    assert "history full" in result["error"]


def test_append_history_message_session_missing_method() -> None:
    """Forward-compat: session without append_history_message
    method (rolling-upgrade gap) surfaces as stage='missing_method'."""
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits append_history_message.
        pass

    _install(rpc, _OldSession())

    msg = serialize_message(_user_msg("hi"))
    ok, result = rpc._handle_session_append_history_message({"message": msg})
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_append_history_message() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    msg = _user_msg("routed!")
    env = RequestEnvelope(
        id=1, method="session.append_history_message",
        args={"message": serialize_message(msg)},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"ok": True}
    assert session.calls[0].parts[0].text == "routed!"


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
        target=runner_rpc.serve, name="rpc-append-test", daemon=True,
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
async def test_e2e_append_history_message_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        msg = _user_msg("end-to-end synthetic message")
        await pair.daemon_client.session_append_history_message(msg)

        assert len(session.calls) == 1
        assert session.calls[0].role is Role.USER
        assert session.calls[0].parts[0].text == "end-to-end synthetic message"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_append_history_message_provenance_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        msg = Message(
            role=Role.MODEL,
            parts=[Part(text="provenance-tagged synthetic")],
            model="claude-sonnet-4-6",
            provider="anthropic",
        )
        await pair.daemon_client.session_append_history_message(msg)

        received = session.calls[0]
        assert received.model == "claude-sonnet-4-6"
        assert received.provider == "anthropic"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_append_history_message_multiple_appends() -> None:
    """Multiple sequential appends each fire as separate calls
    (the runner-side method is idempotent in shape)."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        for i in range(3):
            await pair.daemon_client.session_append_history_message(
                _user_msg(f"message {i}"),
            )

        assert len(session.calls) == 3
        for i, m in enumerate(session.calls):
            assert m.parts[0].text == f"message {i}"
    finally:
        await _teardown(pair)
