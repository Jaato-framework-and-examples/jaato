"""Tests for the session.set_initial_history RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.1.1, 1/3 of the
6.6.1 trio).

Seeds the runner-side ``JaatoSession`` with replayed
conversation history.  Replaces the pre-§7c daemon-side call
``jaato_session.set_initial_history(initial_history)`` at
``server/session_manager.py:2130`` (the
``create_headless_session`` path that seeds ``initial_history``
before the first ``send_message``).

Wire shape: ``{"messages": [<dict>, ...]}`` where each dict is
a serialized :class:`Message` from
``shared.plugins.session.serializer.serialize_message``.  The
runner deserializes via
:func:`shared.plugins.session.serializer.deserialize_history` —
the same code disk persistence uses, so the wire-shape is the
round-trippable JSON format already proven in
``test_serializer.py``.

Tests pin:

- happy path: message list round-trips through the wire
- empty list accepted (clean seed of an empty session is a
  legitimate no-op)
- arg validation: missing ``messages`` key, non-list value,
  malformed Part dicts → stage="decode"
- no_host / no_session error paths
- session.set_initial_history raises (idle/empty-history
  guard violation) → stage="set"
- e2e via daemon-side wrapper: serialize + RPC + deserialize
  round-trip
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
        session_id="sess-init-history",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Minimal stand-in capturing every ``set_initial_history``
    call."""

    def __init__(self) -> None:
        self.calls: List[List[Message]] = []
        self.raise_on_set: Optional[Exception] = None

    def set_initial_history(self, messages: List[Message]) -> None:
        if self.raise_on_set is not None:
            raise self.raise_on_set
        self.calls.append(messages)


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _user_msg(text: str) -> Message:
    return Message(role=Role.USER, parts=[Part(text=text)])


def _model_msg(text: str) -> Message:
    return Message(role=Role.MODEL, parts=[Part(text=text)])


# ----------------------------------------------------------------------
# Direct handler tests — happy path
# ----------------------------------------------------------------------


def test_set_initial_history_happy_path_two_messages() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    history = [_user_msg("hello"), _model_msg("hi back")]
    serialized = {"messages": serialize_history(history)}

    ok, result = rpc._handle_session_set_initial_history(serialized)
    assert ok is True
    assert result == {"ok": True}
    assert len(session.calls) == 1
    received = session.calls[0]
    assert len(received) == 2
    assert received[0].role is Role.USER
    assert received[0].parts[0].text == "hello"
    assert received[1].role is Role.MODEL
    assert received[1].parts[0].text == "hi back"


def test_set_initial_history_empty_list_accepted() -> None:
    """An empty list is a legitimate seed (no-op, but the
    handler shouldn't reject it)."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_initial_history({"messages": []})
    assert ok is True
    assert result == {"ok": True}
    assert session.calls == [[]]


def test_set_initial_history_preserves_provenance_fields() -> None:
    """Messages carrying model + provider provenance fields
    round-trip — cross-provider history must survive the wire."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    msg = Message(
        role=Role.MODEL,
        parts=[Part(text="from claude")],
        model="claude-sonnet-4-6",
        provider="anthropic",
    )
    serialized = {"messages": serialize_history([msg])}

    ok, _ = rpc._handle_session_set_initial_history(serialized)
    assert ok is True
    received = session.calls[0][0]
    assert received.model == "claude-sonnet-4-6"
    assert received.provider == "anthropic"


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------


def test_set_initial_history_rejects_missing_messages_key() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_set_initial_history({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "messages" in result["error"]
    assert session.calls == []


def test_set_initial_history_rejects_non_list_messages() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, None, "string", {"dict": True}]:
        ok, result = rpc._handle_session_set_initial_history({"messages": bad})
        assert ok is False, f"should reject messages={bad!r}"
        assert result["stage"] == "decode"
    assert session.calls == []


def test_set_initial_history_rejects_malformed_message_dict() -> None:
    """Per-element decode failure surfaces as stage='decode'
    rather than partial-seed.  Catches malformed dicts coming
    over the wire (missing 'role', unknown part types, etc.)."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    # Malformed: missing 'role' field will raise during deserialize.
    ok, result = rpc._handle_session_set_initial_history({
        "messages": [{"parts": [{"type": "text", "text": "hi"}]}],
    })
    assert ok is False
    assert result["stage"] == "decode"
    assert "deserialize failed" in result["error"]
    assert session.calls == []


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_set_initial_history_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_set_initial_history({"messages": []})
    assert ok is False
    assert result["stage"] == "no_host"


def test_set_initial_history_setter_raises_returns_set_error() -> None:
    """JaatoSession.set_initial_history raises RuntimeError when
    the session is not idle or its history is non-empty.  The
    daemon's persistence-restore path never violates this, but
    the handler defensively wraps it as stage='set'."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_set = RuntimeError("session is mid-turn")
    _install(rpc, session)

    ok, result = rpc._handle_session_set_initial_history({
        "messages": serialize_history([_user_msg("hi")]),
    })
    assert ok is False
    assert result["stage"] == "set"
    assert "mid-turn" in result["error"]


def test_set_initial_history_session_missing_method() -> None:
    """Forward-compat: session without set_initial_history method
    surfaces as stage='missing_method'."""
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits set_initial_history.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_set_initial_history({"messages": []})
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_set_initial_history() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    history = [_user_msg("routed!")]
    env = RequestEnvelope(
        id=1, method="session.set_initial_history",
        args={"messages": serialize_history(history)},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"ok": True}
    assert session.calls[0][0].parts[0].text == "routed!"


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
        target=runner_rpc.serve, name="rpc-init-history-test", daemon=True,
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
async def test_e2e_set_initial_history_round_trip() -> None:
    """End-to-end: daemon serializes via serialize_history, RPC
    sends across the wire, runner deserializes back to Message
    instances with bit-equal field values."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        history = [
            _user_msg("first user message"),
            _model_msg("first model reply"),
            _user_msg("second user message"),
        ]

        await pair.daemon_client.session_set_initial_history(history)

        assert len(session.calls) == 1
        received = session.calls[0]
        assert len(received) == 3
        assert received[0].role is Role.USER
        assert received[0].parts[0].text == "first user message"
        assert received[1].role is Role.MODEL
        assert received[1].parts[0].text == "first model reply"
        assert received[2].role is Role.USER
        assert received[2].parts[0].text == "second user message"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_set_initial_history_empty_history() -> None:
    """Empty-history seed via the wire is accepted as a no-op."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_set_initial_history([])

        assert session.calls == [[]]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_set_initial_history_provenance_fields() -> None:
    """Cross-provider history with model + provider provenance
    round-trips end-to-end."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        history = [
            Message(
                role=Role.MODEL,
                parts=[Part(text="from gemini")],
                model="gemini-2.5-flash",
                provider="google_genai",
            ),
            Message(
                role=Role.MODEL,
                parts=[Part(text="from claude")],
                model="claude-sonnet-4-6",
                provider="anthropic",
            ),
        ]

        await pair.daemon_client.session_set_initial_history(history)

        received = session.calls[0]
        assert received[0].model == "gemini-2.5-flash"
        assert received[0].provider == "google_genai"
        assert received[1].model == "claude-sonnet-4-6"
        assert received[1].provider == "anthropic"
    finally:
        await _teardown(pair)
