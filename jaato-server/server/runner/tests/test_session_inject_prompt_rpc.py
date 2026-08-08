"""Tests for the session.inject_prompt RPC handler + daemon-side
wrapper (Phase 3 §7c step 6.1, 3/3 — final new RPC handler in
the §7c step 6.1 trio).

Injects a prompt into the runner-side session's message queue.
Replaces the pre-§7c daemon-side call
``self._jaato.get_session().inject_prompt(text, ...)`` at
``server/core.py:3238``.

The wire serializes :class:`shared.message_queue.SourceType` as
its string value (e.g. ``"user"``) to keep the JSON schema
clean; the runner-side handler maps it back to the enum before
calling :meth:`JaatoSession.inject_prompt`.

Tests pin:

- happy path: text + source_id + source_type round-trips
- defaults: source_id / source_type omitted both work
- arg validation: missing/non-str text → stage="decode";
  unknown source_type value → stage="decode" with valid-list
  hint; non-str source_id / source_type → stage="decode"
- no_host / no_session error paths
- inject() raises → stage="inject"
- e2e via daemon-side wrapper: end-to-end round-trip with
  every SourceType value
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, List, Optional, Tuple

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient
from shared.message_queue import SourceType
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-inject",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Minimal stand-in capturing every ``inject_prompt`` call."""

    def __init__(self) -> None:
        # Each entry: (text, source_id, source_type-or-None)
        self.calls: List[Tuple[str, Optional[str], Optional[Any]]] = []
        self.raise_on_inject: Optional[Exception] = None

    def inject_prompt(
        self,
        text: str,
        source_id: Optional[str] = None,
        source_type: Optional[Any] = None,
    ) -> None:
        if self.raise_on_inject is not None:
            raise self.raise_on_inject
        self.calls.append((text, source_id, source_type))


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Direct handler tests — happy path
# ----------------------------------------------------------------------


def test_inject_prompt_happy_path_full_args() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({
        "text": "hello agent",
        "source_id": "main",
        "source_type": "parent",
    })
    assert ok is True
    assert result == {"ok": True}
    assert len(session.calls) == 1
    text, source_id, source_type = session.calls[0]
    assert text == "hello agent"
    assert source_id == "main"
    assert source_type is SourceType.PARENT


def test_inject_prompt_defaults_source_id_source_type() -> None:
    """Both optional args omitted → handler passes ``None``
    through to ``JaatoSession.inject_prompt``, which then applies
    its own ``"unknown"`` / ``SourceType.USER`` defaults."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({"text": "minimal"})
    assert ok is True
    assert result == {"ok": True}
    text, source_id, source_type = session.calls[0]
    assert text == "minimal"
    assert source_id is None
    assert source_type is None


@pytest.mark.parametrize("st_value, expected_enum", [
    ("parent", SourceType.PARENT),
    ("child", SourceType.CHILD),
    ("user", SourceType.USER),
    ("system", SourceType.SYSTEM),
    ("event", SourceType.EVENT),
])
def test_inject_prompt_all_source_type_values(
    st_value: str, expected_enum: SourceType,
) -> None:
    """Every SourceType enum value round-trips through the wire
    schema as its lowercase string and rebuilds to the matching
    enum on the runner side."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({
        "text": f"msg-{st_value}",
        "source_type": st_value,
    })
    assert ok is True
    _, _, source_type = session.calls[0]
    assert source_type is expected_enum


# ----------------------------------------------------------------------
# Direct handler tests — argument validation
# ----------------------------------------------------------------------


def test_inject_prompt_missing_text_returns_decode_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "text" in result["error"]
    assert session.calls == []


def test_inject_prompt_non_str_text_returns_decode_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, None, ["list"], {"d": 1}, b"bytes"]:
        ok, result = rpc._handle_session_inject_prompt({"text": bad})
        assert ok is False, f"text={bad!r} should reject"
        assert result["stage"] == "decode"
    assert session.calls == []


def test_inject_prompt_empty_text_accepted() -> None:
    """An empty string IS a valid str (different from missing key
    or non-str type).  JaatoSession's own logic decides what to
    do with it; the handler doesn't reject empty text."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({"text": ""})
    assert ok is True
    assert session.calls[0][0] == ""


def test_inject_prompt_non_str_source_id_returns_decode_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({
        "text": "hi", "source_id": 42,
    })
    assert ok is False
    assert result["stage"] == "decode"
    assert "source_id" in result["error"]


def test_inject_prompt_non_str_source_type_returns_decode_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({
        "text": "hi", "source_type": 42,
    })
    assert ok is False
    assert result["stage"] == "decode"
    assert "source_type" in result["error"]


def test_inject_prompt_unknown_source_type_returns_decode_error() -> None:
    """Unknown SourceType value → stage='decode' with the valid
    list in the error.  Protects against typos that would
    silently misroute message priority."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({
        "text": "hi", "source_type": "stranger",
    })
    assert ok is False
    assert result["stage"] == "decode"
    # Error message should hint at valid values.
    err = result["error"]
    assert "source_type" in err
    for valid_value in ("parent", "child", "user", "system", "event"):
        assert valid_value in err


# ----------------------------------------------------------------------
# Direct handler tests — error paths
# ----------------------------------------------------------------------


def test_inject_prompt_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_inject_prompt({"text": "hi"})
    assert ok is False
    assert result["stage"] == "no_host"


def test_inject_prompt_inject_raises_returns_inject_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_inject = RuntimeError("queue full")
    _install(rpc, session)

    ok, result = rpc._handle_session_inject_prompt({"text": "hi"})
    assert ok is False
    assert result["stage"] == "inject"
    assert "queue full" in result["error"]


def test_inject_prompt_session_missing_method() -> None:
    """Forward-compat: session without inject_prompt method
    surfaces as stage='missing_method'."""
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits inject_prompt.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_inject_prompt({"text": "hi"})
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_inject_prompt() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.inject_prompt",
        args={"text": "routed!", "source_type": "user"},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"ok": True}
    assert session.calls[0][0] == "routed!"
    assert session.calls[0][2] is SourceType.USER


# ----------------------------------------------------------------------
# E2E daemon-side wrapper
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
        target=runner_rpc.serve, name="rpc-inject-test", daemon=True,
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
async def test_e2e_inject_prompt_full_args() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_inject_prompt(
            "hello via wire",
            source_id="main",
            source_type="parent",
        )

        assert len(session.calls) == 1
        text, source_id, source_type = session.calls[0]
        assert text == "hello via wire"
        assert source_id == "main"
        assert source_type is SourceType.PARENT
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_inject_prompt_minimal_text_only() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_inject_prompt("ping")

        text, source_id, source_type = session.calls[0]
        assert text == "ping"
        assert source_id is None
        assert source_type is None
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_inject_prompt_every_source_type_value() -> None:
    """End-to-end: every SourceType value crosses the wire as a
    string and reconstitutes to the enum runner-side."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        for st in SourceType:
            await pair.daemon_client.session_inject_prompt(
                f"msg-{st.value}", source_type=st.value,
            )

        # 5 SourceType values → 5 calls in order.
        assert len(session.calls) == 5
        for (text, _sid, source_type), expected in zip(
            session.calls, list(SourceType),
        ):
            assert text == f"msg-{expected.value}"
            assert source_type is expected
    finally:
        await _teardown(pair)
