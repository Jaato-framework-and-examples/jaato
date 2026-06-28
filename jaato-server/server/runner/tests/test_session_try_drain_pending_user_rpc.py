"""Tests for the session.try_drain_pending_user RPC handler +
daemon-side wrapper + the underlying JaatoSession method.

Multi-turn deadlock fix.  Root cause (confirmed via a 5-point diag
trace, session 20260628_201512): ``turn.completed`` reaches the client
BEFORE the daemon's model-thread ``finally`` clears ``_model_running``
(~12ms wind-down window).  A client that sends its next turn the instant
it sees ``turn.completed`` deterministically arrives while
``_model_running`` is still True, so the daemon gate forwards that send as
an ``inject_prompt``; the runner-side session — idle, with its per-RPC
continuation callback already restored to None — takes the queue/else
branch, leaving the message with no drainer.  The turn never runs.

The fix: after every runner-tier turn the daemon ``finally`` calls
``session.try_drain_pending_user`` (this RPC), atomically pops any pending
high-priority (USER/PARENT/SYSTEM) message, and starts a fresh turn with
it — mirroring the ``_pending_continuation`` drain.

Tests pin:
- handler happy paths: text returned when a message is queued; None when
  empty; None while a turn is running (don't steal from an active turn)
- pop semantics: the message is removed (not re-returned)
- no_host / missing_method (rolling-upgrade) / underlying-raise error paths
- dispatch routing
- e2e via the daemon-side async + threadsafe wrappers
- the real JaatoSession method exists and implements the contract
"""

from __future__ import annotations

import asyncio
import socket
import threading
import types
from typing import Any, Dict, List, Optional

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient, RunnerCallError
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-drain-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in mirroring JaatoSession.try_drain_pending_user semantics:
    pop the first pending high-priority message unless a turn is running."""

    def __init__(
        self,
        *,
        pending: Optional[List[str]] = None,
        is_running: bool = False,
        raise_on_call: Optional[Exception] = None,
    ) -> None:
        self._pending = list(pending or [])
        self._is_running = is_running
        self._raise = raise_on_call
        self.calls = 0

    def try_drain_pending_user(self) -> Optional[str]:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        if self._is_running:
            return None
        if not self._pending:
            return None
        return self._pending.pop(0)


class _SessionWithoutMethod:
    """Stand-in for an old JaatoSession (rolling-upgrade) without the
    public method — handler must surface stage="missing_method"."""

    pass


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Direct handler tests
# ----------------------------------------------------------------------


def test_drain_returns_queued_text_and_pops() -> None:
    """A high-priority message is queued → handler returns it; a second
    call returns None (popped, not re-returned)."""
    rpc = _make_lone_runner()
    session = _FakeSession(pending=["Say B."])
    _install(rpc, session)

    ok, result = rpc._handle_session_try_drain_pending_user()
    assert ok is True
    assert result == {"text": "Say B."}

    ok2, result2 = rpc._handle_session_try_drain_pending_user()
    assert ok2 is True
    assert result2 == {"text": None}
    assert session.calls == 2


def test_drain_empty_returns_none() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(pending=[]))

    ok, result = rpc._handle_session_try_drain_pending_user()
    assert ok is True
    assert result == {"text": None}


def test_drain_while_running_returns_none() -> None:
    """A turn is running → don't steal its queued message; the running
    turn drains the queue itself mid-turn."""
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(pending=["Say B."], is_running=True))

    ok, result = rpc._handle_session_try_drain_pending_user()
    assert ok is True
    assert result == {"text": None}


def test_drain_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_try_drain_pending_user()
    assert ok is False
    assert result["stage"] == "no_host"


def test_drain_missing_method_returns_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _SessionWithoutMethod())

    ok, result = rpc._handle_session_try_drain_pending_user()
    assert ok is False
    assert result["stage"] == "missing_method"


def test_drain_underlying_raise_returns_call_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(raise_on_call=RuntimeError("boom")))

    ok, result = rpc._handle_session_try_drain_pending_user()
    assert ok is False
    assert result["stage"] == "call"
    assert "boom" in result["error"]


def test_dispatch_routes_session_try_drain_pending_user() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(pending=["hi"]))

    env = RequestEnvelope(
        id=1, method="session.try_drain_pending_user", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"text": "hi"}


# ----------------------------------------------------------------------
# E2E: daemon-side wrapper round-trip
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
        target=runner_rpc.serve, name="rpc-drain-test", daemon=True,
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
async def test_e2e_drain_round_trip_text() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _FakeSession(pending=["Say B."]))
        text = await pair.daemon_client.session_try_drain_pending_user()
        assert text == "Say B."
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_drain_round_trip_none() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _FakeSession(pending=[]))
        text = await pair.daemon_client.session_try_drain_pending_user()
        assert text is None
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_drain_threadsafe_from_thread() -> None:
    """The threadsafe wrapper is the path the daemon's model-thread
    ``finally`` uses — pin it works from a background thread."""
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _FakeSession(pending=["Say B."]))
        result_holder: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                result_holder["value"] = (
                    pair.daemon_client
                    .session_try_drain_pending_user_threadsafe()
                )
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        for _ in range(50):
            if "value" in result_holder or "error" in result_holder:
                break
            await asyncio.sleep(0.05)
        t.join(timeout=2)

        assert "error" not in result_holder, result_holder.get("error")
        assert result_holder["value"] == "Say B."
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_drain_missing_method_raises() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _SessionWithoutMethod())
        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_try_drain_pending_user()
        assert "try_drain_pending_user" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# Real JaatoSession: the public method exists and implements the contract
# ----------------------------------------------------------------------


def test_jaato_session_has_public_method() -> None:
    """Pin the missing-method gap is closed — the runner-side handler's
    ``getattr(session, 'try_drain_pending_user', None)`` finds an actual
    method on real JaatoSession."""
    from shared.jaato_session import JaatoSession
    assert callable(getattr(JaatoSession, "try_drain_pending_user", None))


def _drain(running: bool, texts: List[str], injected: List[str]):
    """Invoke the REAL JaatoSession.try_drain_pending_user against a
    minimal duck-typed object carrying a real MessageQueue — exercises the
    actual logic (is_running guard, has_parent_messages, pop, the
    on_prompt_injected notify) without standing up a full session."""
    from shared.jaato_session import JaatoSession
    from shared.message_queue import MessageQueue, SourceType

    q = MessageQueue()
    for t in texts:
        q.put(t, "user", SourceType.USER)
    obj = types.SimpleNamespace(
        _is_running=running,
        _message_queue=q,
        _on_prompt_injected=lambda text: injected.append(text),
    )
    return JaatoSession.try_drain_pending_user(obj)


def test_real_method_returns_queued_user_message() -> None:
    injected: List[str] = []
    result = _drain(running=False, texts=["Say B."], injected=injected)
    assert result == "Say B."
    # The on_prompt_injected hook fired for UI/trace visibility.
    assert injected == ["Say B."]


def test_real_method_empty_queue_returns_none() -> None:
    injected: List[str] = []
    assert _drain(running=False, texts=[], injected=injected) is None
    assert injected == []


def test_real_method_running_returns_none_and_does_not_pop() -> None:
    """While a turn is running, return None without touching the queue."""
    from shared.message_queue import MessageQueue, SourceType
    from shared.jaato_session import JaatoSession

    q = MessageQueue()
    q.put("Say B.", "user", SourceType.USER)
    obj = types.SimpleNamespace(
        _is_running=True, _message_queue=q, _on_prompt_injected=None,
    )
    assert JaatoSession.try_drain_pending_user(obj) is None
    # Not popped — still queued for the running turn / next drain.
    assert q.has_parent_messages() is True
