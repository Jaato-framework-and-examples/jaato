"""End-to-end tests for the daemon-side ``RunnerRPCClient`` named-
method wrappers (Phase 3 §3.3c precursors).

Drives each wrapper over a real :func:`socketpair` connecting a
:class:`RunnerRPC` (runner side) to a :class:`RunnerRPCClient`
(daemon side).  Verifies the round-trip: daemon-side wrapper
serializes args, runner-side handler decodes + dispatches to a
fake JaatoSession, response surfaces back through the wrapper's
return type.

Tests pin the wire contract so daemon-side callers can rely on
the named-method API (``client.session_health_check()``,
``client.session_get_state(key, default)``, etc.) without
constructing the raw RPC envelope by hand.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional

import pytest

from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient, RunnerCallError
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Pair fixture (mirrors test_session_bootstrap_rpc.py)
# ----------------------------------------------------------------------


class _Pair:
    def __init__(
        self,
        runner_rpc: RunnerRPC,
        runner_thread: threading.Thread,
        daemon_client: RunnerRPCClient,
    ) -> None:
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair() -> _Pair:
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    runner_rpc = RunnerRPC(runner_sock, _no_executor)
    thread = threading.Thread(
        target=runner_rpc.serve, name="rpc-serve-test", daemon=True,
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


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-e2e-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.running = False
        self.stop_outcome = False
        self.stop_calls: List[str] = []
        self.history: List[Any] = []

    def get_session_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set_session_state(self, key: str, value: Any) -> None:
        import json
        json.dumps(value)  # raises if not serialisable
        self.state[key] = value

    def is_running(self) -> bool:
        return self.running

    def request_stop(self, reason: str = "") -> bool:
        self.stop_calls.append(reason)
        return self.stop_outcome

    def get_history(self) -> List[Any]:
        return list(self.history)

    def get_history_raw(self) -> List[Any]:
        return list(self.history)

    def get_context_usage(self) -> Dict[str, Any]:
        return {
            "model": "test-model",
            "context_limit": 100000,
            "total_tokens": 5,
            "prompt_tokens": 5,
            "output_tokens": 0,
            "turns": 1,
            "percent_used": 0.005,
            "tokens_remaining": 99995,
        }


class _Msg:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


def _install_session(pair: _Pair, session: Any) -> None:
    pair.runner_rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# session_health_check
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_session_health_check_no_host() -> None:
    pair = await _make_pair()
    try:
        result = await pair.daemon_client.session_health_check()
        assert result == {
            "has_host": False,
            "ready": False,
            "session_id": "",
            "tool_count": -1,
        }
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_session_health_check_with_session() -> None:
    pair = await _make_pair()
    try:
        _install_session(pair, _FakeSession())
        result = await pair.daemon_client.session_health_check()
        assert result["has_host"] is True
        assert result["ready"] is True
        assert result["session_id"] == "sess-e2e-1"
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# session_get_state / session_set_state
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_session_set_then_get_state() -> None:
    pair = await _make_pair()
    try:
        _install_session(pair, _FakeSession())
        await pair.daemon_client.session_set_state("k1", {"nested": [1, 2]})
        value = await pair.daemon_client.session_get_state("k1")
        assert value == {"nested": [1, 2]}
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_session_get_state_missing_returns_default() -> None:
    pair = await _make_pair()
    try:
        _install_session(pair, _FakeSession())
        value = await pair.daemon_client.session_get_state(
            "absent", default="fallback",
        )
        assert value == "fallback"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_session_get_state_raises_when_no_host() -> None:
    pair = await _make_pair()
    try:
        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_get_state("k")
        assert "no_host" in str(exc_info.value) or "not bootstrapped" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# session_is_running / session_request_stop
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_session_is_running_returns_session_flag() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.running = True
        _install_session(pair, session)

        result = await pair.daemon_client.session_is_running()
        assert result is True
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_session_request_stop_returns_outcome() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.stop_outcome = True
        _install_session(pair, session)

        result = await pair.daemon_client.session_request_stop("operator-cancel")
        assert result is True
        assert session.stop_calls == ["operator-cancel"]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_session_request_stop_no_running_message() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.stop_outcome = False
        _install_session(pair, session)

        result = await pair.daemon_client.session_request_stop()
        assert result is False
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# session_get_history
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_session_get_history_returns_serialized_messages() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.history = [
            _Msg("user", "hello"),
            _Msg("assistant", "hi"),
        ]
        _install_session(pair, session)

        history = await pair.daemon_client.session_get_history()
        assert history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_session_get_history_empty() -> None:
    pair = await _make_pair()
    try:
        _install_session(pair, _FakeSession())
        history = await pair.daemon_client.session_get_history()
        assert history == []
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# session_get_context_usage
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_session_get_context_usage_returns_dict() -> None:
    pair = await _make_pair()
    try:
        _install_session(pair, _FakeSession())
        usage = await pair.daemon_client.session_get_context_usage()
        assert usage["model"] == "test-model"
        assert usage["context_limit"] == 100000
        assert usage["turns"] == 1
    finally:
        await _teardown(pair)
