"""Tests for the session.restore_turn_accounting RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.1.2, 2/3 of the
6.6.1 trio).

Replaces the runner-side session's per-turn token-usage /
timing list from a SessionState snapshot.  Replaces the pre-§7c
daemon-side private-attr write at
``server/session_manager.py:2558-2559``:

    jaato_session._turn_accounting = list(state.turn_accounting)

Now wraps the public method
``JaatoSession.restore_turn_accounting`` added in §7c step
6.6.1.0 (commit 13ce5939).

Tests pin:

- happy path: turn list round-trips through the wire
- empty list accepted (clean restore of a fresh session)
- arg validation: missing ``turns`` key, non-list value,
  non-dict elements → stage="decode"
- no_host / no_session error paths
- setter raises → stage="set"
- e2e via daemon-side wrapper: end-to-end round-trip
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-restore-turns",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in capturing every ``restore_turn_accounting`` call."""

    def __init__(self) -> None:
        self.calls: List[List[Dict[str, Any]]] = []
        self.raise_on_set: Optional[Exception] = None

    def restore_turn_accounting(self, turns: List[Dict[str, Any]]) -> None:
        if self.raise_on_set is not None:
            raise self.raise_on_set
        self.calls.append(turns)


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _realistic_turns() -> List[Dict[str, Any]]:
    """Realistic per-turn entry shape from JaatoSession's
    accumulator + persistence serializer."""
    return [
        {
            "turn_id": 1,
            "prompt_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "duration_seconds": 1.5,
        },
        {
            "turn_id": 2,
            "prompt_tokens": 1500,
            "output_tokens": 350,
            "total_tokens": 1850,
            "duration_seconds": 2.1,
        },
        {
            "turn_id": 3,
            "prompt_tokens": 2000,
            "output_tokens": 100,
            "total_tokens": 2100,
            "duration_seconds": 0.8,
        },
    ]


# ----------------------------------------------------------------------
# Direct handler tests — happy path
# ----------------------------------------------------------------------


def test_restore_turn_accounting_happy_path() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    turns = _realistic_turns()
    ok, result = rpc._handle_session_restore_turn_accounting({
        "turns": turns,
    })
    assert ok is True
    assert result == {"ok": True}
    assert len(session.calls) == 1
    assert session.calls[0] == turns


def test_restore_turn_accounting_empty_list_accepted() -> None:
    """Empty list is a legitimate restore (no turns yet).
    Matches the daemon caller's existing semantic."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_restore_turn_accounting({"turns": []})
    assert ok is True
    assert result == {"ok": True}
    assert session.calls == [[]]


def test_restore_turn_accounting_preserves_arbitrary_dict_keys() -> None:
    """Turn-accounting entry shape evolves with provider
    integrations (cache hits/misses, thinking-token counts,
    etc.).  Handler must NOT validate per-key schema — pass
    through whatever the dict contains."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    turns = [{
        "turn_id": 1,
        "prompt_tokens": 1000,
        "cache_creation_tokens": 500,  # Anthropic-specific
        "cache_read_tokens": 200,
        "thinking_tokens": 150,        # extended-thinking specific
        "model": "claude-opus-4",      # provenance
        "provider": "anthropic",
    }]
    ok, _ = rpc._handle_session_restore_turn_accounting({"turns": turns})
    assert ok is True
    received = session.calls[0][0]
    assert received["cache_creation_tokens"] == 500
    assert received["thinking_tokens"] == 150
    assert received["provider"] == "anthropic"


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------


def test_restore_turn_accounting_rejects_missing_turns_key() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_restore_turn_accounting({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "turns" in result["error"]
    assert session.calls == []


def test_restore_turn_accounting_rejects_non_list_turns() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, None, "string", {"dict": True}]:
        ok, result = rpc._handle_session_restore_turn_accounting({"turns": bad})
        assert ok is False, f"should reject turns={bad!r}"
        assert result["stage"] == "decode"
    assert session.calls == []


def test_restore_turn_accounting_rejects_non_dict_element() -> None:
    """Per-element decode failure surfaces as stage='decode'
    rather than partial-restore.  Catches wire-corruption /
    version-skew issues at the boundary."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_restore_turn_accounting({
        "turns": [{"prompt_tokens": 100}, "not_a_dict", {"prompt_tokens": 200}],
    })
    assert ok is False
    assert result["stage"] == "decode"
    assert "turns[1]" in result["error"]
    assert session.calls == []


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_restore_turn_accounting_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_restore_turn_accounting({"turns": []})
    assert ok is False
    assert result["stage"] == "no_host"


def test_restore_turn_accounting_setter_raises_returns_set_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_set = RuntimeError("setter boom")
    _install(rpc, session)

    ok, result = rpc._handle_session_restore_turn_accounting({"turns": []})
    assert ok is False
    assert result["stage"] == "set"
    assert "setter boom" in result["error"]


def test_restore_turn_accounting_session_missing_method() -> None:
    """Forward-compat: session without restore_turn_accounting
    method (rolling-upgrade gap; pre-§7c-step-6.6.1.0)
    surfaces as stage='missing_method'."""
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits restore_turn_accounting.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_restore_turn_accounting({"turns": []})
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_restore_turn_accounting() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    turns = [{"turn_id": 1, "prompt_tokens": 100}]
    env = RequestEnvelope(
        id=1, method="session.restore_turn_accounting",
        args={"turns": turns},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"ok": True}
    assert session.calls[0] == turns


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
        target=runner_rpc.serve, name="rpc-restore-turns-test", daemon=True,
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
async def test_e2e_restore_turn_accounting_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        turns = _realistic_turns()
        await pair.daemon_client.session_restore_turn_accounting(turns)

        assert len(session.calls) == 1
        received = session.calls[0]
        assert len(received) == 3
        # Bit-equality on every key the daemon's persistence path
        # cares about.
        for i, (got, want) in enumerate(zip(received, turns)):
            assert got == want, f"turn[{i}] diverged: got={got!r} want={want!r}"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_restore_turn_accounting_empty_list() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_restore_turn_accounting([])

        assert session.calls == [[]]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_restore_turn_accounting_caller_mutation_isolated() -> None:
    """Daemon-side caller mutating the list AFTER the RPC must
    not affect the runner's view (the wrapper takes a copy via
    list(turns))."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        turns = [{"turn_id": 1, "prompt_tokens": 100}]
        await pair.daemon_client.session_restore_turn_accounting(turns)

        # Mutate the caller's list AFTER the RPC.
        turns.append({"turn_id": 999, "prompt_tokens": 999_999})

        # The runner-side received the original 1-element list.
        assert len(session.calls[0]) == 1
        assert session.calls[0][0]["turn_id"] == 1
    finally:
        await _teardown(pair)
