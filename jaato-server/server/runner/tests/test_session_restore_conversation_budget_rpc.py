"""Tests for the session.restore_conversation_budget RPC handler
+ daemon-side wrapper (Phase 3 §7c step 6.6.1.3, 3/3 of the
6.6.1 trio).

Restores the runner-side session's CONVERSATION instruction-
budget entry from a SessionState snapshot.  Replaces the pre-§7c
daemon-side reach at ``server/session_manager.py:2592-2593``:

    jaato_session.instruction_budget.restore_conversation_from_snapshot(
        state.budget_state)

Now wraps the public method
``JaatoSession.restore_conversation_budget`` added in §7c step
6.6.1.0 (commit 13ce5939).

Tests pin:

- happy path: snapshot dict round-trips through the wire
- empty dict accepted (underlying method documented as no-op
  for empty input)
- arg validation: missing ``snapshot`` key, non-dict value
  → stage="decode"
- no_host / no_session error paths
- setter raises (malformed snapshot structure) → stage="set"
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
        session_id="sess-restore-budget",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in capturing every ``restore_conversation_budget`` call."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.raise_on_set: Optional[Exception] = None

    def restore_conversation_budget(self, snapshot: Dict[str, Any]) -> None:
        if self.raise_on_set is not None:
            raise self.raise_on_set
        self.calls.append(snapshot)


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _realistic_snapshot() -> Dict[str, Any]:
    """Realistic SourceEntry.to_dict() shape from the persistence
    serializer's ``budget_state`` field."""
    return {
        "source": "conversation",
        "tokens": 1234,
        "gc_policy": "incremental",
        "children": {
            "turn_1": {
                "source": "conversation",
                "tokens": 500,
                "gc_policy": "incremental",
                "children": {},
            },
            "turn_2": {
                "source": "conversation",
                "tokens": 734,
                "gc_policy": "incremental",
                "children": {},
            },
        },
    }


# ----------------------------------------------------------------------
# Direct handler tests — happy path
# ----------------------------------------------------------------------


def test_restore_conversation_budget_happy_path() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    snapshot = _realistic_snapshot()
    ok, result = rpc._handle_session_restore_conversation_budget({
        "snapshot": snapshot,
    })
    assert ok is True
    assert result == {"ok": True}
    assert len(session.calls) == 1
    assert session.calls[0] == snapshot


def test_restore_conversation_budget_empty_dict_accepted() -> None:
    """Empty snapshot is permitted — the underlying
    InstructionBudget.restore_conversation_from_snapshot is
    documented as no-op for empty input
    (instruction_budget.py:407 ``if not snapshot: return``)."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_restore_conversation_budget({
        "snapshot": {},
    })
    assert ok is True
    assert result == {"ok": True}
    assert session.calls == [{}]


def test_restore_conversation_budget_preserves_nested_children() -> None:
    """SourceEntry tree round-trips (recursive ``children`` dict
    structure must survive the wire untouched)."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    snapshot = {
        "source": "conversation",
        "tokens": 100,
        "gc_policy": "incremental",
        "children": {
            "turn_1": {
                "source": "conversation",
                "tokens": 50,
                "children": {
                    "subturn_1a": {
                        "source": "conversation",
                        "tokens": 25,
                        "children": {},
                    },
                },
            },
        },
    }
    ok, _ = rpc._handle_session_restore_conversation_budget({"snapshot": snapshot})
    assert ok is True
    received = session.calls[0]
    assert received["children"]["turn_1"]["children"]["subturn_1a"]["tokens"] == 25


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------


def test_restore_conversation_budget_rejects_missing_snapshot_key() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_restore_conversation_budget({})
    assert ok is False
    assert result["stage"] == "decode"
    assert "snapshot" in result["error"]
    assert session.calls == []


def test_restore_conversation_budget_rejects_non_dict_snapshot() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    for bad in [42, None, "string", ["list"]]:
        ok, result = rpc._handle_session_restore_conversation_budget({
            "snapshot": bad,
        })
        assert ok is False, f"should reject snapshot={bad!r}"
        assert result["stage"] == "decode"
    assert session.calls == []


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_restore_conversation_budget_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_restore_conversation_budget({
        "snapshot": {},
    })
    assert ok is False
    assert result["stage"] == "no_host"


def test_restore_conversation_budget_setter_raises_returns_set_error() -> None:
    """Malformed snapshot structure (e.g. unknown gc_policy
    enum value) surfaces as stage='set'."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_set = ValueError("invalid gc_policy enum")
    _install(rpc, session)

    ok, result = rpc._handle_session_restore_conversation_budget({
        "snapshot": {"source": "conversation"},
    })
    assert ok is False
    assert result["stage"] == "set"
    assert "invalid gc_policy" in result["error"]


def test_restore_conversation_budget_session_missing_method() -> None:
    """Forward-compat: session without
    restore_conversation_budget method (rolling-upgrade gap;
    pre-§7c-step-6.6.1.0) surfaces as stage='missing_method'."""
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits restore_conversation_budget.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_restore_conversation_budget({
        "snapshot": {},
    })
    assert ok is False
    assert result["stage"] == "missing_method"


def test_restore_conversation_budget_noop_when_session_has_no_budget() -> None:
    """When the session's restore_conversation_budget is itself
    no-op (because instruction_budget is None pre-configure),
    the handler still returns success.  The public method's
    no-op-on-None contract bubbles up cleanly."""
    rpc = _make_lone_runner()

    class _NoopSession:
        def __init__(self):
            self.calls = []

        def restore_conversation_budget(self, snapshot):
            # Simulates the public method's no-op-when-None contract.
            self.calls.append(snapshot)

    session = _NoopSession()
    _install(rpc, session)

    ok, result = rpc._handle_session_restore_conversation_budget({
        "snapshot": {"tokens": 100},
    })
    assert ok is True
    assert result == {"ok": True}
    # Method WAS called (handler doesn't second-guess the no-op).
    assert session.calls == [{"tokens": 100}]


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_restore_conversation_budget() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)

    snapshot = {"source": "conversation", "tokens": 42}
    env = RequestEnvelope(
        id=1, method="session.restore_conversation_budget",
        args={"snapshot": snapshot},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"ok": True}
    assert session.calls[0] == snapshot


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
        target=runner_rpc.serve, name="rpc-restore-budget-test", daemon=True,
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
async def test_e2e_restore_conversation_budget_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        snapshot = _realistic_snapshot()
        await pair.daemon_client.session_restore_conversation_budget(snapshot)

        assert len(session.calls) == 1
        received = session.calls[0]
        assert received == snapshot
        # Bit-equality on nested children.
        assert received["children"]["turn_1"]["tokens"] == 500
        assert received["children"]["turn_2"]["tokens"] == 734
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_restore_conversation_budget_empty() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        await pair.daemon_client.session_restore_conversation_budget({})

        assert session.calls == [{}]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_restore_conversation_budget_caller_mutation_isolated() -> None:
    """Daemon-side caller mutating the snapshot AFTER the RPC
    must not affect the runner's view (the wrapper takes a copy
    via dict(snapshot))."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        _install(pair.runner_rpc, session)

        snapshot = {"tokens": 100, "source": "conversation"}
        await pair.daemon_client.session_restore_conversation_budget(snapshot)

        # Mutate caller's dict AFTER the RPC.
        snapshot["tokens"] = 999_999
        snapshot["injected"] = "should_not_appear"

        # Runner-side received the original.
        assert session.calls[0]["tokens"] == 100
        assert "injected" not in session.calls[0]
    finally:
        await _teardown(pair)
