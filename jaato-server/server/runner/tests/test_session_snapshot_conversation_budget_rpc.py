"""Tests for the session.snapshot_conversation_budget RPC handler
+ daemon-side wrapper (Phase 3 §7c step 6.6.3.2, 2/5 of the
6.6.3 prerequisite handler series).

Returns the runner-side session's CONVERSATION instruction-
budget snapshot for persistence-save.  Inverse of
``session.restore_conversation_budget`` (6.6.1.3).  Replaces
the pre-§7c daemon-side reach at
``server/session_manager.py:2986``:

    jaato_session.instruction_budget.get_conversation_snapshot()

Now wraps the public method
``JaatoSession.snapshot_conversation_budget`` added in §7c
step 6.6.3.0.

Tests pin:

- happy path: snapshot dict round-trips through the wire
- None handling: budget=None / no conversation entry → None
- nested children preserved (recursive SourceEntry tree)
- deep-copy isolation: daemon mutation doesn't propagate
- defensive: missing snapshot method, non-dict return,
  getter raises → typed error envelopes
- no_host / no_session error paths
- e2e via daemon-side wrapper: dict + None round-trip
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, Optional

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
        session_id="sess-snap-conv-budget",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in for JaatoSession exposing
    ``snapshot_conversation_budget``."""

    def __init__(
        self,
        *,
        snapshot_dict: Optional[Dict[str, Any]] = None,
        raise_on_get: Optional[Exception] = None,
        non_dict_return: Any = None,
    ) -> None:
        self._snapshot_dict = snapshot_dict
        self._raise = raise_on_get
        self._non_dict = non_dict_return
        self.call_count = 0

    def snapshot_conversation_budget(self) -> Any:
        self.call_count += 1
        if self._raise is not None:
            raise self._raise
        if self._non_dict is not None:
            return self._non_dict
        return self._snapshot_dict


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


def test_snapshot_conversation_budget_happy_path() -> None:
    rpc = _make_lone_runner()
    snapshot = _realistic_snapshot()
    session = _FakeSession(snapshot_dict=snapshot)
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_conversation_budget()
    assert ok is True
    assert result["snapshot"] == snapshot


def test_snapshot_conversation_budget_none_when_no_budget() -> None:
    """Pre-configure / no-conversation-entry → ``{"snapshot": None}``.
    Daemon caller skips persistence-save when None."""
    rpc = _make_lone_runner()
    session = _FakeSession(snapshot_dict=None)
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_conversation_budget()
    assert ok is True
    assert result == {"snapshot": None}


def test_snapshot_conversation_budget_preserves_nested_children() -> None:
    """Recursive SourceEntry children round-trip untouched."""
    rpc = _make_lone_runner()
    snapshot = {
        "source": "conversation",
        "tokens": 200,
        "children": {
            "turn_1": {
                "source": "conversation",
                "tokens": 100,
                "children": {
                    "subturn_1a": {
                        "source": "conversation",
                        "tokens": 50,
                        "children": {},
                    },
                },
            },
        },
    }
    session = _FakeSession(snapshot_dict=snapshot)
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_conversation_budget()
    assert ok is True
    assert result["snapshot"]["children"]["turn_1"]["children"]["subturn_1a"]["tokens"] == 50


def test_snapshot_conversation_budget_deep_copies_to_isolate_state() -> None:
    """Daemon-side mutation of returned dict / nested children
    must NOT propagate back to runner state."""
    rpc = _make_lone_runner()
    snapshot = _realistic_snapshot()
    session = _FakeSession(snapshot_dict=snapshot)
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_conversation_budget()
    assert ok is True

    # Mutate returned + nested entry.
    result["snapshot"]["tokens"] = 999_999
    result["snapshot"]["children"]["turn_1"]["tokens"] = 999_999

    # Original (runner-side budget's view) is unmutated.
    assert snapshot["tokens"] == 1234
    assert snapshot["children"]["turn_1"]["tokens"] == 500


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_snapshot_conversation_budget_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_snapshot_conversation_budget()
    assert ok is False
    assert result["stage"] == "no_host"


def test_snapshot_conversation_budget_getter_raises_returns_read_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(raise_on_get=RuntimeError("budget boom"))
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_conversation_budget()
    assert ok is False
    assert result["stage"] == "read"
    assert "budget boom" in result["error"]


def test_snapshot_conversation_budget_non_dict_return_rejected() -> None:
    """Defensive: non-dict + non-None return surfaces as
    stage='read' rather than crashing daemon-side JSON encoding."""
    rpc = _make_lone_runner()
    session = _FakeSession(non_dict_return="not_a_dict")
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_conversation_budget()
    assert ok is False
    assert result["stage"] == "read"
    assert "expected dict" in result["error"]


def test_snapshot_conversation_budget_session_missing_method() -> None:
    """Forward-compat: session without snapshot_conversation_budget
    method (rolling-upgrade gap; pre-§7c-step-6.6.3.0)
    surfaces as stage='missing_method'."""
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits snapshot_conversation_budget.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_snapshot_conversation_budget()
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_snapshot_conversation_budget() -> None:
    rpc = _make_lone_runner()
    snapshot = _realistic_snapshot()
    session = _FakeSession(snapshot_dict=snapshot)
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.snapshot_conversation_budget", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result["snapshot"] == snapshot


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
        target=runner_rpc.serve, name="rpc-snap-budget-test", daemon=True,
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
async def test_e2e_snapshot_conversation_budget_round_trip() -> None:
    pair = await _make_pair()
    try:
        snapshot = _realistic_snapshot()
        session = _FakeSession(snapshot_dict=snapshot)
        _install(pair.runner_rpc, session)

        result = await pair.daemon_client.session_snapshot_conversation_budget()
        assert result == snapshot
        # Bit-equality on nested children.
        assert result["children"]["turn_1"]["tokens"] == 500
        assert result["children"]["turn_2"]["tokens"] == 734
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_snapshot_conversation_budget_returns_none() -> None:
    """Pre-configure case: budget=None → wrapper returns Python None."""
    pair = await _make_pair()
    try:
        session = _FakeSession(snapshot_dict=None)
        _install(pair.runner_rpc, session)

        result = await pair.daemon_client.session_snapshot_conversation_budget()
        assert result is None
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_snapshot_then_restore_round_trip() -> None:
    """Full save → restore cycle: snapshot from session A, restore
    into session B.  The two new RPCs (6.6.1.3 + 6.6.3.2) are
    inverse pairs — pin that they preserve the snapshot bit-for-bit
    across the wire in both directions."""
    pair = await _make_pair()
    try:
        snapshot = _realistic_snapshot()

        # Session A produces the snapshot.
        session_a = _FakeSession(snapshot_dict=snapshot)
        _install(pair.runner_rpc, session_a)
        saved = await pair.daemon_client.session_snapshot_conversation_budget()

        # Session B receives the restore.
        class _RestoreReceiver:
            def __init__(self):
                self.received = None

            def restore_conversation_budget(self, snap):
                self.received = snap

        session_b = _RestoreReceiver()
        _install(pair.runner_rpc, session_b)
        await pair.daemon_client.session_restore_conversation_budget(saved)

        # The dict that left session A is bit-equal to what arrived
        # at session B.
        assert session_b.received == snapshot
    finally:
        await _teardown(pair)
