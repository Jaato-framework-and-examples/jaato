"""Tests for the session.snapshot_instruction_budget RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.1, 2/3).

Reads the runner-side ``JaatoSession``'s ``InstructionBudget``
snapshot for the daemon's ``emit_current_state`` call site.
Replaces the pre-§7c daemon-side read
``self._jaato.get_session().instruction_budget.snapshot()`` at
``server/core.py:1091``.

Tests pin:

- happy path: handler returns the snapshot dict via RPC
- None handling: budget=None returns ``{"snapshot": None}``
- defensive contract: missing snapshot() method, non-dict
  return, snapshot() raises → typed error envelopes
- deep-copy isolation: daemon-side mutation doesn't propagate
  back to runner state
- no_host / no_session error paths
- e2e via daemon-side wrapper: dict round-trip + None case
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
        session_id="sess-budget",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeBudget:
    """Stand-in for InstructionBudget exposing snapshot()."""

    def __init__(
        self,
        *,
        snapshot_dict: Optional[Dict[str, Any]] = None,
        raise_on_snapshot: Optional[Exception] = None,
        snapshot_returns_non_dict: Any = None,
    ) -> None:
        self._snapshot_dict = snapshot_dict
        self._raise = raise_on_snapshot
        self._non_dict = snapshot_returns_non_dict
        self.call_count = 0

    def snapshot(self) -> Any:
        self.call_count += 1
        if self._raise is not None:
            raise self._raise
        if self._non_dict is not None:
            return self._non_dict
        return self._snapshot_dict


class _FakeSession:
    """Minimal stand-in with the ``instruction_budget`` property."""

    def __init__(self, budget: Any) -> None:
        self._budget = budget

    @property
    def instruction_budget(self) -> Any:
        return self._budget


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _realistic_snapshot() -> Dict[str, Any]:
    """A realistic InstructionBudget.snapshot() shape with the
    keys the daemon's emit_current_state caller reads."""
    return {
        "session_id": "sess-budget",
        "agent_id": "main",
        "agent_type": "main",
        "context_limit": 200_000,
        "total_tokens": 1234,
        "gc_eligible_tokens": 800,
        "locked_tokens": 434,
        "preservable_tokens": 100,
        "utilization_percent": 0.62,
        "available_tokens": 198_766,
        "gc_headroom_percent": 79.38,
        "entries": {
            "system": {"tokens": 200, "items": []},
            "plugin": {"tokens": 234, "items": []},
            "conversation": {"tokens": 800, "items": []},
        },
    }


# ----------------------------------------------------------------------
# Direct handler tests
# ----------------------------------------------------------------------


def test_snapshot_instruction_budget_happy_path() -> None:
    rpc = _make_lone_runner()
    snapshot = _realistic_snapshot()
    session = _FakeSession(_FakeBudget(snapshot_dict=snapshot))
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_instruction_budget()
    assert ok is True
    assert result["snapshot"] == snapshot


def test_snapshot_instruction_budget_none_when_budget_absent() -> None:
    """``session.instruction_budget = None`` → ``{"snapshot": None}``.

    Daemon caller treats None as "skip emit", matching pre-§7c
    ``if session.instruction_budget:`` guard."""
    rpc = _make_lone_runner()
    session = _FakeSession(None)
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_instruction_budget()
    assert ok is True
    assert result == {"snapshot": None}


def test_snapshot_instruction_budget_session_missing_property() -> None:
    """Forward-compat: a JaatoSession variant without
    ``instruction_budget`` property surfaces as snapshot=None
    (treated like budget=None) rather than AttributeError."""
    rpc = _make_lone_runner()

    class _OldSession:
        # Deliberately omits instruction_budget property.
        pass

    _install(rpc, _OldSession())

    ok, result = rpc._handle_session_snapshot_instruction_budget()
    assert ok is True
    assert result == {"snapshot": None}


def test_snapshot_instruction_budget_deep_copies_to_isolate_state() -> None:
    """Daemon-side mutation of the returned dict (or its nested
    ``entries`` sub-dict) must NOT propagate back to runner state.
    Prevents subtle state-corruption bugs."""
    rpc = _make_lone_runner()
    snapshot = _realistic_snapshot()
    session = _FakeSession(_FakeBudget(snapshot_dict=snapshot))
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_instruction_budget()
    assert ok is True

    # Mutate the returned dict + a nested entry.
    result["snapshot"]["total_tokens"] = 999_999
    result["snapshot"]["entries"]["system"]["tokens"] = 999_999

    # Original (the source of the runner-side budget's view) is
    # unmutated.
    assert snapshot["total_tokens"] == 1234
    assert snapshot["entries"]["system"]["tokens"] == 200


def test_snapshot_instruction_budget_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_snapshot_instruction_budget()
    assert ok is False
    assert result["stage"] == "no_host"


def test_snapshot_instruction_budget_snapshot_raises_returns_read_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(_FakeBudget(
        raise_on_snapshot=RuntimeError("snapshot boom"),
    ))
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_instruction_budget()
    assert ok is False
    assert result["stage"] == "read"
    assert "snapshot boom" in result["error"]


def test_snapshot_instruction_budget_non_dict_return_rejected() -> None:
    """Defensive: a budget whose snapshot() returns a non-dict
    surfaces as stage='read' rather than crashing daemon-side
    JSON encoding."""
    rpc = _make_lone_runner()
    session = _FakeSession(_FakeBudget(
        snapshot_returns_non_dict="not_a_dict",
    ))
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_instruction_budget()
    assert ok is False
    assert result["stage"] == "read"
    assert "expected dict" in result["error"]


def test_snapshot_instruction_budget_missing_method() -> None:
    """A budget object without a snapshot() method (rolling-upgrade
    gap) surfaces as stage='missing_method'."""
    rpc = _make_lone_runner()

    class _BudgetWithoutSnapshot:
        # Deliberately omits snapshot().
        pass

    session = _FakeSession(_BudgetWithoutSnapshot())
    _install(rpc, session)

    ok, result = rpc._handle_session_snapshot_instruction_budget()
    assert ok is False
    assert result["stage"] == "missing_method"


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_snapshot_instruction_budget() -> None:
    rpc = _make_lone_runner()
    snapshot = _realistic_snapshot()
    session = _FakeSession(_FakeBudget(snapshot_dict=snapshot))
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.snapshot_instruction_budget", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result["snapshot"] == snapshot


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
        target=runner_rpc.serve, name="rpc-budget-test", daemon=True,
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
async def test_e2e_snapshot_instruction_budget_round_trip() -> None:
    pair = await _make_pair()
    try:
        snapshot = _realistic_snapshot()
        session = _FakeSession(_FakeBudget(snapshot_dict=snapshot))
        _install(pair.runner_rpc, session)

        result = await pair.daemon_client.session_snapshot_instruction_budget()
        assert result == snapshot

        # Toolbar-relevant keys round-trip with bit-equality.
        for key in (
            "session_id", "agent_id", "agent_type",
            "context_limit", "total_tokens", "utilization_percent",
            "available_tokens",
        ):
            assert result[key] == snapshot[key], (
                f"field {key!r} diverged across RPC: "
                f"direct={snapshot[key]!r} via_rpc={result[key]!r}"
            )
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_snapshot_instruction_budget_returns_none() -> None:
    """Pre-configure case: budget=None → wrapper returns Python
    None.  Daemon caller's ``if budget_snapshot:`` guard skips
    the emit."""
    pair = await _make_pair()
    try:
        session = _FakeSession(None)
        _install(pair.runner_rpc, session)

        result = await pair.daemon_client.session_snapshot_instruction_budget()
        assert result is None
    finally:
        await _teardown(pair)
