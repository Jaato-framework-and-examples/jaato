"""Phase 2 cascade-sharing — end-to-end teardown gate.

This is the Phase 2 sign-off test per ``docs/design/runner-cascade-sharing.md`` §6:

  > Sign-off gate: behavior validated end-to-end with a stub plugin
  > that tracks reuse hits

The test simulates the daemon-side teardown path without spawning a real
runner subprocess.  It uses stubs for:

  - ``RunnerRPCClient`` — captures ``session_end_threadsafe`` calls + their
    timing; returns canned result dicts (success / errors / exception).
  - ``PoolManager`` — real ``PoolManager`` instance with a real
    ``acquire_slot`` / ``return_slot_after_session`` / sweep, but the
    template fork is mocked so no real subprocess is forked.

Coverage:

  1. Happy path — pool-served session, session_end returns errors=[],
     slot returns to pool with cascade affinity preserved.
  2. Errors path — session_end returns errors, slot does NOT return to
     pool (cold teardown path engaged).
  3. RPC failure path — session_end raises, slot does NOT return to pool.
  4. Cold-spawn path — spawned.pool_slot is None, no session_end RPC
     called, existing teardown path runs.
  5. Two-session reuse — first session ends successfully, second session
     acquires the SAME slot via cascade affinity (reuse hit telemetry).
"""

from __future__ import annotations

import socket
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from server.runner_pool import PoolManager, PoolSlot


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------


class _StubRunnerRPC:
    """Mimics RunnerRPCClient enough for shutdown() to call session_end +
    session_shutdown + close.

    Configure ``session_end_result`` / ``session_end_raises`` per-test.
    """

    def __init__(
        self,
        *,
        session_end_result: Optional[Dict[str, Any]] = None,
        session_end_raises: Optional[Exception] = None,
    ) -> None:
        self.session_end_result = session_end_result
        self.session_end_raises = session_end_raises
        self.session_end_calls = 0
        self.session_shutdown_calls = 0
        self.close_calls = 0
        # JaatoServer.shutdown reaches into rpc._loop.is_running() — give
        # it a fake loop that says "not running" so the asyncio.close
        # path is skipped (the close path is exercised separately).
        self._loop = MagicMock()
        self._loop.is_running.return_value = False

    def session_end_threadsafe(self, *, timeout: float = 10.0):
        self.session_end_calls += 1
        if self.session_end_raises is not None:
            raise self.session_end_raises
        return self.session_end_result

    def session_shutdown_threadsafe(self, *, timeout: float = 5.0):
        self.session_shutdown_calls += 1
        return "stub-session-id"

    async def close(self):
        self.close_calls += 1


class _StubServer:
    """Minimal JaatoServer surrogate carrying just the fields shutdown()
    reads.  Avoids the real construction cost (provider connect, plugin
    discovery, etc.) — we only care about the cascade-teardown branch.
    """

    def __init__(
        self,
        rpc: Optional[_StubRunnerRPC],
        spawned: Optional[Any],
        pool_manager: Optional[PoolManager],
    ) -> None:
        self._runner_rpc = rpc
        self._spawned_runner = spawned
        self._pool_manager_ref = pool_manager

    def _invoke_shutdown_branch(self) -> None:
        """Mirror the cascade-teardown branch of JaatoServer.shutdown.

        Reproduces the relevant code so the test can exercise it without
        instantiating a full JaatoServer (which requires env_file +
        provider + ...).  Kept in sync with core.py's shutdown body —
        if that drifts, this test should drift with it.
        """
        rpc = self._runner_rpc
        spawned = self._spawned_runner
        pool_manager = self._pool_manager_ref
        self._runner_rpc = None
        self._spawned_runner = None
        self._pool_manager_ref = None

        pool_slot = getattr(spawned, "pool_slot", None) if spawned else None
        cascade_returned = False
        if rpc is not None and pool_slot is not None and pool_manager is not None:
            session_end = getattr(rpc, "session_end_threadsafe", None)
            if callable(session_end):
                try:
                    result = session_end(timeout=10.0)
                    errors = result.get("errors") if isinstance(result, dict) else None
                    if errors == []:
                        pool_manager.return_slot_after_session(pool_slot)
                        cascade_returned = True
                except Exception:  # noqa: BLE001
                    pass

        if cascade_returned:
            return

        # Cold close path — call session_shutdown if available.
        if rpc is not None:
            shutdown_method = getattr(rpc, "session_shutdown_threadsafe", None)
            if callable(shutdown_method):
                try:
                    shutdown_method(timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass


def _stub_spawned(pool_slot: Optional[PoolSlot] = None) -> Any:
    """Build a SpawnedRunner-shaped object with the pool_slot field."""
    spawned = MagicMock()
    spawned.pool_slot = pool_slot
    spawned.pid = pool_slot.pid if pool_slot else 9999
    return spawned


def _make_pool() -> PoolManager:
    """Build a real PoolManager with a mocked template (no fork)."""
    # target_size=8: the constructor starts no threads, so 0 was never
    # what kept them off — and 0 means "pool disabled", which now also
    # means "every returned slot is over capacity".
    return PoolManager(template_manager=MagicMock(), target_size=8)


# ======================================================================
# Path 1: Happy path — session_end returns errors=[]; slot pools back
# ======================================================================


class TestCascadeTeardownHappyPath:
    def test_slot_returned_to_pool_when_session_end_succeeds(self) -> None:
        pool = _make_pool()
        slot = PoolSlot(
            pid=1234, sock=MagicMock(), cascade_id="cascade-X",
        )
        rpc = _StubRunnerRPC(session_end_result={
            "plugins_reset": 5,
            "errors": [],
        })
        server = _StubServer(rpc, _stub_spawned(slot), pool)

        server._invoke_shutdown_branch()

        # Slot back in pool, affinity preserved, timestamp stamped.
        assert pool.idle_count() == 1
        returned = pool._idle_slots[0]
        assert returned.pid == 1234
        assert returned.cascade_id == "cascade-X"
        assert returned.last_session_end_ts is not None
        # session_end called; session_shutdown / close did NOT fire.
        assert rpc.session_end_calls == 1
        assert rpc.session_shutdown_calls == 0
        assert rpc.close_calls == 0

    def test_returned_slot_is_reused_for_same_cascade(self) -> None:
        """The actual cascade-sharing payoff: second session of the same
        cascade reuses the slot."""
        pool = _make_pool()
        slot = PoolSlot(
            pid=1234, sock=MagicMock(), cascade_id="cascade-X",
        )
        rpc = _StubRunnerRPC(session_end_result={
            "plugins_reset": 5, "errors": [],
        })
        server = _StubServer(rpc, _stub_spawned(slot), pool)

        # Session 1 ends → pool gets slot back.
        server._invoke_shutdown_branch()
        assert pool.idle_count() == 1

        # Session 2 (same cascade) acquires.
        slot2 = pool.acquire_slot(cascade_driver_id="cascade-X")
        assert slot2 is not None
        assert slot2.pid == 1234  # same physical slot
        # Telemetry: one reuse HIT.
        counters = pool.get_telemetry()
        assert counters["cascade_slot_reuse_hits_total"] == 1


# ======================================================================
# Path 2: session_end returns errors → slot NOT returned
# ======================================================================


class TestCascadeTeardownErrorsPath:
    def test_slot_not_pooled_when_errors_present(self) -> None:
        pool = _make_pool()
        slot = PoolSlot(
            pid=1234, sock=MagicMock(), cascade_id="cascade-X",
        )
        rpc = _StubRunnerRPC(session_end_result={
            "plugins_reset": 3,
            "errors": ["plugin-a: RuntimeError: boom!"],
        })
        server = _StubServer(rpc, _stub_spawned(slot), pool)

        server._invoke_shutdown_branch()

        # Slot is NOT in pool — cold teardown engaged.
        assert pool.idle_count() == 0
        assert rpc.session_end_calls == 1
        # Cold path fired (session_shutdown called).
        assert rpc.session_shutdown_calls == 1


# ======================================================================
# Path 3: session_end RPC raises → slot NOT returned
# ======================================================================


class TestCascadeTeardownRPCFailure:
    def test_slot_not_pooled_when_rpc_raises(self) -> None:
        pool = _make_pool()
        slot = PoolSlot(
            pid=1234, sock=MagicMock(), cascade_id="cascade-X",
        )
        rpc = _StubRunnerRPC(
            session_end_raises=ConnectionError("transport dead"),
        )
        server = _StubServer(rpc, _stub_spawned(slot), pool)

        server._invoke_shutdown_branch()

        assert pool.idle_count() == 0
        assert rpc.session_end_calls == 1
        assert rpc.session_shutdown_calls == 1


# ======================================================================
# Path 4: Cold-spawned session — no pool_slot, no session_end RPC fired
# ======================================================================


class TestColdSpawnTeardownUnchanged:
    def test_cold_spawn_skips_session_end(self) -> None:
        """A session whose runner came from cold-spawn (no pool reuse
        ever happened) doesn't call session_end — the existing
        session_shutdown / close path runs."""
        pool = _make_pool()
        rpc = _StubRunnerRPC(session_end_result={"errors": []})
        # spawned.pool_slot is None — cold spawn.
        server = _StubServer(rpc, _stub_spawned(pool_slot=None), pool)

        server._invoke_shutdown_branch()

        # session_end NOT called.
        assert rpc.session_end_calls == 0
        # Cold teardown ran.
        assert rpc.session_shutdown_calls == 1
        # Pool stays empty.
        assert pool.idle_count() == 0

    def test_no_pool_manager_skips_cascade_branch(self) -> None:
        """A pool-served session whose pool_manager ref is None
        (defensive) takes the cold path."""
        pool_slot = PoolSlot(pid=1234, sock=MagicMock(), cascade_id="C")
        rpc = _StubRunnerRPC(session_end_result={"errors": []})
        server = _StubServer(rpc, _stub_spawned(pool_slot), pool_manager=None)

        server._invoke_shutdown_branch()

        assert rpc.session_end_calls == 0
        assert rpc.session_shutdown_calls == 1


# ======================================================================
# Path 5: Cross-cascade isolation — different cascade_id can NOT reuse
# ======================================================================


class TestCrossCascadeIsolation:
    def test_different_cascade_does_not_reuse_returned_slot(self) -> None:
        """Slot affined to cascade-A is in the pool.  A request for
        cascade-B doesn't get it — cross-cascade reuse is forbidden by
        design."""
        pool = _make_pool()
        slot = PoolSlot(pid=1234, sock=MagicMock(), cascade_id="cascade-A")
        rpc = _StubRunnerRPC(session_end_result={"plugins_reset": 1, "errors": []})
        server = _StubServer(rpc, _stub_spawned(slot), pool)

        server._invoke_shutdown_branch()
        assert pool.idle_count() == 1

        # Request a different cascade — no PURE IDLE slot, no match.
        slot2 = pool.acquire_slot(cascade_driver_id="cascade-B")
        assert slot2 is None
        counters = pool.get_telemetry()
        assert counters["cascade_slot_reuse_misses_total"] == 1
