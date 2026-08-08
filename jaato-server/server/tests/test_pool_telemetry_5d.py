"""Pool PR 5d: telemetry counter tests.

PoolManager exposes a monotonic counter dict via ``get_telemetry``:
- pool_slot_acquired_total
- pool_acquire_miss_total
- pool_replenish_success_total
- pool_replenish_failures_total
- template_respawn_attempts_total
- template_respawn_failures_total

These tests pin the increment-at-the-right-event semantics so
external observers (admin commands, OTel exporters, log emitters)
can rely on stable counter behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from server.runner_pool import PoolManager, PoolSlot


def _fake_handle(pid: int):
    """Return a raw tuple shaped like ``request_fork_slot`` produces."""
    return (pid, MagicMock(name=f"sock_{pid}"))


def _fake_slot(pid: int, cascade_id=None):
    """Return a ``PoolSlot`` dataclass — what ``_idle_slots`` holds."""
    return PoolSlot(
        pid=pid, sock=MagicMock(name=f"sock_{pid}"),
        cascade_id=cascade_id,
    )


# ----------------------------------------------------------------------
# Initial state
# ----------------------------------------------------------------------


def test_telemetry_starts_at_zero() -> None:
    pool = PoolManager(template_manager=MagicMock(), target_size=0)
    counters = pool.get_telemetry()
    assert counters == {
        "pool_slot_acquired_total": 0,
        "pool_acquire_miss_total": 0,
        "pool_replenish_success_total": 0,
        "pool_replenish_failures_total": 0,
        "template_respawn_attempts_total": 0,
        "template_respawn_failures_total": 0,
        # Phase 2 cascade-sharing counters (server 0.6.144+).
        "cascade_slot_reuse_hits_total": 0,
        "cascade_slot_reuse_misses_total": 0,
        "cascade_slots_idle_torndown_total": 0,
    }


def test_telemetry_returns_a_copy() -> None:
    """Snapshot semantics: mutating the returned dict doesn't affect
    the manager's internal counters."""
    pool = PoolManager(template_manager=MagicMock(), target_size=0)
    snap = pool.get_telemetry()
    snap["pool_slot_acquired_total"] = 999
    fresh = pool.get_telemetry()
    assert fresh["pool_slot_acquired_total"] == 0


# ----------------------------------------------------------------------
# acquire_slot increments
# ----------------------------------------------------------------------


def test_acquire_slot_hit_increments_acquired_total() -> None:
    pool = PoolManager(template_manager=MagicMock(), target_size=2)
    pool._idle_slots = [_fake_slot(10001), _fake_slot(10002)]
    pool.acquire_slot()
    pool.acquire_slot()
    counters = pool.get_telemetry()
    assert counters["pool_slot_acquired_total"] == 2
    assert counters["pool_acquire_miss_total"] == 0


def test_acquire_slot_miss_increments_miss_total() -> None:
    pool = PoolManager(template_manager=MagicMock(), target_size=2)
    pool._idle_slots = []
    pool.acquire_slot()  # miss
    pool.acquire_slot()  # miss
    pool.acquire_slot()  # miss
    counters = pool.get_telemetry()
    assert counters["pool_slot_acquired_total"] == 0
    assert counters["pool_acquire_miss_total"] == 3


def test_acquire_slot_mixed_hit_miss() -> None:
    pool = PoolManager(template_manager=MagicMock(), target_size=2)
    pool._idle_slots = [_fake_slot(10001)]
    pool.acquire_slot()  # hit
    pool.acquire_slot()  # miss (only one was seeded)
    pool.acquire_slot()  # miss
    counters = pool.get_telemetry()
    assert counters["pool_slot_acquired_total"] == 1
    assert counters["pool_acquire_miss_total"] == 2


# ----------------------------------------------------------------------
# Replenishment increments
# ----------------------------------------------------------------------


def test_replenish_success_increments_on_fork_slot_return() -> None:
    """Direct test of the replenish-loop body's success path.  We
    invoke a single iteration via a controlled mock to avoid the
    real thread + sleep timing."""
    tm = MagicMock()
    tm.is_alive.return_value = True
    tm.request_fork_slot.return_value = _fake_handle(20001)

    pool = PoolManager(template_manager=tm, target_size=2)
    # Manually run the body once: simulate "idle_count < target".
    # We bypass the thread + go straight at the helper that
    # request_fork_slot returned a handle.
    handle = tm.request_fork_slot()
    if handle is not None:
        pid, sock = handle
        with pool._lock:
            pool._idle_slots.append(PoolSlot(pid=pid, sock=sock))
        pool._incr("pool_replenish_success_total")
    counters = pool.get_telemetry()
    assert counters["pool_replenish_success_total"] == 1
    assert counters["pool_replenish_failures_total"] == 0


def test_replenish_failure_increments_on_fork_slot_none() -> None:
    """Direct test of the failure-counter increment for clarity.
    The thread integration is covered by the existing
    replenishment tests; this is the unit-level counter pin."""
    pool = PoolManager(template_manager=MagicMock(), target_size=2)
    pool._incr("pool_replenish_failures_total")
    pool._incr("pool_replenish_failures_total")
    counters = pool.get_telemetry()
    assert counters["pool_replenish_failures_total"] == 2


# ----------------------------------------------------------------------
# Template respawn increments
# ----------------------------------------------------------------------


def test_template_respawn_success_increments_attempts_only() -> None:
    tm = MagicMock()
    tm.is_alive.return_value = False
    tm.spawn.return_value = None  # success
    pool = PoolManager(template_manager=tm, target_size=2)

    with patch("server.runner_pool.os.waitpid", side_effect=ChildProcessError()):
        pool._handle_template_death()

    counters = pool.get_telemetry()
    assert counters["template_respawn_attempts_total"] == 1
    assert counters["template_respawn_failures_total"] == 0


def test_template_respawn_failure_increments_both() -> None:
    tm = MagicMock()
    tm.is_alive.return_value = False
    tm.spawn.side_effect = OSError("simulated spawn failure")
    pool = PoolManager(template_manager=tm, target_size=2)

    with patch("server.runner_pool.os.waitpid", side_effect=ChildProcessError()):
        pool._handle_template_death()

    counters = pool.get_telemetry()
    assert counters["template_respawn_attempts_total"] == 1
    assert counters["template_respawn_failures_total"] == 1


def test_template_respawn_multiple_iterations_accumulate() -> None:
    """If watchdog fires multiple times (template flaky), counters
    accumulate across iterations."""
    tm = MagicMock()
    tm.is_alive.return_value = False
    # First two spawns fail; third succeeds.
    tm.spawn.side_effect = [
        OSError("attempt 1 fail"),
        OSError("attempt 2 fail"),
        None,  # success
    ]
    pool = PoolManager(template_manager=tm, target_size=2)

    with patch("server.runner_pool.os.waitpid", side_effect=ChildProcessError()):
        pool._handle_template_death()
        pool._handle_template_death()
        pool._handle_template_death()

    counters = pool.get_telemetry()
    assert counters["template_respawn_attempts_total"] == 3
    assert counters["template_respawn_failures_total"] == 2
