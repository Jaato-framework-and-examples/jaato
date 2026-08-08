"""Phase 2 cascade-sharing pool semantics — unit tests.

Pins the behaviour of:
  - ``PoolManager.acquire_slot(cascade_driver_id)`` affinity routing
  - ``PoolManager.return_slot_after_session`` re-pool path
  - ``PoolManager._sweep_cascade_idle`` teardown after idle timeout
  - Telemetry counters: ``cascade_slot_reuse_hits_total``,
    ``cascade_slot_reuse_misses_total``,
    ``cascade_slots_idle_torndown_total``

Design reference: ``docs/design/runner-cascade-sharing.md`` §4.2.

These are pure pool-side tests — no real runner subprocesses, no
real template fork.  The PoolSlot's ``sock`` is a MagicMock; the
``_sweep_cascade_idle`` test uses ``time.monotonic`` directly so
we can age out a slot synchronously without ``time.sleep(300)``.
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from server.runner_pool import (
    DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS,
    PoolManager,
    PoolSlot,
)


def _slot(pid: int, cascade_id=None, last_end_ts=None) -> PoolSlot:
    """Build a PoolSlot with a MagicMock socket."""
    return PoolSlot(
        pid=pid,
        sock=MagicMock(name=f"sock_{pid}"),
        cascade_id=cascade_id,
        last_session_end_ts=last_end_ts,
    )


def _pool(*slots, idle_timeout=DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS) -> PoolManager:
    """Build a PoolManager pre-seeded with the supplied idle slots.

    target_size=0 so spawn_initial_slots / replenish loop stay off
    — these tests pin behaviour at the API level, not the
    background-thread level.
    """
    pool = PoolManager(
        template_manager=MagicMock(),
        target_size=0,
        cascade_idle_timeout_seconds=idle_timeout,
    )
    pool._idle_slots = list(slots)
    return pool


# ======================================================================
# acquire_slot — cascade affinity routing
# ======================================================================


class TestAcquireCascadeRouting:
    """Per design doc §4.2 claim flow."""

    def test_no_cascade_id_behaves_like_phase1(self) -> None:
        """When caller passes no cascade_driver_id, acquire takes
        any PURE IDLE slot (FIFO from the tail per Phase 1 semantics)."""
        pool = _pool(_slot(1), _slot(2))
        out = pool.acquire_slot()
        assert out is not None
        assert out.pid in (1, 2)
        # cascade_id remains None — slot was never affined.
        assert out.cascade_id is None

    def test_cascade_match_returns_affined_slot(self) -> None:
        """An IDLE_FOR_CASCADE(C) slot is returned when caller
        requests cascade C — the reuse hit."""
        pool = _pool(
            _slot(1, cascade_id=None),                      # PURE
            _slot(2, cascade_id="cascade-A"),               # affined elsewhere
            _slot(3, cascade_id="cascade-B"),               # match
        )
        out = pool.acquire_slot(cascade_driver_id="cascade-B")
        assert out is not None
        assert out.pid == 3
        assert out.cascade_id == "cascade-B"
        # Counters: one cascade hit.
        counters = pool.get_telemetry()
        assert counters["cascade_slot_reuse_hits_total"] == 1
        assert counters["cascade_slot_reuse_misses_total"] == 0

    def test_cascade_miss_uses_pure_idle_and_stamps(self) -> None:
        """No matching cascade → take a PURE IDLE slot, stamp it
        with the cascade so subsequent sessions of the same
        cascade hit."""
        pool = _pool(
            _slot(1, cascade_id=None),                      # PURE
            _slot(2, cascade_id="cascade-A"),               # affined elsewhere
        )
        out = pool.acquire_slot(cascade_driver_id="cascade-B")
        assert out is not None
        assert out.pid == 1  # the PURE IDLE one
        assert out.cascade_id == "cascade-B"  # stamped on the way out
        counters = pool.get_telemetry()
        assert counters["cascade_slot_reuse_hits_total"] == 0
        assert counters["cascade_slot_reuse_misses_total"] == 1

    def test_cascade_miss_no_pure_returns_none(self) -> None:
        """All slots affined to other cascades + no PURE IDLE → None.
        Cross-cascade reuse is forbidden by design (warm plugin state
        belongs to the original cascade)."""
        pool = _pool(
            _slot(1, cascade_id="cascade-A"),
            _slot(2, cascade_id="cascade-B"),
        )
        out = pool.acquire_slot(cascade_driver_id="cascade-C")
        assert out is None
        # Both miss AND empty-pool semantics: the cascade miss
        # increments + the pool_acquire_miss_total also increments.
        counters = pool.get_telemetry()
        assert counters["cascade_slot_reuse_hits_total"] == 0
        assert counters["cascade_slot_reuse_misses_total"] == 1
        assert counters["pool_acquire_miss_total"] == 1

    def test_empty_pool_with_cascade_request(self) -> None:
        """No slots at all + cascade requested → None, miss counter."""
        pool = _pool()
        out = pool.acquire_slot(cascade_driver_id="cascade-X")
        assert out is None
        counters = pool.get_telemetry()
        assert counters["cascade_slot_reuse_misses_total"] == 1
        assert counters["pool_acquire_miss_total"] == 1

    def test_pure_slot_preferred_over_other_cascade(self) -> None:
        """When a request for cascade C comes in and there's both a
        PURE slot AND a slot affined to OTHER cascade, take the
        PURE one — never cross-cascade."""
        pool = _pool(
            _slot(1, cascade_id="other-cascade"),
            _slot(2, cascade_id=None),  # PURE
        )
        out = pool.acquire_slot(cascade_driver_id="cascade-C")
        assert out is not None
        assert out.pid == 2
        assert out.cascade_id == "cascade-C"  # stamped


# ======================================================================
# return_slot_after_session
# ======================================================================


class TestReturnSlot:
    """Pool-side bookkeeping for slots that have completed a session
    and are returning to idle."""

    def test_return_appends_and_stamps_timestamp(self) -> None:
        pool = _pool()  # empty
        slot = _slot(7, cascade_id="cascade-A")
        before_ts = time.monotonic()
        pool.return_slot_after_session(slot)
        # Slot back in idle pool.
        assert pool.idle_count() == 1
        # Timestamp stamped.
        assert slot.last_session_end_ts is not None
        assert slot.last_session_end_ts >= before_ts
        # Cascade affinity preserved.
        assert slot.cascade_id == "cascade-A"

    def test_returned_slot_is_reused_on_next_match(self) -> None:
        pool = _pool()
        slot = _slot(7, cascade_id="cascade-A")
        pool.return_slot_after_session(slot)
        out = pool.acquire_slot(cascade_driver_id="cascade-A")
        assert out is not None
        assert out.pid == 7
        counters = pool.get_telemetry()
        assert counters["cascade_slot_reuse_hits_total"] == 1


# ======================================================================
# _sweep_cascade_idle — teardown after idle timeout
# ======================================================================


class TestCascadeIdleSweep:
    """The replenish loop's per-iteration cascade-idle teardown sweep."""

    def test_fresh_idle_slot_not_reaped(self) -> None:
        """A slot returned just now is well within the idle window."""
        pool = _pool(idle_timeout=300.0)
        slot = _slot(1, cascade_id="cascade-A", last_end_ts=time.monotonic())
        pool._idle_slots = [slot]
        pool._sweep_cascade_idle()
        assert pool.idle_count() == 1
        counters = pool.get_telemetry()
        assert counters["cascade_slots_idle_torndown_total"] == 0

    def test_expired_idle_slot_reaped(self) -> None:
        """A slot whose last_session_end_ts is older than the timeout
        is torn down."""
        pool = _pool(idle_timeout=60.0)
        # Idle for 120s — well past timeout.
        stale = _slot(
            1, cascade_id="cascade-A",
            last_end_ts=time.monotonic() - 120.0,
        )
        pool._idle_slots = [stale]
        with patch("os.waitpid", return_value=(stale.pid, 0)) as waitpid_mock:
            pool._sweep_cascade_idle()
        assert pool.idle_count() == 0
        stale.sock.close.assert_called_once()
        waitpid_mock.assert_called_once_with(stale.pid, 0)
        counters = pool.get_telemetry()
        assert counters["cascade_slots_idle_torndown_total"] == 1

    def test_pure_idle_slots_exempt(self) -> None:
        """PURE IDLE slots (cascade_id=None) have no cascade affinity
        to time out — sweep ignores them even if last_session_end_ts
        is ancient."""
        pool = _pool(idle_timeout=60.0)
        pure = _slot(1, cascade_id=None, last_end_ts=time.monotonic() - 999.0)
        pool._idle_slots = [pure]
        pool._sweep_cascade_idle()
        assert pool.idle_count() == 1
        counters = pool.get_telemetry()
        assert counters["cascade_slots_idle_torndown_total"] == 0

    def test_mixed_slots_only_expired_reaped(self) -> None:
        """Mix of fresh + stale + pure — only the stale cascade slot
        gets torn down."""
        pool = _pool(idle_timeout=60.0)
        fresh = _slot(
            1, cascade_id="cascade-A", last_end_ts=time.monotonic(),
        )
        stale = _slot(
            2, cascade_id="cascade-B",
            last_end_ts=time.monotonic() - 200.0,
        )
        pure = _slot(3, cascade_id=None, last_end_ts=None)
        pool._idle_slots = [fresh, stale, pure]
        with patch("os.waitpid", return_value=(stale.pid, 0)):
            pool._sweep_cascade_idle()
        assert pool.idle_count() == 2
        remaining_pids = {s.pid for s in pool._idle_slots}
        assert remaining_pids == {1, 3}
        counters = pool.get_telemetry()
        assert counters["cascade_slots_idle_torndown_total"] == 1

    def test_slot_with_cascade_but_no_endts_exempt(self) -> None:
        """Edge: a slot with a cascade_id but last_session_end_ts=None
        (e.g. directly seeded for tests, never actually returned) is
        not eligible for sweep — only returned slots are."""
        pool = _pool(idle_timeout=60.0)
        slot = _slot(1, cascade_id="cascade-A", last_end_ts=None)
        pool._idle_slots = [slot]
        pool._sweep_cascade_idle()
        assert pool.idle_count() == 1


# ======================================================================
# Configurable idle timeout
# ======================================================================


class TestConfigurableTimeout:
    def test_default_timeout_matches_design_doc(self) -> None:
        """§3 decision 5: default is 300s."""
        assert DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS == 300.0
        pool = PoolManager(template_manager=MagicMock(), target_size=0)
        assert pool._cascade_idle_timeout == 300.0

    def test_custom_timeout_honored(self) -> None:
        pool = PoolManager(
            template_manager=MagicMock(),
            target_size=0,
            cascade_idle_timeout_seconds=42.0,
        )
        assert pool._cascade_idle_timeout == 42.0
