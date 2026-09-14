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


import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: the return path appends unconditionally again.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="            if len(self._idle_slots) >= self.max_size:",
        replace="            if False:",
        test="TestPoolCapacity::test_returning_to_a_full_pool_does_not_grow_it",
        because="the idle pool growing without a ceiling",
    ),
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="                if self.unreserved_idle_count() >= self.target_size:",
        replace="                if self.idle_count() >= self.target_size:",
        test=("TestMultiTenantCapacity::"
              "test_replenish_forks_when_every_idle_slot_is_another_tenants"),
        because=("replenishment counting slots the waiting tenant is "
                 "forbidden to use, so a pool full of one cascade's "
                 "reservations reads full and never forks"),
    ),
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="        if reservations:",
        replace="        if False:",
        test=("TestPoolCapacity::"
              "test_a_live_returner_displaces_the_stalest_reservation"),
        because=("eviction firing only against unaffiliated residents, so "
                 "a live cascade's own slot is destroyed to preserve "
                 "reservations of a cascade that may never come back"),
    ),
]


def _pool(*slots, idle_timeout=DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS) -> PoolManager:
    """Build a PoolManager pre-seeded with the supplied idle slots.

    Threads stay off because the constructor starts none: both
    ``spawn_initial_slots`` and ``start_replenishment`` are explicit
    calls these tests never make.  The fixture used to pass
    ``target_size=0`` for that reason, which was never the mechanism —
    and 0 is not an inert value: it is "pool disabled", and it now also
    means "every returned slot is over capacity".  A fixture that picks
    a sentinel for a side effect it does not have will eventually
    disagree with production about what the sentinel means.
    """
    pool = PoolManager(
        template_manager=MagicMock(),
        target_size=8,
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


# ======================================================================
# The pool had a floor and no ceiling
#
# ``target_size`` decided how many slots to fork at startup and when
# replenishment should STOP.  Nothing said how many at most, and
# ``return_slot_after_session`` appended unconditionally -- so the pool
# grew by one runner per cleanly-ended pool session and never shrank
# while the daemon lived.  Measured live at 32 slots / 4237 MB.
#
# It surfaced as ``pass show`` failing because gpg could not fork: a
# resource leak wearing a credentials error, found only because three
# unlike failures shared "late in the run".
#
# The log line made it look guarded -- it printed ``idle_count=%d/%d``,
# length against target_size, so it read ``idle_count=32/2``: a
# comparison the code did not make.
# ======================================================================

class TestPoolCapacity:

    def test_returning_to_a_full_pool_does_not_grow_it(self) -> None:
        pool = _pool(_slot(1), _slot(2), idle_timeout=300.0)
        pool.target_size = 2
        pool.max_size = 2

        pool.return_slot_after_session(_slot(3, cascade_id="c"))

        assert pool.idle_count() == 2, (
            "the pool grew past its ceiling. This is the unbounded path: "
            "one extra runner per cleanly-ended session, ~132 MB each, "
            "never reclaimed while the daemon lives."
        )

    def test_a_pure_idle_slot_is_evicted_when_there_is_no_reservation(self) -> None:
        """The original rule, and the branch of it that survives #898.

        A pure-idle resident carries no warm state; a returning affine
        slot does.  With nothing else to spend, trading the
        unaffiliated one for the affine one keeps the warm path the
        pool exists to serve.
        """
        pool = _pool(_slot(1), _slot(2), idle_timeout=300.0)
        pool.target_size = 2
        pool.max_size = 2

        pool.return_slot_after_session(_slot(3, cascade_id="also-warm"))

        assert {s.pid for s in pool._idle_slots} == {2, 3}, (
            "the returning affine slot was not admitted over a resident "
            "that carries nothing"
        )
        assert [s.pid for s in pool._pending_teardown] == [1]
        assert pool.get_telemetry()["pool_stale_reservation_evicted_total"] == 0

    def test_a_live_returner_displaces_the_stalest_reservation(self) -> None:
        """Liveness outranks warmth (#898).

        The returner belongs to a cascade demonstrably mid-run -- it
        just finished a stage and its next one is typically already
        queued.  A resident reservation belongs to a cascade that may
        never come back, and the stalest of them is the one the 300 s
        cascade-idle sweep was going to reap anyway.  Destroying the
        live cascade's own slot to preserve those is how a second
        tenant starved for a full 60 s.
        """
        now = time.monotonic()
        pool = _pool(
            _slot(1, cascade_id="a", last_end_ts=now - 5.0),
            _slot(2, cascade_id="b", last_end_ts=now - 120.0),   # stalest
            idle_timeout=300.0,
        )
        pool.target_size = 2
        pool.max_size = 2

        pool.return_slot_after_session(_slot(3, cascade_id="c"))

        assert {s.pid for s in pool._idle_slots} == {1, 3}, (
            "the live cascade's returning slot was destroyed while a "
            "staler reservation was preserved"
        )
        assert [s.pid for s in pool._pending_teardown] == [2]
        assert pool.get_telemetry()["pool_stale_reservation_evicted_total"] == 1

    def test_a_reservation_that_never_served_is_spent_first(self) -> None:
        """No ``last_session_end_ts`` means no warm session state.

        Such a slot is affined but carries nothing the next stage would
        reuse, so it is the cheapest thing in the pool to lose.
        """
        now = time.monotonic()
        pool = _pool(
            _slot(1, cascade_id="a", last_end_ts=now - 200.0),
            _slot(2, cascade_id="b", last_end_ts=None),
            idle_timeout=300.0,
        )
        pool.target_size = 2
        pool.max_size = 2

        pool.return_slot_after_session(_slot(3, cascade_id="c"))

        assert {s.pid for s in pool._idle_slots} == {1, 3}

    def test_a_pure_returner_still_goes(self) -> None:
        """A returner with no cascade has no claim a resident lacks.

        It also carries its session's accumulated heap (129-187 MB),
        while a resident may still be a CoW-cheap template fork.
        """
        pool = _pool(_slot(1, cascade_id="a"), _slot(2), idle_timeout=300.0)
        pool.target_size = 2
        pool.max_size = 2

        pool.return_slot_after_session(_slot(3))

        assert {s.pid for s in pool._idle_slots} == {1, 2}
        assert [s.pid for s in pool._pending_teardown] == [3]

    def test_the_evicted_slot_is_QUEUED_not_torn_down_inline(self) -> None:
        """Teardown blocks on the daemon loop; this path must not.

        ``_teardown_slot`` closes the rpc via ``run_coroutine_threadsafe``
        and BLOCKS on the future.  ``return_slot_after_session`` runs
        wherever ``JaatoServer.shutdown`` runs, so doing it here would
        make a cap inherit a thread-context assumption -- the same shape
        as the circular wait fixed in #657, where a worker held a lock
        and waited for the loop while the loop waited for the lock.
        """
        pool = _pool(_slot(1), _slot(2), idle_timeout=300.0)
        pool.target_size = 2
        pool.max_size = 2
        torn: list = []
        pool._teardown_slot = lambda slot, reason: torn.append(slot.pid)

        pool.return_slot_after_session(_slot(3))

        assert not torn, (
            "the return path tore a slot down inline, blocking on the "
            "daemon loop from a thread that has no guarantee of being off it"
        )
        assert [s.pid for s in pool._pending_teardown] == [3]

    def test_the_drain_tears_down_what_the_cap_queued(self) -> None:
        pool = _pool(_slot(1), _slot(2), idle_timeout=300.0)
        pool.target_size = 2
        pool.max_size = 2
        torn: list = []
        pool._teardown_slot = lambda slot, reason: torn.append((slot.pid, reason))

        pool.return_slot_after_session(_slot(3))
        pool._drain_pending_teardown()

        assert torn == [(3, "over-capacity")], (
            "the queued slot was never reclaimed; the count is bounded but "
            "the PROCESS is not, which is the memory this fixes"
        )
        assert pool._pending_teardown == []


# ======================================================================
# Capacity was accounted globally while slots were allocated per-tenant
#
# Cascade affinity is the design, and correctly so: cross-cascade reuse
# is forbidden because warm plugin state belongs to the original
# cascade.  The defect (#898) was that the two capacity sites counted
# every idle slot as capacity anyway -- so a pool at capacity could be
# EMPTY from the point of view of every tenant but one.
#
# Reported live: two cascades on one daemon, ``target_size=2``, both
# idle slots affined to cascade B.  Cascade A's ``session.new`` was
# accepted and then received no daemon attention for a full 60 s;
# ``acquire_slot(cascade=A)`` returned None (no affine match, no
# pure-idle) and replenishment read ``idle_count() == 2 >= 2`` and
# never forked.  Nothing to run on, and nothing in the system that
# would ever create one, until B released 31 s too late.
# ======================================================================


class TestMultiTenantCapacity:

    def test_unreserved_idle_count_excludes_other_tenants_slots(self) -> None:
        """The quantity a waiter can actually draw on."""
        pool = _pool(
            _slot(1, cascade_id="B"),
            _slot(2, cascade_id="B"),
            _slot(3),
        )
        assert pool.idle_count() == 3
        assert pool.unreserved_idle_count() == 1, (
            "reservations were counted as free capacity; that is the "
            "arithmetic that starved the second tenant"
        )

    def test_replenish_forks_when_every_idle_slot_is_another_tenants(self) -> None:
        """The reported pool state, one replenish iteration.

        Two idle slots, both affined to B, ``target_size=2``.  A tenant
        that is not B has nothing to acquire, so the loop must fork --
        reading "2/2, full" is what left A with no path to a slot.
        """
        tm = MagicMock()
        tm.is_alive.return_value = True
        tm.request_fork_slot.return_value = (9001, MagicMock(name="sock_9001"))
        pool = PoolManager(template_manager=tm, target_size=2, max_size=4)
        pool._idle_slots = [
            _slot(1, cascade_id="B", last_end_ts=time.monotonic()),
            _slot(2, cascade_id="B", last_end_ts=time.monotonic()),
        ]

        pool._replenish_stop.set()   # one iteration, no sleeping
        pool._replenish_loop()

        assert tm.request_fork_slot.called is False, (
            "the stop event should short-circuit before the loop body"
        )

        pool._replenish_stop.clear()
        _run_one_replenish_iteration(pool)

        assert tm.request_fork_slot.call_count == 1, (
            "replenishment counted slots the waiting tenant is forbidden "
            "to use and concluded the pool was full"
        )
        assert pool.unreserved_idle_count() == 1

    def test_replenish_stops_at_the_ceiling(self) -> None:
        """Reservations still cost 129-187 MB each.

        Unreserved accounting is what un-starves the second tenant;
        ``max_size`` is what keeps that from becoming one runner per
        cascade forever.  Hitting it is recorded, because it is the
        signal to raise the knob.
        """
        tm = MagicMock()
        tm.is_alive.return_value = True
        tm.request_fork_slot.return_value = (9002, MagicMock())
        pool = PoolManager(template_manager=tm, target_size=2, max_size=2)
        pool._idle_slots = [
            _slot(1, cascade_id="B", last_end_ts=time.monotonic()),
            _slot(2, cascade_id="C", last_end_ts=time.monotonic()),
        ]

        _run_one_replenish_iteration(pool)

        assert tm.request_fork_slot.called is False
        assert pool.get_telemetry()["pool_replenish_ceiling_blocked_total"] == 1

    def test_a_second_tenant_can_acquire_once_replenishment_has_run(self) -> None:
        """End to end, at pool level: A no longer starves behind B."""
        tm = MagicMock()
        tm.is_alive.return_value = True
        forked = iter([(9101, MagicMock()), (9102, MagicMock())])
        tm.request_fork_slot.side_effect = lambda: next(forked, None)
        pool = PoolManager(template_manager=tm, target_size=2, max_size=4)
        pool._idle_slots = [
            _slot(1, cascade_id="B", last_end_ts=time.monotonic()),
            _slot(2, cascade_id="B", last_end_ts=time.monotonic()),
        ]

        assert pool.acquire_slot(cascade_driver_id="A") is None, (
            "precondition: with only B's reservations resident there is "
            "nothing for A to take"
        )

        _run_one_replenish_iteration(pool)
        out = pool.acquire_slot(cascade_driver_id="A")

        assert out is not None, "cascade A starved behind cascade B"
        assert out.cascade_id == "A"

    def test_max_size_defaults_to_twice_the_target(self) -> None:
        pool = PoolManager(template_manager=MagicMock(), target_size=3)
        assert pool.max_size == 6

    def test_a_ceiling_below_the_floor_is_raised_to_it(self) -> None:
        """Not a policy: it would evict on every single return."""
        pool = PoolManager(
            template_manager=MagicMock(), target_size=4, max_size=1,
        )
        assert pool.max_size == 4


def _run_one_replenish_iteration(pool: PoolManager) -> None:
    """Run exactly one pass of the replenish loop body.

    The loop is ``while not stop``, so it is driven here by arming the
    stop event at BOTH of the body's exits -- the backoff wait and a
    successful fork.  Without the second, a loop that forks does not
    wait, so it would keep going until the pool reached target and the
    test would be measuring several iterations while claiming one.
    There is no sleeping and no thread.
    """
    real_wait = pool._replenish_stop.wait
    real_fork = pool._template_manager.request_fork_slot

    def _stop_then(fn):
        def _wrapped(*args, **kwargs):
            pool._replenish_stop.set()
            return fn(*args, **kwargs)
        return _wrapped

    pool._replenish_stop.wait = _stop_then(  # type: ignore[assignment]
        lambda timeout=None: True)
    pool._template_manager.request_fork_slot = _stop_then(real_fork)
    try:
        pool._replenish_loop()
    finally:
        pool._replenish_stop.wait = real_wait  # type: ignore[assignment]
        pool._template_manager.request_fork_slot = real_fork
        pool._replenish_stop.clear()
