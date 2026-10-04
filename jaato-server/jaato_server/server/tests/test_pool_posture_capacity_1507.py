"""A served slot is capacity for its own posture only (#1507).

Since #1100 a slot that has served a session fits only a session of the
same posture -- AppArmor profile, uid (#1168), SELinux boundary.  That gate
is right.  What was wrong is that replenishment still counted those served
slots toward ``target_size``: a pool full of slots that had served
unconfined IPC sessions read "full", never forked a virgin, and every
confined WS session cold-spawned (21/21 in the reporting bench).

The rule now: the floor counts VIRGIN unaffined slots only.  Served
unaffined slots are warm capacity for their posture and sit on top of the
floor, like reservations (#898), bounded by ``max_size``.  At the ceiling a
served slot of the least recently demanded posture is spent first.

The tests drive the real :class:`PoolManager` with a fake template (no
processes) and run the replenish loop body synchronously until it would
pause, which is what "the pool has settled" means.
"""

from __future__ import annotations

import itertools
import time
from typing import Optional
from unittest.mock import MagicMock

from jaato_server.server.runner_pool import PoolManager, PoolSlot
from jaato_server.shared.tests.reversion import Reversion

_POOL = "jaato-server/jaato_server/server/runner_pool.py"

CONFINED = "jaato-ws-repo-3f2a9c1b7d4e"

REVERSIONS = [
    Reversion(
        target=_POOL,
        find="                if self.virgin_idle_count() >= self.target_size:",
        replace="                if self.unreserved_idle_count() >= self.target_size:",
        test=("TestTheFloorCountsVirginSlots::"
              "test_unconfined_served_slots_do_not_satisfy_a_confined_arrival"),
        because=("the floor counting served slots, so a pool of unconfined-"
                 "served slots reads full and a confined session cold-spawns"),
    ),
    Reversion(
        target=_POOL,
        find="                        and not self._spend_served_slot_for_floor()):",
        replace="                        and True):",
        test=("TestTheCeiling::"
              "test_at_the_ceiling_a_served_slot_is_spent_for_a_virgin"),
        because=("a pool at its ceiling with served slots of one posture "
                 "never making room for a virgin, so the other posture "
                 "cold-spawns forever"),
    ),
    Reversion(
        target=_POOL,
        find="                if (self.idle_count() >= self.max_size",
        replace="                if (False",
        test="TestTheCeiling::test_the_ceiling_is_never_exceeded",
        because="replenishment forking past max_size",
    ),
    Reversion(
        target=_POOL,
        find=("            if not candidate.has_served:\n"
              "                if virgin_idx is None:"),
        replace=("            if False:\n"
                 "                if virgin_idx is None:"),
        test=("TestTheFloorCountsVirginSlots::"
              "test_a_fitting_served_slot_is_spent_before_a_virgin"),
        because=("a virgin taken when a served slot of the right posture "
                 "was idle, costing a fork and a virgin for nothing"),
    ),
    Reversion(
        target=_POOL,
        find=("            if (self._posture_demand(victim)\n"
              "                    < self._posture_demand(returner)):"),
        replace="            if False:",
        test=("TestTheCeiling::"
              "test_a_returner_displaces_a_slot_of_a_posture_nobody_asks_for"),
        because=("a full pool of one posture's served slots refusing every "
                 "returner of the posture actually in demand"),
    ),
    Reversion(
        target=_POOL,
        find='                    self._incr("pool_posture_miss_total")',
        replace="                    pass",
        test="TestTelemetry::test_a_posture_miss_is_counted",
        because="the #1507 miss being invisible in telemetry",
    ),
    Reversion(
        target=_POOL,
        find='                    name = "served:" + posture_label(posture_of_slot(slot))',
        replace='                    name = "served"',
        test="TestTelemetry::test_idle_slots_are_counted_per_posture",
        because="served slots of different postures reported as one number",
    ),
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _pool(target: int, max_size: Optional[int] = None) -> PoolManager:
    tm = MagicMock(name="template")
    tm.is_alive.return_value = True
    pids = itertools.count(7000)
    tm.request_fork_slot.side_effect = lambda: (next(pids), MagicMock())
    pool = PoolManager(template_manager=tm, target_size=target,
                       max_size=max_size)
    # Teardown would close sockets and waitpid fake pids; record instead.
    pool.torn_down = []  # type: ignore[attr-defined]
    pool._teardown_slot = (  # type: ignore[method-assign]
        lambda slot, reason: pool.torn_down.append(slot.pid))
    return pool


def _settle(pool: PoolManager) -> None:
    """Run the replenish loop until its first pause: the pool is settled.

    A fork does not pause, so the loop keeps forking while the floor is
    short; it pauses when the floor is met or the ceiling blocks it.  The
    pause is replaced with "stop", so no thread and no sleeping.
    """
    pool._replenish_stop.clear()
    pool._pause = pool._replenish_stop.set  # type: ignore[method-assign]
    try:
        pool._replenish_loop()
    finally:
        del pool._pause
        pool._replenish_stop.clear()


def _virgin(pid: int) -> PoolSlot:
    return PoolSlot(pid=pid, sock=MagicMock())


def _served(pid: int, profile: Optional[str] = None,
            uid: Optional[int] = None, ended: float = 0.0) -> PoolSlot:
    slot = PoolSlot(pid=pid, sock=MagicMock(), profile_name=profile,
                    runner_uid=uid, last_session_end_ts=ended)
    slot.has_served = True
    return slot


def _acquire(pool: PoolManager, profile: Optional[str]) -> Optional[PoolSlot]:
    return pool.acquire_slot(profile_name=profile)


def _round(pool: PoolManager, profile: Optional[str],
           concurrency: int) -> int:
    """``concurrency`` sessions of one posture at once; how many pool-served.

    Acquire all, let replenishment react, then return all -- the shape of
    a bench round with several sessions in flight.
    """
    held = [_acquire(pool, profile) for _ in range(concurrency)]
    _settle(pool)
    for slot in held:
        if slot is not None:
            pool.return_slot_after_session(slot)
    _settle(pool)
    return sum(1 for s in held if s is not None)


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


class TestTheFloorCountsVirginSlots:

    def test_unconfined_served_slots_do_not_satisfy_a_confined_arrival(self) -> None:
        """The issue's guard: only unconfined-served idle slots, then a
        confined request -> a virgin is forked, and the NEXT confined
        arrival is pool-served."""
        pool = _pool(target=2, max_size=6)
        pool._idle_slots = [_served(1), _served(2)]

        assert _acquire(pool, CONFINED) is None, (
            "an unconfined-served slot was handed to a confined session "
            "(#1100's gate must still hold)")
        _settle(pool)

        assert pool.virgin_idle_count() == 2
        slot = _acquire(pool, CONFINED)
        assert slot is not None, (
            "replenishment read the unconfined-served slots as capacity "
            "and forked nothing the confined posture could take")
        assert slot.profile_name == CONFINED

    def test_a_fitting_served_slot_is_spent_before_a_virgin(self) -> None:
        pool = _pool(target=1, max_size=4)
        virgin = _virgin(1)
        served = _served(2)
        pool._idle_slots = [virgin, served]

        got = _acquire(pool, None)

        assert got is served, (
            "a virgin was taken while a served slot of the session's own "
            "posture sat idle -- that spends the one slot that fits anyone")
        assert pool.virgin_idle_count() == 1

    def test_alternating_postures_are_both_pool_served_after_warm_up(self) -> None:
        """A, B, A, B with two sessions in flight per round."""
        pool = _pool(target=2)            # derived ceiling: 4
        _settle(pool)                     # the startup fill

        served = [(_round(pool, p, concurrency=2))
                  for p in [None, CONFINED] * 4]

        # The first round of each posture may need a fork; after that,
        # every session of either posture is pool-served.
        assert served[2:] == [2] * 6, served
        assert pool.get_telemetry()["pool_posture_miss_total"] == 0

    def test_a_shrink_keeps_served_slots_under_the_ceiling(self) -> None:
        pool = _pool(target=2, max_size=6)
        pool._start_replenish_thread = lambda: None  # type: ignore[method-assign]
        pool._idle_slots = [_virgin(1), _virgin(2), _served(3)]

        pool.resize(1, max_size=6)

        assert [s.pid for s in pool._idle_slots] == [1, 3]


# ---------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------


class TestTheCeiling:

    def test_at_the_ceiling_a_served_slot_is_spent_for_a_virgin(self) -> None:
        pool = _pool(target=1, max_size=3)
        pool._idle_slots = [_served(1, ended=1.0), _served(2, ended=2.0),
                            _served(3, ended=3.0)]

        assert _acquire(pool, CONFINED) is None
        _settle(pool)

        assert pool.virgin_idle_count() == 1
        assert len(pool._idle_slots) == 3
        # The stalest one goes (no posture demand recorded for any).
        assert pool.torn_down == [1]  # type: ignore[attr-defined]
        assert _acquire(pool, CONFINED) is not None

    def test_reservations_are_not_spent_for_a_virgin(self) -> None:
        """#898's ceiling is unchanged: a full pool of reservations still
        reports itself as blocked rather than evicting them."""
        pool = _pool(target=1, max_size=2)
        # Recent, so the cascade-idle sweep does not reap them first.
        now = time.monotonic()
        r1 = _served(1, ended=now)
        r1.cascade_id = "B"
        r2 = _served(2, ended=now)
        r2.cascade_id = "B"
        pool._idle_slots = [r1, r2]

        _settle(pool)

        assert [s.pid for s in pool._idle_slots] == [1, 2]
        assert pool.get_telemetry()["pool_replenish_ceiling_blocked_total"] == 1

    def test_the_ceiling_is_never_exceeded(self) -> None:
        pool = _pool(target=2, max_size=3)
        pool._idle_slots = [_served(1), _served(2), _served(3, CONFINED)]
        for profile in [None, CONFINED, None, CONFINED, None]:
            slot = _acquire(pool, profile)
            _settle(pool)
            assert len(pool._idle_slots) <= pool.max_size
            if slot is not None:
                pool.return_slot_after_session(slot)
            assert len(pool._idle_slots) <= pool.max_size
            _settle(pool)
            assert len(pool._idle_slots) <= pool.max_size

    def test_a_returner_displaces_a_slot_of_a_posture_nobody_asks_for(self) -> None:
        pool = _pool(target=0, max_size=2)
        pool._idle_slots = [_served(1, CONFINED), _served(2, CONFINED)]
        slot = _acquire(pool, None)       # nothing fits; records demand
        assert slot is None
        returner = _served(3)

        assert pool.return_slot_after_session(returner) is True
        assert 3 in [s.pid for s in pool._idle_slots]
        assert len(pool._idle_slots) == 2

    def test_a_returner_does_not_displace_a_virgin(self) -> None:
        pool = _pool(target=2, max_size=2)
        pool._idle_slots = [_virgin(1), _virgin(2)]

        assert pool.return_slot_after_session(_served(3)) is False
        assert [s.pid for s in pool._idle_slots] == [1, 2]


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:

    def test_idle_slots_are_counted_per_posture(self) -> None:
        pool = _pool(target=1, max_size=8)
        reserved = _served(5)
        reserved.cascade_id = "C"
        pool._idle_slots = [_virgin(1), _served(2), _served(3),
                            _served(4, CONFINED, uid=1001), reserved]

        t = pool.get_telemetry()

        assert t["pool_idle_virgin"] == 1
        assert t["pool_idle_reserved"] == 1
        assert t["pool_idle_served:unconfined/uid=daemon"] == 2
        assert t[f"pool_idle_served:{CONFINED}/uid=1001"] == 1
        snap = pool.snapshot()
        assert (snap["virgin"], snap["served"], snap["reserved"]) == (1, 3, 1)

    def test_a_posture_miss_is_counted(self) -> None:
        pool = _pool(target=1, max_size=4)
        pool._idle_slots = [_served(1)]

        assert _acquire(pool, CONFINED) is None

        t = pool.get_telemetry()
        assert t["pool_posture_miss_total"] == 1
        assert t["pool_acquire_miss_total"] == 1

    def test_an_empty_pool_miss_is_not_a_posture_miss(self) -> None:
        pool = _pool(target=1, max_size=4)
        assert _acquire(pool, CONFINED) is None
        assert pool.get_telemetry()["pool_posture_miss_total"] == 0

