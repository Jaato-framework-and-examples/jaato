"""A pool slot whose RPC client has died is never served again (#1058).

THE DEFECT.  ``PoolManager`` had no notion of RPC liveness.  ``_closed``
is set in exactly two places -- ``RunnerRPCClient.close`` and the read
loop's ``finally`` -- and the second one runs long after the session that
created the client has ended, while the slot sits in ``_idle_slots``.
Nothing told the pool.  ``acquire_slot`` handed back whatever was at the
head of the list, and the next session discovered the corpse by failing
its own ``session.bootstrap``::

    19:00:25,826  spawn_session_runner: session ... served by pool slot pid=5307
    19:00:25,826  spawn_session_runner: session ... reused slot's rpc client
    19:00:25,828  [ERROR] runner session.bootstrap FAILED:
                  error_type=RunnerCallError error=RunnerRPCClient is closed

The failure was discovered by the CONSUMER, one layer too late, and
#1033's refusal is what made it visible rather than a hang.

WHY THE ONE REAPER THE POOL HAD COULD NOT SEE IT.  ``_sweep_cascade_idle``
is exempt for ``cascade_id is None``, which is correct -- a PURE IDLE slot
has no cascade affinity to time out -- and a standalone session returns
its slot as PURE IDLE (#1033).  So the reported slots were in the one
state nothing in the pool ever looked at.  Warmth has a timeout; liveness
must not be confused with it.

WHAT IS PINNED HERE.

  Layer 1 (``PoolManager``) -- the one that makes the bug unreachable:
    * neither affinity path hands out a slot whose client is closed;
    * a dropped corpse is QUEUED FOR TEARDOWN, not merely skipped --
      skipping leaks the runner process AND leaves the corpse counting
      as capacity, so replenishment reads the pool as full and never
      forks the replacement (#898's failure in a new costume);
    * pure-idle slots are swept for liveness despite being exempt from
      the cascade sweep;
    * only POSITIVE evidence counts -- a slot with no client, or one
      whose client cannot answer, is alive.

  Layer 2 (``spawn_session_runner``) -- the check at the point of use,
  which is genuinely reachable because the spawn path runs OFF the daemon
  loop while the read loop that sets the flag runs on it.  It REPAIRS
  the one cause it can: an externally cancelled read task leaves the
  runner and the transport intact, and a task cannot be un-cancelled, so
  a fresh read task is started rather than a flag cleared.

  The capacity path, where a wrong-slot close would live -- pinned as an
  invariant (every slot reaching teardown has already left the pool)
  rather than asserted in prose.

  One slot, one entry -- a double return is refused by identity, because
  two entries share one client and tearing either down leaves the other
  a corpse in the pool.

  The client itself -- ``is_closed`` / ``close_reason`` / ``can_revive``,
  so an evicted slot's log line names WHY the channel died (the runner
  process is gone, versus the loop raising, versus its reader being
  cancelled) instead of leaving three incidents the same evidence.

No real runner subprocesses here: a ``PoolSlot``'s ``sock`` is a
``MagicMock`` and its ``rpc`` is a stand-in reporting the attributes the
predicate reads.  The tests that drive a real client use an ``asyncio``
socketpair, the convention ``test_runner_rpc_client.py`` already uses.
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Optional
from unittest.mock import MagicMock

import pytest

from server.runner_pool import PoolManager, PoolSlot, slot_rpc_death

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: Each entry puts the defect back and names the test that must go red.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="            self._evict_dead_slots_locked()\n\n            # Path (1)",
        replace="            # Path (1)",
        test=("TestAcquireNeverServesACorpse::"
              "test_a_pure_idle_slot_with_a_closed_client_is_not_served"),
        because=("acquire_slot handing out a slot whose RunnerRPCClient "
                 "is closed, so the next session fails session.bootstrap "
                 "with 'RunnerRPCClient is closed'"),
    ),
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="            self._pending_teardown.append(slot)\n            self._incr(\"pool_dead_slot_evicted_total\")",
        replace="            self._incr(\"pool_dead_slot_evicted_total\")",
        test=("TestACorpseIsReapedNotSkipped::"
              "test_the_dropped_slot_is_queued_for_teardown"),
        because=("a dead slot being skipped rather than reaped, which "
                 "leaks the runner process"),
    ),
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="                self._sweep_dead_slots()\n\n                # Phase 2: cascade-idle",
        replace="                # Phase 2: cascade-idle",
        test=("TestPureIdleSlotsHaveALivenessFloor::"
              "test_the_replenish_loop_reaps_a_dead_pure_idle_slot"),
        because=("pure-idle slots having no health check at all, which is "
                 "the state the reported incident sat in -- exempt from "
                 "the cascade sweep by design and liveness-checked by "
                 "nothing"),
    ),
    Reversion(
        target="jaato-server/server/runner_spawn.py",
        find="    existing_rpc, slot_discarded = _reusable_slot_rpc(\n        pool_slot, session_id, pool_manager, daemon_loop)",
        replace="    existing_rpc = getattr(pool_slot, \"rpc\", None) if pool_slot else None\n    slot_discarded = False",
        test=("TestTheReusePathVerifies::"
              "test_a_closed_client_is_refused_at_the_point_of_use"),
        because=("the reuse fast-path taking the slot's rpc client "
                 "unconditionally"),
    ),
    Reversion(
        target="jaato-server/server/runner_spawn.py",
        find="    if not getattr(rpc, \"is_closed\", False):\n        return rpc, False",
        replace="    if True:\n        return rpc, False",
        test=("TestTheReusePathVerifies::"
              "test_a_closed_client_is_refused_at_the_point_of_use"),
        because=("the point-of-use check being wired in and then not "
                 "actually judging the client"),
    ),
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="            if any(s is slot for s in self._idle_slots):",
        replace="            if False:",
        test="TestOneSlotOneEntry::test_the_second_return_is_refused",
        because=("one slot occupying two entries in the idle pool, whose "
                 "shared rpc client is closed by whichever entry is torn "
                 "down first -- leaving the other servable and dead"),
    ),
    Reversion(
        target="jaato-server/server/runner_pool.py",
        find="        pooled = evicted is not slot",
        replace="        pooled = True",
        test=("TestNoTeardownTouchesASlotStillInThePool::"
              "test_a_pure_idle_returner_at_capacity_is_the_one_dropped"),
        because=("the return path logging \"returned to pool\" for a slot "
                 "it dropped at capacity, which is what a reader of an "
                 "at-capacity daemon log is shown"),
    ),
    Reversion(
        target="jaato-server/server/runner_rpc_client.py",
        find="        if not self._closed and task.cancelled():\n            self._mark_closed(\"cancelled\")",
        replace="        return",
        test=("TestTheClientRecordsHowItDied::"
              "test_a_reader_cancelled_before_it_started_is_still_recorded"),
        because=("a read task cancelled before its first step leaving the "
                 "channel reporting itself healthy with nobody reading "
                 "it -- every call then waits out its deadline"),
    ),
    Reversion(
        target="jaato-server/server/runner_rpc_client.py",
        find="            self._mark_closed(reason)",
        replace="            self._closed = True",
        test=("TestTheClientRecordsHowItDied::"
              "test_eof_is_recorded_as_eof"),
        because=("the channel recording THAT it closed but not WHY, so a "
                 "runner that died and a read loop that raised leave "
                 "identical evidence"),
    ),
]


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------

class _FakeRPC:
    """Stand-in for ``RunnerRPCClient`` exposing only what the pool reads."""

    def __init__(self, closed: bool = False, reason: Optional[str] = None):
        self.is_closed = closed
        self.close_reason = reason


class _RevivableRPC(_FakeRPC):
    """A client whose READER was cancelled: closed, but reopenable."""

    def __init__(self, revive_ok: bool = True):
        super().__init__(closed=True, reason="cancelled")
        self.can_revive = True
        self._revive_ok = revive_ok
        self.revived = 0

    async def revive_read_loop(self) -> bool:
        self.revived += 1
        if not self._revive_ok:
            return False
        self.is_closed = False
        self.close_reason = None
        self.can_revive = False
        return True


class _OpaqueRPC:
    """A client that answers neither question -- an out-of-tree or
    duck-typed stand-in.  Must read as ALIVE: absence of evidence is not
    evidence of death, and reading it as death would empty the pool."""


def _slot(pid: int, *, rpc=None, cascade_id=None, last_end_ts=None,
          config_root=None, workspace_root=None,
          profile_name=None) -> PoolSlot:
    return PoolSlot(
        pid=pid,
        sock=MagicMock(name=f"sock_{pid}"),
        cascade_id=cascade_id,
        config_root=config_root,
        workspace_root=workspace_root,
        profile_name=profile_name,
        last_session_end_ts=last_end_ts,
        rpc=rpc,
    )


def _pool(*slots) -> PoolManager:
    tm = MagicMock(name="template_manager")
    tm.is_alive.return_value = True
    tm.request_fork_slot.return_value = None
    mgr = PoolManager(tm, target_size=2)
    mgr._idle_slots.extend(slots)
    return mgr


# --------------------------------------------------------------------
# the predicate
# --------------------------------------------------------------------

class TestOnlyPositiveEvidenceCounts:
    """``slot_rpc_death`` reports death only when a client says so."""

    def test_a_slot_that_never_served_is_alive(self) -> None:
        # No rpc at all: a freshly forked slot.  There is no channel to
        # have died, and the first session on the slot creates one.
        assert slot_rpc_death(_slot(1, rpc=None)) is None

    def test_a_live_client_is_alive(self) -> None:
        assert slot_rpc_death(_slot(1, rpc=_FakeRPC(closed=False))) is None

    def test_a_closed_client_is_dead_and_names_its_reason(self) -> None:
        dead = _slot(1, rpc=_FakeRPC(closed=True, reason="eof"))
        assert slot_rpc_death(dead) == "eof"

    def test_a_closed_client_with_no_reason_still_reads_as_dead(self) -> None:
        dead = _slot(1, rpc=_FakeRPC(closed=True, reason=None))
        assert slot_rpc_death(dead) == "closed"

    def test_a_client_that_cannot_answer_is_treated_as_alive(self) -> None:
        # The safe direction.  Failing closed on a client that does not
        # implement ``is_closed`` would evict every slot on a daemon
        # running an out-of-tree client -- an outage built out of a
        # health check.
        assert slot_rpc_death(_slot(1, rpc=_OpaqueRPC())) is None


# --------------------------------------------------------------------
# layer 1 -- acquire
# --------------------------------------------------------------------

class TestAcquireNeverServesACorpse:
    """Neither affinity path may hand out a dead slot."""

    def test_a_pure_idle_slot_with_a_closed_client_is_not_served(self) -> None:
        # THE REPORTED SHAPE: a standalone session returns its slot as
        # pure idle (#1033); the client dies; the next standalone session
        # arrives and is handed the corpse.
        mgr = _pool(_slot(5307, rpc=_FakeRPC(closed=True, reason="eof")))
        assert mgr.acquire_slot() is None

    def test_a_live_pure_idle_slot_is_still_served(self) -> None:
        mgr = _pool(_slot(11, rpc=_FakeRPC(closed=False)))
        got = mgr.acquire_slot()
        assert got is not None and got.pid == 11

    def test_the_corpse_does_not_shadow_a_live_slot_behind_it(self) -> None:
        mgr = _pool(
            _slot(5307, rpc=_FakeRPC(closed=True, reason="eof")),
            _slot(12, rpc=_FakeRPC(closed=False)),
        )
        got = mgr.acquire_slot()
        assert got is not None and got.pid == 12

    def test_the_cascade_affinity_path_is_gated_too(self) -> None:
        # Path (1) walks for a whole-key match and returns from inside
        # the lock.  A check written per-path is a check that gets
        # forgotten on one of them; this pins that the cascade path is
        # covered by the same filter.
        mgr = _pool(_slot(
            77, cascade_id="cascade-A",
            rpc=_FakeRPC(closed=True, reason="read-loop-crash"),
        ))
        assert mgr.acquire_slot(cascade_driver_id="cascade-A") is None

    def test_a_live_cascade_affined_slot_is_still_reused(self) -> None:
        mgr = _pool(_slot(78, cascade_id="cascade-A", rpc=_FakeRPC()))
        got = mgr.acquire_slot(cascade_driver_id="cascade-A")
        assert got is not None and got.pid == 78
        assert mgr.get_telemetry()["cascade_slot_reuse_hits_total"] == 1


class TestACorpseIsReapedNotSkipped:
    """Skipping a dead slot leaks a runner AND leaves phantom capacity."""

    def test_the_dropped_slot_is_queued_for_teardown(self) -> None:
        dead = _slot(5307, rpc=_FakeRPC(closed=True, reason="eof"))
        mgr = _pool(dead)
        mgr.acquire_slot()
        assert [s.pid for s in mgr._pending_teardown] == [5307]

    def test_the_corpse_stops_counting_as_capacity(self) -> None:
        # The sharper half of the reason not to merely skip.  A corpse
        # left in ``_idle_slots`` keeps counting toward
        # ``unreserved_idle_count``, so the replenish loop reads the pool
        # as full and never forks the replacement -- capacity no arriving
        # session can use, counted as capacity for everybody (#898).
        mgr = _pool(
            _slot(5307, rpc=_FakeRPC(closed=True, reason="eof")),
            _slot(5308, rpc=_FakeRPC(closed=True, reason="eof")),
        )
        assert mgr.unreserved_idle_count() == 2
        mgr._sweep_dead_slots()
        assert mgr.unreserved_idle_count() == 0

    def test_the_eviction_is_counted(self) -> None:
        mgr = _pool(_slot(5307, rpc=_FakeRPC(closed=True, reason="eof")))
        assert mgr.get_telemetry()["pool_dead_slot_evicted_total"] == 0
        mgr._sweep_dead_slots()
        assert mgr.get_telemetry()["pool_dead_slot_evicted_total"] == 1

    def test_a_healthy_pool_reports_the_counter_as_zero(self) -> None:
        # Declared, not sprung into existence by ``_incr``: an absent key
        # reads as "never measured", which is the one thing it must not
        # say.
        mgr = _pool(_slot(1, rpc=_FakeRPC()))
        assert mgr.get_telemetry()["pool_dead_slot_evicted_total"] == 0

    def test_discarding_an_already_acquired_slot_queues_it(self) -> None:
        # An acquired slot is not in ``_idle_slots``, so nothing in the
        # pool would ever reap it if its consumer walked away.
        mgr = _pool()
        slot = _slot(99, rpc=_FakeRPC(closed=True, reason="eof"))
        mgr.discard_acquired_slot(slot, reason="dead-rpc:eof")
        assert [s.pid for s in mgr._pending_teardown] == [99]


class TestPureIdleSlotsHaveALivenessFloor:
    """The cascade sweep's exemption is about warmth, not health."""

    def test_the_cascade_sweep_still_exempts_a_live_pure_idle_slot(self) -> None:
        # Unchanged behaviour, pinned so the liveness sweep is not
        # mistaken for permission to time pure-idle slots out.
        mgr = _pool(_slot(1, rpc=_FakeRPC(), last_end_ts=0.0))
        mgr._cascade_idle_timeout = 0.0
        mgr._sweep_cascade_idle()
        assert [s.pid for s in mgr._idle_slots] == [1]

    def test_the_replenish_loop_reaps_a_dead_pure_idle_slot(self) -> None:
        # ONE iteration of the REAL loop body, not of the sweep it is
        # supposed to call: the loop is where the sweep has to be wired,
        # and a test that calls the sweep directly cannot notice it
        # being unwired.  Deterministic and clockless -- the fork request
        # stops the loop, so exactly one pass runs.
        mgr = _pool(_slot(5307, rpc=_FakeRPC(closed=True, reason="eof")))
        mgr._replenish_interval = 0.0
        torn: list = []
        mgr._teardown_slot = lambda slot, *, reason: torn.append(
            (slot.pid, reason))

        def _stop_after_one_pass():
            mgr._replenish_stop.set()
            return None

        mgr._template_manager.request_fork_slot.side_effect = \
            _stop_after_one_pass
        mgr._replenish_loop()

        assert mgr._idle_slots == [], (
            "the replenish loop left a dead slot in the pool -- it is the "
            "only thing that runs on its own, so a liveness sweep it does "
            "not call is a sweep that never runs"
        )
        assert torn == [(5307, "dead-rpc:eof")]

    def test_an_over_capacity_slot_keeps_its_own_label(self) -> None:
        # The drain used to label every queued slot "over-capacity".
        # That was true of its only producer and became a lie the moment
        # a second one existed, so the label is per-slot now -- and the
        # original producer still gets the original word.
        mgr = _pool()
        slot = _slot(42, rpc=_FakeRPC())
        with mgr._lock:
            mgr._pending_teardown.append(slot)
        torn: list = []
        mgr._teardown_slot = lambda s, *, reason: torn.append((s.pid, reason))
        mgr._drain_pending_teardown()
        assert torn == [(42, "over-capacity")]

    def test_a_dead_slot_is_reaped_through_the_shared_teardown(self) -> None:
        # ``_drain_pending_teardown`` is what closes the transport and
        # reaps the process; pin that a dead slot reaches it.
        mgr = _pool(_slot(5307, rpc=_FakeRPC(closed=True, reason="eof")))
        torn: list = []
        mgr._teardown_slot = lambda slot, *, reason: torn.append(
            (slot.pid, reason))
        mgr._sweep_dead_slots()
        mgr._drain_pending_teardown()
        assert torn == [(5307, "dead-rpc:eof")]


# --------------------------------------------------------------------
# the capacity path -- where a wrong-slot close would live
# --------------------------------------------------------------------

class TestNoTeardownTouchesASlotStillInThePool:
    """A retained slot's client is never closed by its neighbour's exit.

    The first hypothesis for #1058, and the one worth ruling out
    explicitly: the reported slot returned into a pool already at
    ``idle_count=4/4``, an over-capacity teardown ran ten seconds later,
    and the only close in the log named nobody.  If an eviction could
    close a client belonging to a slot that STAYS in the pool -- a
    shared transport, a wrong slot object, ``slot`` written where
    ``evicted`` was meant -- that would be the root cause, and a
    liveness check would paper over it.

    It cannot, and these pin why: every slot handed to
    ``_teardown_slot`` has already left ``_idle_slots``, on both
    branches of the capacity check and on the cascade sweep.
    """

    def _pool_at_capacity(self, *slots) -> PoolManager:
        mgr = _pool(*slots)
        mgr.max_size = len(slots)
        return mgr

    def test_a_pure_idle_returner_at_capacity_is_the_one_dropped(self) -> None:
        residents = [_slot(1, rpc=_FakeRPC()), _slot(2, rpc=_FakeRPC())]
        mgr = self._pool_at_capacity(*residents)
        returner = _slot(5307, rpc=_FakeRPC())
        assert mgr.return_slot_after_session(returner) is False
        assert [s.pid for s in mgr._idle_slots] == [1, 2]
        assert [s.pid for s in mgr._pending_teardown] == [5307]

    def test_an_affine_returner_displaces_a_resident_that_LEAVES(self) -> None:
        resident = _slot(1, cascade_id="B", rpc=_FakeRPC(), last_end_ts=0.0)
        mgr = self._pool_at_capacity(resident, _slot(2, rpc=_FakeRPC()))
        returner = _slot(5307, cascade_id="A", rpc=_FakeRPC())
        assert mgr.return_slot_after_session(returner) is True
        # The displaced resident is queued for teardown AND gone from
        # the pool.  Queued-but-retained is the wrong-slot close.
        assert [s.pid for s in mgr._pending_teardown] == [1]
        assert 1 not in [s.pid for s in mgr._idle_slots]

    def test_every_slot_reaching_teardown_has_left_the_pool(self) -> None:
        # The invariant itself, over both capacity branches and the
        # cascade sweep, asserted at the moment of teardown rather than
        # after it.
        mgr = self._pool_at_capacity(
            _slot(1, cascade_id="B", rpc=_FakeRPC(), last_end_ts=0.0),
            _slot(2, rpc=_FakeRPC()),
        )
        violations: list = []
        mgr._teardown_slot = lambda slot, *, reason: violations.extend(
            [slot.pid] if any(s is slot for s in mgr._idle_slots) else [])
        mgr.return_slot_after_session(_slot(3, cascade_id="A", rpc=_FakeRPC()))
        mgr.return_slot_after_session(_slot(4, rpc=_FakeRPC()))
        mgr._cascade_idle_timeout = 0.0
        mgr._sweep_dead_slots()
        mgr._sweep_cascade_idle()
        mgr._drain_pending_teardown()
        assert violations == [], (
            f"slot(s) {violations} were torn down while still in "
            f"_idle_slots — their client is closed and the pool will "
            f"hand them out again (#1058)"
        )


class TestOneSlotOneEntry:
    """A slot returned twice must not appear in the pool twice.

    ``JaatoServer.shutdown`` captures and nulls ``_runner_rpc`` /
    ``_spawned_runner`` / ``_pool_manager_ref`` without a lock, and is
    called from session unload, ``session.stop``, the #812 orphan sweep
    and daemon shutdown.  Two concurrent callers therefore CAN both see
    the live triple and both return the same slot.  Two entries share
    one ``rpc``: tear either down and the other is a corpse in the pool
    with no log line naming it.

    Guarded at the pool rather than at the four teardown paths -- the
    pool owns the list, so one check covers callers that do not exist
    yet.  It does not fix the double shutdown; it makes it audible.
    """

    def test_the_second_return_is_refused(self) -> None:
        mgr = _pool()
        slot = _slot(5307, rpc=_FakeRPC())
        assert mgr.return_slot_after_session(slot) is True
        assert mgr.return_slot_after_session(slot) is False
        assert [s.pid for s in mgr._idle_slots] == [5307]

    def test_the_refusal_is_counted(self) -> None:
        mgr = _pool()
        slot = _slot(5307, rpc=_FakeRPC())
        mgr.return_slot_after_session(slot)
        mgr.return_slot_after_session(slot)
        tel = mgr.get_telemetry()
        assert tel["pool_duplicate_return_refused_total"] == 1

    def test_a_distinct_slot_with_the_same_pid_is_not_a_duplicate(self) -> None:
        # Identity, not pid: a recycled pid on a genuinely different
        # slot object is a different runner, and refusing it would
        # shrink the pool for no reason.
        mgr = _pool()
        assert mgr.return_slot_after_session(_slot(5307, rpc=_FakeRPC())) is True
        assert mgr.return_slot_after_session(_slot(5307, rpc=_FakeRPC())) is True
        assert len(mgr._idle_slots) == 2

    def test_the_duplicate_shape_is_what_leaves_a_corpse_in_the_pool(self) -> None:
        # Why the guard is worth having, demonstrated end to end: with
        # the duplicate admitted, acquiring one copy and closing its
        # shared client leaves the other copy servable and dead.  The
        # guard is what stops the pool reaching that state; layer 1 is
        # what stops it being SERVED if it ever does.
        mgr = _pool()
        rpc = _FakeRPC()
        slot = _slot(5307, rpc=rpc)
        mgr._idle_slots.extend([slot, slot])       # the unguarded outcome
        taken = mgr.acquire_slot()
        assert taken is slot
        rpc.is_closed, rpc.close_reason = True, "explicit-close"
        assert mgr.acquire_slot() is None, (
            "the surviving duplicate was served with a closed client"
        )


# --------------------------------------------------------------------
# layer 2 -- the reuse fast path
# --------------------------------------------------------------------

class TestTheReusePathVerifies:
    """``spawn_session_runner`` must not consume a closed client."""

    def test_a_live_client_is_reused(self) -> None:
        from server.runner_spawn import _reusable_slot_rpc
        rpc = _FakeRPC(closed=False)
        slot = _slot(5307, rpc=rpc)
        assert _reusable_slot_rpc(slot, "sess", MagicMock()) == (rpc, False)

    def test_a_closed_client_is_refused_at_the_point_of_use(self) -> None:
        """The check itself, AND that the reuse branch is fed by it.

        Two assertions because they fail separately.  The helper can be
        correct and bypassed -- which is exactly the state the defect was
        in, ``existing_rpc`` coming straight off the slot -- and a test
        of the helper alone cannot see that.  The source assertion is the
        cheapest thing that can: driving ``spawn_session_runner`` end to
        end needs a runner subprocess and a daemon loop, and would pin
        far more than the one line that matters.
        """
        import inspect

        from server import runner_spawn
        from server.runner_spawn import _reusable_slot_rpc

        slot = _slot(5307, rpc=_FakeRPC(closed=True, reason="eof"))
        assert _reusable_slot_rpc(slot, "sess", MagicMock()) == (None, True)

        src = inspect.getsource(runner_spawn.spawn_session_runner)
        assert "_reusable_slot_rpc(" in src, (
            "the reuse fast-path no longer takes its client through the "
            "liveness check.  A slot's rpc consumed unconditionally is "
            "#1058."
        )
        assert 'getattr(pool_slot, "rpc"' not in src, (
            "the reuse fast-path reads the slot's rpc directly again, "
            "bypassing the liveness check beside it (#1058)."
        )

    def test_the_refused_slot_is_handed_back_for_teardown(self) -> None:
        from server.runner_spawn import _reusable_slot_rpc
        mgr = _pool()
        slot = _slot(5307, rpc=_FakeRPC(closed=True, reason="eof"))
        assert _reusable_slot_rpc(slot, "sess", mgr) == (None, True)
        assert [s.pid for s in mgr._pending_teardown] == [5307]

    def test_a_first_session_on_a_slot_reuses_nothing(self) -> None:
        from server.runner_spawn import _reusable_slot_rpc
        assert _reusable_slot_rpc(
            _slot(5307, rpc=None), "s", MagicMock()) == (None, False)

    def test_a_cold_spawned_session_has_no_slot_to_check(self) -> None:
        from server.runner_spawn import _reusable_slot_rpc
        assert _reusable_slot_rpc(None, "s", MagicMock()) == (None, False)

    def test_an_unwired_pool_manager_still_refuses_the_corpse(self) -> None:
        # Refusing the reuse beats handing back a closed client even
        # where nothing can reap the slot.
        from server.runner_spawn import _reusable_slot_rpc
        slot = _slot(5307, rpc=_FakeRPC(closed=True, reason="eof"))
        assert _reusable_slot_rpc(slot, "sess", None) == (None, True)


class TestACancelledReaderIsReopened:
    """A cancelled read task is repaired, not merely detected.

    ``close()`` reaps the runner, so a client it closed has nothing left
    to talk to.  A CANCELLED read task is the opposite: the runner is
    alive, the socket is open, the writer works -- only the task
    consuming frames is gone, and a task cannot be un-cancelled.  So the
    repair is a NEW read task on the surviving transport, and clearing
    the boolean would hand back a channel nobody reads.

    ``close()`` can never produce this state: it stamps
    ``"explicit-close"`` before cancelling, and the first cause wins.
    So ``close_reason == "cancelled"`` is a statement of PROVENANCE --
    something outside the client cancelled it.
    """

    def test_the_pool_does_not_bury_a_revivable_slot(self) -> None:
        # Layer 1 must not reap what layer 2 can repair; the pool holds
        # no event loop, so it cannot do the repair itself.
        assert slot_rpc_death(_slot(1, rpc=_RevivableRPC())) is None

    def test_a_revivable_slot_is_still_acquirable(self) -> None:
        mgr = _pool(_slot(5307, rpc=_RevivableRPC()))
        got = mgr.acquire_slot()
        assert got is not None and got.pid == 5307

    def test_the_reuse_path_reopens_it_before_handing_it_over(self) -> None:
        from server.runner_spawn import _reusable_slot_rpc
        rpc = _RevivableRPC()
        slot = _slot(5307, rpc=rpc)
        loop = asyncio.new_event_loop()
        import threading
        threading.Thread(target=loop.run_forever, daemon=True).start()
        try:
            got, discarded = _reusable_slot_rpc(slot, "sess", _pool(), loop)
        finally:
            loop.call_soon_threadsafe(loop.stop)
        assert got is rpc and discarded is False
        assert rpc.revived == 1
        assert rpc.is_closed is False

    def test_a_revive_that_refuses_falls_back_to_discarding(self) -> None:
        from server.runner_spawn import _reusable_slot_rpc
        rpc = _RevivableRPC(revive_ok=False)
        slot = _slot(5307, rpc=rpc)
        mgr = _pool()
        loop = asyncio.new_event_loop()
        import threading
        threading.Thread(target=loop.run_forever, daemon=True).start()
        try:
            got, discarded = _reusable_slot_rpc(slot, "sess", mgr, loop)
        finally:
            loop.call_soon_threadsafe(loop.stop)
        assert got is None and discarded is True
        assert [s.pid for s in mgr._pending_teardown] == [5307]

    def test_no_loop_means_discard_not_a_wrong_answer(self) -> None:
        from server.runner_spawn import _reusable_slot_rpc
        mgr = _pool()
        slot = _slot(5307, rpc=_RevivableRPC())
        assert _reusable_slot_rpc(slot, "sess", mgr, None) == (None, True)
        assert [s.pid for s in mgr._pending_teardown] == [5307]

    def test_an_eof_client_is_never_revived(self) -> None:
        # The peer is gone.  A reader on a socket whose runner is a
        # corpse reads EOF and closes again -- the same failure, one
        # layer later.
        from server.runner_spawn import _reusable_slot_rpc
        rpc = _RevivableRPC()
        rpc.close_reason, rpc.can_revive = "eof", False
        got, discarded = _reusable_slot_rpc(
            _slot(5307, rpc=rpc), "s", _pool(), None)
        assert got is None and discarded is True and rpc.revived == 0


# --------------------------------------------------------------------
# the client's own account of how it died
# --------------------------------------------------------------------

def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


#: Why every socketpair client below is built with
#: ``runner_pid=os.getpid()`` and never ``0``: ``close()`` reaps with
#: ``os.waitpid(runner_pid, WNOHANG)`` and escalates with
#: ``os.kill(runner_pid, ...)``, and in both calls **0 means the caller's
#: whole process group**.  A test passing 0 reaps any child of the pytest
#: process that happens to exit inside the close window -- which is how a
#: test over here makes a real-subprocess test over there fail on a pid it
#: no longer owns -- and, past the timeout, SIGTERMs the entire run.  The
#: pytest process's own pid is not our child, so ``waitpid`` answers
#: ``ChildProcessError`` immediately and the ladder is never reached.  It
#: is also the convention ``server/test_runner_rpc_client.py`` already
#: uses for its socketpair-only client.


class TestTheClientRecordsHowItDied:
    """``close_reason`` tells the two incidents apart.

    "The runner died" (``eof``) and "the read loop raised"
    (``read-loop-crash``) have different operator responses and used to
    leave identical evidence -- ``_closed = True`` and nothing else.
    """

    def test_a_fresh_client_is_live_and_nameless(self) -> None:
        from server.runner_rpc_client import RunnerRPCClient

        async def go():
            a, b = socket.socketpair()
            c = RunnerRPCClient(a, runner_pid=os.getpid())
            await c.start()
            try:
                assert c.is_closed is False
                assert c.close_reason is None
            finally:
                b.close()
                await c.close(timeout=1)
        _drive(go())

    def test_eof_is_recorded_as_eof(self) -> None:
        from server.runner_rpc_client import RunnerRPCClient

        async def go():
            a, b = socket.socketpair()
            c = RunnerRPCClient(a, runner_pid=os.getpid())
            await c.start()
            b.close()                      # the runner goes away
            for _ in range(200):
                if c.is_closed:
                    break
                await asyncio.sleep(0.005)
            assert c.is_closed is True, "read loop never noticed EOF"
            assert c.close_reason == "eof"
        _drive(go())

    def test_an_explicit_close_says_so(self) -> None:
        from server.runner_rpc_client import RunnerRPCClient

        async def go():
            a, b = socket.socketpair()
            c = RunnerRPCClient(a, runner_pid=os.getpid())
            await c.start()
            await c.close(timeout=1)
            b.close()
            assert c.is_closed is True
            assert c.close_reason == "explicit-close"
        _drive(go())

    def test_an_explicitly_closed_client_is_never_revivable(self) -> None:
        # close() stamps its reason BEFORE cancelling the read task, so
        # the cancellation the finally sees never wins -- which is what
        # keeps ``can_revive`` a statement about EXTERNAL cancellation
        # and stops it offering to revive a client whose runner close()
        # has already reaped.
        from server.runner_rpc_client import RunnerRPCClient

        async def go():
            a, b = socket.socketpair()
            c = RunnerRPCClient(a, runner_pid=os.getpid())
            await c.start()
            await c.close(timeout=1)
            b.close()
            assert c.close_reason == "explicit-close"
            assert c.can_revive is False
        _drive(go())

    def test_a_cancelled_reader_is_revivable_and_revives(self) -> None:
        # The one shape that can be repaired, driven on a real socket.
        from server.runner_rpc_client import RunnerRPCClient

        async def go():
            a, b = socket.socketpair()
            c = RunnerRPCClient(a, runner_pid=os.getpid())
            await c.start()
            await asyncio.sleep(0)                # let the reader start
            c._read_task.cancel()                 # an outside canceller
            for _ in range(200):
                if c.is_closed:
                    break
                await asyncio.sleep(0.005)
            assert c.is_closed is True
            assert c.close_reason == "cancelled"
            assert c.can_revive is True
            assert await c.revive_read_loop() is True
            assert c.is_closed is False
            assert c.close_reason is None
            b.close()
            for _ in range(200):                  # the new reader works
                if c.is_closed:
                    break
                await asyncio.sleep(0.005)
            assert c.close_reason == "eof", (
                "the revived read task never ran"
            )
        _drive(go())

    def test_a_reader_cancelled_before_it_started_is_still_recorded(self) -> None:
        """The one exit the loop body's ``finally`` cannot cover.

        ``Task.cancel()`` before the task's first step never enters the
        coroutine, so no ``finally`` runs.  Left unrecorded, the channel
        reports itself HEALTHY with no reader: writes land, replies are
        never consumed, every call waits out its deadline.  That is
        worse than a closed client, because the pool's liveness poll
        answers "alive".
        """
        from server.runner_rpc_client import RunnerRPCClient

        async def go():
            a, b = socket.socketpair()
            c = RunnerRPCClient(a, runner_pid=os.getpid())
            await c.start()
            c._read_task.cancel()        # NO sleep(0): never scheduled
            await asyncio.sleep(0.01)
            assert c._read_task.cancelled() is True
            assert c.is_closed is True, (
                "a channel with no reader reported itself healthy"
            )
            assert c.close_reason == "cancelled"
            assert c.can_revive is True
            b.close()
        _drive(go())

    def test_a_dead_transport_refuses_the_revive(self) -> None:
        from server.runner_rpc_client import RunnerRPCClient

        async def go():
            a, b = socket.socketpair()
            c = RunnerRPCClient(a, runner_pid=os.getpid())
            await c.start()
            await asyncio.sleep(0)
            c._read_task.cancel()
            for _ in range(200):
                if c.is_closed:
                    break
                await asyncio.sleep(0.005)
            c._writer.close()                     # transport gone
            assert c.can_revive is False
            assert await c.revive_read_loop() is False
            assert c.is_closed is True
            b.close()
        _drive(go())

    def test_a_closed_client_is_what_the_pool_predicate_reads(self) -> None:
        # The two halves joined: a real client that died by EOF is what
        # ``slot_rpc_death`` reports on, with the cause carried through.
        from server.runner_rpc_client import RunnerRPCClient

        async def go():
            a, b = socket.socketpair()
            c = RunnerRPCClient(a, runner_pid=os.getpid())
            await c.start()
            b.close()
            for _ in range(200):
                if c.is_closed:
                    break
                await asyncio.sleep(0.005)
            return c
        client = _drive(go())
        mgr = _pool(_slot(5307, rpc=client))
        assert slot_rpc_death(mgr._idle_slots[0]) == "eof"
        assert mgr.acquire_slot() is None
