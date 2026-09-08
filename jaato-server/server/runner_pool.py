"""Pool slot manager for pre-warm runner pool (pool PR 3).

Owns the collection of pre-forked pool slots between the template
subprocess (PR 2) and the session-routing layer (PR 4).

Each pool slot is:
  - A child process forked from the template (inherits warm imports)
  - Identified by its PID
  - Reachable via a per-slot socket on the daemon side

PR 3 ships only the pool's idle-management:
  - ``spawn_initial_slots(n)`` — ask the template to fork N slots
    at daemon startup
  - ``shutdown_all()`` — politely SHUTDOWN every idle slot at daemon
    exit
  - Idle-slot tracking dict

PR 4 will add:
  - ``acquire_slot()`` — pop an idle slot off the pool, return it
    to the session-routing code to send a bootstrap envelope
  - Replenishment thread that asks the template for a new slot
    whenever the pool drops below the target size

This module is **daemon-tier** infrastructure, like
``runner_template.py``.  Not a plugin in the discovery sense.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# Cascade-sharing idle timeout (Phase 2).  An IDLE_FOR_CASCADE slot
# whose last_session_end_ts is older than this gets torn down by the
# replenish loop.  Default matches design doc §3 decision 5 (300s).
DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS = 300.0


@dataclass
class PoolSlot:
    """Pre-warm pool slot — daemon's handle on a template-forked child.

    Replaces the Phase 1 ``Tuple[int, socket.socket]`` shape.  Carries
    cascade-affinity bookkeeping for Phase 2 slot reuse.

    Attributes:
        pid: Slot process id.  Template-child; reaped via waitpid by
            the daemon's subreaper bit (or by shutdown_all on exit).
        sock: Daemon-side end of the slot's RPC socket.  Closing this
            socket signals the slot to exit (peer-EOF).
        config_root: The ``.jaato`` config root the slot's WARM PLUGIN STATE
            was built from.  Reuse requires a match.

            ``cascade_driver_id`` alone was the whole reuse key, so two
            sessions differing in config root -- different profiles, agents,
            prompt library, permission config -- shared a slot as long as the
            cascade id matched.  The path variables self-heal (the runner
            re-runs ``set_config_root`` every bootstrap); what does not is
            everything DERIVED from that root during the first session's
            bootstrap, which the pool exists to keep warm.  A slot's identity
            has to include what the slot actually carries.

            ``None`` for a slot that has never served.
        cascade_id: Optional cascade-driver ID this slot is currently
            affined to.  ``None`` for a fresh slot or a slot that has
            never served a session (PURE IDLE).  Set when a session
            with ``cascade_driver_id != None`` is acquired against
            this slot; cleared only on slot teardown (slots never
            switch cascades mid-life).
        last_session_end_ts: Wall-clock monotonic timestamp of the
            most recent ``session_end`` RPC for this slot.  ``None``
            until the slot has served (+ returned from) at least one
            session.  Used by the idle-teardown sweep to detect
            cascades that have gone idle longer than
            ``cascade_idle_timeout_seconds``.
    """

    pid: int
    sock: socket.socket
    cascade_id: Optional[str] = None
    config_root: Optional[str] = None
    last_session_end_ts: Optional[float] = None
    # Phase 3 cascade-sharing (server 0.6.146+): identifier of the
    # most recent session this slot served.  Set when the slot
    # returns to the pool after a successful ``session_end`` RPC
    # (JaatoServer.shutdown sets it via the
    # ``return_slot_after_session`` call site).  The NEXT session
    # to acquire this slot uses this value to
    # ``apparmor_parser --remove`` the prior session's profile
    # AFTER its own ``aa_change_profile`` transition succeeds.
    # ``None`` for slots that have never served a session.
    last_session_id: Optional[str] = None
    # Phase 3 cascade-sharing hotfix (server 0.6.150+): the
    # ``RunnerRPCClient`` adopted onto this slot's socket.  Created
    # ONCE per slot lifetime by the first session's spawn helper;
    # reset via ``reset_for_slot_reuse`` between cascade sessions;
    # closed at slot teardown (cascade-idle sweep OR daemon shutdown).
    #
    # Why the rpc lives on the slot, not on JaatoServer: the asyncio
    # transport that ``RunnerRPCClient.start`` adopts via
    # ``loop.connect_accepted_socket`` binds the socket exclusively.
    # Creating a second ``RunnerRPCClient`` on the same socket fails;
    # session 2 spawn falls back to in-process and ``server._runner_rpc``
    # is None — observed v152-retry-4 as
    # ``NoneType.session_send_message_threadsafe``.  See PR #173.
    #
    # Typed ``Any`` to avoid the cycle through ``runner_rpc_client``.
    rpc: Optional[Any] = None


# Phase 1 alias.  All in-tree callers were updated in Phase 2 to use
# ``PoolSlot``; the alias remains so external imports of
# ``server.runner_pool.SlotHandle`` keep type-checking.  No tuple
# semantics — ``PoolSlot`` is a dataclass with ``.pid`` / ``.sock``.
SlotHandle = PoolSlot


class PoolManager:
    """Manages the pre-warm pool of forked-from-template slots.

    Lifecycle:

      daemon startup ─→ template spawned (PR 2)
                         │
                         ▼
      daemon startup ─→ PoolManager(template_mgr).spawn_initial_slots(n=2)
                         │  forks N slots via template's FORK_SLOT
                         ▼
                  (N slots sit idle, waiting for bootstrap envelopes)
                         │
                         ▼
      session arrives ─→ acquire_slot() (PR 4)
                         │
                         ▼
                  (slot serves the session; replenish in background)
                         │
                         ▼
      daemon shutdown ─→ shutdown_all() — SHUTDOWN every idle slot

    Thread-safe — concurrent ``acquire_slot``/``shutdown_all`` calls
    serialize via the manager's lock.

    Attributes:
        target_size: Floor on UNRESERVED idle slots — those with no
            cascade affinity, which any arriving session may take
            (default 2).
        max_size: Ceiling on TOTAL idle slots, reservations included
            (default ``2 * target_size``).  Floor and ceiling count
            different things on purpose: a cascade-affined idle slot
            is capacity for exactly one tenant, so counting it as pool
            capacity starves everybody else (#898).
        _idle_slots: List of currently-idle slot handles.
    """

    def __init__(
        self,
        template_manager,
        target_size: int = 2,
        replenish_interval: float = 0.5,
        cascade_idle_timeout_seconds: float = DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS,
        max_size: Optional[int] = None,
    ) -> None:
        """Initialize the pool manager.

        Args:
            template_manager: The daemon's :class:`TemplateManager`.
                Source of fork-slot requests; must already have
                ``spawn()`` been called.
            target_size: Number of **unreserved** idle slots to keep
                available — slots any arriving session may take,
                i.e. those with no cascade affinity stamped.  Default
                2 — reasonable for typical workstation; cascade
                harnesses spawning many concurrent sessions can raise
                this via ``JAATO_RUNNER_POOL_SIZE`` env var (consumed
                by daemon ``__main__.py`` and threaded here).  Values
                <= 0 disable the pool (sessions fall back to cold-spawn
                session-mode).

                UNRESERVED, not total (#898).  A cascade-affined idle
                slot is a RESERVATION: cross-cascade reuse is forbidden
                by design, so such a slot is capacity for exactly one
                tenant and is unavailable to everybody else.  Counting
                it as pool capacity is what let a pool read "full" while
                being empty from the point of view of every tenant but
                one — two idle slots affined to cascade B, and cascade
                A's ``acquire_slot`` returning ``None`` with nothing in
                the system that would ever create a slot A could use.
            max_size: Hard ceiling on TOTAL idle slots (unreserved +
                reservations).  ``None`` (default) means
                ``2 * target_size``.  This is what bounds the growth
                that the unreserved-accounting above implies: without
                it, one reservation per live cascade accumulates
                without limit, and a slot is 129-187 MB by this file's
                own measurement.  Clamped up to ``target_size`` — a
                ceiling below the floor is not a policy.  Operators
                raise it via ``JAATO_RUNNER_POOL_MAX_SIZE``.
            replenish_interval: Seconds the replenishment thread sleeps
                between idle-count checks.  Default 0.5s — fast enough
                that a 6-step cascade refilling slots between steps
                doesn't perceive replenishment latency, slow enough
                that the thread doesn't burn CPU when the pool is at
                target size.  Pool PR 4.
            cascade_idle_timeout_seconds: Phase 2 — how long an
                IDLE_FOR_CASCADE slot may sit idle before the
                replenish loop tears it down.  Default 300s (per
                design doc §3 decision 5).  Override for tests that
                exercise the teardown path without sleeping for 5
                minutes.
        """
        self._template_manager = template_manager
        self.target_size = max(0, int(target_size))
        #: Hard ceiling on ``len(self._idle_slots)``.  ``target_size``
        #: is the floor on the UNRESERVED subset; this is the ceiling on
        #: the whole list, and the two are deliberately different
        #: numbers because reservations sit on top of the floor rather
        #: than consuming it (#898).
        self.max_size = (
            max(self.target_size, int(max_size))
            if max_size is not None else self.target_size * 2
        )
        #: Slots dropped by the capacity check, awaiting teardown by the
        #: replenish thread.  They are already out of ``_idle_slots``, so
        #: the pool count is bounded the moment a slot is queued here.
        self._pending_teardown: List[PoolSlot] = []
        self._idle_slots: List[PoolSlot] = []
        self._lock = threading.Lock()
        # Pool PR 4 replenishment thread state.
        self._replenish_interval = float(replenish_interval)
        self._replenish_stop = threading.Event()
        self._replenish_thread: Optional[threading.Thread] = None
        # Phase 2 cascade-sharing idle timeout.
        self._cascade_idle_timeout = float(cascade_idle_timeout_seconds)
        # Pool PR 5d telemetry counters.  Monotonically-incrementing
        # process-lifetime totals.  Snapshot via :meth:`get_telemetry`
        # for diagnostic surfaces (logs, admin commands, OTel
        # exporters wired by external code).
        self._counters: Dict[str, int] = {
            # Number of times ``acquire_slot`` returned a real handle.
            "pool_slot_acquired_total": 0,
            # Number of times ``acquire_slot`` returned None (pool
            # empty when called).  Sessions in this state fall back
            # to cold-spawn — useful for sizing decisions.
            "pool_acquire_miss_total": 0,
            # Number of times the replenishment thread forked a new
            # slot to top up the pool.
            "pool_replenish_success_total": 0,
            # Number of times ``request_fork_slot`` returned None
            # during replenishment (template stuck, IPC error, etc.).
            "pool_replenish_failures_total": 0,
            # Number of times the watchdog initiated a template
            # respawn.  Each unit of this counter corresponds to one
            # detected template death.
            "template_respawn_attempts_total": 0,
            # Number of those respawn attempts that raised an
            # exception (template_manager.spawn() failed).  Watchdog
            # retries on next iteration, so attempts >> failures is
            # normal during a flaky-template incident.
            "template_respawn_failures_total": 0,
            # Phase 2 cascade-sharing.  Number of times acquire_slot
            # found a slot already affined to the requested cascade
            # and reused it (warm pool slot AND warm plugin state
            # AND warm LSP server connections).
            "cascade_slot_reuse_hits_total": 0,
            # Phase 2.  Number of times acquire_slot received a
            # cascade_driver_id but no IDLE_FOR_CASCADE(C) slot
            # existed — fell through to a PURE-IDLE slot or to
            # cold-spawn.  Pairs with hits_total to measure
            # cascade-reuse efficacy on the workload.
            "cascade_slot_reuse_misses_total": 0,
            # Phase 2.  Number of cascade-affined idle slots torn
            # down by the replenish-loop sweep because they exceeded
            # ``cascade_idle_timeout_seconds`` without serving a new
            # session.  Should match the number of cascades that
            # actually finished (long-pause cascades within timeout
            # don't count).
            "cascade_slots_idle_torndown_total": 0,
            # Slots dropped by the capacity check (queued for
            # teardown), and the subset of those actually reclaimed by
            # the replenish thread's drain.  Declared here rather than
            # sprung into existence by ``_incr`` so ``get_telemetry``
            # reports 0 instead of omitting the key on a daemon that
            # has never hit the ceiling.
            "pool_slots_over_cap_total": 0,
            "pool_slots_over_cap_torndown_total": 0,
            # #898.  Times a RESERVATION (cascade-affined idle slot)
            # was evicted to admit a returning slot whose cascade is
            # demonstrably live.  Growth in this counter alongside
            # ``cascade_slot_reuse_misses_total`` means ``max_size`` is
            # too small for the number of concurrent tenants.
            "pool_stale_reservation_evicted_total": 0,
            # #898.  Times the replenish loop wanted to fork an
            # unreserved slot (unreserved < target_size) and could not
            # because the pool was at ``max_size``.  THE sizing signal
            # for a multi-tenant daemon: a nonzero value means some
            # tenant is being served by cold-spawn while reservations
            # hold the ceiling.  Raise ``JAATO_RUNNER_POOL_MAX_SIZE``.
            "pool_replenish_ceiling_blocked_total": 0,
        }
        self._counters_lock = threading.Lock()

    def spawn_initial_slots(self) -> int:
        """Fork ``target_size`` slots from the template.

        Called once at daemon startup, AFTER the template has
        finished its plugin discovery (which is asynchronous in
        the template — daemon should wait briefly or hope the
        template is fast enough).  Failures are individually
        non-fatal; the pool ends up smaller than target.

        PR 3 ships this as a synchronous loop at daemon startup —
        callers ride out the blocking time.  PR 5 will move
        replenishment into a background thread + add the
        slot-replenishment trigger on acquire.

        Returns:
            Number of slots successfully forked (≤ ``target_size``).
        """
        if self.target_size <= 0:
            logger.info(
                "PoolManager: target_size=%d; pool disabled",
                self.target_size,
            )
            return 0

        forked = 0
        with self._lock:
            for i in range(self.target_size):
                raw = self._template_manager.request_fork_slot()
                if raw is None:
                    logger.warning(
                        "PoolManager: fork-slot %d/%d failed; pool "
                        "will be smaller than target.  Sessions fall "
                        "back to cold-spawn session-mode when pool is "
                        "empty.", i + 1, self.target_size,
                    )
                    break
                pid, sock = raw
                self._idle_slots.append(PoolSlot(pid=pid, sock=sock))
                forked += 1

        logger.info(
            "PoolManager: spawned %d pool slot(s); target=%d",
            forked, self.target_size,
        )
        return forked

    def idle_count(self) -> int:
        """Return the current count of idle slots — reservations included.

        This is the number ``max_size`` bounds.  It is NOT the number a
        waiter can draw on: see :meth:`unreserved_idle_count`.
        """
        with self._lock:
            return len(self._idle_slots)

    def unreserved_idle_count(self) -> int:
        """Return the count of idle slots usable by ANY arriving session.

        A slot carrying a ``cascade_id`` is a reservation — cross-cascade
        reuse is forbidden (warm plugin state belongs to the original
        cascade), so it is capacity for one tenant and for nobody else.
        Only PURE-IDLE slots are the pool's free capacity, and this is
        the quantity the replenish loop compares against
        ``target_size`` (#898).

        Reading total idle there instead meant a pool holding two
        B-affined slots read "full" while ``acquire_slot(cascade=A)``
        returned ``None`` — no affine match, no pure-idle, and no
        replenishment that would ever produce one.  A starved until B
        released, which for the reporting incident was 31 s past the
        client's 60 s budget.
        """
        with self._lock:
            return sum(1 for s in self._idle_slots if s.cascade_id is None)

    def acquire_slot(
        self, cascade_driver_id: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Optional[PoolSlot]:
        """Pop an idle slot off the pool — cascade-affinity aware (Phase 2).

        Affinity routing (per design doc §4.2 claim flow):

        1. If ``cascade_driver_id`` is provided, walk the idle list
           for the first slot whose ``cascade_id`` matches.  Match
           → return it (slot stays affined to its cascade; reuse hit).
        2. Else (no cascade requested OR no match), take any PURE
           IDLE slot (``cascade_id is None``).  If the caller did
           supply a cascade_id, stamp it on the returned slot — the
           slot is now affined to that cascade for the rest of its
           life.
        3. If no idle slot exists, return ``None``.  Caller falls
           back to cold-spawn (which spawns a fresh session-mode
           runner, NOT a pool slot).

        Caller becomes responsible for the slot — must either send
        a bootstrap envelope (the daemon's spawn_session_runner path)
        or close the daemon-side socket (which signals the slot to
        exit).

        Telemetry (in addition to the Phase 1 counters):
          - ``cascade_slot_reuse_hits_total`` incremented on path (1).
          - ``cascade_slot_reuse_misses_total`` incremented when
            ``cascade_driver_id`` is supplied but path (1) didn't fire
            (whether or not we fell through to a PURE IDLE on path
            (2)).  Pairs with hits_total to score cascade-reuse
            efficacy on the workload.

        Args:
            cascade_driver_id: Optional cascade tenant ID.  ``None``
                (default) means "no cascade affinity" — behaves like
                Phase 1 (any IDLE slot, FIFO).

        Returns:
            A :class:`PoolSlot` (with ``cascade_id`` set to the
            requested cascade if reuse fired OR if a PURE IDLE slot
            was stamped) or ``None`` if the pool is empty.
        """
        with self._lock:
            # Path (1): cascade-affinity match.
            if cascade_driver_id is not None:
                for i, slot in enumerate(self._idle_slots):
                    # BOTH must match.  The cascade id says "same cascade";
                    # the config root says "same warm state".  Matching on the
                    # id alone let a session reuse a slot warmed from a
                    # different .jaato -- different profiles, agents, prompts,
                    # permission config -- and inherit whatever the first
                    # session's bootstrap had already derived from it.
                    if (slot.cascade_id == cascade_driver_id
                            and slot.config_root == config_root):
                        slot = self._idle_slots.pop(i)
                        self._incr("pool_slot_acquired_total")
                        self._incr("cascade_slot_reuse_hits_total")
                        logger.info(
                            "PoolManager.acquire_slot: cascade reuse "
                            "HIT — slot pid=%d cascade=%s",
                            slot.pid, cascade_driver_id,
                        )
                        return slot
                # Cascade requested but no match.
                self._incr("cascade_slot_reuse_misses_total")

            # Path (2) / (3): any PURE IDLE slot, else None.  PURE
            # IDLE is preferred over cascade-affined-to-other-cascade
            # because cross-cascade reuse is forbidden by design
            # (warm plugin state belongs to the original cascade).
            pure_idle_idx = next(
                (i for i, s in enumerate(self._idle_slots)
                 if s.cascade_id is None),
                None,
            )
            if pure_idle_idx is None:
                self._incr("pool_acquire_miss_total")
                return None
            slot = self._idle_slots.pop(pure_idle_idx)

        # Stamp cascade affinity if caller supplied one.  Slot is
        # now affined for the rest of its life.
        if cascade_driver_id is not None:
            slot.cascade_id = cascade_driver_id
            # Stamped together: the pair IS the slot's identity, and a slot
            # affined to a cascade without recording what it was warmed from
            # is a slot whose reuse check cannot be right.
            slot.config_root = config_root
            logger.info(
                "PoolManager.acquire_slot: cascade reuse MISS — fresh "
                "slot pid=%d stamped cascade=%s",
                slot.pid, cascade_driver_id,
            )
        self._incr("pool_slot_acquired_total")
        return slot

    def return_slot_after_session(self, slot: PoolSlot) -> None:
        """Return a slot to the idle pool after its session ended (Phase 2).

        Pre-condition: caller has ALREADY invoked the slot's
        ``session_end`` RPC successfully (which fired
        ``reset_for_next_session`` on each plugin).  This method is
        purely the pool-side bookkeeping — it does NOT call the
        slot's RPC.  On ``session_end`` failure, caller must NOT
        invoke this method; the slot should be torn down via
        ``slot.sock.close()`` + waitpid (the slot is in undefined
        state and cannot be reused).

        Stamps ``last_session_end_ts`` to the current monotonic
        wall-clock for the idle-teardown sweep.

        Args:
            slot: The slot to return.  ``slot.cascade_id`` controls
                whether the slot becomes IDLE_FOR_CASCADE (cascade_id
                set) or PURE IDLE (cascade_id None — only legal if
                the slot was never stamped with a cascade, i.e. it
                served a standalone session).
        """
        slot.last_session_end_ts = time.monotonic()
        evicted: Optional[PoolSlot] = None
        stale_reservation_evicted = False
        with self._lock:
            if len(self._idle_slots) >= self.max_size:
                # AT CAPACITY.  ``target_size`` said how many to keep at
                # LEAST and nothing said how many at most, so this list
                # grew by one runner per cleanly-ended pool session and
                # never shrank while the daemon lived.  Measured live at
                # 32 slots / 4237 MB, surfacing as ``pass show`` failing
                # because gpg could not fork -- a resource leak wearing a
                # credentials error.
                #
                # The ceiling is ``max_size``, not ``target_size``:
                # ``target_size`` is the floor on the UNRESERVED subset,
                # and reservations sit on top of it (#898).
                #
                # DEFAULT: the RETURNING slot goes.  It has just served a
                # session, so it carries that session's accumulated heap
                # (measured 129-187 MB on long-lived trees), while a
                # resident may still be a CoW-cheap template fork.  Keeping
                # what is already here also avoids churning the pool on
                # every return once it is full.
                victim_ix = (
                    self._pick_capacity_victim()
                    if slot.cascade_id is not None else None
                )
                if victim_ix is not None:
                    evicted = self._idle_slots.pop(victim_ix)
                    stale_reservation_evicted = evicted.cascade_id is not None
                    self._idle_slots.append(slot)
                else:
                    evicted = slot
            else:
                self._idle_slots.append(slot)
        if stale_reservation_evicted:
            self._incr("pool_stale_reservation_evicted_total")

        if evicted is not None:
            # Handed to the replenish thread rather than torn down here.
            # ``_teardown_slot`` blocks on the daemon loop, and THIS path
            # runs wherever ``JaatoServer.shutdown`` runs.  The count is
            # already bounded above; only the process teardown is deferred,
            # by at most one replenish interval (~0.5s).
            with self._lock:
                self._pending_teardown.append(evicted)
            self._incr("pool_slots_over_cap_total")
            logger.info(
                "PoolManager: pool at capacity (%d/%d); slot pid=%d "
                "(cascade=%s) queued for teardown",
                len(self._idle_slots), self.max_size,
                evicted.pid, evicted.cascade_id or "(pure)",
            )
        logger.info(
            "PoolManager.return_slot_after_session: slot pid=%d "
            "returned to pool (cascade=%s; idle_count=%d/%d, "
            "unreserved=%d/%d)",
            slot.pid, slot.cascade_id or "(pure)",
            len(self._idle_slots), self.max_size,
            self.unreserved_idle_count(), self.target_size,
        )

    def _pick_capacity_victim(self) -> Optional[int]:
        """Choose which resident an AFFINE returning slot displaces.

        Caller holds ``self._lock``.  Returns an index into
        ``self._idle_slots``, or ``None`` when the returner should be
        the one dropped.  Only consulted when the returner carries a
        ``cascade_id``: a returner with none has no claim on the pool
        that a resident does not also have, so it goes.

        LIVENESS OUTRANKS WARMTH (#898).  The rule used to be "an affine
        slot displaces a PURE-IDLE resident, otherwise the returner
        goes", which fires only against *unaffiliated* residents.  When
        both residents were affined to a second cascade, the returner --
        belonging to a cascade demonstrably mid-run, whose next stage
        was already queued -- was destroyed to preserve two slots of a
        cascade that might never come back.

        So the order is:

        1. **The stalest RESERVATION** (largest time since its
           ``last_session_end_ts``; a reservation that never served
           sorts first, since it carries no warm state at all).  Losing
           a reservation costs its cascade a cold plugin bootstrap on
           its next stage; that cascade still RUNS, because it falls
           through to a pure-idle slot.  A slot idle long enough to be
           the stalest is also the one the 300 s cascade-idle sweep was
           going to reap anyway.
        2. **A pure-idle resident**, when there is no reservation to
           drop.  This is the original rule and its original reason: a
           pure-idle slot carries no warm state, so trading it for a
           slot that does keeps the warm path the pool exists to serve.
        3. **Neither** (``None``) -- the pool holds nothing at all, so
           the returner is the only candidate.

        Note what (1) does NOT do: it never leaves a tenant with no way
        to run.  Reservations are the thing being spent, and a cascade
        that loses one takes an unreserved slot instead -- which
        ``target_size`` keeps stocked.  That is the whole trade: warm
        state for one tenant is negotiable, capacity for every tenant
        is not.
        """
        reservations = [
            (i, s) for i, s in enumerate(self._idle_slots)
            if s.cascade_id is not None
        ]
        if reservations:
            return min(
                reservations,
                key=lambda pair: (
                    pair[1].last_session_end_ts
                    if pair[1].last_session_end_ts is not None
                    else float("-inf")
                ),
            )[0]
        return next(
            (i for i, s in enumerate(self._idle_slots)
             if s.cascade_id is None),
            None,
        )

    def _incr(self, key: str, delta: int = 1) -> None:
        """Atomic-ish bump of a telemetry counter.  Internal helper."""
        with self._counters_lock:
            self._counters[key] = self._counters.get(key, 0) + delta

    def get_telemetry(self) -> Dict[str, int]:
        """Return a snapshot of the pool's telemetry counters.

        Snapshot semantics: caller gets a stable dict copy at the
        moment of the call.  Counters keep incrementing concurrently
        on the replenishment thread + acquire path; subsequent
        snapshots reflect those increments.

        Keys (pool PR 5d):
          - ``pool_slot_acquired_total``: sessions served by a pool
            slot.  High value relative to ``pool_acquire_miss_total``
            means pool is well-sized for the workload.
          - ``pool_acquire_miss_total``: ``acquire_slot`` returned
            None.  Sessions fell back to cold-spawn.  If this is
            growing fast, raise ``target_size`` or check
            ``pool_replenish_failures_total``.
          - ``pool_stale_reservation_evicted_total``: a cascade-affined
            idle slot was dropped at the ceiling to admit a returning
            slot of a live cascade.  Its cascade pays a cold plugin
            bootstrap next stage; it does not stall.
          - ``pool_replenish_ceiling_blocked_total``: replenishment
            wanted an unreserved slot and ``max_size`` forbade it.
            Nonzero on a multi-tenant daemon means raise
            ``JAATO_RUNNER_POOL_MAX_SIZE``.
          - ``pool_replenish_success_total``: replenishment thread
            successfully forked a new slot.
          - ``pool_replenish_failures_total``: replenishment thread's
            ``request_fork_slot`` returned None.  Investigate
            template health if growing.
          - ``template_respawn_attempts_total``: watchdog initiated a
            template respawn (template died).
          - ``template_respawn_failures_total``: respawn ``spawn()``
            raised.  Attempts >> failures means flaky template;
            attempts == failures means template can't be respawned
            (operator action needed).
        """
        with self._counters_lock:
            return dict(self._counters)

    def shutdown_all(self) -> None:
        """Tear down the replenishment thread + every idle slot.

        Called once at daemon shutdown.  Idempotent: a second call
        after a successful first call finds an empty pool + stopped
        thread and returns.

        Order:
          1. Stop the replenishment thread (so it doesn't race-fork
             a new slot while we're closing the pool).
          2. Close each idle slot's socket — the slot's serve loop
             reads EOF and exits cleanly (pool PR 4 replaced PR 3's
             ``SHUTDOWN\\n`` command-loop with RPC-serve; an RPC
             slot exits on peer-EOF, not on a custom shutdown line).
          3. Reap each slot's PID via ``waitpid``.  Slots are template-
             children not daemon-children, so this can raise
             ``ChildProcessError`` (the init process already reaped
             them on SIGCHLD).  Treating that as success is fine for
             PR 4 — the subreaper fix is PR 5 work.

        Best-effort: any per-slot error logs but doesn't stop the
        rest of the teardown.
        """
        # 1. Stop the replenishment thread first.
        self.stop_replenishment()

        # 2. Drain idle slots.
        with self._lock:
            slots = list(self._idle_slots)
            self._idle_slots.clear()

        for slot in slots:
            try:
                slot.sock.close()
            except OSError:
                pass
            # 3. Reap.  Slots are template-children; ChildProcessError
            # is expected and benign.
            try:
                os.waitpid(slot.pid, 0)
            except ChildProcessError:
                pass

        if slots:
            logger.info(
                "PoolManager.shutdown_all: tore down %d idle slot(s)",
                len(slots),
            )

    # --------------------- replenishment thread ----------------------

    def start_replenishment(self) -> None:
        """Start the background thread that keeps the pool topped up.

        Pool PR 4: the thread watches
        :meth:`unreserved_idle_count` against ``target_size`` and,
        whenever the count of slots ANY tenant could take drops below
        target, asks the template for a fresh fork-slot — subject to
        the ``max_size`` ceiling on total idle slots.  This is what makes
        the pool useful for cascades — a 6-step cascade with
        target_size=2 cold-spawns 4 of 6 steps otherwise; with
        replenishment a slot's gone-and-refilled cycle is small
        compared to the model-call time of each step, so cascade
        steps 3+ find a warm slot waiting.

        Idempotent: a second :meth:`start_replenishment` call after
        a successful first one is a no-op (logs a warning).

        Skipped when ``target_size <= 0`` (pool is disabled).
        """
        if self.target_size <= 0:
            logger.debug(
                "PoolManager.start_replenishment: target_size=%d; "
                "thread not started", self.target_size,
            )
            return
        if self._replenish_thread is not None and self._replenish_thread.is_alive():
            logger.warning(
                "PoolManager.start_replenishment: already running; ignoring",
            )
            return

        self._replenish_stop.clear()
        self._replenish_thread = threading.Thread(
            target=self._replenish_loop,
            name="jaato-pool-replenish",
            daemon=True,
        )
        self._replenish_thread.start()
        logger.info(
            "PoolManager: replenishment thread started "
            "(target_size=%d interval=%.2fs)",
            self.target_size, self._replenish_interval,
        )

    def stop_replenishment(self, timeout: float = 5.0) -> None:
        """Signal the replenishment thread to stop and wait for it.

        Idempotent: safe to call when no thread is running.
        """
        if self._replenish_thread is None:
            return
        self._replenish_stop.set()
        self._replenish_thread.join(timeout=timeout)
        if self._replenish_thread.is_alive():
            logger.warning(
                "PoolManager.stop_replenishment: thread didn't join "
                "within %.1fs; leaking as daemon thread", timeout,
            )
        self._replenish_thread = None

    def _replenish_loop(self) -> None:
        """Background loop body — wakes every ``_replenish_interval``,
        watchdogs the template + tops up the pool by ONE slot per
        iteration.

        One-slot-per-iteration matters: when the cascade rapidly
        drains the pool (multiple steps in a tight loop) the loop
        keeps refilling without blocking the daemon's main asyncio
        loop or holding the pool lock across a multi-slot batch.
        On a hot cascade the loop body fires effectively-continuously
        until the pool is back at target_size.

        Pool PR 5b watchdog (this iteration): detect template death
        + auto-respawn.  When ``template_manager.is_alive()`` flips
        False (template OOM-killed, segfaulted, or otherwise gone)
        the loop:

          1. Drains any idle slot handles — slots may have re-parented
             to the daemon via subreaper (PR 5b daemon startup bit);
             ``waitpid`` them so they don't zombie.
          2. Respawns the template via ``template_manager.spawn()``.
          3. Sleeps briefly to let the new template's plugin discovery
             complete before the next iteration tries fork-slot.
             (PR 5c will replace this sleep with a proper ready
             handshake.)

        Stops cleanly when ``_replenish_stop`` is set (daemon
        shutdown).
        """
        while not self._replenish_stop.is_set():
            try:
                # Template watchdog (PR 5b): detect death + respawn.
                # PR 5c: ``template_manager.spawn`` (called inside
                # _handle_template_death) blocks for the new
                # template's READY signal, so no explicit sleep
                # needed before the next iteration tries fork-slot.
                if not self._template_manager.is_alive():
                    self._handle_template_death()
                    continue

                # Phase 2: cascade-idle teardown sweep.  Reaps slots
                # that have sat IDLE_FOR_CASCADE longer than the
                # configured timeout — these are cascades that
                # finished (no further sessions arriving) or stalled
                # past the operator's tolerance.
                self._sweep_cascade_idle()

                # Cheap check: enough UNRESERVED slots, sleep.
                #
                # Unreserved, not total (#898).  ``idle_count()``
                # counts slots a waiting tenant is FORBIDDEN to use:
                # with two idle slots affined to cascade B and
                # target_size=2 the pool read "full" and never forked
                # another, while ``acquire_slot(cascade=A)`` returned
                # None.  Nothing to run on, and nothing in the system
                # that would ever create one, until B released.
                if self.unreserved_idle_count() >= self.target_size:
                    self._replenish_stop.wait(self._replenish_interval)
                    continue
                # ... but reservations still occupy memory, so the
                # total is what the ceiling bounds.  Hitting it is the
                # signal that ``max_size`` is too small for the number
                # of concurrent tenants: some of them are being served
                # by cold-spawn while reservations hold the ceiling.
                if self.idle_count() >= self.max_size:
                    self._incr("pool_replenish_ceiling_blocked_total")
                    self._replenish_stop.wait(self._replenish_interval)
                    continue
                raw = self._template_manager.request_fork_slot()
                if raw is None:
                    # request_fork_slot already logged the cause; back
                    # off briefly so we don't tight-loop on a flaky
                    # template.
                    self._incr("pool_replenish_failures_total")
                    self._replenish_stop.wait(self._replenish_interval)
                    continue
                pid, sock = raw
                new_slot = PoolSlot(pid=pid, sock=sock)
                with self._lock:
                    self._idle_slots.append(new_slot)
                self._incr("pool_replenish_success_total")
                logger.info(
                    "PoolManager replenish: forked slot pid=%d "
                    "(unreserved=%d/%d, idle_count=%d/%d)",
                    new_slot.pid, self.unreserved_idle_count(),
                    self.target_size, len(self._idle_slots), self.max_size,
                )
            except Exception:  # noqa: BLE001 — boundary surface
                logger.exception(
                    "PoolManager replenish: unhandled error; sleeping "
                    "and continuing",
                )
                self._replenish_stop.wait(self._replenish_interval)

    def _teardown_slot(self, slot: PoolSlot, *, reason: str) -> None:
        """Close one idle slot's transport and reap its process.

        ONE COPY, called by every path that drops a slot.  It was the
        cascade-idle sweep's inline tail; the capacity check needs the
        identical sequence, and a second copy of a forty-line teardown
        is a second copy that rots.

        MUST RUN OFF THE DAEMON LOOP.  It closes the rpc via
        ``run_coroutine_threadsafe`` and BLOCKS on the future, so calling
        it from the loop thread would have the loop wait for itself.
        Both callers are the replenish thread, which is why the capacity
        check hands slots here instead of tearing them down on the
        session-return path -- that path runs wherever
        ``JaatoServer.shutdown`` runs, and a cap is not worth inheriting
        a thread-context assumption to enforce.
        """
        # Phase 3 hotfix (server 0.6.150+): close the slot's rpc
        # client before tearing down the socket.  rpc.close()
        # closes the writer + sigterm-ladder the runner pid;
        # doing it via the rpc rather than bare ``sock.close()``
        # cancels the read task + drains in-flight futures
        # cleanly.  Falls back to bare sock.close() if no rpc
        # was ever stashed (rare — slot has run zero sessions).
        rpc = slot.rpc
        if rpc is not None:
            # Best-effort close via the daemon loop; we run
            # close synchronously here because we're already
            # in the replenish thread, not the daemon loop.
            try:
                import asyncio as _asyncio
                loop = getattr(rpc, "_loop", None)
                if loop is not None and loop.is_running():
                    fut = _asyncio.run_coroutine_threadsafe(
                        rpc.close(timeout=5.0), loop,
                    )
                    fut.result(timeout=8.0)
                else:
                    # No loop running — fall back to bare
                    # socket close.  Runner gets EOF + dies.
                    slot.sock.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PoolManager cascade-idle sweep: rpc.close "
                    "raised for pid=%d: %s — falling back to "
                    "bare sock.close()", slot.pid, exc,
                )
                try:
                    slot.sock.close()
                except OSError:
                    pass
        else:
            try:
                slot.sock.close()
            except OSError:
                pass
        try:
            os.waitpid(slot.pid, 0)
        except ChildProcessError:
            pass

    def _drain_pending_teardown(self) -> None:
        """Tear down slots the capacity check dropped.

        Runs in the replenish thread, which is the context
        ``_teardown_slot`` requires.  Called from the sweep so both
        reclaim paths share one caller and one thread.
        """
        with self._lock:
            pending, self._pending_teardown = self._pending_teardown, []
        for slot in pending:
            self._teardown_slot(slot, reason="over-capacity")
            self._incr("pool_slots_over_cap_torndown_total")
            logger.info(
                "PoolManager: torn down over-capacity slot pid=%d "
                "(cascade=%s)", slot.pid, slot.cascade_id or "(pure)",
            )

    def _sweep_cascade_idle(self) -> None:
        """Tear down IDLE_FOR_CASCADE slots whose idle window expired.

        Phase 2.  An IDLE_FOR_CASCADE(C) slot is one that returned
        from a session (``last_session_end_ts`` set) AND has a
        ``cascade_id`` set.  If the elapsed monotonic-time since
        the last session end exceeds ``cascade_idle_timeout_seconds``,
        the cascade is considered finished (or paused longer than
        the operator's tolerance).  We close the daemon-side socket
        (slot sees EOF, exits), waitpid it, and let the
        replenishment loop top the pool back up.

        Acquired (in-use) slots are NEVER in ``_idle_slots`` so this
        sweep cannot race with an active session.  PURE IDLE slots
        (``cascade_id is None``) are also exempt — they have no
        cascade affinity to time out.

        Called every replenish iteration (~0.5s).  Cheap when the
        pool has no IDLE_FOR_CASCADE entries.
        """
        self._drain_pending_teardown()

        now = time.monotonic()
        timeout = self._cascade_idle_timeout
        to_reap: List[PoolSlot] = []
        with self._lock:
            survivors: List[PoolSlot] = []
            for slot in self._idle_slots:
                if (
                    slot.cascade_id is not None
                    and slot.last_session_end_ts is not None
                    and (now - slot.last_session_end_ts) > timeout
                ):
                    to_reap.append(slot)
                else:
                    survivors.append(slot)
            if to_reap:
                self._idle_slots = survivors

        for slot in to_reap:
            self._teardown_slot(slot, reason="cascade-idle")
            self._incr("cascade_slots_idle_torndown_total")
            logger.info(
                "PoolManager cascade-idle sweep: torn down slot pid=%d "
                "cascade=%s (idle %.1fs > timeout %.0fs)",
                slot.pid, slot.cascade_id,
                now - (slot.last_session_end_ts or now), timeout,
            )

    def _handle_template_death(self) -> None:
        """Recover from template-subprocess death (pool PR 5b watchdog).

        Pool slots are template-children (forked, no exec).  When the
        template dies, slots get orphaned.  With the daemon's
        ``PR_SET_CHILD_SUBREAPER`` bit set (PR 5b startup), orphaned
        descendants re-parent to the daemon — so ``waitpid`` works.

        Recovery sequence:
          1. Drain idle slot handles — close their sockets (slot sees
             EOF, exits), waitpid them.  Acquired slots (handed out to
             active sessions) keep running independently; the
             session's bootstrap already completed and the slot serves
             RPC until session end.  Only IDLE slot handles in our
             possession need cleanup here.
          2. Respawn the template.  ``spawn`` blocks for the new
             template's READY signal (PR 5c) before returning, so
             the next replenish iteration can immediately fork-slot
             without a sleep.  On spawn failure, the outer loop's
             exception handler retries on the next iteration.

        Best-effort throughout — daemon shutdown signals via
        ``_replenish_stop`` are honored.
        """
        with self._lock:
            stale = list(self._idle_slots)
            self._idle_slots.clear()

        for slot in stale:
            try:
                slot.sock.close()
            except OSError:
                pass
            # Subreaper re-parented slot to daemon → waitpid works.
            # If subreaper wasn't set OR if init already reaped, we
            # get ChildProcessError — tolerate it.
            try:
                os.waitpid(slot.pid, 0)
            except ChildProcessError:
                pass

        if stale:
            logger.warning(
                "PoolManager watchdog: template died; reaped %d "
                "orphaned idle slot(s).  Acquired (in-use) slots "
                "keep running independently.", len(stale),
            )
        else:
            logger.warning(
                "PoolManager watchdog: template died; idle pool was "
                "empty so no orphaned slots to reap.",
            )

        # Respawn the template.  ``spawn`` is idempotent if the
        # previous spawn cleaned up properly (is_alive() returning
        # False clears self.pid via waitpid).
        self._incr("template_respawn_attempts_total")
        try:
            self._template_manager.spawn()
        except Exception as exc:  # noqa: BLE001 — boundary surface
            self._incr("template_respawn_failures_total")
            logger.error(
                "PoolManager watchdog: template respawn failed: %s; "
                "will retry on next iteration", exc,
            )
            return

        logger.info(
            "PoolManager watchdog: template respawned successfully — "
            "pool will refill on subsequent iterations",
        )
