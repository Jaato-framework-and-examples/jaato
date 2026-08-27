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
        target_size: Configured pool size (default 2).
        _idle_slots: List of currently-idle slot handles.
    """

    def __init__(
        self,
        template_manager,
        target_size: int = 2,
        replenish_interval: float = 0.5,
        cascade_idle_timeout_seconds: float = DEFAULT_CASCADE_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the pool manager.

        Args:
            template_manager: The daemon's :class:`TemplateManager`.
                Source of fork-slot requests; must already have
                ``spawn()`` been called.
            target_size: Number of idle slots to keep available.
                Default 2 — reasonable for typical workstation; cascade
                harnesses spawning many concurrent sessions can raise
                this via ``JAATO_RUNNER_POOL_SIZE`` env var (consumed
                by daemon ``__main__.py`` and threaded here).  Values
                <= 0 disable the pool (sessions fall back to cold-spawn
                session-mode).
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
        """Return the current count of idle slots."""
        with self._lock:
            return len(self._idle_slots)

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
        with self._lock:
            self._idle_slots.append(slot)
        logger.info(
            "PoolManager.return_slot_after_session: slot pid=%d "
            "returned to pool (cascade=%s; idle_count=%d/%d)",
            slot.pid, slot.cascade_id or "(pure)",
            len(self._idle_slots), self.target_size,
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

        Pool PR 4: the thread watches ``idle_count()`` against
        ``target_size`` and, whenever the count drops below target,
        asks the template for a fresh fork-slot.  This is what makes
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

                # Cheap check: pool already at target, sleep.
                if self.idle_count() >= self.target_size:
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
                    "(idle_count=%d/%d)",
                    new_slot.pid, len(self._idle_slots), self.target_size,
                )
            except Exception:  # noqa: BLE001 — boundary surface
                logger.exception(
                    "PoolManager replenish: unhandled error; sleeping "
                    "and continuing",
                )
                self._replenish_stop.wait(self._replenish_interval)

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
