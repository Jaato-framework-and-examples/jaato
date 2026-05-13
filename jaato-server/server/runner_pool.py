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
from typing import List, Optional, Tuple


logger = logging.getLogger(__name__)


# Per-slot handle: (pid, daemon-side socket).
SlotHandle = Tuple[int, socket.socket]


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
        """
        self._template_manager = template_manager
        self.target_size = max(0, int(target_size))
        self._idle_slots: List[SlotHandle] = []
        self._lock = threading.Lock()
        # Pool PR 4 replenishment thread state.
        self._replenish_interval = float(replenish_interval)
        self._replenish_stop = threading.Event()
        self._replenish_thread: Optional[threading.Thread] = None

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
                handle = self._template_manager.request_fork_slot()
                if handle is None:
                    logger.warning(
                        "PoolManager: fork-slot %d/%d failed; pool "
                        "will be smaller than target.  Sessions fall "
                        "back to cold-spawn session-mode when pool is "
                        "empty.", i + 1, self.target_size,
                    )
                    break
                self._idle_slots.append(handle)
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

    def acquire_slot(self) -> Optional[SlotHandle]:
        """Pop an idle slot off the pool (PR 4 will use this).

        PR 3 ships the API but the daemon doesn't call it yet.
        Caller becomes responsible for the slot — must either send
        a bootstrap envelope (PR 4) or close the daemon-side socket
        (which signals the slot to exit).

        Returns:
            ``(pid, sock)`` of an idle slot, or ``None`` if the pool
            is empty (session-mode fallback applies).
        """
        with self._lock:
            if not self._idle_slots:
                return None
            return self._idle_slots.pop()

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

        for slot_pid, slot_sock in slots:
            try:
                slot_sock.close()
            except OSError:
                pass
            # 3. Reap.  Slots are template-children; ChildProcessError
            # is expected and benign.
            try:
                os.waitpid(slot_pid, 0)
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
        """Background loop body — wakes every ``_replenish_interval``
        and tops up the pool by ONE slot per iteration.

        One-slot-per-iteration matters: when the cascade rapidly
        drains the pool (multiple steps in a tight loop) the loop
        keeps refilling without blocking the daemon's main asyncio
        loop or holding the pool lock across a multi-slot batch.
        On a hot cascade the loop body fires effectively-continuously
        until the pool is back at target_size.

        Stops cleanly when ``_replenish_stop`` is set (daemon
        shutdown) OR when the template manager reports dead.
        """
        while not self._replenish_stop.is_set():
            try:
                # Cheap check: pool already at target, sleep.
                if self.idle_count() >= self.target_size:
                    self._replenish_stop.wait(self._replenish_interval)
                    continue
                # Template dead = nothing we can do until daemon restarts.
                if not self._template_manager.is_alive():
                    logger.warning(
                        "PoolManager replenish: template is dead; "
                        "sleeping %.1fs before re-checking",
                        self._replenish_interval * 10,
                    )
                    self._replenish_stop.wait(self._replenish_interval * 10)
                    continue
                handle = self._template_manager.request_fork_slot()
                if handle is None:
                    # request_fork_slot already logged the cause; back
                    # off briefly so we don't tight-loop on a flaky
                    # template.
                    self._replenish_stop.wait(self._replenish_interval)
                    continue
                with self._lock:
                    self._idle_slots.append(handle)
                logger.info(
                    "PoolManager replenish: forked slot pid=%d "
                    "(idle_count=%d/%d)",
                    handle[0], len(self._idle_slots), self.target_size,
                )
            except Exception:  # noqa: BLE001 — boundary surface
                logger.exception(
                    "PoolManager replenish: unhandled error; sleeping "
                    "and continuing",
                )
                self._replenish_stop.wait(self._replenish_interval)
