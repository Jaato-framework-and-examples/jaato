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
        """
        self._template_manager = template_manager
        self.target_size = max(0, int(target_size))
        self._idle_slots: List[SlotHandle] = []
        self._lock = threading.Lock()

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
        """Send SHUTDOWN to every idle slot + wait for them to exit.

        Called once at daemon shutdown.  Idempotent: a second call
        after a successful first call finds an empty pool and returns.

        Best-effort: a slot that's already dead (SIGCHLD reaped
        elsewhere) doesn't raise; slot's socket already closed is
        fine.
        """
        with self._lock:
            slots = list(self._idle_slots)
            self._idle_slots.clear()

        for slot_pid, slot_sock in slots:
            try:
                slot_sock.sendall(b"SHUTDOWN\n")
            except OSError as exc:
                logger.debug(
                    "PoolManager: slot pid=%d SHUTDOWN send failed "
                    "(slot already gone?): %s", slot_pid, exc,
                )
            try:
                slot_sock.close()
            except OSError:
                pass
            try:
                os.waitpid(slot_pid, 0)
            except ChildProcessError:
                pass  # slot already reaped

        if slots:
            logger.info(
                "PoolManager.shutdown_all: tore down %d idle slot(s)",
                len(slots),
            )
