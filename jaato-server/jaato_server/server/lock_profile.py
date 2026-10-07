"""A re-entrant lock that says who holds it, since when, and from where.

**Why this exists (#1452).**  The daemon's event loop stopped for 121 s,
parked in ``SessionManager._emit_to_session`` on ``with self._lock:``.
The loop watchdog (#625, #631) printed the loop's stack, which named the
WAITER, and every other thread's, and still could not name the HOLDER:
the thread holding the lock was waiting on a future, and a future wait
sits in ``threading.py`` on ``waiter.acquire()``, which the dump read as
"blocked acquiring a lock".  A lock that records its own owner removes
that inference: the owner is a fact the lock knows, not something a
reader reconstructs from 70 stacks.

What it records, per OUTERMOST acquisition (a re-entrant acquire by the
owner changes nothing but the depth):

* the owning thread's ident and name;
* the monotonic time the hold began;
* the acquiring site (file, line, function).

What it does with that:

* on release, a hold longer than the threshold is logged at WARNING with
  its duration, its site and the stack at release time.  The threshold is
  ``JAATO_LOCK_HOLD_WARN_MS`` (default 500 ms; ``0`` disables the log).
  This fires AFTER the hold, so it answers "which holds are long";
* :func:`held_locks` returns every profiled lock currently held, with the
  owner's CURRENT stack (``sys._current_frames``).  The loop watchdog
  calls it in every ``LOOP_STALL`` dump, so a stall on a profiled lock
  names its holder WHILE the stall is happening.

**Cost when healthy.**  One ``get_ident``, one ``monotonic`` and one
``sys._getframe`` per outermost acquire, and one ``monotonic`` per
outermost release.  No stack is formatted unless a hold is long or a
stall dump asks.

**Semantics.**  Exactly ``threading.RLock``'s: ``acquire(blocking,
timeout)``, ``release``, the context-manager protocol, and the three
private methods ``threading.Condition`` uses on an RLock
(``_is_owned``, ``_release_save``, ``_acquire_restore``), so a
``Condition`` built on this lock keeps working and its waits are shown
as releases (a thread waiting on the condition does not hold the lock).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
import weakref
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

#: The env var, read once per lock at construction.
HOLD_WARN_ENV = "JAATO_LOCK_HOLD_WARN_MS"
#: Default long-hold threshold, milliseconds.
DEFAULT_HOLD_WARN_MS = 500.0
#: Greppable token on the long-hold WARNING.
LONG_HOLD_TOKEN = "LOCK_HELD_LONG"

_REGISTRY: "weakref.WeakSet[ProfiledRLock]" = weakref.WeakSet()
_REGISTRY_GUARD = threading.Lock()


def hold_warn_seconds() -> float:
    """The long-hold threshold in seconds; ``0.0`` means "do not log".

    A value that does not parse, or is negative, falls back to the
    default rather than disabling the log: a typo must not remove the
    only record of a long hold.
    """
    # A literal, not HOLD_WARN_ENV: the env-scope catalog is derived by an
    # AST scan for literal reads (shared/env_scope.py).
    raw = os.environ.get("JAATO_LOCK_HOLD_WARN_MS", "").strip()  # env: ms SessionManager._lock may be held before LOCK_HELD_LONG is logged with the holder's stack (default 500; 0 off)
    if not raw:
        return DEFAULT_HOLD_WARN_MS / 1000.0
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_HOLD_WARN_MS / 1000.0
    if value < 0:
        return DEFAULT_HOLD_WARN_MS / 1000.0
    return value / 1000.0


@dataclass(frozen=True)
class HeldLock:
    """A snapshot of one profiled lock that was held when asked.

    ``stack`` is the owner's stack at the moment of the snapshot, not at
    acquisition: for a lock held across a wait, that is the wait, which is
    the thing a reader needs to see.
    """

    name: str
    owner_ident: int
    owner_name: str
    held_for: float
    site: str
    stack: str

    def render(self) -> str:
        """Human-readable form used by the loop watchdog's dump."""
        return (
            f"lock {self.name!r} is HELD by thread {self.owner_ident} "
            f"{self.owner_name!r} for {self.held_for:.1f}s, acquired at "
            f"{self.site}; the holder is executing:\n{self.stack}"
        )


def _site_of(frame) -> str:
    if frame is None:
        return "<unknown>"
    code = frame.f_code
    return f"{code.co_filename}:{frame.f_lineno} in {code.co_name}"


class ProfiledRLock:
    """``threading.RLock`` that records its owner (see module docstring).

    Lifecycle of the owner fields: set by the outermost successful
    :meth:`acquire` (depth 0 -> 1), cleared by the release that returns
    depth to 0, in both cases while the inner lock is held, so the owner
    thread is the only writer.  Readers (:func:`held_locks`, from the
    watchdog thread) read without the lock and accept a snapshot that may
    be a moment stale; they never block the lock's users.
    """

    def __init__(self, name: str, *, warn_after: Optional[float] = None) -> None:
        """
        Args:
            name: Shown in every log line and dump (e.g.
                ``"SessionManager._lock"``).
            warn_after: Long-hold threshold in seconds; ``None`` reads
                :data:`HOLD_WARN_ENV`; ``0`` disables the release log.
        """
        self.name = name
        self._inner = threading.RLock()
        self._warn_after = (
            hold_warn_seconds() if warn_after is None else warn_after
        )
        self._owner: Optional[int] = None
        self._depth = 0
        self._since = 0.0
        self._site_frame = None
        self._site = ""
        with _REGISTRY_GUARD:
            _REGISTRY.add(self)

    # ------------------------------------------------------------ RLock API

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire, recording the owner on the outermost acquisition."""
        got = self._inner.acquire(blocking, timeout)
        if got:
            self._note_acquired(sys._getframe(1), 1)
        return got

    __enter__ = acquire

    def release(self) -> None:
        """Release; log the hold if it was long and this was the outermost."""
        if self._owner != threading.get_ident():
            # Let the inner lock raise the error RLock raises.
            self._inner.release()
            return
        self._depth -= 1
        if self._depth:
            self._inner.release()
            return
        held = time.monotonic() - self._since
        frame, site = self._site_frame, self._site
        self._clear_owner()
        self._inner.release()
        if self._warn_after and held >= self._warn_after:
            self._log_long_hold(held, frame, site)

    def __exit__(self, *exc) -> None:
        self.release()

    def locked(self) -> bool:
        """Whether any thread holds the lock."""
        return self._owner is not None

    # ------------------------------------- the Condition protocol on RLock

    def _is_owned(self) -> bool:
        return self._owner == threading.get_ident()

    def _release_save(self):
        state = (self._depth, self._since, self._site_frame, self._site)
        self._clear_owner()
        return (self._inner._release_save(), state)

    def _acquire_restore(self, saved) -> None:
        inner_state, (depth, _since, frame, site) = saved
        self._inner._acquire_restore(inner_state)
        # A condition wait released the lock; the new hold starts now.
        self._owner = threading.get_ident()
        self._depth = depth
        self._since = time.monotonic()
        self._site_frame, self._site = frame, site

    # ------------------------------------------------------------- helpers

    def _note_acquired(self, frame, depth: int) -> None:
        if self._depth == 0:
            self._owner = threading.get_ident()
            self._since = time.monotonic()
            self._site_frame = frame
            self._site = _site_of(frame)
        self._depth += depth

    def _clear_owner(self) -> None:
        self._owner = None
        self._depth = 0
        self._site_frame = None
        self._site = ""

    def _log_long_hold(self, held: float, frame, site: str) -> None:
        stack = "".join(traceback.format_stack(frame)) if frame else ""
        logger.warning(
            "%s: %r was held for %.0f ms by thread %r, acquired at %s "
            "(threshold %.0f ms, %s):\n%s",
            LONG_HOLD_TOKEN, self.name, held * 1000.0,
            threading.current_thread().name, site,
            self._warn_after * 1000.0, HOLD_WARN_ENV, stack,
        )

    def snapshot(self) -> Optional[HeldLock]:
        """This lock's owner and the owner's current stack, or ``None``."""
        owner, since, site = self._owner, self._since, self._site
        if owner is None:
            return None
        frame = sys._current_frames().get(owner)
        names = {t.ident: t.name for t in threading.enumerate()}
        return HeldLock(
            name=self.name,
            owner_ident=owner,
            owner_name=names.get(owner, "<unnamed>"),
            held_for=time.monotonic() - since,
            site=site or "<unknown>",
            stack="".join(traceback.format_stack(frame)) if frame else "",
        )


def held_locks() -> List[HeldLock]:
    """Every profiled lock currently held, longest hold first."""
    with _REGISTRY_GUARD:
        locks: Tuple[ProfiledRLock, ...] = tuple(_REGISTRY)
    found = [snap for snap in (lk.snapshot() for lk in locks) if snap]
    found.sort(key=lambda h: h.held_for, reverse=True)
    return found
