"""A stalled event loop that names what it was doing.

**Why this exists.**  The daemon's asyncio loop demonstrably fails to run
scheduled coroutines for 5-35 seconds at a stretch — established via #625's
timeout attribution: every caller arms an inner timeout strictly smaller than
its outer, the inner is armed only once the coroutine runs, and across 18
stalls in 28 minutes on two independent harnesses the inner fired zero times.
The loop was not running them.

What we could NOT establish is WHY.  A stalled loop leaves no witness: it
cannot log (logging from the loop needs the loop's thread to be free), its
timers do not fire, and by the time anything observes the stall it is over.
Both benches saw only silence bracketed by a burst.  Inline notification
dispatch on the RPC read loop is a verified structural fact and the leading
candidate — and per the no-guessing rule it stays a candidate until something
NAMES the code holding the loop.

**How it works.**  Two halves:

- an asyncio **heartbeat** task on the watched loop that stamps a monotonic
  timestamp every ``interval`` seconds.  While the loop runs, the stamp is
  fresh; the moment the loop stops scheduling, the stamp freezes.  The
  heartbeat is the *absence detector*: its silence is the signal.

- a plain **thread** that polls the stamp.  Threads keep running while the
  loop is blocked — that asymmetry is the whole trick.  When the stamp goes
  stale past ``threshold``, the thread grabs the loop thread's CURRENT STACK
  via ``sys._current_frames()`` and logs it at WARNING.  That stack is the
  code holding the loop, captured mid-stall, with no cooperation from the
  loop required.

  Past ``all_threads_after`` it dumps EVERY thread's stack, named and
  stamped with how long each has sat on the same frames.  This half exists
  because the first half answers only half the question: when the loop is
  parked on a lock, its own stack names the thread that is BLOCKED, and the
  holder is somewhere else entirely.  Observed 2026-08-28 — a 36.5s stall
  whose loop stack ended at ``session_manager.py:4351  with self._lock:``,
  which says what the loop wanted and nothing about who had it.

  ``sys._current_frames()`` is per-PROCESS, so this can only be done from
  inside the daemon.  No client, harness or external observer can obtain it;
  from outside they get their own threads.

On recovery it logs the stall's total duration, so the log carries the pair
the investigation needs: *what was running* and *for how long*.

**Cost when healthy:** one no-op coroutine per second on the loop and one
timestamp comparison per second on a daemon thread.  The stack dump happens
only during a stall, rate-limited to one per ``resample_every`` while a single
stall persists.

**What this deliberately is not.**  Not a fix — nothing here prevents or
shortens a stall, and it must not: the stall is the evidence, and softening
it before it is understood would be masking the symptom that routes
attention.  Not ``loop.set_debug(True)`` — debug mode's slow-callback log
only reports AFTER the callback returns (nothing mid-stall), and its
per-callback overhead taxes the healthy path.
"""

from __future__ import annotations

import linecache
import logging
import os
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional

import asyncio

logger = logging.getLogger(__name__)

#: Greppable tokens.  LOOP_STALL fires mid-stall with the stack; LOOP_RESUMED
#: fires on recovery with the duration.  A reader greps either and gets the
#: other from the adjacent lines.
_STALL_TOKEN = "LOOP_STALL"
_RESUME_TOKEN = "LOOP_RESUMED"


class LoopWatchdog:
    """Watches one asyncio loop; names the code holding it when it stalls.

    Lifecycle: construct, then ``start()`` FROM the loop being watched (it
    needs the loop's thread id, and reading it from inside is the only
    race-free way).  ``stop()`` from anywhere.
    """

    def __init__(
        self,
        *,
        interval: float = 1.0,
        threshold: float = 2.0,
        resample_every: float = 10.0,
        all_threads_after: float = 10.0,
    ) -> None:
        """
        Args:
            interval: Heartbeat period, seconds.
            threshold: Staleness that counts as a stall.  Above ``2 *
                interval`` so ordinary scheduling jitter never fires it.
            resample_every: While one stall persists, re-dump the stack at
                most this often — a 30s stall yields ~3 stacks, enough to see
                whether it is one frame or a churn, without flooding the log.
            all_threads_after: Stall age past which EVERY thread's stack is
                dumped, not just the loop's.  Below it the log is unchanged.

                Why a second threshold rather than always-on: the loop thread
                alone answers "what was the loop doing", and for a short stall
                that is the whole question.  It does not answer "who was
                holding the thing the loop waited on" — and a daemon has
                enough threads that dumping all of them on every 2s hiccup
                would cost more than it tells.  The default sits where the
                two questions separate in practice: #633's stalls clustered
                at 10-12s on a timeout ceiling and are understood; the
                unexplained ones observed since run past 30s on a held mutex,
                where the loop's own stack names only the BLOCKED party.
        """
        self._interval = interval
        self._threshold = threshold
        self._resample = resample_every
        self._all_threads_after = all_threads_after
        #: ident -> (stack digest, monotonic time that digest FIRST appeared).
        #: Lets each dump say how long a thread has been sitting on the same
        #: stack: one that appears once at 36s and one that appears in every
        #: dump are different bugs, and without this they read identically.
        self._digest_seen: dict = {}
        self._beat = time.monotonic()
        self._loop_thread_id: Optional[int] = None
        self._task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---------------------------------------------------------------- loop side

    def start(self) -> None:
        """Install the heartbeat on the RUNNING loop and start the monitor."""
        loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()
        self._beat = time.monotonic()
        self._task = loop.create_task(self._heartbeat(), name="loop-watchdog")
        self._thread = threading.Thread(
            target=self._monitor, name="loop-watchdog-monitor", daemon=True,
        )
        self._thread.start()
        # The armed line deliberately does NOT contain the stall token: a
        # reader grepping for stalls must find only stalls, and a filter
        # matching the token must not fire on the announcement of it.
        logger.info(
            "LoopWatchdog armed: interval=%.1fs threshold=%.1fs "
            "all_threads_after=%.1fs -- a stall past the threshold logs the "
            "loop thread's stack at WARNING, and past all_threads_after "
            "every thread's stack as well",
            self._interval, self._threshold, self._all_threads_after,
        )

    async def _heartbeat(self) -> None:
        try:
            while not self._stop.is_set():
                self._beat = time.monotonic()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------- thread side

    def _monitor(self) -> None:
        stall_started: Optional[float] = None
        last_dump = 0.0
        while not self._stop.wait(self._interval):
            age = time.monotonic() - self._beat
            if age <= self._threshold:
                if stall_started is not None:
                    # Recovery.  The duration is measured from the last GOOD
                    # beat, which is when the loop actually stopped.
                    logger.warning(
                        "%s: loop ran again after %.1fs", _RESUME_TOKEN,
                        time.monotonic() - stall_started,
                    )
                    stall_started = None
                continue

            now = time.monotonic()
            if stall_started is None:
                stall_started = self._beat
            if now - last_dump < self._resample:
                continue
            last_dump = now

            stack = self._loop_stack()
            logger.warning(
                "%s: event loop has not run for %.1fs -- the loop thread is "
                "executing:\n%s", _STALL_TOKEN, age,
                stack or "<loop thread not found>",
            )

            # Past ``all_threads_after`` the loop's own stack is no longer the
            # whole answer.  When it is parked on a lock, that stack names the
            # BLOCKED party; the holder is another thread entirely, and no
            # amount of detail about the waiter identifies it.  ``_current_
            # frames()`` is per-PROCESS, so this can only be done from inside
            # the daemon -- an out-of-process observer sees its own threads.
            if age >= self._all_threads_after:
                logger.warning(
                    "%s: stall is %.1fs (>= %.1fs) -- every thread's stack "
                    "follows; the loop thread above is the WAITER, look here "
                    "for the holder:\n%s",
                    _STALL_TOKEN, age, self._all_threads_after,
                    self._all_thread_stacks(),
                )

    def _innermost_line(self, frame) -> str:
        """The source line the thread is actually sitting on.

        ``sys._current_frames()`` hands back each thread's INNERMOST frame, so
        this is where it stopped -- not where it started.
        """
        return linecache.getline(
            frame.f_code.co_filename, frame.f_lineno,
        ).strip()

    @staticmethod
    def _is_waiting_for_a_lock(line: str) -> bool:
        """Is this source line a thread blocking to ACQUIRE a lock?

        Acquiring a ``threading.Lock`` blocks inside C, which creates no Python
        frame -- so the innermost Python frame of a blocked thread IS the
        ``with ...lock:`` statement itself.  That is what makes this decidable
        from the frame alone.

        A thread parked here is BLOCKED, never blocking: it holds nothing it
        acquired at this line, because it never got it.
        """
        low = line.lower()
        return (
            (low.startswith("with ") and "lock" in low)
            or ".acquire(" in low
        )

    def _all_thread_stacks(self) -> str:
        """Every thread, PARTITIONED into waiters and holder-candidates.

        WHY PARTITION.  The first version of this dump reported every thread
        with how long it had sat on the same stack, on the theory that a
        holder sits still while busy threads move.  Run against a real daemon
        that turned out to answer the wrong question: 47 threads, and about
        forty read "unchanged for 20.0s", because A PARKED THREAD IS UNCHANGED
        BY DEFINITION.  Idle readers, reapers, pool threads and a dozen
        blocked watchers all looked exactly like a holder.  The signal is
        necessary and nowhere near sufficient.

        What actually found the holder was reading stack CONTENT: waiters end
        at a lock acquire, and the holder passes THROUGH and blocks somewhere
        else.  That distinction is mechanical, so it belongs here rather than
        in the reader's head.

        Output order follows the question: candidates FIRST and in full, since
        the holder is among them and they are the smaller set; waiters after,
        GROUPED BY THE LINE THEY ARE BLOCKED ON, so a twelve-thread convoy is
        one entry naming twelve threads instead of twelve near-identical
        stacks.

        Named by the perpetual-monologue bench, from the first live run.
        """
        frames = sys._current_frames()
        named = {t.ident: t.name for t in threading.enumerate()}
        now = time.monotonic()

        # Retire threads that have gone away, so the map cannot grow across a
        # long-lived daemon's many stalls.
        for ident in list(self._digest_seen):
            if ident not in frames:
                del self._digest_seen[ident]

        candidates: List[str] = []
        waiters: Dict[str, List[str]] = {}

        for ident, frame in sorted(frames.items()):
            label = self._label(ident, named, frame, now)
            line = self._innermost_line(frame)
            if self._is_waiting_for_a_lock(line):
                where = (f"{os.path.basename(frame.f_code.co_filename)}:"
                         f"{frame.f_lineno}  {line}")
                waiters.setdefault(where, []).append(label)
            else:
                stack = "".join(traceback.format_stack(frame))
                candidates.append(f"--- {label}\n{stack}")

        return self._render(candidates, waiters)

    def _label(self, ident: int, named: dict, frame, now: float) -> str:
        """``thread <id> '<name>'`` plus how long it has sat on this stack."""
        stack = "".join(traceback.format_stack(frame))
        digest = hash(stack)
        prev = self._digest_seen.get(ident)
        if prev is not None and prev[0] == digest:
            held = f", unchanged for {now - prev[1]:.1f}s"
        else:
            self._digest_seen[ident] = (digest, now)
            held = ", first seen at this stack"

        if ident in named:
            name = named[ident]
        else:
            name = "<not in threading.enumerate() -- created outside threading>"
        marker = "  <-- THE LOOP" if ident == self._loop_thread_id else ""
        return f"thread {ident} {name!r}{held}{marker}"

    @staticmethod
    def _render(candidates: List[str], waiters: Dict[str, List[str]]) -> str:
        blocked = sum(len(v) for v in waiters.values())
        # The two headers must not be substrings of one another.  "NOT
        # WAITING FOR A LOCK" contains "WAITING FOR A LOCK", so anything
        # splitting the dump on the second lands inside the first -- which
        # is exactly what happened to the first test written against it.
        parts = [
            f"=== HOLDER CANDIDATES ({len(candidates)}) -- THE HOLDER IS IN "
            f"HERE. These threads got past every lock they asked for; one of "
            f"them is sitting on the one everything else wants.",
            *candidates,
            f"=== BLOCKED ACQUIRING A LOCK ({blocked}) -- blocked, not "
            f"blocking. Grouped by the line each is stopped on; a thread here "
            f"holds nothing it acquired at that line, because it never got "
            f"it.",
        ]
        for where, labels in sorted(waiters.items()):
            parts.append(f"  {where}\n    " + "\n    ".join(sorted(labels)))
        return "\n".join(parts)

    def _loop_stack(self) -> str:
        """The loop thread's stack, captured from outside, mid-stall."""
        if self._loop_thread_id is None:
            return ""
        frame = sys._current_frames().get(self._loop_thread_id)
        if frame is None:
            return ""
        return "".join(traceback.format_stack(frame))

    # ---------------------------------------------------------------- teardown

    def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 3)
