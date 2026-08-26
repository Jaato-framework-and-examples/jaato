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

import logging
import sys
import threading
import time
import traceback
from typing import Optional

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
    ) -> None:
        """
        Args:
            interval: Heartbeat period, seconds.
            threshold: Staleness that counts as a stall.  Above ``2 *
                interval`` so ordinary scheduling jitter never fires it.
            resample_every: While one stall persists, re-dump the stack at
                most this often — a 30s stall yields ~3 stacks, enough to see
                whether it is one frame or a churn, without flooding the log.
        """
        self._interval = interval
        self._threshold = threshold
        self._resample = resample_every
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
            "LoopWatchdog armed: interval=%.1fs threshold=%.1fs -- a stall "
            "longer than the threshold logs the loop thread's stack at "
            "WARNING",
            self._interval, self._threshold,
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
