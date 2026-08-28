"""The watchdog must name the code holding a stalled loop, from outside it.

These tests run a REAL event loop and REALLY block it — a fake would prove
the watchdog works against fakes, and the entire point of this instrument is
that nothing inside a stalled loop can testify.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import List

from server.loop_watchdog import LoopWatchdog


def _capture() -> tuple[logging.Handler, List[logging.LogRecord]]:
    records: List[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Cap()
    lg = logging.getLogger("server.loop_watchdog")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    return handler, records


def _release(handler) -> None:
    logging.getLogger("server.loop_watchdog").removeHandler(handler)


def _the_function_that_blocks_the_loop() -> None:
    """Named so the stack dump has something unmistakable to contain."""
    time.sleep(1.2)


def test_a_blocked_loop_is_reported_with_the_blocking_frame():
    handler, records = _capture()
    dog = LoopWatchdog(interval=0.1, threshold=0.3, resample_every=0.2)
    try:
        async def scenario():
            dog.start()
            await asyncio.sleep(0.3)          # a few healthy beats
            _the_function_that_blocks_the_loop()   # synchronous: loop stalls
            await asyncio.sleep(0.3)          # recovery beats

        asyncio.run(scenario())
    finally:
        dog.stop()
        _release(handler)

    stalls = [r for r in records if "LOOP_STALL" in r.getMessage()]
    assert stalls, "a 1.2s block on a 0.3s threshold produced no report"
    dump = stalls[0].getMessage()
    assert "_the_function_that_blocks_the_loop" in dump, (
        f"the report does not name the code holding the loop -- without the "
        f"frame this is a lag meter, not a diagnosis.  Got:\n{dump}"
    )

    resumes = [r for r in records if "LOOP_RESUMED" in r.getMessage()]
    assert resumes, "recovery was not reported, so duration is unmeasurable"


def test_a_healthy_loop_is_silent():
    """No stall, no noise — a witness that chatters gets filtered out."""
    handler, records = _capture()
    dog = LoopWatchdog(interval=0.05, threshold=0.5)
    try:
        async def scenario():
            dog.start()
            for _ in range(10):
                await asyncio.sleep(0.03)     # busy but never blocked

        asyncio.run(scenario())
    finally:
        dog.stop()
        _release(handler)

    noisy = [
        r for r in records
        if "LOOP_STALL" in r.getMessage() or "LOOP_RESUMED" in r.getMessage()
    ]
    assert not noisy, (
        f"a healthy loop produced stall traffic: "
        f"{[r.getMessage()[:80] for r in noisy]}"
    )


def test_one_long_stall_is_resampled_not_flooded():
    handler, records = _capture()
    dog = LoopWatchdog(interval=0.05, threshold=0.1, resample_every=0.35)
    try:
        async def scenario():
            dog.start()
            await asyncio.sleep(0.2)
            time.sleep(1.0)                    # ~1s stall, resample at 0.35s

        asyncio.run(scenario())
    finally:
        dog.stop()
        _release(handler)

    stalls = [r for r in records if "LOOP_STALL" in r.getMessage()]
    assert 1 <= len(stalls) <= 4, (
        f"expected a handful of resamples for a 1s stall, got {len(stalls)} "
        f"-- either the rate limit failed (flood) or sampling stopped"
    )


def test_stop_tears_down_both_halves():
    dog = LoopWatchdog(interval=0.05, threshold=0.5)

    async def scenario():
        dog.start()
        await asyncio.sleep(0.15)

    asyncio.run(scenario())
    dog.stop()
    assert not dog._thread.is_alive(), "monitor thread survived stop()"


# ---------------------------------------------------------------------------
# Naming the HOLDER, not just the waiter
#
# The tests above prove the watchdog names what the loop was doing.  That is
# the whole answer only while the loop is doing something.  When it is parked
# on a lock, its stack names the thread that is BLOCKED and says nothing about
# who has it -- observed 2026-08-28 as a 36.5s stall ending at
# ``session_manager.py:4351  with self._lock:``.
#
# These block a real loop on a lock a real other thread is holding, because a
# fake holder would prove the dump works against fakes.
# ---------------------------------------------------------------------------

def _the_function_that_HOLDS_the_lock(lock, release_after: float) -> None:
    """Named so the all-thread dump has something unmistakable to contain."""
    with lock:
        time.sleep(release_after)


def test_a_long_stall_names_the_thread_that_held_the_lock():
    handler, records = _capture()
    lock = threading.Lock()
    holder = threading.Thread(
        target=_the_function_that_HOLDS_the_lock,
        args=(lock, 1.5),
        name="the-holder-thread",
        daemon=True,
    )
    dog = LoopWatchdog(
        interval=0.1, threshold=0.3, resample_every=0.2,
        all_threads_after=0.5,
    )
    try:
        async def scenario():
            dog.start()
            await asyncio.sleep(0.2)
            holder.start()
            time.sleep(0.2)          # let the holder actually take it
            with lock:               # SYNCHRONOUS: the loop now waits on it
                pass
            await asyncio.sleep(0.3)

        asyncio.run(scenario())
    finally:
        dog.stop()
        holder.join(timeout=3)
        _release(handler)

    dumps = [r.getMessage() for r in records
             if "every thread's stack follows" in r.getMessage()]
    assert dumps, (
        "a stall well past all_threads_after produced no all-thread dump, so "
        "the holder could not have been named"
    )
    dump = dumps[0]

    assert "_the_function_that_HOLDS_the_lock" in dump, (
        f"the dump does not contain the HOLDER's frame. The loop's own stack "
        f"names only the blocked party; if this is missing the instrument "
        f"still cannot answer 'who had it'.  Got:\n{dump}"
    )
    assert "the-holder-thread" in dump, (
        f"the holder's frame is present but its THREAD NAME is not. A bare "
        f"thread id cannot say which subsystem it belongs to, which is the "
        f"whole reason names are dumped.  Got:\n{dump}"
    )
    assert "<-- THE LOOP" in dump, (
        "the loop thread is not marked in the dump, so a reader cannot tell "
        "the waiter from the candidates"
    )


def test_a_short_stall_does_not_dump_every_thread():
    """Below the second threshold the log is exactly as it was.

    A daemon has enough threads that dumping all of them on every brief
    hiccup would cost more than it tells.
    """
    handler, records = _capture()
    dog = LoopWatchdog(
        interval=0.1, threshold=0.3, resample_every=0.2,
        all_threads_after=30.0,      # far above the stall this test creates
    )
    try:
        async def scenario():
            dog.start()
            await asyncio.sleep(0.3)
            _the_function_that_blocks_the_loop()   # ~1.2s, well under 30s
            await asyncio.sleep(0.3)

        asyncio.run(scenario())
    finally:
        dog.stop()
        _release(handler)

    assert [r for r in records if "LOOP_STALL" in r.getMessage()], (
        "the stall itself was not reported, so this test proves nothing "
        "about the all-thread gate"
    )
    assert not [r for r in records
                if "every thread's stack follows" in r.getMessage()], (
        "a stall shorter than all_threads_after dumped every thread anyway "
        "-- the gate does not gate"
    )


def test_a_thread_sitting_still_is_reported_as_unchanged():
    """The elapsed-held signal.

    A thread holding one lock for a whole stall and a thread churning through
    work both appear in every dump. Only the time on an IDENTICAL stack
    separates them, and without it the two read the same.
    """
    handler, records = _capture()
    lock = threading.Lock()
    holder = threading.Thread(
        target=_the_function_that_HOLDS_the_lock,
        args=(lock, 2.0),
        name="the-holder-thread",
        daemon=True,
    )
    dog = LoopWatchdog(
        interval=0.1, threshold=0.3, resample_every=0.4,
        all_threads_after=0.4,
    )
    try:
        async def scenario():
            dog.start()
            await asyncio.sleep(0.2)
            holder.start()
            time.sleep(0.2)
            with lock:
                pass
            await asyncio.sleep(0.3)

        asyncio.run(scenario())
    finally:
        dog.stop()
        holder.join(timeout=4)
        _release(handler)

    dumps = [r.getMessage() for r in records
             if "every thread's stack follows" in r.getMessage()]
    assert len(dumps) >= 2, (
        f"need at least two dumps to say anything was UNCHANGED between "
        f"them; got {len(dumps)}"
    )
    assert "unchanged for" in dumps[-1], (
        f"no thread was reported as sitting on the same stack across two "
        f"dumps, so a held lock and a busy thread still read identically.  "
        f"Got:\n{dumps[-1]}"
    )
