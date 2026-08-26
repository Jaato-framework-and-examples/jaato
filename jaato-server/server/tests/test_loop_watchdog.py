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
