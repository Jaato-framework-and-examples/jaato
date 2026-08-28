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


# ---------------------------------------------------------------------------
# Separating the holder from the waiters
#
# The dump above names every thread and how long each has sat still.  Run
# against a real daemon that answered the wrong question: 47 threads, ~40
# reading "unchanged for 20.0s", because A PARKED THREAD IS UNCHANGED BY
# DEFINITION.  Idle readers, reapers and a dozen blocked watchers all looked
# exactly like a holder.
#
# What discriminates is stack CONTENT: a waiter's innermost frame IS the
# `with ...lock:` statement (acquiring a lock blocks in C, creating no Python
# frame), while the holder got past it and is stopped somewhere else.
#
# These build a REAL convoy: one holder, several blocked waiters, on one lock.
# ---------------------------------------------------------------------------

def _the_function_that_WAITS_for_the_lock(lock) -> None:
    """Named so the waiter group has something unmistakable to contain."""
    with lock:
        pass


def _run_a_convoy(dog, lock, n_waiters, hold_for=1.6):
    """Hold *lock*, park *n_waiters* threads on it, capture the dumps."""
    handler, records = _capture()
    holder = threading.Thread(
        target=_the_function_that_HOLDS_the_lock, args=(lock, hold_for),
        name="the-holder-thread", daemon=True,
    )
    waiters = [
        threading.Thread(target=_the_function_that_WAITS_for_the_lock,
                         args=(lock,), name=f"waiter-{i}", daemon=True)
        for i in range(n_waiters)
    ]
    try:
        async def scenario():
            dog.start()
            await asyncio.sleep(0.2)
            holder.start()
            time.sleep(0.2)              # let the holder take it
            for w in waiters:
                w.start()
            time.sleep(0.2)              # let them pile up on it
            with lock:                   # the LOOP joins the convoy
                pass
            await asyncio.sleep(0.3)

        asyncio.run(scenario())
    finally:
        dog.stop()
        holder.join(timeout=4)
        for w in waiters:
            w.join(timeout=4)
        _release(handler)

    return [r.getMessage() for r in records
            if "every thread's stack follows" in r.getMessage()]


def test_the_holder_is_not_buried_among_the_waiters():
    """The holder lands in the candidate set; the waiters do not."""
    dog = LoopWatchdog(interval=0.1, threshold=0.3, resample_every=0.3,
                       all_threads_after=0.4)
    dumps = _run_a_convoy(dog, threading.Lock(), n_waiters=4)
    assert dumps, "no all-thread dump was produced"
    dump = dumps[0]

    head, _, tail = dump.partition("=== BLOCKED ACQUIRING A LOCK")
    assert tail, f"the dump is not partitioned at all:\n{dump}"

    assert "_the_function_that_HOLDS_the_lock" in head, (
        "the HOLDER is not in the candidate section. It is the one thread the "
        "reader needs; putting it among forty parked threads is the problem "
        f"this partition exists to fix.\n\n{dump}"
    )
    assert "_the_function_that_WAITS_for_the_lock" not in head, (
        "a thread blocked ACQUIRING the lock was listed as a holder "
        f"candidate. It holds nothing — it never got the lock.\n\n{dump}"
    )
    assert "waiter-0" in tail, (
        f"the blocked waiters are not in the waiter section.\n\n{dump}"
    )


def test_a_convoy_is_one_entry_not_one_stack_each():
    """Threads blocked on the SAME line collapse to a single entry.

    The live report had twelve workspace-monitor threads on one line. Twelve
    near-identical stacks is what makes a dump unreadable at daemon scale.
    """
    dog = LoopWatchdog(interval=0.1, threshold=0.3, resample_every=0.3,
                       all_threads_after=0.4)
    dumps = _run_a_convoy(dog, threading.Lock(), n_waiters=6)
    assert dumps, "no all-thread dump was produced"
    dump = dumps[0]

    _, _, tail = dump.partition("=== BLOCKED ACQUIRING A LOCK")
    for i in range(6):
        assert f"waiter-{i}" in tail, f"waiter-{i} missing from the dump"

    # Assert the PROPERTY: all six waiters sit under ONE group entry.
    #
    # Two earlier versions of this were wrong in opposite directions. The
    # first counted the waiter FUNCTION's name, which the waiter section
    # never prints -- 0 either way, so it passed with the grouping torn out.
    # The second counted the source text "with lock:", which the LOOP's own
    # blocking line also starts with, so a correct dump counted 2.
    #
    # A group entry is a line at two-space indent; its members are the
    # deeper-indented lines under it. Counting members per group is the thing
    # grouping actually does.
    groups: dict = {}
    current = None
    for raw in tail.splitlines():
        if raw.startswith("    ") and current is not None:
            groups[current].append(raw.strip())
        elif raw.startswith("  ") and raw.strip():
            current = raw.strip()
            groups[current] = []

    holding = [g for g, members in groups.items()
               if any("waiter-" in m for m in members)]
    assert len(holding) == 1, (
        f"the six waiters are spread over {len(holding)} group entries; "
        f"threads stopped on ONE line must collapse to ONE entry. Twelve "
        f"near-identical stacks is what made the live dump unreadable.\n\n"
        f"{tail}"
    )
    members = groups[holding[0]]
    assert len(members) == 6, (
        f"the waiters' group lists {len(members)} threads, expected 6 — "
        f"grouping is dropping or duplicating members.\n\n{tail}"
    )


def test_the_loop_is_reported_as_a_waiter_when_it_is_one():
    """The loop blocked on a lock is BLOCKED, and must not read as a holder.

    It is the thread the reader arrives caring about, so mislabelling it is
    the most expensive mistake the dump can make.
    """
    dog = LoopWatchdog(interval=0.1, threshold=0.3, resample_every=0.3,
                       all_threads_after=0.4)
    dumps = _run_a_convoy(dog, threading.Lock(), n_waiters=2)
    assert dumps, "no all-thread dump was produced"
    dump = dumps[0]

    head, _, tail = dump.partition("=== BLOCKED ACQUIRING A LOCK")
    assert "<-- THE LOOP" in tail, (
        "the loop was blocked acquiring the lock, so it belongs in the waiter "
        f"section. Reporting it as a candidate points the reader at the one "
        f"thread that certainly is not the holder.\n\n{dump}"
    )
    assert "<-- THE LOOP" not in head, (
        f"the loop appears in the candidate section while blocked.\n\n{dump}"
    )
