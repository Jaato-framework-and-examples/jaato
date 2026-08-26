"""A blocking threadsafe call from the loop thread must refuse, not deadlock.

#631's watchdog answered the loop-stall question in its first ninety seconds:
seven stalls, seven identical stacks — the RPC read loop dispatching a
notification inline, the ``on_turn_progress`` hook hitting a context-limit
cache miss, and the miss path calling ``session_get_context_limit_threadsafe``
FROM the loop thread.  ``run_coroutine_threadsafe`` schedules onto the loop,
``future.result`` blocks the loop waiting for it: a self-deadlock broken only
by its own 10s ceiling.  Recovery times were a CONSTANT — 10.3/10.6/11.1/11.9s
(21s = two stacked) — and the timeout's exception path left the cache
unhealed, so one cold miss repeated on every streaming notification forever.

Three changes, one per test group:

1. the guard: every ``*_threadsafe`` entry refuses from the loop thread,
   BEFORE scheduling, loudly;
2. the hooks: a cache miss emits with the limit unknown and schedules a
   NON-BLOCKING off-band fill;
3. E.1: a ``usage_update`` payload carrying a limit heals the cache — the
   recovery path the /model invalidation comment always promised and never
   had.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from server.runner_rpc_client import RunnerRPCClient


# ---------------------------------------------------------------- the guard


def _client_on(loop: asyncio.AbstractEventLoop) -> RunnerRPCClient:
    client = RunnerRPCClient.__new__(RunnerRPCClient)
    client._loop = loop
    return client


def test_a_threadsafe_call_from_the_loop_thread_refuses_immediately():
    async def scenario():
        client = _client_on(asyncio.get_running_loop())
        started = time.monotonic()
        with pytest.raises(RuntimeError) as caught:
            client._run_threadsafe(asyncio.sleep(0), timeout=10.0)
        elapsed = time.monotonic() - started
        return str(caught.value), elapsed

    msg, elapsed = asyncio.run(scenario())
    assert "self-deadlock" in msg
    assert elapsed < 1.0, (
        f"the refusal took {elapsed:.1f}s -- it must fire BEFORE scheduling, "
        f"not after a timeout; live this was a constant 10s stall per event"
    )


def test_the_guard_refuses_before_scheduling_not_after():
    """Raising after ``run_coroutine_threadsafe`` would leave the coroutine
    queued to run with its side effects once the loop frees up, detached
    from any caller.  So nothing may reach the loop."""
    ran = []

    async def side_effect():
        ran.append(True)

    async def scenario():
        client = _client_on(asyncio.get_running_loop())
        with pytest.raises(RuntimeError):
            client._run_threadsafe(side_effect(), timeout=1.0)
        await asyncio.sleep(0.05)      # give a leaked coroutine time to run

    asyncio.run(scenario())
    assert not ran, (
        "the guard scheduled the coroutine before refusing -- its side "
        "effects ran detached from any caller"
    )


def test_the_same_call_from_a_worker_thread_still_works():
    """The guard must not tax the correct usage the wrappers exist for."""
    result = []

    async def scenario():
        loop = asyncio.get_running_loop()
        client = _client_on(loop)

        async def answer():
            return 42

        def worker():
            result.append(client._run_threadsafe(answer(), timeout=5.0))

        t = threading.Thread(target=worker)
        t.start()
        while t.is_alive():
            await asyncio.sleep(0.01)
        t.join()

    asyncio.run(scenario())
    assert result == [42]


def test_all_four_blocking_entries_are_guarded():
    """Checked in source: the guard must cover the CLASS, not one instance."""
    import inspect

    import server.runner_rpc_client as mod

    src = inspect.getsource(mod)
    for name in ("call_threadsafe", "bootstrap_session_threadsafe",
                 "session_send_message_threadsafe", "_run_threadsafe"):
        idx = src.index(f"def {name}(")
        window = src[idx:idx + 3000]
        assert "_guard_not_on_loop(" in window, (
            f"{name} blocks on the loop without the on-loop guard -- a "
            f"callback calling it from the loop thread deadlocks for its "
            f"full timeout, silently, per event"
        )


# ------------------------------------------------- the non-blocking miss path


def _bare_server():
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    srv._cached_context_limit = None
    srv._context_limit_fill_inflight = False
    srv._runner_rpc = None
    return srv


def test_the_fill_never_blocks_the_calling_thread():
    """The scheduler must only ENQUEUE — from any thread, including the
    loop's own.  Blocking here is the entire original defect."""
    srv = _bare_server()

    class _RPC:
        async def session_get_context_limit(self, *, timeout=5.0):
            await asyncio.sleep(0.05)
            return 128000

    async def scenario():
        rpc = _RPC()
        rpc._loop = asyncio.get_running_loop()
        srv._runner_rpc = rpc

        started = time.monotonic()
        srv._schedule_context_limit_fill()      # from the loop thread itself
        elapsed = time.monotonic() - started
        assert elapsed < 0.02, f"scheduling blocked for {elapsed:.3f}s"

        await asyncio.sleep(0.15)               # let the fill land

    asyncio.run(scenario())
    assert srv._cached_context_limit == 128000, (
        "the off-band fill never healed the cache -- the miss repeats on "
        "every notification, which is the forever-stall shape"
    )


def test_a_stampede_of_misses_schedules_one_fill():
    srv = _bare_server()
    calls = []

    class _RPC:
        async def session_get_context_limit(self, *, timeout=5.0):
            calls.append(1)
            await asyncio.sleep(0.1)
            return 1000

    async def scenario():
        rpc = _RPC()
        rpc._loop = asyncio.get_running_loop()
        srv._runner_rpc = rpc
        for _ in range(10):                     # ten notifications, one miss
            srv._schedule_context_limit_fill()
        await asyncio.sleep(0.25)

    asyncio.run(scenario())
    assert len(calls) == 1, (
        f"{len(calls)} fills for one cold cache -- a busy stream would "
        f"stampede the runner"
    )


def test_no_rpc_no_loop_is_a_noop_not_a_crash():
    srv = _bare_server()
    srv._schedule_context_limit_fill()          # rpc is None
    assert srv._cached_context_limit is None
