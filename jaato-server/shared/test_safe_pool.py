"""Tests for the safe-pool pre-task hook infrastructure (server 0.6.47+).

Pins the contract that:
- Every task submitted to a SafeThreadPoolExecutor runs all registered
  pre-task hooks BEFORE the user's callable.
- Hook exceptions never block the wrapped task.
- sweep_registered_executors dispatches one task per worker per pool.
- Pools are auto-registered on construction; dropped pools don't leak
  via the WeakSet.
"""

import gc
import threading
import time
import weakref

from shared.safe_pool import (
    SafeThreadPoolExecutor,
    register_pre_task_hook,
    sweep_registered_executors,
    _PRE_TASK_HOOKS,
    _REGISTERED_EXECUTORS,
)


def test_pre_task_hook_runs_before_task():
    order = []

    def hook():
        order.append("hook")

    # Register on a fresh hook list to avoid cross-test pollution.
    original = list(_PRE_TASK_HOOKS)
    _PRE_TASK_HOOKS.clear()
    try:
        register_pre_task_hook(hook)
        with SafeThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(lambda: order.append("task"))
            future.result(timeout=2)
        assert order == ["hook", "task"], f"expected hook→task, got {order}"
    finally:
        _PRE_TASK_HOOKS[:] = original


def test_register_idempotent():
    def hook():
        pass

    original = list(_PRE_TASK_HOOKS)
    _PRE_TASK_HOOKS.clear()
    try:
        register_pre_task_hook(hook)
        register_pre_task_hook(hook)  # second call is a no-op
        assert _PRE_TASK_HOOKS == [hook]
    finally:
        _PRE_TASK_HOOKS[:] = original


def test_hook_exception_does_not_block_task():
    """A failing hook is logged-and-swallowed; the wrapped task still runs."""
    ran = []

    def bad_hook():
        raise RuntimeError("boom")

    original = list(_PRE_TASK_HOOKS)
    _PRE_TASK_HOOKS.clear()
    try:
        register_pre_task_hook(bad_hook)
        with SafeThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: ran.append("ran"))
            future.result(timeout=2)
        assert ran == ["ran"]
    finally:
        _PRE_TASK_HOOKS[:] = original


def test_pool_auto_registered():
    """Constructing a SafeThreadPoolExecutor auto-registers it; the
    weak ref drops on garbage collection.
    """
    pool = SafeThreadPoolExecutor(max_workers=1)
    assert pool in _REGISTERED_EXECUTORS

    pool_ref = weakref.ref(pool)
    pool.shutdown(wait=True)
    del pool
    gc.collect()

    # WeakSet drops the entry once the pool is collected.  Asserting
    # exact set size is fragile because other tests in the same run
    # may leak SafeThreadPoolExecutors that haven't been collected yet
    # (e.g. tests that don't shutdown(wait=True) explicitly).  The
    # weak-ref behaviour is what matters; the WeakSet drop follows.
    assert pool_ref() is None


def test_sweep_dispatches_per_worker():
    """sweep_registered_executors fires the registered hook on each
    worker in each pool (verified by counting hook invocations).
    """
    counter = {"n": 0}
    counter_lock = threading.Lock()

    def counting_hook():
        with counter_lock:
            counter["n"] += 1

    original = list(_PRE_TASK_HOOKS)
    _PRE_TASK_HOOKS.clear()
    try:
        register_pre_task_hook(counting_hook)
        with SafeThreadPoolExecutor(max_workers=3) as p1, \
             SafeThreadPoolExecutor(max_workers=2) as p2:
            # Prime each pool so the workers are alive (otherwise the
            # ThreadPoolExecutor lazy-spawns workers and the sweep
            # only reaches workers that already exist).
            for _ in range(3):
                p1.submit(lambda: None).result(timeout=2)
            for _ in range(2):
                p2.submit(lambda: None).result(timeout=2)

            # Reset the counter — primer tasks each fired the hook.
            with counter_lock:
                counter["n"] = 0

            n_dispatched = sweep_registered_executors()
            # 3 workers in p1 + 2 workers in p2 = 5.  The reported
            # number includes only registered SafeThreadPoolExecutors.
            assert n_dispatched >= 5, (
                f"expected at least 5 dispatched (3+2), got {n_dispatched}"
            )

            # Wait for the sweep tasks to complete.  Since each barrier-
            # waiter blocks until all max_workers tasks are running, we
            # need to give them up to 2.5s (the 2s barrier timeout +
            # cleanup).
            time.sleep(0.5)

            # Hook must have fired at least once per worker per pool.
            with counter_lock:
                fired = counter["n"]
            assert fired >= 5, (
                f"hook fired {fired} times; expected at least 5 (one per worker)"
            )
    finally:
        _PRE_TASK_HOOKS[:] = original
