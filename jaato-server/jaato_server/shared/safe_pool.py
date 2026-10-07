"""Thread-pool executor with pre-task hooks (server 0.6.47+).

Closes a residual gap in :func:`server.apparmor.apparmor_confine`:
while the context manager defensively resets per-thread AppArmor
confinement on entry and restores it on exit, the restoration write
to ``/proc/self/task/<tid>/attr/current`` can fail (kernel transient,
profile removed mid-tool, signal interruption).  Tasks that subsequently
run on the stuck worker WITHOUT going through ``apparmor_confine`` —
enrichment plugins, event-bus broadcasts, workspace-monitor callbacks,
log I/O, IPC ``run_in_executor`` calls — would hit EACCES.

The :class:`SafeThreadPoolExecutor` here wraps :meth:`submit` to
prepend every registered pre-task hook on every submitted task.  The
``server.apparmor`` module registers ``_thread_unconfine_safe`` as a
hook when first imported, so every submission to a SafeThreadPoolExecutor
begins with a best-effort defensive reset.

The module also tracks every constructed pool in a :class:`weakref.WeakSet`
(so dropped pools don't leak) and exposes :func:`sweep_registered_executors`,
which dispatches a one-shot reset to every worker in every registered
pool.  ``server.apparmor.AppArmorManager._teardown_profile_impl`` calls
the sweep BEFORE invoking ``apparmor_parser -R``, ensuring no worker
is left flagged-as-confined to a profile that's about to vanish (which
would surface as a phantom-profile evaluation in the kernel).

Layering: this module lives in ``shared/`` so shared-layer code (tool
runner, subagent plugin, background mixin) can use it without violating
the ``shared → server`` import rule.  Server-side hooks (apparmor,
future security layers) register their reset callables through
:func:`register_pre_task_hook`.
"""

from __future__ import annotations

import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


_PRE_TASK_HOOKS: List[Callable[[], None]] = []
_REGISTERED_EXECUTORS: "weakref.WeakSet[SafeThreadPoolExecutor]" = weakref.WeakSet()


def register_pre_task_hook(hook: Callable[[], None]) -> None:
    """Register a callable to run at the start of every task submitted
    to a :class:`SafeThreadPoolExecutor`.

    Hooks are invoked in registration order.  Each hook's exceptions
    are caught and logged at debug level — hook failures must not
    block the wrapped task.

    Idempotent: registering the same callable twice is a no-op (the
    hook list is checked with ``in`` before appending) so import
    cycles don't double-register.
    """
    if hook not in _PRE_TASK_HOOKS:
        _PRE_TASK_HOOKS.append(hook)


class SafeThreadPoolExecutor(ThreadPoolExecutor):
    """:class:`ThreadPoolExecutor` that runs registered pre-task hooks
    before every submitted task.

    Auto-registers in the module-level ``_REGISTERED_EXECUTORS`` set so
    :func:`sweep_registered_executors` can dispatch reset tasks to every
    worker across every pool.  Cost per task: one Python function call
    plus the cost of the registered hooks (typically a single 27-byte
    syscall to ``/proc/self/task/<tid>/attr/current``, sub-millisecond).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _REGISTERED_EXECUTORS.add(self)

    def submit(self, fn: Callable, /, *args: Any, **kwargs: Any):  # type: ignore[override]
        def _safe_call(*a: Any, **kw: Any):
            for hook in _PRE_TASK_HOOKS:
                try:
                    hook()
                except Exception:  # pragma: no cover — hooks must not block
                    logger.debug(
                        "safe_pool: pre-task hook %r raised; ignoring", hook,
                        exc_info=True,
                    )
            return fn(*a, **kw)
        return super().submit(_safe_call, *args, **kwargs)


def sweep_registered_executors() -> int:
    """Dispatch a one-shot pre-task-hook task to every worker in every
    registered :class:`SafeThreadPoolExecutor`.

    Used by ``server.apparmor.AppArmorManager._teardown_profile_impl``
    before ``apparmor_parser -R`` removes a session's profile, so no
    worker thread is left flagged-as-confined to a soon-to-vanish
    profile.

    Uses a :class:`threading.Barrier` per pool to maximise the chance
    that one task lands on each worker (the barrier's parties=max_workers
    blocks early arrivers until all max_workers tasks have started, so
    each task is picked up by a different worker).  Barrier timeout (2s)
    prevents deadlock if some workers are busy with unrelated work; in
    that case the early arrivers' hooks still fire, the barrier breaks,
    and the remaining busy workers will run the hooks on their NEXT
    submitted task via the wrapped ``submit``.

    Returns:
        Total number of reset tasks dispatched (sum of max_workers
        across all registered, non-shutdown pools).  Useful for
        diagnostics.
    """
    n = 0
    for executor in list(_REGISTERED_EXECUTORS):
        try:
            workers = getattr(executor, "_max_workers", 0) or 0
            if workers <= 0:
                continue
            barrier = threading.Barrier(workers)

            def _trigger_hooks_and_wait() -> None:
                # The wrapped ``submit`` already runs hooks before
                # invoking us; this body just waits on the barrier
                # to keep the worker occupied long enough for sibling
                # workers to claim their own tasks.
                try:
                    barrier.wait(timeout=2.0)
                except threading.BrokenBarrierError:
                    pass  # busy workers timed out — early arrivers reset

            for _ in range(workers):
                executor.submit(_trigger_hooks_and_wait)
            n += workers
        except RuntimeError:
            continue  # executor was shut down; skip
    return n
