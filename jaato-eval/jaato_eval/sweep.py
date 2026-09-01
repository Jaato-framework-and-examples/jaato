"""Sweep driver — the cartesian product of task × profile set × repeat.

This is the module that turns the three questions into commands:

===============================  ===================================
Can I use a cheaper model?       vary ``profile_set``, hold task fixed
Can I simplify the prompt?       vary the persona, hold profile fixed
Can I drop the harness tools?    vary ``plugins`` in a profile variant
===============================  ===================================

Only the first is a sweep axis here; the other two are profile-set or
task edits, because they change what the arm *is* rather than which cell
of the matrix it occupies.

CONCURRENCY AND THE RUNNER POOL
===============================

Each simultaneous arm needs its own pre-warm runner slot.  Sequential
stages reuse one slot via the framework's ``slot.settled`` handoff, but a
parallel sweep does not — so ``JAATO_RUNNER_POOL_SIZE`` must be at least
``concurrency`` or arms will queue on a cold spawn (~30s each) instead of
claiming a warm slot (~7s).  :func:`pool_size_advice` states the number;
the driver logs it rather than silently under-performing.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .arm import ArmResult, ArmSpec
from .manifest import TaskManifest
from .pool import CascadePools
from .results import ResultStore
from .runner import run_arm

#: Default parallelism.  Conservative: every arm is a live model session,
#: and the daemon's own pool is the real ceiling.
DEFAULT_CONCURRENCY = 2


def build_matrix(tasks: Sequence[TaskManifest],
                 profile_sets: Sequence[Optional[str]]) -> List[ArmSpec]:
    """Expand tasks × profile sets × repeats into arms.

    ``repeats`` comes from each task, so a flaky task can ask for more
    samples than a deterministic one without inflating the whole sweep.

    An empty ``profile_sets`` means "whatever each task declares" — one
    arm per task per repeat, which is the single-configuration case.
    """
    sets: Sequence[Optional[str]] = profile_sets or [None]
    return [
        ArmSpec(task=task, profile_set=ps, repeat=r)
        for task in tasks
        for ps in sets
        for r in range(task.repeats)
    ]


def pool_size_advice(concurrency: int) -> str:
    """The ``JAATO_RUNNER_POOL_SIZE`` this sweep wants, as advice text."""
    return (f"JAATO_RUNNER_POOL_SIZE should be >= {concurrency} for this sweep; "
            "each simultaneous arm needs its own warm slot, and a short pool "
            "means arms cold-spawn (~30s) instead of claiming a slot (~7s)")


async def run_sweep(arms: Sequence[ArmSpec], *, store: ResultStore,
                    workspace_root: Path,
                    concurrency: int = DEFAULT_CONCURRENCY,
                    socket_path: Optional[str] = None,
                    keep_workspaces: bool = False,
                    resume: bool = False,
                    arm_timeout_seconds: Optional[float] = None,
                    on_result: Optional[Callable[[ArmResult], None]] = None,
                    ) -> List[ArmResult]:
    """Run every arm with bounded concurrency, appending as they land.

    Args:
        arms: From :func:`build_matrix`.
        store: Results are appended here the moment each arm finishes, so
            a sweep killed halfway leaves everything completed readable.
        workspace_root: Parent for per-arm scratch workspaces.
        concurrency: Simultaneous arms.  See the module docstring on
            matching this to the daemon's runner pool.
        resume: Skip arms already present in ``store``.  Makes a killed
            sweep restartable without re-spending on completed arms.
        on_result: Called with each result as it lands, for progress
            reporting.  Exceptions from it are not caught — a broken
            progress callback should be loud, not silently swallowed
            alongside real results.

    Returns:
        Results in completion order (not matrix order).
    """
    todo = list(arms)
    if resume:
        done = store.completed_arm_ids()
        todo = [a for a in todo if a.arm_id not in done]

    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: List[ArmResult] = []
    lock = asyncio.Lock()

    # One pool per budgeted task, declared before any arm starts and owned
    # for the whole sweep: a cascade pool belongs to the connection that
    # declared it, so an owner opened and closed around a single arm would
    # take the pool with it.
    tasks = {a.task.task_id: a.task for a in todo}

    async def one(spec: ArmSpec, pools: CascadePools) -> None:
        async with semaphore:
            # READ INSIDE THE SEMAPHORE, immediately before the arm starts.
            # Taken up front for every arm it would describe the pool as it
            # was before the sweep began — the same number on every row, and
            # useless for the question it exists to answer: how much was
            # already gone when THIS arm arrived.
            on_arrival = await pools.snapshot(spec.task.task_id)
            result = await run_arm(
                spec, workspace_root=workspace_root,
                socket_path=socket_path, keep_workspace=keep_workspaces,
                cascade_driver_id=pools.cid_for(spec.task.task_id),
                arm_timeout_seconds=arm_timeout_seconds,
                pool_on_arrival=on_arrival)
        # Serialise the append: JSONL tolerates interleaved *records* but
        # not interleaved *bytes* from concurrent writers.
        async with lock:
            store.append(result)
            results.append(result)
            if on_result is not None:
                on_result(result)

    async with CascadePools(list(tasks.values()), socket_path=socket_path,
                            workspace_path=str(workspace_root)) as pools:
        await asyncio.gather(*(one(spec, pools) for spec in todo))
    return results
