"""Per-task cascade budget pools — the second, independent budget gate.

TWO GATES, NOT ONE
==================

jaato has two budget mechanisms and they are independent by design; an
eval sweep wants both, for different reasons.

**The profile gate** is a per-arm ceiling: ``budget_control:`` in the
arm's own profile, which every task ships in its ``config_root``.  A
session carrying one is a delegation *with its own books* — it runs with
exactly what it declared, is never clamped to a pool's remainder, never
depletes a pool, and is not degraded when a pool crosses a rung.  That is
the right shape for an arm, because arms must be independent
measurements: an arm whose ceiling depended on what earlier arms spent
would produce results that changed with sweep order, and under
concurrency would not be reproducible at all.  This gate needs no code
here — the task declares it and the daemon enforces it.

**The cascade gate** is the aggregate this module implements: a pool
declared once, against which many sessions draw.  Its scope here is one
task — a task's arms (repeats × profile sets) share a pool, so a task
with ``repeats: 20`` cannot run away, and no task can starve another.
Sessions drawing on it are clamped at spawn to what remains, degraded
mid-flight when it crosses a rung, and **refused outright** once it is
empty.

The two do not compose on a single session, and that is the framework's
rule rather than this module's: a session declaring its own
``budget_control`` is on its own books and simply does not draw on the
pool.  So a task whose profile sets all declare ceilings will see its
pool untouched — which is correct, not a bug, and worth knowing before
reading a pool snapshot as evidence that nothing ran.

WHY THE POOL IS NOT A MANIFEST PROPERTY OF THE PROFILE
======================================================

A ceiling is a property of a reusable template; a pool is a runtime
aggregate over one live cascade id.  The same task run twice gets two
pools, and they must not share depletion.  So the manifest's ``budget:``
block names the *limits*, and the cid is minted per sweep.

Reference implementation this follows:
``budget_control_and_model_degradation/runners/fanout_run.py``.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional, Sequence

from .manifest import TaskManifest

#: How long :meth:`CascadePools.snapshot` waits for the daemon's reply.
#: Short deliberately: the snapshot is a REPORTING nicety, and an arm must
#: never be delayed — let alone fail — because a pool reading was slow.
SNAPSHOT_TIMEOUT_SECONDS = 5.0


def _slug(task_id: str) -> str:
    """A cid-safe fragment of the task id, for legible daemon logs."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in task_id)


class CascadePools:
    """Declares one pool per budgeted task, on a single owner client.

    One client declares every cid: ``cascade_budget_set`` takes the cid as
    a parameter, so N pools do not need N connections.  The owner must
    outlive the arms — the pool belongs to the connection that declared
    it — which is why this is an async context manager wrapping the whole
    sweep rather than something a single arm sets up.

    Use::

        async with CascadePools(tasks, socket_path=…, workspace_path=…) as pools:
            ...  run arms, passing pools.cid_for(task.id)  ...

    A task with no ``budget:`` block gets ``None`` and simply runs
    un-pooled, subject only to whatever its profile declares.
    """

    def __init__(self, tasks: Sequence[TaskManifest], *,
                 socket_path: Optional[str] = None,
                 workspace_path: Optional[str] = None) -> None:
        self._tasks = [t for t in tasks if t.budget.limits]
        self._socket_path = socket_path
        self._workspace_path = workspace_path
        self._cids: Dict[str, str] = {}
        self._client: Any = None
        #: cid -> the last ``cascade.budget.get`` reply seen for it.  The
        #: daemon answers on the EVENT STREAM rather than as a return
        #: value, so the reply has to be latched by a handler and matched
        #: back to its cid.
        self._budget_replies: Dict[str, Dict[str, Any]] = {}
        self._reply_arrived = asyncio.Event()

    def cid_for(self, task_id: str) -> Optional[str]:
        """The cascade id this task's arms draw on, or ``None``."""
        return self._cids.get(task_id)

    @property
    def declared(self) -> Dict[str, str]:
        """``task_id`` -> cid, for the tasks that got a pool."""
        return dict(self._cids)

    async def __aenter__(self) -> "CascadePools":
        if not self._tasks:
            # Nothing declared a pool.  Opening an owner connection anyway
            # would make every sweep pay for a feature no task asked for.
            return self

        from jaato_sdk.client.ipc import IPCClient
        from jaato_sdk.events import ClientType

        kwargs: Dict[str, Any] = {"client_type": ClientType.API}
        if self._socket_path:
            kwargs["socket_path"] = self._socket_path
        if self._workspace_path:
            kwargs["workspace_path"] = self._workspace_path
        self._client = IPCClient(**kwargs)
        await self._client.connect(timeout=120)
        self._subscribe_budget_replies()

        for task in self._tasks:
            cid = f"jaato-eval-{_slug(task.task_id)}-{uuid.uuid4().hex[:8]}"
            await self._client.cascade_budget_set(
                cid,
                limits=dict(task.budget.limits),
                degrade=list(task.budget.degrade) or None,
            )
            self._cids[task.task_id] = cid
        return self

    def _subscribe_budget_replies(self) -> None:
        """Latch ``cascade.budget.get`` answers off the event stream.

        The daemon replies to that command with a ``SystemMessageEvent``
        whose ``message`` is JSON — so this handler has to distinguish the
        reply from every other system message the connection sees, and it
        does so by SHAPE: a JSON object carrying ``cascade_driver_id``.
        Matching on anything looser would let an unrelated message be
        recorded as a pool reading, which is the one thing a budget column
        must not do.

        Best-effort: a daemon that answers in a shape this does not
        recognise leaves the reading absent, and an absent reading prints
        as unknown rather than as an empty pool.
        """
        from jaato_sdk.events import EventType

        def on_system_message(event: Any) -> None:
            raw = getattr(event, "message", None)
            if not isinstance(raw, str) or "cascade_driver_id" not in raw:
                return
            try:
                data = json.loads(raw)
            except ValueError:
                return
            if not isinstance(data, dict):
                return
            cid = data.get("cascade_driver_id")
            if not cid:
                return
            self._budget_replies[str(cid)] = data
            self._reply_arrived.set()

        self._client.subscribe(EventType.SYSTEM_MESSAGE, on_system_message)

    async def snapshot(self, task_id: str) -> Optional[Dict[str, Any]]:
        """What this task's pool had left, read now.

        Called by the sweep driver just before an arm starts, so the arm's
        result records the headroom it ARRIVED with.  That is the fact that
        makes a ``budget_exhausted`` terminal legible: three arms sharing a
        $6.00 pool spent $3.81 + $0.17 + $2.03, and the last was killed
        mid-work and recorded BLOCKED — indistinguishable, from the results
        file alone, from a model that failed.

        Returns:
            The daemon's reply — ``{cascade_driver_id, declared, limits,
            remaining, usage_fraction, pressure}`` — or ``None`` when the
            task has no pool, the owner connection is closed, or no reply
            arrives within :data:`SNAPSHOT_TIMEOUT_SECONDS`.  ``None`` is
            "not read", never "empty": a reporting read must not be able to
            manufacture evidence that a pool was exhausted.
        """
        cid = self._cids.get(task_id)
        if not cid or self._client is None:
            return None
        # Drop any earlier reading for this cid before asking, so a STALE
        # one cannot satisfy this wait.  The owner declares every pool on
        # one connection and the wake-up event is shared across cids, hence
        # the loop: another cid's reply sets the event without answering us.
        self._budget_replies.pop(cid, None)
        self._reply_arrived.clear()
        try:
            await self._client.cascade_budget_get(cid)
        except Exception:  # noqa: BLE001 — reporting must not fail an arm
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SNAPSHOT_TIMEOUT_SECONDS
        while True:
            reply = self._budget_replies.get(cid)
            if reply is not None:
                return reply
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(self._reply_arrived.wait(),
                                       timeout=remaining)
            except asyncio.TimeoutError:
                return None
            self._reply_arrived.clear()

    async def __aexit__(self, *exc: Any) -> bool:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None
        return False
