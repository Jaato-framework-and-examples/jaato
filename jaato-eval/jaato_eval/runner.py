"""Run one arm: materialise, execute, grade.

This is the only module that talks to the daemon, and it does so through
``jaato_sdk`` alone — never ``shared.*``.  The constraint is load-bearing:
if the eval engine can be built on the SDK, the SDK is sufficient for
third-party drivers.  Where it cannot, the gap is a real SDK defect and
belongs in the SDK, not in a private import here.  (One such gap is
already live — see :mod:`jaato_eval.ledger`.)

FAILURE TAXONOMY
================

Everything that can go wrong is sorted into exactly one of two buckets,
and the sorting is the point:

*The agent was exercised and did something wrong* → its graders decide,
and they may return FAIL.

*The agent was not exercised* → the arm is BLOCKED and carries a reason.
Fixture could not be copied, daemon unreachable, session errored, budget
tripped, provider cut the turn short.  None of these say anything about
the configuration under test, and averaging them into a pass rate is how
a sweep concludes that the cheap model is worse when its provider merely
rate-limited you.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .arm import ArmResult, ArmSpec
from .fixture import FixtureError, Workspace, discard, materialise
from .graders import REGISTRY, GraderContext
from .ledger import LedgerResult, build_ledger
from .results import canonical_hash
from .verdict import Verdict

#: Usage keys summed across turns rather than taken from the last turn.
#: ``total_tokens`` is deliberately excluded: for a prompt-inclusive
#: provider it is the end-of-turn CONTEXT SIZE, not spend, so summing it
#: across turns overcounts.  ``spend_total_tokens`` is the billed figure.
_SUMMED_USAGE = ("prompt_tokens", "output_tokens", "spend_total_tokens",
                 "cache_read_tokens", "cache_creation_tokens",
                 "reasoning_tokens", "thinking_tokens")


class _TurnAccumulator:
    """Collects per-turn facts as ``TurnCompletedEvent``s arrive.

    Usage arrives per turn, not once at the end, so an arm's real spend is
    only knowable by accumulating.  ``cost_usd`` stays ``None`` unless at
    least one turn reported a cost — a zero would be indistinguishable
    from "free", which it is not.
    """

    def __init__(self) -> None:
        self.turns = 0
        self.finish_reason = "stop"
        self.usage: Dict[str, Any] = {k: 0 for k in _SUMMED_USAGE}
        self.cost_usd: Optional[float] = None

    def on_turn(self, event: Any) -> None:
        self.turns += 1
        reason = getattr(event, "finish_reason", None)
        if reason:
            self.finish_reason = reason
        usage = getattr(event, "usage", None)
        if usage is None:
            return
        for key in _SUMMED_USAGE:
            value = getattr(usage, key, None)
            if isinstance(value, (int, float)):
                self.usage[key] += value
        cost = getattr(usage, "cost_usd", None)
        if isinstance(cost, (int, float)):
            self.cost_usd = (self.cost_usd or 0.0) + float(cost)

    def snapshot(self) -> Dict[str, Any]:
        out = dict(self.usage)
        out["cost_usd"] = self.cost_usd
        return out


async def run_arm(spec: ArmSpec, *, workspace_root: Path,
                  socket_path: Optional[str] = None,
                  keep_workspace: bool = False) -> ArmResult:
    """Execute and grade one arm.

    Args:
        spec: What to run.
        workspace_root: Parent directory for this arm's scratch workspace.
        socket_path: Daemon IPC socket; ``None`` uses the client default.
        keep_workspace: Leave the workspace on disk after grading.  Set
            when a human needs to inspect what the agent actually did.

    Returns:
        An :class:`ArmResult`, always — this coroutine does not raise for
        conditions it can describe, because one arm blowing up must not
        take the sweep with it.
    """
    task = spec.task
    result = ArmResult(spec=spec)

    try:
        workspace = materialise(
            task.resolved_fixture(),
            workspace_root / spec.arm_id.replace("/", "_").replace("#", "_"),
            profile_set=spec.profile_set or task.harness.profile_set,
        )
    except FixtureError as exc:
        result.blocked_reason = f"fixture: {exc}"
        return result

    started = time.monotonic()
    try:
        payload, accumulator, history = await _run_session(
            spec, workspace, socket_path=socket_path)
    except Exception as exc:  # noqa: BLE001 — any session failure is BLOCKED
        result.blocked_reason = f"session: {exc!r}"
        result.duration_seconds = time.monotonic() - started
        if not keep_workspace:
            discard(workspace)
        return result

    result.duration_seconds = time.monotonic() - started
    result.turns = accumulator.turns
    result.finish_reason = accumulator.finish_reason
    result.usage = accumulator.snapshot()
    if payload is not None:
        result.payload_hash = canonical_hash(payload)

    ledger = build_ledger(history)
    context = GraderContext(
        workspace_path=workspace.path,
        config_root=task.resolved_config_root(),
        agent_params=dict(task.input.agent_params),
        payload=payload,
        ledger=ledger,
        history=history,
        usage=result.usage,
        finish_reason=accumulator.finish_reason,
        turns=accumulator.turns,
    )

    result.verdicts = await _grade(task, context)

    if not keep_workspace:
        discard(workspace)
    return result


async def _grade(task, context: GraderContext) -> List[Verdict]:
    """Run every grader in manifest order, off the event loop.

    Graders are synchronous by contract (see ``graders.judge``), and some
    of them block — a build can take minutes.  Dispatching each through
    ``asyncio.to_thread`` keeps a slow grader on one arm from stalling
    every other arm's event handling.

    Order is significant: ``context.prior_verdicts`` is filled in as we
    go, so a gated grader sees the outcomes of the ones declared above it.
    """
    verdicts: List[Verdict] = []
    for spec in task.graders:
        adapter = REGISTRY[spec.kind](spec)
        verdict = await asyncio.to_thread(adapter.grade, context)
        verdicts.append(verdict)
        context.prior_verdicts[verdict.grader_id] = verdict.state
    return verdicts


async def _run_session(spec: ArmSpec, workspace: Workspace, *,
                       socket_path: Optional[str]):
    """Open the session, send the prompt, return payload + facts + history.

    The ``.env`` written into the workspace carries ``JAATO_PROFILE_SET``;
    ``env_file`` is resolved relative to ``workspace_path``, which is how
    the sweep's model axis reaches profile discovery without the engine
    having to mutate the task's own configuration.
    """
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import EventType

    task = spec.task
    kwargs: Dict[str, Any] = {
        "profile": task.harness.profile,
        "workspace_path": str(workspace.path),
        "config_root": str(task.resolved_config_root()),
        "env_file": ".env",
    }
    if task.input.agent:
        kwargs["agent"] = task.input.agent
    if task.input.agent_params:
        kwargs["agent_params"] = dict(task.input.agent_params)
    if task.environment.apparmor:
        kwargs["apparmor"] = True
    if socket_path:
        kwargs["socket_path"] = socket_path

    accumulator = _TurnAccumulator()
    history: List[Dict[str, Any]] = []

    async with IPCClient.session(**kwargs) as session:
        client = session.client
        unsub_turn = client.subscribe(EventType.TURN_COMPLETED, accumulator.on_turn)

        history_ready = asyncio.Event()

        def on_history(event: Any) -> None:
            history.extend(getattr(event, "history", []) or [])
            history_ready.set()

        unsub_history = client.subscribe_once(EventType.HISTORY, on_history)
        try:
            payload = await session.complete(task.input.prompt)
            await client.request_history()
            try:
                await asyncio.wait_for(history_ready.wait(), timeout=30)
            except asyncio.TimeoutError:
                # No history means graders that need the ledger will find
                # it empty and unfaithful, and will BLOCK — which is the
                # correct outcome, not a reason to fail the arm here.
                pass
        finally:
            unsub_turn()
            unsub_history()

    return payload, accumulator, history
