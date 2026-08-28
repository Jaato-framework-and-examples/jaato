"""Run one arm: materialise, execute, grade.

This is the only module that talks to the daemon, and it does so through
``jaato_sdk`` alone — never ``shared.*``.  The constraint is load-bearing:
if the eval engine can be built on the SDK, the SDK is sufficient for
third-party drivers.  Where it cannot, the gap is a real SDK defect and
belongs in the SDK, not in a private import here.  That constraint has
already paid once: the tool-call ledger was unreachable over the SDK, and
jaato #639 / #640 closed it in the SDK rather than here (see
:mod:`jaato_eval.ledger`).

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
from .ledger import build_ledger_result
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
        self.termination_reason = ""
        self.termination_detail = ""

    def on_terminated(self, event: Any) -> None:
        """Record why the session wound down.

        ``SessionTerminatedEvent.reason`` is the only place an abnormal
        stop names ITSELF.  A budget ceiling in particular short-circuits
        BEFORE any turn runs, so no ``TurnCompletedEvent`` fires and the
        per-turn ``finish_reason`` never mentions it — the SDK's own
        docstring warns that a driver reading only turns reports "a
        generic failure ... a ceiling stop indistinguishable from a
        break".  That is exactly what this engine did until it subscribed
        here.

        ``natural`` / ``client_request`` / ``stopped`` are ordinary
        wind-downs and say nothing about completeness; only
        ``budget_exhausted`` and ``error`` name a stop.
        """
        self.termination_reason = getattr(event, "reason", "") or ""
        detail = (getattr(event, "details", None)
                  or getattr(event, "error_summary", None) or "")
        self.termination_detail = str(detail)

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


#: Wall-clock ceiling for one arm's session, in seconds.  The harness owns
#: this because nothing else can: a task pool's ``seconds`` is reconciled
#: when a session ENDS, so a session that never ends never consumes it and
#: the pool cannot abort it.  Measured twice — a slow model kept turning
#: past sixteen minutes while its sibling finished in one, and each time the
#: sweep died on the operator's own `timeout`, losing the report and one
#: arm's result with it.
DEFAULT_ARM_TIMEOUT_SECONDS = 900.0


async def run_arm(spec: ArmSpec, *, workspace_root: Path,
                  socket_path: Optional[str] = None,
                  keep_workspace: bool = False,
                  cascade_driver_id: Optional[str] = None,
                  arm_timeout_seconds: Optional[float] = None) -> ArmResult:
    """Execute and grade one arm.

    Args:
        spec: What to run.
        workspace_root: Parent directory for this arm's scratch workspace.
        socket_path: Daemon IPC socket; ``None`` uses the client default.
        keep_workspace: Leave the workspace on disk after grading.  Set
            when a human needs to inspect what the agent actually did.
        arm_timeout_seconds: Wall-clock ceiling for the session.  An arm
            that exceeds it is BLOCKED — it was cut short, so it says
            nothing about the configuration under test.  ``None`` uses
            :data:`DEFAULT_ARM_TIMEOUT_SECONDS`; pass ``0`` to disable.
        cascade_driver_id: The task's cascade pool (see
            :mod:`jaato_eval.pool`).  ``None`` runs the arm un-pooled.
            An arm whose profile declares its own ``budget_control`` is on
            its own books and does not draw on the pool even when given
            one — that is the framework's rule, not this engine's.

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
        limit = (DEFAULT_ARM_TIMEOUT_SECONDS if arm_timeout_seconds is None
                 else float(arm_timeout_seconds))
        run = _run_session(spec, workspace, socket_path=socket_path,
                           cascade_driver_id=cascade_driver_id)
        payload, accumulator, history = (
            await asyncio.wait_for(run, timeout=limit) if limit > 0 else await run)
    except asyncio.TimeoutError:
        result.blocked_reason = (
            f"arm exceeded the harness ceiling of {limit:.0f}s and was cut "
            "short — BLOCKED, not FAIL: a run that did not finish says "
            "nothing about the configuration under test")
        result.duration_seconds = time.monotonic() - started
        if not keep_workspace:
            discard(workspace)
        return result
    except Exception as exc:  # noqa: BLE001 — any session failure is BLOCKED
        result.blocked_reason = _describe_session_failure(exc)
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

    ledger = build_ledger_result(history)
    context = GraderContext(
        workspace_path=workspace.path,
        config_root=task.resolved_config_root(),
        agent_params=dict(task.input.agent_params),
        payload=payload,
        ledger=ledger,
        history=history,
        usage=result.usage,
        finish_reason=accumulator.finish_reason,
        termination_reason=accumulator.termination_reason,
        termination_detail=accumulator.termination_detail,
        turns=accumulator.turns,
        socket_path=socket_path,
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


def _describe_session_failure(exc: Exception) -> str:
    """Name the failure, and name a pool refusal as itself.

    The daemon states WHY it refused in ``SessionRefused.error_type``; a
    cascade pool with no headroom refuses the spawn rather than starting
    a session that cannot run a turn.  Reported as a bare ``repr`` that
    reads as an unexplained daemon failure — the operator cannot tell "the
    pool I declared did its job" from "the daemon is broken", which are
    opposite calls to action.

    The type is read, never inferred: a refusal whose type the daemon did
    not supply keeps its generic description rather than being given a
    likely-looking one.
    """
    if isinstance(exc, PoolRefused):
        return (f"spawn refused: the task's cascade budget pool is exhausted "
                f"({exc}) — no arm was started, so this says nothing about "
                f"the configuration under test")
    error_type = getattr(exc, "error_type", None)
    if error_type and "CascadeExhausted" in str(error_type):
        return (f"spawn refused: the task's cascade budget pool is exhausted "
                f"({error_type}) — no arm was started, so this says nothing "
                f"about the configuration under test")
    if error_type:
        return f"session refused by the daemon ({error_type}): {exc}"
    return f"session: {exc!r}"


class PoolRefused(RuntimeError):
    """The task's cascade pool had no headroom and refused this spawn."""


class _ArmSession:
    """``IPCClient.session`` with the one hook the facade does not offer.

    A cascade pool that refuses a spawn announces it as an ``ERROR`` event
    carrying ``error_type="CascadeExhaustedError"`` — and ``create_session``
    still returns a session id for the refused session.  Verified live: a
    spawn into an exhausted pool comes back with a perfectly ordinary sid,
    then dies thirty seconds later on a generic "session runner not ready"
    timeout that names nothing.

    So the refusal is only visible to a handler subscribed BEFORE
    ``create_session``, which the facade gives no way to install (it
    connects and creates inside ``__aenter__``).  This mirrors
    ``_SessionContext`` — including disconnecting on any exception out of
    create, which is where the facade's own comment records a leak it had
    to fix — and adds the subscription.  The send-and-wait recipe still
    comes from the public ``Session`` wrapper, so PR #399's
    SESSION_TERMINATED-only hang is not reproduced here.

    The SDK pins the ordering this relies on: handlers are dispatched
    before ``create_session``'s waiter is released, so a refusal is
    already latched by the time the sid is in hand.
    """

    def __init__(self, kwargs: Dict[str, Any],
                 handlers: Dict[str, Any]) -> None:
        self._kwargs = dict(kwargs)
        self._handlers = dict(handlers)
        self._client: Any = None
        self.refusal: Optional[str] = None

    async def __aenter__(self):
        """Connect, subscribe, create — and in that order.

        The ORDER is required for the refusal watch and only for it: a
        cascade pool with no headroom announces the refusal while
        ``create_session`` is still in flight, so a handler installed
        afterwards never sees it.  The other subscriptions go in here for
        consistency rather than necessity — moving them after create was
        tried against a live daemon and changed nothing, which is worth
        recording so the next reader does not re-derive it.

        This class once also registered as a cascade observer, because a
        cid'd session that reached its signal_completion terminus received
        no TURN_COMPLETED and no HISTORY at all.  jaato #643 found the
        cause — SessionTerminatedEvent was emitted BEFORE the final
        TurnCompletedEvent, so a policy that detached on the terminal
        event stranded the turn event — and fixed it for every consumer.
        Retested on 9a4bf437: a pooled arm reports turns=1 with no
        registration, where it reported turns=0 before.  The call is gone
        rather than kept as insurance; it encoded an explanation that is
        now false, which is worse than a redundant RPC.
        """
        from jaato_sdk.client.convenience import Session
        from jaato_sdk.client.ipc import IPCClient
        from jaato_sdk.events import ClientType, EventType

        create_keys = ("profile", "agent", "agent_params", "cascade_driver_id")
        create_kwargs = {k: v for k, v in self._kwargs.items() if k in create_keys}
        ctor_kwargs = {k: v for k, v in self._kwargs.items() if k not in create_keys}
        ctor_kwargs.setdefault("client_type", ClientType.API)

        self._client = IPCClient(**ctor_kwargs)
        if not await self._client.connect(timeout=120):
            raise ConnectionError(
                "could not connect to / autostart the jaato daemon — "
                "run `python -m jaato_sdk.doctor`")

        def on_error(event: Any) -> None:
            error_type = str(getattr(event, "error_type", "") or "")
            if "CascadeExhausted" in error_type:
                self.refusal = (str(getattr(event, "error", "")) or error_type)

        for event_type in (EventType.ERROR, EventType.AGENT_ERROR):
            self._client.subscribe(event_type, on_error)
        for name, handler in self._handlers.items():
            self._client.subscribe(getattr(EventType, name), handler)

        try:
            sid = await self._client.create_session(**create_kwargs)
        except BaseException:
            await self._client.disconnect()
            raise
        if self.refusal:
            await self._client.disconnect()
            raise PoolRefused(self.refusal)
        return Session(self._client, sid)

    async def __aexit__(self, *exc: Any) -> bool:
        if self._client is not None:
            await self._client.disconnect()
        return False


async def _run_session(spec: ArmSpec, workspace: Workspace, *,
                       socket_path: Optional[str],
                       cascade_driver_id: Optional[str] = None):
    """Open the session, send the prompt, return payload + facts + history.

    The ``.env`` written into the workspace carries ``JAATO_PROFILE_SET``;
    ``env_file`` is resolved relative to ``workspace_path``, which is how
    the sweep's model axis reaches profile discovery without the engine
    having to mutate the task's own configuration.
    """
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
    if socket_path:
        kwargs["socket_path"] = socket_path
    if cascade_driver_id:
        kwargs["cascade_driver_id"] = cascade_driver_id

    accumulator = _TurnAccumulator()
    # None until a HistoryEvent actually lands.  Seeding this with [] would
    # make "no history arrived" indistinguishable from "the agent made no
    # tool calls" — see build_ledger_result on why that difference decides
    # whether a verdict is about the agent or about this engine.
    history: Optional[List[Dict[str, Any]]] = None
    history_ready = asyncio.Event()

    def on_history(event: Any) -> None:
        nonlocal history
        history = list(getattr(event, "history", []) or [])
        history_ready.set()

    handlers = {
        "TURN_COMPLETED": accumulator.on_turn,
        "SESSION_TERMINATED": accumulator.on_terminated,
        "HISTORY": on_history,
    }

    async with _ArmSession(kwargs, handlers) as session:
        client = session.client
        payload = await session.complete(task.input.prompt)
        await client.request_history()
        try:
            await asyncio.wait_for(history_ready.wait(), timeout=30)
        except asyncio.TimeoutError:
            # ``history`` stays None, so the ledger comes back UNFAITHFUL
            # and ledger-reading graders BLOCK.  This comment used to claim
            # that outcome while the code seeded history with [] — which
            # the ledger judged faithful-and-empty, so those graders ran on
            # a phantom and returned FAIL about the agent instead.
            pass

    return payload, accumulator, history
