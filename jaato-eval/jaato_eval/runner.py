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

The sorting is by whether the agent was EXERCISED, not by whether the
session ended cleanly, and one terminal makes the difference visible: an
agent that spends the framework's completion-nudge budget
(``NudgeExhausted``) has run, worked and left a workspace, and is missing
only its sign-off.  That belongs in the first bucket, and it reaches it —
see :mod:`jaato_eval.sign_off`, which owns the rule, and ``_run_session``,
which grades through such a terminal instead of raising past the grading.
Every other error terminal still lands in the second.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .arm import ArmResult, ArmSpec
from .fixture import FixtureError, Workspace, discard, materialise
from .graders import REGISTRY, GraderContext
from .ledger import build_ledger_result
from .profile import resolve_budget_ceiling
from .results import canonical_hash
from .sign_off import MAX_COMPLETION_NUDGES, is_unsigned_terminal
from .verdict import Verdict

#: Usage keys summed across turns rather than taken from the last turn.
#: ``total_tokens`` is deliberately excluded: for a prompt-inclusive
#: provider it is the end-of-turn CONTEXT SIZE, not spend, so summing it
#: across turns overcounts.  ``spend_total_tokens`` is the billed figure.
#:
#: The cache pair follows the same rule, and used to break it.
#: ``cache_read_tokens`` / ``cache_creation_tokens`` are the turn's LAST
#: RESPONSE's figures — a level, not spend — so adding them across turns
#: produced neither.  The SDK documents the distinction as load-bearing
#: (``jaato_sdk.events``): under ``model_tiers`` a mid-turn tier switch
#: re-reads the whole prefix cold at the new model, and the last-response
#: figures hide exactly that miss.  The fingerprint of the bug was visible
#: in the archived corpus — three of four Gemini arms reported
#: ``cache_creation`` equal to ``cache_read`` to the token, which is one
#: level reading copied into two fields, not two independent billed sums
#: (jaato #800).  ``spend_cache_read_tokens`` /
#: ``spend_cache_creation_tokens`` are already summed over the turn's
#: responses, the same shape as ``spend_total_tokens``, so summing them
#: across turns is the right operation.
#:
#: ``prompt_tokens`` / ``output_tokens`` were the last pair here with the
#: same defect, and could not be fixed with the cache pair because no spend
#: counterpart reached the wire: the session accumulated ``spend_prompt`` /
#: ``spend_output`` per response and dropped both at the boundary.
#: jaato #802 carries them, so every member of this tuple is now a billed
#: figure and the tuple's name is true of all of it.
_SUMMED_USAGE = ("spend_prompt_tokens", "spend_output_tokens",
                 "spend_total_tokens",
                 "spend_cache_read_tokens", "spend_cache_creation_tokens",
                 "reasoning_tokens", "thinking_tokens")


#: Test seam: the graded context, for suites that must assert on what the
#: graders were HANDED rather than on a verdict downstream of it.  A no-op
#: in production; a stub suite rebinds it.
def _CONTEXT_SPY(context):  # noqa: N802 - a seam, not a class
    return None


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
        self.termination_error_type = ""
        self.agent_error: Optional[str] = None
        self.completion_gap: Optional[str] = None
        # PROVIDER-SIDE FACTS THE WIRE DOES NOT CARRY YET.  Both stay None
        # on every arm today: the OpenRouter provider reads
        # ``native_finish_reason`` off the choice and the routed upstream
        # off the response, and neither reaches TurnCompletedEvent (jaato
        # #766).  Read here anyway, by name, so the per-arm report fills
        # these columns the day the framework reports them — the
        # alternative is a report that keeps printing "—" for a fact the
        # daemon has started sending.
        self.native_finish_reason: Optional[str] = None
        self.upstream_provider: Optional[str] = None

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
        # The terminal's TYPE, alongside its prose.  ``reason="error"``
        # says only that something failed; the type is what separates a
        # daemon that died mid-turn from an agent that finished its work
        # and never called signal_completion (see :mod:`jaato_eval.sign_off`).
        self.termination_error_type = str(
            getattr(event, "error_type", "") or "")

    def note_unsigned(self, exc: Exception) -> None:
        """Record an error terminal the arm is being graded through anyway.

        Called by :func:`_run_session` when ``complete()`` raises a
        terminal :mod:`jaato_eval.sign_off` classifies as *unsigned* — the
        agent worked and left a workspace, it just never called
        ``signal_completion``.  The facts land here rather than on a local
        so that the arm's result and every grader see the same account of
        why no payload arrived, and so a session whose
        ``SessionTerminatedEvent`` never reached us (the exception carries
        the same two fields) is described just as fully.

        Never overwrites what the terminal event already said: the event is
        the daemon's own account, the exception is the SDK's relay of it.
        """
        error_type = str(getattr(exc, "error_type", "") or "")
        if error_type and not self.termination_error_type:
            self.termination_error_type = error_type
        summary = str(getattr(exc, "error_summary", "") or "")
        if summary and not self.termination_detail:
            self.termination_detail = summary
        if not self.termination_reason:
            self.termination_reason = "error"
        self.agent_error = str(exc)

    def on_turn(self, event: Any) -> None:
        self.turns += 1
        reason = getattr(event, "finish_reason", None)
        if reason:
            self.finish_reason = reason
        # LATCHED PER TURN, not read off the last one.  completion_gap
        # rides EXACTLY ONE event and is read-and-cleared, so a session
        # that gave up and then received more work stops reporting it —
        # sampling only the final turn would miss the very turn that
        # carried the fact.  It means "asked twice and refused", not
        # "did not signal on this turn", so a legitimately multi-turn
        # session never sets it.
        gap = getattr(event, "completion_gap", None)
        if gap:
            self.completion_gap = str(gap)
        # LATCHED, not overwritten by a later turn that omits them: a
        # gateway reports the upstream once per response and a normalised
        # finish reason has no native twin on most turns, so "the last turn
        # did not say" must not erase what an earlier one did.
        native = getattr(event, "native_finish_reason", None)
        if native:
            self.native_finish_reason = str(native)
        upstream = getattr(event, "upstream_provider", None)
        if upstream:
            self.upstream_provider = str(upstream)
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
                  arm_timeout_seconds: Optional[float] = None,
                  pool_on_arrival: Optional[Dict[str, Any]] = None) -> ArmResult:
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
        pool_on_arrival: What that pool had left when this arm STARTED, as
            :meth:`jaato_eval.pool.CascadePools.snapshot` read it.  Recorded
            on the result because spend is only legible against what was
            still available: an arm that arrives at a 63%-consumed pool and
            is terminated ``budget_exhausted`` reads as a model failure
            until the reader can see it was billed for a sibling's
            appetite.  ``None`` when the task declared no pool, or when the
            snapshot could not be read — never invented.

    Returns:
        An :class:`ArmResult`, always — this coroutine does not raise for
        conditions it can describe, because one arm blowing up must not
        take the sweep with it.
    """
    task = spec.task
    result = ArmResult(spec=spec)
    # Before anything can fail.  These three are properties of what the arm
    # was ALLOWED, not of what it did, so a fixture error must not cost them
    # — an arm blocked at materialisation still belongs in the per-arm table
    # with its ceilings shown.
    _record_declared_budget(result, spec, pool_on_arrival)

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
    session_ref: Dict[str, Any] = {}
    try:
        limit = (DEFAULT_ARM_TIMEOUT_SECONDS if arm_timeout_seconds is None
                 else float(arm_timeout_seconds))
        # Held by us, not by the coroutine: `asyncio.wait_for` cancels the
        # task on timeout and we would never receive its return value.
        accumulator = _TurnAccumulator()
        run = _run_session(spec, workspace, socket_path=socket_path,
                           cascade_driver_id=cascade_driver_id,
                           accumulator=accumulator,
                           session_ref=session_ref)
        payload, accumulator, history = (
            await asyncio.wait_for(run, timeout=limit) if limit > 0 else await run)
    except asyncio.TimeoutError:
        result.blocked_reason = (
            f"arm exceeded the harness ceiling of {limit:.0f}s and was cut "
            "short — BLOCKED, not FAIL: a run that did not finish says "
            "nothing about the configuration under test")
        result.duration_seconds = time.monotonic() - started
        _record_partial_usage(
            result, accumulator,
            _tracker_usage(workspace, session_ref.get('id')))
        # AFTER the usage read and BEFORE the discard: the session log this
        # counts nudges from lives in the workspace that is about to go.
        _record_binding(result, workspace, session_ref, accumulator)
        if not keep_workspace:
            discard(workspace)
        return result
    except Exception as exc:  # noqa: BLE001 — any session failure is BLOCKED
        result.blocked_reason = _describe_session_failure(exc)
        result.duration_seconds = time.monotonic() - started
        _record_partial_usage(
            result, accumulator,
            _tracker_usage(workspace, session_ref.get('id')))
        _record_binding(result, workspace, session_ref, accumulator)
        if not keep_workspace:
            discard(workspace)
        return result

    _record_binding(result, workspace, session_ref, accumulator)
    result.duration_seconds = time.monotonic() - started
    result.turns = accumulator.turns
    result.finish_reason = accumulator.finish_reason
    result.usage = accumulator.snapshot()
    # Set ONLY for an arm graded through an error terminal (today: a
    # missing signal_completion).  ``blocked_reason`` stays None -- the
    # state must roll up from the verdicts, which is the difference this
    # field exists to record: the arm ran and produced evidence, and what
    # it lacked was the sign-off, not the work.
    result.error = accumulator.agent_error
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
        termination_error_type=accumulator.termination_error_type,
        completion_gap=accumulator.completion_gap,
        turns=accumulator.turns,
        socket_path=socket_path,
        error=accumulator.agent_error,
    )

    _CONTEXT_SPY(context)
    result.verdicts = await _grade(task, context)

    if not keep_workspace:
        discard(workspace)
    return result


def _record_declared_budget(result: "ArmResult", spec: ArmSpec,
                            pool_on_arrival: Optional[Dict[str, Any]]) -> None:
    """Record the two ceilings this arm ran under, and the pool's state.

    THE TWO ARE NOT ALTERNATIVES TO EACH OTHER and the report shows both
    for that reason: a session declaring its own ``budget_control`` is on
    its own books and does not draw on the pool, so a populated
    ``budget_ceiling`` beside an untouched pool is the framework working
    as designed rather than a pool that failed to bind.

    ``budget_ceiling`` stays ``None`` when the profile could not be
    resolved — see :mod:`jaato_eval.profile` on why that must not read as
    "unbudgeted".
    """
    result.budget_ceiling = resolve_budget_ceiling(
        spec.task.resolved_config_root(),
        spec.task.harness.profile,
        spec.profile_set or spec.task.harness.profile_set,
    )
    result.pool_limits = dict(spec.task.budget.limits) or None
    result.pool_on_arrival = pool_on_arrival


def _record_binding(result: "ArmResult", workspace: Workspace,
                    session_ref: Dict[str, Any],
                    accumulator: "_TurnAccumulator") -> None:
    """Record WHICH session this arm was, and what served it.

    Called on every path that opened a session — success, harness timeout
    and session failure alike — because the arm a reader most needs to
    look up upstream is the one that did not finish.  The session id in
    particular is the join key: OpenRouter's console groups by exactly
    this id, so a row carrying it links to the provider's own record of
    the arm, and a row without it cannot be joined to anything.

    Every field is left ``None`` when unknown.  A cut arm may never have
    received its ``SessionInfoEvent``, and naming a model it might have
    bound would be worse than a blank.
    """
    result.session_id = session_ref.get("id")
    result.model = session_ref.get("model")
    result.provider = session_ref.get("provider")
    result.upstream_provider = accumulator.upstream_provider
    result.native_finish_reason = accumulator.native_finish_reason
    result.completion_nudges = _completion_nudges(
        workspace, result.session_id, accumulator.completion_gap)


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
                 handlers: Dict[str, Any],
                 session_ref: Optional[Dict[str, Any]] = None) -> None:
        self._kwargs = dict(kwargs)
        self._handlers = dict(handlers)
        self._client: Any = None
        self.refusal: Optional[str] = None
        # Written to DIRECTLY by the SESSION_INFO handler rather than read
        # off this object afterwards, for the same reason the sid is
        # published early: a timeout cancels the session body, and the
        # binding the report describes must survive that.
        self._session_ref = session_ref if session_ref is not None else {}

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

        def on_session_info(event: Any) -> None:
            """Latch the model and provider the daemon actually BOUND.

            ``profile_set`` is a directory name someone chose; this is the
            binding.  Subscribed BEFORE create for the same reason the
            refusal watch is: the first ``SessionInfoEvent`` is emitted
            while ``create_session`` is still in flight.

            Two of them arrive on a normal create — a snapshot at creation
            and a second once the provider is fully ready — and the second
            is the one that can carry a name the first did not have yet.
            So an empty field never overwrites a populated one, and a
            populated one always wins.
            """
            for key, attribute in (("model", "model_name"),
                                   ("provider", "model_provider")):
                value = getattr(event, attribute, None)
                if value:
                    self._session_ref[key] = str(value)

        for event_type in (EventType.ERROR, EventType.AGENT_ERROR):
            self._client.subscribe(event_type, on_error)
        self._client.subscribe(EventType.SESSION_INFO, on_session_info)
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
        # Recorded on the CONTEXT MANAGER, not only returned: a timeout
        # cancels the body, and the caller still needs the sid to find the
        # session's persisted tracker snapshot (see _tracker_usage).
        self.session_id = sid
        return Session(self._client, sid)

    async def __aexit__(self, *exc: Any) -> bool:
        if self._client is not None:
            await self._client.disconnect()
        return False


#: Dimensions the persisted tracker snapshot reports, mapped onto the
#: accumulator's vocabulary.  Only unambiguous pairs are carried: the
#: snapshot's ``tokens`` is a single total with no prompt/output split, so
#: it cannot fill those two without inventing a division.
_TRACKER_TO_USAGE = {"usd": "cost_usd", "tokens": "spend_total_tokens"}


def _tracker_usage(workspace: Workspace,
                   session_id: Optional[str]) -> Dict[str, float]:
    """Read the session's own BudgetTracker snapshot from its workspace.

    ``JaatoSession.get_budget_usage`` is the authoritative figure — the
    tracker accumulates it PER RESPONSE — and its docstring says why an
    event stream is not: events are "both duplicable (turn.progress
    re-emits) and droppable (a cancelled turn's TurnCompletedEvent)".  A
    cut arm is exactly the droppable case.

    That method is not exposed to SDK clients, but it does not need to be:
    the daemon persists the snapshot into the session record inside the
    arm's own workspace, which ``--keep-workspaces`` preserves anyway.
    Reading the file needs no new wire surface and no live session, so it
    works after the coroutine has already been cancelled.

    Measured 2026-08-30: an arm reporting ``cost=$0.0000, turns=0`` had
    ``budget_usage: {"usd": 0.4458212, "tokens": 3057027.0,
    "tool_calls": 36.0}`` sitting in its workspace (#723).

    Returns ``{}`` when the record is absent or unreadable — the caller
    then keeps whatever the accumulator saw.  A missing snapshot must
    never be worse than no snapshot.
    """
    if not session_id:
        return {}
    record = Path(workspace.path) / ".jaato" / "sessions" / f"{session_id}.json"
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    usage = data.get("budget_usage")
    if not isinstance(usage, dict):
        return {}
    out: Dict[str, float] = {}
    for src, dst in _TRACKER_TO_USAGE.items():
        value = usage.get(src)
        if isinstance(value, (int, float)):
            out[dst] = float(value)
    turns = usage.get("turns")
    if isinstance(turns, (int, float)):
        out["turns"] = float(turns)
    return out


def _completion_nudges(workspace: Workspace,
                       session_id: Optional[str],
                       completion_gap: Optional[str]) -> Optional[int]:
    """How many completion nudges this arm drew, or ``None`` if unknowable.

    Three of one sweep's BLOCKED arms were explained only by grepping
    ``COMPLETION_NUDGE`` out of session logs, and an arm sitting at
    ``2/2`` is one nudge from BLOCKED — so the count belongs in the
    result rather than in whatever log the operator still has.

    There is no event carrying it: the framework announces each nudge with
    ``JaatoServer._trace``, which is ``logger.debug``.  So this reads the
    session's OWN log out of the arm's workspace, the same move
    :func:`_tracker_usage` makes for the budget snapshot and for the same
    reason — the daemon writes it there (``JAATO_SESSION_LOG_DIR``,
    default ``.jaato/logs``, resolved against the workspace), so no new
    wire surface and no live session is needed.

    ``completion_gap`` is the corroborating witness and the fallback: the
    framework sets it exactly when it asked ``MAX_COMPLETION_NUDGES``
    times and gave up, so an arm carrying it is at the ceiling whether or
    not a log survives.

    Returns:
        The count, or ``None`` when it cannot be established.  ``None``
        rather than ``0`` for the case that matters: a daemon logging at
        INFO writes the session file without ever writing a nudge line,
        and reporting that as "no nudges" would be a fact this engine
        made up.  A log with no ``DEBUG`` record at all is therefore read
        as "not recorded", not as "none fired".
    """
    if completion_gap:
        return MAX_COMPLETION_NUDGES
    if not session_id:
        return None
    log_dir = Path(workspace.path) / ".jaato" / "logs"
    # The handler names files ``session_{sid}_client_{cid}.log`` and one
    # session can be written by more than one client, so this is a glob
    # over the session's files rather than a single path.
    logs = sorted(log_dir.glob(f"session_{session_id}*.log"))
    if not logs:
        return None
    nudges = 0
    debug_seen = False
    for log in logs:
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        debug_seen = debug_seen or "[DEBUG]" in text
        nudges += text.count("COMPLETION_NUDGE:")
    if nudges:
        return nudges
    return 0 if debug_seen else None


def _record_partial_usage(result: "ArmResult",
                          accumulator: "_TurnAccumulator",
                          tracker: Optional[Dict[str, float]] = None) -> None:
    """Carry whatever the arm spent onto a BLOCKED result.

    BLOCKED means "we learned nothing about the configuration", NOT "this
    was free".  An arm cut mid-turn has usually spent real money: one
    observed arm ran its full 900s ceiling across 467 billed
    `chat/completions` calls and reported `cost=$0.0000`, because usage was
    only ever copied onto the result along the success path (#723).

    That is not merely a wrong number.  The task pool's `usd` ceiling is
    evaluated against reported spend, so an arm that never completes a turn
    — an agent looping on tool calls, exactly the shape this harness runs —
    could burn without bound while the pool read zero.  Reporting what we
    have makes the ceiling enforceable in the case it exists for.

    Still a FLOOR, not the truth: usage rides on turn-completion events, so
    an arm cut inside its first turn reports what completed, which may be
    nothing.  The invariant is that reported cost never UNDERSTATES what
    the accumulator saw; closing the in-flight gap needs per-response usage
    (#723).
    """
    result.turns = accumulator.turns
    if accumulator.finish_reason and not result.finish_reason:
        result.finish_reason = accumulator.finish_reason
    usage = accumulator.snapshot()
    for key, value in (tracker or {}).items():
        if key == "turns":
            result.turns = max(result.turns, int(value))
            continue
        # Never report LESS than either source saw.  The tracker counts per
        # response and normally wins for a cut arm; the accumulator can still
        # be ahead on a dimension the snapshot does not carry, or if the
        # record was written before the final response landed.
        current = usage.get(key)
        if not isinstance(current, (int, float)) or value > current:
            usage[key] = value
    result.usage = usage


async def _run_session(spec: ArmSpec, workspace: Workspace, *,
                       socket_path: Optional[str],
                       cascade_driver_id: Optional[str] = None,
                       accumulator: Optional["_TurnAccumulator"] = None,
                       session_ref: Optional[Dict[str, Any]] = None):
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

    # Owned by the CALLER when supplied: a timeout cancels this coroutine,
    # so anything created HERE is unreachable afterwards and its usage is
    # lost with it.  The arm that most needs its spend reported is the one
    # that was cut short (#723).
    if accumulator is None:
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

    arm = _ArmSession(kwargs, handlers, session_ref=session_ref)
    async with arm as session:
        # Published as soon as it exists, for the same reason the
        # accumulator is caller-owned: a timeout cancels this body, and the
        # sid is what locates the session's persisted tracker snapshot.
        if session_ref is not None:
            session_ref["id"] = getattr(arm, "session_id", None)
        client = session.client
        # RETURNS AT THE SESSION'S TERMINUS, and everything after this line
        # -- the ledger, and every grader reading the workspace -- depends on
        # that.  It used to return at the first turn boundary, which for a
        # completion-gated profile is not the terminus: an agent that ends a
        # turn in prose is re-prompted by the daemon and keeps working.  This
        # engine graded an arm 19s before its agent's first commit and
        # recorded FAIL on a tree that compiles.  Fixed in the SDK rather than
        # worked around here, per this module's own rule about where SDK gaps
        # belong (jaato #767); the guard lives in the SDK's conformance suite,
        # where a live daemon can actually be re-prompted.
        #
        # NOT EVERY ERROR TERMINAL IS "NOTHING TO GRADE".  ``complete()``
        # raises ``AgentError`` on an error terminal, and this used to let
        # every one of them out to run_arm's blanket handler, which records
        # BLOCKED.  For ``NudgeExhausted`` that is the wrong state: the
        # agent ran, committed, and left a workspace, and only its sign-off
        # is missing -- so the arm was reported as unmeasured while its tree
        # sat on disk, and (since blocked arms leave the pass-rate
        # denominator) a genuinely failing arm silently improved the model's
        # score.  :mod:`jaato_eval.sign_off` owns which terminals qualify;
        # everything else still propagates, because a daemon that died
        # mid-turn really does leave a tree nobody can vouch for.
        try:
            payload = await session.complete(task.input.prompt)
        except Exception as exc:  # noqa: BLE001 -- sorted, not swallowed
            if not is_unsigned_terminal(getattr(exc, "error_type", None)):
                raise
            payload = None
            accumulator.note_unsigned(exc)
        try:
            await client.request_history()
        except Exception:  # noqa: BLE001 -- the same fallback as a timeout
            # A session that ended in an error may not accept the request at
            # all.  That is the timeout case one step earlier, and it takes
            # the same course: ``history`` stays None, the ledger comes back
            # UNFAITHFUL, ledger-reading graders BLOCK -- and the graders
            # that read the workspace still get their verdict, which is the
            # whole point of grading this arm.
            history_ready.set()
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
