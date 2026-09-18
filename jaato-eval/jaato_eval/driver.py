"""A driver arm — ``harness.kind: driver`` (jaato #1110).

An arm was always one session: the engine opened it from
``harness.profile``, sent ``input.prompt``, waited, and graded the
workspace.  Some units of work are not one session.  A backtest cell is a
DRIVER PROCESS orchestrating about ten sessions in a fixed order
(analysts → debate → judges), with market data reaching the model as host
tools that live in the driver's own process — the driver-as-graph shape,
typed payloads and a deterministic DAG, not a supervisor session.  What
that cell wanted from this engine is every matrix mechanic it already
owns: repeats, arms across profile sets, per-arm scratch workspaces, a
cascade pool per task, resume, concurrency, the per-arm ceiling with
BLOCKED accounting, and the report.  What it could not express was the
arm.

This module is the arm.  The engine hands the driver a CONTRACT as
environment, attaches an observer to the arm's cascade id BEFORE the
process starts, runs the process in its own process group under the arm
ceiling, and reads its exit code.

THE CONTRACT (``JAATO_EVAL_CONTRACT=1``)
=========================================

==========================  =================================================
variable                    value
==========================  =================================================
``JAATO_EVAL``              ``1`` — a graded run, as a script grader sees
``JAATO_EVAL_CONTRACT``     ``1`` — the version of this table
``JAATO_EVAL_WORKSPACE``    the materialised fixture, carrying the ``.env``
                            the engine already writes (``JAATO_PROFILE_SET``)
``JAATO_EVAL_CONFIG_ROOT``  the task's read-only ``.jaato/``
``JAATO_EVAL_SOCKET``       the daemon the arm runs on — the same rule as
                            ``GraderContext.socket_path``, so ABSENT when the
                            sweep uses the SDK's default socket
``JAATO_EVAL_CASCADE_ID``   the arm's cid.  EVERY session the driver opens
                            must be stamped with it — that is what makes the
                            pool, the observer and the per-stage records work
``JAATO_EVAL_PARAM_<KEY>``  one per ``input.params`` entry, and
``JAATO_EVAL_PARAMS``       the whole mapping as JSON — the encoding a
                            ``script`` grader already receives
==========================  =================================================

The contract is versioned by name so a driver can refuse a table it does
not understand rather than guess at one.

THE EXIT-CODE VOCABULARY
========================

Mirrors the one rule in :mod:`jaato_eval.sign_off`:

* ``0`` — ran to its end; the tree is gradeable.
* ``75`` (``EX_TEMPFAIL``) — an environment fault: daemon unreachable,
  fixture unusable.  BLOCKED — "we learned nothing".
* anything else — ran and stopped short: an UNSIGNED arm.  Script graders
  still run; payload-reading graders BLOCK, naming the driver.  The
  driver's stderr tail becomes ``termination_detail``.

A driver killed at the arm ceiling has no exit code of its own and is
BLOCKED exactly as a session arm is.

WHAT THE OBSERVER MEASURES, AND HOW IT KNOWS WHOSE IT IS
=========================================================

Measured on a live daemon: a cascade observer attached to the cid of a
driver run sees every session the driver opens — nine ``AgentCreatedEvent``,
nine ``TurnCompletedEvent`` with per-turn usage, the terminals — from its
own connection.  So the engine accounts a driver arm the way it accounts a
session arm, one :class:`~jaato_eval.accounting._TurnAccumulator` PER
SESSION ID, summed at the end.

The cid is the task POOL's cid when the task declares a ``budget:`` block
— that is what lets the pool apply to the driver's sessions with no engine
work, since the daemon applies a cid's pool to every session stamped with
it — and one minted per arm otherwise.  The pool case has a consequence
the single-run measurement did not show: two arms of one task running
concurrently share the cid, so the observer sees the sibling's sessions
too.  Attribution therefore does not come from the event stream.  The
daemon persists every session's record into the session's OWN workspace
at creation (``<workspace>/.jaato/sessions/<sid>.json`` — the file
:func:`jaato_eval.runner._tracker_usage` already reads), and the contract
puts the driver's sessions in THIS arm's workspace, so the set of records
there is the set of sessions that are this arm's.  Sessions the observer
saw with no record here are a sibling's and are dropped; a record here
the observer never saw (opened before the registration was applied) is
still this arm's, with its usage read from the record.

Two things the observer cannot see, said plainly: ``SessionInfoEvent`` is
answered to the creating client and is not routed to cascade observers,
so a driver arm's ``model`` / ``provider`` come from the first session's
persisted record rather than from the wire; and ``cascade.register`` is
fire-and-forget, so the observer is registered before the process is
spawned and relies on the daemon applying the frame before a process that
has yet to start can connect.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from .accounting import _SUMMED_USAGE, _TurnAccumulator
from .params import param_env
from .pool import _slug

#: The version of the contract table in the module docstring.  Bumped when
#: a variable changes meaning; a new variable is additive and is not.
CONTRACT_VERSION = "1"

#: ``sysexits.h``'s temporary-failure code: the driver's way of saying "the
#: environment, not the run" — daemon unreachable, fixture unusable.
EX_TEMPFAIL = 75

#: How long a driver killed at the ceiling gets between SIGTERM and
#: SIGKILL.  SIGTERM first because a driver has sessions open on the
#: daemon and a graceful exit is its chance to end them; a session left
#: behind keeps spending until the daemon's orphan sweep reaches it
#: (jaato #812, 900s by default), so the engine also stops the sessions it
#: can name — see :meth:`CascadeObserver.stop_sessions`.
TERM_GRACE_SECONDS = 10.0

#: Lines of stdout / stderr kept as evidence.  The driver's output is not
#: the deliverable — the workspace is — so a bounded tail is what a
#: verdict quotes, and an unbounded capture of a chatty driver is a memory
#: bill the sweep should not pay.
TAIL_LINES = 40

#: Read chunk for the drain.  Chunked rather than ``readline`` so a driver
#: that prints a very long line cannot raise ``LimitOverrunError`` out of
#: the reader and lose the tail.
_READ_CHUNK = 65536

#: How long the engine waits for the daemon to confirm a best-effort
#: ``session.stop`` of a session the killed driver left behind.
_STOP_TIMEOUT_SECONDS = 10.0


def driver_environment(*, workspace: Path, config_root: Path,
                       cascade_id: str, params: Mapping[str, Any],
                       socket_path: Optional[str]) -> Dict[str, str]:
    """The contract, as the environment ADDED to the driver's.

    Returns only the contract variables; the caller overlays them on
    ``os.environ`` at spawn.  Kept pure so the table in the module
    docstring is testable as a table.

    Raises:
        ValueError: when two ``params`` keys collide on one variable name.
            The manifest parser refuses that before any arm runs, so this
            is a guard on the contract rather than a path a task reaches.
    """
    env, collision = param_env(params)
    if collision:
        raise ValueError(f"input.params: {collision}")
    contract = {
        "JAATO_EVAL": "1",
        "JAATO_EVAL_CONTRACT": CONTRACT_VERSION,
        "JAATO_EVAL_WORKSPACE": str(workspace),
        "JAATO_EVAL_CONFIG_ROOT": str(config_root),
        "JAATO_EVAL_CASCADE_ID": cascade_id,
        **env,
    }
    if socket_path:
        contract["JAATO_EVAL_SOCKET"] = str(socket_path)
    return contract


def arm_cascade_id(arm_id: str, pool_cid: Optional[str]) -> str:
    """The cid handed to the driver.

    The task pool's cid when the task declared a pool — the pool binds to
    that cid and to nothing else — and a per-arm id otherwise, because a
    driver arm ALWAYS needs one: it is what the observer attaches to.  A
    session arm with no pool runs un-cid'd; a driver arm cannot.
    """
    if pool_cid:
        return pool_cid
    return f"jaato-eval-{_slug(arm_id)}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# The observer
# ---------------------------------------------------------------------------

class CascadeObserver:
    """One IPC connection watching every session under the arm's cid.

    Keeps a :class:`_TurnAccumulator` per session id, in first-seen order,
    and nothing else: whose sessions they are is decided afterwards by
    :func:`attributed_sessions`, from the workspace, because the event
    stream alone cannot say (see the module docstring).

    Lifecycle: ``async with`` connects, subscribes, registers — in that
    order, and before the caller starts the driver, so the first session's
    first event is not missed.  ``__aexit__`` disconnects, which is also
    what unregisters the observer daemon-side.

    Attributes:
        sessions: ``session_id`` -> the accumulator for that session.
        order: session ids in the order the observer first saw them —
            the creation order, since ``AgentCreatedEvent`` is the first
            routed event of a session's life.
        unattributed: routed events that carried NO session id.  Counted
            rather than dropped silently: the daemon stamps every routed
            event (protocol 1.2+), so a nonzero count is a daemon this
            engine cannot account against, and worth seeing.
        refusals: pool refusals seen under the cid (``CascadeExhausted``),
            so a driver whose stage was refused a spawn has that fact in
            its termination detail even if its own stderr does not say.
    """

    def __init__(self, cascade_id: str, *,
                 socket_path: Optional[str] = None) -> None:
        self.cascade_id = cascade_id
        self._socket_path = socket_path
        self.sessions: Dict[str, _TurnAccumulator] = {}
        self.order: List[str] = []
        self.unattributed = 0
        self.refusals: List[str] = []
        self._client: Any = None

    # -- handlers ---------------------------------------------------------

    def _bucket(self, event: Any) -> Optional[_TurnAccumulator]:
        sid = str(getattr(event, "session_id", "") or "")
        if not sid:
            self.unattributed += 1
            return None
        accumulator = self.sessions.get(sid)
        if accumulator is None:
            accumulator = self.sessions[sid] = _TurnAccumulator()
            self.order.append(sid)
        return accumulator

    def on_created(self, event: Any) -> None:
        """Register the session; the first routed event of its life."""
        self._bucket(event)

    def on_turn(self, event: Any) -> None:
        accumulator = self._bucket(event)
        if accumulator is not None:
            accumulator.on_turn(event)

    def on_terminated(self, event: Any) -> None:
        accumulator = self._bucket(event)
        if accumulator is not None:
            accumulator.on_terminated(event)

    def on_error(self, event: Any) -> None:
        error_type = str(getattr(event, "error_type", "") or "")
        if "CascadeExhausted" in error_type:
            self.refusals.append(str(getattr(event, "error", "")) or error_type)

    # -- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> "CascadeObserver":
        from jaato_sdk.client.ipc import IPCClient
        from jaato_sdk.events import (AgentCreatedEvent, ClientType,
                                      ErrorEvent, EventType,
                                      SessionTerminatedEvent,
                                      TurnCompletedEvent)

        kwargs: Dict[str, Any] = {"client_type": ClientType.API}
        if self._socket_path:
            kwargs["socket_path"] = self._socket_path
        self._client = IPCClient(**kwargs)
        if not await self._client.connect(timeout=120):
            # ``__aexit__`` never runs when ``__aenter__`` raises, so the
            # half-open client is closed here or not at all.
            await self._client.disconnect()
            self._client = None
            raise ConnectionError(
                "could not connect to / autostart the jaato daemon — "
                "run `python -m jaato_sdk.doctor`")
        self._client.subscribe(EventType.AGENT_CREATED, self.on_created)
        self._client.subscribe(EventType.TURN_COMPLETED, self.on_turn)
        self._client.subscribe(EventType.SESSION_TERMINATED, self.on_terminated)
        self._client.subscribe(EventType.ERROR, self.on_error)
        # Event CLASSES, not wire values: the cascade filter matches
        # ``type(event).__name__``, and a name that can never match makes a
        # deaf observer (jaato #821).
        await self._client.cascade_register(
            self.cascade_id, role="observer",
            event_types=[AgentCreatedEvent, TurnCompletedEvent,
                         SessionTerminatedEvent, ErrorEvent])
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None
        return False

    # -- after the driver -------------------------------------------------

    def unfinished(self, session_ids: Sequence[str]) -> List[str]:
        """Those of ``session_ids`` the observer saw no terminal for."""
        return [sid for sid in session_ids
                if sid not in self.sessions
                or not self.sessions[sid].termination_reason]

    async def stop_sessions(self, session_ids: Sequence[str]) -> List[str]:
        """Best-effort ``session.stop`` for sessions a killed driver left.

        Returns the ids for which the stop was SENT.  Never raises: a
        daemon predating the verb (protocol 1.7) refuses it in the SDK,
        and a stop that cannot be sent leaves the orphan sweep as the
        bound — which is what would have happened anyway.  Nothing here
        is a kill: ``session.stop`` trips the session's cancel token, the
        same path a budget ``abort`` rung takes.
        """
        stop = getattr(self._client, "stop_session", None)
        if stop is None:
            return []
        stopped: List[str] = []
        for sid in session_ids:
            try:
                await asyncio.wait_for(stop(sid), timeout=_STOP_TIMEOUT_SECONDS)
            except Exception:  # noqa: BLE001 — best-effort, see docstring
                continue
            stopped.append(sid)
        return stopped

    def merged(self, session_ids: Sequence[str]) -> _TurnAccumulator:
        """One accumulator summing the named sessions.

        Usage and turns are summed; the provider-side latches
        (``upstream_provider``, ``native_finish_reason``) keep the last
        populated value, as they do within one session.  Termination
        facts are deliberately NOT merged — a driver arm's termination is
        its exit code, and the per-session terminals are reported through
        :meth:`abnormal_terminals` instead.
        """
        total = _TurnAccumulator()
        for sid in session_ids:
            part = self.sessions.get(sid)
            if part is None:
                continue
            total.turns += part.turns
            for key in _SUMMED_USAGE:
                total.usage[key] += part.usage[key]
            if part.cost_usd is not None:
                total.cost_usd = (total.cost_usd or 0.0) + part.cost_usd
            if part.upstream_provider:
                total.upstream_provider = part.upstream_provider
            if part.native_finish_reason:
                total.native_finish_reason = part.native_finish_reason
        return total

    def abnormal_terminals(self, session_ids: Sequence[str]) -> List[str]:
        """``sid: reason`` for each named session that named a stop.

        ``natural`` / ``client_request`` / ``stopped`` are ordinary
        wind-downs; only ``budget_exhausted`` and ``error`` name one.
        """
        out: List[str] = []
        for sid in session_ids:
            part = self.sessions.get(sid)
            if part is None or part.termination_reason not in ("budget_exhausted", "error"):
                continue
            label = part.termination_reason
            if part.termination_error_type:
                label += f"({part.termination_error_type})"
            out.append(f"{sid}: {label}")
        return out


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------

@dataclass
class DriverOutcome:
    """How the driver process ended.

    Attributes:
        exit_code: The process's exit status, or ``None`` when the engine
            killed it at the arm ceiling — a code the engine caused is not
            a code the driver chose, and must not be read through the
            exit-code vocabulary.
        timed_out: The ceiling fired.
        stderr_tail: The last :data:`TAIL_LINES` lines of stderr.
        stdout_tail: Likewise stdout.  Kept because a driver that prints
            its progress there and dies with an empty stderr would
            otherwise leave no evidence at all.
        duration_seconds: Wall clock from spawn to exit (or kill).
    """

    exit_code: Optional[int]
    timed_out: bool
    stderr_tail: List[str]
    stdout_tail: List[str]
    duration_seconds: float

    @property
    def gradeable(self) -> bool:
        return self.exit_code == 0

    @property
    def environment_fault(self) -> bool:
        return self.exit_code == EX_TEMPFAIL

    @property
    def evidence(self) -> str:
        """The tail a verdict quotes: stderr, else stdout, else nothing."""
        lines = self.stderr_tail or self.stdout_tail
        return "\n".join(lines).strip()


async def run_driver(command: str, *, cwd: Path, env: Mapping[str, str],
                     timeout: float) -> DriverOutcome:
    """Run ``command`` through the shell in ``cwd`` under ``timeout``.

    The process gets its OWN PROCESS GROUP (``start_new_session``), so the
    kill at the ceiling reaches the driver's children — the stage
    subprocesses a driver-as-graph spawns — and not only the shell.
    ``timeout <= 0`` disables the ceiling, as it does for a session arm.

    ``env`` is the contract; it is overlaid on this process's environment
    rather than replacing it, since the driver needs a ``PATH`` and its
    interpreter's own variables.
    """
    started = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        command, cwd=str(cwd), env={**os.environ, **env},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True)
    out_tail: Deque[str] = deque(maxlen=TAIL_LINES)
    err_tail: Deque[str] = deque(maxlen=TAIL_LINES)
    drains = asyncio.gather(_drain(proc.stdout, out_tail),
                            _drain(proc.stderr, err_tail))
    timed_out = False
    try:
        if timeout > 0:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        else:
            await proc.wait()
    except asyncio.TimeoutError:
        timed_out = True
        await _kill_process_group(proc)
    # The pipes close when the last holder exits; a grandchild that
    # survived the group kill by re-parenting could hold them open, so the
    # drain is bounded rather than awaited outright.
    try:
        await asyncio.wait_for(drains, timeout=5.0)
    except asyncio.TimeoutError:
        drains.cancel()
    return DriverOutcome(
        exit_code=None if timed_out else proc.returncode,
        timed_out=timed_out,
        stderr_tail=list(err_tail), stdout_tail=list(out_tail),
        duration_seconds=time.monotonic() - started)


async def _drain(stream: Any, tail: Deque[str]) -> None:
    """Keep the last lines of ``stream`` in ``tail`` until it closes."""
    if stream is None:
        return
    pending = ""
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            break
        pending += chunk.decode("utf-8", errors="replace")
        lines = pending.split("\n")
        pending = lines.pop()
        tail.extend(line.rstrip() for line in lines)
    if pending.strip():
        tail.append(pending.rstrip())


async def _kill_process_group(proc: Any) -> None:
    """SIGTERM the group, wait :data:`TERM_GRACE_SECONDS`, then SIGKILL."""
    _signal_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=TERM_GRACE_SECONDS)
        return
    except asyncio.TimeoutError:
        pass
    _signal_group(proc, signal.SIGKILL)
    await proc.wait()


def _signal_group(proc: Any, sig: int) -> None:
    """Signal the driver's process group, falling back to the process."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, sig)
        else:  # pragma: no cover — no process groups on this platform
            proc.send_signal(sig)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            return


# ---------------------------------------------------------------------------
# Attribution — whose sessions were these
# ---------------------------------------------------------------------------

def workspace_session_records(workspace: Path) -> Dict[str, Dict[str, Any]]:
    """``sid`` -> the record the daemon persisted into this workspace.

    The daemon writes ``<workspace>/.jaato/sessions/<sid>.json`` when a
    session is CREATED and again as it runs, so every session the driver
    opened in this arm's workspace has a record here from its first
    moment.  A file that will not parse is skipped, not fatal: a record
    mid-write is a normal state on a live daemon.
    """
    directory = Path(workspace) / ".jaato" / "sessions"
    records: Dict[str, Dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            records[path.stem] = data
    return records


def attributed_sessions(observer: CascadeObserver,
                        records: Mapping[str, Any]) -> List[str]:
    """Which sessions are THIS arm's, in creation order.

    Those the observer saw that have a record in the arm's workspace,
    in the order seen; then any record the observer never saw (opened
    before its registration was applied), in id order — session ids are
    timestamp-shaped, so that is creation order too.

    With NO records at all, every session the observer saw is taken.
    That is the one case where a concurrent sibling arm under a shared
    pool cid could be mis-attributed, and it needs a daemon that
    persisted nothing into the workspace — which would also have broken
    every tracker read this engine makes.  Reporting nothing there would
    understate an arm that demonstrably ran sessions; the trade is
    stated rather than hidden.
    """
    if not records:
        return list(observer.order)
    seen = [sid for sid in observer.order if sid in records]
    unseen = sorted(sid for sid in records if sid not in observer.sessions)
    return seen + unseen


def record_binding(record: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """``(model, provider)`` as a session record persisted them.

    The record freezes the resolved profile (``profile_snapshot``, record
    2.8+), which is the recipe the daemon bound.  Older keys are honoured
    first where a record carries them.  ``None`` where the record says
    nothing — never the profile-set name, which is a directory someone
    chose.
    """
    snapshot = record.get("profile_snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    model = record.get("model_name") or snapshot.get("model")
    provider = record.get("model_provider") or snapshot.get("provider")
    return (str(model) if model else None, str(provider) if provider else None)
