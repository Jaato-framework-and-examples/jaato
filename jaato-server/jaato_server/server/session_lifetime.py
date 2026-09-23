"""The daemon-side wall-clock bound on a loaded session (#812).

Every ceiling a session had was held by something the session could outlive:

* ``jaato_eval``'s ``--arm-timeout`` is enforced **client-side**, in the runner
  loop — the process that died is the one that was supposed to stop the arm;
* the task pool's ``seconds`` is reconciled **when a session ends**, so a
  session that never ends never consumes it;
* ``budget_control`` works, and is the only thing that did — the reported
  session was stopped by an ``abort`` rung at $2.52.  A profile declaring no
  ``budget_control`` had nothing at all.

The common shape is that the holder of the bound was not the thing that
outlives the failure.  The daemon is, so the bound lives here, and it does not
wait for the session to end: the sweep runs on a timer and judges sessions
while they are running.

Two bounds, deliberately different in kind
------------------------------------------

============================  ==========================  ======================
Bound                         Measures                    Default
============================  ==========================  ======================
``max_session_seconds``       total loaded wall-clock      unbounded (opt-in)
``max_orphan_seconds``        continuous time with NO      900 s
                              consumer
============================  ==========================  ======================

``max_session_seconds`` is opt-in because an interactive session left open over
a lunch break is not a defect, and a default would kill it.

``max_orphan_seconds`` carries a default because the session it exists for is
precisely the one whose author declared nothing — a bound you must remember to
write would have left #812's session running exactly as long.

What "orphaned" means, and why it is safe to default
----------------------------------------------------

A session is judged an orphan only when **both** hold:

1. it is LOADED and has no attached clients at all, and
2. the sweep has previously OBSERVED it carrying at least one client.

The second clause is the load-bearing one and was added after the first,
weaker predicate turned out to be wrong.  "Has no client" is a broader state
than "its client went away", and the difference is not academic:
``_load_session_impl`` uses its ``client_id`` argument for config, env and
progress events and **never attaches it**, so a session revived through
``wake_session`` → ``resume_session`` has an empty ``attached_clients`` by
construction.  ``wake_session`` knows this and branches on it — the log line
reads *"revived cold, no client — DEFERRED"* — so under the weaker predicate a
cold revive driving a long turn would have been cancelled by a bound written
for an entirely different situation.

Requiring an observed attachment makes the bound depend on something the sweep
MEASURES rather than on an invariant maintained at call sites it cannot see.
That is deliberate: an earlier draft of this module asserted "every path that
drives a session attaches a client id", which is simply **false** in this
tree.  A guard policing that claim would have failed on ``main``.

The clause also fails safe.  A session the sweep never saw attached is never
stopped by the orphan bound; an explicit ``max_session_seconds`` still applies
to it, because that one is an operator's own ceiling rather than an inference
about who is watching.

What survives both clauses is the #812 state: loaded, running, with a client
that existed and is now gone, and nothing that will ever read the result.  The
other detached shapes are excluded structurally:

* a completion-gated session that ended has been UNLOADED — the sweep only
  walks ``_sessions``, so the documented ``signal_completion`` →
  ``session.wake`` resume path cannot be reached at all;
* a cold wake revive fails clause 2;
* a cascade stage and an attached interactive session fail clause 1;
* an idle orphan is normally unloaded by ``_maybe_unload_session`` before the
  grace expires anyway.

A third field, and a different VERB (#1106)
-------------------------------------------

``unload_grace_seconds`` lives in the same block and is read here, and it is
not a third bound.  The two above decide whether to **stop** a session that
has run too long; it decides how long to wait before **unloading** one that is
merely unwatched.

The distinction is the whole reason it is a separate field rather than a
smaller ``max_orphan_seconds``.  Stopping is a verdict about work nobody will
read; unloading is a cache eviction.  Before #1106 the unload path had no time
dimension at all: a WebSocket close reached ``SessionManager.detach_client``
and the teardown ran synchronously, so a tab reload cost a full save, a
plugin teardown, a ``server.shutdown`` and a pool-slot return, and the browser
came back to a session id the daemon no longer held in memory.

The two compose by ordering rather than by arithmetic.  The grace default
(60 s) sits far below the orphan default (900 s), so an unwatched session is
normally UNLOADED long before the watchdog would stop it, and the watchdog
stays the outer bound for the case the grace cannot reach — a session whose
model thread is still running, which ``_maybe_unload_session`` declines to
unload at all.  The sentence in the list above about an idle orphan being
"normally unloaded before the grace expires anyway" describes a race; #1106
makes it deliberate.

``0`` is the explicit opt-out and restores the pre-#1106 behaviour.  Unlike
the two bounds, ``0`` here is the most restrictive value, not the least:
:func:`resolve_unload_grace` therefore reads it as "no grace" rather than as
"unbounded", and the inheritance rule (``shared/plugins/subagent/config.py``)
takes a plain ``min()`` for it.

Composition with ``budget_control``
-----------------------------------

This grows no second cancellation mechanism.  A verdict is carried out by the
same ``server.stop()`` the ``budget_control`` ``abort`` rung reaches through
``request_stop`` — the cancel token the session already honours mid-turn — so a
bound crossed here behaves exactly like a ceiling crossed there, and a session
that declares both is stopped by whichever fires first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from jaato_server.shared.runtime_limits import (
    DEFAULT_MAX_ORPHAN_SECONDS,
    DEFAULT_UNLOAD_GRACE_SECONDS,
    UNBOUNDED_SECONDS,
)

#: How often the daemon's watchdog thread evaluates every loaded session.
#: Far below the smallest sensible bound, so the overshoot a coarse tick adds
#: is noise against a ceiling measured in minutes; cheap enough to be
#: irrelevant, since a sweep is a walk over the loaded-session dict.
DEFAULT_SWEEP_INTERVAL_SECONDS = 15.0

#: ``verdict.reason`` values.  Strings rather than an enum because they are
#: rendered into a ``SessionTerminatedEvent.reason``, a log line and an
#: operator listing, and each of those wants the literal.
REASON_MAX_SESSION_SECONDS = "max_session_seconds"
REASON_MAX_ORPHAN_SECONDS = "max_orphan_seconds"


@dataclass(frozen=True)
class SessionLifetimeObservation:
    """One loaded session as the sweep sees it, at one instant.

    A plain value rather than a ``Session`` so the policy can be evaluated
    without a daemon: :func:`evaluate` takes these and returns verdicts, and
    everything that touches locks, threads or servers stays in
    ``SessionManager``.

    Attributes:
        session_id: The session this describes.
        loaded_at: Monotonic timestamp at which the session became loaded.
            Monotonic, not wall-clock: a bound must not be shortened or
            lengthened by an NTP step or a DST change.
        orphaned_since: Monotonic timestamp at which the session last became
            orphaned (no attached clients), or ``None`` while it has one.
            Reset to ``None`` on every re-attach, so the grace measures
            CONTINUOUS orphanhood — a client that reconnects has renewed the
            session's claim on being wanted.
        limits: The session's resolved ``runtime_limits``, or ``None``.
    """

    session_id: str
    loaded_at: float
    orphaned_since: Optional[float]
    limits: Any = None


@dataclass(frozen=True)
class LifetimeVerdict:
    """A session the sweep has decided to stop, and why.

    Attributes:
        session_id: The session to stop.
        reason: :data:`REASON_MAX_SESSION_SECONDS` or
            :data:`REASON_MAX_ORPHAN_SECONDS`.  Becomes the
            ``SessionTerminatedEvent.reason``, so a client can distinguish a
            wall-clock stop from a user cancel or a budget abort.
        elapsed_seconds: How long the measured clock had run.
        limit_seconds: The bound it crossed.
    """

    session_id: str
    reason: str
    elapsed_seconds: float
    limit_seconds: float

    def describe(self) -> str:
        """One line for the log and the operator, naming bound and overshoot.

        Returns:
            e.g. ``"session 20260903_084517 exceeded max_orphan_seconds:
            orphaned for 931.4s > 900.0s"``.
        """
        measured = (
            "orphaned for" if self.reason == REASON_MAX_ORPHAN_SECONDS
            else "loaded for"
        )
        return (
            f"session {self.session_id} exceeded {self.reason}: "
            f"{measured} {self.elapsed_seconds:.1f}s > "
            f"{self.limit_seconds:.1f}s"
        )


def resolve_bounds(limits: Any) -> Tuple[Optional[float], Optional[float]]:
    """The effective ``(max_session_seconds, max_orphan_seconds)`` for a session.

    Three states per field, and the difference between two of them is the
    whole reason this function exists rather than a pair of attribute reads:

    * **declared and positive** — that value;
    * **declared as 0** — explicitly unbounded, so ``None``.  This is the
      operator's escape hatch and it must outrank the framework default;
    * **not declared** — the framework default, which is ``None`` for
      ``max_session_seconds`` and :data:`DEFAULT_MAX_ORPHAN_SECONDS` for
      ``max_orphan_seconds``.

    A ``limits`` object that is ``None``, or that predates these fields (an
    older session snapshot revived by a newer daemon), reads as "not declared"
    and takes the defaults — never raises.

    Args:
        limits: The session's ``RuntimeLimits``, or ``None``.

    Returns:
        ``(session_bound, orphan_bound)``; ``None`` in either slot means
        unbounded.
    """
    session_raw = getattr(limits, "max_session_seconds", None)
    orphan_raw = getattr(limits, "max_orphan_seconds", None)

    session_bound: Optional[float] = None
    if isinstance(session_raw, (int, float)) and session_raw != UNBOUNDED_SECONDS:
        session_bound = float(session_raw)

    if orphan_raw is None:
        orphan_bound: Optional[float] = DEFAULT_MAX_ORPHAN_SECONDS
    elif isinstance(orphan_raw, (int, float)) and orphan_raw != UNBOUNDED_SECONDS:
        orphan_bound = float(orphan_raw)
    else:
        orphan_bound = None

    return session_bound, orphan_bound


def resolve_unload_grace(limits: Any) -> float:
    """The effective ``unload_grace_seconds`` for a session (#1106).

    Three states, as with :func:`resolve_bounds`, but the third resolves to a
    number rather than to ``None`` — there is no "unbounded" grace, only a
    zero one:

    * **declared and positive** — that value;
    * **declared as 0** — no grace, the pre-#1106 behaviour, and the
      operator's explicit opt-out;
    * **not declared** — :data:`~shared.runtime_limits.DEFAULT_UNLOAD_GRACE_SECONDS`.

    A ``limits`` object that is ``None``, or that predates the field (an older
    session snapshot revived by a newer daemon, or a stand-in server in a
    test), reads as "not declared" and takes the default — never raises.

    Args:
        limits: The session's ``RuntimeLimits``, or ``None``.

    Returns:
        Seconds; ``0.0`` means unload immediately.
    """
    raw = getattr(limits, "unload_grace_seconds", None)
    if raw is None:
        return DEFAULT_UNLOAD_GRACE_SECONDS
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return DEFAULT_UNLOAD_GRACE_SECONDS
    if raw == UNBOUNDED_SECONDS:
        return 0.0
    return float(raw)


def unload_grace_remaining(
    clientless_since: Optional[float], now: float, limits: Any,
) -> float:
    """Seconds of grace left before a clientless session may be unloaded.

    A pure function of three values so the whole decision can be exercised
    with no daemon, no thread and no sleeping — ``now`` is a parameter for the
    reason #996 and #713 made it one elsewhere: a test states the instant it
    means instead of betting on a clock.

    ``clientless_since is None`` is deliberately read as "it just became
    clientless, so the full grace remains" rather than as "no clock, unload
    now".  The clock is DERIVED — by the sweep, and by the unload gate itself
    at the moment it observes an empty ``attached_clients`` — never stamped at
    the ten-odd sites that mutate that set (#735, and this module's own
    docstring).  A future call site that learns to unload and forgets
    everything else therefore still defers, and the sweep picks it up; the
    opposite default would let that call site silently disarm the grace.

    Args:
        clientless_since: Monotonic instant the session last became
            clientless, or ``None`` when no observation has recorded one.
        now: Monotonic now.
        limits: The session's ``RuntimeLimits``, or ``None``.

    Returns:
        ``0.0`` when the grace has elapsed (or is disabled), else how much is
        left.  Never negative.
    """
    grace = resolve_unload_grace(limits)
    if grace <= 0:
        return 0.0
    if clientless_since is None:
        return grace
    return max(0.0, grace - (now - clientless_since))


def evaluate_one(
    observation: SessionLifetimeObservation, now: float,
) -> Optional[LifetimeVerdict]:
    """Judge one session against its bounds.

    The total-lifetime bound is checked FIRST: when a session crosses both at
    the same sweep, "it ran too long" is the more informative verdict than "and
    also nobody was watching", and the orphan bound is the fallback for
    sessions that declared no total.

    Args:
        observation: The session's clocks and limits.
        now: Monotonic now, taken once per sweep so every session in one pass
            is judged against the same instant.

    Returns:
        A verdict, or ``None`` when the session is within both bounds.
    """
    session_bound, orphan_bound = resolve_bounds(observation.limits)

    if session_bound is not None:
        elapsed = now - observation.loaded_at
        if elapsed > session_bound:
            return LifetimeVerdict(
                session_id=observation.session_id,
                reason=REASON_MAX_SESSION_SECONDS,
                elapsed_seconds=elapsed,
                limit_seconds=session_bound,
            )

    if orphan_bound is not None and observation.orphaned_since is not None:
        orphaned = now - observation.orphaned_since
        if orphaned > orphan_bound:
            return LifetimeVerdict(
                session_id=observation.session_id,
                reason=REASON_MAX_ORPHAN_SECONDS,
                elapsed_seconds=orphaned,
                limit_seconds=orphan_bound,
            )

    return None


def evaluate(
    observations: Iterable[SessionLifetimeObservation], now: float,
) -> List[LifetimeVerdict]:
    """Judge every observed session, returning only those to stop.

    Args:
        observations: One per loaded session.
        now: Monotonic now, shared across the pass.

    Returns:
        Verdicts in observation order; empty when nothing crossed a bound.
    """
    verdicts = []
    for obs in observations:
        verdict = evaluate_one(obs, now)
        if verdict is not None:
            verdicts.append(verdict)
    return verdicts


def describe_armed_bounds(limits: Any) -> str:
    """Render the effective bounds for the arming log line.

    #735 is the cautionary tale this exists for: ``tool_timeout_seconds`` was
    documented, parsed, validated and delivered to nothing, and a cap that
    silently does not apply is worse than no cap.  So the daemon logs what it
    armed, with the effective numbers rather than the declared ones — the two
    differ whenever a field was omitted or set to 0.

    All three daemon-layer fields are named, including ``unload_grace_seconds``
    (#1106) — a grace that silently does not apply is the same defect as a cap
    that silently does not apply, and it is harder to notice, because its
    failure mode is the behaviour that was there before.

    Args:
        limits: The session's ``RuntimeLimits``, or ``None``.

    Returns:
        e.g. ``"max_session_seconds=unbounded max_orphan_seconds=900.0s
        (framework default) unload_grace_seconds=60.0s (framework default)"``.
    """
    session_bound, orphan_bound = resolve_bounds(limits)
    declared_orphan = getattr(limits, "max_orphan_seconds", None)
    session_txt = (
        "unbounded" if session_bound is None else f"{session_bound:.1f}s"
    )
    orphan_txt = "unbounded" if orphan_bound is None else f"{orphan_bound:.1f}s"
    if declared_orphan is None and orphan_bound is not None:
        orphan_txt += " (framework default)"
    grace = resolve_unload_grace(limits)
    grace_txt = "0.0s (no grace)" if grace <= 0 else f"{grace:.1f}s"
    if getattr(limits, "unload_grace_seconds", None) is None:
        grace_txt += " (framework default)"
    return (
        f"max_session_seconds={session_txt} max_orphan_seconds={orphan_txt} "
        f"unload_grace_seconds={grace_txt}"
    )
