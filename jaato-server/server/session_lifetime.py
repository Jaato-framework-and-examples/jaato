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

**No attached clients at all** — not even the synthetic ``_headless`` marker.
That is a narrower condition than "the client disconnected", and it is what
keeps this from becoming the terminate-on-client-loss behaviour the framework
deliberately does not want:

* a session woken by ``session.wake`` / ``resume_session`` is attached to
  ``SessionManager._HEADLESS_CLIENT_ID`` and is **never** orphaned;
* a cascade stage likewise carries a client for the duration of its run;
* a completion-gated session that ended has been UNLOADED to disk — the sweep
  only sees loaded sessions, so the documented ``signal_completion`` →
  ``session.wake`` resume path is untouched;
* an idle orphan is normally unloaded by ``_maybe_unload_session`` before the
  grace expires anyway.

What is left, once those are excluded, is the #812 state: loaded, running, and
with nothing that will ever read the result.

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

from shared.runtime_limits import (
    DEFAULT_MAX_ORPHAN_SECONDS,
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

    Args:
        limits: The session's ``RuntimeLimits``, or ``None``.

    Returns:
        e.g. ``"max_session_seconds=unbounded max_orphan_seconds=900.0s
        (framework default)"``.
    """
    session_bound, orphan_bound = resolve_bounds(limits)
    declared_orphan = getattr(limits, "max_orphan_seconds", None)
    session_txt = (
        "unbounded" if session_bound is None else f"{session_bound:.1f}s"
    )
    orphan_txt = "unbounded" if orphan_bound is None else f"{orphan_bound:.1f}s"
    if declared_orphan is None and orphan_bound is not None:
        orphan_txt += " (framework default)"
    return (
        f"max_session_seconds={session_txt} max_orphan_seconds={orphan_txt}"
    )
