"""Raising an incident -- the server-side half of the register (#1122).

The record, the vocabulary and the parser live in
:mod:`jaato_sdk.incidents`, because the reader of a register is
``jaato-doctor`` and the SDK cannot import ``shared``.  What lives HERE
is the one thing that needs a session: the raiser, which reads the
binding off whichever session is executing and writes the line through
its trace sink.

Everything from the SDK module is re-exported, so an in-tree caller
writes ``from shared.incidents import ...`` exactly as before and there
is one definition of each name -- the ``shared.session_context`` /
``jaato_sdk.session_env`` shape (#918).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from jaato_sdk.incidents import (  # noqa: F401 -- re-exported surface
    ARTICLE_73_WINDOWS,
    INCIDENT_KINDS,
    INCIDENT_TRACE_PREFIX,
    KIND_BUDGET_EXHAUSTED,
    KIND_CIRCUIT_OPENED,
    KIND_CONFINEMENT_REFUSED,
    KIND_DESCRIPTIONS,
    KIND_NUDGE_EXHAUSTED,
    KIND_SESSION_ERROR,
    Incident,
    clocks,
    parse_incident_trace,
    read_register,
)

__all__ = [
    "ARTICLE_73_WINDOWS", "INCIDENT_KINDS", "INCIDENT_TRACE_PREFIX",
    "KIND_BUDGET_EXHAUSTED", "KIND_CIRCUIT_OPENED",
    "KIND_CONFINEMENT_REFUSED", "KIND_DESCRIPTIONS",
    "KIND_NUDGE_EXHAUSTED", "KIND_SESSION_ERROR",
    "Incident", "clocks", "parse_incident_trace", "read_register",
    "raise_incident",
]


def raise_incident(
    kind: str,
    cause: str,
    *,
    site: str,
    session: Any = None,
    session_id: Optional[str] = None,
    emit: Any = None,
    now: Optional[float] = None,
) -> Optional[Incident]:
    """Record an incident, everywhere it belongs.  Never raises.

    The ONE writer.  Every site that detects something a person should
    look at calls this rather than formatting its own line, so the five
    formats #1122 is about become one register -- and so an AST guard can
    check that each named site still calls it.

    **It must never fail the thing it reports on.**  An incident is
    recorded ABOUT a failure -- a terminal error, a refused bootstrap --
    so a raiser that could itself raise would turn a reported failure
    into an unreported one, at exactly the moment the record matters
    most.  Everything here is wrapped.

    Args:
        kind: One of :data:`INCIDENT_KINDS`.  An unknown kind is still
            recorded, under its own name: refusing it would drop the
            report of something that DID happen because the vocabulary
            was not updated first.
        cause: One line.  Goes LAST on the trace line, so it may contain
            anything.
        site: ``file.py::function`` -- what noticed.
        session: The ``JaatoSession``, when there is one.  Read for its
            trace sink and its binding; never stored.
        session_id: Used when ``session`` cannot supply one.
        emit: A callable taking an :class:`Incident`, for the typed
            event.  Optional: the trace line is what every deployment
            gets, and the event is what a client can branch on.
        now: The instant (#996).

    Returns:
        The :class:`Incident`, or ``None`` when nothing could be
        recorded at all.
    """
    try:
        incident = _build(kind, cause, site, session, session_id, now)
    except Exception:  # noqa: BLE001 -- see the docstring
        return None

    try:
        tracer = getattr(session, "_trace", None) if session is not None else None
        if callable(tracer):
            tracer(incident.to_trace())
        else:
            _trace_without_a_session(incident)
    except Exception:  # noqa: BLE001
        pass

    if emit is not None:
        try:
            emit(incident)
        except Exception:  # noqa: BLE001
            pass
    return incident


def _build(
    kind: str,
    cause: str,
    site: str,
    session: Any,
    session_id: Optional[str],
    now: Optional[float],
) -> Incident:
    """Assemble the record, reading the binding off the session if any."""
    provider = model = tier = None
    if session is not None:
        provider = getattr(session, "_active_provider_name", None) or getattr(
            getattr(session, "_runtime", None), "provider_name", None)
        model = getattr(session, "_model_name", None)
        tier = getattr(session, "_active_tier", None)
        session_id = session_id or getattr(session, "_daemon_session_id", None)
    return Incident(
        kind=kind,
        at=time.time() if now is None else now,
        cause=(cause or "").replace("\n", " ").strip(),
        session_id=session_id,
        provider=provider,
        model=model,
        tier=tier,
        site=site,
    )


def _trace_without_a_session(incident: Incident) -> None:
    """Write the line when there is no session to write it through.

    A confinement refusal happens BEFORE any session exists, which is
    exactly when the record matters -- so the register must not be
    reachable only from inside a session.
    """
    from jaato_sdk.trace import resolve_trace_path, trace_write
    path = resolve_trace_path("JAATO_TRACE_LOG")
    trace_write("INCIDENT", incident.to_trace(), path)
