"""Which sessions are FINISHED, and when a finished session stops being one.

A session ends in several ways and, before this module, none of them left a
mark on its record.  The web client's End deleted the record outright, so a
session a person had chosen to end could only be kept as a SLEEPING session,
indistinguishable from one they had merely walked away from.

A session is finished when a ``SessionTerminatedEvent`` names one of
:data:`FINISHED_REASONS`:

==================== =========================================================
``client_request``   the person ended it (``session.end``) while it was idle
``stopped``          the person ended it (``session.end``) mid-turn
``natural``          the agent completed its task (``signal_completion``)
``budget_exhausted`` its ``budget_control`` ceiling stopped it
==================== =========================================================

Deliberately NOT finished:

- ``error`` -- a session that failed is not done, and burying it under
  Finished hides the one a person most needs to look at again.
- ``cascade_cancelled`` and the operator / wall-clock stops
  (``operator_request``, ``max_session_seconds``, ``max_orphan_seconds``) --
  nobody who owns the session chose to end it, and a cascade stage is not a
  session a person opens from a picker.

**A finished session runs again the moment a turn starts in it.**  The mark
is cleared by the main agent going ``active``, never by an attach: opening a
finished session to read it keeps it finished, and sending it a message makes
it a working session again.  A ``budget_exhausted`` session that is sent a
message is refused and re-marked by the same terminal, which is the truth.

The mark is two fields on the record (version 2.11): ``ended_at`` (ISO-8601
UTC) and ``end_reason`` (one of :data:`FINISHED_REASONS`).  Both ride the
session listing (protocol 1.29) through ``session_picker_fields``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, FrozenSet, Optional

#: ``SessionTerminatedEvent.reason`` values that mark a session finished.
FINISHED_REASONS: FrozenSet[str] = frozenset({
    "client_request", "stopped", "natural", "budget_exhausted",
})


def finished_reason_of(event: Any) -> Optional[str]:
    """The finishing reason an event carries, or ``None``.

    Args:
        event: Any routed event.  Only a ``SessionTerminatedEvent`` whose
            ``reason`` is in :data:`FINISHED_REASONS` answers.

    Returns:
        The reason, or ``None`` for every other event.
    """
    if type(event).__name__ != "SessionTerminatedEvent":
        return None
    reason = getattr(event, "reason", None)
    return reason if reason in FINISHED_REASONS else None


def reopens(event: Any, main_agent_id: str) -> bool:
    """Whether an event starts a turn in the session's main agent.

    Args:
        event: Any routed event.
        main_agent_id: The session's main agent id; a subagent going
            active is part of a turn that already started.

    Returns:
        ``True`` for ``AgentStatusChangedEvent(status="active")`` of the
        main agent.
    """
    return (type(event).__name__ == "AgentStatusChangedEvent"
            and getattr(event, "status", None) == "active"
            and getattr(event, "agent_id", None) == main_agent_id)


def note_lifecycle(session: Any, event: Any) -> bool:
    """Mark or clear ``session``'s finished state from one routed event.

    Args:
        session: A session record; ``ended_at`` / ``end_reason`` /
            ``is_dirty`` are written on it, and read with ``getattr`` so a
            record that predates them is treated as unfinished.
        event: The event being routed to it.

    Returns:
        ``True`` when the session was MARKED finished (the caller persists
        it); ``False`` otherwise, including when it was cleared -- a turn
        is starting, and the save at its end writes the cleared record.
    """
    reason = finished_reason_of(event)
    if reason is not None:
        session.ended_at = datetime.now(timezone.utc).isoformat()
        session.end_reason = reason
        session.is_dirty = True
        return True
    # ``getattr``: a duck-typed record (a test double, an out-of-tree
    # session) that never finished has no such attribute, and routing an
    # event to it must not raise.
    if getattr(session, "ended_at", None) is None:
        return False
    server = getattr(session, "server", None)
    main = getattr(server, "_main_agent_id", None) or "main"
    if reopens(event, main):
        session.ended_at = None
        session.end_reason = None
        session.is_dirty = True
    return False
