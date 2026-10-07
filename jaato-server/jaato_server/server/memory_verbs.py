"""The daemon side of the memory verbs (#1232): the owner gate and the answer event.

The rail's four requests (``MemoryListRequest`` / ``MemoryGetRequest`` /
``MemoryUpdateRequest`` / ``MemoryDeleteRequest``) arrive at
``SessionManager.handle_request``, which resolves the session and the
workspace owner and hands them here.  This module decides two things and
nothing else:

* **who may curate** -- :func:`may_curate`, the one predicate both the
  refusal and the list's ``may_curate`` flag read, so the rail can never
  offer a button the daemon then refuses;
* **what the answer looks like** -- one correlated result event per request,
  whatever happened, carrying the caller's ``request_id``.

What a verb DOES is not decided here: the answer comes from
``JaatoServer.memory_op``, which asks the runner holding the store (or, with
no runner at all, the daemon's own plugin).  Keeping the transport-facing
half here keeps ``handle_request`` -- baselined on the complexity ratchet --
at one arm.
"""

from typing import Any, Dict, Optional

from jaato_sdk.events import (
    Event,
    MemoryDeleteRequest,
    MemoryDeleteResultEvent,
    MemoryGetRequest,
    MemoryGetResultEvent,
    MemoryListEvent,
    MemoryListRequest,
    MemoryUpdateRequest,
    MemoryUpdateResultEvent,
)
from jaato_server.shared.plugins.memory.verbs import failure

#: Every request this module answers, for ``handle_request``'s one arm.
MEMORY_REQUEST_TYPES = (
    MemoryListRequest,
    MemoryGetRequest,
    MemoryUpdateRequest,
    MemoryDeleteRequest,
)

#: The verbs that CHANGE the store, and so pass the owner gate.
MUTATING_REQUEST_TYPES = (MemoryUpdateRequest, MemoryDeleteRequest)


def may_curate(owner: Optional[str], user_id: Optional[str]) -> bool:
    """Whether ``user_id`` may update / approve / dismiss / remove memories.

    The workspace OWNER may, and anyone may on an UNOWNED workspace -- the
    shape ``WorkspaceManager.visible_to`` already gives viewing (#1113).
    Deliberately stricter than viewing in one case: on an OWNED workspace a
    connection with no identity is not the owner, so it may look and may not
    change.  Viewing needs no check here -- the request is served for the
    session the caller is attached to, and ``session.attach`` already
    admitted only a session the caller may see.
    """
    return owner is None or (user_id is not None and owner == user_id)


def curator_stamp(user_id: Optional[str]) -> Dict[str, Any]:
    """The ``curated_by`` identity a rail approval records.

    The PERSON the transport authenticated, never the model: a rail action
    has no model in context, and ``_stamp_curation`` would otherwise record
    no approver at all.  ``user`` is omitted -- not set to ``None`` -- when
    the connection carries no identity, so the stamp claims nothing it did
    not observe.
    """
    stamp: Dict[str, Any] = {"kind": "human", "via": "memory.update"}
    if user_id:
        stamp["user"] = user_id
    return stamp


def _update_args(event: MemoryUpdateRequest, user_id: Optional[str]) -> Dict[str, Any]:
    fields = {
        "description": event.description,
        "content": event.content,
        "tags": event.tags,
        "maturity": event.maturity,
    }
    return {
        "memory_id": event.memory_id,
        "fields": {k: v for k, v in fields.items() if v is not None},
        "curator": curator_stamp(user_id),
    }


def _request_op(event: Event, session_id: str, user_id: Optional[str]) -> tuple:
    """``(op, args)`` for the verb ``event`` asks for."""
    if isinstance(event, MemoryListRequest):
        return "list", {"session_id": session_id}
    if isinstance(event, MemoryUpdateRequest):
        return "update", _update_args(event, user_id)
    op = "get" if isinstance(event, MemoryGetRequest) else "delete"
    return op, {"memory_id": event.memory_id}


def result_event(event: Event, answer: Dict[str, Any], *, allowed: bool) -> Event:
    """The correlated answer to ``event``, built from a verb's ``answer``."""
    common = {
        "request_id": getattr(event, "request_id", "") or "",
        "ok": bool(answer.get("ok")),
        "error": str(answer.get("error") or ""),
        "category": str(answer.get("category") or ""),
        "source": str(answer.get("source") or ""),
    }
    if isinstance(event, MemoryListRequest):
        memories = answer.get("memories") if common["ok"] else []
        return MemoryListEvent(memories=memories or [], may_curate=allowed, **common)
    memory_id = getattr(event, "memory_id", "") or ""
    if isinstance(event, MemoryGetRequest):
        return MemoryGetResultEvent(memory_id=memory_id, memory=answer.get("memory"), **common)
    if isinstance(event, MemoryUpdateRequest):
        return MemoryUpdateResultEvent(memory_id=memory_id, memory=answer.get("memory"), **common)
    return MemoryDeleteResultEvent(memory_id=memory_id, **common)


def answer_memory_request(
    server: Any,
    event: Event,
    *,
    session_id: str,
    user_id: Optional[str],
    owner: Optional[str],
) -> Event:
    """Serve one memory request end to end: gate, ask, answer.

    Args:
        server: The session's ``JaatoServer``, or ``None`` when the caller
            is attached to no loaded session (answered ``no_session``).
        event: One of :data:`MEMORY_REQUEST_TYPES`.
        session_id: The session the request is served for -- the "this
            session" the rail highlights.
        user_id: The identity the TRANSPORT authenticated for the caller,
            never a field of the request.
        owner: The session workspace's qualified owner, ``None`` when
            unowned.

    A mutation the owner gate refuses is answered ``not_owner`` WITHOUT
    asking the runner, so nothing is changed and nothing is read.
    """
    allowed = may_curate(owner, user_id)
    if server is None:
        answer = failure("no_session", "no session is attached to this connection")
    elif isinstance(event, MUTATING_REQUEST_TYPES) and not allowed:
        answer = failure(
            "not_owner",
            "only the owner of this session's workspace may change its memories",
        )
    else:
        op, args = _request_op(event, session_id, user_id)
        answer = server.memory_op(op, args)
    return result_event(event, answer, allowed=allowed)


__all__ = [
    "MEMORY_REQUEST_TYPES",
    "MUTATING_REQUEST_TYPES",
    "answer_memory_request",
    "curator_stamp",
    "may_curate",
    "result_event",
]

