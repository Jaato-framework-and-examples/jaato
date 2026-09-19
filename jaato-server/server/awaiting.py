"""What a session is waiting on a HUMAN for, as one scalar (#1138).

A session that raises a permission ASK or a ``request_clarification`` is
**blocked** until somebody answers it, and until now that fact reached only
the clients attached to *that* session: events go to
``session.attached_clients`` and ``_client_to_session`` is 1:1, so a browser
working in session A never learned that session B wanted it.  Someone with
five sessions open could not tell which one was asking.

This module owns the vocabulary and the one precedence rule behind
``RuntimeSessionInfo.awaiting`` / ``.awaiting_since``, the two fields
``SessionManager.list_sessions`` publishes on the in-memory branch.

WHY A SCALAR, AND WHY A SECOND FIELD RATHER THAN A UNION
=========================================================

``awaiting`` is a plain string (or ``None``) because the whole degradation
argument rests on an older client being able to *ignore* it: an additive
scalar on an already free-form listing row costs a client that does not know
it exactly nothing, where a new EVENT would reach ``deserialize_event``'s
raise-on-unknown-type.  So ``awaiting_since`` is a SECOND field rather than
``awaiting`` widening into ``{kind, since}`` — a reader doing
``typeof row.awaiting === "string"`` keeps working, and a reader that wants
the clock opts into one more key.

``is_processing`` cannot carry either of them.  A session blocked on a prompt
is still processing; telling *working* from *waiting on you* is the entire
point, and one boolean cannot.

WHICH PROMPT THE PAIR DESCRIBES
================================

Both prompts can be in flight at once — tool execution is 8-wide by default,
so one tool in a batch can hit a permission gate while another calls
``request_clarification`` — and the pair has room for one.  The rule is
**the oldest unanswered prompt**, because that is the question a reader is
actually asking ("which of these has been sitting longest") and because it
needs no invented ranking between the two kinds.

Ranking is the TIEBREAK only, for the case the clock cannot separate:
an undated prompt (a test that sets the daemon-side field by hand, or a
future raise site that forgets to stamp) still reports its KIND, and
``awaiting_since`` is then absent.  Absent means "not measured", never
"just now" — the rule this tree applies to every other absent measurement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple


#: ``awaiting`` when a permission ASK is unanswered.
AWAITING_PERMISSION = "permission"

#: ``awaiting`` when a ``request_clarification`` batch is unanswered.
AWAITING_CLARIFICATION = "clarification"

#: The closed vocabulary, in TIEBREAK order.  Only consulted when two
#: prompts cannot be separated by their raise instants; see the module
#: docstring.  Permission first because it gates a side effect that has not
#: happened yet, where a clarification gates an answer inside a tool call
#: that was already allowed to run.
AWAITING_KINDS: Tuple[str, ...] = (AWAITING_PERMISSION, AWAITING_CLARIFICATION)

_RANK = {kind: index for index, kind in enumerate(AWAITING_KINDS)}


class PendingPrompt(NamedTuple):
    """One unanswered human-facing prompt.

    Attributes:
        kind: A member of :data:`AWAITING_KINDS`.
        since: Epoch seconds (wall clock) at which the prompt was raised,
            or ``None`` when the holder carries no stamp.
    """

    kind: str
    since: Optional[float]


def resolve_awaiting(
    candidates: Iterable[PendingPrompt],
) -> Optional[PendingPrompt]:
    """The one prompt ``awaiting`` / ``awaiting_since`` describe.

    The oldest dated candidate wins.  When nothing is dated the vocabulary
    order breaks the tie, so the answer is deterministic whatever order the
    holders were read in — a UI marker that flickers because two dicts
    happened to iterate differently is worse than no marker.

    Args:
        candidates: Every prompt currently unanswered on this session.

    Returns:
        The winning prompt, or ``None`` when nothing is pending.
    """
    pending: List[PendingPrompt] = [
        prompt for prompt in candidates if prompt.kind in _RANK
    ]
    if not pending:
        return None
    dated = [prompt for prompt in pending if prompt.since is not None]
    if dated:
        return min(dated, key=lambda p: (p.since, _RANK[p.kind]))
    return min(pending, key=lambda p: _RANK[p.kind])


def awaiting_fields(
    prompt: Optional[PendingPrompt],
) -> Tuple[Optional[str], Optional[str]]:
    """Render *prompt* as the two listing fields.

    The instant is rendered as an ISO-8601 UTC string rather than passed
    through as epoch seconds, so it matches the row it rides on
    (``created_at`` / ``last_activity`` are both ``.isoformat()`` strings)
    and so a browser can subtract it from its own clock.

    A MONOTONIC clock would be wrong here even though the daemon's own
    lifetime bounds use one: monotonic instants mean nothing outside the
    process that produced them, and this number's only consumer is a
    client rendering "waiting 4 min".

    THE VOCABULARY IS ENFORCED HERE, not only produced here.  This is the
    boundary between "whatever the session server answered" and "what goes
    on the wire", and :func:`awaiting_of` accepts any duck-typed server --
    so a ``kind`` outside :data:`AWAITING_KINDS`, or a ``since`` that is
    not a real number, is dropped rather than published.  A closed
    vocabulary a client may branch on is only closed if something closes
    it; and the alternative failure -- a listing that raises because one
    session's server answered oddly -- would be worse than the fact it
    was reporting.

    Args:
        prompt: The winner from :func:`resolve_awaiting`, or ``None``.

    Returns:
        ``(awaiting, awaiting_since)``, either or both ``None``.
    """
    kind = getattr(prompt, "kind", None)
    if kind not in _RANK:
        return (None, None)
    since = getattr(prompt, "since", None)
    if not isinstance(since, (int, float)) or isinstance(since, bool):
        return (kind, None)
    try:
        stamp = datetime.fromtimestamp(float(since), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return (kind, None)
    return (kind, stamp.isoformat())


def awaiting_of(server: object) -> Tuple[Optional[str], Optional[str]]:
    """``(awaiting, awaiting_since)`` for a loaded session's ``JaatoServer``.

    The listing's in-memory branch reads this the way it already reads
    ``session.server.is_processing`` — one more read on the object it
    already has in hand.

    ``getattr`` rather than a direct call, and a server that answers
    nothing degrades to ``(None, None)``: an out-of-tree or duck-typed
    session server predating the method then behaves exactly as it did
    before the field existed, which is the ``_turns_ran_snapshot``
    precedent (#881).  A real ``JaatoServer`` never takes that path, and
    :meth:`JaatoServer.awaiting_prompt` is written not to raise, so this
    is a capability check and not an error swallow.

    Args:
        server: The session's ``JaatoServer`` (or a stand-in for one).

    Returns:
        ``(awaiting, awaiting_since)``.
    """
    probe = getattr(server, "awaiting_prompt", None)
    if not callable(probe):
        return (None, None)
    return awaiting_fields(probe())


def pending_prompt(
    relay: object,
    local_request_id: Optional[str],
    local_since: Optional[float],
    kind: str,
) -> Optional[PendingPrompt]:
    """One kind's unanswered prompt, read from whichever path holds it.

    A prompt of one kind has TWO possible holders and they belong to
    different deployments rather than to different moments:

    * the runner-RPC **relay** (``PromptOperatorHandler`` /
      ``ClarificationRelayHandler``) on the default, runner-served path,
      where the plugin that raises the prompt is ``PLUGIN_TIER = "runner"``
      and the daemon-side hook is not in the loop;
    * the daemon-local ``_pending_*_request_id`` pair, on the embedded and
      standalone-WS paths, and as the legacy fallback
      ``JaatoServer.respond_to_permission`` still documents as Path 2.

    The relay is consulted first because it is the live path where both
    exist; the daemon-local field is consulted whenever the relay holds
    nothing, which is what keeps the fallback covered rather than shadowed.

    Args:
        relay: The relay handler, or ``None`` when this session has no
            runner (and so no relay was ever registered).
        local_request_id: The daemon-local pending id, falsy when none.
        local_since: The daemon-local stamp, ``None`` when unmeasured.
        kind: The :data:`AWAITING_KINDS` member these holders carry.

    Returns:
        The pending prompt, or ``None`` when neither holder has one.
    """
    if relay is not None:
        probe = getattr(relay, "has_pending_prompt", None)
        if callable(probe) and probe():
            since = getattr(relay, "pending_since", None)
            return PendingPrompt(kind, since() if callable(since) else None)
    if local_request_id:
        return PendingPrompt(kind, local_since)
    return None


def oldest(stamps: Sequence[float]) -> Optional[float]:
    """The earliest of *stamps*, or ``None`` when there are none.

    Shared by the two runner-RPC relay handlers so "when did the oldest
    in-flight prompt arrive" has one definition rather than one per
    handler.
    """
    return min(stamps) if stamps else None
