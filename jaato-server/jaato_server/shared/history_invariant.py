"""The one place the conversation-history pairing invariant is written down.

Every provider this framework supports requires that an assistant turn's
function calls and the function results that answer them stay **paired**:
every ``FunctionCall`` has exactly one ``ToolResult`` carrying its
``call_id``, no result precedes the call it answers, and no content block
is empty.  Break the pairing and the upstream rejects the *whole* request
with a 400 — several turns after the damage was done, which is what makes
this class of defect so expensive to diagnose.

**Why a boundary validator rather than four careful subsystems.**  Four
subsystems edit history independently and none of them can see the others:

======================  ====================================================
subsystem               how it breaks the pairing
======================  ====================================================
GC plugins              ``gc_truncate`` / ``gc_summarize`` / ``gc_hybrid`` /
                        ``gc_budget`` drop or rewrite turns.  ``gc_hybrid``
                        has three boundaries (recent preserved, middle
                        summarized, ancient truncated) and each is a chance
                        to cut between a call and its result.
cancellation            a :class:`CancelToken` can stop a turn after the
                        model emitted N calls and fewer than N ran.
                        Parallel execution (8 wide by default) widens the
                        window, and the batch is answered *partially*.
rewind                  ``rewind.py`` restores to an earlier point; a
                        history that now *begins* with a tool result is
                        exactly the leading-orphan shape.
the wire                a third-party OpenAI-compatible endpoint that
                        streams a tool call with **no id** puts a call into
                        history that no result can ever be matched to.
                        Fixed at the seam too (see
                        :func:`synthetic_tool_call_id`); this is the
                        backstop for the histories already on disk.
======================  ====================================================

Enforcing at each site means auditing four subsystems plus every future GC
plugin.  Enforcing at the boundary — the single point where history is
handed to a provider — catches all four, plus subagent history sharing,
without any of them knowing about the others.

**The boundary is** :meth:`JaatoSession._history_for_provider`, and this
module follows the precedent that seam already sets (#847): it operates on
a **per-request copy** and never mutates stored history.  That distinction
is load-bearing rather than stylistic.  A turn cancelled mid-batch leaves
three calls and one result in the store; repairing the *store* would write
two synthetic "cancelled" results next to calls the session may still be
about to execute, and the real results would then arrive as duplicates.
Repairing the *copy* lets the stored history stay the truth while every
request built from it is well-formed.

**Repair, do not reject.**  Letting the provider 400 is the behaviour being
fixed, and dropping the offending turn loses work the user paid for.  Each
repair is deterministic and, critically, **byte-stable across requests**:
the synthesised result text carries no timestamp and no random id, because
an unstable prefix would invalidate the Anthropic/Gemini prompt cache on
every single turn (see ``cache_anthropic`` and the ``cache_prompt`` knob).

**Reasoning replay is why repair adds rather than removes.**  ``minimax``,
``kimi`` and ``mimo`` set ``replay_reasoning = True``, so the session keeps
``Part.thought`` in history and replays it as ``reasoning_content`` — MiMo
answers **400** to a tool loop that omits it and Kimi K3 wants the
assistant message back as-is.  The obvious repair for an unanswered call
("delete the assistant message") would therefore trade a pairing 400 for a
reasoning-replay 400.  So a MODEL message is **never** removed by this
module: an unanswered call is *answered* with a synthetic result, which
leaves the turn's reasoning, text and call ids exactly where the upstream
expects them.

**Two callers, one policy, different reach.**
:func:`shared.plugins.gc.utils.ensure_tool_call_integrity` is the *post-GC*
repair the session applies to **stored** history at four GC call sites, and
since #674 it delegates to :func:`repair_history` rather than implementing a
second, contradictory policy of its own.  It used to repair by deletion, and
on a partially-answered batch that produced the very orphan it existed to
prevent (see that function's docstring for the reproducing sequence).  The
split of responsibility is now:

* **producer-side, destructive, between turns** — the GC path fixes the
  record once, so the repair is persisted;
* **boundary, per-request, non-destructive** — this module's call from
  :meth:`JaatoSession._history_for_provider` is the backstop for the
  producers GC never sees: cancellation mid-batch, ``rewind``, subagent
  history sharing, and a wire that streamed no tool-call id.

**What happens to history that is ALREADY corrupt on disk.**  Nothing
migrates it, deliberately — there is no record rewrite in this change.  A
session persisted in an orphaned state (GC'd by the old deletion policy,
then written out, then revived through ``SessionManager._load_session``) is
covered in two ways and neither is a migration:

1. it is never *sent* corrupt, because every request goes through
   :meth:`JaatoSession._history_for_provider`, which repairs the copy;
2. the first GC pass in the revived session heals the stored record, because
   the producer that used to corrupt it now repairs it.

So an untouched corrupt record stays corrupt on disk until one of those two
happens, and any *future* reader that bypasses both would still see it.  That
is a stated limitation rather than an oversight: rewriting persisted session
records is a migration with its own failure modes, and the two paths above
cover every route this tree actually reads history by.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, replace as _dc_replace
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
)

__all__ = [
    "SYNTHETIC_CALL_ID_PREFIX",
    "CANCELLED_RESULT",
    "HistoryDefect",
    "new_tool_call_nonce",
    "synthetic_tool_call_id",
    "validate_history",
    "repair_history",
]


#: Marks an id this framework minted because the producer supplied none.
#: Shared by the wire seam (:func:`synthetic_tool_call_id`, called from the
#: provider streaming loops) and by the boundary backstop, so an operator
#: grepping a trace for one concept finds both.
SYNTHETIC_CALL_ID_PREFIX = "jaato_synth_call_"

logger = logging.getLogger(__name__)

#: One entry per repair pass: (severity, message template).
#:
#: **Severity is the diagnostic half of this module, and it is not
#: uniform.**  A repair here is cheap and silent by construction — it turns
#: a provider 400 into a correction nobody sees — which is a better user
#: experience and a worse diagnostic.  Since #674 also fixed the producers
#: (``ensure_tool_call_integrity`` and the four streaming loops), a repair
#: at the boundary now means one of two very different things, and the
#: level says which:
#:
#: ``WARNING``
#:     the producer that should have prevented this did not.  A minted id
#:     means a tool call reached history with no id, and every streaming
#:     loop in this tree mints one at the seam — so either the history
#:     predates that fix, or a producer is not minting.  A rising count is
#:     how that announces itself; nothing else would.
#:
#: ``INFO``
#:     a shape that legitimately still arrives here, because no producer
#:     sees it: cancellation mid-batch, ``rewind`` / resume leaving a
#:     leading orphan, subagent history sharing, and a thought-only turn.
#:     These are the reason the backstop exists and are not defects.
#:
#: The split is honest rather than clean: an orphaned result *could* also
#: mean GC regressed, and nothing at this layer can tell that from a
#: rewind.  So the rule is "WARNING only where the producer-side fix makes
#: recurrence a defect by construction", which is the line that can be
#: defended.  Every kind is still counted and named at both sinks.
_REPAIR_REPORTS = (
    (logging.WARNING, "minted {n} missing tool call id(s) {items}"),
    (logging.INFO, "dropped {n} orphaned tool result(s) {items}"),
    (logging.INFO,
     "synthesised cancelled result(s) for {n} unanswered call(s) {items}"),
    (logging.INFO,
     "removed empty content block(s) from {n} message(s) at indices {items}"),
    (logging.INFO,
     "dropped {n} message(s) left with no content at indices {items}"),
)

#: The payload a synthesised result carries.  Deliberately identical to the
#: one :meth:`JaatoSession._inject_synthetic_cancelled_results` already
#: writes, so a model sees one shape whether the reconciliation happened at
#: the cancellation site or here, and **constant**, so the request prefix is
#: byte-stable across turns and the prompt cache survives.
CANCELLED_RESULT = {
    "error": "cancelled",
    "detail": (
        "This tool call was never executed -- the turn ended before it ran "
        "(cancellation, context collection, or a restored session). No "
        "action was taken. Re-issue the call if the result is still needed."
    ),
}


@dataclass(frozen=True)
class HistoryDefect:
    """One violation of the pairing invariant, as found by validation.

    Attributes:
        kind: Stable machine-readable code — one of ``"unmatched_call"``,
            ``"orphan_result"``, ``"missing_call_id"``,
            ``"duplicate_call_id"`` or ``"empty_content"``.  Tests and
            traces key on this rather than on the prose.
        message_index: Index into the history list the defect was found
            at.  Meaningful only against the list that was validated.
        detail: Human-readable description for the trace line.
    """

    kind: str
    message_index: int
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind}@{self.message_index}: {self.detail}"


def new_tool_call_nonce() -> str:
    """A fresh per-response discriminator for :func:`synthetic_tool_call_id`.

    Providers call this **once per streaming response**, next to the
    tool-call accumulator it belongs to, and pass the result to every id
    they mint from that response.  Living here rather than being four
    inlined ``uuid`` calls keeps the whole "what a minted id looks like"
    question in one module, which is what lets the boundary backstop and
    the wire seam stay recognisably the same concept.
    """
    return uuid.uuid4().hex[:8]


def synthetic_tool_call_id(index: int, nonce: str) -> str:
    """Mint a tool-call id for a producer that supplied none.

    Called from the streaming loops of every provider that accumulates
    OpenAI-shaped tool-call deltas keyed by index (``_openai_compat`` and
    therefore nim / nebius / ovhcloud / doubleword / zhipuai_openai, plus
    ``openrouter``, ``vllm`` and the models endpoint provider, each of
    which owns its own loop).  Those endpoints are *supposed* to send an
    id on the delta that opens a call; several self-hosted servers whose
    OpenAI compatibility is approximate do not, and the accumulated id
    stays ``None``.  Downstream that reaches history as ``call_id=""`` and
    the call can never be matched to its result — an orphan arriving from
    the wire rather than from GC, cancellation or rewind.

    Fixing it here rather than only at the boundary matters because the id
    is what the *model* is told it called; minting at the seam means the
    result the framework sends back names the same id the assistant
    message does, which is the pairing the upstream checks.

    Args:
        index: The delta's ``index`` field — unique within one response,
            which is what makes the ids of a parallel batch distinct.
        nonce: A per-response discriminator.  Required rather than
            defaulted because ``index`` alone is unique only *within* a
            response: two separate turns both minting ``..._0`` would put
            two different calls under one id in the same history, which is
            the defect wearing the fix as a disguise.  Providers pass a
            short uuid fragment; tests pass a fixed string.

    Returns:
        An id carrying :data:`SYNTHETIC_CALL_ID_PREFIX`, so the trace and
        any later audit can tell a minted id from an upstream one.
    """
    return f"{SYNTHETIC_CALL_ID_PREFIX}{nonce}_{index}"


# ==================== Validation (pure) ====================


def _is_result_message(msg: Message) -> bool:
    """Whether ``msg`` carries tool results.

    Results normally ride a ``Role.TOOL`` message, but the Gemini-shaped
    path puts them on a ``Role.USER`` message, so both are recognised.
    """
    if msg.role == Role.TOOL:
        return True
    return msg.role == Role.USER and any(
        p.function_response is not None for p in (msg.parts or [])
    )


def _calls_of(msg: Message) -> List[FunctionCall]:
    """Every ``FunctionCall`` carried by ``msg``, in part order."""
    return [p.function_call for p in (msg.parts or []) if p.function_call]


def _results_of(msg: Message) -> List[ToolResult]:
    """Every ``ToolResult`` carried by ``msg``, in part order."""
    return [
        p.function_response for p in (msg.parts or []) if p.function_response
    ]


def _part_is_empty_content(part: Part) -> bool:
    """Whether ``part`` is an empty content block no wire will accept.

    A text part whose text is empty or whitespace-only. This is the shape
    behind *"text content blocks must be non-empty"* and behind the
    whitespace-next-to-a-thinking-block variant that bypassed
    normalisation: both are a text block with nothing in it.  A
    ``thought``-only part is **not** empty content — for a
    ``replay_reasoning`` provider it is required payload.
    """
    if part.text is None:
        return False
    if part.text.strip():
        return False
    return not (
        part.function_call
        or part.function_response
        or part.inline_data
        or part.thought
    )


def _scan_ids(
    messages: Sequence[Message],
) -> Tuple[List[HistoryDefect], Set[str], Set[str]]:
    """Collect id-shaped defects and the call/result id sets.

    Returns ``(defects, call_ids, answered_ids)`` where ``call_ids`` is
    every id the model asked under and ``answered_ids`` every id a result
    claims to answer.  A result whose id is not yet in ``call_ids`` when it
    is reached is a *leading* orphan — the resume shape — and is reported
    with its own position.
    """
    defects: List[HistoryDefect] = []
    call_ids: Set[str] = set()
    answered: Set[str] = set()
    for idx, msg in enumerate(messages):
        for call in _calls_of(msg):
            if not call.id:
                defects.append(HistoryDefect(
                    "missing_call_id", idx,
                    f"call to {call.name!r} carries no id"))
                continue
            if call.id in call_ids:
                defects.append(HistoryDefect(
                    "duplicate_call_id", idx,
                    f"id {call.id!r} was already used by an earlier call"))
            call_ids.add(call.id)
        for res in _results_of(msg):
            if not res.call_id:
                defects.append(HistoryDefect(
                    "missing_call_id", idx,
                    f"result for {res.name!r} carries no call_id"))
                continue
            if res.call_id not in call_ids:
                defects.append(HistoryDefect(
                    "orphan_result", idx,
                    f"result answers {res.call_id!r}, which no preceding "
                    f"call requested"))
            answered.add(res.call_id)
    return defects, call_ids, answered


def validate_history(messages: Sequence[Message]) -> List[HistoryDefect]:
    """Report every way ``messages`` violates the pairing invariant.

    Pure: reads the list, allocates nothing that outlives the call, and
    never mutates a ``Message``.  :func:`repair_history` is the half that
    acts; this half is what the property tests assert against, before and
    after every GC strategy and every cancel-at-turn-N cut.

    Args:
        messages: A history slice, oldest first.

    Returns:
        The defects found, in history order.  Empty means the history is
        safe to hand to any provider this framework supports.
    """
    defects, call_ids, answered = _scan_ids(messages)
    for idx, msg in enumerate(messages):
        for call in _calls_of(msg):
            if call.id and call.id not in answered:
                defects.append(HistoryDefect(
                    "unmatched_call", idx,
                    f"call {call.id!r} to {call.name!r} was never answered"))
        for part in (msg.parts or []):
            if _part_is_empty_content(part):
                defects.append(HistoryDefect(
                    "empty_content", idx,
                    f"empty text block on a {msg.role.value} message"))
    defects.sort(key=lambda d: d.message_index)
    return defects


# ==================== Repair ====================


def _mint_scope(msg: Message) -> str:
    """The per-message discriminator minted ids are namespaced under.

    ``Message.message_id`` is the right source: it is stable across
    requests (GC and the modality gate both preserve it through
    ``dataclasses.replace``, because GC's history-budget sync keys on it),
    so the same stored history mints the same ids every turn and the repair
    does not itself invalidate the upstream prompt cache.

    It has a ``uuid4`` default and the session serializer round-trips it, so
    it is effectively always present — but a hand-built or hand-edited
    ``Message`` can carry ``""``, and an empty scope would make two
    different id-less messages mint the *same* ids at the same part
    positions.  The fallback is therefore a digest of the message's own
    shape rather than a constant: still deterministic, still distinct.
    """
    if msg.message_id:
        return msg.message_id[:8]
    shape = "|".join(
        f"{p.function_call.name if p.function_call else ''}"
        f":{p.function_response.name if p.function_response else ''}"
        f":{p.text or ''}"
        for p in (msg.parts or [])
    )
    return hashlib.sha256(shape.encode("utf-8", "replace")).hexdigest()[:8]


def _mint_ids_for_message(
    msg: Message, pending: List[str], minted: List[str]
) -> Message:
    """Give ids to the id-less calls and results of one message.

    Calls are minted from the message's own ``message_id`` plus the part
    position, which is stable across requests (``message_id`` survives GC
    and the modality gate's ``dataclasses.replace``) — so the same history
    mints the same ids on every turn and the prompt cache is not
    invalidated by the repair itself.

    Results are matched **positionally** to the calls still pending from
    the most recent batch, which is the only correspondence available once
    the ids are gone, and is the order the wire produced them in.

    Args:
        msg: The message to repair.
        pending: Ids from the most recent call batch not yet answered;
            consumed (in order) by id-less results.
        minted: Accumulator of the ids this pass created, for the trace.

    Returns:
        ``msg`` itself when it had nothing to mint, else a repaired copy.
    """
    parts = msg.parts or []
    if not any(
        (p.function_call and not p.function_call.id)
        or (p.function_response and not p.function_response.call_id)
        for p in parts
    ):
        return msg
    new_parts: List[Part] = []
    for pos, part in enumerate(parts):
        if part.function_call and not part.function_call.id:
            new_id = f"{SYNTHETIC_CALL_ID_PREFIX}{_mint_scope(msg)}_{pos}"
            minted.append(new_id)
            new_parts.append(Part.from_function_call(
                _dc_replace(part.function_call, id=new_id)))
        elif part.function_response and not part.function_response.call_id:
            new_id = pending.pop(0) if pending else (
                f"{SYNTHETIC_CALL_ID_PREFIX}{_mint_scope(msg)}_{pos}")
            minted.append(new_id)
            new_parts.append(Part.from_function_response(
                _dc_replace(part.function_response, call_id=new_id)))
        else:
            new_parts.append(part)
    return _dc_replace(msg, parts=new_parts)


def _mint_missing_ids(
    messages: List[Message], minted: List[str]
) -> List[Message]:
    """Backstop for the wire seam: mint ids the producer never sent.

    Walks the history keeping the ids of the most recent unanswered call
    batch, so an id-less *result* can be paired with the id-less *call* it
    plainly answers rather than being dropped as an orphan.
    """
    out: List[Message] = []
    pending: List[str] = []
    for msg in messages:
        repaired = _mint_ids_for_message(msg, pending, minted)
        if _calls_of(repaired):
            pending = [c.id for c in _calls_of(repaired) if c.id]
        elif _is_result_message(repaired):
            answered = {r.call_id for r in _results_of(repaired)}
            pending = [i for i in pending if i not in answered]
        out.append(repaired)
    return out


def _drop_orphan_results(
    messages: List[Message], dropped: List[str]
) -> List[Message]:
    """Remove results that answer a call this history does not contain.

    The resume shape: ``rewind`` (or a GC cut, or a restored session whose
    snapshot began mid-batch) leaves a history that *opens* with a tool
    result.  Every wire rejects it — Anthropic by name, with *"unexpected
    ``tool_use_id`` found in ``tool_result`` blocks"*.

    Dropping is the right repair here and synthesis is not: there is no
    call to attach a synthetic partner to, and inventing an assistant turn
    would put words in the model's mouth.  A result message that loses
    *all* its parts is dropped whole; one that keeps some is copied with
    the survivors, so a mixed batch does not lose its good results.
    """
    out: List[Message] = []
    seen: Set[str] = set()
    for msg in messages:
        for call in _calls_of(msg):
            if call.id:
                seen.add(call.id)
        if not _is_result_message(msg):
            out.append(msg)
            continue
        parts = msg.parts or []
        kept: List[Part] = []
        for part in parts:
            res = part.function_response
            if res is None or res.call_id in seen:
                kept.append(part)
            else:
                dropped.append(res.call_id)
        if len(kept) == len(parts):
            out.append(msg)
        elif kept:
            out.append(_dc_replace(msg, parts=kept))
    return out


def _synthetic_result_part(call: FunctionCall) -> Part:
    """The stand-in result for a call that was never answered.

    Carries :data:`CANCELLED_RESULT` — a constant, so two requests built
    from the same history are byte-identical and the upstream prompt cache
    still hits — and ``is_error=True``, so a model reading it treats the
    call as failed rather than as having returned nothing.
    """
    return Part.from_function_response(ToolResult(
        call_id=call.id,
        name=call.name,
        result=CANCELLED_RESULT,
        is_error=True,
    ))


def _place_synthetic_results(
    out: List[Message], nxt: Optional[Message], parts: List[Part]
) -> bool:
    """Put the synthesised results where the wire expects to find them.

    Immediately after the assistant message that made the calls: merged
    into that batch's own result message when one follows, so the upstream
    sees a single result block per call in one place — the shape an
    uninterrupted turn produces — and as a fresh ``Role.TOOL`` message
    when the batch was answered by nothing at all.

    Args:
        out: The result list being built; appended to in place.
        nxt: The message that followed the assistant turn, if any.
        parts: The synthesised ``function_response`` parts.

    Returns:
        ``True`` when ``nxt`` was merged into and the caller must not
        append it again.
    """
    if nxt is not None and _is_result_message(nxt):
        out.append(_dc_replace(nxt, parts=list(nxt.parts or []) + parts))
        return True
    out.append(Message(role=Role.TOOL, parts=parts))
    return False


def _answer_unmatched_calls(
    messages: List[Message], synthesized: List[str]
) -> List[Message]:
    """Give every unanswered call a result, without touching its message.

    The cancellation shape, and the one the pre-existing GC repair gets
    wrong: a batch of three calls of which one ran leaves two unanswered,
    and *deleting the assistant message* both orphans the result that did
    arrive and discards the turn's reasoning (which ``replay_reasoning``
    providers require back verbatim).  So the MODEL message is left
    untouched and the missing results are inserted immediately after it —
    merged into the batch's own result message when there is one, so the
    wire sees one result block per call in one place, exactly as an
    uninterrupted turn would have produced.
    """
    answered = {
        r.call_id for m in messages for r in _results_of(m) if r.call_id
    }
    out: List[Message] = []
    skip_next = False
    for idx, msg in enumerate(messages):
        if skip_next:
            skip_next = False
            continue
        out.append(msg)
        missing = [c for c in _calls_of(msg) if c.id and c.id not in answered]
        if not missing:
            continue
        synthesized.extend(c.id for c in missing)
        answered.update(c.id for c in missing)
        nxt = messages[idx + 1] if idx + 1 < len(messages) else None
        # ``skip_next`` stops the loop re-appending a message that
        # :func:`_place_synthetic_results` merged the new results into.
        skip_next = _place_synthetic_results(
            out, nxt, [_synthetic_result_part(c) for c in missing])
    return out


def _drop_empty_content(
    messages: List[Message], emptied: List[int], dropped: List[int]
) -> List[Message]:
    """Remove empty text blocks, and messages left with no content at all.

    *"Conversations getting stuck on 'text content blocks must be
    non-empty' after a turn where the model produced only thinking"* is
    this, and reasoning replay makes it **more** reachable rather than
    less: a ``replay_reasoning`` session deliberately keeps thought parts
    in history, so a thought-only turn is a live shape rather than a
    curiosity.  The thought part is kept and the empty text beside it is
    removed — dropping the message instead would strip the
    ``reasoning_content`` MiMo answers 400 without.

    Args:
        messages: History to filter.
        emptied: Accumulator of indices that merely *lost* parts.  Kept
            separate from ``dropped`` because the two are different
            events in the trace, and because :func:`repair_history`
            decides whether anything changed by looking at these lists —
            folding a part-level removal into silence is how an empty
            block survived the repair that removed it.
        dropped: Accumulator of indices removed entirely.
    """
    out: List[Message] = []
    for idx, msg in enumerate(messages):
        parts = msg.parts or []
        kept = [p for p in parts if not _part_is_empty_content(p)]
        if len(kept) == len(parts):
            out.append(msg)
        elif kept:
            emptied.append(idx)
            out.append(_dc_replace(msg, parts=kept))
        else:
            dropped.append(idx)
    return out


def repair_history(
    messages: Sequence[Message],
    trace_fn: Optional[Callable[[str], None]] = None,
) -> List[Message]:
    """Return a history that satisfies the invariant, repairing as needed.

    The four repairs run in a fixed order because each depends on the one
    before it:

    1. **mint missing ids** — until every call has an id, nothing else can
       tell a pair from an orphan;
    2. **drop orphan results** — a result answering a call this history
       does not contain has no partner to synthesise;
    3. **answer unmatched calls** — every remaining call gets a result;
    4. **drop empty content** — last, so a message emptied by an earlier
       step is noticed.

    **This returns a copy and never mutates its input.**  Callers that
    hand the result to a provider (the sole production caller is
    :meth:`JaatoSession._history_for_provider`) are therefore repairing
    the per-request view while the session's stored history stays the
    record of what actually happened — the same non-destructive contract
    :meth:`JaatoSession._gate_history_for_active_modalities` holds to, and
    for the same reason: the truth on disk must survive a repair aimed at
    one request.

    Args:
        messages: A history slice, oldest first.  Not modified.
        trace_fn: Optional sink for one line per repair kind.  A silent
            repair is nearly as bad as a 400 — it hides a GC strategy or a
            provider that is producing broken histories — so anything
            repaired is announced.

    Returns:
        ``messages`` itself (the same object, when it is a ``list``) when
        nothing needed repairing, so the overwhelmingly common healthy
        history allocates nothing; otherwise a new list of messages, in
        which unrepaired messages are the *stored* objects rather than
        copies.
    """
    if not messages:
        return list(messages)
    minted: List[str] = []
    dropped_results: List[str] = []
    synthesized: List[str] = []
    emptied_msgs: List[int] = []
    dropped_msgs: List[int] = []

    out = _mint_missing_ids(list(messages), minted)
    out = _drop_orphan_results(out, dropped_results)
    out = _answer_unmatched_calls(out, synthesized)
    out = _drop_empty_content(out, emptied_msgs, dropped_msgs)

    findings = (minted, dropped_results, synthesized,
                emptied_msgs, dropped_msgs)
    changed = any(findings)
    if changed:
        # Reported even when the caller passed no trace sink: the logger
        # is the half a deployment aggregates, and a repair nobody can
        # see is the defect this reporting exists to prevent.
        _report_repairs(trace_fn, findings)
    else:
        return messages if isinstance(messages, list) else list(messages)
    return out


def _report_repairs(
    trace_fn: Optional[Callable[[str], None]],
    findings: Sequence[Sequence[object]],
) -> None:
    """Announce every repair that fired, to BOTH sinks, with counts.

    Two sinks because they have different audiences and different
    availability.  ``trace_fn`` is the session's own trace (the artefact
    every deployment gets, beside the permission DECISION lines and the
    BUDGET lines an operator correlates against); ``logger`` is where a
    deployment that aggregates logs will actually notice a rising count.
    A repair recorded in neither is the silent-correction failure this
    module would otherwise introduce — a validator that repairs everything
    and says nothing turns a loud 400 into permanent quiet damage.

    The message names **what** was repaired and how much of it, never
    merely that a repair happened: a count is what a future producer
    defect moves, and "a repair occurred" is not.

    Args:
        trace_fn: The session trace sink, or ``None``.
        findings: One sequence per pass, positionally matching
            :data:`_REPAIR_REPORTS`; an empty one means that pass did
            nothing and is not reported.
    """
    for (level, template), items in zip(_REPAIR_REPORTS, findings):
        if not items:
            continue
        line = ("HISTORY_INVARIANT: "
                + template.format(n=len(items), items=list(items)))
        if trace_fn:
            trace_fn(line)
        logger.log(level, "%s", line)
