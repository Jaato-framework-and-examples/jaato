"""Utility functions for Context Garbage Collection.

Provides helpers for turn splitting, token estimation, and history manipulation.
"""

import hashlib
import json
from dataclasses import dataclass, replace as _dc_replace
from typing import (
    Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING,
)

from jaato_sdk.media_identity import (
    ATTACHMENT_ID_KEY,
    audio_duration_seconds,
    describe_attachment,
    mint_attachment_id,
    split_mime,
)
from jaato_sdk.plugins.model_provider.types import (
    Message,
    Part,
    Role,
    ToolResult,
)
# The pairing invariant has ONE definition, and this module is a consumer of
# it rather than a second author (#674).  Import direction is safe:
# ``history_invariant`` depends only on the SDK types, never on GC.
from jaato_server.shared.history_invariant import repair_history

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .base import GCConfig, GCTriggerReason


@dataclass
class Turn:
    """Represents a conversation turn (user message + model response(s)).

    A turn typically consists of:
    - One user Message (role=USER)
    - One or more model Message objects (role=MODEL)
    - Possibly function response Message objects (role=USER with function_response parts,
      or role=TOOL for Anthropic provider)
    """

    index: int
    """Turn index (0-based)."""

    contents: List[Message]
    """All Message objects in this turn."""

    estimated_tokens: int = 0
    """Estimated token count for this turn."""

    @property
    def is_empty(self) -> bool:
        """Check if this turn has no content."""
        return len(self.contents) == 0


def split_into_turns(history: List[Message]) -> List[Turn]:
    """Split conversation history into logical turns.

    A turn starts with a user message and includes all subsequent
    model responses until the next user message. Function responses
    (user role with function_response parts) are grouped with the
    preceding model response.

    Args:
        history: List of Message objects from conversation history.

    Returns:
        List of Turn objects, each containing related Message objects.
    """
    if not history:
        return []

    turns: List[Turn] = []
    current_turn_contents: List[Message] = []
    turn_index = 0

    for message in history:
        # Check if this is a new user message (not a function response)
        is_user_message = message.role == Role.USER
        is_function_response = False

        # Role.TOOL messages are always function responses (Anthropic provider uses this)
        if message.role == Role.TOOL:
            is_function_response = True
        elif is_user_message and message.parts:
            # Check if it's a function response (has function_response parts)
            is_function_response = any(
                part.function_response is not None
                for part in message.parts
            )

        # Start new turn on user message (not function response)
        if is_user_message and not is_function_response and current_turn_contents:
            # Save current turn
            turns.append(Turn(
                index=turn_index,
                contents=current_turn_contents,
                estimated_tokens=estimate_turn_tokens(current_turn_contents)
            ))
            turn_index += 1
            current_turn_contents = []

        current_turn_contents.append(message)

    # Don't forget the last turn
    if current_turn_contents:
        turns.append(Turn(
            index=turn_index,
            contents=current_turn_contents,
            estimated_tokens=estimate_turn_tokens(current_turn_contents)
        ))

    return turns


def flatten_turns(turns: List[Turn]) -> List[Message]:
    """Flatten a list of turns back into a content list.

    Args:
        turns: List of Turn objects.

    Returns:
        Flattened list of Message objects preserving order.
    """
    result: List[Message] = []
    for turn in turns:
        result.extend(turn.contents)
    return result


# ============================================================
# Media accounting
# ============================================================

#: Bytes of payload per estimated token, by top-level mime type.
#:
#: GC reasoned in tokens and turns and had **no concept of media at all**
#: (#850): ``estimate_message_tokens`` walked ``text`` / ``function_call``
#: / ``function_response`` and never looked at ``inline_data``, so a turn
#: carrying 600 KB of audio was sized at the one-token floor.  A threshold
#: set at 60% cannot fire on a payload the denominator does not contain,
#: and eviction ordering cannot prefer a turn it believes is empty.
#:
#: These rates exist to make that payload VISIBLE, not to reproduce any
#: vendor's billing.  They are order-of-magnitude anchors:
#:
#: - ``audio`` — 24 kHz mono s16le runs at 48 kB/s, and audio-input
#:   pricing across the OpenAI-shaped wires is roughly tens of tokens per
#:   second of speech; 4.8 kB/token places a minute of audio in the
#:   high hundreds of tokens.  An estimate, not a quoted rate: the
#:   framework has no per-model audio-token table and inventing one it
#:   could not keep current would be worse than a stated approximation.
#: - ``image`` / ``application`` (PDF pages are images) — a large image
#:   costs on the order of a thousand tokens; same caveat.
#: - the default — an unclassified blob is charged at its WIRE cost:
#:   base64 is 4/3 of the payload and the framework's own text heuristic
#:   is 4 chars per token, so 3 bytes per token.  Deliberately the
#:   harshest rate: a payload nobody can name is the one whose real cost
#:   is least knowable, and over-counting it makes GC notice rather than
#:   ignore it.
#:
#: A wrong-but-visible number beats an invisible one; the exact rate only
#: shifts WHEN collection fires, never whether the bytes are seen.
MEDIA_BYTES_PER_TOKEN: Dict[str, int] = {
    "audio": 4800,
    "image": 1500,
    "video": 1500,
    "application": 1500,
}

#: Rate for a payload whose mime names no known top-level type.
DEFAULT_MEDIA_BYTES_PER_TOKEN = 3


def inline_data_bytes(inline_data: Optional[Dict[str, Any]]) -> int:
    """Payload size of one ``Part.inline_data``, in bytes.

    Handles both forms the dict is seen in: raw ``bytes`` (in-process,
    what :meth:`JaatoSession._parts_from_user_message` builds) and a
    base64 ``str`` (a part rehydrated from a persisted record), whose
    decoded size is 3/4 of its length.  Returns 0 for anything else.
    """
    if not inline_data:
        return 0
    data = inline_data.get("data")
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    if isinstance(data, str):
        return (len(data) * 3) // 4
    return 0


def estimate_media_tokens(inline_data: Optional[Dict[str, Any]]) -> int:
    """Estimated token cost of one binary part — see :data:`MEDIA_BYTES_PER_TOKEN`.

    Returns 0 for an empty payload so a part carrying nothing is not
    charged for existing.
    """
    num_bytes = inline_data_bytes(inline_data)
    if num_bytes <= 0:
        return 0
    base, _params = split_mime((inline_data or {}).get("mime_type"))
    top = base.split("/", 1)[0]
    rate = MEDIA_BYTES_PER_TOKEN.get(top, DEFAULT_MEDIA_BYTES_PER_TOKEN)
    return max(1, num_bytes // rate)


def attachment_media_view(attachment: Any) -> Dict[str, Any]:
    """An ``inline_data``-shaped view of one :class:`ToolResult` attachment.

    A tool result carries its binary payload as
    :class:`~jaato_sdk.plugins.model_provider.types.Attachment` objects
    (``mime_type`` / ``data`` / ``display_name``) rather than as the
    ``Part.inline_data`` dict, so the two shapes are the SAME three fields
    wearing different clothes.  Rendering the attachment into the dict
    form here means the rate table, the byte count, the marker text and
    the mime match are one implementation for both, instead of a parallel
    set that can disagree about what a megabyte of audio costs.

    The attachment id is **minted from the payload** rather than read off
    a field: :class:`Attachment` has none (it is a public SDK dataclass
    used by every multimodal tool, and widening it is a separate
    decision), and the id is a digest of the bytes — so recomputing it
    here yields exactly the value ingest would have given it.
    """
    data = getattr(attachment, "data", None)
    return {
        "mime_type": getattr(attachment, "mime_type", None),
        "data": data,
        "display_name": getattr(attachment, "display_name", None),
        ATTACHMENT_ID_KEY: mint_attachment_id(data),
    }


def part_media_views(part: Part) -> List[Dict[str, Any]]:
    """Every binary payload one part carries, in ``inline_data`` shape.

    Two sources, which GC previously knew one of (#850, #989):

    * ``Part.inline_data`` — a user-message attachment, the inbound audio
      #850 was filed about.
    * ``Part.function_response.attachments`` — a TOOL RESULT's bytes.
      ``function_response`` appeared in this module only for message
      grouping, so these were sized at the one-token floor and were
      immune to :func:`evict_consumed_media`.  Pre-existing for the
      ``_multimodal`` image tools; it is a clarification answered by
      voice (#989) that makes it recur every turn.
    """
    if part.inline_data:
        return [part.inline_data]
    response = part.function_response
    if response is None:
        return []
    return [
        attachment_media_view(att)
        for att in (getattr(response, "attachments", None) or [])
    ]


def part_media_bytes(part: Part) -> int:
    """Binary payload carried by one part, in bytes (either source)."""
    return sum(inline_data_bytes(view) for view in part_media_views(part))


def part_media_tokens(part: Part) -> int:
    """Estimated token cost of one part's binary payload (either source)."""
    return sum(estimate_media_tokens(view) for view in part_media_views(part))


def message_media_bytes(message: Message) -> int:
    """Total binary payload carried by one message, in bytes.

    Counts both a part's own ``inline_data`` and the attachments of a
    tool result it carries — see :func:`part_media_views`.
    """
    return sum(part_media_bytes(part) for part in (message.parts or []))


def message_media_tokens(message: Message) -> int:
    """Estimated token cost of every binary part in one message.

    A whole-message sum rather than a per-part branch so callers walking
    parts in their own loop -- notably
    ``JaatoSession._update_conversation_budget``, a function already frozen
    at the top of the cyclomatic-complexity baseline -- can account for
    media with a single statement and no additional decision point.
    """
    return sum(part_media_tokens(part) for part in (message.parts or []))


def history_media_bytes(history: Sequence[Message]) -> int:
    """Total ``inline_data`` payload across a whole history, in bytes.

    The raw number behind ``context_usage["media_bytes"]``, which is what
    lets a GC strategy trigger on media PRESSURE — a quantity in bytes,
    kept as bytes rather than laundered through a token estimate, because
    the thing that killed a voice session was request SIZE and an operator
    setting a limit on it thinks in megabytes.
    """
    return sum(message_media_bytes(m) for m in history)


def estimate_message_tokens(message: Message) -> int:
    """Estimate token count for a single Message object.

    Uses a simple heuristic: ~4 characters per token for text, and
    :data:`MEDIA_BYTES_PER_TOKEN` for binary ``inline_data`` parts (which
    were previously counted as nothing at all — see #850).

    Args:
        message: A Message object to estimate.

    Returns:
        Estimated token count.
    """
    total_chars = 0
    media_tokens = 0

    if message.parts:
        for part in message.parts:
            # Text parts
            if part.text:
                total_chars += len(part.text)

            # Function call parts
            elif part.function_call:
                fc = part.function_call
                total_chars += len(fc.name) if fc.name else 0
                if fc.args:
                    # Args is typically a dict, estimate from string repr
                    total_chars += len(str(fc.args))

            # Function response parts
            elif part.function_response:
                fr = part.function_response
                total_chars += len(fr.name) if fr.name else 0
                if fr.result:
                    total_chars += len(str(fr.result))
                # A tool result's own binary payload — an image a tool
                # produced, or the audio a clarification was answered with
                # (#989).  Invisible here until now, exactly as
                # ``inline_data`` was before #850.
                media_tokens += part_media_tokens(part)

            # Binary parts (audio, images, PDFs) — the payload that
            # dominates a voice request and used to be sized at zero.
            elif part.inline_data:
                media_tokens += estimate_media_tokens(part.inline_data)

            # Reasoning a replaying wire sends back on every later request
            # (docs/design/minimax-kimi-mimo-providers.md §3).  It is
            # context in the plainest sense, and a Kimi K3 turn at maximum
            # effort carries tens of thousands of tokens of it — the #850
            # blind spot again, in text, unless it is counted here.
            elif part.thought:
                total_chars += len(part.thought)

    # Rough estimate: 4 chars per token (conservative)
    return max(1, total_chars // 4 + media_tokens)


def estimate_turn_tokens(contents: List[Message]) -> int:
    """Estimate token count for a list of Message objects.

    Args:
        contents: List of Message objects.

    Returns:
        Total estimated token count.
    """
    return sum(estimate_message_tokens(c) for c in contents)


def estimate_history_tokens(history: List[Message]) -> int:
    """Estimate total token count for entire history.

    Args:
        history: Full conversation history.

    Returns:
        Total estimated token count.
    """
    return estimate_turn_tokens(history)


def create_summary_message(summary_text: str) -> Message:
    """Create a Message object containing a context summary.

    The summary is marked with special delimiters so the model
    understands it's compressed context, not a user message.

    Args:
        summary_text: The summary text to include.

    Returns:
        A Message object with role=USER containing the summary.
    """
    formatted_summary = (
        "[Context Summary - Previous conversation compressed]\n"
        f"{summary_text}\n"
        "[End Context Summary]"
    )

    return Message(
        role=Role.USER,
        parts=[Part(text=formatted_summary)]
    )


def create_gc_notification_message(message: str) -> Message:
    """Create a Message object notifying about GC.

    Args:
        message: The notification message.

    Returns:
        A Message object with the notification.
    """
    formatted_message = f"[System: {message}]"

    return Message(
        role=Role.USER,
        parts=[Part(text=formatted_message)]
    )


def ensure_tool_call_integrity(
    history: List[Message],
    trace_fn=None,
) -> List[Message]:
    """Repair tool_use/tool_result pairing in **stored** history after GC.

    GC removes messages individually, which can cut between a MODEL
    message's ``function_call`` parts (tool_use) and the
    ``function_response`` parts that answer them (tool_result) — a pairing
    every provider requires.  The four GC call sites in
    :mod:`shared.jaato_session` install this function's output as the
    session's history, so unlike the per-request boundary repair this one
    **is** destructive by design: it fixes the record once instead of
    hiding the damage on every later request.

    **The policy is** :func:`shared.history_invariant.repair_history`, and
    that is the whole implementation.  One invariant must have one
    definition — this function existing alongside a boundary validator
    with a *different* repair policy is what produced #674's worst case.

    Until #674 this function repaired by DELETION, and pass 2 removed a
    MODEL message whose tool calls were not all answered.  On a
    **partially** answered batch — the routine outcome of cancelling
    8-wide parallel execution — that deletes the assistant turn while its
    real result stays behind, so the function named for the invariant
    produced the violation it exists to prevent::

        MODEL calls {A, B}  ->  TOOL answers A  ->  USER turn

    Pass 1 kept the TOOL message (``A`` is valid); pass 2 reached the USER
    branch with ``B`` still pending and deleted the MODEL message, leaving
    ``A``'s result answering a call present nowhere.  Measured across
    widths 2/3/5/8: every case with at least one real result, 14 of 18.
    A stored history in that state is then **persisted**, so a session
    revived from the record (``SessionManager._load_session``) comes back
    corrupt — which is why repairing only the per-request copy was not
    enough.

    Now an unanswered call is *answered* with a synthetic cancelled result
    and no MODEL message is ever removed, which also preserves the turn's
    ``Part.thought`` — the ``reasoning_content`` ``replay_reasoning``
    providers require back (MiMo answers 400 without it).  The synthetic
    payload is the same ``{"error": "cancelled", ...}`` shape
    :meth:`JaatoSession._inject_synthetic_cancelled_results` already
    writes into stored history for this exact situation.

    **Writing results into the store is safe at all four call sites**
    because each runs between turns, never mid-batch:
    ``_maybe_collect_after_turn`` and ``_maybe_collect_before_send`` sit on
    turn boundaries, ``manual_gc`` is operator-driven, and
    ``_try_gc_for_context_recovery`` pops the trailing MODEL message with
    pending calls *before* calling this and re-appends it after — so the
    calls the session is about to execute are not present to be answered.

    Fully orphaned tool results (a result whose call this history no longer
    contains — the rewind/resume shape) are still dropped: there is no call
    to attach a partner to, and inventing an assistant turn would put words
    in the model's mouth.

    Args:
        history: Conversation history, possibly with broken pairs.
        trace_fn: Optional trace sink, ``(str) -> None``.  Receives one
            ``HISTORY_INVARIANT: ...`` line per repair kind that fired;
            the GC call sites prefix it with their own phase name.

    Returns:
        A history satisfying the pairing invariant — ``history`` itself
        when nothing needed repair.

    See also:
        :func:`shared.history_invariant.validate_history` for the defects
        this repairs, and :meth:`JaatoSession._history_for_provider` for
        the per-request backstop that catches the producers this function
        does not see (cancellation, rewind, subagent history sharing, a
        wire that streams no tool-call id).
    """
    return repair_history(history, trace_fn=trace_fn)


def get_preserved_indices(
    total_turns: int,
    preserve_recent: int,
    pinned_indices: Optional[List[int]] = None
) -> set:
    """Calculate which turn indices should be preserved.

    Args:
        total_turns: Total number of turns in history.
        preserve_recent: Number of recent turns to preserve.
        pinned_indices: Additional indices to preserve.

    Returns:
        Set of turn indices that should not be collected.
    """
    preserved = set()

    # Always preserve recent turns
    if preserve_recent > 0:
        start_recent = max(0, total_turns - preserve_recent)
        preserved.update(range(start_recent, total_turns))

    # Add pinned indices
    if pinned_indices:
        for idx in pinned_indices:
            if 0 <= idx < total_turns:
                preserved.add(idx)

    return preserved


# ============================================================
# Identical tool-result dedup
# ============================================================


def _tool_result_payload(tr: ToolResult) -> str:
    """Canonical string form of a tool result's payload for hashing/sizing.

    Uses sorted-key JSON so equal dicts hash equal regardless of key
    order; falls back to ``repr`` for non-JSON-serializable payloads.
    """
    try:
        return json.dumps(tr.result, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(tr.result)


def _tool_result_signature(tr: ToolResult) -> str:
    """Stable hash of ``(name, result)`` — the dedup grouping key.

    Two function-response parts are duplicates iff the SAME tool returned
    the SAME payload (e.g. a model re-calling ``listAvailableTemplates``
    and getting the identical catalog back).
    """
    payload = _tool_result_payload(tr)
    raw = f"{tr.name or ''}\x00{payload}".encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()


def _elided_result(tr: ToolResult) -> ToolResult:
    """Return a copy of ``tr`` with its payload replaced by a tiny marker.

    Keeps the result a dict (stable downstream shape) and preserves
    ``call_id``/``name`` so tool_call↔tool_result pairing is intact —
    only the redundant body is dropped.
    """
    marker = {
        "_gc_deduped": True,
        "note": (
            f"Identical to a later '{tr.name}' result — body elided by GC "
            f"to reclaim context (the duplicate carried the same payload)."
        ),
    }
    return _dc_replace(tr, result=marker)


def dedup_identical_tool_results(
    history: List[Message],
    min_payload_chars: int = 200,
) -> Tuple[List[Message], int, List[str]]:
    """Collapse byte-identical tool-result payloads in ``history``.

    A model that re-invokes the same tool and gets the identical output
    back (the qwen3 ``listAvailableTemplates`` re-call pathology) bloats
    the wire with redundant copies that GC eviction can't reclaim when
    they sit in the recency-preserved window.  This **shrinks** each
    earlier duplicate's result body to a compact marker WITHOUT removing
    the message — so recency, message structure, and tool_call/tool_result
    pairing are all preserved, and only EXACT duplicates are touched
    (zero data loss: the surviving copy holds the full payload).

    Strategy: group function-response parts by ``(name, result)``
    signature; for each group with >1 member, keep the MOST RECENT
    occurrence intact (it's recency-protected, so it survives any later
    eviction) and elide all earlier ones.  Groups whose payload is below
    ``min_payload_chars`` are skipped (not worth eliding).

    Returns ``(new_history, chars_reclaimed, elided_message_ids)``.
    ``history`` is returned unchanged (same object) and the id list empty
    when nothing is deduped.  Only the messages that actually change are
    copied; the rest are shared by reference.  ``elided_message_ids`` lets
    the caller invalidate any per-message token cache so the budget
    re-sync recomputes the shrunken sizes (message_id is preserved across
    the shrink, so a cache keyed on it would otherwise return the stale
    pre-dedup count).
    """
    # 1) Group occurrences by signature: sig -> [(msg_idx, part_idx, payload_len)]
    groups: Dict[str, List[Tuple[int, int, int]]] = {}
    for mi, msg in enumerate(history):
        for pi, part in enumerate(msg.parts or []):
            tr = part.function_response
            if tr is None or tr.result is None:
                continue
            payload_len = len(_tool_result_payload(tr))
            if payload_len < min_payload_chars:
                continue
            groups.setdefault(_tool_result_signature(tr), []).append(
                (mi, pi, payload_len)
            )

    # 2) For each duplicated group, mark all-but-last for elision.
    #    to_elide: msg_idx -> set(part_idx)
    to_elide: Dict[int, set] = {}
    chars_reclaimed = 0
    for occ in groups.values():
        if len(occ) < 2:
            continue
        for (mi, pi, payload_len) in occ[:-1]:  # keep occ[-1] (most recent)
            to_elide.setdefault(mi, set()).add(pi)
            chars_reclaimed += payload_len

    if not to_elide:
        return history, 0, []

    # 3) Rebuild history, copying only the touched messages.
    new_history: List[Message] = []
    elided_message_ids: List[str] = []
    for mi, msg in enumerate(history):
        elide_parts = to_elide.get(mi)
        if not elide_parts:
            new_history.append(msg)
            continue
        new_parts = []
        for pi, part in enumerate(msg.parts):
            if pi in elide_parts and part.function_response is not None:
                new_parts.append(
                    _dc_replace(part, function_response=_elided_result(part.function_response))
                )
            else:
                new_parts.append(part)
        new_history.append(_dc_replace(msg, parts=new_parts))
        elided_message_ids.append(msg.message_id)

    return new_history, chars_reclaimed, elided_message_ids


# ============================================================
# Consumed-media eviction
# ============================================================

#: Mime prefixes evicted from history once the turn that consumed them
#: has completed.  Audio only, by default, and the asymmetry is the
#: argument: a model handed the same recording a second time re-hears it
#: exactly as it did the first, so the bytes buy nothing after the turn
#: that answered them — whereas an image is routinely re-examined ("what
#: does the third column say?") and a PDF is a document the conversation
#: keeps referring back to.  Widen it per session via
#: ``GCConfig.media_evict_mime_prefixes`` when that trade differs.
DEFAULT_MEDIA_EVICT_MIME_PREFIXES: Tuple[str, ...] = ("audio/",)


def _evicted_media_marker(
    inline_data: Dict[str, Any],
    attachment_id: Optional[str],
) -> str:
    """The text that stands in for evicted bytes.

    **Naming the id is the load-bearing part.**  A marker that only says
    "audio was here" turns a purge into a deletion, and in a domain with
    call-retention duties "the agent handled a claim from audio nobody can
    produce" is precisely the audit finding this is written to avoid.  With
    the id present, a transcript alone locates the archived recording — and
    because the id is a digest of the payload, the match can be verified
    from the recording rather than trusted from a mapping.

    An attachment that reached ingest with an undecodable payload has no
    id (see :func:`~jaato_sdk.media_identity.mint_attachment_id`); the
    marker then says so rather than inventing one, because a fabricated id
    is worse than an absent one.
    """
    mime = inline_data.get("mime_type")
    num_bytes = inline_data_bytes(inline_data)
    duration = audio_duration_seconds(mime, inline_data.get("data"))
    fields = [inline_data.get("display_name") or None,
              describe_attachment(mime, num_bytes, duration),
              f"ref {attachment_id}" if attachment_id else "no ref (payload "
              "did not decode at ingest)"]
    return (
        "[Media evicted after the turn that consumed it — "
        + ", ".join(f for f in fields if f)
        + ". The recording is not discarded by this; cite the ref to locate "
          "the archived original.]"
    )


def evict_consumed_media(
    history: List[Message],
    mime_prefixes: Iterable[str] = DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
    protect_last_messages: int = 0,
) -> Tuple[List[Message], int, List[str]]:
    """Replace consumed binary payloads with a traceable text marker.

    Covers both sources :func:`part_media_views` knows: a ``Part``'s own
    ``inline_data`` (swapped for a text part carrying the marker) and a
    tool result's ``attachments`` (dropped, with the marker appended to
    that result's ``model_suffix`` — see :func:`_evict_tool_result_media`).

    Inbound media was the one direction with no lifecycle at all (#850).
    Outbound already did the right thing — model media is ``CLIENT``
    audience so it never enters history, and ``ensure_spoken_part`` leaves
    the *transcript* in its place — while every utterance a session had
    HEARD stayed in history and was re-sent on every later request.  Five
    questions on one helpdesk call measured ~2.8 MB of audio, all of it on
    the wire of the last request and growing with every turn.

    This is shape **A** of the two the issue names: evict the bytes, keep a
    marker carrying the id.  It is deliberately not shape B (substitute a
    transcript), which is higher fidelity and needs a transcript source
    that does not exist for inbound audio on chat-completions today.  They
    compose — B where a transcript exists, A as the floor — and both need
    the ingest id this marker prints.

    **Destructive, unlike the modality gate.**
    :meth:`JaatoSession._gate_history_for_active_modalities` filters a
    *copy* per request precisely so a later ``enter_tier`` back to a voice
    model can still hear.  This one rewrites the stored history, and the
    model genuinely cannot re-listen afterwards.  That is the accepted
    trade: re-hearing a recording yields the same understanding it yielded
    the first time, and the conversation already carries that
    understanding in words.

    Args:
        history: Conversation history, newest last.
        mime_prefixes: Mimes to evict, matched case-insensitively as
            prefixes (``("audio/",)`` by default).
        protect_last_messages: Trailing messages left untouched.  0 is
            correct at the standard call site — eviction runs at the START
            of a turn, so everything in history belongs to a turn that has
            already completed — and exists for callers that run it
            elsewhere.

    Returns:
        ``(new_history, bytes_reclaimed, evicted_message_ids)``.  When
        nothing matches, ``history`` itself is returned (same object) with
        ``0`` and an empty list, so a text-only session allocates nothing.
        Only the messages that change are copied; the rest are shared by
        reference.  The id list lets the caller invalidate a per-message
        token cache keyed on ``message_id``, which the rewrite preserves.
    """
    prefixes = tuple(p.lower() for p in mime_prefixes if p)
    if not prefixes or not history:
        return history, 0, []

    cutoff = len(history) - max(0, protect_last_messages)

    new_history: List[Message] = []
    evicted_ids: List[str] = []
    bytes_reclaimed = 0

    for index, msg in enumerate(history):
        if index >= cutoff:
            new_history.append(msg)
            continue
        new_parts, reclaimed = _evict_parts_media(msg.parts or [], prefixes)
        if new_parts is None:
            new_history.append(msg)
            continue
        bytes_reclaimed += reclaimed
        new_history.append(_dc_replace(msg, parts=new_parts))
        evicted_ids.append(msg.message_id)

    if not evicted_ids:
        return history, 0, []
    return new_history, bytes_reclaimed, evicted_ids


def _media_matches(view: Dict[str, Any], prefixes: Tuple[str, ...]) -> bool:
    """Whether one ``inline_data``-shaped payload is evictable."""
    base, _params = split_mime(view.get("mime_type"))
    return base.startswith(prefixes) and inline_data_bytes(view) > 0


def _evict_parts_media(
    parts: Sequence[Part],
    prefixes: Tuple[str, ...],
) -> Tuple[Optional[List[Part]], int]:
    """Rewrite one message's parts, dropping evictable binary payloads.

    Returns ``(None, 0)`` when nothing in this message matches, so the
    caller shares the original message by reference instead of copying it.
    """
    new_parts: List[Part] = []
    reclaimed = 0
    for part in parts:
        if part.inline_data and _media_matches(part.inline_data, prefixes):
            inline = part.inline_data
            reclaimed += inline_data_bytes(inline)
            new_parts.append(Part.from_text(_evicted_media_marker(
                inline, inline.get(ATTACHMENT_ID_KEY),
            )))
            continue
        replacement, freed = _evict_tool_result_media(
            part.function_response, prefixes,
        )
        if replacement is None:
            new_parts.append(part)
            continue
        reclaimed += freed
        # ``_dc_replace`` rather than ``Part.from_function_response``:
        # swap only the field that changed, so anything else the part
        # carries survives the rewrite.
        new_parts.append(_dc_replace(part, function_response=replacement))
    return (new_parts, reclaimed) if reclaimed else (None, 0)


def _evict_tool_result_media(
    response: Optional[ToolResult],
    prefixes: Tuple[str, ...],
) -> Tuple[Optional[ToolResult], int]:
    """Strip evictable attachments off one tool result.

    A tool result's bytes live in ``ToolResult.attachments``, not in a
    ``Part``, so eviction cannot simply swap the part for a text one — the
    call id, the name and the structured ``result`` the model answered from
    all have to survive.  The attachment is dropped and its marker appended
    to ``model_suffix``, which is the existing model-facing-only channel
    (:func:`render_result_for_model`): the structured ``result`` stays the
    source of truth every non-model consumer reads — the tool-call ledger,
    completion processors, enrichment — while the model still learns that
    a recording was here and what its id was.

    Returns ``(None, 0)`` when nothing matched, so the caller keeps the
    original part.
    """
    attachments = list(getattr(response, "attachments", None) or [])
    if not attachments:
        return None, 0
    kept: List[Any] = []
    markers: List[str] = []
    reclaimed = 0
    for attachment in attachments:
        view = attachment_media_view(attachment)
        if not _media_matches(view, prefixes):
            kept.append(attachment)
            continue
        reclaimed += inline_data_bytes(view)
        markers.append(_evicted_media_marker(view, view.get(ATTACHMENT_ID_KEY)))
    if not markers:
        return None, 0
    suffix = "\n".join(
        [s for s in [response.model_suffix] if s] + markers
    )
    return _dc_replace(
        response,
        attachments=kept or None,
        model_suffix=suffix,
    ), reclaimed


def media_pressure_reason(
    context_usage: Dict[str, Any],
    config: "GCConfig",
) -> Optional["GCTriggerReason"]:
    """``MEDIA_PRESSURE`` when history carries too much binary payload.

    The one media question every strategy asks, asked in one place so all
    four give the same answer.  Each ``should_collect`` consults this
    ALONGSIDE its percentage check rather than instead of it: the two
    denominators are independent, and a session can be well under its token
    threshold while carrying megabytes of audio — which is precisely the
    state #850 measured and the state no strategy could see.

    Returns ``None`` when the check is disabled (``media_bytes_threshold``
    of 0), when the session reports no ``media_bytes`` (a client or test
    harness that predates the key — absence must not be read as zero-and-
    fine or as pressure), or when the payload is under the ceiling.
    """
    from .base import GCTriggerReason
    threshold = getattr(config, "media_bytes_threshold", 0) or 0
    if threshold <= 0:
        return None
    media_bytes = context_usage.get("media_bytes")
    if not isinstance(media_bytes, int):
        return None
    if media_bytes < threshold:
        return None
    return GCTriggerReason.MEDIA_PRESSURE
