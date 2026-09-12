"""Attachments on a clarification ANSWER (#989).

A clarification answer is a **tool result**, not a user message.  That one
fact decides everything here: the bytes ride
:attr:`~jaato_sdk.plugins.model_provider.types.ToolResult.attachments`,
where every converter family already marshals them (``image_url`` /
``input_audio`` / a Gemini ``Blob``) and where the modality gate already
runs — rather than being ferried in as a separate user turn the user never
typed.

Three jobs live here, deliberately apart from :mod:`.models`, because the
DAEMON runs the first of them and never imports the plugin:

1. **wire ↔ object** — ``{mime_type, data: base64, display_name,
   attachment_id}`` is the same shape ``send_message(attachments=...)``
   uses, so a client that can attach to a message can attach to an answer
   with no second encoder.
2. **validation at submit** (:func:`validate_answer_attachments`), which is
   where a bad batch has somewhere to be reported.
3. **the model-facing description** (:func:`describe_wire_attachment`) — the
   mime, the name and the ingest id beside the answer they belong to, never
   the payload.

WHY A BYTE CAP, AND WHY IT REFUSES RATHER THAN TRUNCATES.  The answers
cross daemon→runner as the *response* to ``client.request_clarification``,
under the 10 MB ``MAX_MESSAGE_SIZE``.  An oversized response is not written
at all (`RunnerRPCClient._write_frame_json` raises `FrameTooLargeError`,
which is logged and swallowed), so the runner's call **never resolves** and
the turn hangs behind a clarification nobody can answer.  A 120 s utterance
is 3.84 MB raw / ~5.12 MB base64: one fits, two do not.  So the cap is
enforced at submit, in bytes of decoded payload, with the future left
unresolved — the client is told the number and can retry with less.  What
the right multi-question voice policy is (per-question delivery, a
per-answer cap, chunking) is still open; failing loudly is what keeps that
question open instead of answering it with a hang.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Sequence, Tuple

from jaato_sdk.media_identity import (
    ATTACHMENT_ID_KEY,
    audio_duration_seconds,
    describe_attachment,
    ensure_attachment_id,
)
from jaato_sdk.plugins.model_provider.types import Attachment

#: Total decoded payload one clarification batch may carry, in bytes.
#:
#: 6 MiB raw is ~8 MiB base64, which leaves headroom under the 10 MB
#: ``shared.framing.MAX_MESSAGE_SIZE`` for the rest of the RPC frame.  One
#: full-length utterance fits; two do not, and that is reported rather than
#: silently trimmed — see the module docstring.
MAX_CLARIFICATION_ATTACHMENT_BYTES = 6 * 1024 * 1024


def wire_attachment_bytes(entry: Dict[str, Any]) -> Optional[bytes]:
    """Decode one wire attachment's payload, or ``None`` if it cannot be.

    Accepts the two forms the canonical dict is seen in: raw ``bytes``
    (an in-process caller) and a base64 ``str`` (the wire).  ``None``
    means the entry is unusable, which callers report rather than repair.
    """
    data = entry.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except Exception:  # noqa: BLE001 — any decode failure is "unusable"
            return None
    return None


def attachment_to_wire(attachment: Attachment) -> Dict[str, Any]:
    """Render an :class:`Attachment` as the canonical wire dict."""
    return ensure_attachment_id({
        "mime_type": attachment.mime_type,
        "data": base64.b64encode(attachment.data or b"").decode("ascii"),
        "display_name": attachment.display_name,
    })


def attachment_from_wire(entry: Dict[str, Any]) -> Optional[Attachment]:
    """Build an :class:`Attachment` from one wire dict, or ``None``.

    ``None`` for an entry with no mime or an undecodable payload — the
    same "unusable, do not repair" rule :func:`wire_attachment_bytes`
    states.  The id is NOT carried onto the object: ``Attachment`` has no
    field for it, and it is a digest of the payload, so any consumer that
    needs it recomputes it (``mint_attachment_id``) and gets the same
    value.
    """
    raw = wire_attachment_bytes(entry)
    mime = (entry.get("mime_type") or "").strip()
    if raw is None or not mime:
        return None
    return Attachment(
        mime_type=mime,
        data=raw,
        display_name=entry.get("display_name") or None,
    )


def attachments_from_wire(
    entries: Sequence[Dict[str, Any]],
) -> List[Attachment]:
    """Build the usable :class:`Attachment` objects from wire dicts.

    Unusable entries are dropped here because the daemon already refused
    them at submit (:func:`validate_answer_attachments`); anything that
    reaches this point and still will not decode is a transport fault, and
    inventing an empty attachment for it would be worse than carrying one
    fewer.
    """
    out: List[Attachment] = []
    for entry in entries or []:
        attachment = attachment_from_wire(entry)
        if attachment is not None:
            out.append(attachment)
    return out


def describe_wire_attachment(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The model-facing descriptor for one attachment — never the payload.

    #845's rule 1 transfers here as **labelling, not defanging**: today's
    typed answer goes into the tool result verbatim with no wrapper, so
    wrapping only the spoken one would weigh a voice answer differently
    from the identical typed one — the exact asymmetry ``_wrap_wake_content``
    exists to avoid, inverted.  What the model does need is to know that an
    answer arrived as audio, which question it belongs to, and the id under
    which the recording is archived.
    """
    raw = wire_attachment_bytes(entry) or b""
    mime = entry.get("mime_type")
    descriptor: Dict[str, Any] = {
        "mime_type": mime,
        "description": describe_attachment(
            mime, len(raw), audio_duration_seconds(mime, raw),
        ),
    }
    if entry.get("display_name"):
        descriptor["display_name"] = entry["display_name"]
    if entry.get(ATTACHMENT_ID_KEY):
        descriptor[ATTACHMENT_ID_KEY] = entry[ATTACHMENT_ID_KEY]
    return descriptor


def validate_answer_attachments(
    raw: Any,
    question_count: int,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[str]]:
    """Validate a submitted ``answer_attachments`` map; normalise the keys.

    Returns ``(normalised, errors)`` where *normalised* maps a 1-based
    question index to its list of canonical wire dicts (id back-filled),
    and *errors* is every problem found — all of them, so a client fixes
    one round trip rather than N.  A non-empty *errors* means the caller
    must NOT resolve the clarification: leaving the future pending is what
    lets the same request be answered again.

    Checked here, and only here, is what the daemon can actually know:
    the question exists, the payload decodes, and the batch fits the
    frame.  Whether the ACTIVE MODEL can consume a given mime is not on
    that list — the daemon does not hold the runner's provider, and the
    tier can change between the question and the answer (#847), so a
    submit-time guess would be wrong in both directions.  That question is
    answered authoritatively by ``JaatoSession._gate_one_tool_result``,
    against the model the bytes are about to reach.
    """
    errors: List[str] = []
    normalised: Dict[int, List[Dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        return {}, [f"answer_attachments must be an object, got "
                    f"{type(raw).__name__}"]

    total_bytes = 0
    for key, entries in raw.items():
        index = _as_question_index(key)
        if index is None or not (1 <= index <= question_count):
            errors.append(
                f"answer_attachments key {key!r} is not a question index in "
                f"1..{question_count}"
            )
            continue
        if not isinstance(entries, (list, tuple)):
            errors.append(
                f"answer_attachments[{index}] must be a list of attachments"
            )
            continue
        cleaned: List[Dict[str, Any]] = []
        for position, entry in enumerate(entries, 1):
            payload = _validate_entry(entry, index, position, errors)
            if payload is None:
                continue
            total_bytes += len(wire_attachment_bytes(payload) or b"")
            cleaned.append(payload)
        if cleaned:
            normalised[index] = cleaned

    if total_bytes > MAX_CLARIFICATION_ATTACHMENT_BYTES:
        errors.append(
            f"clarification attachments total {total_bytes} bytes, over the "
            f"{MAX_CLARIFICATION_ATTACHMENT_BYTES}-byte cap for one batch "
            f"(the answers cross the runner RPC in a single frame). Answer "
            f"fewer questions with media per submission, or send shorter "
            f"recordings."
        )
    if errors:
        return {}, errors
    return normalised, []


def _as_question_index(key: Any) -> Optional[int]:
    """A 1-based question index from a map key, or ``None``.

    JSON object keys are strings, so the wire always delivers ``"1"``;
    an in-process caller may pass ``1``.  Both mean the same question.
    """
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        try:
            return int(key.strip())
        except ValueError:
            return None
    return None


def _validate_entry(
    entry: Any,
    index: int,
    position: int,
    errors: List[str],
) -> Optional[Dict[str, Any]]:
    """Validate one attachment entry, appending any problem to *errors*."""
    where = f"answer_attachments[{index}][{position}]"
    if not isinstance(entry, dict):
        errors.append(f"{where} must be an object with mime_type and data")
        return None
    if not (entry.get("mime_type") or "").strip():
        errors.append(f"{where} is missing mime_type")
        return None
    if wire_attachment_bytes(entry) is None:
        errors.append(
            f"{where} carries no decodable payload (data must be base64 or "
            f"bytes)"
        )
        return None
    return ensure_attachment_id(dict(entry))
