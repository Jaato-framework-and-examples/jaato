"""Inbound binary attachments on OpenAI-shaped wires — shared machinery.

The inbound counterpart to ``_media_deltas``: a plain module, not a base
class, so a provider opts in by *importing* rather than by *inheriting*.
Same reasoning as its sibling — the providers that speak OpenAI's
chat-completions format are not the same set as the providers that
inherit ``_openai_compat.OpenAICompatProvider`` (``openrouter`` speaks the
format and inherits nothing), so machinery parked on that base class is
unreachable for half the fleet that needs it.

**Why this module exists.**  ``_openai_compat/converters.py`` used to
marshal *every* ``inline_data`` part on a user message into an
``image_url`` block, defaulting a missing mime to ``image/png`` (#829).  A
PDF became ``image_url`` with ``data:application/pdf;base64,...``; audio
became ``image_url`` with ``data:audio/wav;...``; a part with no mime was
*asserted* to be a PNG.  Nothing was refused, logged, or stripped — the
bytes reached the wire mislabelled, which is the one outcome worse than
either carrying them or declining them.

The correct shape already existed one directory over, in
``openrouter/converters.py``: dispatch on the mime, and return ``None``
for anything the wire cannot carry.  Two converters, one wire family,
opposite policies.  This module is that dispatch, lifted so both call it.

**The policy is per-wire, and it is declared, not assumed.**  Two axes
today, one per non-image mime family:

``pdf_as_file`` — OpenRouter has a PDF-input extension (a ``file``
content block, parsed by the upstream model or by OpenRouter's
``file-parser`` plugin) and declares ``pdf_input=True``; every
``_openai_compat`` sharer (nim, vllm, lmstudio, tensorrt_llm,
zhipuai_openai, triton, nebius, ovhcloud, doubleword) declares
``pdf_input=False``, so for them a PDF is content the wire does not carry
and is withheld.  The capability declaration and the converter now agree
— before this module they contradicted each other.

``audio_as_input_audio`` — the inbound half of #830.  The framework grew
a mouth in #824/#828 (model-emitted audio reaches a client and plays) and
had no ears: ``input_audio``, OpenAI's content-block form for audio
*input*, appeared nowhere in the tree, so every ``audio/*`` part was
withheld by the clause above no matter how loudly the model's catalog
entry declared ``audio`` as an input modality.  OpenRouter carries audio
input and declares ``audio_input=True``; the ``_openai_compat`` sharers
declare ``False`` until someone verifies one of them on the wire.  The
gate that decides whether audio *may* be sent already existed on both the
tier (``_validate_modality_tier_capabilities``) and the tool-result
(``_gate_one_tool_result``) paths — what was missing was somewhere for it
to send the bytes to.

**Withholding is visible.**  ``withheld_attachment_note`` renders the
model-facing note in the same ``[Attachment withheld: ...]`` shape the
session's modality gate uses (``JaatoSession._build_withheld_attachment_note``),
so a model that is told "here is a document" and receives nothing can at
least tell that something was dropped rather than confabulating over the
silence.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# One entry as the callers hold it: (mime, data, filename).  ``data`` is
# raw ``bytes`` or an already-base64-encoded ``str``; ``filename`` is the
# attachment's display name, used only by the PDF ``file`` block.
AttachmentEntry = Tuple[str, Any, Optional[str]]

_UNKNOWN_MIME = "unknown type"

# Container mime -> the token OpenAI's ``input_audio`` block wants in its
# ``format`` field.  The vocabulary is CLOSED and belongs to the wire, not
# to us: ``wav``, ``mp3``, ``aiff``, ``aac``, ``ogg``, ``flac``, ``m4a``,
# ``pcm16``, ``pcm24`` (OpenRouter's documented set; OpenAI's own endpoint
# accepts the first two, and an upstream may accept fewer than its gateway
# lists — that refusal is the upstream's to make, not ours to pre-empt).
# We map onto a SUBSET: ``pcm24`` has no mime that unambiguously means it,
# and inventing one would be guessing.  A mime with no entry here is
# withheld — sending ``format: "opus"`` because the mime said so would be
# #829 in a new costume.
_AUDIO_CONTAINER_FORMATS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/vnd.wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/x-mp3": "mp3",
    "audio/mpga": "mp3",
    "audio/aiff": "aiff",
    "audio/x-aiff": "aiff",
    "audio/aac": "aac",
    "audio/x-aac": "aac",
    "audio/aacp": "aac",
    "audio/ogg": "ogg",
    "audio/x-ogg": "ogg",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
}

# Raw PCM has no container, so everything that distinguishes one PCM
# stream from another lives in the mime PARAMETERS — and ``pcm16`` is not
# a format so much as an assertion about them: 16-bit signed
# little-endian, mono, 24 kHz.  That is exactly what the framework itself
# EMITS (``_media_deltas.STREAM_AUDIO_MIME``), so the ears accept what the
# mouth produces.  A parameter that CONTRADICTS the assertion withholds;
# an absent parameter is taken as agreement, because a bare ``audio/pcm``
# in this ecosystem means pcm16 and refusing it would reject the common
# case to guard against a hypothetical one.
#
# ``audio/L16`` is deliberately absent: RFC 2586 defines it as
# BIG-endian, and relabelling big-endian samples ``pcm16`` yields noise,
# not audio.  Transcode it to WAV before sending.
_PCM_MIMES = frozenset({"audio/pcm", "audio/x-pcm"})
_PCM16_PARAMS = {"encoding": "s16le", "channels": "1", "rate": "24000"}


def _split_mime(mime: str) -> Tuple[str, Dict[str, str]]:
    """``"audio/pcm;rate=24000"`` -> ``("audio/pcm", {"rate": "24000"})``.

    Lower-cases both halves and tolerates a parameter with no ``=``, which
    a hand-built mime string occasionally carries.
    """
    head, _, tail = (mime or "").partition(";")
    params: Dict[str, str] = {}
    for chunk in tail.split(";"):
        key, sep, value = chunk.partition("=")
        if sep:
            params[key.strip().lower()] = value.strip().strip('"').lower()
    return head.strip().lower(), params


def audio_wire_format(mime: str) -> Optional[str]:
    """The ``input_audio.format`` token for ``mime``, or ``None``.

    ``None`` means *this audio cannot be described in the wire's
    vocabulary* — an unmapped container, or raw PCM whose parameters
    contradict what ``pcm16`` means.  It is the same refusal
    :func:`attachment_content_block` makes for a mime the wire does not
    carry at all, and it reaches the model through the same note.
    """
    base, params = _split_mime(mime)
    container = _AUDIO_CONTAINER_FORMATS.get(base)
    if container is not None:
        return container
    if base in _PCM_MIMES:
        if all(params.get(k, want) == want for k, want in _PCM16_PARAMS.items()):
            return "pcm16"
    return None


def _b64(data: Any) -> str:
    """Base64 payload for ``data``, whatever form the caller holds it in."""
    if isinstance(data, (bytes, bytearray)):
        return base64.b64encode(bytes(data)).decode("utf-8")
    return data if isinstance(data, str) else ""


def attachment_content_block(
    mime: str,
    data: Any,
    filename: Optional[str] = None,
    *,
    pdf_as_file: bool = False,
    audio_as_input_audio: bool = False,
) -> Optional[Dict[str, Any]]:
    """OpenAI content block for one binary attachment, or ``None``.

    ``image/*`` becomes an ``image_url`` data-URL block — the one shape
    every OpenAI-compatible wire in this tree carries.  ``application/pdf``
    becomes a ``file`` block **only** when the caller passes
    ``pdf_as_file=True``, which is OpenRouter's PDF-input extension and is
    not part of the base OpenAI chat format.  ``audio/*`` becomes an
    ``input_audio`` block **only** when the caller passes
    ``audio_as_input_audio=True`` *and* the mime maps to a token in the
    wire's closed ``format`` vocabulary (:func:`audio_wire_format`); a
    wire that carries audio still cannot carry every container, and the
    two refusals read the same to the model.

    Every other mime — video, ``application/octet-stream``, and the empty
    string a part with no declared mime carries — returns ``None``.
    ``None`` means *this wire does not carry this content*; it never means
    "send it as an image and hope".  Callers pair it with
    :func:`withheld_attachment_note` so the drop is stated rather than
    silent.

    Args:
        mime: The attachment's declared mime type.  Falsy (absent) is a
            withhold, not a PNG — guessing is what #829 was.
        data: Raw ``bytes`` or an already-base64-encoded ``str``.
        filename: Display name; used as the PDF block's ``filename``.
        pdf_as_file: Whether this wire carries PDFs as ``file`` blocks.
        audio_as_input_audio: Whether this wire carries audio INPUT as
            ``input_audio`` blocks (#830).

    Returns:
        The content-block dict, or ``None`` if the wire cannot carry it.
    """
    mime = mime or ""
    if mime.startswith("image/"):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{_b64(data)}"},
        }
    if pdf_as_file and mime == "application/pdf":
        return {
            "type": "file",
            "file": {
                "filename": filename or "document.pdf",
                "file_data": f"data:application/pdf;base64,{_b64(data)}",
            },
        }
    if audio_as_input_audio and mime.lower().startswith("audio/"):
        wire_format = audio_wire_format(mime)
        if wire_format is not None:
            return {
                "type": "input_audio",
                "input_audio": {"data": _b64(data), "format": wire_format},
            }
    return None


def _log_withheld(
    withheld: Sequence[str],
    pdf_as_file: bool,
    audio_as_input_audio: bool = False,
) -> None:
    """WARNING-log a withhold, so an operator sees what never left the box.

    Both policy flags are named in the message because "audio was
    withheld" reads very differently on a wire that carries no audio at
    all and on one that carries audio but not *this container* — the
    first is a provider choice, the second a transcode away from working.
    """
    if withheld:
        logger.warning(
            "Attachment(s) withheld — this wire does not carry %s "
            "(pdf_as_file=%s, audio_as_input_audio=%s). Mislabelling them "
            "as images is the bug this replaces (#829).",
            sorted(set(withheld)), pdf_as_file, audio_as_input_audio,
        )


def marshal_attachments(
    entries: Iterable[AttachmentEntry],
    *,
    pdf_as_file: bool = False,
    audio_as_input_audio: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split attachments into wire blocks and withheld mimes.

    Args:
        entries: ``(mime, data, filename)`` triples.
        pdf_as_file: Passed through to :func:`attachment_content_block`.
        audio_as_input_audio: Passed through to
            :func:`attachment_content_block`.

    Returns:
        ``(blocks, withheld)`` — the content blocks that reached the wire,
        and the mime of each attachment that did not (in encounter order,
        duplicates kept so the note can count them).
    """
    blocks: List[Dict[str, Any]] = []
    withheld: List[str] = []
    for mime, data, filename in entries:
        block = attachment_content_block(
            mime, data, filename,
            pdf_as_file=pdf_as_file,
            audio_as_input_audio=audio_as_input_audio,
        )
        if block is not None:
            blocks.append(block)
        else:
            withheld.append(mime or _UNKNOWN_MIME)
    _log_withheld(withheld, pdf_as_file, audio_as_input_audio)
    return blocks, withheld


def _accepted_forms(pdf_as_file: bool, audio_as_input_audio: bool) -> str:
    """Prose list of what THIS wire does accept, for the withheld note.

    The note used to say "text, or an image" unconditionally, which was
    already wrong for OpenRouter (it carries PDFs) and is wronger now that
    a wire may carry audio.  Telling a model to retry in a form the wire
    also refuses wastes a turn; telling it the form is unavailable when it
    is not wastes the capability.
    """
    forms = ["text", "images"]
    if pdf_as_file:
        forms.append("PDF documents")
    if audio_as_input_audio:
        forms.append("audio (" + "/".join(
            sorted(set(_AUDIO_CONTAINER_FORMATS.values()) | {"pcm16"})) + ")")
    return ", ".join(forms[:-1]) + f", or {forms[-1]}"


def withheld_attachment_note(
    withheld: Sequence[str],
    *,
    pdf_as_file: bool = False,
    audio_as_input_audio: bool = False,
) -> Optional[str]:
    """Model-facing note for attachments this wire could not carry.

    Mirrors the ``[Attachment withheld: ...]`` shape the session's
    modality gate produces, so the two sources of withholding read the
    same to the model.  Returns ``None`` when nothing was withheld, so a
    caller can append unconditionally.

    The note names the mimes and counts them, because "a document was
    dropped" and "four documents were dropped" lead to different
    follow-up questions.  It also names what this particular wire DOES
    accept (:func:`_accepted_forms`), so the remedy it suggests is one
    that would actually work here — the two policy flags default to the
    conservative wire (images only) so an existing caller's note is
    unchanged.

    Args:
        withheld: Mimes that did not reach the wire, duplicates kept.
        pdf_as_file: Whether this wire carries PDFs.
        audio_as_input_audio: Whether this wire carries audio input.
    """
    if not withheld:
        return None
    counts: Dict[str, int] = {}
    for mime in withheld:
        counts[mime] = counts.get(mime, 0) + 1
    kinds = ", ".join(
        f"{mime} (x{n})" if n > 1 else mime for mime, n in sorted(counts.items())
    )
    return (
        f"[Attachment withheld: this provider's API cannot carry {kinds} "
        f"content, so it was not sent.  Ask for the content in a form this "
        f"wire accepts ({_accepted_forms(pdf_as_file, audio_as_input_audio)}), "
        f"or switch to a provider that declares support for it.]"
    )


def attachment_entries_from_parts(parts: Iterable[Any]) -> List[AttachmentEntry]:
    """``(mime, data, filename)`` triples from ``Part.inline_data`` dicts.

    ``inline_data`` is ``{"mime_type": str, "data": bytes}`` (plus an
    optional ``display_name``).  ``mime_type`` may be present-but-``None``
    — the wire dict is built from a client payload that may not have
    guessed one — so this reads it defensively rather than trusting a
    ``dict.get`` default.
    """
    entries: List[AttachmentEntry] = []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        entries.append((
            inline.get("mime_type") or "",
            inline.get("data", b""),
            inline.get("display_name"),
        ))
    return entries


def attachment_entries_from_attachments(
    attachments: Any,
) -> List[AttachmentEntry]:
    """``(mime, data, filename)`` triples from ``Attachment`` objects.

    The tool-result side of the same split: ``ToolResult.attachments``
    carries typed ``Attachment`` objects rather than ``inline_data``
    dicts, but the marshalling decision is identical.
    """
    entries: List[AttachmentEntry] = []
    for att in attachments or []:
        entries.append((
            getattr(att, "mime_type", "") or "",
            getattr(att, "data", b""),
            getattr(att, "display_name", None),
        ))
    return entries


def user_message_with_attachments(
    content: str,
    parts: Iterable[Any],
    *,
    pdf_as_file: bool = False,
    audio_as_input_audio: bool = False,
) -> List[Dict[str, Any]]:
    """The ``role:"user"`` wire message(s) for a text + ``inline_data`` turn.

    The whole user-message tail of an OpenAI-shaped converter, shared so
    the three that had it copied (``_openai_compat``, ``nebius``,
    ``openrouter``) cannot drift apart again — drift is what #829 was.

    Three outcomes:

    1. **Something is carried** — multimodal block list: the text first
       (so the question precedes what it is about), then the media, then
       the withheld-note if anything was also dropped.
    2. **Nothing is carried but something was dropped** — plain-string
       content with the note appended, so the turn says what is missing
       instead of arriving mysteriously content-free.
    3. **No attachments at all** — plain-string content, byte-identical to
       the pre-multimodal wire shape.

    Args:
        content: The already-joined text of the message's text parts.
        parts: The message's ``Part`` objects (non-``inline_data`` ignored).
        pdf_as_file: Whether this wire carries PDFs (OpenRouter only).
        audio_as_input_audio: Whether this wire carries audio input as
            ``input_audio`` blocks (OpenRouter only, #830).

    Returns:
        A single-element list, matching ``message_to_openai``'s contract.
    """
    blocks, withheld = marshal_attachments(
        attachment_entries_from_parts(parts),
        pdf_as_file=pdf_as_file,
        audio_as_input_audio=audio_as_input_audio,
    )
    note = withheld_attachment_note(
        withheld,
        pdf_as_file=pdf_as_file,
        audio_as_input_audio=audio_as_input_audio,
    )
    if blocks:
        out: List[Dict[str, Any]] = []
        if content:
            out.append({"type": "text", "text": content})
        out.extend(blocks)
        if note:
            out.append({"type": "text", "text": note})
        return [{"role": "user", "content": out}]
    if note:
        content = f"{content}\n\n{note}" if content else note
    return [{"role": "user", "content": content}]


def tool_result_followup_message(
    attachments: Any,
    *,
    pdf_as_file: bool = False,
    audio_as_input_audio: bool = False,
    label: str = "Attachment",
) -> Optional[Dict[str, Any]]:
    """Follow-up ``role:"user"`` message carrying a tool result's attachments.

    OpenAI-shaped ``tool`` messages cannot carry image or file content —
    it lives only in ``user`` messages — so a tool that returns a PNG or a
    PDF surfaces it as a follow-up user turn, correlated to the result only
    by adjacency and by the lead line naming the files.

    Returns ``None`` when there is nothing to say: no carried attachment
    and nothing withheld.  A result whose attachments were ALL withheld
    still produces a message — the note — because a model told "here is the
    file you asked for" and shown nothing will otherwise invent its
    contents.

    Args:
        attachments: ``ToolResult.attachments`` (may be ``None``).
        pdf_as_file: Whether this wire carries PDFs (OpenRouter only).
        audio_as_input_audio: Whether this wire carries audio input as
            ``input_audio`` blocks (OpenRouter only, #830).
        label: Lead-line noun.  ``"Image"`` for the wires that carry only
            images; ``"Attachment"`` where PDFs and audio also ride.

    Returns:
        The wire message dict, or ``None``.
    """
    blocks: List[Dict[str, Any]] = []
    withheld: List[str] = []
    names: List[str] = []
    for mime, data, filename in attachment_entries_from_attachments(attachments):
        block = attachment_content_block(
            mime, data, filename,
            pdf_as_file=pdf_as_file,
            audio_as_input_audio=audio_as_input_audio,
        )
        if block is None:
            withheld.append(mime or _UNKNOWN_MIME)
        else:
            blocks.append(block)
            names.append(filename or mime)
    _log_withheld(withheld, pdf_as_file, audio_as_input_audio)
    note = withheld_attachment_note(
        withheld,
        pdf_as_file=pdf_as_file,
        audio_as_input_audio=audio_as_input_audio,
    )
    if not blocks and not note:
        return None
    lead = (
        f"[{label} returned by tool call: {', '.join(names)}]" if blocks
        else f"[{label} returned by tool call]"
    )
    text = f"{lead}\n{note}" if note else lead
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}] + blocks,
    }
