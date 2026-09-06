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

**The policy is per-wire, and it is declared, not assumed.**
``pdf_as_file`` is the only axis today: OpenRouter has a PDF-input
extension (a ``file`` content block, parsed by the upstream model or by
OpenRouter's ``file-parser`` plugin) and declares ``pdf_input=True``;
every ``_openai_compat`` sharer (nim, vllm, lmstudio, tensorrt_llm,
zhipuai_openai, triton, nebius, ovhcloud, doubleword) declares
``pdf_input=False``, so for them a PDF is content the wire does not carry
and is withheld.  The capability declaration and the converter now agree
— before this module they contradicted each other.

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
) -> Optional[Dict[str, Any]]:
    """OpenAI content block for one binary attachment, or ``None``.

    ``image/*`` becomes an ``image_url`` data-URL block — the one shape
    every OpenAI-compatible wire in this tree carries.  ``application/pdf``
    becomes a ``file`` block **only** when the caller passes
    ``pdf_as_file=True``, which is OpenRouter's PDF-input extension and is
    not part of the base OpenAI chat format.

    Every other mime — audio, video, ``application/octet-stream``, and the
    empty string a part with no declared mime carries — returns ``None``.
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
    return None


def _log_withheld(withheld: Sequence[str], pdf_as_file: bool) -> None:
    """WARNING-log a withhold, so an operator sees what never left the box."""
    if withheld:
        logger.warning(
            "Attachment(s) withheld — this wire does not carry %s "
            "(pdf_as_file=%s). Mislabelling them as images is the bug "
            "this replaces (#829).",
            sorted(set(withheld)), pdf_as_file,
        )


def marshal_attachments(
    entries: Iterable[AttachmentEntry],
    *,
    pdf_as_file: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split attachments into wire blocks and withheld mimes.

    Args:
        entries: ``(mime, data, filename)`` triples.
        pdf_as_file: Passed through to :func:`attachment_content_block`.

    Returns:
        ``(blocks, withheld)`` — the content blocks that reached the wire,
        and the mime of each attachment that did not (in encounter order,
        duplicates kept so the note can count them).
    """
    blocks: List[Dict[str, Any]] = []
    withheld: List[str] = []
    for mime, data, filename in entries:
        block = attachment_content_block(
            mime, data, filename, pdf_as_file=pdf_as_file
        )
        if block is not None:
            blocks.append(block)
        else:
            withheld.append(mime or _UNKNOWN_MIME)
    _log_withheld(withheld, pdf_as_file)
    return blocks, withheld


def withheld_attachment_note(withheld: Sequence[str]) -> Optional[str]:
    """Model-facing note for attachments this wire could not carry.

    Mirrors the ``[Attachment withheld: ...]`` shape the session's
    modality gate produces, so the two sources of withholding read the
    same to the model.  Returns ``None`` when nothing was withheld, so a
    caller can append unconditionally.

    The note names the mimes and counts them, because "a document was
    dropped" and "four documents were dropped" lead to different
    follow-up questions.
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
        f"wire accepts (text, or an image), or switch to a provider that "
        f"declares support for it.]"
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

    Returns:
        A single-element list, matching ``message_to_openai``'s contract.
    """
    blocks, withheld = marshal_attachments(
        attachment_entries_from_parts(parts), pdf_as_file=pdf_as_file
    )
    note = withheld_attachment_note(withheld)
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
        label: Lead-line noun.  ``"Image"`` for the wires that carry only
            images; ``"Attachment"`` where PDFs also ride.

    Returns:
        The wire message dict, or ``None``.
    """
    blocks: List[Dict[str, Any]] = []
    withheld: List[str] = []
    names: List[str] = []
    for mime, data, filename in attachment_entries_from_attachments(attachments):
        block = attachment_content_block(
            mime, data, filename, pdf_as_file=pdf_as_file
        )
        if block is None:
            withheld.append(mime or _UNKNOWN_MIME)
        else:
            blocks.append(block)
            names.append(filename or mime)
    _log_withheld(withheld, pdf_as_file)
    note = withheld_attachment_note(withheld)
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
