"""Converters between internal types and OpenAI's **Responses API** shape.

The Responses API is not a dialect of Chat Completions; it is a different
request and reply *shape*, and the differences are the whole reason this
module exists rather than a flag on the chat converter:

======================  ===============================  =============================
concept                 Chat Completions                 Responses
======================  ===============================  =============================
conversation            ``messages``                     ``input`` (a flat item list)
system prompt           a ``role:"system"`` message      the ``instructions`` field
a tool definition       ``{"type":"function",            ``{"type":"function",
                          "function":{name,...}}``         name, parameters, ...}``
                        (nested)                         (flat)
a tool call             ``message.tool_calls[]``          a top-level ``function_call``
                        keyed by ``id``                   item keyed by ``call_id``
a tool result           ``role:"tool"`` message          a ``function_call_output`` item
an image                ``image_url``                    ``input_image``
a PDF                   ``file`` / ``file_data``         ``input_file`` / ``file_data``
assistant text          ``content`` (a string)           an ``output_text`` content part
usage                   ``prompt_tokens`` /              ``input_tokens`` /
                        ``completion_tokens``              ``output_tokens``
======================  ===============================  =============================

**The mime dispatch is NOT reimplemented here.**  Whether a wire carries a
PDF, an audio container, or nothing at all is decided once, in
``_attachments.attachment_content_block``, and this module *translates* the
chat block it returns into the Responses spelling.  Two converters for one
policy is what #829 was; the translation table below is deliberately
mechanical so the policy has nowhere to fork.

Tool names are hashed to opaque wire ids exactly as on the chat path
(``shared.tool_id_map``), so nothing here weakens the guarantee that
``test_tool_id_wire_conformance`` pins: the human name never reaches the
model, and a returned id is resolved back before the call is executed.

Importable as a FILE, with no package context and no vendor SDK — the
conformance guards load provider converters that way — hence the absolute
``shared.`` / ``jaato_sdk.`` imports.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    TokenUsage,
    ToolSchema,
    normalize_inclusive_usage,
    parse_tool_call_arguments,
    render_result_for_model,
)

from shared.tool_id_map import id_to_name, name_to_id
from shared.plugins.model_provider._attachments import (
    attachment_entries_from_attachments,
    attachment_entries_from_parts,
    marshal_attachments,
    withheld_attachment_note,
)

#: Responses carries everything ``api.openai.com``'s chat wire carries.
PDF_AS_FILE = True
AUDIO_AS_INPUT_AUDIO = True


# ==================== Tool schemas ====================

def tool_schema_to_responses(schema: ToolSchema) -> Dict[str, Any]:
    """One ``ToolSchema`` as a Responses tool definition.

    The Responses shape is FLAT — ``name`` / ``description`` /
    ``parameters`` sit beside ``type``, where Chat Completions nests them
    under a ``function`` key.  Sending the chat shape here is a 400, which
    is why the two converters cannot be shared by aliasing.

    Args:
        schema: The internal tool schema.

    Returns:
        The Responses tool-definition dict, with the name hashed.
    """
    return {
        "type": "function",
        "name": name_to_id(schema.name),
        "description": schema.description,
        "parameters": schema.parameters,
    }


def tool_schemas_to_responses(
    schemas: Optional[List[ToolSchema]],
) -> Optional[List[Dict[str, Any]]]:
    """Convert tool schemas to the Responses ``tools`` array (or ``None``)."""
    if not schemas:
        return None
    return [tool_schema_to_responses(s) for s in schemas]


def get_original_tool_name(tool_id: str) -> str:
    """Resolve a hashed wire id back to the human tool name.

    Falls back to the id itself when the mapping has no entry — a
    hallucinated id must stay visible as the invention it is rather than
    be dressed up as a resolved name (#873).
    """
    return id_to_name(tool_id) or tool_id


# ==================== Attachments ====================

def _chat_block_to_responses(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate one chat content block into its Responses spelling.

    The three carried forms, and nothing else:

    - ``image_url`` → ``input_image`` (the data URL moves up one level)
    - ``file`` → ``input_file`` (same ``file_data`` data URL, flattened)
    - ``input_audio`` → ``input_audio`` (identical in both shapes)

    Returns ``None`` for a block shape this translation does not know,
    which cannot happen for blocks produced by
    :func:`attachment_content_block` today but keeps an added block type
    from reaching the wire in a shape the API never defined.
    """
    kind = block.get("type")
    if kind == "image_url":
        return {
            "type": "input_image",
            "image_url": block.get("image_url", {}).get("url", ""),
        }
    if kind == "file":
        file_block = block.get("file", {})
        return {
            "type": "input_file",
            "filename": file_block.get("filename", "document.pdf"),
            "file_data": file_block.get("file_data", ""),
        }
    if kind == "input_audio":
        return {"type": "input_audio", "input_audio": block.get("input_audio", {})}
    return None


def _input_content(text: str, entries) -> List[Dict[str, Any]]:
    """The ``content`` list of one user item: text first, then media.

    Text leads so the question precedes what it is about (the same order
    the chat converter uses).  Anything the wire cannot carry is stated
    in-band via :func:`withheld_attachment_note` rather than dropped
    silently — a model told "here is the file" and shown nothing will
    otherwise confabulate its contents.
    """
    blocks, withheld = marshal_attachments(
        entries,
        pdf_as_file=PDF_AS_FILE,
        audio_as_input_audio=AUDIO_AS_INPUT_AUDIO,
    )
    content: List[Dict[str, Any]] = []
    if text:
        content.append({"type": "input_text", "text": text})
    for block in blocks:
        translated = _chat_block_to_responses(block)
        if translated is not None:
            content.append(translated)
    note = withheld_attachment_note(
        withheld,
        pdf_as_file=PDF_AS_FILE,
        audio_as_input_audio=AUDIO_AS_INPUT_AUDIO,
    )
    if note:
        content.append({"type": "input_text", "text": note})
    return content


# ==================== History → input items ====================

def _tool_result_items(message: Message) -> List[Dict[str, Any]]:
    """``function_call_output`` items for one internal TOOL message.

    One item per ``function_response`` part: a parallel tool batch arrives
    as a single ``Message`` carrying N results, and the API keys each
    output by its own ``call_id``.  Emitting only the first is how N-1
    results silently leave the conversation.

    Attachments cannot ride a ``function_call_output`` (its ``output`` is
    a string), so a result carrying media is followed by a ``user`` item
    holding it — the Responses counterpart of the chat path's follow-up
    user message.
    """
    items: List[Dict[str, Any]] = []
    for fr in (p.function_response for p in message.parts if p.function_response):
        items.append({
            "type": "function_call_output",
            "call_id": fr.call_id,
            "output": render_result_for_model(
                fr.result, fr.model_suffix,
                untrusted=fr.untrusted, untrusted_source=fr.untrusted_source,
            ),
        })
        entries = attachment_entries_from_attachments(
            getattr(fr, "attachments", None))
        if not entries:
            continue
        content = _input_content(
            f"Attachment(s) returned by {fr.name}:", entries)
        if len(content) > 1:
            items.append({"role": "user", "content": content})
    return items


def _assistant_items(message: Message) -> List[Dict[str, Any]]:
    """Assistant text and ``function_call`` items for one MODEL message.

    Text and calls are separate top-level items in this shape (Chat
    Completions carries both on one message), and the call's identity is
    ``call_id`` — the same value the matching ``function_call_output``
    quotes.  ``arguments`` is a JSON *string*, as on the chat wire.
    """
    items: List[Dict[str, Any]] = []
    text = "".join(p.text for p in message.parts if p.text)
    if text:
        items.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })
    for fc in (p.function_call for p in message.parts if p.function_call):
        items.append({
            "type": "function_call",
            "call_id": fc.id,
            "name": name_to_id(fc.name),
            "arguments": json.dumps(fc.args or {}),
        })
    return items


def message_to_responses(message: Message) -> List[Dict[str, Any]]:
    """Convert one internal ``Message`` into Responses ``input`` items.

    Returns a LIST because one internal message can be several items: a
    TOOL message carries N parallel results, and a MODEL message carries
    text *and* calls as separate items.

    Args:
        message: Internal message.

    Returns:
        Responses input items, in wire order.
    """
    if any(p.function_response for p in message.parts):
        return _tool_result_items(message)
    if message.role == Role.MODEL:
        return _assistant_items(message)
    text = "".join(p.text for p in message.parts if p.text)
    content = _input_content(text, attachment_entries_from_parts(message.parts))
    if not content:
        # A turn with neither text nor carriable media still has to be an
        # item, or the request loses a turn of the conversation entirely.
        content = [{"type": "input_text", "text": ""}]
    return [{"role": "user", "content": content}]


def history_to_responses(history: List[Message]) -> List[Dict[str, Any]]:
    """Convert internal history to the Responses ``input`` array."""
    return [
        item for m in (history or []) for item in message_to_responses(m)
    ]


# ==================== Response → internal ====================

def _parts_from_content(content: Any) -> List[Part]:
    """Text parts from one output message item's ``content`` list."""
    parts: List[Part] = []
    for block in content or []:
        kind = _get(block, "type")
        if kind in ("output_text", "text"):
            text = _get(block, "text") or ""
            if text:
                parts.append(Part.from_text(text))
        elif kind == "refusal":
            # A refusal IS the model's answer for that turn; dropping it
            # leaves an empty response and an unexplained silence.
            refusal = _get(block, "refusal") or ""
            if refusal:
                parts.append(Part.from_text(refusal))
    return parts


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off an SDK model object or a plain dict.

    The Responses SDK returns pydantic models, tests and replayed
    fixtures carry dicts, and both must convert identically.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def part_from_function_call_item(item: Any) -> Part:
    """One ``function_call`` output item as an internal ``Part``.

    ``call_id`` (not the item's own ``id``) is the identity the matching
    ``function_call_output`` must quote, so it is what becomes
    ``FunctionCall.id``.  Arguments that will not parse stay unreadable:
    the session refuses the call and tells the model, rather than
    executing a zero-argument call it never made (#750).
    """
    args, unreadable = parse_tool_call_arguments(_get(item, "arguments"))
    return Part.from_function_call(FunctionCall(
        id=_get(item, "call_id") or _get(item, "id"),
        name=get_original_tool_name(_get(item, "name") or ""),
        args=args,
        unreadable_args=unreadable,
    ))


def parts_from_responses_output(output: Any) -> List[Part]:
    """Internal parts for a Responses ``output`` array, in wire order.

    Reasoning items are skipped: their summaries reach the caller through
    the thinking channel, and their encrypted content is not something a
    later turn of ours replays.
    """
    parts: List[Part] = []
    for item in output or []:
        kind = _get(item, "type")
        if kind == "function_call":
            parts.append(part_from_function_call_item(item))
        elif kind == "message":
            parts.extend(_parts_from_content(_get(item, "content")))
    return parts


def usage_from_responses(usage: Any) -> TokenUsage:
    """Token usage from a Responses ``usage`` object.

    The field names differ from Chat Completions (``input_tokens`` /
    ``output_tokens``), and the cached count sits under
    ``input_tokens_details.cached_tokens``.  That count is a SUBSET of
    ``input_tokens`` on this wire, so it is normalised out of the prompt
    total — otherwise the cached tokens are counted on both sides of
    every downstream sum (#758).
    """
    if usage is None:
        return TokenUsage()
    details = _get(usage, "input_tokens_details")
    cached = _get(details, "cached_tokens") if details is not None else None
    return normalize_inclusive_usage(TokenUsage(
        prompt_tokens=_get(usage, "input_tokens") or 0,
        output_tokens=_get(usage, "output_tokens") or 0,
        total_tokens=_get(usage, "total_tokens") or 0,
        cache_read_tokens=cached if isinstance(cached, int) and cached else None,
    ))


def thinking_from_responses_output(output: Any) -> Optional[str]:
    """Concatenated reasoning summaries from a Responses ``output`` array.

    Only the *summary* is available: the model's raw reasoning is not
    returned, and its encrypted form is opaque by construction.
    """
    chunks: List[str] = []
    for item in output or []:
        if _get(item, "type") != "reasoning":
            continue
        for block in _get(item, "summary") or []:
            text = _get(block, "text") or ""
            if text:
                chunks.append(text)
    return "".join(chunks) or None


#: ``response.status`` / ``incomplete_details.reason`` → internal reason.
#:
#: ``incomplete`` is the API's word for "the turn ended early", and the
#: reason under it says why.  It is NOT jaato's ``INCOMPLETE``, which
#: means the stream never ended at all — the two would be a dangerous
#: synonym, since one is a turn the API completed and reported on and the
#: other is a severed connection (#687).
_INCOMPLETE_REASONS = {
    "max_output_tokens": FinishReason.MAX_TOKENS,
    "content_filter": FinishReason.SAFETY,
}


def finish_reason_from_response(response: Any) -> FinishReason:
    """Map a Responses ``status`` (+ ``incomplete_details``) to a reason.

    ``completed`` is ``STOP``; ``incomplete`` consults the reason;
    ``failed`` is ``ERROR``.  An unrecognised status maps to ``UNKNOWN``
    — "the turn ended and we do not map the label", which stays a
    success, and is deliberately not the sentinel a severed stream uses.
    """
    status = _get(response, "status")
    if status == "completed":
        return FinishReason.STOP
    if status == "failed":
        return FinishReason.ERROR
    if status == "incomplete":
        details = _get(response, "incomplete_details")
        reason = _get(details, "reason") if details is not None else None
        return _INCOMPLETE_REASONS.get(reason, FinishReason.UNKNOWN)
    return FinishReason.UNKNOWN
