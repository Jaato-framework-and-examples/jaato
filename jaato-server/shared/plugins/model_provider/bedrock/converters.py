"""Converters between internal types and the Bedrock **Converse** wire.

Converse is Amazon's one message-shaped API across every vendor it hosts
(Anthropic, Amazon Nova, Meta, Mistral, ...), so this module is the whole
of the wire knowledge: the provider below it only moves bytes.

How it differs from every other wire in this tree:

- **Blocks are single-key dicts, not tagged unions.**  ``{"text": ...}``,
  ``{"image": {...}}``, ``{"toolUse": {...}}`` -- the key IS the type.
- **Binary is raw ``bytes``**, under ``source: {"bytes": b"..."}``.  botocore
  base64-encodes blob members itself, so handing it a base64 *string* would
  encode it twice.  This is why the wire this module produces carries the
  payload verbatim rather than base64 (see the note in the capability
  conformance guard).
- **Every media block names a format from a CLOSED vocabulary** --
  ``png|jpeg|gif|webp`` for images, ``pdf|csv|doc|docx|xls|xlsx|html|txt|md``
  for documents, and a fixed audio list.  A mime outside the vocabulary is
  WITHHELD, never relabelled to one inside it: relabelling is #829, and the
  ears (#830) settled that a container the wire does not name is refused
  with a note rather than renamed.
- **Two roles only** (``user`` / ``assistant``), like Anthropic: tool results
  ride in a ``user`` message as ``toolResult`` blocks.
- **Tool arguments are a JSON object, not a JSON string** -- ``toolUse.input``
  is a document member.

Tool names are hashed to opaque wire ids (:func:`name_to_id`) on both
surfaces that carry one -- the ``toolConfig.tools`` array and a name-bearing
``toolChoice`` -- and resolved back with :func:`id_to_name` on the way in.
"""

import base64
import json
from typing import Any, Dict, List, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import (
    Attachment,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolResult,
    ToolSchema,
    render_result_for_model,
)

from shared.tool_id_map import id_to_name, name_to_id, tool_choice_to_wire


# ==================== Wire vocabularies ====================
#
# Each of these is Bedrock's OWN closed enum, transcribed.  A mime that maps
# to nothing here is withheld -- see ``_media_block``.

#: ``image.format`` -- Bedrock names four container formats and no others.
IMAGE_FORMATS: Dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

#: ``document.format``.  Bedrock's document block is how Converse carries a
#: PDF (and a handful of office / plain-text formats) to the models that read
#: them; there is no generic "file" block.
DOCUMENT_FORMATS: Dict[str, str] = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}

#: ``audio.format``.  Bedrock names containers, not codecs, so a mime whose
#: whole meaning lives in its parameters (``audio/L16``) has no entry: see
#: the ``pcm`` note below.
AUDIO_FORMATS: Dict[str, str] = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/opus": "opus",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    # Raw PCM.  Bedrock's ``pcm`` means signed 16-bit little-endian, so the
    # framework's own STREAM_AUDIO_MIME (audio/pcm;rate=...;channels=1) maps
    # here -- the ears accept what the mouth produces.  ``audio/L16`` is
    # deliberately ABSENT: RFC 2586 makes it big-endian, and relabelled
    # big-endian samples are noise, not audio.
    "audio/pcm": "pcm",
}

#: ``video.format``.  Carried because Converse carries it and Amazon Nova
#: reads it; the framework has no ``video`` capability column yet, so this is
#: reachable only for a model whose ``modalities`` say ``video``.
VIDEO_FORMATS: Dict[str, str] = {
    "video/x-matroska": "mkv",
    "video/quicktime": "mov",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/x-flv": "flv",
    "video/mpeg": "mpeg",
    "video/x-ms-wmv": "wmv",
    "video/3gpp": "three_gp",
}

#: Bedrock's ``stopReason`` enum -> the framework's :class:`FinishReason`.
#: ``guardrail_intervened`` and ``content_filtered`` are Bedrock's own
#: additions and both mean the same thing to the session: the model was
#: stopped by a policy, not by finishing.
_STOP_REASONS: Dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_USE,
    "max_tokens": FinishReason.MAX_TOKENS,
    "guardrail_intervened": FinishReason.SAFETY,
    "content_filtered": FinishReason.SAFETY,
}


def _normalise_mime(mime: Optional[str]) -> str:
    """Lower-case the mime and drop its parameters (``;rate=24000``).

    The parameters matter for exactly one family -- raw PCM, whose shape
    lives entirely in them -- and that case is handled by
    :func:`_pcm_parameters_agree` before this strips them.
    """
    return (mime or "").split(";")[0].strip().lower()


def _pcm_parameters_agree(mime: str) -> bool:
    """Do a raw-PCM mime's parameters agree with Bedrock's ``pcm``?

    ``pcm`` on this wire asserts signed 16-bit little-endian samples.  An
    ABSENT parameter is agreement (nothing contradicts the assertion); a
    parameter that CONTRADICTS it -- another bit depth, big-endian -- is a
    withhold, because sending it would relabel the samples rather than
    carry them.
    """
    lowered = (mime or "").lower()
    for token, allowed in (("bits=", "16"), ("format=", "s16le")):
        if token in lowered:
            value = lowered.split(token, 1)[1].split(";")[0].strip()
            if value != allowed:
                return False
    return "big" not in lowered


# ==================== Tool schema conversion ====================

def tool_schema_to_bedrock(schema: ToolSchema) -> Dict[str, Any]:
    """One :class:`ToolSchema` as a Converse ``toolSpec`` entry.

    The name is hashed (:func:`name_to_id`) so the model cannot read
    behaviour off the string, and because Bedrock enforces
    ``^[a-zA-Z0-9_-]{1,64}$`` -- which an MCP name like ``mcp.server.tool``
    does not pass.
    """
    return {
        "toolSpec": {
            "name": name_to_id(schema.name),
            "description": schema.description,
            "inputSchema": {"json": schema.parameters or {"type": "object",
                                                          "properties": {}}},
        }
    }


def tool_schemas_to_bedrock(
    schemas: Optional[List[ToolSchema]],
) -> Optional[List[Dict[str, Any]]]:
    """The ``toolConfig.tools`` array, or ``None`` when there are no tools."""
    if not schemas:
        return None
    return [tool_schema_to_bedrock(s) for s in schemas]


def tool_choice_to_bedrock(tool_choice: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Translate the session's OpenAI-shaped ``tool_choice`` to Converse's.

    The session speaks one dialect to every provider
    (``{"type": "function", "function": {"name": ...}}``, or the strings
    ``auto`` / ``required`` / ``none``); Converse spells the same three
    choices ``{"auto": {}}``, ``{"any": {}}`` and ``{"tool": {"name": ...}}``.

    ``none`` has no Converse spelling at all -- the wire's way to say "do not
    call a tool" is to send no ``toolConfig`` -- so it returns ``None`` here
    and the provider drops the whole block.

    A name-bearing choice is routed through :func:`tool_choice_to_wire` --
    the framework's ONE tool-name mapper -- before the shape is translated,
    because the ``tools`` array this choice must match is hashed.  Doing the
    hash there rather than calling ``name_to_id`` directly keeps a single
    definition of what "the wire id of a tool_choice" means.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        lowered = tool_choice.strip().lower()
        if lowered == "auto":
            return {"auto": {}}
        if lowered in ("required", "any"):
            return {"any": {}}
        return None
    if isinstance(tool_choice, dict):
        mapped = tool_choice_to_wire(tool_choice)
        fn = mapped.get("function")
        name = fn.get("name") if isinstance(fn, dict) else mapped.get("name")
        if isinstance(name, str) and name:
            return {"tool": {"name": name}}
        kind = str(mapped.get("type", "")).lower()
        if kind in ("required", "any"):
            return {"any": {}}
        if kind == "auto":
            return {"auto": {}}
    return None


# ==================== Message conversion ====================

def role_to_bedrock(role: Role) -> str:
    """Converse has ``user`` and ``assistant``; TOOL results ride as ``user``."""
    return "assistant" if role == Role.MODEL else "user"


def role_from_bedrock(role: str) -> Role:
    """Inverse of :func:`role_to_bedrock` for a response message."""
    return Role.MODEL if role == "assistant" else Role.USER


def _media_block(
    mime: Optional[str], data: bytes, name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """The Converse content block for a binary payload, or ``None``.

    ``None`` means *this wire does not name this format* -- the caller
    withholds it and tells the model so.  Dispatching on the mime rather
    than defaulting to an image block is the #829 lesson: silently
    mislabelling was the one outcome worse than either carrying the bytes
    or declining them.

    ``name`` is used only by the document block, whose ``name`` member
    Bedrock requires; it is sanitised because Bedrock rejects a document
    name outside ``[a-zA-Z0-9\\s\\-()\\[\\]]``.
    """
    base = _normalise_mime(mime)
    if base in IMAGE_FORMATS:
        return {"image": {"format": IMAGE_FORMATS[base],
                          "source": {"bytes": data}}}
    if base in DOCUMENT_FORMATS:
        return {"document": {"format": DOCUMENT_FORMATS[base],
                             "name": _document_name(name),
                             "source": {"bytes": data}}}
    if base in AUDIO_FORMATS:
        if AUDIO_FORMATS[base] == "pcm" and not _pcm_parameters_agree(mime or ""):
            return None
        return {"audio": {"format": AUDIO_FORMATS[base],
                          "source": {"bytes": data}}}
    if base in VIDEO_FORMATS:
        return {"video": {"format": VIDEO_FORMATS[base],
                          "source": {"bytes": data}}}
    return None


def _document_name(name: Optional[str]) -> str:
    """A document name Bedrock will accept.

    Bedrock constrains the member to alphanumerics, whitespace, hyphens,
    parentheses and square brackets -- so the dots in ``report.pdf`` are a
    400.  Everything else becomes a space, runs are collapsed, and an empty
    result falls back to a constant rather than an empty string (also a 400).
    """
    raw = name or "document"
    kept = [c if (c.isalnum() or c in " -()[]") else " " for c in raw]
    cleaned = " ".join("".join(kept).split())
    return cleaned or "document"


def part_to_bedrock_content_block(part: Part) -> Optional[Dict[str, Any]]:
    """One :class:`Part` as a Converse content block, or ``None`` to drop it.

    ``None`` is returned for a part this wire has no block for -- an empty
    text part, or binary in a format Bedrock does not name.  Callers append
    only truthy blocks, because Converse rejects a message whose ``content``
    array is empty.
    """
    if part.text is not None:
        # Converse rejects an empty ``text`` member outright, so an empty
        # text part is dropped rather than sent as "".
        return {"text": part.text} if part.text else None

    if part.function_call is not None:
        fc = part.function_call
        return {
            "toolUse": {
                "toolUseId": fc.id,
                "name": name_to_id(fc.name),
                # A document member, not a JSON string: unlike OpenAI's
                # ``arguments``, Converse wants the object itself.
                "input": fc.args or {},
            }
        }

    if part.function_response is not None:
        return _tool_result_block(part.function_response)

    if part.inline_data is not None:
        data = part.inline_data.get("data") or b""
        if isinstance(data, str):
            data = base64.b64decode(data)
        return _media_block(
            part.inline_data.get("mime_type"),
            data,
            part.inline_data.get("display_name"),
        )

    if part.thought is not None:
        # Deliberately NOT emitted.  Converse's ``reasoningContent`` block
        # requires the upstream's own ``signature`` alongside the text, and
        # ``Part.thought`` carries no signature -- so replaying the text
        # alone would be rejected by the very models that produce it.  See
        # the reasoning-replay note in the package docstring.
        return None

    return None


def _tool_result_block(result: ToolResult) -> Dict[str, Any]:
    """A ``toolResult`` block: the rendered text, then any attachments.

    The text goes through :func:`render_result_for_model` -- the same
    rendering every other provider uses -- so the untrusted-content boundary
    and the result's ``model_suffix`` steering survive this wire too.

    An attachment in a format Bedrock does not name is replaced by a note,
    not dropped silently: a model handed a result whose picture vanished
    answers as though the tool returned nothing.
    """
    text = render_result_for_model(
        result.result,
        result.model_suffix,
        untrusted=result.untrusted,
        untrusted_source=result.untrusted_source,
    )
    content: List[Dict[str, Any]] = []
    if text:
        content.append({"text": text})

    for att in result.attachments or []:
        block = _media_block(att.mime_type, att.data, att.display_name)
        if block:
            content.append(block)
        else:
            content.append({"text": _withheld_note(att)})

    if not content:
        # Converse rejects an empty ``toolResult.content``; an executed tool
        # that produced nothing is still an answer to the call.
        content.append({"text": "(no output)"})

    block_out: Dict[str, Any] = {
        "toolUseId": result.call_id,
        "content": content,
    }
    if result.is_error:
        block_out["status"] = "error"
    return {"toolResult": block_out}


def _withheld_note(att: Attachment) -> str:
    """Tell the model what was withheld, and what this wire does carry."""
    name = f" {att.display_name}" if att.display_name else ""
    return (
        f"[Attachment withheld:{name} {att.mime_type or 'unknown type'} is not "
        f"a format the Bedrock Converse API carries. It accepts images "
        f"(png, jpeg, gif, webp), documents (pdf, csv, doc, docx, xls, xlsx, "
        f"html, txt, md), audio and video.]"
    )


def message_to_bedrock(message: Message) -> Dict[str, Any]:
    """One :class:`Message` as a Converse message dict.

    Returns ``{"role": ..., "content": [...]}``.  The content array may come
    back empty -- :func:`messages_to_bedrock` is what drops such a message,
    since only it can see whether dropping breaks the alternation.
    """
    content: List[Dict[str, Any]] = []
    for part in message.parts:
        block = part_to_bedrock_content_block(part)
        if block:
            content.append(block)
    return {"role": role_to_bedrock(message.role), "content": content}


def _paired_tool_ids(messages: List[Message]) -> Tuple[set, set]:
    """The tool-call ids that have BOTH halves, in each direction.

    Converse rejects a ``toolUse`` with no answering ``toolResult`` and a
    ``toolResult`` answering nothing, and a history can acquire either from
    a turn that died mid-flight -- an exception between the model's request
    and the tool's answer, or a cancelled stream whose parts were saved.
    One orphan makes every later request in that session a 400, which is a
    session that cannot be recovered by talking to it.

    Returns ``(answered_call_ids, known_result_ids)``; the caller emits a
    half only when its id is in the matching set, so an orphan is dropped
    rather than sent.  Order-independent on purpose: an out-of-order pair
    is still a pair.
    """
    call_ids = {p.function_call.id for m in messages for p in m.parts
                if p.function_call is not None}
    result_ids = {p.function_response.call_id for m in messages for p in m.parts
                  if p.function_response is not None}
    return call_ids & result_ids, result_ids & call_ids


def _is_orphan(part: Part, paired: set) -> bool:
    """Is this part half of a tool call whose other half is missing?"""
    if part.function_call is not None:
        return part.function_call.id not in paired
    if part.function_response is not None:
        return part.function_response.call_id not in paired
    return False


def messages_to_bedrock(messages: List[Message]) -> List[Dict[str, Any]]:
    """The Converse ``messages`` array.

    Three wire rules are enforced here rather than left to the caller,
    because each is a rejection rather than a degradation:

    * **consecutive same-role messages are merged** -- Converse, like the
      Anthropic API beneath much of it, requires the roles to alternate, and
      a tool-result turn is several internal messages that all become
      ``user``;
    * **an unpaired ``toolUse`` or ``toolResult`` is dropped** -- see
      :func:`_paired_tool_ids`; one orphan makes every later request in the
      session a 400;
    * **a message that converted to no blocks is dropped** -- an empty
      ``content`` array is a 400, and the merge above is what makes dropping
      safe (the surviving neighbour absorbs the turn).
    """
    paired, _ = _paired_tool_ids(messages)
    out: List[Dict[str, Any]] = []
    for msg in messages:
        usable = [p for p in msg.parts if not _is_orphan(p, paired)]
        converted = message_to_bedrock(Message(role=msg.role, parts=usable))
        if not converted["content"]:
            continue
        if out and out[-1]["role"] == converted["role"]:
            out[-1]["content"].extend(converted["content"])
        else:
            out.append(converted)
    return out


def system_to_bedrock(
    system_instruction: Optional[str],
    *,
    cache: bool = False,
    cache_ttl: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """The ``system`` array, optionally closed by a cache point.

    Converse's ``cachePoint`` is the same idea as Anthropic's
    ``cache_control``: everything BEFORE the marker is cacheable.  Placing
    one at the end of the system array is the breakpoint that pays for
    itself on every turn, because the system prompt is the part that does
    not change.
    """
    if not system_instruction:
        return None
    blocks: List[Dict[str, Any]] = [{"text": system_instruction}]
    if cache:
        blocks.append({"cachePoint": _cache_point(cache_ttl)})
    return blocks


def _cache_point(ttl: Optional[str]) -> Dict[str, Any]:
    """A ``cachePoint`` block; ``ttl`` is omitted unless it is ``5m``/``1h``.

    An unrecognised TTL is dropped rather than forwarded: Bedrock rejects
    anything outside its two-value enum, and defaulting to the shorter of
    the two is the cheaper mistake.
    """
    point: Dict[str, Any] = {"type": "default"}
    if ttl in ("5m", "1h"):
        point["ttl"] = ttl
    return point


def tool_config_to_bedrock(
    tools: Optional[List[ToolSchema]],
    tool_choice: Optional[Any] = None,
    *,
    cache: bool = False,
    cache_ttl: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The ``toolConfig`` block, or ``None`` when the turn sends no tools.

    A ``toolChoice`` without ``tools`` is a 400 on this wire (as on OpenAI's),
    so the choice is only ever emitted inside a config that has tools.
    """
    wire_tools = tool_schemas_to_bedrock(tools)
    if not wire_tools:
        return None
    if cache:
        wire_tools = list(wire_tools) + [{"cachePoint": _cache_point(cache_ttl)}]
    config: Dict[str, Any] = {"tools": wire_tools}
    choice = tool_choice_to_bedrock(tool_choice)
    if choice:
        config["toolChoice"] = choice
    return config


# ==================== Response conversion ====================

def finish_reason_from_bedrock(stop_reason: Optional[str]) -> FinishReason:
    """Map ``stopReason`` to :class:`FinishReason`.

    An unrecognised reason maps to ``UNKNOWN`` rather than ``STOP``: Bedrock
    adds reasons as it adds models, and reading a new one as a clean finish
    is how a refusal becomes an answer.
    """
    return _STOP_REASONS.get((stop_reason or "").lower(), FinishReason.UNKNOWN)


def usage_from_bedrock(raw_usage: Optional[Dict[str, Any]]) -> TokenUsage:
    """Read a Converse ``usage`` object into :class:`TokenUsage`.

    Bedrock reports ``inputTokens`` alongside ``cacheReadInputTokens`` /
    ``cacheWriteInputTokens`` as DISJOINT buckets -- Anthropic's convention,
    which is also the one :class:`TokenUsage` documents -- so the three are
    carried across unchanged and ``total`` is their sum plus output.
    """
    usage = TokenUsage()
    if not raw_usage:
        return usage
    usage.prompt_tokens = int(raw_usage.get("inputTokens") or 0)
    usage.output_tokens = int(raw_usage.get("outputTokens") or 0)
    cache_read = raw_usage.get("cacheReadInputTokens")
    cache_write = raw_usage.get("cacheWriteInputTokens")
    if cache_read is not None:
        usage.cache_read_tokens = int(cache_read)
    if cache_write is not None:
        usage.cache_creation_tokens = int(cache_write)
    reported_total = raw_usage.get("totalTokens")
    usage.total_tokens = int(reported_total) if reported_total is not None else (
        usage.prompt_tokens + usage.output_tokens
    )
    return usage


def content_block_to_part(block: Dict[str, Any]) -> Tuple[Optional[Part], Optional[str]]:
    """One response content block as a ``(Part, reasoning_text)`` pair.

    Both halves can be ``None``: an unrecognised block yields neither.
    Reasoning is returned SEPARATELY rather than as a ``Part`` because it
    belongs on :attr:`ProviderResponse.thinking` -- the session shows it and
    does not replay it, for the signature reason in the module docstring.
    """
    if "text" in block:
        return Part.from_text(block["text"]), None
    if "toolUse" in block:
        use = block["toolUse"] or {}
        return Part.from_function_call(FunctionCall(
            id=use.get("toolUseId", ""),
            name=id_to_name(use.get("name", "")),
            args=use.get("input") or {},
        )), None
    if "reasoningContent" in block:
        reasoning = (block["reasoningContent"] or {}).get("reasoningText") or {}
        return None, reasoning.get("text")
    return None, None


def response_from_bedrock(response: Dict[str, Any]) -> ProviderResponse:
    """A batch ``converse()`` response as a :class:`ProviderResponse`."""
    message = ((response or {}).get("output") or {}).get("message") or {}
    parts: List[Part] = []
    reasoning: List[str] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        part, thought = content_block_to_part(block)
        if part is not None:
            parts.append(part)
        if thought:
            reasoning.append(thought)

    return ProviderResponse(
        parts=parts,
        usage=usage_from_bedrock((response or {}).get("usage")),
        finish_reason=finish_reason_from_bedrock((response or {}).get("stopReason")),
        raw=response,
        thinking="".join(reasoning) or None,
    )


# ==================== History serialization ====================

def serialize_message(message: Message) -> Dict[str, Any]:
    """Serialize a :class:`Message` to a JSON-safe dict."""
    parts: List[Dict[str, Any]] = []
    for part in message.parts:
        if part.text is not None:
            parts.append({"type": "text", "text": part.text})
        elif part.thought is not None:
            parts.append({"type": "thought", "thought": part.thought})
        elif part.function_call is not None:
            fc = part.function_call
            parts.append({"type": "function_call", "id": fc.id,
                          "name": fc.name, "args": fc.args})
        elif part.function_response is not None:
            fr = part.function_response
            parts.append({"type": "function_response", "call_id": fr.call_id,
                          "name": fr.name, "result": fr.result,
                          "is_error": fr.is_error})
        elif part.inline_data is not None:
            data = part.inline_data.get("data") or b""
            if isinstance(data, bytes):
                data = base64.b64encode(data).decode("utf-8")
            parts.append({"type": "inline_data",
                          "mime_type": part.inline_data.get("mime_type"),
                          "data": data})
    return {"role": message.role.value, "parts": parts}


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Inverse of :func:`serialize_message`."""
    parts: List[Part] = []
    for p in data.get("parts", []):
        kind = p.get("type")
        if kind == "text":
            parts.append(Part(text=p.get("text", "")))
        elif kind == "thought":
            parts.append(Part(thought=p.get("thought", "")))
        elif kind == "function_call":
            parts.append(Part(function_call=FunctionCall(
                id=p.get("id", ""), name=p.get("name", ""),
                args=p.get("args") or {})))
        elif kind == "function_response":
            parts.append(Part(function_response=ToolResult(
                call_id=p.get("call_id", ""), name=p.get("name", ""),
                result=p.get("result"), is_error=p.get("is_error", False))))
        elif kind == "inline_data":
            raw = p.get("data")
            if raw and isinstance(raw, str):
                raw = base64.b64decode(raw)
            parts.append(Part(inline_data={"mime_type": p.get("mime_type"),
                                           "data": raw}))
    return Message(role=Role(data["role"]), parts=parts)


def serialize_history(history: List[Message]) -> str:
    """Serialize history to a JSON string."""
    return json.dumps([serialize_message(m) for m in history])


def deserialize_history(data: str) -> List[Message]:
    """Deserialize a JSON string produced by :func:`serialize_history`."""
    return [deserialize_message(m) for m in json.loads(data)]
