"""Message conversion between jaato's wire types and the Prompt API shape.

The Chrome built-in AI Prompt API takes conversation context as
``initialPrompts`` — a list of ``{"role": "system"|"user"|"assistant",
"content": str}`` dicts — and a final prompt string.  This module converts
jaato ``Message`` objects into those dicts.

The Prompt API has no native tool-calling on stable Chrome, so tool
traffic is text-encoded in both directions:

- MODEL-role ``function_call`` parts are re-serialized as ``tool_call``
  fenced blocks (the same syntax the model is instructed to emit, see
  ``tooling.py``) so history round-trips consistently.
- TOOL-role ``function_response`` parts become user messages carrying the
  rendered tool result.  One jaato TOOL message with N parallel
  function_response parts emits N wire messages — dropping all but the
  first is the classic silent-parallel-results bug this mirrors
  ``_openai_compat/converters.py`` in avoiding.

IMPORTANT: this file is loaded STANDALONE (by file path, bypassing the
package ``__init__``) by ``test_provider_capability_conformance``, so it
must use only absolute ``jaato_sdk`` imports — no relative imports, no
vendor SDKs.
"""

import json
from typing import Any, Dict, List

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
    ToolSchema,
    render_result_for_model,
)
from shared.tool_id_map import name_to_id

#: Info string of the fenced block used for text-encoded tool calls.
TOOL_CALL_FENCE = "tool_call"

#: Placeholder inserted where a message carried content this text-only
#: adapter cannot marshal (images, audio).  Keeps the model aware that
#: something was elided instead of silently rewriting history.
NON_TEXT_PLACEHOLDER = "[non-text content omitted: not supported by this provider]"


def serialize_tool_call(wire_id: str, args: Dict[str, Any]) -> str:
    """Render one tool call as the ``tool_call`` fenced block.

    This is the canonical wire syntax in BOTH directions: the system
    prompt (:func:`tool_schemas_to_prompt`) instructs the model to emit
    it, and history replay re-serializes past model calls through here so
    the model sees its own prior turns in the exact syntax it produced
    them.

    ``wire_id`` is the HASHED tool id (``name_to_id``), never the human
    name — the framework-wide contract that no human tool name reaches
    the model (see ``shared/tool_id_map.py``); callers hash before
    calling.
    """
    body = json.dumps({"name": wire_id, "arguments": args},
                      ensure_ascii=False)
    return f"```{TOOL_CALL_FENCE}\n{body}\n```"


def tool_schemas_to_prompt(tools: List[ToolSchema]) -> str:
    """Render tool schemas as the model-facing system-prompt section.

    This is chrome_ai's "tools array": the Prompt API has no native tool
    parameter on stable Chrome, so the schemas are injected as prompt
    text.  Tool names are hashed to opaque wire ids (``name_to_id``) so
    the model relies on the DESCRIPTION, per the framework-wide
    no-human-name-on-the-wire contract — the wire-leak conformance guard
    (``test_tool_id_wire_conformance``) scans this function's output.

    The rendering is deliberately terse (compact JSON schema, short
    preamble): Gemini Nano's whole context is ~6-9k tokens.

    Returns the full section, or ``""`` when ``tools`` is empty.
    """
    if not tools:
        return ""
    lines = [
        "# Tools",
        "",
        "You can call tools. Tools are identified by opaque ids (e.g. "
        "t_1a2b3c4d); pick by DESCRIPTION and use the id exactly as "
        "listed. To call one, output ONLY a fenced block:",
        "",
        f"```{TOOL_CALL_FENCE}",
        '{"name": "<tool id>", "arguments": {<arguments as JSON>}}',
        "```",
        "",
        "Rules:",
        "- Only the tool ids listed below exist. Never invent ids.",
        "- Emit one block per call; you may emit several blocks for"
        " independent calls.",
        "- After emitting tool_call blocks, stop. The results arrive in"
        " the next user message as 'Tool result ...'.",
        "- If no tool is needed, answer in plain text without any"
        f" {TOOL_CALL_FENCE} block.",
        "",
        "Available tools:",
        "",
    ]
    for tool in tools:
        lines.append(f"## {name_to_id(tool.name)}")
        if tool.description:
            lines.append(tool.description.strip())
        if tool.parameters:
            schema = json.dumps(tool.parameters, separators=(",", ":"),
                                ensure_ascii=False)
            lines.append(f"Arguments JSON schema: {schema}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_tool_response(fr: ToolResult) -> str:
    """Render one executed tool's result as model-facing text.

    Uses :func:`render_result_for_model` (never ``str(dict)``) so the
    structured result stays on ``ToolResult.result`` for the ledger/GC
    while the model receives clean JSON plus any trusted steering suffix,
    with untrusted content boundary-wrapped.  The tool is referenced by
    its hashed wire id, matching the id the model called it by.
    """
    rendered = render_result_for_model(
        fr.result,
        getattr(fr, "model_suffix", None),
        untrusted=getattr(fr, "untrusted", False),
        untrusted_source=getattr(fr, "untrusted_source", None),
    )
    label = name_to_id(fr.name) if fr.name else "tool"
    return f"Tool result for {label} (call {fr.call_id}):\n{rendered}"


def message_to_prompt_api(message: Message) -> List[Dict[str, str]]:
    """Convert one jaato ``Message`` to Prompt API message dicts.

    Returns a LIST because one internal message can legitimately expand
    to several wire messages (N parallel tool results) or to zero (a
    message holding only unmarshalable parts still yields a placeholder,
    so the return is in practice never empty for non-empty input).
    """
    if message.role == Role.TOOL:
        out: List[Dict[str, str]] = []
        for part in message.parts:
            if part.function_response is not None:
                out.append({
                    "role": "user",
                    "content": _render_tool_response(part.function_response),
                })
        return out

    role = "assistant" if message.role == Role.MODEL else "user"
    chunks: List[str] = []
    for part in message.parts:
        if part.text is not None:
            chunks.append(part.text)
        elif part.function_call is not None:
            fc = part.function_call
            # History replay: the model emitted (and only ever knows) the
            # hashed wire id, so re-serialize with it, not the human name.
            chunks.append(serialize_tool_call(name_to_id(fc.name), fc.args))
        elif part.thought is not None:
            continue  # internal reasoning is never replayed to the model
        elif part.inline_data is not None:
            chunks.append(NON_TEXT_PLACEHOLDER)
    content = "\n".join(c for c in chunks if c)
    if not content:
        return []
    return [{"role": role, "content": content}]


def messages_to_prompt_api(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert a jaato history slice to a Prompt API message list."""
    out: List[Dict[str, str]] = []
    for message in messages:
        out.extend(message_to_prompt_api(message))
    return out


# ==================== History persistence round-trip ====================
# Standard Message JSON serialization (same shape as the echo provider's,
# kept here so the provider package stays dependency-free).

def serialize_message(message: Message) -> Dict[str, Any]:
    """Serialize a Message to a JSON-safe dict (text / call / response parts)."""
    parts: List[Dict[str, Any]] = []
    for part in message.parts:
        if part.text is not None:
            parts.append({"type": "text", "text": part.text})
        elif part.function_call is not None:
            fc = part.function_call
            parts.append({
                "type": "function_call",
                "id": fc.id,
                "name": fc.name,
                "args": fc.args,
            })
        elif part.function_response is not None:
            fr = part.function_response
            parts.append({
                "type": "function_response",
                "call_id": fr.call_id,
                "name": fr.name,
                "result": fr.result,
                "is_error": fr.is_error,
            })
    return {"role": message.role.value, "parts": parts}


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a dict produced by :func:`serialize_message`."""
    parts: List[Part] = []
    for p in data.get("parts", []):
        ptype = p.get("type")
        if ptype == "text":
            parts.append(Part(text=p["text"]))
        elif ptype == "function_call":
            parts.append(Part(function_call=FunctionCall(
                id=p.get("id", ""),
                name=p["name"],
                args=p.get("args", {}),
            )))
        elif ptype == "function_response":
            parts.append(Part(function_response=ToolResult(
                call_id=p.get("call_id", ""),
                name=p.get("name", ""),
                result=p.get("result"),
                is_error=p.get("is_error", False),
            )))
    return Message(role=Role(data["role"]), parts=parts)
