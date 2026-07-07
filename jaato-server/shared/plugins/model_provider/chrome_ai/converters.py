"""Message conversion between jaato's wire types and the Prompt API shape.

The Chrome built-in AI Prompt API takes conversation context as
``initialPrompts`` — a list of ``{"role": "system"|"user"|"assistant",
"content": str}`` dicts — and a final prompt string.  That is exactly the
least-common-denominator chat shape produced by the shared prose-tools
machinery (``model_provider/_prose_tools.py``), where this provider's
text-encoded tool protocol was hoisted so OpenAI-compat providers can
reuse it behind the ``prose_tool_calls`` quirk.  This module re-exports
those conversions under the provider's historical names (the wire-leak
conformance guard loads this FILE standalone and looks up
``message_to_prompt_api`` / ``tool_schemas_to_prompt`` here) and keeps
the chrome_ai-specific history-persistence round-trip.

IMPORTANT: standalone loading (by file path, bypassing the package
``__init__``) means only absolute ``jaato_sdk``/``shared`` imports are
allowed here — no relative imports, no vendor SDKs.
"""

from typing import Any, Dict, List

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
)
from shared.plugins.model_provider._prose_tools import (
    NON_TEXT_PLACEHOLDER,
    TOOL_CALL_FENCE,
    message_to_prose_chat as message_to_prompt_api,
    messages_to_prose_chat as messages_to_prompt_api,
    serialize_tool_call,
    tool_schemas_to_prompt,
)

__all__ = [
    "NON_TEXT_PLACEHOLDER",
    "TOOL_CALL_FENCE",
    "message_to_prompt_api",
    "messages_to_prompt_api",
    "serialize_tool_call",
    "tool_schemas_to_prompt",
    "serialize_message",
    "deserialize_message",
]


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
