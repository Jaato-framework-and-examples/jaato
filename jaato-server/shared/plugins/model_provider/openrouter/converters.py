"""Converters between internal types and OpenAI chat completions format.

OpenRouter exposes the OpenAI chat-completions wire format faithfully,
so this module is a direct port of the NIM converter.  It handles the
bidirectional translation between the provider-agnostic types
(``Message``, ``ToolSchema``, ...) and the OpenAI SDK's native shapes.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion  # noqa: F401

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolResult,
    ToolSchema,
)

from shared.tool_id_map import id_to_name, name_to_id


# ==================== Tool Name Mapping ====================


def sanitize_tool_name(name: str) -> str:
    """Map a tool name to its hash-derived ID.

    OpenRouter routes raw tool names through to upstream providers.
    Some upstreams (notably OpenAI) reject names containing dots or
    other characters MCP tools commonly use, so we always hash to a
    safe deterministic ID.
    """
    return name_to_id(name)


def get_original_tool_name(tool_id: str) -> str:
    """Resolve a hash-derived ID back to the original tool name."""
    return id_to_name(tool_id)


def clear_tool_name_mapping() -> None:
    """No-op. Hash-derived IDs are deterministic and need no clearing."""
    pass


def register_tool_name_mapping(sanitized: str, original: str) -> None:
    """No-op. Hash-derived IDs handle reverse mapping automatically."""
    pass


# ==================== ToolSchema Conversion ====================

def tool_schema_to_openai(schema: ToolSchema) -> Dict[str, Any]:
    """Convert a ``ToolSchema`` to the OpenAI tool definition dict."""
    return {
        "type": "function",
        "function": {
            "name": name_to_id(schema.name),
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


def tool_schemas_to_openai(
    schemas: Optional[List[ToolSchema]],
) -> Optional[List[Dict[str, Any]]]:
    """Convert a list of ``ToolSchema`` objects to OpenAI tool definitions."""
    if not schemas:
        return None
    return [tool_schema_to_openai(s) for s in schemas]


# ==================== Message Conversion ====================

def message_to_openai(message: Message) -> Dict[str, Any]:
    """Convert an internal ``Message`` to the OpenAI chat-message dict."""
    role = message.role

    text_parts = [p.text for p in message.parts if p.text]
    content = "".join(text_parts) if text_parts else ""

    function_calls = [p.function_call for p in message.parts if p.function_call]
    function_responses = [
        p.function_response for p in message.parts if p.function_response
    ]

    if function_responses:
        fr = function_responses[0]
        result_str = (
            json.dumps(fr.result) if not isinstance(fr.result, str) else fr.result
        )
        return {
            "role": "tool",
            "tool_call_id": fr.call_id,
            "content": result_str,
        }

    if role == Role.MODEL:
        msg: Dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if function_calls:
            msg["tool_calls"] = [
                {
                    "id": fc.id,
                    "type": "function",
                    "function": {
                        "name": sanitize_tool_name(fc.name),
                        "arguments": json.dumps(fc.args),
                    },
                }
                for fc in function_calls
            ]
            if not content:
                msg["content"] = None
        return msg

    return {
        "role": "user",
        "content": content,
    }


def message_from_openai(msg: Dict[str, Any]) -> Message:
    """Convert an OpenAI chat-message dict back to an internal ``Message``."""
    parts: List[Part] = []
    role_str = msg.get("role", "user")

    if role_str == "tool":
        result = msg.get("content", "")
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            pass
        parts.append(Part(function_response=ToolResult(
            call_id=msg.get("tool_call_id", ""),
            name="",
            result=result,
        )))
        return Message(role=Role.TOOL, parts=parts)

    if role_str == "assistant":
        content = msg.get("content")
        if content:
            parts.append(Part(text=content))
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            args: Dict[str, Any] = {}
            if func.get("arguments"):
                try:
                    args = json.loads(func["arguments"])
                except json.JSONDecodeError:
                    args = {"raw": func["arguments"]}
            parts.append(Part(function_call=FunctionCall(
                id=tc.get("id", ""),
                name=get_original_tool_name(func.get("name", "")),
                args=args,
            )))
        return Message(role=Role.MODEL, parts=parts)

    if role_str == "system":
        content = msg.get("content", "")
        if content:
            parts.append(Part(text=content))
        return Message(role=Role.USER, parts=parts)

    content = msg.get("content", "")
    if content:
        parts.append(Part(text=content))
    return Message(role=Role.USER, parts=parts)


def history_to_openai(history: List[Message]) -> List[Dict[str, Any]]:
    """Convert internal history to an OpenAI message list."""
    return [message_to_openai(m) for m in (history or [])]


# ==================== Response Conversion ====================

def extract_parts_from_response(response: "ChatCompletion") -> List[Part]:
    """Extract ``Part`` objects from an OpenAI ``ChatCompletion``.

    Preserves the natural ordering: text first, then any tool calls.
    """
    parts: List[Part] = []

    if not response or not response.choices:
        return parts

    for choice in response.choices:
        if not choice.message:
            continue

        if choice.message.content:
            parts.append(Part.from_text(choice.message.content))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args: Dict[str, Any] = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                fc = FunctionCall(
                    id=tc.id,
                    name=get_original_tool_name(tc.function.name),
                    args=args,
                )
                parts.append(Part.from_function_call(fc))

    return parts


def extract_finish_reason(response: "ChatCompletion") -> FinishReason:
    """Extract the ``FinishReason`` from an OpenAI response."""
    if not response or not response.choices:
        return FinishReason.UNKNOWN

    for choice in response.choices:
        reason = choice.finish_reason
        if reason:
            reason_str = str(reason).lower()
            if reason_str == "stop":
                return FinishReason.STOP
            elif reason_str in ("length", "max_tokens"):
                return FinishReason.MAX_TOKENS
            elif reason_str == "tool_calls":
                return FinishReason.TOOL_USE
            elif reason_str == "content_filter":
                return FinishReason.SAFETY

    return FinishReason.UNKNOWN


def extract_usage(response: "ChatCompletion") -> TokenUsage:
    """Extract token usage from an OpenAI response."""
    usage = TokenUsage()

    if not response or not response.usage:
        return usage

    usage.prompt_tokens = response.usage.prompt_tokens or 0
    usage.output_tokens = response.usage.completion_tokens or 0
    usage.total_tokens = response.usage.total_tokens or 0

    return usage


def extract_reasoning_from_response(response: "ChatCompletion") -> Optional[str]:
    """Extract reasoning/thinking content if the upstream model exposes it.

    OpenRouter forwards a ``reasoning`` field for models that support
    chain-of-thought (DeepSeek-R1, OpenAI o-series, ...).  Some upstreams
    use ``reasoning_content`` instead, so we accept both spellings.
    """
    if not response or not getattr(response, "choices", None):
        return None

    reasoning_parts: List[str] = []
    for choice in response.choices:
        msg = getattr(choice, "message", None)
        if not msg:
            continue
        for attr in ("reasoning", "reasoning_content"):
            value = getattr(msg, attr, None)
            if value and isinstance(value, str):
                reasoning_parts.append(value)
                break

    return "\n".join(reasoning_parts) if reasoning_parts else None


def response_from_openai(response: "ChatCompletion") -> ProviderResponse:
    """Convert an OpenAI ``ChatCompletion`` to a ``ProviderResponse``."""
    return ProviderResponse(
        parts=extract_parts_from_response(response),
        usage=extract_usage(response),
        finish_reason=extract_finish_reason(response),
        raw=response,
        thinking=extract_reasoning_from_response(response),
    )


# ==================== Streaming Helpers ====================

def map_finish_reason(reason: Optional[str]) -> FinishReason:
    """Map an OpenAI streaming finish reason to a ``FinishReason``."""
    if not reason:
        return FinishReason.UNKNOWN

    reason_lower = reason.lower()
    if reason_lower == "stop":
        return FinishReason.STOP
    elif reason_lower in ("length", "max_tokens"):
        return FinishReason.MAX_TOKENS
    elif reason_lower in ("tool_calls", "function_call"):
        return FinishReason.TOOL_USE
    elif reason_lower == "content_filter":
        return FinishReason.SAFETY

    return FinishReason.UNKNOWN


# ==================== Serialization ====================

def serialize_message(message: Message) -> Dict[str, Any]:
    """Serialize a ``Message`` to a JSON-friendly dict."""
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
        elif part.inline_data is not None:
            parts.append({
                "type": "inline_data",
                "mime_type": part.inline_data.get("mime_type"),
                "data": (
                    base64.b64encode(part.inline_data.get("data", b"")).decode("utf-8")
                    if part.inline_data.get("data")
                    else None
                ),
            })
        elif part.thought is not None:
            parts.append({"type": "thought", "thought": part.thought})

    return {
        "role": message.role.value,
        "parts": parts,
    }


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a dict back to a ``Message``."""
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
                name=p["name"],
                result=p.get("result"),
                is_error=p.get("is_error", False),
            )))
        elif ptype == "inline_data":
            raw_data = None
            if p.get("data"):
                raw_data = base64.b64decode(p["data"])
            parts.append(Part(inline_data={
                "mime_type": p.get("mime_type"),
                "data": raw_data,
            }))
        elif ptype == "thought":
            parts.append(Part(thought=p.get("thought", "")))

    return Message(
        role=Role(data["role"]),
        parts=parts,
    )


def serialize_history(history: List[Message]) -> str:
    """Serialize history to a JSON string."""
    return json.dumps([serialize_message(m) for m in history])


def deserialize_history(data: str) -> List[Message]:
    """Deserialize a JSON string back to a list of ``Message`` objects."""
    items = json.loads(data)
    return [deserialize_message(m) for m in items]
