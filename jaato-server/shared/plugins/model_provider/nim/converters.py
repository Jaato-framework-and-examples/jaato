"""Converters between internal types and OpenAI chat completions format.

This module handles bidirectional conversion between provider-agnostic
types (Message, ToolSchema, etc.) and the OpenAI SDK types used by
NVIDIA NIM's OpenAI-compatible API.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

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


# ==================== Tool Name Sanitization ====================

# OpenAI API requires function names to match ^[a-zA-Z0-9_-]{1,64}$
_OPENAI_TOOL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

# Module-level mapping from sanitized names back to original names
_tool_name_mapping: Dict[str, str] = {}


def sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to match OpenAI's pattern ^[a-zA-Z0-9_-]{1,64}$.

    Replaces invalid characters (like dots, colons, spaces) with underscores
    and truncates to 64 characters if needed.

    Args:
        name: Original tool name.

    Returns:
        Sanitized tool name safe for OpenAI-compatible APIs.
    """
    if _OPENAI_TOOL_NAME_PATTERN.match(name):
        return name
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return sanitized[:64]


def get_original_tool_name(sanitized_name: str) -> str:
    """Get the original tool name from a sanitized name.

    Args:
        sanitized_name: The sanitized tool name received from the API.

    Returns:
        Original tool name if mapping exists, otherwise returns the input unchanged.
    """
    return _tool_name_mapping.get(sanitized_name, sanitized_name)


def clear_tool_name_mapping() -> None:
    """Clear the tool name mapping. Call when tools are reconfigured."""
    _tool_name_mapping.clear()


def register_tool_name_mapping(sanitized: str, original: str) -> None:
    """Register a mapping from sanitized to original tool name.

    Args:
        sanitized: The sanitized tool name sent to the API.
        original: The original tool name used internally.
    """
    if sanitized != original:
        _tool_name_mapping[sanitized] = original


# ==================== ToolSchema Conversion ====================

def tool_schema_to_openai(schema: ToolSchema) -> Dict[str, Any]:
    """Convert ToolSchema to OpenAI tool definition dict.

    Tool names are sanitized to match OpenAI's pattern ^[a-zA-Z0-9_-]{1,64}$.
    A mapping from sanitized to original names is maintained for reverse lookup.

    Args:
        schema: Internal tool schema.

    Returns:
        Dict in OpenAI tool format.
    """
    sanitized_name = sanitize_tool_name(schema.name)
    register_tool_name_mapping(sanitized_name, schema.name)

    return {
        "type": "function",
        "function": {
            "name": sanitized_name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


def tool_schemas_to_openai(schemas: Optional[List[ToolSchema]]) -> Optional[List[Dict[str, Any]]]:
    """Convert list of ToolSchemas to OpenAI tool definitions.

    Args:
        schemas: List of internal tool schemas.

    Returns:
        List of OpenAI tool dicts, or None if no schemas.
    """
    if not schemas:
        return None
    return [tool_schema_to_openai(s) for s in schemas]


# ==================== Message Conversion ====================

def message_to_openai(message: Message) -> Dict[str, Any]:
    """Convert internal Message to OpenAI message dict.

    Args:
        message: Internal message.

    Returns:
        Dict in OpenAI chat message format.
    """
    role = message.role

    # Collect text content
    text_parts = [p.text for p in message.parts if p.text]
    content = "".join(text_parts) if text_parts else ""

    # Check for function calls (assistant messages)
    function_calls = [p.function_call for p in message.parts if p.function_call]

    # Check for function responses (tool messages)
    function_responses = [p.function_response for p in message.parts if p.function_response]

    if function_responses:
        # Tool result message
        fr = function_responses[0]
        result_str = json.dumps(fr.result) if not isinstance(fr.result, str) else fr.result
        return {
            "role": "tool",
            "tool_call_id": fr.call_id,
            "content": result_str,
        }

    if role == Role.MODEL:
        msg: Dict[str, Any] = {
            "role": "assistant",
        }
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

    # Default to user message
    return {
        "role": "user",
        "content": content,
    }


def message_from_openai(msg: Dict[str, Any]) -> Message:
    """Convert OpenAI message dict to internal Message.

    Args:
        msg: Dict in OpenAI chat message format.

    Returns:
        Internal Message.
    """
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
            args = {}
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

    # User message
    content = msg.get("content", "")
    if content:
        parts.append(Part(text=content))
    return Message(role=Role.USER, parts=parts)


def history_to_openai(history: List[Message]) -> List[Dict[str, Any]]:
    """Convert internal history to OpenAI message list.

    Args:
        history: List of internal messages.

    Returns:
        List of OpenAI message dicts.
    """
    return [message_to_openai(m) for m in (history or [])]


# ==================== Response Conversion ====================

def extract_parts_from_response(response: "ChatCompletion") -> List[Part]:
    """Extract parts from OpenAI ChatCompletion response.

    Preserves order of text and function calls.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        List of Part objects.
    """
    parts = []

    if not response or not response.choices:
        return parts

    for choice in response.choices:
        if not choice.message:
            continue

        # Text content first
        if choice.message.content:
            parts.append(Part.from_text(choice.message.content))

        # Tool calls follow text
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = {}
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
    """Extract finish reason from OpenAI response.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        Internal FinishReason.
    """
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
    """Extract token usage from OpenAI response.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        TokenUsage with counts.
    """
    usage = TokenUsage()

    if not response or not response.usage:
        return usage

    usage.prompt_tokens = response.usage.prompt_tokens or 0
    usage.output_tokens = response.usage.completion_tokens or 0
    usage.total_tokens = response.usage.total_tokens or 0

    return usage


def extract_reasoning_from_response(response: "ChatCompletion") -> Optional[str]:
    """Extract reasoning/thinking content from a non-streaming response.

    Models like DeepSeek-R1 on NIM expose chain-of-thought via a
    ``reasoning_content`` field on the message.

    Args:
        response: OpenAI ChatCompletion response.

    Returns:
        Reasoning text if present, None otherwise.
    """
    if not response or not getattr(response, "choices", None):
        return None

    reasoning_parts: List[str] = []
    for choice in response.choices:
        msg = getattr(choice, "message", None)
        if not msg:
            continue
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning and isinstance(reasoning, str):
            reasoning_parts.append(reasoning)

    return "\n".join(reasoning_parts) if reasoning_parts else None


def response_from_openai(response: "ChatCompletion") -> ProviderResponse:
    """Convert OpenAI ChatCompletion to internal ProviderResponse.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        Internal ProviderResponse.
    """
    return ProviderResponse(
        parts=extract_parts_from_response(response),
        usage=extract_usage(response),
        finish_reason=extract_finish_reason(response),
        raw=response,
        thinking=extract_reasoning_from_response(response),
    )


# ==================== Streaming Helpers ====================

def map_finish_reason(reason: Optional[str]) -> FinishReason:
    """Map OpenAI finish reason string to internal FinishReason.

    Args:
        reason: Finish reason string from streaming chunk.

    Returns:
        Internal FinishReason.
    """
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
    """Serialize a Message to a dictionary for JSON storage.

    Args:
        message: Internal message.

    Returns:
        Serializable dict.
    """
    parts = []
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
                "data": base64.b64encode(part.inline_data.get("data", b"")).decode("utf-8")
                        if part.inline_data.get("data") else None,
            })
        elif part.thought is not None:
            parts.append({"type": "thought", "thought": part.thought})

    return {
        "role": message.role.value,
        "parts": parts,
    }


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a dictionary to a Message.

    Args:
        data: Serialized message dict.

    Returns:
        Internal Message.
    """
    parts = []
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
    """Serialize history to JSON string.

    Args:
        history: List of messages to serialize.

    Returns:
        JSON string representation.
    """
    return json.dumps([serialize_message(m) for m in history])


def deserialize_history(data: str) -> List[Message]:
    """Deserialize JSON string to history.

    Args:
        data: Previously serialized history string.

    Returns:
        List of Message objects.
    """
    items = json.loads(data)
    return [deserialize_message(m) for m in items]
