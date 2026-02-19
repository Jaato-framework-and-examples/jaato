"""Converters between internal types and Antigravity API format.

This module handles bidirectional conversion between provider-agnostic
types (Message, ToolSchema, etc.) and the Antigravity/Generative Language
API JSON format.

The Antigravity API uses the same format as Google's Generative Language API,
but accessed through a different endpoint with OAuth authentication.
"""

import base64
import json
import uuid
from typing import Any, Dict, List, Optional, Union

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
)


# ==================== Role Conversion ====================


def role_to_api(role: Role) -> str:
    """Convert internal Role to API role string."""
    mapping = {
        Role.USER: "user",
        Role.MODEL: "model",
        Role.TOOL: "user",  # Tool responses are sent as user
    }
    return mapping.get(role, "user")


def role_from_api(role: str) -> Role:
    """Convert API role string to internal Role."""
    mapping = {
        "user": Role.USER,
        "model": Role.MODEL,
    }
    return mapping.get(role, Role.USER)


# ==================== ToolSchema Conversion ====================


def tool_schema_to_api(schema: ToolSchema) -> Dict[str, Any]:
    """Convert ToolSchema to API function declaration format."""
    return {
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters,
    }


def tool_schemas_to_api(schemas: Optional[List[ToolSchema]]) -> Optional[List[Dict[str, Any]]]:
    """Convert list of ToolSchemas to API tools format."""
    if not schemas:
        return None
    return [{
        "functionDeclarations": [tool_schema_to_api(s) for s in schemas]
    }]


# ==================== Part Conversion ====================


def part_to_api(part: Part) -> Optional[Dict[str, Any]]:
    """Convert internal Part to API part format.

    Returns None if the part type is not representable.
    """
    if part.text is not None:
        return {"text": part.text}

    if part.function_call is not None:
        fc = part.function_call
        return {
            "functionCall": {
                "name": fc.name,
                "args": fc.args,
            }
        }

    if part.function_response is not None:
        fr = part.function_response
        response = fr.result if isinstance(fr.result, dict) else {"result": fr.result}
        return {
            "functionResponse": {
                "name": fr.name,
                "response": response,
            }
        }

    if part.inline_data is not None:
        mime_type = part.inline_data.get("mime_type", "image/png")
        data = part.inline_data.get("data", b"")
        if isinstance(data, bytes):
            data = base64.b64encode(data).decode("utf-8")
        return {
            "inlineData": {
                "mimeType": mime_type,
                "data": data,
            }
        }

    if part.thought is not None:
        return {"thought": part.thought}

    return None


def part_from_api(part_data: Dict[str, Any]) -> Part:
    """Convert API part to internal Part."""
    # Text part
    if "text" in part_data:
        return Part(text=part_data["text"])

    # Function call part
    if "functionCall" in part_data:
        fc = part_data["functionCall"]
        call_id = str(uuid.uuid4())[:8]
        return Part(function_call=FunctionCall(
            id=call_id,
            name=fc.get("name", ""),
            args=fc.get("args", {}),
        ))

    # Function response part
    if "functionResponse" in part_data:
        fr = part_data["functionResponse"]
        return Part(function_response=ToolResult(
            call_id="",
            name=fr.get("name", ""),
            result=fr.get("response", {}),
        ))

    # Inline data
    if "inlineData" in part_data:
        inline = part_data["inlineData"]
        data = inline.get("data", "")
        if isinstance(data, str):
            data = base64.b64decode(data)
        return Part(inline_data={
            "mime_type": inline.get("mimeType", "application/octet-stream"),
            "data": data,
        })

    # Thought part (thinking/reasoning)
    if "thought" in part_data:
        return Part(thought=part_data["thought"])

    # Unknown part type
    return Part(text="")


# ==================== Message Conversion ====================


def message_to_api(message: Message) -> Dict[str, Any]:
    """Convert internal Message to API content format."""
    parts = []
    for part in message.parts:
        api_part = part_to_api(part)
        if api_part:
            parts.append(api_part)

    return {
        "role": role_to_api(message.role),
        "parts": parts,
    }


def message_from_api(content: Dict[str, Any]) -> Message:
    """Convert API content to internal Message."""
    role = role_from_api(content.get("role", "user"))
    parts = []

    for part_data in content.get("parts", []):
        part = part_from_api(part_data)
        parts.append(part)

    return Message(role=role, parts=parts)


def messages_to_api(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert internal message list to API contents format."""
    return [message_to_api(msg) for msg in messages]


def messages_from_api(contents: List[Dict[str, Any]]) -> List[Message]:
    """Convert API contents to internal message list."""
    return [message_from_api(c) for c in contents]


# ==================== Response Conversion ====================


def finish_reason_from_api(reason: Optional[str]) -> FinishReason:
    """Convert API finish reason to internal FinishReason."""
    if reason is None:
        return FinishReason.UNKNOWN

    # Normalize to uppercase for comparison
    reason_upper = reason.upper()

    mapping = {
        "STOP": FinishReason.STOP,
        "MAX_TOKENS": FinishReason.MAX_TOKENS,
        "SAFETY": FinishReason.SAFETY,
        "RECITATION": FinishReason.SAFETY,
        "STOP_SEQUENCE": FinishReason.STOP,
        # Function calling reasons
        "TOOL_USE": FinishReason.TOOL_USE,
        "FUNCTION_CALL": FinishReason.TOOL_USE,
        # Error reasons
        "ERROR": FinishReason.ERROR,
        "BLOCKLIST": FinishReason.SAFETY,
        "PROHIBITED_CONTENT": FinishReason.SAFETY,
    }

    return mapping.get(reason_upper, FinishReason.UNKNOWN)


def response_from_api(response_data: Dict[str, Any]) -> ProviderResponse:
    """Convert API response to ProviderResponse.

    Handles both streaming and non-streaming response formats.
    """
    parts: List[Part] = []
    thinking_text: List[str] = []

    # Extract candidates
    candidates = response_data.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        content = candidate.get("content", {})

        for part_data in content.get("parts", []):
            part = part_from_api(part_data)

            # Collect thinking parts separately
            if part.thought:
                thinking_text.append(part.thought)
            else:
                parts.append(part)

        # Get finish reason
        finish_reason = finish_reason_from_api(candidate.get("finishReason"))

        # Check if there are function calls - override finish reason
        has_function_calls = any(p.function_call for p in parts)
        if has_function_calls:
            finish_reason = FinishReason.TOOL_USE
    else:
        finish_reason = FinishReason.UNKNOWN

    # Extract usage metadata
    usage_metadata = response_data.get("usageMetadata", {})
    usage = TokenUsage(
        prompt_tokens=usage_metadata.get("promptTokenCount", 0),
        output_tokens=usage_metadata.get("candidatesTokenCount", 0),
        total_tokens=usage_metadata.get("totalTokenCount", 0),
        reasoning_tokens=usage_metadata.get("thoughtsTokenCount", 0),
    )

    # Build thinking string
    thinking = "\n".join(thinking_text) if thinking_text else None

    return ProviderResponse(
        parts=parts,
        usage=usage,
        finish_reason=finish_reason,
        raw=response_data,
        thinking=thinking,
    )


# ==================== History Serialization ====================


def serialize_history(history: List[Message]) -> str:
    """Serialize history to JSON string for persistence."""
    data = []
    for msg in history:
        msg_data = {
            "role": msg.role.value,
            "parts": [],
        }
        for part in msg.parts:
            part_data = {}
            if part.text is not None:
                part_data["text"] = part.text
            if part.function_call is not None:
                part_data["function_call"] = {
                    "id": part.function_call.id,
                    "name": part.function_call.name,
                    "args": part.function_call.args,
                }
            if part.function_response is not None:
                part_data["function_response"] = {
                    "call_id": part.function_response.call_id,
                    "name": part.function_response.name,
                    "result": part.function_response.result,
                    "is_error": part.function_response.is_error,
                }
            if part.inline_data is not None:
                # Base64 encode binary data
                inline = part.inline_data.copy()
                if isinstance(inline.get("data"), bytes):
                    inline["data"] = base64.b64encode(inline["data"]).decode("utf-8")
                part_data["inline_data"] = inline
            if part.thought is not None:
                part_data["thought"] = part.thought
            if part_data:
                msg_data["parts"].append(part_data)
        data.append(msg_data)
    return json.dumps(data)


def deserialize_history(data: str) -> List[Message]:
    """Deserialize history from JSON string."""
    messages = []
    for msg_data in json.loads(data):
        role = Role(msg_data["role"])
        parts = []
        for part_data in msg_data.get("parts", []):
            if "text" in part_data:
                parts.append(Part(text=part_data["text"]))
            elif "function_call" in part_data:
                fc = part_data["function_call"]
                parts.append(Part(function_call=FunctionCall(
                    id=fc["id"],
                    name=fc["name"],
                    args=fc["args"],
                )))
            elif "function_response" in part_data:
                fr = part_data["function_response"]
                parts.append(Part(function_response=ToolResult(
                    call_id=fr["call_id"],
                    name=fr["name"],
                    result=fr["result"],
                    is_error=fr.get("is_error", False),
                )))
            elif "inline_data" in part_data:
                inline = part_data["inline_data"]
                # Decode base64 data
                if isinstance(inline.get("data"), str):
                    inline["data"] = base64.b64decode(inline["data"])
                parts.append(Part(inline_data=inline))
            elif "thought" in part_data:
                parts.append(Part(thought=part_data["thought"]))
        messages.append(Message(role=role, parts=parts))
    return messages


# ==================== SSE Stream Parsing ====================


def parse_sse_event(line: str) -> Optional[Dict[str, Any]]:
    """Parse a Server-Sent Event line.

    Args:
        line: A line from the SSE stream.

    Returns:
        Parsed JSON data if this is a data event, None otherwise.
    """
    line = line.strip()
    if not line:
        return None

    if line.startswith("data: "):
        data_str = line[6:]
        if data_str == "[DONE]":
            return {"done": True}
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            return None

    return None


def extract_text_from_stream_chunk(chunk_data: Dict[str, Any]) -> Optional[str]:
    """Extract text content from a streaming chunk.

    Args:
        chunk_data: Parsed JSON data from SSE event.

    Returns:
        Text content if present, None otherwise.
    """
    candidates = chunk_data.get("candidates", [])
    if not candidates:
        return None

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    for part in parts:
        if "text" in part:
            return part["text"]

    return None


def extract_thinking_from_stream_chunk(chunk_data: Dict[str, Any]) -> Optional[str]:
    """Extract thinking content from a streaming chunk.

    Args:
        chunk_data: Parsed JSON data from SSE event.

    Returns:
        Thinking content if present, None otherwise.
    """
    candidates = chunk_data.get("candidates", [])
    if not candidates:
        return None

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    for part in parts:
        if "thought" in part:
            return part["thought"]

    return None


def extract_function_calls_from_stream_chunk(
    chunk_data: Dict[str, Any]
) -> List[FunctionCall]:
    """Extract function calls from a streaming chunk.

    Args:
        chunk_data: Parsed JSON data from SSE event.

    Returns:
        List of function calls found in the chunk.
    """
    calls = []
    candidates = chunk_data.get("candidates", [])
    if not candidates:
        return calls

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            call_id = str(uuid.uuid4())[:8]
            calls.append(FunctionCall(
                id=call_id,
                name=fc.get("name", ""),
                args=fc.get("args", {}),
            ))

    return calls


def extract_usage_from_stream_chunk(chunk_data: Dict[str, Any]) -> Optional[TokenUsage]:
    """Extract usage metadata from a streaming chunk.

    Usage is typically only present in the final chunk.

    Args:
        chunk_data: Parsed JSON data from SSE event.

    Returns:
        TokenUsage if present, None otherwise.
    """
    usage_metadata = chunk_data.get("usageMetadata")
    if not usage_metadata:
        return None

    return TokenUsage(
        prompt_tokens=usage_metadata.get("promptTokenCount", 0),
        output_tokens=usage_metadata.get("candidatesTokenCount", 0),
        total_tokens=usage_metadata.get("totalTokenCount", 0),
        reasoning_tokens=usage_metadata.get("thoughtsTokenCount", 0),
    )


def extract_finish_reason_from_stream_chunk(
    chunk_data: Dict[str, Any]
) -> Optional[FinishReason]:
    """Extract finish reason from a streaming chunk.

    Args:
        chunk_data: Parsed JSON data from SSE event.

    Returns:
        FinishReason if present, None otherwise.
    """
    candidates = chunk_data.get("candidates", [])
    if not candidates:
        return None

    reason = candidates[0].get("finishReason")
    if reason:
        return finish_reason_from_api(reason)

    return None


# ==================== Request Building ====================


def build_generate_request(
    contents: List[Dict[str, Any]],
    system_instruction: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a generateContent request body.

    Args:
        contents: List of content objects (converted messages).
        system_instruction: Optional system prompt.
        tools: Optional list of tool declarations.
        generation_config: Optional generation configuration.

    Returns:
        Request body dictionary.
    """
    request: Dict[str, Any] = {
        "contents": contents,
    }

    if system_instruction:
        request["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    if tools:
        request["tools"] = tools

    if generation_config:
        request["generationConfig"] = generation_config

    return request


def build_generation_config(
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    thinking_config: Optional[Dict[str, Any]] = None,
    response_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build generation configuration.

    Args:
        max_output_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        thinking_config: Thinking/reasoning configuration.
        response_schema: JSON schema for structured output.

    Returns:
        Generation config dictionary.
    """
    config: Dict[str, Any] = {}

    if max_output_tokens is not None:
        config["maxOutputTokens"] = max_output_tokens

    if temperature is not None:
        config["temperature"] = temperature

    if top_p is not None:
        config["topP"] = top_p

    if top_k is not None:
        config["topK"] = top_k

    if thinking_config is not None:
        config["thinkingConfig"] = thinking_config

    if response_schema is not None:
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = response_schema

    return config
