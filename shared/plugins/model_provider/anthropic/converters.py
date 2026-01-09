"""Converters between internal types and Anthropic SDK types.

This module handles bidirectional conversion between provider-agnostic
types (Message, ToolSchema, etc.) and the Anthropic SDK types.

Key differences from other providers:
- Uses `input_schema` instead of `parameters` for tool definitions
- Only 2 roles: "user" and "assistant" (no "model" or "tool")
- Tool results are content blocks in user messages, not separate role
- Content is always an array of typed blocks
- Supports "thinking" blocks for extended reasoning
"""

import base64
import json
from typing import Any, Dict, List, Optional

from ..types import (
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


# ==================== Tool Schema Conversion ====================

def tool_schema_to_anthropic(schema: ToolSchema) -> Dict[str, Any]:
    """Convert ToolSchema to Anthropic tool format.

    Note: Anthropic uses `input_schema` instead of `parameters`.
    """
    return {
        "name": schema.name,
        "description": schema.description,
        "input_schema": schema.parameters,
    }


def tool_schemas_to_anthropic(schemas: Optional[List[ToolSchema]]) -> Optional[List[Dict[str, Any]]]:
    """Convert list of ToolSchemas to Anthropic tool definitions."""
    if not schemas:
        return None
    return [tool_schema_to_anthropic(s) for s in schemas]


# ==================== Message Conversion ====================

def role_to_anthropic(role: Role) -> str:
    """Convert internal Role to Anthropic role string.

    Anthropic only has "user" and "assistant" roles.
    Tool results go in user messages with tool_result blocks.
    """
    if role == Role.MODEL:
        return "assistant"
    # USER and TOOL both map to "user"
    return "user"


def role_from_anthropic(role: str) -> Role:
    """Convert Anthropic role string to internal Role."""
    if role == "assistant":
        return Role.MODEL
    return Role.USER


def part_to_anthropic_content_block(part: Part) -> Optional[Dict[str, Any]]:
    """Convert a Part to an Anthropic content block.

    Returns None if the part type is not supported.
    """
    if part.text is not None:
        return {"type": "text", "text": part.text}

    if part.function_call is not None:
        fc = part.function_call
        return {
            "type": "tool_use",
            "id": fc.id,
            "name": fc.name,
            "input": fc.args,
        }

    if part.function_response is not None:
        fr = part.function_response
        # Result can be string or dict - Anthropic expects string or list of blocks
        content: Any
        if isinstance(fr.result, str):
            content = fr.result
        else:
            content = json.dumps(fr.result)

        block: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": fr.call_id,
            "content": content,
        }

        if fr.is_error:
            block["is_error"] = True

        # Handle attachments (images in tool results)
        if fr.attachments:
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for att in fr.attachments:
                if att.mime_type.startswith("image/"):
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": att.mime_type,
                            "data": base64.b64encode(att.data).decode("utf-8"),
                        }
                    })
            block["content"] = content_blocks

        return block

    if part.inline_data is not None:
        mime_type = part.inline_data.get("mime_type", "image/png")
        data = part.inline_data.get("data", b"")
        if isinstance(data, bytes):
            data = base64.b64encode(data).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data,
            }
        }

    return None


def message_to_anthropic(message: Message) -> Dict[str, Any]:
    """Convert internal Message to Anthropic format.

    Returns a dict with 'role' and 'content' (array of blocks).
    """
    content = []

    for part in message.parts:
        block = part_to_anthropic_content_block(part)
        if block:
            content.append(block)

    return {
        "role": role_to_anthropic(message.role),
        "content": content,
    }


def messages_to_anthropic(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert internal message list to Anthropic format.

    Handles merging consecutive same-role messages (Anthropic requirement).
    """
    if not messages:
        return []

    result = []
    current_role = None
    current_content: List[Dict[str, Any]] = []

    for msg in messages:
        anthropic_msg = message_to_anthropic(msg)
        role = anthropic_msg["role"]

        if role == current_role:
            # Merge content into current message
            current_content.extend(anthropic_msg["content"])
        else:
            # Flush previous message
            if current_content:
                result.append({
                    "role": current_role,
                    "content": current_content,
                })
            # Start new message
            current_role = role
            current_content = anthropic_msg["content"]

    # Flush last message
    if current_content:
        result.append({
            "role": current_role,
            "content": current_content,
        })

    return result


def validate_tool_use_pairing(messages: List[Message]) -> List[Message]:
    """Validate and repair tool_use/tool_result pairing in history.

    Anthropic requires every tool_use block to have a corresponding tool_result
    immediately following it. This function:
    1. Tracks all tool_use IDs from assistant messages
    2. Removes them when matching tool_result is found
    3. If unpaired tool_use blocks exist before a user message (text),
       removes the assistant message containing them

    This is a defensive measure to prevent API errors from corrupted history
    due to cancellation or exceptions during streaming.

    Args:
        messages: List of internal Message objects

    Returns:
        Cleaned list of messages with unpaired tool_use blocks removed
    """
    if not messages:
        return messages

    result = []
    pending_tool_use_ids: set = set()  # Track tool_use IDs awaiting results
    pending_assistant_msg_idx: Optional[int] = None  # Index of assistant msg with pending tool calls

    for msg in messages:
        if msg.role == Role.MODEL:
            # Check if this assistant message has function calls (tool_use)
            has_tool_use = any(p.function_call is not None for p in msg.parts)
            if has_tool_use:
                # If we already have pending tool_use from a previous assistant message,
                # that's an error - remove it
                if pending_tool_use_ids and pending_assistant_msg_idx is not None:
                    # Remove the previous assistant message that has unpaired tool_use
                    result = result[:pending_assistant_msg_idx] + result[pending_assistant_msg_idx + 1:]
                    # Adjust index for removal
                    pending_assistant_msg_idx = None
                    pending_tool_use_ids.clear()

                # Track this message and its tool_use IDs
                pending_assistant_msg_idx = len(result)
                for p in msg.parts:
                    if p.function_call is not None:
                        pending_tool_use_ids.add(p.function_call.id)

            result.append(msg)

        elif msg.role == Role.TOOL:
            # Tool results - match them with pending tool_use
            for p in msg.parts:
                if p.function_response is not None:
                    pending_tool_use_ids.discard(p.function_response.call_id)

            # If all tool_use IDs are resolved, clear tracking
            if not pending_tool_use_ids:
                pending_assistant_msg_idx = None

            result.append(msg)

        elif msg.role == Role.USER:
            # User message - if we still have pending tool_use, we have a problem
            # The assistant message with unpaired tool_use must be removed
            if pending_tool_use_ids and pending_assistant_msg_idx is not None:
                # Remove the assistant message that has unpaired tool_use
                result = result[:pending_assistant_msg_idx] + result[pending_assistant_msg_idx + 1:]
                pending_tool_use_ids.clear()
                pending_assistant_msg_idx = None

            result.append(msg)

        else:
            result.append(msg)

    # Final check - if history ends with unpaired tool_use, remove that assistant message
    if pending_tool_use_ids and pending_assistant_msg_idx is not None:
        result = result[:pending_assistant_msg_idx] + result[pending_assistant_msg_idx + 1:]

    return result


def content_block_to_part(block: Dict[str, Any]) -> Optional[Part]:
    """Convert an Anthropic content block to a Part."""
    block_type = block.get("type")

    if block_type == "text":
        return Part(text=block.get("text", ""))

    if block_type == "tool_use":
        return Part(function_call=FunctionCall(
            id=block.get("id", ""),
            name=block.get("name", ""),
            args=block.get("input", {}),
        ))

    if block_type == "tool_result":
        result = block.get("content", "")
        # Try to parse JSON result
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass
        return Part(function_response=ToolResult(
            call_id=block.get("tool_use_id", ""),
            name="",  # Name not available in tool_result
            result=result,
            is_error=block.get("is_error", False),
        ))

    if block_type == "image":
        source = block.get("source", {})
        data = source.get("data", "")
        if isinstance(data, str):
            data = base64.b64decode(data)
        return Part(inline_data={
            "mime_type": source.get("media_type", "image/png"),
            "data": data,
        })

    # Thinking blocks are handled separately in response conversion
    if block_type == "thinking":
        # Store thinking in text with a marker (or we handle it specially)
        return None  # Handled at response level

    return None


def message_from_anthropic(msg: Dict[str, Any]) -> Message:
    """Convert Anthropic message dict to internal Message."""
    role = role_from_anthropic(msg.get("role", "user"))
    parts = []

    content = msg.get("content", [])
    if isinstance(content, str):
        # Simple string content
        parts.append(Part(text=content))
    elif isinstance(content, list):
        for block in content:
            part = content_block_to_part(block)
            if part:
                parts.append(part)

    return Message(role=role, parts=parts)


# ==================== Tool Result Conversion ====================

def tool_result_to_anthropic(result: ToolResult) -> Dict[str, Any]:
    """Convert ToolResult to Anthropic tool_result content block."""
    content: Any
    if isinstance(result.result, str):
        content = result.result
    else:
        content = json.dumps(result.result)

    block: Dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": content,
    }

    if result.is_error:
        block["is_error"] = True

    # Handle attachments
    if result.attachments:
        content_blocks = []
        if content:
            content_blocks.append({"type": "text", "text": str(content)})
        for att in result.attachments:
            if att.mime_type.startswith("image/"):
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": att.mime_type,
                        "data": base64.b64encode(att.data).decode("utf-8"),
                    }
                })
        block["content"] = content_blocks

    return block


def tool_results_to_anthropic(results: List[ToolResult]) -> List[Dict[str, Any]]:
    """Convert list of ToolResults to Anthropic tool_result blocks."""
    return [tool_result_to_anthropic(r) for r in results]


# ==================== Response Conversion ====================

def extract_text_from_response(response: Any) -> Optional[str]:
    """Extract text from Anthropic response."""
    if not response or not hasattr(response, "content"):
        return None

    texts = []
    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            texts.append(block.text)

    return "".join(texts) if texts else None


def extract_thinking_from_response(response: Any) -> Optional[str]:
    """Extract thinking/reasoning from Anthropic response."""
    if not response or not hasattr(response, "content"):
        return None

    thinking_parts = []
    for block in response.content:
        if hasattr(block, "type") and block.type == "thinking":
            thinking_parts.append(block.thinking)

    return "\n".join(thinking_parts) if thinking_parts else None


def extract_function_calls_from_response(response: Any) -> List[FunctionCall]:
    """Extract function calls from Anthropic response."""
    calls = []

    if not response or not hasattr(response, "content"):
        return calls

    for block in response.content:
        if hasattr(block, "type") and block.type == "tool_use":
            calls.append(FunctionCall(
                id=block.id,
                name=block.name,
                args=block.input if hasattr(block, "input") else {},
            ))

    return calls


def extract_finish_reason_from_response(response: Any) -> FinishReason:
    """Extract finish reason from Anthropic response."""
    if not response or not hasattr(response, "stop_reason"):
        return FinishReason.UNKNOWN

    reason = response.stop_reason
    if reason == "end_turn":
        return FinishReason.STOP
    elif reason == "tool_use":
        return FinishReason.TOOL_USE
    elif reason == "max_tokens":
        return FinishReason.MAX_TOKENS
    elif reason == "stop_sequence":
        return FinishReason.STOP

    return FinishReason.UNKNOWN


def extract_usage_from_response(response: Any) -> TokenUsage:
    """Extract token usage from Anthropic response.

    Extracts standard token counts plus cache token information
    when prompt caching is enabled.
    """
    usage = TokenUsage()

    if not response or not hasattr(response, "usage"):
        return usage

    resp_usage = response.usage
    usage.prompt_tokens = getattr(resp_usage, "input_tokens", 0)
    usage.output_tokens = getattr(resp_usage, "output_tokens", 0)
    usage.total_tokens = usage.prompt_tokens + usage.output_tokens

    # Extract cache token information (prompt caching)
    cache_creation = getattr(resp_usage, "cache_creation_input_tokens", None)
    cache_read = getattr(resp_usage, "cache_read_input_tokens", None)
    if cache_creation is not None and cache_creation > 0:
        usage.cache_creation_tokens = cache_creation
    if cache_read is not None and cache_read > 0:
        usage.cache_read_tokens = cache_read

    return usage


def response_from_anthropic(response: Any) -> ProviderResponse:
    """Convert Anthropic response to internal ProviderResponse.

    Uses parts-based API for proper text/function_call interleaving.
    """
    parts = []

    # Extract parts in order from response content
    if response and hasattr(response, "content"):
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    parts.append(Part.from_text(block.text))
                elif block.type == "tool_use":
                    fc = FunctionCall(
                        id=block.id,
                        name=block.name,
                        args=block.input if hasattr(block, "input") else {},
                    )
                    parts.append(Part.from_function_call(fc))
                # Thinking blocks are extracted separately

    return ProviderResponse(
        parts=parts,
        usage=extract_usage_from_response(response),
        finish_reason=extract_finish_reason_from_response(response),
        raw=response,
        thinking=extract_thinking_from_response(response),
    )


# ==================== Streaming Helpers ====================


def extract_text_from_stream_event(event: Any) -> Optional[str]:
    """Extract text delta from a streaming event.

    Anthropic streaming uses various event types:
    - content_block_delta with text_delta contains text chunks
    - Other event types don't contain text

    Returns:
        Text chunk if this is a text delta event, None otherwise.
    """
    if not event:
        return None

    # Check event type
    event_type = getattr(event, "type", None)

    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        if delta:
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                return getattr(delta, "text", None)

    return None


def extract_input_json_from_stream_event(event: Any) -> Optional[str]:
    """Extract input JSON delta from a streaming event (for tool calls).

    Returns:
        JSON string chunk if this is an input_json_delta event, None otherwise.
    """
    if not event:
        return None

    event_type = getattr(event, "type", None)

    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        if delta:
            delta_type = getattr(delta, "type", None)
            if delta_type == "input_json_delta":
                return getattr(delta, "partial_json", None)

    return None


def extract_content_block_start(event: Any) -> Optional[Dict[str, Any]]:
    """Extract content block info from content_block_start event.

    Returns:
        Dict with block type and info, or None.
    """
    if not event:
        return None

    event_type = getattr(event, "type", None)

    if event_type == "content_block_start":
        index = getattr(event, "index", 0)
        content_block = getattr(event, "content_block", None)
        if content_block:
            block_type = getattr(content_block, "type", None)
            if block_type == "tool_use":
                return {
                    "type": "tool_use",
                    "index": index,
                    "id": getattr(content_block, "id", ""),
                    "name": getattr(content_block, "name", ""),
                }
            elif block_type == "text":
                return {
                    "type": "text",
                    "index": index,
                }
            elif block_type == "thinking":
                return {
                    "type": "thinking",
                    "index": index,
                }

    return None


def extract_message_delta(event: Any) -> Optional[Dict[str, Any]]:
    """Extract message delta info (stop reason, usage) from message_delta event.

    Returns:
        Dict with stop_reason and usage info, or None.
    """
    if not event:
        return None

    event_type = getattr(event, "type", None)

    if event_type == "message_delta":
        result: Dict[str, Any] = {}

        delta = getattr(event, "delta", None)
        if delta:
            stop_reason = getattr(delta, "stop_reason", None)
            if stop_reason:
                result["stop_reason"] = stop_reason

        usage = getattr(event, "usage", None)
        if usage:
            result["usage"] = TokenUsage(
                prompt_tokens=0,  # Not available in delta
                output_tokens=getattr(usage, "output_tokens", 0),
                total_tokens=getattr(usage, "output_tokens", 0),
            )

        return result if result else None

    return None


def extract_message_start(event: Any) -> Optional[TokenUsage]:
    """Extract initial usage from message_start event.

    Extracts standard token counts plus cache token information
    when prompt caching is enabled.

    Returns:
        TokenUsage with input tokens and cache info, or None.
    """
    if not event:
        return None

    event_type = getattr(event, "type", None)

    if event_type == "message_start":
        message = getattr(event, "message", None)
        if message:
            usage = getattr(message, "usage", None)
            if usage:
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)

                # Extract cache token information
                cache_creation = getattr(usage, "cache_creation_input_tokens", None)
                cache_read = getattr(usage, "cache_read_input_tokens", None)

                token_usage = TokenUsage(
                    prompt_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )

                if cache_creation is not None and cache_creation > 0:
                    token_usage.cache_creation_tokens = cache_creation
                if cache_read is not None and cache_read > 0:
                    token_usage.cache_read_tokens = cache_read

                return token_usage

    return None


def extract_thinking_from_stream_event(event: Any) -> Optional[str]:
    """Extract thinking delta from a streaming event.

    Returns:
        Thinking text chunk if this is a thinking_delta event, None otherwise.
    """
    if not event:
        return None

    event_type = getattr(event, "type", None)

    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        if delta:
            delta_type = getattr(delta, "type", None)
            if delta_type == "thinking_delta":
                return getattr(delta, "thinking", None)

    return None


# ==================== Serialization ====================
# For session persistence - converts internal types to/from JSON

def serialize_message(message: Message) -> Dict[str, Any]:
    """Serialize a Message to a dictionary for JSON storage."""
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
            data = part.inline_data.get("data", b"")
            if isinstance(data, bytes):
                data = base64.b64encode(data).decode("utf-8")
            parts.append({
                "type": "inline_data",
                "mime_type": part.inline_data.get("mime_type"),
                "data": data,
            })

    return {
        "role": message.role.value,
        "parts": parts,
    }


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a dictionary to a Message."""
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
                name=p.get("name", ""),
                result=p.get("result"),
                is_error=p.get("is_error", False),
            )))
        elif ptype == "inline_data":
            raw_data = p.get("data")
            if raw_data and isinstance(raw_data, str):
                raw_data = base64.b64decode(raw_data)
            parts.append(Part(inline_data={
                "mime_type": p.get("mime_type"),
                "data": raw_data,
            }))

    return Message(
        role=Role(data["role"]),
        parts=parts,
    )


def serialize_history(history: List[Message]) -> str:
    """Serialize history to JSON string."""
    return json.dumps([serialize_message(m) for m in history])


def deserialize_history(data: str) -> List[Message]:
    """Deserialize JSON string to history."""
    items = json.loads(data)
    return [deserialize_message(m) for m in items]
