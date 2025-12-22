"""Converters between internal types and GitHub Models/Azure AI Inference SDK types.

This module handles bidirectional conversion between provider-agnostic
types (Message, ToolSchema, etc.) and the azure-ai-inference SDK types
which follow the OpenAI chat completions format.
"""

import base64
import json
import uuid
from typing import Any, Dict, List, Optional

from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletions,
    ChatCompletionsToolCall,
    ChatCompletionsToolDefinition,
    ChatRequestMessage,
    FunctionCall as AzureFunctionCall,
    FunctionDefinition,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

from ..types import (
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

def role_to_sdk(role: Role) -> str:
    """Convert internal Role to SDK role string."""
    mapping = {
        Role.USER: "user",
        Role.MODEL: "assistant",
        Role.TOOL: "tool",
    }
    return mapping.get(role, "user")


def role_from_sdk(role: str) -> Role:
    """Convert SDK role string to internal Role."""
    mapping = {
        "user": Role.USER,
        "assistant": Role.MODEL,
        "system": Role.USER,  # System messages are treated as context
        "tool": Role.TOOL,
    }
    return mapping.get(role, Role.USER)


# ==================== ToolSchema Conversion ====================

def tool_schema_to_sdk(schema: ToolSchema) -> ChatCompletionsToolDefinition:
    """Convert ToolSchema to SDK ChatCompletionsToolDefinition."""
    return ChatCompletionsToolDefinition(
        function=FunctionDefinition(
            name=schema.name,
            description=schema.description,
            parameters=schema.parameters,
        )
    )


def tool_schema_from_sdk(tool_def: ChatCompletionsToolDefinition) -> ToolSchema:
    """Convert SDK ChatCompletionsToolDefinition to ToolSchema."""
    func = tool_def.function
    return ToolSchema(
        name=func.name,
        description=func.description or "",
        parameters=func.parameters or {},
    )


def tool_schemas_to_sdk(schemas: Optional[List[ToolSchema]]) -> Optional[List[ChatCompletionsToolDefinition]]:
    """Convert list of ToolSchemas to SDK tool definitions."""
    if not schemas:
        return None
    return [tool_schema_to_sdk(s) for s in schemas]


# ==================== Message Conversion ====================

def message_to_sdk(message: Message) -> ChatRequestMessage:
    """Convert internal Message to SDK ChatRequestMessage.

    The SDK uses different message classes for different roles:
    - SystemMessage for system prompts
    - UserMessage for user input
    - AssistantMessage for model responses (may include tool calls)
    - ToolMessage for tool results
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
        # Tool result message - use the first response
        fr = function_responses[0]
        result_str = json.dumps(fr.result) if not isinstance(fr.result, str) else fr.result
        return ToolMessage(
            tool_call_id=fr.call_id,
            content=result_str,
        )

    if role == Role.MODEL:
        # Assistant message - may include tool calls
        if function_calls:
            tool_calls = [
                ChatCompletionsToolCall(
                    id=fc.id,
                    function=AzureFunctionCall(
                        name=fc.name,
                        arguments=json.dumps(fc.args),
                    ),
                )
                for fc in function_calls
            ]
            return AssistantMessage(
                content=content if content else None,
                tool_calls=tool_calls,
            )
        return AssistantMessage(content=content)

    # Default to user message
    return UserMessage(content=content)


def message_from_sdk(msg: ChatRequestMessage) -> Message:
    """Convert SDK ChatRequestMessage to internal Message."""
    parts = []

    # Handle different message types
    if isinstance(msg, SystemMessage):
        if msg.content:
            parts.append(Part(text=msg.content))
        return Message(role=Role.USER, parts=parts)

    if isinstance(msg, UserMessage):
        if msg.content:
            if isinstance(msg.content, str):
                parts.append(Part(text=msg.content))
            elif isinstance(msg.content, list):
                # Handle multimodal content
                for item in msg.content:
                    if hasattr(item, 'text'):
                        parts.append(Part(text=item.text))
                    # Image parts would need special handling
        return Message(role=Role.USER, parts=parts)

    if isinstance(msg, AssistantMessage):
        if msg.content:
            parts.append(Part(text=msg.content))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                parts.append(Part(function_call=FunctionCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args,
                )))
        return Message(role=Role.MODEL, parts=parts)

    if isinstance(msg, ToolMessage):
        result = msg.content
        try:
            result = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            pass
        parts.append(Part(function_response=ToolResult(
            call_id=msg.tool_call_id,
            name="",  # Tool name not available in ToolMessage
            result=result,
        )))
        return Message(role=Role.TOOL, parts=parts)

    # Fallback
    return Message(role=Role.USER, parts=[Part(text=str(msg))])


def history_to_sdk(history: List[Message]) -> List[ChatRequestMessage]:
    """Convert internal history to SDK message list."""
    return [message_to_sdk(m) for m in (history or [])]


def history_from_sdk(history: List[ChatRequestMessage]) -> List[Message]:
    """Convert SDK message list to internal history."""
    return [message_from_sdk(m) for m in (history or [])]


# ==================== Tool Result Conversion ====================

def tool_result_to_sdk(result: ToolResult) -> ToolMessage:
    """Convert ToolResult to SDK ToolMessage."""
    content = result.result
    if result.is_error:
        content = {"error": str(result.result)}

    content_str = json.dumps(content) if not isinstance(content, str) else content

    return ToolMessage(
        tool_call_id=result.call_id,
        content=content_str,
    )


def tool_results_to_sdk(results: List[ToolResult]) -> List[ToolMessage]:
    """Convert list of ToolResults to SDK ToolMessages."""
    return [tool_result_to_sdk(r) for r in (results or [])]


# ==================== Streaming Helpers ====================

def extract_function_calls_from_stream_delta(tool_calls) -> List[FunctionCall]:
    """Extract function calls from streaming delta tool_calls.

    In streaming mode, tool calls may be spread across multiple chunks.
    Each chunk contains partial information that needs to be accumulated.

    Args:
        tool_calls: List of tool call deltas from a streaming chunk.

    Returns:
        List of FunctionCall objects (may be partial during streaming).
    """
    calls = []

    if not tool_calls:
        return calls

    for tc in tool_calls:
        # Get the tool call index and id
        tc_id = getattr(tc, 'id', None) or str(uuid.uuid4())[:8]

        # Get function info
        func = getattr(tc, 'function', None)
        if not func:
            continue

        name = getattr(func, 'name', None) or ''
        arguments = getattr(func, 'arguments', None) or ''

        # Parse arguments if present
        args = {}
        if arguments:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                # Arguments may be partial during streaming
                args = {"_partial": arguments}

        if name:  # Only add if we have a function name
            calls.append(FunctionCall(
                id=tc_id,
                name=name,
                args=args,
            ))

    return calls


# ==================== Response Conversion ====================

def extract_text_from_response(response: ChatCompletions) -> Optional[str]:
    """Extract text from SDK response."""
    if not response or not response.choices:
        return None

    texts = []
    for choice in response.choices:
        if choice.message and choice.message.content:
            texts.append(choice.message.content)

    return "".join(texts) if texts else None


def extract_function_calls_from_response(response: ChatCompletions) -> List[FunctionCall]:
    """Extract function calls from SDK response."""
    calls = []

    if not response or not response.choices:
        return calls

    for choice in response.choices:
        if choice.message and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                calls.append(FunctionCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args,
                ))

    return calls


def extract_parts_from_response(response: ChatCompletions) -> List[Part]:
    """Extract parts from SDK response, preserving order of text and function calls.

    In OpenAI/GitHub Models API, text content comes before tool_calls in the message,
    so we preserve that ordering in the parts list.
    """
    parts = []

    if not response or not response.choices:
        return parts

    for choice in response.choices:
        if not choice.message:
            continue

        # Text content comes first
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
                    name=tc.function.name,
                    args=args,
                )
                parts.append(Part.from_function_call(fc))

    return parts


def extract_finish_reason_from_response(response: ChatCompletions) -> FinishReason:
    """Extract finish reason from SDK response."""
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


def extract_usage_from_response(response: ChatCompletions) -> TokenUsage:
    """Extract token usage from SDK response."""
    usage = TokenUsage()

    if not response or not response.usage:
        return usage

    usage.prompt_tokens = response.usage.prompt_tokens or 0
    usage.output_tokens = response.usage.completion_tokens or 0
    usage.total_tokens = response.usage.total_tokens or 0

    return usage


def response_from_sdk(response: ChatCompletions) -> ProviderResponse:
    """Convert SDK ChatCompletions to internal ProviderResponse."""
    return ProviderResponse(
        parts=extract_parts_from_response(response),
        usage=extract_usage_from_response(response),
        finish_reason=extract_finish_reason_from_response(response),
        raw=response,
    )


# ==================== Serialization ====================
# For session persistence - converts internal types to/from JSON

def serialize_message(message: Message) -> Dict[str, Any]:
    """Serialize a Message to a dictionary for JSON storage."""
    parts = []
    for part in message.parts:
        if part.text is not None:
            parts.append({'type': 'text', 'text': part.text})
        elif part.function_call is not None:
            fc = part.function_call
            parts.append({
                'type': 'function_call',
                'id': fc.id,
                'name': fc.name,
                'args': fc.args,
            })
        elif part.function_response is not None:
            fr = part.function_response
            parts.append({
                'type': 'function_response',
                'call_id': fr.call_id,
                'name': fr.name,
                'result': fr.result,
                'is_error': fr.is_error,
            })
        elif part.inline_data is not None:
            parts.append({
                'type': 'inline_data',
                'mime_type': part.inline_data.get('mime_type'),
                'data': base64.b64encode(part.inline_data.get('data', b'')).decode('utf-8')
                        if part.inline_data.get('data') else None,
            })
        elif part.thought is not None:
            parts.append({'type': 'thought', 'thought': part.thought})
        elif part.executable_code is not None:
            parts.append({'type': 'executable_code', 'code': part.executable_code})
        elif part.code_execution_result is not None:
            parts.append({'type': 'code_execution_result', 'output': part.code_execution_result})

    return {
        'role': message.role.value,
        'parts': parts,
    }


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a dictionary to a Message."""
    parts = []
    for p in data.get('parts', []):
        ptype = p.get('type')
        if ptype == 'text':
            parts.append(Part(text=p['text']))
        elif ptype == 'function_call':
            parts.append(Part(function_call=FunctionCall(
                id=p.get('id', ''),
                name=p['name'],
                args=p.get('args', {}),
            )))
        elif ptype == 'function_response':
            parts.append(Part(function_response=ToolResult(
                call_id=p.get('call_id', ''),
                name=p['name'],
                result=p.get('result'),
                is_error=p.get('is_error', False),
            )))
        elif ptype == 'inline_data':
            raw_data = None
            if p.get('data'):
                raw_data = base64.b64decode(p['data'])
            parts.append(Part(inline_data={
                'mime_type': p.get('mime_type'),
                'data': raw_data,
            }))
        elif ptype == 'thought':
            parts.append(Part(thought=p.get('thought', '')))
        elif ptype == 'executable_code':
            parts.append(Part(executable_code=p.get('code', '')))
        elif ptype == 'code_execution_result':
            parts.append(Part(code_execution_result=p.get('output', '')))

    return Message(
        role=Role(data['role']),
        parts=parts,
    )


def serialize_history(history: List[Message]) -> str:
    """Serialize history to JSON string."""
    return json.dumps([serialize_message(m) for m in history])


def deserialize_history(data: str) -> List[Message]:
    """Deserialize JSON string to history."""
    items = json.loads(data)
    return [deserialize_message(m) for m in items]
