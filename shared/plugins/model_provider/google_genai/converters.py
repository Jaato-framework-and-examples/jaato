"""Converters between internal types and Google GenAI SDK types.

This module handles bidirectional conversion between provider-agnostic
types (Message, ToolSchema, etc.) and Google's SDK types (Content,
FunctionDeclaration, etc.).
"""

import base64
import json
import uuid
from typing import Any, Dict, List, Optional

from google.genai import types

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


# ==================== Role Conversion ====================

def role_to_sdk(role: Role) -> str:
    """Convert internal Role to SDK role string."""
    mapping = {
        Role.USER: "user",
        Role.MODEL: "model",
        Role.TOOL: "user",  # Tool responses are sent as user in Gemini
    }
    return mapping.get(role, "user")


def role_from_sdk(role: str) -> Role:
    """Convert SDK role string to internal Role."""
    mapping = {
        "user": Role.USER,
        "model": Role.MODEL,
    }
    return mapping.get(role, Role.USER)


# ==================== ToolSchema Conversion ====================

def tool_schema_to_sdk(schema: ToolSchema) -> types.FunctionDeclaration:
    """Convert ToolSchema to SDK FunctionDeclaration."""
    return types.FunctionDeclaration(
        name=schema.name,
        description=schema.description,
        parameters_json_schema=schema.parameters
    )


def tool_schema_from_sdk(decl: types.FunctionDeclaration) -> ToolSchema:
    """Convert SDK FunctionDeclaration to ToolSchema."""
    # Handle both dict and object forms of parameters
    params = {}
    if hasattr(decl, 'parameters_json_schema') and decl.parameters_json_schema:
        params = decl.parameters_json_schema
    elif hasattr(decl, 'parameters') and decl.parameters:
        # Convert Schema object to dict if needed
        if hasattr(decl.parameters, 'to_dict'):
            params = decl.parameters.to_dict()
        elif isinstance(decl.parameters, dict):
            params = decl.parameters

    return ToolSchema(
        name=decl.name,
        description=decl.description or "",
        parameters=params
    )


def tool_schemas_to_sdk_tool(schemas: List[ToolSchema]) -> Optional[types.Tool]:
    """Convert list of ToolSchemas to SDK Tool object.

    Deduplicates by tool name to avoid 'Duplicate function declaration' errors.
    """
    if not schemas:
        return None
    # Deduplicate by name (keep first occurrence)
    seen = set()
    unique_schemas = []
    for s in schemas:
        if s.name not in seen:
            seen.add(s.name)
            unique_schemas.append(s)
    declarations = [tool_schema_to_sdk(s) for s in unique_schemas]
    return types.Tool(function_declarations=declarations)


# ==================== Part Conversion ====================

def part_to_sdk(part: Part) -> types.Part:
    """Convert internal Part to SDK Part."""
    if part.text is not None:
        return types.Part.from_text(text=part.text)

    if part.function_call is not None:
        fc = part.function_call
        return types.Part(
            function_call=types.FunctionCall(
                name=fc.name,
                args=fc.args
            )
        )

    if part.function_response is not None:
        fr = part.function_response
        response = fr.result if isinstance(fr.result, dict) else {"result": fr.result}
        return types.Part.from_function_response(
            name=fr.name,
            response=response
        )

    if part.inline_data is not None:
        return types.Part(
            inline_data=types.Blob(
                mime_type=part.inline_data.get("mime_type", "application/octet-stream"),
                data=part.inline_data.get("data")
            )
        )

    # Fallback to empty text
    return types.Part.from_text(text="")


def part_from_sdk(part: types.Part) -> Part:
    """Convert SDK Part to internal Part."""
    # Text part
    if hasattr(part, 'text') and part.text is not None:
        return Part(text=part.text)

    # Function call part
    if hasattr(part, 'function_call') and part.function_call is not None:
        fc = part.function_call
        # Generate a unique ID for the function call
        call_id = str(uuid.uuid4())[:8]
        return Part(function_call=FunctionCall(
            id=call_id,
            name=fc.name,
            args=dict(fc.args) if fc.args else {}
        ))

    # Function response part
    if hasattr(part, 'function_response') and part.function_response is not None:
        fr = part.function_response
        response = fr.response
        if hasattr(response, 'items'):
            response = dict(response)
        return Part(function_response=ToolResult(
            call_id="",  # SDK doesn't track call IDs
            name=fr.name,
            result=response
        ))

    # Inline data
    if hasattr(part, 'inline_data') and part.inline_data is not None:
        inline = part.inline_data
        return Part(inline_data={
            "mime_type": inline.mime_type,
            "data": inline.data
        })

    # Unknown - return empty text
    return Part(text="")


# ==================== Message/Content Conversion ====================

def message_to_sdk(message: Message) -> types.Content:
    """Convert internal Message to SDK Content."""
    sdk_parts = [part_to_sdk(p) for p in (message.parts or [])]
    return types.Content(
        role=role_to_sdk(message.role),
        parts=sdk_parts
    )


def message_from_sdk(content: types.Content) -> Message:
    """Convert SDK Content to internal Message."""
    parts = [part_from_sdk(p) for p in (content.parts or [])]
    return Message(
        role=role_from_sdk(content.role),
        parts=parts
    )


def history_to_sdk(history: List[Message]) -> List[types.Content]:
    """Convert internal history to SDK history."""
    return [message_to_sdk(m) for m in (history or [])]


def history_from_sdk(history: List[types.Content]) -> List[Message]:
    """Convert SDK history to internal history."""
    return [message_from_sdk(c) for c in (history or [])]


# ==================== ToolResult Conversion ====================

def tool_result_to_sdk_part(result: ToolResult) -> types.Part:
    """Convert ToolResult to SDK function response Part.

    Handles both simple results and multimodal results with attachments.
    When attachments are present, builds a multimodal function response
    using FunctionResponsePart/FunctionResponseBlob structure.
    """
    response = result.result if isinstance(result.result, dict) else {"result": result.result}
    if result.is_error:
        response = {"error": str(result.result)}

    # Handle multimodal attachments
    if result.attachments:
        return _build_multimodal_function_response(result.name, response, result.attachments)

    return types.Part.from_function_response(
        name=result.name,
        response=response
    )


def _build_multimodal_function_response(
    name: str,
    response: Dict[str, Any],
    attachments: List[Attachment]
) -> types.Part:
    """Build a multimodal function response with attachments.

    Creates a function response that includes inline binary data using
    the FunctionResponsePart/FunctionResponseBlob structure. The displayName
    field links the $ref in the response to the actual data.

    Args:
        name: The function name.
        response: The response dict (may contain $ref placeholders).
        attachments: List of Attachment objects with binary data.

    Returns:
        A types.Part with nested multimodal data.
    """
    # Build FunctionResponsePart list from attachments
    parts = []
    for attachment in attachments:
        display_name = attachment.display_name or f"attachment_{len(parts)}"

        # Add $ref to response if not already present
        if display_name not in str(response):
            response[display_name] = {"$ref": display_name}

        parts.append(
            types.FunctionResponsePart(
                inlineData=types.FunctionResponseBlob(
                    mimeType=attachment.mime_type,
                    data=attachment.data,
                    displayName=display_name
                )
            )
        )

    try:
        return types.Part.from_function_response(
            name=name,
            response=response,
            parts=parts
        )
    except Exception:
        # Fallback to simple response if multimodal fails
        return types.Part.from_function_response(
            name=name,
            response={**response, "error": "Failed to attach multimodal data"}
        )


def tool_results_to_sdk_parts(results: List[ToolResult]) -> List[types.Part]:
    """Convert list of ToolResults to SDK Parts."""
    return [tool_result_to_sdk_part(r) for r in (results or [])]


# ==================== Response Conversion ====================

def extract_text_from_response(response) -> Optional[str]:
    """Extract text from SDK response, handling function call parts safely."""
    if not response or not hasattr(response, 'candidates') or not response.candidates:
        return None

    texts = []
    for candidate in response.candidates:
        if hasattr(candidate, 'content') and candidate.content:
            for part in (candidate.content.parts or []):
                if hasattr(part, 'text') and part.text:
                    texts.append(part.text)

    return ''.join(texts) if texts else None


def extract_function_calls_from_response(response) -> List[FunctionCall]:
    """Extract function calls from SDK response."""
    calls = []

    if not response:
        return calls

    # Use SDK's function_calls property if available
    if hasattr(response, 'function_calls') and response.function_calls:
        for fc in response.function_calls:
            call_id = str(uuid.uuid4())[:8]
            calls.append(FunctionCall(
                id=call_id,
                name=fc.name,
                args=dict(fc.args) if fc.args else {}
            ))

    return calls


def extract_finish_reason_from_response(response) -> FinishReason:
    """Extract finish reason from SDK response."""
    if not response or not hasattr(response, 'candidates') or not response.candidates:
        return FinishReason.UNKNOWN

    for candidate in response.candidates:
        if hasattr(candidate, 'finish_reason'):
            reason = str(candidate.finish_reason).upper()
            if 'STOP' in reason:
                return FinishReason.STOP
            elif 'MAX' in reason or 'LENGTH' in reason:
                return FinishReason.MAX_TOKENS
            elif 'SAFETY' in reason:
                return FinishReason.SAFETY
            elif 'TOOL' in reason or 'FUNCTION' in reason:
                return FinishReason.TOOL_USE

    return FinishReason.UNKNOWN


def extract_usage_from_response(response) -> TokenUsage:
    """Extract token usage from SDK response."""
    usage = TokenUsage()

    if not response:
        return usage

    metadata = getattr(response, 'usage_metadata', None)
    if metadata:
        usage.prompt_tokens = getattr(metadata, 'prompt_token_count', 0) or 0
        usage.output_tokens = getattr(metadata, 'candidates_token_count', 0) or 0
        usage.total_tokens = getattr(metadata, 'total_token_count', 0) or 0

    return usage


def response_from_sdk(response) -> ProviderResponse:
    """Convert SDK response to internal ProviderResponse."""
    return ProviderResponse(
        text=extract_text_from_response(response),
        function_calls=extract_function_calls_from_response(response),
        usage=extract_usage_from_response(response),
        finish_reason=extract_finish_reason_from_response(response),
        raw=response
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
                'args': fc.args
            })
        elif part.function_response is not None:
            fr = part.function_response
            parts.append({
                'type': 'function_response',
                'call_id': fr.call_id,
                'name': fr.name,
                'result': fr.result,
                'is_error': fr.is_error
            })
        elif part.inline_data is not None:
            parts.append({
                'type': 'inline_data',
                'mime_type': part.inline_data.get('mime_type'),
                'data': base64.b64encode(part.inline_data.get('data', b'')).decode('utf-8')
                        if part.inline_data.get('data') else None
            })

    return {
        'role': message.role.value,
        'parts': parts
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
                args=p.get('args', {})
            )))
        elif ptype == 'function_response':
            parts.append(Part(function_response=ToolResult(
                call_id=p.get('call_id', ''),
                name=p['name'],
                result=p.get('result'),
                is_error=p.get('is_error', False)
            )))
        elif ptype == 'inline_data':
            raw_data = None
            if p.get('data'):
                raw_data = base64.b64decode(p['data'])
            parts.append(Part(inline_data={
                'mime_type': p.get('mime_type'),
                'data': raw_data
            }))

    return Message(
        role=Role(data['role']),
        parts=parts
    )


def serialize_history(history: List[Message]) -> str:
    """Serialize history to JSON string."""
    return json.dumps([serialize_message(m) for m in history])


def deserialize_history(data: str) -> List[Message]:
    """Deserialize JSON string to history."""
    items = json.loads(data)
    return [deserialize_message(m) for m in items]
