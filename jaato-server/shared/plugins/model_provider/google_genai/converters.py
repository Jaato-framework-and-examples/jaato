"""Converters between internal types and Google GenAI SDK types.

This module handles bidirectional conversion between provider-agnostic
types (Message, ToolSchema, etc.) and Google's SDK types (Content,
FunctionDeclaration, etc.).
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Lazy imports - SDK is only loaded when actually used
from ._lazy import get_types

if TYPE_CHECKING:
    from google.genai import types

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
    normalize_inclusive_usage,
)

from shared.tool_id_map import id_to_name, name_to_id


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

def tool_schema_to_sdk(schema: ToolSchema) -> get_types().FunctionDeclaration:
    """Convert ToolSchema to SDK FunctionDeclaration."""
    return get_types().FunctionDeclaration(
        name=name_to_id(schema.name),
        description=schema.description,
        parameters_json_schema=schema.parameters
    )


def tool_schema_from_sdk(decl: get_types().FunctionDeclaration) -> ToolSchema:
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


def tool_schemas_to_sdk_tool(schemas: List[ToolSchema]) -> Optional[get_types().Tool]:
    """Convert list of ToolSchemas to SDK Tool object."""
    if not schemas:
        return None
    declarations = [tool_schema_to_sdk(s) for s in schemas]
    return get_types().Tool(function_declarations=declarations)


# ==================== Part Conversion ====================

def part_to_sdk(part: Part) -> get_types().Part:
    """Convert internal Part to SDK Part."""
    if part.text is not None:
        return get_types().Part.from_text(text=part.text)

    if part.function_call is not None:
        fc = part.function_call
        return get_types().Part(
            function_call=get_types().FunctionCall(
                name=name_to_id(fc.name),
                args=fc.args
            )
        )

    if part.function_response is not None:
        # Delegate to the multimodal-aware builder so a tool result's image
        # attachments actually reach the model.  The inline
        # ``from_function_response`` here dropped ``attachments`` silently — the
        # live history_to_sdk path never used tool_result_to_sdk_part, so
        # google_genai declared-but-didn't-deliver tool-result vision.
        return tool_result_to_sdk_part(part.function_response)

    if part.inline_data is not None:
        return get_types().Part(
            inline_data=get_types().Blob(
                mime_type=part.inline_data.get("mime_type", "application/octet-stream"),
                data=part.inline_data.get("data")
            )
        )

    # Fallback to empty text
    return get_types().Part.from_text(text="")


def part_from_sdk(part: get_types().Part) -> Part:
    """Convert SDK Part to internal Part."""
    # Text part
    if hasattr(part, 'text') and part.text is not None:
        return Part(text=part.text)

    # Function call part
    if hasattr(part, 'function_call') and part.function_call is not None:
        fc = part.function_call
        call_id = str(uuid.uuid4())[:8]
        return Part(function_call=FunctionCall(
            id=call_id,
            name=id_to_name(fc.name),
            args=dict(fc.args) if fc.args else {}
        ))

    # Function response part
    if hasattr(part, 'function_response') and part.function_response is not None:
        fr = part.function_response
        response = fr.response
        if hasattr(response, 'items'):
            response = dict(response)
        return Part(function_response=ToolResult(
            call_id="",
            name=id_to_name(fr.name),
            result=response
        ))

    # Inline data
    if hasattr(part, 'inline_data') and part.inline_data is not None:
        inline = part.inline_data
        return Part(inline_data={
            "mime_type": inline.mime_type,
            "data": inline.data
        })

    # Thought part (Gemini 2.0+ thinking mode)
    if hasattr(part, 'thought') and part.thought is not None:
        return Part(thought=part.thought)

    # Executable code part
    if hasattr(part, 'executable_code') and part.executable_code is not None:
        # Extract code string from SDK's ExecutableCode type
        code = part.executable_code
        code_str = getattr(code, 'code', str(code)) if code else ""
        return Part(executable_code=code_str)

    # Code execution result part
    if hasattr(part, 'code_execution_result') and part.code_execution_result is not None:
        result = part.code_execution_result
        # Extract output from SDK's CodeExecutionResult type
        output = getattr(result, 'output', str(result)) if result else ""
        return Part(code_execution_result=output)

    # Unknown part type - log a warning and return empty text
    # This helps diagnose when the SDK returns new/unexpected part types
    import sys
    part_attrs = []
    for attr in ['text', 'function_call', 'function_response', 'inline_data',
                 'executable_code', 'code_execution_result', 'thought']:
        if hasattr(part, attr):
            val = getattr(part, attr)
            if val is not None:
                part_attrs.append(f"{attr}={type(val).__name__}")
    if part_attrs:
        print(f"[google_genai/converters] Warning: Unknown SDK part type with attrs: {part_attrs}",
              file=sys.stderr)

    return Part(text="")


# ==================== Message/Content Conversion ====================

def message_to_sdk(message: Message) -> get_types().Content:
    """Convert internal Message to SDK Content."""
    sdk_parts = [part_to_sdk(p) for p in (message.parts or [])]
    return get_types().Content(
        role=role_to_sdk(message.role),
        parts=sdk_parts
    )


def message_from_sdk(content: get_types().Content) -> Message:
    """Convert SDK Content to internal Message."""
    parts = [part_from_sdk(p) for p in (content.parts or [])]
    return Message(
        role=role_from_sdk(content.role),
        parts=parts
    )


def history_to_sdk(history: List[Message]) -> List[get_types().Content]:
    """Convert internal history to SDK history."""
    return [message_to_sdk(m) for m in (history or [])]


def history_from_sdk(history: List[get_types().Content]) -> List[Message]:
    """Convert SDK history to internal history."""
    return [message_from_sdk(c) for c in (history or [])]


# ==================== ToolResult Conversion ====================

def tool_result_to_sdk_part(result: ToolResult) -> get_types().Part:
    """Convert ToolResult to SDK function response Part.

    Handles both simple results and multimodal results with attachments.
    When attachments are present, builds a multimodal function response
    using FunctionResponsePart/FunctionResponseBlob structure.
    """
    response = result.result if isinstance(result.result, dict) else {"result": result.result}
    if result.is_error:
        response = {"error": str(result.result)}
    elif result.untrusted:
        # Untrusted external content (web_fetch/web_search/MCP): deliver the
        # result as a single boundary-wrapped field so the model treats it as
        # data, not instructions (indirect-prompt-injection mitigation).
        from jaato_sdk.plugins.model_provider.types import wrap_untrusted_content
        _text = result.result if isinstance(result.result, str) else json.dumps(result.result)
        response = {"untrusted_external_content":
                    wrap_untrusted_content(_text, result.untrusted_source)}
    # Dict-response provider: deliver the model-facing steering suffix as a
    # reserved key (model still sees it; the structured result + ledger stay
    # clean — the ledger reads history, not this converter output).
    if result.model_suffix:
        response = {**response, "_agent_guidance": result.model_suffix}

    # Names must be id-mapped to match the function_call name emitted at
    # part_to_sdk (``name_to_id(fc.name)``) — otherwise google can't pair the
    # response to its call.
    fn_name = name_to_id(result.name)

    # Handle multimodal attachments
    if result.attachments:
        return _build_multimodal_function_response(fn_name, response, result.attachments)

    return get_types().Part.from_function_response(
        name=fn_name,
        response=response
    )


def _build_multimodal_function_response(
    name: str,
    response: Dict[str, Any],
    attachments: List[Attachment]
) -> get_types().Part:
    """Build a multimodal function response with attachments.

    Creates a function response that includes inline binary data using
    the FunctionResponsePart/FunctionResponseBlob structure. The displayName
    field links the $ref in the response to the actual data.

    Args:
        name: The function name.
        response: The response dict (may contain $ref placeholders).
        attachments: List of Attachment objects with binary data.

    Returns:
        A get_types().Part with nested multimodal data.
    """
    # Build FunctionResponsePart list from attachments
    parts = []
    for attachment in attachments:
        display_name = attachment.display_name or f"attachment_{len(parts)}"

        # Add $ref to response if not already present
        if display_name not in str(response):
            response[display_name] = {"$ref": display_name}

        parts.append(
            get_types().FunctionResponsePart(
                inlineData=get_types().FunctionResponseBlob(
                    mimeType=attachment.mime_type,
                    data=attachment.data,
                    displayName=display_name
                )
            )
        )

    try:
        return get_types().Part.from_function_response(
            name=name,
            response=response,
            parts=parts
        )
    except Exception:
        # Fallback to simple response if multimodal fails
        return get_types().Part.from_function_response(
            name=name,
            response={**response, "error": "Failed to attach multimodal data"}
        )


def tool_results_to_sdk_parts(results: List[ToolResult]) -> List[get_types().Part]:
    """Convert list of ToolResults to SDK Parts."""
    return [tool_result_to_sdk_part(r) for r in (results or [])]


# ==================== Streaming Helpers ====================

def extract_text_from_chunk(chunk) -> Optional[str]:
    """Extract text from a streaming chunk without triggering SDK warnings.

    The SDK's chunk.text accessor prints warnings when there are non-text
    parts (like function calls). This function safely extracts text by
    iterating through parts directly.

    Args:
        chunk: A streaming chunk from send_message_stream.

    Returns:
        Text content if present, None otherwise.
    """
    if not chunk or not hasattr(chunk, 'candidates') or not chunk.candidates:
        return None

    texts = []
    for candidate in chunk.candidates:
        if hasattr(candidate, 'content') and candidate.content:
            for part in (candidate.content.parts or []):
                # Check for text attribute directly, avoid using chunk.text
                if hasattr(part, 'text') and part.text:
                    texts.append(part.text)

    return ''.join(texts) if texts else None


def function_call_from_sdk(fc) -> Optional[FunctionCall]:
    """Convert SDK FunctionCall to internal FunctionCall.

    Used during streaming to convert function calls from chunks.

    Args:
        fc: SDK FunctionCall object.

    Returns:
        Internal FunctionCall or None if invalid.
    """
    if not fc or not hasattr(fc, 'name'):
        return None

    call_id = str(uuid.uuid4())[:8]
    return FunctionCall(
        id=call_id,
        name=fc.name,
        args=dict(fc.args) if hasattr(fc, 'args') and fc.args else {}
    )


#: Google's ``FinishReason`` members, mapped by NAME.
#:
#: The mapping used to be four substring tests, and two of them were
#: wrong in opposite directions (#687).
#:
#: ``'TOOL' in name or 'FUNCTION' in name -> TOOL_USE`` never matched a
#: tool-use turn, because Gemini reports ``STOP`` for a turn that emits
#: function calls -- it has no tool-use finish reason at all.  What it
#: DID match was every error whose name happens to mention tools:
#: ``MALFORMED_FUNCTION_CALL`` (the model emitted a call its own
#: serialiser rejected), ``UNEXPECTED_TOOL_CALL``,
#: ``TOO_MANY_TOOL_CALLS``.  Each of those became "the model wants a
#: tool run", so the session executed or nudged on a turn that had
#: failed.
#:
#: And everything unlisted fell through to ``UNKNOWN``, which is a
#: SUCCESS value downstream -- so ``RECITATION``, ``BLOCKLIST``,
#: ``PROHIBITED_CONTENT``, ``SPII`` and ``OTHER`` all read as clean
#: stops.
#:
#: Mapping by name rather than by substring is what makes both
#: unambiguous.  An unrecognised name still resolves to ``UNKNOWN``: a
#: reason we do not know is not a reason to guess, and Google adds
#: members (the image-generation set landed in 2025) faster than this
#: table can.
_GOOGLE_FINISH_REASONS = {
    "STOP": FinishReason.STOP,

    "MAX_TOKENS": FinishReason.MAX_TOKENS,

    # Content filters.  ``RECITATION`` (copyrighted material),
    # ``BLOCKLIST``, ``PROHIBITED_CONTENT``, ``SPII`` (personal data)
    # and the image-side equivalents are all "the filter stopped it".
    "SAFETY": FinishReason.SAFETY,
    "RECITATION": FinishReason.SAFETY,
    "BLOCKLIST": FinishReason.SAFETY,
    "PROHIBITED_CONTENT": FinishReason.SAFETY,
    "SPII": FinishReason.SAFETY,
    "IMAGE_SAFETY": FinishReason.SAFETY,
    "IMAGE_PROHIBITED_CONTENT": FinishReason.SAFETY,
    "IMAGE_RECITATION": FinishReason.SAFETY,

    # Generation failures.  Named errors, not stops -- and emphatically
    # not tool-use requests.
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
    "UNEXPECTED_TOOL_CALL": FinishReason.ERROR,
    "TOO_MANY_TOOL_CALLS": FinishReason.ERROR,
    "LANGUAGE": FinishReason.ERROR,
    "UNSUPPORTED_LANGUAGE": FinishReason.ERROR,
    "NO_IMAGE": FinishReason.ERROR,
    "IMAGE_OTHER": FinishReason.ERROR,
    "OTHER": FinishReason.ERROR,
}


def finish_reason_from_sdk(reason) -> FinishReason:
    """Convert an SDK finish reason to the internal :class:`FinishReason`.

    Used by both the streaming accumulator (per chunk) and the batch
    converter (per candidate), so the two cannot disagree about what a
    given Google reason means.

    Args:
        reason: SDK finish reason -- an enum member, its ``str()``
            (``"FinishReason.MALFORMED_FUNCTION_CALL"``), or a bare
            name.  All three reduce to the member name.

    Returns:
        The mapped reason, or ``FinishReason.UNKNOWN`` for a name this
        table does not carry.
    """
    if not reason:
        return FinishReason.UNKNOWN

    # ``str()`` of an SDK enum is ``"FinishReason.STOP"``; a plain
    # string is already the name.  Take the last dotted segment either
    # way, and prefer ``.name`` when the object offers one.
    name = getattr(reason, "name", None) or str(reason)
    name = name.rsplit(".", 1)[-1].strip().upper()

    return _GOOGLE_FINISH_REASONS.get(name, FinishReason.UNKNOWN)


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


def extract_parts_from_response(response) -> List[Part]:
    """Extract parts from SDK response, preserving order of text and function calls."""
    parts = []

    if not response or not hasattr(response, 'candidates') or not response.candidates:
        return parts

    for candidate in response.candidates:
        if not hasattr(candidate, 'content') or not candidate.content:
            continue

        for sdk_part in (candidate.content.parts or []):
            if hasattr(sdk_part, 'text') and sdk_part.text is not None:
                parts.append(Part.from_text(sdk_part.text))
            elif hasattr(sdk_part, 'function_call') and sdk_part.function_call:
                fc = function_call_from_sdk(sdk_part.function_call)
                if fc:
                    parts.append(Part.from_function_call(fc))

    return parts


def extract_finish_reason_from_response(response) -> FinishReason:
    """Extract the finish reason from a batch SDK response.

    Delegates to :func:`finish_reason_from_sdk` so the batch and
    streaming paths cannot drift apart about what a Google reason means
    -- they carried two separate copies of the same substring mapping,
    and therefore the same two defects, until #687.
    """
    if not response or not hasattr(response, 'candidates') or not response.candidates:
        return FinishReason.UNKNOWN

    for candidate in response.candidates:
        reason = getattr(candidate, 'finish_reason', None)
        if reason:
            return finish_reason_from_sdk(reason)

    return FinishReason.UNKNOWN


def extract_usage_from_response(response) -> TokenUsage:
    """Extract token usage from SDK response.

    Extracts standard token counts plus cached content token count
    when context caching is used.

    Gemini's ``prompt_token_count`` is the WHOLE prompt and
    ``cached_content_token_count`` is the cached PART of it — the
    OpenAI convention, not the Anthropic one :class:`TokenUsage`
    carries.  So the cached count is subtracted back out here, at the
    seam, and every consumer downstream reads one convention (issue
    #758).  Google reports no cache-creation count on a response
    (explicit ``CachedContent`` is billed at creation time, on its own
    call), so there is nothing else to remove.
    """
    usage = TokenUsage()

    if not response:
        return usage

    metadata = getattr(response, 'usage_metadata', None)
    if metadata:
        usage.prompt_tokens = getattr(metadata, 'prompt_token_count', 0) or 0
        usage.output_tokens = getattr(metadata, 'candidates_token_count', 0) or 0
        usage.total_tokens = getattr(metadata, 'total_token_count', 0) or 0

        # Extract cached content token count (context caching)
        cached_tokens = getattr(metadata, 'cached_content_token_count', None)
        if isinstance(cached_tokens, int) and cached_tokens > 0:
            usage.cache_read_tokens = cached_tokens
            normalize_inclusive_usage(usage)

    return usage


def response_from_sdk(response) -> ProviderResponse:
    """Convert SDK response to internal ProviderResponse."""
    return ProviderResponse(
        parts=extract_parts_from_response(response),
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
        elif part.thought is not None:
            parts.append({'type': 'thought', 'thought': part.thought})
        elif part.executable_code is not None:
            parts.append({'type': 'executable_code', 'code': part.executable_code})
        elif part.code_execution_result is not None:
            parts.append({'type': 'code_execution_result', 'output': part.code_execution_result})

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
        elif ptype == 'thought':
            parts.append(Part(thought=p.get('thought', '')))
        elif ptype == 'executable_code':
            parts.append(Part(executable_code=p.get('code', '')))
        elif ptype == 'code_execution_result':
            parts.append(Part(code_execution_result=p.get('output', '')))

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
