"""Detection helper for tool-call truncation via max_tokens.

When an LLM is serializing a tool call and hits its ``max_tokens`` cap
mid-arguments, the tool name survives but the ``arguments`` object ends
up truncated — typically as ``{}``.  The executor then rejects with a
"required field missing" error, and the model's coherent pre-tool
narration is lost to history as context the model can't act on.

This module detects that pathology by looking for two signals in the
same :class:`ProviderResponse`:

1. ``finish_reason == FinishReason.MAX_TOKENS`` — the cap was hit.
2. At least one :class:`FunctionCall` with ``args`` missing keys that
   the tool's :class:`ToolSchema` declares as ``required``.  The empty-
   args case (``{}``) is subsumed by this — an empty dict is missing
   every required key.

Tools with no required parameters (e.g. ``shell_list``) legitimately
take ``{}`` and are never flagged.  Tools whose schema has no
parameters object or no ``required`` list also pass through — we err
conservative.

Consumers: :mod:`shared.jaato_session` calls
:func:`detect_truncated_tool_call` between
``_add_model_response_to_history`` and ``_classify_finish_reason``,
so the detection fires *before* the classifier converts MAX_TOKENS
into an early-return abnormal termination.  On a hit, the session
rewrites the last assistant message to preserve narration text,
appends a synthetic user-role hint, and re-enters the chat loop
(bounded by a per-session counter).

See ``docs/design/rewind-with-hint.md`` for the full design.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

from jaato_sdk.plugins.model_provider.types import FinishReason

if TYPE_CHECKING:
    from jaato_sdk.plugins.model_provider.types import (
        FunctionCall,
        ProviderResponse,
        ToolSchema,
    )


# Reason strings surfaced on telemetry (``jaato.rewind.reason`` attribute
# on the turn span).  Keep short and stable — dashboards key off them.
REASON_MAX_TOKENS_EMPTY_ARGS = "max_tokens_empty_args"
REASON_MAX_TOKENS_MISSING_REQUIRED = "max_tokens_missing_required"


def detect_truncated_tool_call(
    response: 'ProviderResponse',
    tool_schemas: List['ToolSchema'],
) -> Optional[str]:
    """Return a reason string when the response looks like a truncated tool call.

    Args:
        response: The :class:`ProviderResponse` that just arrived from
            the provider.  Its ``finish_reason`` and ``parts`` are the
            detection signals.
        tool_schemas: All tool schemas currently exposed in the session
            (``registry.get_exposed_tool_schemas()``).  Used to resolve
            each call's ``required`` parameter list.  When a schema is
            not found for a called tool, that call is conservatively
            skipped — we never want to false-positive.

    Returns:
        - :data:`REASON_MAX_TOKENS_EMPTY_ARGS` when at least one call
          has ``args == {}`` on a tool that declares required params.
        - :data:`REASON_MAX_TOKENS_MISSING_REQUIRED` when at least one
          call has non-empty ``args`` but is missing one or more keys
          from the tool's ``required`` list.
        - ``None`` otherwise (includes all cases where the response
          does NOT look like a truncation pathology: non-MAX_TOKENS
          finish reasons, no function calls, legitimate empty-args
          tools, and fully-formed args).

    Detection rules (strict v1 — deliberately conservative):
        - ``response.finish_reason`` MUST be ``MAX_TOKENS``.
        - At least one :class:`FunctionCall` must exist in ``response.parts``.
        - For each call, look up its schema.  If the schema has
          ``required`` params and the call's ``args`` is missing any,
          the detector fires.
        - Empty-args vs missing-required are reported separately so
          telemetry can distinguish the two shapes.
    """
    if response.finish_reason != FinishReason.MAX_TOKENS:
        return None

    calls = [p.function_call for p in response.parts if p.function_call]
    if not calls:
        return None

    by_name = {schema.name: schema for schema in tool_schemas}

    found_empty = False
    found_missing = False

    for call in calls:
        schema = by_name.get(call.name)
        if schema is None:
            # Unknown tool — conservative: don't flag.  The executor
            # will emit its own "no executor registered" error.
            continue

        required = _required_keys(schema)
        if not required:
            # Tool legitimately takes no required args (e.g. shell_list).
            # Empty args are valid here, never a truncation signal.
            continue

        args = call.args or {}
        missing = [k for k in required if k not in args]
        if not missing:
            continue

        if not args:
            found_empty = True
        else:
            found_missing = True

    if found_empty:
        return REASON_MAX_TOKENS_EMPTY_ARGS
    if found_missing:
        return REASON_MAX_TOKENS_MISSING_REQUIRED
    return None


def find_truncated_call_name(
    response: 'ProviderResponse',
    tool_schemas: List['ToolSchema'],
) -> Optional[str]:
    """Return the name of the first call in ``response`` flagged as truncated.

    Used by the session to tag the telemetry span
    (``jaato.rewind.tool`` attribute) and to name the tool in the
    injected hint message.  Returns ``None`` when no call is flagged
    or when ``detect_truncated_tool_call`` would also return ``None``.
    """
    if response.finish_reason != FinishReason.MAX_TOKENS:
        return None

    by_name = {schema.name: schema for schema in tool_schemas}

    for part in response.parts:
        call = part.function_call
        if call is None:
            continue
        schema = by_name.get(call.name)
        if schema is None:
            continue
        required = _required_keys(schema)
        if not required:
            continue
        args = call.args or {}
        if any(k not in args for k in required):
            return call.name
    return None


def _required_keys(schema: 'ToolSchema') -> List[str]:
    """Extract the ``required`` list from a tool's JSON Schema parameters.

    Tolerates missing ``parameters`` dict and missing ``required`` key
    (both resolve to an empty list).  Non-list values are normalized
    to ``[]`` — we never want to crash on a malformed schema.
    """
    params = schema.parameters or {}
    required = params.get("required", [])
    if not isinstance(required, list):
        return []
    return [str(k) for k in required]
