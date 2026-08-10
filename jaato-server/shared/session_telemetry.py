"""Pure telemetry/observability helpers extracted from ``JaatoSession``.

These functions translate jaato's provider/conversation types into the
shapes OpenTelemetry/OpenInference spans expect, and classify cache
outcomes from usage data. They are deliberately **pure** — no session
state, no I/O, no span mutation — so they can be unit-tested in isolation
and reused across sessions/runners.

``JaatoSession`` (in :mod:`shared.jaato_session`) is the sole caller
today; the recorder methods there read/accumulate per-turn span state and
delegate the data-mapping to these functions. Keeping the mapping logic
here shrinks the session God object and makes the conversions
independently verifiable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from jaato_sdk.plugins.model_provider.types import (
    Message,
    ProviderResponse,
    Role,
)


def response_to_openinference(response: ProviderResponse) -> List[Dict[str, Any]]:
    """Convert a ``ProviderResponse`` to OpenInference output message dicts.

    Returns a list with a single assistant message containing text content
    and any tool calls from the response parts. An empty response (no text
    and no function calls) maps to an empty list, so callers can skip
    ``span.set_output_messages`` entirely.

    Args:
        response: The provider response to convert.

    Returns:
        List of message dicts with ``role``, ``content``, and optional
        ``tool_calls`` suitable for ``span.set_output_messages()``.
    """
    text = response.get_text()
    function_calls = response.get_function_calls()

    if not text and not function_calls:
        return []

    msg: Dict[str, Any] = {"role": "assistant", "content": text or ""}
    if function_calls:
        msg["tool_calls"] = [
            {
                "name": fc.name,
                "arguments": json.dumps(fc.args) if fc.args else "{}",
            }
            for fc in function_calls
        ]
    return [msg]


def history_to_openinference(messages: Iterable[Message]) -> List[Dict[str, Any]]:
    """Convert session history messages to OpenInference input message dicts.

    Maps jaato ``Message`` objects to the dict format expected by
    ``span.set_input_messages()``: each dict has ``role`` and ``content``,
    plus ``tool_calls`` for model messages that carry function calls. The
    jaato ``model`` role is normalized to OpenInference's ``assistant``.

    Args:
        messages: The conversation messages being sent to the provider
            (typically ``session._history.messages``).

    Returns:
        List of message dicts suitable for ``span.set_input_messages()``.
    """
    result: List[Dict[str, Any]] = []
    for msg in messages:
        # Map jaato roles to OpenInference roles
        role = msg.role.value  # "user", "model", "tool"
        if role == "model":
            role = "assistant"

        # Extract text content from parts
        texts = [p.text for p in msg.parts if p.text]
        content = "".join(texts) if texts else ""

        entry: Dict[str, Any] = {"role": role, "content": content}

        # Include tool calls from model messages
        if msg.role == Role.MODEL:
            tool_calls = [
                {
                    "name": p.function_call.name,
                    "arguments": json.dumps(p.function_call.args)
                    if p.function_call.args else "{}",
                }
                for p in msg.parts
                if p.function_call
            ]
            if tool_calls:
                entry["tool_calls"] = tool_calls

        result.append(entry)
    return result


def build_input_messages(
    system_instruction: Optional[str],
    messages: Iterable[Message],
) -> List[Dict[str, Any]]:
    """OpenInference input messages for a turn: the conversation history with
    the system instruction prepended as a ``system``-role message.

    The system prompt is **not** part of session history — it reaches the
    provider as the API's separate top-level ``system`` parameter — so it must
    be added here or the span shows only the conversation, dropping the largest
    and most important input. Emitting it as ``input_messages[0]`` is the
    standard OpenInference representation. Omitted when falsy (no system prompt
    configured, or the whole prompt was suppressed for a small-context model).
    """
    result = history_to_openinference(messages)
    if system_instruction:
        result.insert(0, {"role": "system", "content": system_instruction})
    return result


def classify_cache_outcome(
    prompt_tokens: int,
    cache_read_tokens: Optional[int],
    cache_creation_tokens: Optional[int],
) -> str:
    """Classify a request's cache hit/miss outcome from usage counts.

    ``prompt_tokens`` is the *new* (uncached) input only — matches
    Anthropic's ``input_tokens`` semantics, which jaato normalizes other
    providers to (see ``model_provider/anthropic/converters.py``). Total
    input therefore = ``cache_read_tokens + prompt_tokens``, and the hit
    ratio is ``cache_read_tokens / total_input`` — which naturally caps at
    1.0. Dividing by ``prompt_tokens`` alone produces ratios above 1.0 on
    cache-warm turns (e.g. 36.97 from 26580 / 719) and misclassifies them
    as anomalies.

    Returns:
        ``"hit"``     — most input tokens served from cache (>= 80%)
        ``"partial"`` — some input tokens served from cache (10-80%)
        ``"warm"``    — cache was being written but not read (creation only)
        ``"miss"``    — no cache reads, no creation
        ``"unknown"`` — usage data missing
    """
    read = cache_read_tokens or 0
    creation = cache_creation_tokens or 0
    new_input = prompt_tokens or 0
    total_input = read + new_input
    if total_input <= 0:
        return "unknown"
    ratio = read / total_input
    if ratio >= 0.8:
        return "hit"
    if ratio >= 0.1:
        return "partial"
    if creation > 0:
        return "warm"
    return "miss"
