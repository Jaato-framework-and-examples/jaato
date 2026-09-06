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
    TERMINAL_FINISH_REASONS,
    TokenUsage,
    ToolResult,
    normalize_inclusive_usage,
    parse_tool_call_arguments,
    render_result_for_model,
    ToolSchema,
)

from shared.tool_id_map import id_to_name, name_to_id

from shared.plugins.model_provider._attachments import (
    tool_result_followup_message,
    user_message_with_attachments,
)


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

def _sanitize_parameters_for_strict_upstreams(
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Strip JSON-Schema constructs that strict OpenAI-compat upstreams reject.

    OpenRouter routes to many upstreams, several of which (AtlasCloud's
    Qwen serving observed 2026-05-02; others suspected) run our tool
    definitions through a strict JSON-Schema validator before invoking
    the model and reject borderline-valid forms with HTTP 400 ``bad
    request``.  The most common culprit is ``required: []`` on object
    schemas — an empty array is technically valid JSON Schema but
    several validators trip on it.  Other minor cleanups land here too.

    Mutations:
      - Drop ``required`` keys whose value is an empty list (recursively).
      - Rewrite ``const: X`` to ``enum: [X]`` (semantically equivalent
        but more widely supported — JSON Schema's ``const`` keyword is
        2019-09+ and several strict validators don't recognize it).

    The function returns a deep-cloned, sanitized copy; the input
    dict is left untouched.  When the input is ``None`` (tool with no
    parameters), an empty ``{"type": "object", "properties": {}}`` is
    returned so the OpenAI tool-definition shape stays valid.
    """
    if parameters is None:
        return {"type": "object", "properties": {}}

    def _clean(node: Any) -> Any:
        if isinstance(node, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in node.items():
                # Strip ``required: []`` — strict upstreams reject empty arrays.
                if k == "required" and isinstance(v, list) and len(v) == 0:
                    continue
                # Rewrite ``const: X`` → ``enum: [X]`` for older validators.
                # Skip the rewrite when ``enum`` is also present at the same
                # level (caller intentionally combined them — leave alone).
                if k == "const" and "enum" not in node:
                    cleaned["enum"] = [v]
                    continue
                cleaned[k] = _clean(v)
            return cleaned
        if isinstance(node, list):
            return [_clean(item) for item in node]
        return node

    return _clean(parameters)


def tool_schema_to_openai(
    schema: ToolSchema,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Convert a ``ToolSchema`` to the OpenAI tool definition dict.

    Args:
        schema: The internal ``ToolSchema`` to convert.
        strict: When ``True``, emits ``"strict": True`` as a sibling of
            ``parameters`` inside the ``function`` dict.  OpenRouter
            forwards this to supported upstreams (Anthropic Sonnet 4.5 /
            Opus 4.1+, OpenAI GPT-4o+, Gemini, OSS, Fireworks per
            https://openrouter.ai/docs/guides/features/structured-outputs),
            which grammar-constrain tool-argument sampling to the
            schema.  Set via ``plugin_configs.openrouter.api_params.strict_tools``
            on the profile.  The framework intentionally does NOT auto-
            rewrite ``parameters`` to satisfy OpenAI's strict-mode
            schema requirements (``additionalProperties: false`` on
            every object, exhaustive ``required`` arrays, no
            ``oneOf``/``anyOf``); kb authors own schema shape and
            wire-side errors surface mismatches.  Default ``False``
            preserves the legacy advisory-mode wire shape.
    """
    function: Dict[str, Any] = {
        "name": name_to_id(schema.name),
        "description": schema.description,
        "parameters": _sanitize_parameters_for_strict_upstreams(schema.parameters),
    }
    if strict:
        function["strict"] = True
    return {
        "type": "function",
        "function": function,
    }


def tool_schemas_to_openai(
    schemas: Optional[List[ToolSchema]],
    *,
    cache_control: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Convert a list of ``ToolSchema`` objects to OpenAI tool definitions.

    When ``cache_control`` is provided, the tools are sorted by name (so
    the cache prefix is stable across sessions) and the dict is stamped
    onto the **last** tool object as a sibling of ``type`` / ``function``
    — that's the wire shape OpenRouter forwards to Anthropic/Gemini for
    tool-catalog caching.  Sorting matters: the cache prefix invalidates
    if tool registration order shifts between turns, so we always
    canonicalise to alphabetical when caching.

    When ``strict`` is ``True``, every function definition in the
    output carries ``"strict": True``.  See ``tool_schema_to_openai``
    for the full contract.
    """
    if not schemas:
        return None
    converted = [tool_schema_to_openai(s, strict=strict) for s in schemas]
    if cache_control:
        converted.sort(key=lambda t: t["function"]["name"])
        converted[-1] = {**converted[-1], "cache_control": dict(cache_control)}
    return converted


def system_message_with_cache(
    text: str,
    cache_control: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a system message, optionally annotated for prompt caching.

    Without ``cache_control`` this returns the standard flat OpenAI
    shape (``{"role": "system", "content": "<text>"}``) so the wire
    stays minimal.  With ``cache_control``, the content is promoted to
    a content-part list with the breakpoint on the last part — that's
    the form OpenRouter requires for explicit caching on Anthropic and
    Gemini upstreams.
    """
    if not cache_control:
        return {"role": "system", "content": text}
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": dict(cache_control),
            }
        ],
    }


# ==================== Message Conversion ====================

def message_to_openai(message: Message) -> List[Dict[str, Any]]:
    """Convert an internal ``Message`` to OpenAI chat-message dict(s).

    Returns a LIST: one internal ``TOOL`` message can carry N parallel
    ``function_response`` parts (a parallel tool-call batch is appended as a
    single ``Message(role=TOOL, parts=[...N...])`` — see
    ``jaato_session._do_send_tool_results``), and the OpenAI chat format
    requires ONE ``role:"tool"`` message per ``tool_call_id``.  Emitting only
    ``function_responses[0]`` silently dropped results #2..N off the wire —
    the model saw only the first parallel result.  Non-tool messages map to a
    single-element list.
    """
    role = message.role

    text_parts = [p.text for p in message.parts if p.text]
    content = "".join(text_parts) if text_parts else ""

    function_calls = [p.function_call for p in message.parts if p.function_call]
    function_responses = [
        p.function_response for p in message.parts if p.function_response
    ]

    if function_responses:
        # One wire ``role:"tool"`` message PER function_response so all N
        # parallel results reach the model (each keyed by its own call_id).
        tool_msgs: List[Dict[str, Any]] = []
        image_followups: List[Dict[str, Any]] = []
        for fr in function_responses:
            result_str = render_result_for_model(fr.result, fr.model_suffix, untrusted=fr.untrusted, untrusted_source=fr.untrusted_source)
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": fr.call_id,
                "content": result_str,
            })
            # tool messages can't carry image/file content — surface such
            # attachments as a follow-up user message so the model SEES them.
            # A mime this wire doesn't carry (audio, video, no declared mime)
            # is withheld and SAID so, rather than dropped into silence (#829).
            followup = tool_result_followup_message(
                getattr(fr, "attachments", None), pdf_as_file=True
            )
            if followup is not None:
                image_followups.append(followup)
        # All tool messages first (each keyed to its tool_call_id), then any
        # attachment follow-ups, then the model generates its next turn.
        return tool_msgs + image_followups

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
        return [msg]

    # User message.  Marshal any inline_data (image / PDF) parts into OpenAI
    # multimodal content blocks so a vision/file-declared model actually
    # RECEIVES them.  OpenRouter declares these via the catalog
    # (resolve_modalities catalog-detect), but this wire converter only emitted
    # text — the binary part was silently dropped and the model confabulated.
    # Text-only turns keep a plain-string ``content`` (unchanged wire shape).
    #
    # A part whose mime this wire doesn't carry — audio, video, or one with no
    # declared mime — is withheld rather than asserted to be a PNG (#829); the
    # note states it so the model doesn't confabulate over the gap.
    return user_message_with_attachments(content, message.parts, pdf_as_file=True)


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
            # A ``{"raw": ...}`` stand-in was still a fabricated argument
            # mapping; the raw text rides on its own field (#750).
            args, unreadable_args = parse_tool_call_arguments(
                func.get("arguments")
            )
            parts.append(Part(function_call=FunctionCall(
                id=tc.get("id", ""),
                name=get_original_tool_name(func.get("name", "")),
                args=args,
                unreadable_args=unreadable_args,
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


def _repair_history_shape_for_strict_upstreams(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Repair an OpenAI message list for upstreams that strictly enforce
    the OpenAI Chat Completions spec on the ``messages`` field.

    The framework's session-level ``ensure_tool_call_integrity`` enforces
    Anthropic-format validity (no orphan ``tool_use`` / ``tool_result``
    pairs).  OpenAI's spec adds further constraints that some upstream
    providers (AtlasCloud's Qwen serving observed 2026-05-02; others
    suspected) enforce strictly and reject with HTTP 400 when violated:

      - The first message in ``messages[]`` MUST NOT be role=``tool``.
        GC pruning that drops the leading system/user prefix can leave
        a ``tool`` message exposed at index 0.
      - Each role=``tool`` message's ``tool_call_id`` MUST match an
        ``id`` in a preceding role=``assistant`` message's
        ``tool_calls[]`` array.  GC pruning that drops the assistant
        frame while keeping the tool result produces an "orphan"
        tool message that strict validators reject.

    The repair is conservative — it only **drops** invalid frames;
    it never invents content.  Drops are silent (no error raised) so
    the agent can recover gracefully with whatever history remains.

    Returns a new list; the input is not mutated.

    Failure modes the repair catches:
      - ``[user, tool, ...]``                       → drop tool head
      - ``[user, tool, tool, ...]``                  → drop tool head(s)
      - ``[..., tool(id=X), ...]`` no preceding asst → drop the orphan
      - GC removed an asst frame's tool_calls       → tool responses become orphans → drop

    Failure modes the repair does NOT catch (deferred to future work):
      - assistant with ``tool_calls`` followed by NO tool responses
        (orphan assistant) — generally OpenAI accepts this for an
        in-progress turn; only some upstreams reject.
      - tool ordering: spec wants tool responses in the same order
        as assistant's tool_calls; harmless if shuffled in practice.
    """
    if not messages:
        return messages

    out: List[Dict[str, Any]] = []
    seen_assistant_tool_ids: set = set()

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in seen_assistant_tool_ids:
                out.append(msg)
            # else: drop (orphan tool response — its assistant frame is gone)
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    seen_assistant_tool_ids.add(tc_id)

        out.append(msg)

    # Drop any leading tool messages — they made it through the per-message
    # check (e.g. their assistant precedes the GC truncation point) but a
    # tool message at position 0 still violates the OpenAI spec.
    while out and out[0].get("role") == "tool":
        out.pop(0)

    return out


def history_to_openai(history: List[Message]) -> List[Dict[str, Any]]:
    """Convert internal history to an OpenAI message list.

    Runs a final shape-repair pass so strict OpenAI-compat upstreams
    (e.g. AtlasCloud / Qwen) don't reject post-GC history with HTTP 400
    for orphan tool responses or leading tool messages.  See
    :func:`_repair_history_shape_for_strict_upstreams` for the
    constraints enforced.
    """
    # Flatten: message_to_openai returns a LIST (a TOOL message with N
    # parallel function_responses → N wire tool messages).
    converted = [
        wire for m in (history or []) for wire in message_to_openai(m)
    ]
    return _repair_history_shape_for_strict_upstreams(converted)


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
                args, unreadable_args = parse_tool_call_arguments(
                    tc.function.arguments
                )
                fc = FunctionCall(
                    id=tc.id,
                    name=get_original_tool_name(tc.function.name),
                    args=args,
                    unreadable_args=unreadable_args,
                )
                parts.append(Part.from_function_call(fc))

    return parts


def extract_finish_reason(response: "ChatCompletion") -> FinishReason:
    """Extract the ``FinishReason`` from an OpenAI response.

    Shares :func:`resolve_choice_finish_reason` with the streaming
    path, so the batch path sees a truncation that OpenRouter
    normalised away for exactly the same reasons — a non-streamed turn
    can hit the output cap mid-tool-call just as easily.  Choices whose
    reason resolves to ``UNKNOWN`` (typically: not reported yet) are
    skipped, preserving the original scan-for-a-reported-reason
    behaviour.
    """
    if not response or not response.choices:
        return FinishReason.UNKNOWN

    for choice in response.choices:
        resolved = resolve_choice_finish_reason(choice)
        if resolved is not FinishReason.UNKNOWN:
            return resolved

    return FinishReason.UNKNOWN


def extract_usage(response: "ChatCompletion") -> TokenUsage:
    """Extract token usage from an OpenAI response.

    In addition to the standard prompt / completion / total counts, this
    pulls the OpenRouter prompt-caching telemetry when present:

    - ``prompt_tokens_details.cached_tokens`` — tokens served from cache
      (90% Anthropic discount, varying for other upstreams).  Surfaced
      via :attr:`TokenUsage.cache_read_tokens`, and SUBTRACTED from
      ``prompt_tokens``, which on this wire includes it.
    - ``prompt_tokens_details.cache_write_tokens`` — tokens written to
      cache on this turn (1.25x premium for 5-minute TTL, 2x for
      1-hour), with the Anthropic-native top-level
      ``cache_creation_input_tokens`` accepted as a fallback.  Surfaced
      via :attr:`TokenUsage.cache_creation_tokens`.

    The OpenAI SDK doesn't have typed fields for OpenRouter's extras,
    so we use ``getattr`` with the dict-/Pydantic-aware
    :func:`_read_usage_extra` helper instead of touching ``model_extra``
    directly — the same accessor works on real ``CompletionUsage``
    objects and on the ``MagicMock`` doubles tests use.
    """
    usage = TokenUsage()

    if not response or not response.usage:
        return usage

    raw_usage = response.usage
    usage.prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
    usage.output_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
    usage.total_tokens = getattr(raw_usage, "total_tokens", 0) or 0
    apply_cache_usage(raw_usage, usage)

    return usage


def _read_usage_extra(raw_usage: Any, key: str) -> Optional[Any]:
    """Read an OpenRouter-specific usage field that the OpenAI SDK
    doesn't model with a typed attribute.

    Tries (in order): direct ``getattr`` (works for ``MagicMock`` /
    ``SimpleNamespace`` and any future SDK update that adds the field),
    then ``model_extra`` (Pydantic's bag for unknown fields on the
    real ``CompletionUsage``).  Returns ``None`` when neither carries
    the key — caller treats that as "field not reported".
    """
    if raw_usage is None:
        return None
    direct = getattr(raw_usage, key, None)
    if direct is not None:
        return direct
    extra = getattr(raw_usage, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(key)
    return None


def _read_details(details: Any, key: str) -> Optional[int]:
    """Read one integer key out of ``prompt_tokens_details``.

    That block arrives as a Pydantic model on the real SDK and as a plain
    dict on some paths (and as a ``SimpleNamespace`` in tests), so both
    accesses are tried.  Returns ``None`` for absent or non-integer
    values, which callers treat as "not reported" -- distinct from a
    reported zero.

    Exists because the read and write counts live side by side in this
    block and were being read two different ways; the write side looked
    only at the top level and therefore never saw its value.
    """
    if details is None:
        return None
    value = getattr(details, key, None)
    if value is None and isinstance(details, dict):
        value = details.get(key)
    return value if isinstance(value, int) else None


def apply_cache_usage(raw_usage: Any, usage: TokenUsage) -> None:
    """Populate the cache-related fields on ``usage`` from a raw usage object.

    Mutates ``usage`` in place so the same helper works for the batch
    path (:func:`extract_usage`) and both streaming usage updates in
    :meth:`OpenRouterProvider._stream_response`.  Silent on missing
    fields — older OpenRouter responses (and non-cache-capable
    upstreams) simply leave the optional ``cache_*`` attributes as
    ``None``.

    AND CONVERTS THE CONVENTION.  OpenRouter counts both cached
    quantities INSIDE ``prompt_tokens``; :class:`TokenUsage` counts them
    beside it.  So this is also the seam that subtracts them back out —
    it is the one place all three OpenRouter paths pass through, and the
    only place that has the raw counts and the destination object at the
    same time.  Callers must therefore pass a FRESHLY built ``usage``
    still carrying the wire's ``prompt_tokens``; the subtraction is
    arithmetic, not idempotent.

    Without it the same tokens land on both sides of every downstream
    sum: a live GLM-5.3 turn reported a 50% cache hit against a bill
    that says 99.3%, because ``cache_read + prompt`` double-counted the
    129,825 tokens that were served from cache (issue #758).
    """
    if raw_usage is None:
        return

    details = getattr(raw_usage, "prompt_tokens_details", None)

    cached_tokens = _read_details(details, "cached_tokens")
    if cached_tokens is not None and cached_tokens > 0:
        usage.cache_read_tokens = cached_tokens

    # Writes sit BESIDE the reads, in the same nested block, under
    # OpenRouter's own spelling.  Reading only the top-level Anthropic
    # name meant `cache_creation_tokens` was always None on OpenRouter
    # while the write was billed: a 27,438-token Sonnet turn cost
    # $0.10271, which is $3.74/Mtok -- the cache-WRITE rate ($3.75), not
    # the $3.00 input rate.  Verified against the wire (issue #699):
    #
    #   "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4403}
    #
    # The top-level name is kept as a fallback because the surrounding
    # parsing is deliberately shape-tolerant -- this one helper serves
    # several upstreams, and an upstream that does pass the Anthropic
    # field through should keep working.
    creation = _read_details(details, "cache_write_tokens")
    if creation is None:
        creation = _read_usage_extra(raw_usage, "cache_creation_input_tokens")
    if isinstance(creation, int) and creation > 0:
        usage.cache_creation_tokens = creation

    # OpenRouter exposes cost telemetry via ``cost`` (USD) and a derived
    # ``cache_discount`` (negative number = savings).  We forward ``cost``
    # into the framework's ``cost_usd`` field so per-turn accounting
    # reflects the actual gateway charge after cache savings.
    cost = _read_usage_extra(raw_usage, "cost")
    if isinstance(cost, (int, float)) and cost >= 0:
        usage.cost_usd = float(cost)

    # LAST, once both cached counts are known: take them out of
    # ``prompt_tokens``.  Order matters — reading the writes above is what
    # makes the cold-arrival turn (all input written, none read) normalize
    # correctly rather than reporting a whole cold prefix as new input.
    normalize_inclusive_usage(usage)


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

#: Finish-reason spellings that all mean "the output cap was hit".
#:
#: OpenRouter normalises ``finish_reason`` to the OpenAI vocabulary but
#: passes the upstream's own word through untouched in
#: ``native_finish_reason``, so the vocabulary this has to recognise is
#: the union over every provider it fronts, not the OpenAI four:
#: ``length`` (OpenAI chat completions, Together, Fireworks),
#: ``max_tokens`` (Anthropic, Bedrock), ``max_output_tokens`` (OpenAI's
#: Responses API, which is what gpt-5 family models are served over) and
#: ``model_length`` (Mistral).  Google's ``MAX_TOKENS`` folds in via the
#: caller's lowercasing.
#:
#: Anything not listed here falls through to ``UNKNOWN`` rather than
#: being guessed at — an unrecognised reason is a reason to look, not a
#: reason to assume truncation.
TRUNCATION_FINISH_REASONS = frozenset({
    "length",
    "max_tokens",
    "max_output_tokens",
    "model_length",
})


def map_finish_reason(reason: Optional[str]) -> FinishReason:
    """Map an OpenAI streaming finish reason to a ``FinishReason``.

    ``"error"`` is OpenRouter-specific: it accompanies the mid-stream
    error event documented at
    https://openrouter.ai/docs/api/reference/streaming
    ("Errors After Tokens Have Been Sent"), where the upstream
    disconnects partway through a response.  The framework's
    ``FinishReason.ERROR`` is its dedicated outcome.

    Truncation spellings beyond OpenAI's ``length`` are accepted
    because this same function is used to interpret OpenRouter's
    ``native_finish_reason`` — see :data:`TRUNCATION_FINISH_REASONS`.
    """
    if not reason:
        return FinishReason.UNKNOWN

    reason_lower = reason.lower()
    if reason_lower == "stop":
        return FinishReason.STOP
    elif reason_lower in TRUNCATION_FINISH_REASONS:
        return FinishReason.MAX_TOKENS
    elif reason_lower in ("tool_calls", "function_call"):
        return FinishReason.TOOL_USE
    elif reason_lower == "content_filter":
        return FinishReason.SAFETY
    elif reason_lower == "error":
        return FinishReason.ERROR

    return FinishReason.UNKNOWN


def read_native_finish_reason(choice: Any) -> Optional[str]:
    """Return OpenRouter's ``native_finish_reason`` for a choice, if any.

    OpenRouter documents this field as a sibling of ``finish_reason`` on
    every choice (streaming chunk and batch response alike): the raw
    word the *upstream* used, before OpenRouter mapped it into the
    OpenAI vocabulary.  The two disagree in practice — an OpenRouter
    activity export for issue #745 shows turns that ran to a 65,536
    token cap reported as ``native_finish_reason: "max_output_tokens"``
    and ``finish_reason: "tool_calls"``, i.e. the normalised field
    hides the truncation behind the fact that a call was in flight when
    the cap landed.

    The OpenAI SDK's ``Choice`` models don't declare the field, so on
    real responses it lands in Pydantic's ``model_extra``.  Both
    accesses are tried, mirroring :func:`_read_usage_extra`.

    Returns:
        The reason string when one is reported as a genuine ``str``,
        else ``None``.  The ``isinstance`` check is what keeps
        ``MagicMock``'s auto-vivified attributes from being mistaken
        for a reported value.
    """
    if choice is None:
        return None
    direct = getattr(choice, "native_finish_reason", None)
    if isinstance(direct, str):
        return direct
    extra = getattr(choice, "model_extra", None)
    if isinstance(extra, dict):
        candidate = extra.get("native_finish_reason")
        if isinstance(candidate, str):
            return candidate
    return None


def resolve_choice_finish_reason(choice: Any) -> FinishReason:
    """Map a choice's finish reason, consulting the native reason too.

    ``finish_reason`` is authoritative whenever it already reports a
    terminal outcome (truncation, safety, a mid-stream error): those
    are precise, and a native reason cannot improve on the *mapping*.
    It is *not* authoritative when it reports ``stop`` / ``tool_calls``
    / nothing at all, because OpenRouter's normalisation can flatten an
    upstream truncation into one of those — which is the #745 bug.  In
    that case a truncating ``native_finish_reason`` wins.

    ``error`` is the one terminal outcome that describes nothing, and
    the native reason is the only thing carrying information about it
    (#766).  That diagnosis cannot ride *this* return value — a
    :class:`FinishReason` has no room for ``MALFORMED_FUNCTION_CALL``
    — so callers read :func:`read_native_finish_reason` alongside this
    and raise :class:`~.errors.UpstreamFinishError` with it.  The
    mapping here is unchanged: ``error`` still means
    :attr:`FinishReason.ERROR`.

    Args:
        choice: A streaming ``Choice`` delta or a batch ``Choice``.

    Returns:
        The resolved :class:`FinishReason`.
    """
    normalised = map_finish_reason(getattr(choice, "finish_reason", None))
    if normalised in TERMINAL_FINISH_REASONS:
        return normalised
    native = read_native_finish_reason(choice)
    if native and native.strip().lower() in TRUNCATION_FINISH_REASONS:
        return FinishReason.MAX_TOKENS
    return normalised


def read_response_native_finish_reason(response: Any) -> Optional[str]:
    """Return the first ``native_finish_reason`` a batch response reports.

    The streaming loop reads the native reason per choice as the chunks
    arrive; the non-streamed path has no loop to hang that on, so this
    scans the response's choices for the first one that reports a
    reason.  Mirrors :func:`extract_finish_reason`'s first-reported-wins
    scan so the two agree about *which* choice they are describing.

    Returns:
        The native reason string, or ``None`` when no choice reports one
        (which includes every non-OpenRouter-shaped double).
    """
    if response is None:
        return None
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    for choice in choices:
        native = read_native_finish_reason(choice)
        if native:
            return native
    return None


def read_chunk_error(chunk: Any) -> Optional[Dict[str, Any]]:
    """Return OpenRouter's mid-stream ``error`` payload, or ``None``.

    Per https://openrouter.ai/docs/api/reference/streaming, when an
    upstream disconnects after some tokens have been sent OpenRouter
    cannot change the HTTP status (already 200 OK), so it emits a final
    SSE chunk shaped like::

        {"id":"...", "object":"chat.completion.chunk", ...,
         "error":{"code":"server_error","message":"..."},
         "choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}

    The ``error`` field is top-level (sibling of ``choices``).  The
    OpenAI SDK's ``ChatCompletionChunk`` Pydantic model doesn't declare
    it, so on real responses it lands in ``model_extra``; tests can
    populate it as either a dict or a small namespace object.

    We accept an error only when it materialises as a real dict or
    as an object exposing a string ``message`` — that way the
    helper Just Works on test doubles without being fooled by
    ``MagicMock``'s auto-attributes (which are never dicts and never
    have a string ``message``).
    """
    if chunk is None:
        return None
    direct = getattr(chunk, "error", None)
    if isinstance(direct, dict):
        return direct
    if direct is not None:
        msg = getattr(direct, "message", None)
        code = getattr(direct, "code", None)
        if isinstance(msg, str):
            return {
                "code": code if isinstance(code, (str, int)) else None,
                "message": msg,
            }
    extra = getattr(chunk, "model_extra", None)
    if isinstance(extra, dict):
        candidate = extra.get("error")
        if isinstance(candidate, dict):
            return candidate
    return None


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
