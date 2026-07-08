"""Prose-emulated tool calling — shared machinery.

Some backends expose models that cannot emit native tool calls: the
Chrome built-in AI Prompt API has no tool parameter on stable Chrome, and
cheap models behind OpenAI-compatible gateways (OpenRouter et al.) ignore
the ``tools`` array and answer in prose.  Historically jaato simply
marked those models as unable to drive the harness; this module is the
opt-in alternative — a text-encoded tool-call protocol:

- **Request side**: :func:`tool_schemas_to_prompt` renders the tool
  schemas into a terse system-prompt section, and
  :func:`messages_to_prose_chat` converts history so past tool traffic is
  replayed as text (assistant ``tool_call`` fences, tool results as user
  messages) instead of native ``tool_calls`` / ``role:"tool"`` structures
  a non-tool model would choke on.
- **Response side**: :func:`parse_tool_calls` extracts fenced
  ``tool_call`` JSON blocks from the model's text, and
  :func:`rewrite_prose_tool_calls` turns a text-only
  ``ProviderResponse`` into one carrying real ``FunctionCall`` parts.

Consumers:

- The ``chrome_ai`` provider (always — its backend has no native path).
- ``OpenAICompatProvider`` and the standalone ``openrouter`` provider,
  behind the opt-in ``prose_tool_calls`` model quirk
  (``profile.quirks.prose_tool_calls: true``) — see
  :func:`read_prose_tool_calls_quirk`.

Framework contracts honored here so every consumer inherits them:

- **No human tool name on the wire** (``shared/tool_id_map``): the prompt
  section, history replay, and result labels all use hashed ``t_<hash>``
  ids; the wire-leak conformance guard scans
  :func:`tool_schemas_to_prompt`.  Callers map parsed ids back via
  ``id_to_name`` (identity for unknown/hallucinated ids, which then
  surface as structured unknown-tool executor errors the model can
  recover from — deliberately not discarded).
- **Tool results** go through ``render_result_for_model`` (structured
  result stays on ``ToolResult.result`` for ledger/GC; untrusted content
  stays boundary-wrapped).
- **Parallel tool results**: one TOOL message with N
  ``function_response`` parts emits N wire messages.

Expectation-setting: prose emulation is a reliability tier BELOW native
tool calling (community bridges driving Gemini Nano this way report
hallucinated ids and occasional malformed JSON — the parser degrades
gracefully, leaving broken blocks visible in text).  It exists so models
that would otherwise get NO tools can still drive the harness.

This module is loaded STANDALONE (by file path) by the wire-leak
conformance guard, so it uses only absolute ``jaato_sdk``/``shared``
imports — no relative imports, no vendor SDKs.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ToolResult,
    ToolSchema,
    render_result_for_model,
)
from shared.tool_id_map import id_to_name, name_to_id

logger = logging.getLogger(__name__)

#: Name of the opt-in model quirk that switches an OpenAI-compat /
#: openrouter provider onto this module's text-encoded protocol.
PROSE_TOOL_CALLS_QUIRK = "prose_tool_calls"

#: Info string of the fenced block used for text-encoded tool calls.
TOOL_CALL_FENCE = "tool_call"

#: Placeholder inserted where a message carried content the prose path
#: cannot marshal (images, audio).  Keeps the model aware that something
#: was elided instead of silently rewriting history.
NON_TEXT_PLACEHOLDER = "[non-text content omitted: not supported by this provider]"

#: Matches one ``tool_call`` fenced block; group 1 is the JSON body.
_FENCE_RE = re.compile(
    r"```" + TOOL_CALL_FENCE + r"[ \t]*\n(.*?)\n?```",
    re.DOTALL,
)


# ==================== Request side ====================

def serialize_tool_call(wire_id: str, args: Dict[str, Any]) -> str:
    """Render one tool call as the ``tool_call`` fenced block.

    This is the canonical wire syntax in BOTH directions: the system
    prompt (:func:`tool_schemas_to_prompt`) instructs the model to emit
    it, and history replay re-serializes past model calls through here so
    the model sees its own prior turns in the exact syntax it produced
    them.

    ``wire_id`` is the HASHED tool id (``name_to_id``), never the human
    name — callers hash before calling.
    """
    body = json.dumps({"name": wire_id, "arguments": args},
                      ensure_ascii=False)
    return f"```{TOOL_CALL_FENCE}\n{body}\n```"


def tool_schemas_to_prompt(tools: List[ToolSchema]) -> str:
    """Render tool schemas as the model-facing system-prompt section.

    This is the prose path's "tools array".  Tool names are hashed to
    opaque wire ids (``name_to_id``) so the model relies on the
    DESCRIPTION, per the framework-wide no-human-name-on-the-wire
    contract — the wire-leak conformance guard
    (``test_tool_id_wire_conformance``) scans this function's output.

    The rendering is deliberately terse (compact JSON schema, short
    preamble): the primary consumers are small models, some with tiny
    context windows (Gemini Nano: ~6-9k tokens total).

    Returns the full section, or ``""`` when ``tools`` is empty.
    """
    if not tools:
        return ""
    lines = [
        "# Tools",
        "",
        "You can call the tools listed under 'Available tools' below. "
        "Each tool appears as a '## <id>' heading followed by its "
        "description. Pick the tool whose description fits the task and "
        "use its id copied EXACTLY from the heading. To call one, output "
        "ONLY a fenced block:",
        "",
        f"```{TOOL_CALL_FENCE}",
        '{"name": "<tool id>", "arguments": {<arguments as JSON>}}',
        "```",
        "",
        "Rules:",
        "- Use only an id that appears as a '##' heading below, copied "
        "exactly. Never invent an id or use one that is not listed.",
        "- Emit one block per call; you may emit several blocks for"
        " independent calls.",
        "- After emitting tool_call blocks, stop. The results arrive in"
        " the next user message as 'Tool result ...'.",
        "- If no tool is needed, answer in plain text without any"
        f" {TOOL_CALL_FENCE} block.",
        "",
        "Available tools:",
        "",
    ]
    for tool in tools:
        lines.append(f"## {name_to_id(tool.name)}")
        if tool.description:
            lines.append(tool.description.strip())
        if tool.parameters:
            schema = json.dumps(tool.parameters, separators=(",", ":"),
                                ensure_ascii=False)
            lines.append(f"Arguments JSON schema: {schema}")
        lines.append("")
    return "\n".join(lines).rstrip()


def augment_system_with_tools(system_instruction: Optional[str],
                              tools: Optional[List[ToolSchema]]) -> Optional[str]:
    """Combine the session's system instruction with the tool section.

    Returns the combined system text, or ``None`` when there is neither
    an instruction nor any tools (caller then sends no system message).
    """
    parts = []
    if system_instruction:
        parts.append(system_instruction)
    if tools:
        section = tool_schemas_to_prompt(tools)
        if section:
            parts.append(section)
    return "\n\n".join(parts) if parts else None


def _render_tool_response(fr: ToolResult) -> str:
    """Render one executed tool's result as model-facing text.

    Uses :func:`render_result_for_model` (never ``str(dict)``) so the
    structured result stays on ``ToolResult.result`` for the ledger/GC
    while the model receives clean JSON plus any trusted steering suffix,
    with untrusted content boundary-wrapped.  The tool is referenced by
    its hashed wire id, matching the id the model called it by.
    """
    rendered = render_result_for_model(
        fr.result,
        getattr(fr, "model_suffix", None),
        untrusted=getattr(fr, "untrusted", False),
        untrusted_source=getattr(fr, "untrusted_source", None),
    )
    label = name_to_id(fr.name) if fr.name else "tool"
    return f"Tool result for {label} (call {fr.call_id}):\n{rendered}"


def message_to_prose_chat(message: Message) -> List[Dict[str, str]]:
    """Convert one jaato ``Message`` to plain ``{"role", "content"}`` dicts.

    The output shape is deliberately the least common denominator —
    string content, ``user``/``assistant`` roles only — valid both as
    OpenAI Chat Completions messages and as Prompt API
    ``initialPrompts``.  Tool traffic is text-encoded:

    - MODEL ``function_call`` parts re-serialize as ``tool_call`` fences
      (hashed wire ids — what the model originally emitted).
    - TOOL ``function_response`` parts become user messages; N parallel
      results emit N messages (dropping all but the first is the classic
      silent-parallel-results bug this mirrors
      ``_openai_compat/converters.py`` in avoiding).
    - Thought parts are never replayed; unmarshalable content becomes
      :data:`NON_TEXT_PLACEHOLDER`.

    Returns a LIST because one internal message can expand to several
    wire messages, or to zero (a message with no renderable content).
    """
    if message.role == Role.TOOL:
        out: List[Dict[str, str]] = []
        for part in message.parts:
            if part.function_response is not None:
                out.append({
                    "role": "user",
                    "content": _render_tool_response(part.function_response),
                })
        return out

    role = "assistant" if message.role == Role.MODEL else "user"
    chunks: List[str] = []
    for part in message.parts:
        if part.text is not None:
            chunks.append(part.text)
        elif part.function_call is not None:
            fc = part.function_call
            # History replay: the model emitted (and only ever knows) the
            # hashed wire id, so re-serialize with it, not the human name.
            chunks.append(serialize_tool_call(name_to_id(fc.name), fc.args))
        elif part.thought is not None:
            continue  # internal reasoning is never replayed to the model
        elif part.inline_data is not None:
            chunks.append(NON_TEXT_PLACEHOLDER)
    content = "\n".join(c for c in chunks if c)
    if not content:
        return []
    return [{"role": role, "content": content}]


def messages_to_prose_chat(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert a jaato history slice via :func:`message_to_prose_chat`."""
    out: List[Dict[str, str]] = []
    for message in messages:
        out.extend(message_to_prose_chat(message))
    return out


# ==================== Response side ====================

def parse_tool_calls(text: str) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
    """Extract ``tool_call`` fenced blocks from model output.

    Returns ``(clean_text, calls)`` where ``clean_text`` is the output
    with successfully-parsed blocks removed, and ``calls`` is a list of
    ``(name, arguments)`` tuples in emission order.  ``name`` is whatever
    the model emitted — normally a hashed wire id; callers map it back to
    the human name via ``tool_id_map.id_to_name``.

    A block whose body is a JSON array yields one call per element.
    Malformed blocks (invalid JSON, or no ``name`` key) are left in the
    text verbatim — the model's words reach the user unaltered and the
    failure is visible instead of vanishing — and logged at WARNING.
    """
    calls: List[Tuple[str, Dict[str, Any]]] = []

    def _consume(match: "re.Match[str]") -> str:
        body = match.group(1).strip()
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            logger.warning("prose_tools: unparseable tool_call block left "
                           "in text: %r", body[:200])
            return match.group(0)
        entries = parsed if isinstance(parsed, list) else [parsed]
        parsed_calls: List[Tuple[str, Dict[str, Any]]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("name"):
                logger.warning("prose_tools: tool_call entry missing 'name', "
                               "left in text: %r", str(entry)[:200])
                return match.group(0)
            args = entry.get("arguments", entry.get("args", {}))
            if not isinstance(args, dict):
                args = {"value": args}
            parsed_calls.append((str(entry["name"]), args))
        calls.extend(parsed_calls)
        return ""

    clean = _FENCE_RE.sub(_consume, text)
    # Collapse the whitespace holes left by removed blocks.
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, calls


def rewrite_prose_tool_calls(response: ProviderResponse, *,
                             call_id_prefix: str) -> ProviderResponse:
    """Turn fenced tool calls in a text response into ``FunctionCall`` parts.

    The prose path's counterpart of the native tool-call flush: parses
    the response text, maps wire ids back to human names (unknown ids
    pass through and surface as executor errors), and rebuilds the
    ``parts`` list as clean text + function-call parts with fresh
    correlation ids.  ``finish_reason`` flips to ``TOOL_USE`` when calls
    were found — except on a CANCELLED response, whose partial text is
    returned untouched (executing tools parsed from a truncated stream
    would act on output the user aborted).

    Returns the response unchanged (same object) when there is nothing to
    rewrite; otherwise a NEW ``ProviderResponse`` preserving usage /
    thinking / raw / structured_output.
    """
    if response.finish_reason == FinishReason.CANCELLED:
        return response
    text = response.get_text()
    if not text:
        return response
    clean_text, raw_calls = parse_tool_calls(text)
    if not raw_calls:
        return response

    parts: List[Part] = []
    if clean_text:
        parts.append(Part.from_text(clean_text))
    for index, (wire_id, args) in enumerate(raw_calls):
        parts.append(Part.from_function_call(FunctionCall(
            id=f"{call_id_prefix}-prose-{uuid.uuid4().hex[:8]}-{index}",
            name=id_to_name(wire_id),
            args=args,
        )))
    return ProviderResponse(
        parts=parts,
        usage=response.usage,
        finish_reason=FinishReason.TOOL_USE,
        raw=response.raw,
        structured_output=response.structured_output,
        thinking=response.thinking,
    )


# ==================== Quirk plumbing ====================

def read_prose_tool_calls_quirk(extra: Optional[Dict[str, Any]],
                                provider: str = "") -> bool:
    """Read ``profile.quirks.prose_tool_calls`` from a config ``extra`` dict.

    The runtime injects the profile's ``quirks`` mapping into
    ``config.extra["quirks"]`` (same path the vLLM quirks read).  A
    non-dict ``quirks`` value is ignored with a warning, mirroring
    ``VLLMProvider._parse_quirks``.
    """
    quirks = (extra or {}).get("quirks") or {}
    if not isinstance(quirks, dict):
        logger.warning("%s: ignoring non-dict quirks (got %s)",
                       provider or "prose_tools", type(quirks).__name__)
        return False
    return bool(quirks.get(PROSE_TOOL_CALLS_QUIRK, False))
