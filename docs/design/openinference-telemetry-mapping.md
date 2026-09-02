# OpenInference Telemetry Mapping — Detailed Design

## Status

**Draft** — March 2026

## Problem

jaato's OpenTelemetry integration emits valid OTLP traces, but uses custom
attribute names (`jaato.*`, `gen_ai.*`) that don't match the
[OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md)
expected by [Arize Phoenix](https://phoenix.arize.com/). As a result, Phoenix
dashboards show raw spans without LLM-specific UI features — no message
inspection, no token cost breakdown, no tool call visualization, no agent
graph rendering.

## Goal

Replace jaato's custom attribute schema with OpenInference semantic
conventions so that:

1. Phoenix dashboards render full AI-native UIs (messages, tokens, tool calls,
   agent graphs).
2. Traces are interoperable with any OpenInference-compatible backend.
3. The attribute vocabulary is an industry standard, not jaato-specific.

## Non-Goals

- Supporting the full breadth of OpenInference (embeddings, retrievers,
  rerankers, guardrails, evaluators, prompt templates). Only the span kinds
  jaato actually produces are mapped.
- Adding `openinference-semantic-conventions` as a runtime dependency (we use
  raw strings to stay zero-dep).

---

## 1. Span Kind Mapping

The single most critical attribute is `openinference.span.kind`. Without it,
Phoenix treats spans as generic OTel spans and skips all AI-specific rendering.

| jaato span name      | OTel span name (current → new) | OpenInference `span.kind` | Rationale |
|----------------------|-------------------------------|---------------------------|-----------|
| Turn root            | `jaato.turn` → `jaato.turn`   | `AGENT`                   | A turn is an agent reasoning loop (LLM calls + tool use). |
| LLM API call         | `gen_ai.chat` → `llm`         | `LLM`                     | Direct LLM invocation. Renamed to avoid confusion with GenAI semantic conventions. |
| Tool execution       | `jaato.tool` → `jaato.tool`   | `TOOL`                    | External tool invocation. |
| Retry attempt        | `jaato.retry` → `jaato.retry` | `CHAIN`                   | Internal orchestration step. |
| GC operation         | `jaato.gc` → `jaato.gc`       | `CHAIN`                   | Internal orchestration step. |
| Permission check     | `jaato.permission` → `jaato.permission` | `CHAIN`          | Internal orchestration step. |

Span names stay the same — Phoenix classifies by `openinference.span.kind`,
not by span name.

**Implementation:** Each `*_span()` method sets `openinference.span.kind` in
the initial attributes dict.

---

## 2. Attribute Mapping — Per Span Kind

### 2.1. Turn Span → AGENT

**Replaced attributes:**

| Old attribute               | New attribute (OpenInference) | Notes |
|-----------------------------|-------------------------------|-------|
| `jaato.session_id`          | `session.id`                  | |
| `jaato.agent_name`          | `agent.name`                  | Only set when non-None. |
| `jaato.agent_type`          | `metadata` (JSON string)      | Packed into metadata dict along with other jaato-specific fields. |
| `jaato.turn_index`          | `metadata` (JSON string)      | Same — folded into metadata. |
| `jaato.plan_id`             | `metadata` (JSON string)      | Same. |
| `jaato.step_id`             | `metadata` (JSON string)      | Same. |

**New attributes:**

| Attribute              | Value                                          |
|------------------------|-------------------------------------------------|
| `openinference.span.kind` | `"AGENT"` |
| `input.value`          | User prompt text (subject to redaction). |
| `input.mime_type`      | `"text/plain"` |
| `output.value`         | Agent response text (subject to redaction). |
| `output.mime_type`     | `"text/plain"` |
| `graph.node.id`        | `session_id` (unique per agent) |
| `graph.node.name`      | `agent_name` or `agent_type` (human-readable) |
| `graph.node.parent_id` | Empty string for main agent; parent's `session_id` for subagents |

**`metadata` consolidation:** OpenInference defines a `metadata` attribute
(JSON string) for arbitrary span-level metadata. All jaato-specific context
that has no OI equivalent goes here:

```json
{
  "agent_type": "main",
  "turn_index": 3,
  "plan_id": "plan-xyz",
  "step_id": "step-1",
  "streaming": true
}
```

**How `graph.node.parent_id` is populated:** Add a new optional parameter
`parent_session_id` to `turn_span()`. For subagents, jaato's session creation
flow passes the parent session ID. When absent or `agent_type == "main"`,
`graph.node.parent_id` is `""` (root).

### 2.2. LLM Span → LLM

This is the highest-value mapping — it unlocks Phoenix's message inspector,
token breakdown, and tool call visualization.

#### 2.2.1. Core Model Attributes

| Old attribute                    | New attribute (OpenInference) |
|----------------------------------|-------------------------------|
| `gen_ai.system`                  | `llm.system`                  |
| `gen_ai.request.model`           | `llm.model_name`              |
| `gen_ai.response.finish_reasons` | `metadata` (JSON string)      |
| `jaato.streaming`                | `metadata` (JSON string)      |

#### 2.2.2. Token Counts

| Old attribute                        | New attribute (OpenInference)                    |
|--------------------------------------|--------------------------------------------------|
| `gen_ai.usage.input_tokens`          | `llm.token_count.prompt`                         |
| `gen_ai.usage.output_tokens`         | `llm.token_count.completion`                     |
| *(computed)*                         | `llm.token_count.total`                          |
| `gen_ai.usage.cache_read_tokens`     | `llm.token_count.prompt_details.cache_read`      |
| `gen_ai.usage.cache_creation_tokens` | `llm.token_count.prompt_details.cache_write`     |
| `gen_ai.usage.reasoning_tokens`      | `llm.token_count.completion_details.reasoning`   |

**Implementation:** The session layer sets attributes using the new OI names
directly. `_SpanWrapper` tracks `prompt` and `completion` values to auto-
compute and set `llm.token_count.total` when both are known.

#### 2.2.3. Input/Output Messages

OpenInference uses **flattened indexed prefixes** for messages:

```
llm.input_messages.0.message.role = "system"
llm.input_messages.0.message.content = "You are a helpful assistant."
llm.input_messages.1.message.role = "user"
llm.input_messages.1.message.content = "Hello"
```

**Current state:** jaato's LLM span does NOT currently carry input/output
messages as span attributes. The `gen_ai.prompt` / `gen_ai.completion`
attributes in the redaction list are never actually set.

**Approach:** Add two methods to `SpanContext` / `_SpanWrapper`:

```python
def set_input_messages(self, messages: List[Dict[str, Any]]) -> None:
    """Set OpenInference-formatted input messages on the span.

    Each message dict has 'role' and 'content' keys.
    Flattened to:
      llm.input_messages.{i}.message.role
      llm.input_messages.{i}.message.content
    """

def set_output_messages(self, messages: List[Dict[str, Any]]) -> None:
    """Set OpenInference-formatted output messages on the span.

    Same format. Tool calls within messages flattened to:
      llm.output_messages.{i}.message.tool_calls.{j}.tool_call.function.name
      llm.output_messages.{i}.message.tool_calls.{j}.tool_call.function.arguments
    """
```

These methods handle:
- Content redaction (when `redact_content=True`, content is replaced with
  `[REDACTED: N chars]` but roles and tool call names are preserved).
- Flattening to indexed attribute format.
- Tool call extraction from output messages.

**Session-layer call site** (`jaato_session.py`): Inside the existing
`with self._telemetry.llm_span(...)` block, the session calls
`llm_span.set_input_messages(...)` with the messages sent to the provider and
`llm_span.set_output_messages(...)` with the response. Only messages for the
current LLM call are captured (not the full conversation history).

#### 2.2.4. Tool Calls in LLM Output

When the LLM returns tool calls, they appear in the output messages:

```
llm.output_messages.0.message.role = "assistant"
llm.output_messages.0.message.tool_calls.0.tool_call.function.name = "cli"
llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments = '{"command": "ls"}'
```

Handled by `set_output_messages()`. The session already has the parsed
`FunctionCall` objects; it just needs to pass them.

### 2.3. Tool Span → TOOL

| Old attribute              | New attribute (OpenInference) | Notes |
|----------------------------|-------------------------------|-------|
| `jaato.tool.name`          | `tool.name`                   | |
| `jaato.tool.call_id`       | `tool.id`                     | |
| `jaato.tool.plugin_type`   | `metadata` (JSON string)      | |
| `jaato.tool.success`       | *(use span status)*           | `set_status_ok()` / `set_status_error()` already called. |
| `jaato.tool.error`         | `exception.message`           | OI standard for errors. |
| `jaato.tool.duration_seconds` | `metadata` (JSON string)   | OTel span duration already captures this; keep in metadata for convenience. |
| `jaato.tool.parallel`      | `metadata` (JSON string)      | |
| `jaato.tool.mcp_server`    | `metadata` (JSON string)      | |
| `jaato.tool.streaming`     | `metadata` (JSON string)      | |

**New attributes:**

| Attribute              | Value |
|------------------------|-------|
| `openinference.span.kind` | `"TOOL"` |
| `input.value`          | Tool arguments as JSON string (subject to redaction). |
| `input.mime_type`      | `"application/json"` |
| `output.value`         | Tool result as JSON string (subject to redaction). |
| `output.mime_type`     | `"application/json"` |

### 2.4. Retry Span → CHAIN

| Old attribute              | New attribute (OpenInference) |
|----------------------------|-------------------------------|
| `jaato.retry.attempt`      | `metadata` (JSON string)      |
| `jaato.retry.max_attempts` | `metadata` (JSON string)      |
| `jaato.retry.context`      | `metadata` (JSON string)      |
| `jaato.retry.delay_seconds`| `metadata` (JSON string)      |
| `jaato.retry.error_type`   | `metadata` (JSON string)      |

New: `openinference.span.kind = "CHAIN"`.

### 2.5. GC Span → CHAIN

| Old attribute              | New attribute (OpenInference) |
|----------------------------|-------------------------------|
| `jaato.gc.trigger_reason`  | `metadata` (JSON string)      |
| `jaato.gc.strategy`        | `metadata` (JSON string)      |
| `jaato.gc.items_collected` | `metadata` (JSON string)      |
| `jaato.gc.tokens_freed`    | `metadata` (JSON string)      |
| `jaato.gc.context_before`  | `metadata` (JSON string)      |
| `jaato.gc.context_after`   | `metadata` (JSON string)      |

New: `openinference.span.kind = "CHAIN"`.

### 2.6. Permission Span → CHAIN

| Old attribute                 | New attribute (OpenInference) |
|-------------------------------|-------------------------------|
| `jaato.permission.tool_name`  | `metadata` (JSON string)      |

New: `openinference.span.kind = "CHAIN"`.

---

## 3. `_SpanWrapper` Redesign

### 3.1. Remove Legacy Sensitive Attributes

Replace `_SENSITIVE_ATTRS` with the new OI attribute names:

```python
_SENSITIVE_ATTRS = frozenset({
    "input.value",
    "output.value",
})
```

The old `gen_ai.prompt`, `gen_ai.completion`, `gen_ai.request.prompt`,
`gen_ai.response.completion`, `jaato.tool.args`, `jaato.tool.result` are
deleted — those attributes are no longer emitted.

### 3.2. Token Total Auto-Computation

Add `_prompt_tokens` and `_completion_tokens` tracking to `_SpanWrapper`:

```python
__slots__ = ("_span", "_redact", "_prompt_tokens", "_completion_tokens")

def set_attribute(self, key: str, value: Any) -> None:
    # ... redaction logic ...
    self._span.set_attribute(key, value)

    # Auto-compute total
    if key == "llm.token_count.prompt":
        self._prompt_tokens = value
        if self._completion_tokens is not None:
            self._span.set_attribute(
                "llm.token_count.total",
                self._prompt_tokens + self._completion_tokens,
            )
    elif key == "llm.token_count.completion":
        self._completion_tokens = value
        if self._prompt_tokens is not None:
            self._span.set_attribute(
                "llm.token_count.total",
                self._prompt_tokens + self._completion_tokens,
            )
```

### 3.3. Message Flattening Methods

```python
def set_input_messages(self, messages: List[Dict[str, Any]]) -> None:
    """Flatten input messages to OpenInference indexed attributes."""
    for i, msg in enumerate(messages):
        prefix = f"llm.input_messages.{i}.message"
        self._span.set_attribute(f"{prefix}.role", msg.get("role", ""))
        content = msg.get("content", "")
        if self._redact and content:
            content = f"[REDACTED: {len(content)} chars]"
        self._span.set_attribute(f"{prefix}.content", content)

def set_output_messages(self, messages: List[Dict[str, Any]]) -> None:
    """Flatten output messages to OpenInference indexed attributes."""
    for i, msg in enumerate(messages):
        prefix = f"llm.output_messages.{i}.message"
        self._span.set_attribute(f"{prefix}.role", msg.get("role", ""))
        content = msg.get("content", "")
        if self._redact and content:
            content = f"[REDACTED: {len(content)} chars]"
        self._span.set_attribute(f"{prefix}.content", content)
        for j, tc in enumerate(msg.get("tool_calls", [])):
            tc_prefix = f"{prefix}.tool_calls.{j}.tool_call"
            self._span.set_attribute(
                f"{tc_prefix}.function.name", tc.get("name", ""))
            args = tc.get("arguments", "")
            if self._redact and args:
                args = f"[REDACTED: {len(args)} chars]"
            self._span.set_attribute(f"{tc_prefix}.function.arguments", args)
```

### 3.4. Metadata Helper

A helper to pack jaato-specific attributes into the `metadata` JSON string:

```python
def set_metadata(self, data: Dict[str, Any]) -> None:
    """Set the OpenInference metadata attribute as a JSON string.

    Used for jaato-specific fields that have no OpenInference equivalent.
    """
    import json
    self._span.set_attribute("metadata", json.dumps(data))
```

---

## 4. Span Method Changes in `otel_plugin.py`

### 4.1. Constants

```python
# OpenInference span kind (required for Phoenix rendering)
_OI_SPAN_KIND = "openinference.span.kind"

# Span kind values
_OI_AGENT = "AGENT"
_OI_LLM = "LLM"
_OI_TOOL = "TOOL"
_OI_CHAIN = "CHAIN"
```

### 4.2. `turn_span()` — New Signature and Attributes

```python
@contextmanager
def turn_span(
    self,
    session_id: str,
    agent_type: str,
    agent_name: Optional[str] = None,
    turn_index: Optional[int] = None,
    parent_session_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[_SpanWrapper, None, None]:
    # ...
    attrs = {
        _OI_SPAN_KIND: _OI_AGENT,
        "session.id": session_id,
        "graph.node.id": session_id,
        "graph.node.name": agent_name or agent_type,
        "graph.node.parent_id": parent_session_id or "",
    }
    if agent_name:
        attrs["agent.name"] = agent_name

    # jaato-specific fields packed into metadata
    metadata = {"agent_type": agent_type}
    if turn_index is not None:
        metadata["turn_index"] = turn_index
    plan_id = getattr(ctx, "plan_id", None)
    if plan_id:
        metadata["plan_id"] = plan_id
    step_id = getattr(ctx, "step_id", None)
    if step_id:
        metadata["step_id"] = step_id
    attrs["metadata"] = json.dumps(metadata)

    if attributes:
        attrs.update(attributes)

    with self._tracer.start_as_current_span("jaato.turn", attributes=attrs) as span:
        yield _SpanWrapper(span, self._redact_content)
```

### 4.3. `llm_span()` — OpenInference Attributes

```python
@contextmanager
def llm_span(
    self,
    model: str,
    provider: str,
    streaming: bool = False,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[_SpanWrapper, None, None]:
    # ...
    attrs = {
        _OI_SPAN_KIND: _OI_LLM,
        "llm.system": provider,
        "llm.model_name": model,
    }
    metadata = {"streaming": streaming}
    attrs["metadata"] = json.dumps(metadata)

    if attributes:
        attrs.update(attributes)

    with self._tracer.start_as_current_span("llm", attributes=attrs) as span:
        yield _SpanWrapper(span, self._redact_content)
```

### 4.4. `tool_span()` — OpenInference Attributes

```python
@contextmanager
def tool_span(
    self,
    tool_name: str,
    call_id: str,
    plugin_type: str = "unknown",
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[_SpanWrapper, None, None]:
    # ...
    attrs = {
        _OI_SPAN_KIND: _OI_TOOL,
        "tool.name": tool_name,
        "tool.id": call_id,
    }
    metadata = {"plugin_type": plugin_type}
    attrs["metadata"] = json.dumps(metadata)

    if attributes:
        attrs.update(attributes)

    with self._tracer.start_as_current_span("jaato.tool", attributes=attrs) as span:
        yield _SpanWrapper(span, self._redact_content)
```

### 4.5. `retry_span()`, `gc_span()`, `permission_span()` — CHAIN Kind

All three follow the same pattern: set `openinference.span.kind = "CHAIN"`
and pack their jaato-specific attributes into `metadata`.

Example for `retry_span()`:

```python
attrs = {
    _OI_SPAN_KIND: _OI_CHAIN,
}
metadata = {
    "retry_attempt": attempt,
    "retry_max_attempts": max_attempts,
    "retry_context": context,
}
attrs["metadata"] = json.dumps(metadata)
```

### 4.6. `_get_agent_attrs()` — Removed

This method currently builds a dict of `jaato.*` attributes from thread-local
context. It is no longer needed as those attributes are either mapped to OI
equivalents (`session.id`, `agent.name`) or packed into `metadata`. The
thread-local context is still used, but the attribute construction moves
inline into each span method.

---

## 5. Session-Layer Changes (`jaato_session.py`)

### 5.1. Token Count Attributes

Replace all `gen_ai.usage.*` attribute calls with `llm.token_count.*`:

```python
# Before:
llm_span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
llm_span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
llm_span.set_attribute("gen_ai.usage.cache_read_tokens", usage.cache_read_tokens)
llm_span.set_attribute("gen_ai.usage.cache_creation_tokens", usage.cache_creation_tokens)
llm_span.set_attribute("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens)

# After:
llm_span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
llm_span.set_attribute("llm.token_count.completion", usage.output_tokens)
# llm.token_count.total is auto-computed by _SpanWrapper
if usage.cache_read_tokens:
    llm_span.set_attribute("llm.token_count.prompt_details.cache_read", usage.cache_read_tokens)
if usage.cache_creation_tokens:
    llm_span.set_attribute("llm.token_count.prompt_details.cache_write", usage.cache_creation_tokens)
if usage.reasoning_tokens:
    llm_span.set_attribute("llm.token_count.completion_details.reasoning", usage.reasoning_tokens)
```

### 5.2. LLM Message Capture

Inside the existing `with self._telemetry.llm_span(...)` block:

```python
# Before sending to provider — capture input messages
oi_messages = []
for msg in messages_for_provider:
    oi_messages.append({
        "role": msg.role,
        "content": msg.text_content or "",
    })
llm_telemetry.set_input_messages(oi_messages)

# After receiving response — capture output messages
oi_output = [{
    "role": "assistant",
    "content": response.text or "",
    "tool_calls": [
        {
            "name": fc.name,
            "arguments": json.dumps(fc.args) if fc.args else "{}",
        }
        for fc in response.function_calls
    ] if response.function_calls else [],
}]
llm_telemetry.set_output_messages(oi_output)
```

### 5.3. Tool Span Input/Output

Inside the existing `with self._telemetry.tool_span(...)` block:

```python
# Set tool input (arguments)
tool_span.set_attribute("input.value", json.dumps(fc.args) if fc.args else "{}")
tool_span.set_attribute("input.mime_type", "application/json")

# ... execute tool ...

# Set tool output (result)
tool_span.set_attribute("output.value", json.dumps(result_dict))
tool_span.set_attribute("output.mime_type", "application/json")
```

### 5.4. Turn Span — Pass `parent_session_id`

```python
with self._telemetry.turn_span(
    session_id=self._agent_id,
    agent_type=self._agent_type,
    agent_name=self._agent_name,
    turn_index=self._turn_index,
    parent_session_id=self._parent_session_id,
) as turn_span:
```

If `_parent_session_id` doesn't exist on `JaatoSession` yet, add it as an
optional parameter to `create_session()` in `jaato_runtime.py`.

### 5.5. Tool Error Attributes

Replace `jaato.tool.error` with the OI standard:

```python
# Before:
tool_span.set_attribute("jaato.tool.error", error_message)

# After:
tool_span.set_attribute("exception.message", error_message)
```

### 5.6. Other Session Attribute Replacements

| Old call | New call |
|----------|----------|
| `turn_span.set_attribute("jaato.cancelled", True)` | Pack into metadata or use span status |
| `turn_span.set_attribute("jaato.streaming", True)` | Already in metadata via `turn_span()` |
| `llm_span.set_attribute("gen_ai.response.finish_reasons", [...])` | Pack into metadata |

---

## 6. Protocol / Null Plugin Changes

### 6.1. `plugin.py` — SpanContext Protocol

Add methods:

```python
def set_input_messages(self, messages: List[Dict[str, Any]]) -> None:
    """Set OpenInference input messages (flattened indexed attributes)."""
    ...

def set_output_messages(self, messages: List[Dict[str, Any]]) -> None:
    """Set OpenInference output messages (flattened indexed attributes)."""
    ...

def set_metadata(self, data: Dict[str, Any]) -> None:
    """Set OpenInference metadata attribute (JSON string)."""
    ...
```

Add `parent_session_id` parameter to `turn_span()`.

### 6.2. `null_plugin.py` — _NoOpSpan

Add no-op implementations of `set_input_messages()`, `set_output_messages()`,
and `set_metadata()`.

---

## 7. Redaction

The redaction model simplifies. Content-bearing OI attributes:

| Attribute | Redacted? | Mechanism |
|-----------|-----------|-----------|
| `llm.input_messages.*.message.content` | Yes | `set_input_messages()` checks `_redact` |
| `llm.output_messages.*.message.content` | Yes | `set_output_messages()` checks `_redact` |
| `llm.output_messages.*.message.tool_calls.*.function.arguments` | Yes | `set_output_messages()` checks `_redact` |
| `input.value` | Yes | `_SENSITIVE_ATTRS` set on `_SpanWrapper` |
| `output.value` | Yes | `_SENSITIVE_ATTRS` set on `_SpanWrapper` |

Non-redactable (always emitted):
- `llm.model_name`, `llm.system`
- `agent.name`, `session.id`
- `tool.name`, `tool.id`
- `graph.node.*`
- `llm.token_count.*`
- `metadata` (contains only structural info, not content)
- `openinference.span.kind`
- Message roles and tool call function names

---

## 8. Example: Final Trace Output

```
Span: "jaato.turn"
  openinference.span.kind: "AGENT"
  session.id: "sess-abc"
  agent.name: "main"
  graph.node.id: "sess-abc"
  graph.node.name: "main"
  graph.node.parent_id: ""
  metadata: '{"agent_type":"main","turn_index":3}'
  input.value: "List files in /tmp"
  input.mime_type: "text/plain"
  output.value: "Here are the files in /tmp: ..."
  output.mime_type: "text/plain"

  Child: "llm"
    openinference.span.kind: "LLM"
    llm.system: "anthropic"
    llm.model_name: "claude-sonnet-4-20250514"
    llm.token_count.prompt: 1500
    llm.token_count.completion: 200
    llm.token_count.total: 1700
    llm.token_count.prompt_details.cache_read: 800
    llm.input_messages.0.message.role: "system"
    llm.input_messages.0.message.content: "You are a helpful assistant."
    llm.input_messages.1.message.role: "user"
    llm.input_messages.1.message.content: "List files in /tmp"
    llm.output_messages.0.message.role: "assistant"
    llm.output_messages.0.message.content: ""
    llm.output_messages.0.message.tool_calls.0.tool_call.function.name: "cli"
    llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments: '{"command":"ls /tmp"}'
    metadata: '{"streaming":true}'

    Child: "jaato.tool"
      openinference.span.kind: "TOOL"
      tool.name: "cli"
      tool.id: "call_123"
      input.value: '{"command":"ls /tmp"}'
      input.mime_type: "application/json"
      output.value: '{"status":"success","output":"file1.txt\nfile2.txt"}'
      output.mime_type: "application/json"
      metadata: '{"plugin_type":"cli","duration_seconds":0.5,"parallel":false}'

    Child: "jaato.retry"
      openinference.span.kind: "CHAIN"
      metadata: '{"retry_attempt":1,"retry_max_attempts":5,"retry_context":"api_call"}'
```

---

## 9. Migration Path

### Phase 1 — Core Replacement (this design)

- Replace all `jaato.*` / `gen_ai.*` attributes with OpenInference equivalents.
- Add `openinference.span.kind` to every span.
- Add message capture via `set_input_messages()` / `set_output_messages()`.
- Add graph attributes for DAG visualization.
- Pack jaato-specific context into `metadata` JSON.

**Estimated scope:** ~150 lines changed in `otel_plugin.py`, ~50 lines in
`jaato_session.py`, ~100 lines in tests.

### Phase 2 — Enrichments (future)

- `llm.invocation_parameters` (temperature, max_tokens, etc.).
- `llm.tools` (tool schemas sent to the model, flattened indexed).
- `llm.cost.*` (if token pricing data becomes available).

### Phase 3 — Full Observability (future)

- Embedding spans (EMBEDDING kind) for RAG plugin.
- Retriever spans (RETRIEVER kind) for document retrieval.
- Guardrail spans (GUARDRAIL kind) for permission/safety checks.

---

## 10. File Change Summary

| File | Change |
|------|--------|
| `shared/plugins/telemetry/otel_plugin.py` | Replace attribute schema with OI conventions, add message/metadata methods, remove `_get_agent_attrs()`, add OI constants |
| `shared/plugins/telemetry/plugin.py` | Add `set_input_messages()`, `set_output_messages()`, `set_metadata()` to `SpanContext` protocol; add `parent_session_id` to `turn_span()` |
| `shared/plugins/telemetry/null_plugin.py` | Add no-op `set_input_messages()`, `set_output_messages()`, `set_metadata()` |
| `shared/plugins/telemetry/tests/test_plugin.py` | Rewrite tests for OI attributes, message flattening, redaction, metadata packing |
| `shared/jaato_session.py` | Replace `gen_ai.usage.*` with `llm.token_count.*`, add message capture calls, pass `parent_session_id`, replace `jaato.tool.*` with OI equivalents |
| `docs/opentelemetry-design.md` | Update attribute reference section |
| `CLAUDE.md` | Update telemetry attribute list |

---

## 11. Decisions

1. **Span names stay the same.** Phoenix classifies by
   `openinference.span.kind`, not span name. No reason to change them.

2. **Messages scoped to current LLM call.** Only messages for the individual
   provider call are captured, not the full conversation history. This is what
   the span represents.

3. **No `openinference-semantic-conventions` dependency.** Raw strings are
   sufficient. The constants are simple and stable.

4. **`metadata` as catch-all.** OpenInference defines `metadata` (JSON string)
   for arbitrary data. All jaato-specific attributes that have no OI
   equivalent go here rather than being silently dropped.

5. **`_get_agent_attrs()` removed.** Thread-local context is still used, but
   attribute names are now OI-native. The helper method that built a `jaato.*`
   dict is replaced by inline construction in each span method.
