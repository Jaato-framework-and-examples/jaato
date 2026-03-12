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

Emit OpenInference-compliant attributes **alongside** the existing jaato
attributes so that:

1. Phoenix dashboards render full AI-native UIs (messages, tokens, tool calls,
   agent graphs).
2. Existing jaato-specific attributes (`jaato.tool.plugin_type`,
   `jaato.gc.*`, `jaato.retry.*`, etc.) remain unchanged for jaato's own
   dashboards and downstream consumers.
3. The changes are confined to `otel_plugin.py` — no session-layer or
   provider-layer changes required.

## Non-Goals

- Removing or renaming existing `jaato.*` attributes (backwards-compatible).
- Supporting the full breadth of OpenInference (embeddings, retrievers,
  rerankers, guardrails, evaluators, prompt templates). Only the span kinds
  jaato actually produces are mapped.
- Adding `openinference-semantic-conventions` as a runtime dependency (we use
  raw strings to stay zero-dep).

---

## 1. Span Kind Mapping

The single most critical attribute is `openinference.span.kind`. Without it,
Phoenix treats spans as generic OTel spans and skips all AI-specific rendering.

| jaato span name      | OTel span name (current) | OpenInference `span.kind` | Rationale |
|----------------------|--------------------------|---------------------------|-----------|
| Turn root            | `jaato.turn`             | `AGENT`                   | A turn is an agent reasoning loop (LLM calls + tool use). |
| LLM API call         | `gen_ai.chat`            | `LLM`                     | Direct LLM invocation. |
| Tool execution       | `jaato.tool`             | `TOOL`                    | External tool invocation. |
| Retry attempt        | `jaato.retry`            | `CHAIN`                   | Internal orchestration step. |
| GC operation         | `jaato.gc`               | `CHAIN`                   | Internal orchestration step. |
| Permission check     | `jaato.permission`       | `CHAIN`                   | Internal orchestration step. |

**Implementation:** Each `*_span()` method sets `openinference.span.kind` in
the initial attributes dict, alongside the existing attributes.

---

## 2. Attribute Mapping — Per Span Kind

### 2.1. Turn Span → AGENT

| Current jaato attribute     | OpenInference attribute     | Notes |
|-----------------------------|-----------------------------|-------|
| `jaato.session_id`          | `session.id`                | Direct rename for OI; keep original too. |
| `jaato.agent_name`          | `agent.name`                | Only set when agent_name is non-None. |
| `jaato.agent_type`          | *(keep as-is)*              | No OI equivalent; jaato-specific. |
| `jaato.turn_index`          | *(keep as-is)*              | No OI equivalent; jaato-specific. |
| *(new)*                     | `input.value`               | User prompt text (if not redacted). |
| *(new)*                     | `output.value`              | Agent response text (if not redacted). |
| *(new)*                     | `input.mime_type`           | `"text/plain"` |
| *(new)*                     | `output.mime_type`          | `"text/plain"` |

**Graph attributes** (new — enables Phoenix DAG visualization):

| Attribute              | Value                                          |
|------------------------|-------------------------------------------------|
| `graph.node.id`        | `session_id` (unique per agent)                 |
| `graph.node.name`      | `agent_name` or `agent_type` (human-readable)   |
| `graph.node.parent_id` | Empty string for main agent; parent's `session_id` for subagents |

**How `graph.node.parent_id` is populated:** The turn span already receives
`agent_type` (`"main"` vs `"subagent"`). For subagents, jaato's session
creation flow passes the parent session ID. We add a new optional parameter
`parent_session_id` to `turn_span()` that subagents provide. When absent or
`agent_type == "main"`, `graph.node.parent_id` is set to `""` (root).

### 2.2. LLM Span → LLM

This is the highest-value mapping — it unlocks Phoenix's message inspector,
token breakdown, and tool call visualization.

#### 2.2.1. Core Model Attributes

| Current jaato attribute     | OpenInference attribute     | Notes |
|-----------------------------|-----------------------------|-------|
| `gen_ai.system`             | `llm.system`                | Same semantics; add alias. |
| `gen_ai.request.model`      | `llm.model_name`            | Same semantics; add alias. |
| `gen_ai.response.finish_reasons` | *(keep as-is)*         | No direct OI equivalent. |

#### 2.2.2. Token Counts

| Current jaato attribute             | OpenInference attribute                          |
|--------------------------------------|--------------------------------------------------|
| `gen_ai.usage.input_tokens`          | `llm.token_count.prompt`                         |
| `gen_ai.usage.output_tokens`         | `llm.token_count.completion`                     |
| *(computed)*                         | `llm.token_count.total`                          |
| `gen_ai.usage.cache_read_tokens`     | `llm.token_count.prompt_details.cache_read`      |
| `gen_ai.usage.cache_creation_tokens` | `llm.token_count.prompt_details.cache_write`     |
| `gen_ai.usage.reasoning_tokens`      | `llm.token_count.completion_details.reasoning`   |

**Implementation:** When the session sets a `gen_ai.usage.*` attribute on the
LLM span, the `_SpanWrapper.set_attribute()` method (or a post-processing
step) also writes the corresponding `llm.token_count.*` attribute. The
`llm.token_count.total` is computed as `prompt + completion`.

#### 2.2.3. Input/Output Messages

This is the most complex mapping. OpenInference uses **flattened indexed
prefixes** for messages:

```
llm.input_messages.0.message.role = "system"
llm.input_messages.0.message.content = "You are a helpful assistant."
llm.input_messages.1.message.role = "user"
llm.input_messages.1.message.content = "Hello"
```

**Current state:** jaato's LLM span does NOT currently carry input/output
messages as span attributes — they are not set anywhere in the session layer.
The `gen_ai.prompt` and `gen_ai.completion` sensitive attributes exist in the
redaction list but are never actually set.

**Approach:** Add two new methods to `SpanContext` / `_SpanWrapper`:

```python
def set_input_messages(self, messages: List[Dict[str, Any]]) -> None:
    """Set OpenInference-formatted input messages on the span.

    Each message dict should have 'role' and 'content' keys.
    Messages are flattened to indexed attributes:
      llm.input_messages.{i}.message.role
      llm.input_messages.{i}.message.content
    """

def set_output_messages(self, messages: List[Dict[str, Any]]) -> None:
    """Set OpenInference-formatted output messages on the span.

    Same format as input messages. Tool calls within messages are
    flattened to:
      llm.output_messages.{i}.message.tool_calls.{j}.tool_call.function.name
      llm.output_messages.{i}.message.tool_calls.{j}.tool_call.function.arguments
    """
```

These methods handle:
- Content redaction (when `redact_content=True`, content is replaced with
  `[REDACTED: N chars]` but roles and tool call names are preserved).
- Flattening to indexed attribute format.
- Tool call extraction from output messages.

**Session-layer call site** (`jaato_session.py`): After the LLM response is
received, the session calls `llm_span.set_output_messages(...)` with the
response messages. For input messages, the session calls
`llm_span.set_input_messages(...)` with the messages sent to the provider.

> **Note:** This is the ONE place where the session layer needs a small change
> — adding two method calls inside the existing `with self._telemetry.llm_span(...)`
> block. All other changes are in `otel_plugin.py`.

#### 2.2.4. Tool Calls in LLM Output

When the LLM returns tool calls, they appear in the output messages:

```
llm.output_messages.0.message.role = "assistant"
llm.output_messages.0.message.tool_calls.0.tool_call.function.name = "cli"
llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments = '{"command": "ls"}'
```

This is handled by `set_output_messages()` as described above. The session
already has the parsed `FunctionCall` objects; it just needs to pass them.

### 2.3. Tool Span → TOOL

| Current jaato attribute     | OpenInference attribute     | Notes |
|-----------------------------|-----------------------------|-------|
| `jaato.tool.name`           | `tool.name`                 | Add alias. |
| `jaato.tool.call_id`        | `tool.id`                   | Add alias. |
| *(new)*                     | `tool.description`          | Optional; from tool schema if available. |
| *(new)*                     | `input.value`               | Tool arguments (JSON string, if not redacted). |
| *(new)*                     | `output.value`              | Tool result (JSON string, if not redacted). |
| *(new)*                     | `input.mime_type`           | `"application/json"` |
| *(new)*                     | `output.mime_type`          | `"application/json"` |
| `jaato.tool.plugin_type`    | *(keep as-is)*              | jaato-specific. |
| `jaato.tool.success`        | *(keep as-is)*              | jaato-specific; OI uses span status. |
| `jaato.tool.duration_seconds`| *(keep as-is)*             | jaato-specific. |
| `jaato.tool.parallel`       | *(keep as-is)*              | jaato-specific. |

### 2.4. Chain Spans (Retry, GC, Permission)

These internal orchestration spans get `openinference.span.kind = "CHAIN"` and
`input.value` / `output.value` where meaningful. No further OpenInference-specific
attributes are needed — they are jaato internals that Phoenix will render as
generic chain steps in the trace waterfall.

---

## 3. Implementation Plan

All changes are scoped to the telemetry plugin layer. The session layer gets
minimal additions (passing messages to LLM spans).

### 3.1. Changes to `otel_plugin.py`

#### 3.1.1. Constants

Add OpenInference attribute name constants at module level (raw strings, no
external dependency):

```python
# OpenInference span kind (required for Phoenix rendering)
_OI_SPAN_KIND = "openinference.span.kind"

# OpenInference span kind values
_OI_AGENT = "AGENT"
_OI_LLM = "LLM"
_OI_TOOL = "TOOL"
_OI_CHAIN = "CHAIN"
```

#### 3.1.2. `_SpanWrapper` Additions

Add `set_input_messages()` and `set_output_messages()` methods that flatten
message lists to indexed OpenInference attributes. These methods respect the
existing `_redact` flag.

```python
def set_input_messages(self, messages: List[Dict[str, Any]]) -> None:
    for i, msg in enumerate(messages):
        prefix = f"llm.input_messages.{i}.message"
        self._span.set_attribute(f"{prefix}.role", msg.get("role", ""))
        content = msg.get("content", "")
        if self._redact and content:
            content = f"[REDACTED: {len(content)} chars]"
        self._span.set_attribute(f"{prefix}.content", content)

def set_output_messages(self, messages: List[Dict[str, Any]]) -> None:
    for i, msg in enumerate(messages):
        prefix = f"llm.output_messages.{i}.message"
        self._span.set_attribute(f"{prefix}.role", msg.get("role", ""))
        content = msg.get("content", "")
        if self._redact and content:
            content = f"[REDACTED: {len(content)} chars]"
        self._span.set_attribute(f"{prefix}.content", content)
        # Flatten tool calls
        for j, tc in enumerate(msg.get("tool_calls", [])):
            tc_prefix = f"{prefix}.tool_calls.{j}.tool_call"
            self._span.set_attribute(
                f"{tc_prefix}.function.name",
                tc.get("name", ""),
            )
            args = tc.get("arguments", "")
            if self._redact and args:
                args = f"[REDACTED: {len(args)} chars]"
            self._span.set_attribute(f"{tc_prefix}.function.arguments", args)
```

Add a dual-write helper for the token count aliasing pattern:

```python
# Token attribute aliasing: gen_ai.usage.* → llm.token_count.*
_TOKEN_ALIASES = {
    "gen_ai.usage.input_tokens": "llm.token_count.prompt",
    "gen_ai.usage.output_tokens": "llm.token_count.completion",
    "gen_ai.usage.cache_read_tokens": "llm.token_count.prompt_details.cache_read",
    "gen_ai.usage.cache_creation_tokens": "llm.token_count.prompt_details.cache_write",
    "gen_ai.usage.reasoning_tokens": "llm.token_count.completion_details.reasoning",
}

def set_attribute(self, key: str, value: Any) -> None:
    """Set an attribute, redacting sensitive content if configured.

    Also writes OpenInference aliases for gen_ai.usage.* token
    attributes and tracks prompt/completion for total computation.
    """
    # Existing redaction logic ...

    self._span.set_attribute(key, value)

    # Dual-write OpenInference alias if applicable
    oi_alias = _TOKEN_ALIASES.get(key)
    if oi_alias is not None:
        self._span.set_attribute(oi_alias, value)
        # Track for total computation
        if key == "gen_ai.usage.input_tokens":
            self._prompt_tokens = value
        elif key == "gen_ai.usage.output_tokens":
            self._completion_tokens = value
            # Write total when both are known
            if self._prompt_tokens is not None:
                self._span.set_attribute(
                    "llm.token_count.total",
                    self._prompt_tokens + value,
                )
```

> `_prompt_tokens` and `_completion_tokens` are `Optional[int]` fields added
> to `__slots__` initialized to `None`.

#### 3.1.3. Span Method Changes

**`turn_span()`** — add `parent_session_id` parameter:

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
    # ... existing logic ...
    attrs.update({
        # OpenInference
        _OI_SPAN_KIND: _OI_AGENT,
        "session.id": session_id,
        # Graph visualization
        "graph.node.id": session_id,
        "graph.node.name": agent_name or agent_type,
        "graph.node.parent_id": parent_session_id or "",
    })
    if agent_name:
        attrs["agent.name"] = agent_name
    # ... rest unchanged ...
```

**`llm_span()`**:

```python
attrs.update({
    _OI_SPAN_KIND: _OI_LLM,
    "llm.system": provider,       # OI alias
    "llm.model_name": model,      # OI alias
    # Keep existing gen_ai.* attributes
    "gen_ai.system": provider,
    "gen_ai.request.model": model,
    "jaato.streaming": streaming,
})
```

**`tool_span()`**:

```python
attrs.update({
    _OI_SPAN_KIND: _OI_TOOL,
    "tool.name": tool_name,       # OI alias
    "tool.id": call_id,           # OI alias
    # Keep existing jaato.tool.* attributes
    "jaato.tool.name": tool_name,
    "jaato.tool.call_id": call_id,
    "jaato.tool.plugin_type": plugin_type,
})
```

**`retry_span()`**, **`gc_span()`**, **`permission_span()`**:

```python
attrs[_OI_SPAN_KIND] = _OI_CHAIN
# All other attributes unchanged
```

### 3.2. Changes to `plugin.py` (Protocol)

Add `set_input_messages()` and `set_output_messages()` to the `SpanContext`
protocol. Add optional `parent_session_id` parameter to `turn_span()`.

### 3.3. Changes to `null_plugin.py`

Add no-op `set_input_messages()` and `set_output_messages()` to `_NoOpSpan`.

### 3.4. Changes to `jaato_session.py`

Inside the existing `with self._telemetry.llm_span(...)` block, add calls to
pass messages to the span. This requires converting jaato's internal message
format to simple dicts:

```python
# After constructing the messages list for the provider call:
if hasattr(llm_telemetry, 'set_input_messages'):
    oi_messages = []
    for msg in messages_for_provider:
        oi_messages.append({
            "role": msg.role,
            "content": msg.text_content or "",
        })
    llm_telemetry.set_input_messages(oi_messages)

# After receiving the response:
if hasattr(llm_telemetry, 'set_output_messages'):
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

For the turn span, pass `parent_session_id` when creating subagent sessions:

```python
# In send_message():
with self._telemetry.turn_span(
    session_id=self._agent_id,
    agent_type=self._agent_type,
    agent_name=self._agent_name,
    turn_index=self._turn_index,
    parent_session_id=self._parent_session_id,  # NEW
) as turn_span:
```

> `_parent_session_id` is already available on `JaatoSession` — it's set during
> subagent creation via `runtime.create_session()`. If it doesn't exist yet,
> add it as an optional parameter.

### 3.5. Changes to `jaato_session.py` — Tool Input/Output

Inside the existing `with self._telemetry.tool_span(...)` block, add
input/output values:

```python
with self._telemetry.tool_span(...) as tool_span:
    # Set tool input (arguments)
    tool_span.set_attribute("input.value", json.dumps(fc.args) if fc.args else "{}")
    tool_span.set_attribute("input.mime_type", "application/json")

    # ... execute tool ...

    # Set tool output (result)
    tool_span.set_attribute("output.value", json.dumps(result_dict))
    tool_span.set_attribute("output.mime_type", "application/json")
```

The `input.value` and `output.value` attributes should be added to the
`_SENSITIVE_ATTRS` set for redaction.

---

## 4. Redaction Impact

The new OpenInference attributes containing content must respect the existing
redaction policy:

| New attribute                                          | Redactable? |
|--------------------------------------------------------|-------------|
| `llm.input_messages.*.message.content`                 | Yes — via `set_input_messages()` |
| `llm.output_messages.*.message.content`                | Yes — via `set_output_messages()` |
| `llm.output_messages.*.message.tool_calls.*.function.arguments` | Yes — via `set_output_messages()` |
| `input.value`                                          | Yes — add to `_SENSITIVE_ATTRS` |
| `output.value`                                         | Yes — add to `_SENSITIVE_ATTRS` |
| `llm.input_messages.*.message.role`                    | No |
| `llm.output_messages.*.message.role`                   | No |
| `tool_call.function.name`                              | No |
| `llm.model_name`, `llm.system`, `agent.name`          | No |
| `session.id`, `graph.node.*`                           | No |
| `llm.token_count.*`                                    | No |

When `redact_content=True` (default), messages and tool arguments are redacted
inside `set_input_messages()` / `set_output_messages()` directly — the
redaction happens before attributes are written to the span, same as today.

---

## 5. Configuration

### 5.1. Feature Flag

A new config key `openinference` (bool, default `True`) controls whether
OpenInference attributes are emitted. When `False`, only the existing
`jaato.*` and `gen_ai.*` attributes are written.

```python
# Environment variable
JAATO_TELEMETRY_OPENINFERENCE=true  # default

# Programmatic
plugin.initialize({
    "enabled": True,
    "openinference": True,  # default
})
```

**Rationale:** Default `True` because the whole point is Phoenix compatibility.
The flag exists as an escape hatch for users sending to backends that choke on
the extra attributes or have attribute count limits.

### 5.2. No New Dependencies

We use raw attribute name strings rather than the
`openinference-semantic-conventions` package. The constants are defined as
module-level strings in `otel_plugin.py`. This keeps the telemetry plugin
zero-additional-dep (only `opentelemetry-api` and `opentelemetry-sdk`).

---

## 6. Validation & Testing

### 6.1. Unit Tests (`test_plugin.py`)

Add test cases for:

1. **Span kind presence:** Every span type sets `openinference.span.kind`.
2. **Turn span → AGENT** with `session.id`, `agent.name`, `graph.node.*`.
3. **LLM span → LLM** with `llm.model_name`, `llm.system`.
4. **Tool span → TOOL** with `tool.name`, `tool.id`.
5. **Token aliasing:** Setting `gen_ai.usage.input_tokens` also sets
   `llm.token_count.prompt`.
6. **Total computation:** `llm.token_count.total` = prompt + completion.
7. **Message flattening:** `set_input_messages()` produces correct indexed
   attributes.
8. **Tool call flattening:** Tool calls in output messages produce correct
   `tool_call.function.name` and `tool_call.function.arguments`.
9. **Redaction:** Messages redacted when `redact_content=True`.
10. **Feature flag off:** No OpenInference attributes when
    `openinference=False`.
11. **Graph attributes:** Subagent spans have `graph.node.parent_id` pointing
    to parent session.
12. **Backward compatibility:** All existing `jaato.*` and `gen_ai.*`
    attributes still present.

### 6.2. Integration Validation with Phoenix

Manual validation steps (not automated):

1. Start Phoenix: `phoenix serve`
2. Configure jaato: `JAATO_TELEMETRY_ENABLED=true OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006/v1/traces`
3. Run a multi-tool conversation.
4. Verify in Phoenix UI:
   - Trace waterfall shows AGENT → LLM → TOOL hierarchy.
   - LLM spans show message inspector with input/output.
   - Token counts appear in the cost breakdown.
   - Tool calls are visible with names and arguments.
   - Multi-agent sessions show DAG visualization via `graph.node.*`.

---

## 7. Example: Before and After

### Before (current jaato spans)

```
Span: "jaato.turn"
  jaato.session_id: "sess-abc"
  jaato.agent_type: "main"
  jaato.turn_index: 3

  Child: "gen_ai.chat"
    gen_ai.system: "anthropic"
    gen_ai.request.model: "claude-sonnet-4-20250514"
    gen_ai.usage.input_tokens: 1500
    gen_ai.usage.output_tokens: 200
    gen_ai.usage.cache_read_tokens: 800

    Child: "jaato.tool"
      jaato.tool.name: "cli"
      jaato.tool.call_id: "call_123"
      jaato.tool.plugin_type: "cli"
      jaato.tool.success: true
      jaato.tool.duration_seconds: 0.5
```

**Phoenix rendering:** Generic span waterfall. No LLM details, no messages,
no token breakdown.

### After (with OpenInference attributes added)

```
Span: "jaato.turn"
  openinference.span.kind: "AGENT"
  session.id: "sess-abc"
  agent.name: "main"
  graph.node.id: "sess-abc"
  graph.node.name: "main"
  graph.node.parent_id: ""
  # (all existing jaato.* attributes preserved)
  jaato.session_id: "sess-abc"
  jaato.agent_type: "main"
  jaato.turn_index: 3

  Child: "gen_ai.chat"
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
    # (all existing gen_ai.* attributes preserved)
    gen_ai.system: "anthropic"
    gen_ai.request.model: "claude-sonnet-4-20250514"
    gen_ai.usage.input_tokens: 1500
    gen_ai.usage.output_tokens: 200
    gen_ai.usage.cache_read_tokens: 800

    Child: "jaato.tool"
      openinference.span.kind: "TOOL"
      tool.name: "cli"
      tool.id: "call_123"
      input.value: '{"command":"ls /tmp"}'
      input.mime_type: "application/json"
      output.value: '{"status":"success","output":"file1.txt\nfile2.txt"}'
      output.mime_type: "application/json"
      # (all existing jaato.tool.* attributes preserved)
      jaato.tool.name: "cli"
      jaato.tool.call_id: "call_123"
      jaato.tool.plugin_type: "cli"
      jaato.tool.success: true
      jaato.tool.duration_seconds: 0.5
```

**Phoenix rendering:** Full AI-native UI — agent graph, message inspector,
token cost breakdown, tool call visualization.

---

## 8. Migration Path

### Phase 1 — Core Span Kinds (this design)

- Add `openinference.span.kind` to all spans.
- Add `session.id`, `agent.name`, `graph.node.*` to turn spans.
- Add `llm.system`, `llm.model_name`, `llm.token_count.*` to LLM spans.
- Add `tool.name`, `tool.id`, `input.value`, `output.value` to tool spans.
- Add `set_input_messages()` / `set_output_messages()` for LLM message capture.

**Estimated scope:** ~200 lines in `otel_plugin.py`, ~30 lines in
`jaato_session.py`, ~100 lines in tests.

### Phase 2 — Enrichments (future)

- `llm.invocation_parameters` (temperature, max_tokens, etc.)
- `llm.tools` (tool schemas sent to the model)
- `llm.cost.*` (if token pricing data becomes available)
- `metadata` attribute with jaato-specific structured data

### Phase 3 — Full Observability (future)

- Embedding spans for RAG plugin (if/when added)
- Retriever spans for document retrieval
- Guardrail spans for permission/safety checks (reclassify from CHAIN)

---

## 9. File Change Summary

| File | Change |
|------|--------|
| `shared/plugins/telemetry/otel_plugin.py` | Add OI constants, dual-write attributes, `set_input/output_messages()`, graph attrs, `openinference` config flag |
| `shared/plugins/telemetry/plugin.py` | Add `set_input_messages()` / `set_output_messages()` to `SpanContext` protocol; `parent_session_id` to `turn_span()` |
| `shared/plugins/telemetry/null_plugin.py` | Add no-op `set_input_messages()` / `set_output_messages()` to `_NoOpSpan` |
| `shared/plugins/telemetry/tests/test_plugin.py` | Add tests for all OI attributes, message flattening, redaction, feature flag |
| `shared/jaato_session.py` | Pass messages to LLM span, pass `parent_session_id` to turn span, set `input.value`/`output.value` on tool spans |
| `docs/opentelemetry-design.md` | Add OpenInference section referencing this design |
| `CLAUDE.md` | Update telemetry section with OI attribute list |

---

## 10. Open Questions

1. **Span name preservation:** Should we keep the current span names
   (`jaato.turn`, `gen_ai.chat`, `jaato.tool`) or rename them to match
   OpenInference conventions? Phoenix uses `openinference.span.kind` for
   classification, not span names, so keeping current names is safe and
   preserves backward compatibility.
   **Recommendation:** Keep current names. ✓

2. **Message capture performance:** Flattening messages to indexed attributes
   could produce many attributes for long conversations. Should we cap at N
   messages or only capture the last turn's messages?
   **Recommendation:** Only capture messages for the current LLM call (not
   the full history). This is what the span represents anyway. ✓

3. **`openinference-semantic-conventions` dependency:** Should we add it
   as an optional dependency for type safety?
   **Recommendation:** No. Raw strings are sufficient and keep us zero-dep.
   The constants are simple strings that rarely change. ✓
