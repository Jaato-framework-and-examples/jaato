# JAATO OpenTelemetry Integration

## Executive Summary

JAATO integrates **OpenTelemetry (OTel)** as an opt-in observability layer, providing distributed tracing across the entire agentic pipeline -- from user turns through LLM calls, tool executions, permission checks, retries, and garbage collection. The design follows a **plugin-based architecture** with a zero-cost null implementation as the default, ensuring no overhead when telemetry is disabled. Traces follow the emerging **GenAI semantic conventions**, enabling export to any OTel-compatible backend (Langfuse, Arize Phoenix, Helicone, Jaeger, etc.).

---

## Part 1: Design Principles

### Four Principles

| Principle | How Achieved |
|-----------|-------------|
| **Plugin-based** | Telemetry is a `TelemetryPlugin` protocol with two implementations: `OTelPlugin` (real) and `NullTelemetryPlugin` (no-op) |
| **Zero-cost abstraction** | When disabled, no OTel imports occur; all context managers yield no-op objects |
| **Decorator pattern** | Instrumentation via `with telemetry.turn_span(...)` context managers, not inline code changes |
| **Backwards compatible** | Existing code works unchanged; telemetry is purely additive |

### Opt-In Activation

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TELEMETRY ACTIVATION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Default State: DISABLED                                             │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  NullTelemetryPlugin                                       │      │
│  │  • All methods are no-ops                                  │      │
│  │  • No OTel imports                                         │      │
│  │  • Zero overhead                                           │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                      │
│  Activation: JAATO_TELEMETRY_ENABLED=true                            │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  OTelPlugin                                                │      │
│  │  • Imports opentelemetry-api, opentelemetry-sdk            │      │
│  │  • Creates TracerProvider with configured exporter          │      │
│  │  • Spans flow to OTLP endpoint or console                  │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                      │
│  Dependencies installed separately:                                  │
│  pip install -r requirements-telemetry.txt                           │
│  (opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp)│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Span Hierarchy

Every user interaction produces a trace with a well-defined span hierarchy:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SPAN HIERARCHY                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  jaato.turn                        Root: entire send_message() call  │
│  │                                                                   │
│  ├── gen_ai.chat                   LLM API call (1st round)          │
│  │   └── jaato.retry               Retry attempt (if rate limited)   │
│  │                                                                   │
│  ├── jaato.tool                    Tool execution (may be parallel)   │
│  │   ├── jaato.permission_check    Permission decision               │
│  │   └── jaato.mcp_call            MCP server call (if MCP tool)     │
│  │                                                                   │
│  ├── jaato.tool                    Another parallel tool              │
│  │   └── jaato.permission_check                                      │
│  │                                                                   │
│  ├── gen_ai.chat                   Follow-up LLM call with results   │
│  │                                                                   │
│  ├── jaato.tool                    More tool calls...                 │
│  │                                                                   │
│  ├── gen_ai.chat                   Final LLM response (text only)    │
│  │                                                                   │
│  └── jaato.gc                      GC operation (if triggered)        │
│                                                                   │
│  Subagent traces are children of the parent's jaato.tool span:       │
│                                                                      │
│  jaato.turn (parent)                                                 │
│  └── jaato.tool (delegate)                                           │
│      └── jaato.turn (subagent)                                       │
│          ├── gen_ai.chat                                             │
│          ├── jaato.tool                                              │
│          └── gen_ai.chat                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Integration Points

The telemetry plugin hooks into existing code paths without modifying their logic:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTEGRATION POINTS                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Component         Method                    Span Type               │
│  ─────────         ──────                    ─────────               │
│  JaatoSession      send_message()            jaato.turn (root)       │
│  JaatoSession      _run_chat_loop()          gen_ai.chat             │
│  JaatoSession      _execute_function_calls() jaato.tool              │
│  ToolExecutor      execute()                 jaato.permission_check  │
│  with_retry()      retry wrapper             jaato.retry             │
│  JaatoSession      _maybe_collect()          jaato.gc                │
│  Provider          send_message_streaming()  (attributes on parent)  │
│                                                                      │
│  Each integration point uses a context manager:                      │
│                                                                      │
│  with self._telemetry.turn_span(session_id, "main") as span:        │
│      # ... existing send_message logic ...                           │
│      span.set_attribute("jaato.turn_index", self._turn_count)        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Semantic Conventions

JAATO follows the emerging [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) plus custom `jaato.*` attributes:

### GenAI Chat Span Attributes (gen_ai.chat)

| Attribute | Type | Example |
|-----------|------|---------|
| `gen_ai.system` | string | `"anthropic"`, `"google_genai"` |
| `gen_ai.request.model` | string | `"claude-sonnet-4-5-20250929"` |
| `gen_ai.request.max_tokens` | int | `4096` |
| `gen_ai.response.model` | string | `"claude-sonnet-4-5-20250929"` |
| `gen_ai.response.finish_reasons` | string[] | `["stop"]`, `["tool_use"]` |
| `gen_ai.usage.input_tokens` | int | `1250` |
| `gen_ai.usage.output_tokens` | int | `384` |

### Turn Span Attributes (jaato.turn)

| Attribute | Type | Example |
|-----------|------|---------|
| `jaato.session_id` | string | `"sess_abc123"` |
| `jaato.agent_type` | string | `"main"`, `"subagent"` |
| `jaato.agent_name` | string | `"code_reviewer"` |
| `jaato.turn_index` | int | `5` |
| `jaato.streaming` | bool | `true` |
| `jaato.cancelled` | bool | `false` |

### Tool Span Attributes (jaato.tool)

| Attribute | Type | Example |
|-----------|------|---------|
| `jaato.tool.name` | string | `"cli_based_tool"` |
| `jaato.tool.plugin_type` | string | `"cli"`, `"mcp"`, `"builtin"` |
| `jaato.tool.call_id` | string | `"call_xyz789"` |
| `jaato.tool.success` | bool | `true` |
| `jaato.tool.error` | string | `"Permission denied"` |
| `jaato.tool.mcp_server` | string | `"filesystem"` |

### Retry Span Attributes (jaato.retry)

| Attribute | Type | Example |
|-----------|------|---------|
| `jaato.retry.attempt` | int | `2` |
| `jaato.retry.max_attempts` | int | `5` |
| `jaato.retry.delay_seconds` | float | `4.5` |
| `jaato.retry.error_type` | string | `"rate_limit"` |

### GC Span Attributes (jaato.gc)

| Attribute | Type | Example |
|-----------|------|---------|
| `jaato.gc.trigger_reason` | string | `"threshold"`, `"manual"` |
| `jaato.gc.strategy` | string | `"budget"`, `"truncate"` |
| `jaato.gc.items_collected` | int | `12` |
| `jaato.gc.tokens_freed` | int | `8500` |
| `jaato.gc.context_before` | float | `85.2` |
| `jaato.gc.context_after` | float | `45.1` |

---

## Part 5: TelemetryPlugin Protocol

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TelemetryPlugin Interface                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Lifecycle:                                                          │
│  ──────────                                                          │
│  initialize(config) → None        Set up tracer, exporter            │
│  shutdown() → None                Flush spans, clean up              │
│                                                                      │
│  Span Creation (Context Managers):                                   │
│  ─────────────────────────────────                                   │
│  turn_span(session_id, agent_type, ...) → Span                      │
│  llm_span(model, provider, streaming, ...) → Span                   │
│  tool_span(tool_name, call_id, plugin_type, ...) → Span             │
│  retry_span(attempt, max_attempts, ...) → Span                      │
│  gc_span(trigger_reason, strategy, ...) → Span                      │
│                                                                      │
│  Span Operations:                                                    │
│  ────────────────                                                    │
│  record_exception(exception) → None                                  │
│  set_attribute(key, value) → None                                    │
│  add_event(name, attributes) → None                                  │
│                                                                      │
│  Implementations:                                                    │
│  ┌───────────────────────┐  ┌───────────────────────────────────┐   │
│  │ NullTelemetryPlugin   │  │ OTelPlugin                        │   │
│  │ (Default)             │  │ (When enabled)                    │   │
│  │                       │  │                                   │   │
│  │ All methods: no-op    │  │ Creates real OTel spans           │   │
│  │ Yields _NoOpSpan      │  │ Yields _SpanWrapper               │   │
│  │ Zero imports          │  │ Exports to OTLP/console           │   │
│  │ Zero overhead         │  │ Supports redaction                │   │
│  └───────────────────────┘  └───────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JAATO_TELEMETRY_ENABLED` | `false` | Enable telemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | None | OTLP endpoint URL |
| `OTEL_EXPORTER_OTLP_HEADERS` | None | Auth headers |
| `OTEL_SERVICE_NAME` | `jaato` | Service name for traces |
| `JAATO_TELEMETRY_SAMPLE_RATE` | `1.0` | Sampling rate (0.0-1.0) |
| `JAATO_TELEMETRY_REDACT_CONTENT` | `true` | Redact prompts/responses |
| `JAATO_TELEMETRY_EXPORTER` | `otlp` | Exporter type (otlp/console) |

### Config File (`.jaato/telemetry.json`)

```json
{
  "enabled": true,
  "service_name": "jaato",
  "exporter": "otlp",
  "endpoint": "http://localhost:4317",
  "sample_rate": 1.0,
  "redact_content": true,
  "batch_export": true
}
```

### Programmatic Configuration

```python
from shared.plugins.telemetry import create_otel_plugin

telemetry = create_otel_plugin()
telemetry.initialize({
    "enabled": True,
    "exporter": "otlp",
    "endpoint": "http://localhost:4317",
    "redact_content": True,
})

runtime.set_telemetry_plugin(telemetry)
```

---

## Part 7: Content Redaction

By default, prompts and model responses are NOT included in spans to prevent sensitive data leakage:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTENT REDACTION                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Default (redact_content: true):                                     │
│  ──────────────────────────────                                      │
│  span.set_attribute("gen_ai.prompt", "[REDACTED: 1250 chars]")      │
│  span.set_attribute("gen_ai.completion", "[REDACTED: 384 chars]")   │
│                                                                      │
│  Opt-in (redact_content: false):                                     │
│  ─────────────────────────────                                       │
│  span.set_attribute("gen_ai.prompt", "User's full prompt text...")   │
│  span.set_attribute("gen_ai.completion", "Model's response...")      │
│                                                                      │
│  The _SpanWrapper class intercepts set_attribute() calls and         │
│  applies redaction to sensitive fields automatically.                 │
│                                                                      │
│  Non-sensitive attributes (token counts, model names, tool names)    │
│  are always included regardless of redaction setting.                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 8: Context Propagation

### Thread Boundaries

JAATO uses threads for parallel tool execution. OTel context must propagate across thread boundaries:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THREAD CONTEXT PROPAGATION                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Main Thread (jaato.turn span active)                                │
│       │                                                              │
│       │  Capture context:                                            │
│       │  ctx = context.get_current()                                 │
│       │                                                              │
│       ├──► Thread Pool (tool execution)                              │
│       │    │                                                         │
│       │    │  token = attach(ctx)                                    │
│       │    │  with telemetry.tool_span(...):                         │
│       │    │      execute_tool()     ◄── Child of jaato.turn         │
│       │    │  detach(token)                                          │
│       │                                                              │
│       ├──► Thread Pool (another tool)                                │
│       │    │                                                         │
│       │    │  token = attach(ctx)                                    │
│       │    │  with telemetry.tool_span(...):                         │
│       │    │      execute_tool()     ◄── Also child of jaato.turn    │
│       │    │  detach(token)                                          │
│       │                                                              │
│       └──► All tool spans appear as children of the turn span        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Subagent Sessions

Subagents inherit the telemetry plugin from the parent runtime. Trace context propagates through `JaatoRuntime`:

```
runtime._telemetry → shared by all sessions
    │
    ├──► Main session uses it for jaato.turn spans
    └──► Subagent session uses same plugin
         Subagent spans appear as children in the same trace
```

---

## Part 9: Backend Compatibility

JAATO exports standard OTel traces, compatible with any OTel-capable backend:

| Backend | Configuration |
|---------|--------------|
| **Langfuse** | `endpoint: https://cloud.langfuse.com`, `headers: {Authorization: Bearer KEY}` |
| **Arize Phoenix** | `endpoint: http://localhost:6006/v1/traces` |
| **Helicone** | `endpoint: https://otel.helicone.ai`, `headers: {Helicone-Auth: Bearer KEY}` |
| **Jaeger** | `endpoint: http://localhost:4317` (OTLP gRPC) |
| **Console** | `exporter: console` (prints spans to stdout, useful for debugging) |

---

## Part 10: Relationship to Token Ledger

The `TokenLedger` and telemetry serve complementary but distinct purposes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LEDGER vs TELEMETRY                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TokenLedger (Existing)                                              │
│  ──────────────────────                                              │
│  Purpose: Billing and accounting                                     │
│  Granularity: Per-session aggregate                                  │
│  Output: JSONL file (LEDGER_PATH)                                    │
│  Data: model, prompt_tokens, output_tokens, cost                     │
│  Overhead: Always on, minimal                                        │
│                                                                      │
│  TelemetryPlugin (New)                                               │
│  ─────────────────────                                               │
│  Purpose: Debugging and observability                                │
│  Granularity: Per-span with timing                                   │
│  Output: OTLP to any backend                                         │
│  Data: Spans, attributes, events, exceptions, timing                 │
│  Overhead: Opt-in only                                               │
│                                                                      │
│  They coexist — same code path records both:                         │
│                                                                      │
│  with telemetry.llm_span(model, provider, streaming=True) as span:  │
│      response = provider.send_message_streaming(...)                 │
│      # Ledger records usage (always)                                 │
│      ledger.record_request(model, prompt_tokens, output_tokens)      │
│      # Telemetry records span attributes (when enabled)              │
│      span.set_attribute("gen_ai.usage.input_tokens", ...)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 11: Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              JaatoRuntime                            │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │ PluginRegistry  │    │ TelemetryPlugin │    │ TokenLedger    │  │
│  │                 │    │                 │    │ (existing)     │  │
│  └────────┬────────┘    └────────┬────────┘    └────────────────┘  │
│           │                      │                                   │
│           │                      │ creates & manages                 │
│           │                      ▼                                   │
│           │             ┌─────────────────────────────────────────┐  │
│           │             │              TracerProvider             │  │
│           │             │  ┌─────────┐  ┌──────────┐  ┌────────┐│  │
│           │             │  │ Tracer  │  │ Exporter │  │Processor││  │
│           │             │  │ "jaato" │  │ (OTLP)   │  │ (Batch) ││  │
│           │             │  └─────────┘  └──────────┘  └────────┘│  │
│           │             └─────────────────────────────────────────┘  │
│           │                                                          │
│           ▼                                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                         JaatoSession                           │  │
│  │                                                                │  │
│  │  send_message() ──► jaato.turn span (root)                    │  │
│  │  _run_chat_loop() ──► gen_ai.chat spans (per LLM call)       │  │
│  │  _execute_tools() ──► jaato.tool spans (per tool)             │  │
│  │  with_retry() ──► jaato.retry spans (per retry)               │  │
│  │  _maybe_collect() ──► jaato.gc spans (per GC event)           │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 12: Related Documentation

| Document | Focus |
|----------|-------|
| [opentelemetry-design.md](opentelemetry-design.md) | Full OTel implementation design and code examples |
| [jaato_model_harness.md](jaato_model_harness.md) | The harness layers that telemetry instruments |
| [jaato_tool_system.md](jaato_tool_system.md) | Tool execution pipeline (instrumented with tool spans) |
| [jaato_multi_provider.md](jaato_multi_provider.md) | Provider abstraction (gen_ai.system attribute) |
| [jaato_subagent_architecture.md](jaato_subagent_architecture.md) | Subagent trace correlation |
| [architecture.md](architecture.md) | Server-first architecture overview |

---

## Part 13: Color Coding Suggestion for Infographic

- **Blue:** Turn spans (jaato.turn) -- the root of each trace
- **Green:** LLM spans (gen_ai.chat) -- model API calls
- **Orange:** Tool spans (jaato.tool) -- tool execution
- **Red:** Permission spans (jaato.permission_check) -- safety checks
- **Yellow:** Retry spans (jaato.retry) -- error recovery
- **Purple:** GC spans (jaato.gc) -- garbage collection
- **Gray:** Transport and export infrastructure (TracerProvider, OTLP exporter)
- **Cyan:** Configuration (environment variables, config files, programmatic setup)
