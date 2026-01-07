# OpenTelemetry Integration Design for jaato

This document describes the architecture and implementation plan for adding OpenTelemetry (OTel) export capabilities to jaato.

## 1. Overview

### 1.1 Goals

**Must Have:**
- Trace hierarchy with root span for `send_message()` and child spans for LLM calls, tool executions, and retries
- Follow OpenTelemetry semantic conventions for GenAI (emerging standard)
- Provider-agnostic instrumentation working across Google GenAI, Anthropic, and GitHub Models
- Opt-in design with zero overhead when disabled
- Backend-agnostic export to any OTel-compatible collector

**Nice to Have:**
- Streaming token tracking via `on_chunk` callbacks
- GC event tracing
- Subagent correlation (parent-child trace linking)
- Permission event tracing
- MCP vs CLI tool type distinction in spans

**Non-Goals:**
- Building a custom observability UI
- Evaluation/scoring framework
- Dataset management

### 1.2 Design Principles

1. **Plugin-based**: Telemetry is implemented as an optional plugin (`shared/plugins/telemetry/`)
2. **Zero-cost abstraction**: When disabled, no OTel imports or overhead
3. **Decorator pattern**: Instrumentation via decorators/context managers, not code changes
4. **Backwards compatible**: Existing code works unchanged; telemetry is additive

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JaatoRuntime                                    │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ PluginRegistry  │    │ TelemetryPlugin │    │    TokenLedger          │  │
│  │                 │    │                 │    │    (existing)           │  │
│  └────────┬────────┘    └────────┬────────┘    └─────────────────────────┘  │
│           │                      │                                           │
│           │                      │ creates & manages                         │
│           │                      ▼                                           │
│           │             ┌─────────────────────────────────────────────────┐  │
│           │             │              TracerProvider                     │  │
│           │             │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │  │
│           │             │  │ Tracer  │  │Exporter │  │ SpanProcessor   │  │  │
│           │             │  │ "jaato" │  │ (OTLP)  │  │ (Batch/Simple)  │  │  │
│           │             │  └─────────┘  └─────────┘  └─────────────────┘  │  │
│           │             └─────────────────────────────────────────────────┘  │
└───────────┼─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JaatoSession                                    │
│                                                                              │
│  send_message() ─────────► ┌────────────────┐                               │
│                            │  TurnSpan      │  Root span for entire turn    │
│                            │  (jaato.turn)  │                               │
│                            └───────┬────────┘                               │
│                                    │                                         │
│  _run_chat_loop() ────────────────┼───────────────────────────────────────► │
│                                    │                                         │
│            ┌───────────────────────┼───────────────────────┐                │
│            │                       │                       │                │
│            ▼                       ▼                       ▼                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  LLMSpan        │    │  ToolSpan       │    │  RetrySpan              │  │
│  │ (gen_ai.chat)   │    │ (jaato.tool)    │    │ (jaato.retry)           │  │
│  │                 │    │                 │    │                         │  │
│  │ model, tokens,  │    │ tool_name,      │    │ attempt, delay,         │  │
│  │ finish_reason   │    │ duration, error │    │ error_type              │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Span Hierarchy

```
jaato.turn                          # Root: entire send_message() call
├── gen_ai.chat                     # LLM API call (may repeat for function calling loop)
│   └── jaato.retry                 # Retry attempt (if rate limited)
├── jaato.tool                      # Tool execution
│   ├── jaato.permission_check      # Permission decision
│   └── jaato.mcp_call              # MCP server call (if MCP tool)
├── gen_ai.chat                     # Follow-up LLM call with tool results
└── jaato.gc                        # GC operation (if triggered)
```

### 2.3 Integration Points

| Component | File | Method | Span Type |
|-----------|------|--------|-----------|
| Session | `jaato_session.py:963` | `send_message()` | `jaato.turn` (root) |
| Session | `jaato_session.py:1188` | `_run_chat_loop()` | `gen_ai.chat` |
| Session | `jaato_session.py:1569` | `_execute_function_call_group()` | `jaato.tool` |
| Tool Executor | `ai_tool_runner.py:327` | `execute()` | `jaato.permission_check` |
| Retry | `utils.py` (with_retry) | `with_retry()` | `jaato.retry` |
| GC | `jaato_session.py:1053` | `_maybe_collect_after_turn()` | `jaato.gc` |
| Provider | `provider.py` | `send_message_streaming()` | (attributes on parent) |

## 3. Semantic Conventions

Following [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (draft):

### 3.1 GenAI Chat Span Attributes

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `gen_ai.system` | string | AI provider system | `"anthropic"`, `"google_genai"` |
| `gen_ai.request.model` | string | Model identifier | `"claude-sonnet-4-20250514"` |
| `gen_ai.request.max_tokens` | int | Max output tokens requested | `4096` |
| `gen_ai.request.temperature` | float | Sampling temperature | `0.7` |
| `gen_ai.response.model` | string | Actual model used | `"claude-sonnet-4-20250514"` |
| `gen_ai.response.finish_reasons` | string[] | Why generation stopped | `["stop"]`, `["tool_use"]` |
| `gen_ai.usage.input_tokens` | int | Prompt tokens | `1250` |
| `gen_ai.usage.output_tokens` | int | Completion tokens | `384` |

### 3.2 jaato Custom Attributes

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `jaato.session_id` | string | Session identifier | `"sess_abc123"` |
| `jaato.agent_type` | string | Agent type | `"main"`, `"subagent"` |
| `jaato.agent_name` | string | Agent name | `"code_reviewer"` |
| `jaato.turn_index` | int | Turn number in session | `5` |
| `jaato.streaming` | bool | Whether streaming was used | `true` |
| `jaato.cancelled` | bool | Whether turn was cancelled | `false` |

### 3.3 Tool Span Attributes

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `jaato.tool.name` | string | Tool name | `"cli_based_tool"` |
| `jaato.tool.plugin_type` | string | Plugin type | `"cli"`, `"mcp"`, `"builtin"` |
| `jaato.tool.call_id` | string | Function call ID | `"call_xyz789"` |
| `jaato.tool.success` | bool | Execution success | `true` |
| `jaato.tool.error` | string | Error message if failed | `"Permission denied"` |
| `jaato.tool.mcp_server` | string | MCP server name (if MCP) | `"filesystem"` |

### 3.4 Retry Span Attributes

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `jaato.retry.attempt` | int | Attempt number | `2` |
| `jaato.retry.max_attempts` | int | Maximum attempts | `5` |
| `jaato.retry.delay_seconds` | float | Backoff delay | `4.5` |
| `jaato.retry.error_type` | string | Error classification | `"rate_limit"`, `"transient"` |
| `jaato.retry.error_message` | string | Error details | `"429 Too Many Requests"` |

### 3.5 GC Span Attributes

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `jaato.gc.trigger_reason` | string | Why GC was triggered | `"threshold"`, `"manual"` |
| `jaato.gc.strategy` | string | GC strategy used | `"truncate"`, `"summarize"` |
| `jaato.gc.items_collected` | int | Items removed | `12` |
| `jaato.gc.tokens_freed` | int | Tokens reclaimed | `8500` |
| `jaato.gc.context_before` | float | Context % before GC | `85.2` |
| `jaato.gc.context_after` | float | Context % after GC | `45.1` |

## 4. Plugin Design

### 4.1 TelemetryPlugin Interface

```python
# shared/plugins/telemetry/plugin.py

from typing import Any, Dict, Optional, Protocol
from contextlib import contextmanager

class TelemetryPlugin(Protocol):
    """Protocol for telemetry/tracing plugins."""

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the telemetry system.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default True)
                - service_name: str (default "jaato")
                - exporter: str ("otlp", "console", "none")
                - endpoint: str (OTLP endpoint URL)
                - headers: Dict[str, str] (auth headers)
                - batch_export: bool (default True)
                - sample_rate: float (0.0-1.0, default 1.0)
                - redact_content: bool (redact prompts/responses)
        """
        ...

    def shutdown(self) -> None:
        """Flush and shutdown telemetry."""
        ...

    @contextmanager
    def turn_span(
        self,
        session_id: str,
        agent_type: str,
        agent_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create root span for a turn.

        Usage:
            with telemetry.turn_span("sess_123", "main") as span:
                # ... run turn ...
                span.set_attribute("jaato.turn_index", 5)
        """
        ...

    @contextmanager
    def llm_span(
        self,
        model: str,
        provider: str,
        streaming: bool = False,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create span for LLM API call.

        Usage:
            with telemetry.llm_span("claude-sonnet-4-20250514", "anthropic") as span:
                response = provider.send_message(...)
                span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        """
        ...

    @contextmanager
    def tool_span(
        self,
        tool_name: str,
        call_id: str,
        plugin_type: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create span for tool execution."""
        ...

    @contextmanager
    def retry_span(
        self,
        attempt: int,
        max_attempts: int,
        context: str = "api_call",
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create span for retry attempt."""
        ...

    @contextmanager
    def gc_span(
        self,
        trigger_reason: str,
        strategy: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create span for GC operation."""
        ...

    def record_exception(self, exception: Exception) -> None:
        """Record exception on current span."""
        ...

    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute on current span."""
        ...

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to current span."""
        ...
```

### 4.2 OTel Implementation

```python
# shared/plugins/telemetry/otel_plugin.py

from typing import Any, Dict, Optional
from contextlib import contextmanager
import os

# Lazy imports to avoid overhead when disabled
_tracer = None
_provider = None

class OTelPlugin:
    """OpenTelemetry implementation of TelemetryPlugin."""

    def __init__(self):
        self._enabled = False
        self._tracer = None
        self._redact_content = True

    def initialize(self, config: Dict[str, Any]) -> None:
        self._enabled = config.get("enabled", True)
        if not self._enabled:
            return

        # Import OTel only when enabled
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME

        service_name = config.get("service_name", "jaato")
        resource = Resource.create({SERVICE_NAME: service_name})

        provider = TracerProvider(resource=resource)

        # Configure exporter
        exporter_type = config.get("exporter", "otlp")
        if exporter_type == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            endpoint = config.get("endpoint", os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
            headers = config.get("headers", {})
            exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        elif exporter_type == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        else:
            exporter = None

        if exporter:
            if config.get("batch_export", True):
                processor = BatchSpanProcessor(exporter)
            else:
                processor = SimpleSpanProcessor(exporter)
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer("jaato", "0.1.0")
        self._redact_content = config.get("redact_content", True)

    def shutdown(self) -> None:
        if self._enabled and self._tracer:
            from opentelemetry import trace
            provider = trace.get_tracer_provider()
            if hasattr(provider, 'shutdown'):
                provider.shutdown()

    @contextmanager
    def turn_span(
        self,
        session_id: str,
        agent_type: str,
        agent_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        if not self._enabled:
            yield _NoOpSpan()
            return

        from opentelemetry import trace

        attrs = {
            "jaato.session_id": session_id,
            "jaato.agent_type": agent_type,
        }
        if agent_name:
            attrs["jaato.agent_name"] = agent_name
        if attributes:
            attrs.update(attributes)

        with self._tracer.start_as_current_span("jaato.turn", attributes=attrs) as span:
            yield _SpanWrapper(span, self._redact_content)

    @contextmanager
    def llm_span(
        self,
        model: str,
        provider: str,
        streaming: bool = False,
        attributes: Optional[Dict[str, Any]] = None
    ):
        if not self._enabled:
            yield _NoOpSpan()
            return

        attrs = {
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "jaato.streaming": streaming,
        }
        if attributes:
            attrs.update(attributes)

        with self._tracer.start_as_current_span("gen_ai.chat", attributes=attrs) as span:
            yield _SpanWrapper(span, self._redact_content)

    @contextmanager
    def tool_span(
        self,
        tool_name: str,
        call_id: str,
        plugin_type: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None
    ):
        if not self._enabled:
            yield _NoOpSpan()
            return

        attrs = {
            "jaato.tool.name": tool_name,
            "jaato.tool.call_id": call_id,
            "jaato.tool.plugin_type": plugin_type,
        }
        if attributes:
            attrs.update(attributes)

        with self._tracer.start_as_current_span("jaato.tool", attributes=attrs) as span:
            yield _SpanWrapper(span, self._redact_content)

    @contextmanager
    def retry_span(
        self,
        attempt: int,
        max_attempts: int,
        context: str = "api_call",
        attributes: Optional[Dict[str, Any]] = None
    ):
        if not self._enabled:
            yield _NoOpSpan()
            return

        attrs = {
            "jaato.retry.attempt": attempt,
            "jaato.retry.max_attempts": max_attempts,
            "jaato.retry.context": context,
        }
        if attributes:
            attrs.update(attributes)

        with self._tracer.start_as_current_span("jaato.retry", attributes=attrs) as span:
            yield _SpanWrapper(span, self._redact_content)

    @contextmanager
    def gc_span(
        self,
        trigger_reason: str,
        strategy: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        if not self._enabled:
            yield _NoOpSpan()
            return

        attrs = {
            "jaato.gc.trigger_reason": trigger_reason,
            "jaato.gc.strategy": strategy,
        }
        if attributes:
            attrs.update(attributes)

        with self._tracer.start_as_current_span("jaato.gc", attributes=attrs) as span:
            yield _SpanWrapper(span, self._redact_content)


class _SpanWrapper:
    """Wrapper providing consistent interface with content redaction."""

    def __init__(self, span, redact_content: bool):
        self._span = span
        self._redact = redact_content

    def set_attribute(self, key: str, value: Any) -> None:
        # Redact potentially sensitive content
        if self._redact and key in ("gen_ai.prompt", "gen_ai.completion"):
            value = f"[REDACTED: {len(str(value))} chars]"
        self._span.set_attribute(key, value)

    def record_exception(self, exception: Exception) -> None:
        self._span.record_exception(exception)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self._span.add_event(name, attributes=attributes)

    def set_status(self, status) -> None:
        self._span.set_status(status)


class _NoOpSpan:
    """No-op span for when telemetry is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def set_status(self, status) -> None:
        pass
```

### 4.3 Null Implementation (Default)

```python
# shared/plugins/telemetry/null_plugin.py

from typing import Any, Dict, Optional
from contextlib import contextmanager

class NullTelemetryPlugin:
    """No-op telemetry plugin - zero overhead when telemetry disabled."""

    def initialize(self, config: Dict[str, Any]) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @contextmanager
    def turn_span(self, *args, **kwargs):
        yield _NoOpSpan()

    @contextmanager
    def llm_span(self, *args, **kwargs):
        yield _NoOpSpan()

    @contextmanager
    def tool_span(self, *args, **kwargs):
        yield _NoOpSpan()

    @contextmanager
    def retry_span(self, *args, **kwargs):
        yield _NoOpSpan()

    @contextmanager
    def gc_span(self, *args, **kwargs):
        yield _NoOpSpan()

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass
    def record_exception(self, exception: Exception) -> None:
        pass
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass
    def set_status(self, status) -> None:
        pass
```

## 5. Configuration

### 5.1 Environment Variables

Following standard OTel environment variables plus jaato-specific ones:

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL | None (disabled) |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers (key=value,key2=value2) | None |
| `OTEL_SERVICE_NAME` | Service name | `jaato` |
| `JAATO_TELEMETRY_ENABLED` | Enable telemetry | `false` |
| `JAATO_TELEMETRY_SAMPLE_RATE` | Sample rate (0.0-1.0) | `1.0` |
| `JAATO_TELEMETRY_REDACT_CONTENT` | Redact prompts/responses | `true` |
| `JAATO_TELEMETRY_EXPORTER` | Exporter type (otlp/console) | `otlp` |

### 5.2 Programmatic Configuration

```python
from shared.jaato_client import JaatoClient
from shared.plugins.telemetry import create_otel_plugin

# Create and configure telemetry
telemetry = create_otel_plugin()
telemetry.initialize({
    "enabled": True,
    "service_name": "my-app",
    "exporter": "otlp",
    "endpoint": "http://localhost:4317",
    "headers": {"Authorization": "Bearer xxx"},
    "redact_content": True,
    "sample_rate": 1.0,
})

# Inject into runtime
client = JaatoClient()
client.set_telemetry_plugin(telemetry)

# Or via runtime directly
runtime = client.get_runtime()
runtime.set_telemetry_plugin(telemetry)
```

### 5.3 Config File (`.jaato/telemetry.json`)

```json
{
  "enabled": true,
  "service_name": "jaato",
  "exporter": "otlp",
  "endpoint": "http://localhost:4317",
  "headers": {
    "Authorization": "Bearer ${LANGFUSE_API_KEY}"
  },
  "sample_rate": 1.0,
  "redact_content": true,
  "batch_export": true
}
```

## 6. Integration with Existing Infrastructure

### 6.1 Token Ledger Relationship

The `TokenLedger` and telemetry serve different purposes:
- **TokenLedger**: Billing/accounting (aggregated per-session)
- **Telemetry**: Debugging/observability (per-span with timing)

They will coexist:

```python
# In JaatoSession._run_chat_loop()

# Existing ledger recording (unchanged)
self._ledger.record_request(model, prompt_tokens, output_tokens)

# New telemetry span attributes
with self._telemetry.llm_span(model, provider, streaming=True) as span:
    response = provider.send_message_streaming(...)
    span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
```

### 6.2 UI Hooks Integration

Existing `_ui_hooks` events map naturally to telemetry:

| UI Hook | Telemetry Equivalent |
|---------|---------------------|
| `on_agent_created()` | `turn_span()` start |
| `on_tool_call_start()` | `tool_span()` start |
| `on_tool_call_end()` | `tool_span()` end |
| `on_agent_turn_completed()` | `turn_span()` end |

The hooks will be augmented to also emit telemetry, not replaced.

### 6.3 Server Events Integration

Server events (`server/events.py`) can optionally propagate trace context:

```python
@dataclass
class AgentCreatedEvent(Event):
    # ... existing fields ...
    trace_id: Optional[str] = None      # OTel trace ID
    span_id: Optional[str] = None       # OTel span ID
```

This enables:
- Correlating client-side logs with server traces
- Distributed tracing across IPC/WebSocket boundaries
- External tools to join the same trace

## 7. Context Propagation

### 7.1 Thread Boundaries

jaato uses threads for tool execution and MCP calls. Context must propagate:

```python
from opentelemetry import context
from opentelemetry.context import attach, detach

# Capture context before spawning thread
ctx = context.get_current()

def thread_worker():
    token = attach(ctx)
    try:
        # Spans created here will be children of the captured context
        with telemetry.tool_span(...):
            execute_tool()
    finally:
        detach(token)

thread = Thread(target=thread_worker)
thread.start()
```

### 7.2 Subagent Sessions

Subagents inherit telemetry from parent via `JaatoRuntime`:

```python
# In JaatoRuntime.create_session()
def create_session(self, model, tools, system_instructions):
    session = JaatoSession(self, model, tools, system_instructions)
    # Telemetry plugin is shared via runtime
    session._telemetry = self._telemetry
    # Trace context is automatically inherited
    return session
```

### 7.3 Async/Await

For async providers, use async context managers:

```python
from opentelemetry import trace

async def send_message_async(self, message):
    with self._telemetry.llm_span(self._model, "anthropic") as span:
        response = await self._client.messages.create(...)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
        return response
```

## 8. Sensitive Data Handling

### 8.1 Default Redaction

By default, prompts and responses are NOT logged to spans:

```python
# These are redacted by default
span.set_attribute("gen_ai.prompt", "[REDACTED: 1250 chars]")
span.set_attribute("gen_ai.completion", "[REDACTED: 384 chars]")
```

### 8.2 Opt-in Content Logging

Users can enable full content logging:

```python
telemetry.initialize({
    "redact_content": False,  # Enable full content logging
})
```

### 8.3 Selective Redaction

Future: configurable redaction rules:

```json
{
  "redaction": {
    "patterns": ["password", "api_key", "secret"],
    "fields": ["gen_ai.prompt", "jaato.tool.args.password"]
  }
}
```

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# shared/plugins/telemetry/tests/test_otel_plugin.py

def test_disabled_telemetry_zero_overhead():
    """Verify disabled telemetry has no imports or overhead."""
    plugin = NullTelemetryPlugin()
    plugin.initialize({"enabled": False})

    with plugin.turn_span("sess", "main") as span:
        span.set_attribute("key", "value")  # No-op

    # No OTel imports should have occurred
    assert "opentelemetry" not in sys.modules

def test_span_hierarchy():
    """Verify correct parent-child relationships."""
    plugin = OTelPlugin()
    plugin.initialize({"enabled": True, "exporter": "memory"})

    with plugin.turn_span("sess", "main"):
        with plugin.llm_span("model", "provider"):
            with plugin.tool_span("tool", "call_1"):
                pass

    spans = get_exported_spans()
    assert spans[0].name == "jaato.turn"
    assert spans[1].parent.span_id == spans[0].span_id
    assert spans[2].parent.span_id == spans[1].span_id
```

### 9.2 Integration Tests

```python
def test_full_turn_instrumentation():
    """End-to-end test of turn with tool call."""
    client = JaatoClient()
    client.set_telemetry_plugin(create_test_plugin())

    response = client.send_message("Use the calculator to add 2+2")

    spans = get_exported_spans()
    assert any(s.name == "jaato.turn" for s in spans)
    assert any(s.name == "gen_ai.chat" for s in spans)
    assert any(s.name == "jaato.tool" and s.attributes["jaato.tool.name"] == "calculator" for s in spans)
```

### 9.3 Manual Verification

```bash
# Start local collector
docker run -p 4317:4317 otel/opentelemetry-collector-contrib

# Run with console exporter for debugging
JAATO_TELEMETRY_ENABLED=true \
JAATO_TELEMETRY_EXPORTER=console \
python -c "
from shared.jaato_client import JaatoClient
client = JaatoClient()
client.send_message('Hello')
"
```

## 10. Implementation Plan

### Phase 1: Foundation (Week 1)

**Deliverables:**
- `shared/plugins/telemetry/` directory structure
- `plugin.py` - Protocol definition
- `null_plugin.py` - No-op implementation
- `otel_plugin.py` - Basic OTel implementation
- Unit tests for all span types

**Files to create:**
```
shared/plugins/telemetry/
├── __init__.py
├── plugin.py          # Protocol
├── null_plugin.py     # No-op default
├── otel_plugin.py     # OTel implementation
└── tests/
    ├── __init__.py
    └── test_plugin.py
```

### Phase 2: Session Integration (Week 2)

**Deliverables:**
- `JaatoRuntime` telemetry plugin configuration
- `JaatoSession.send_message()` root span instrumentation
- `JaatoSession._run_chat_loop()` LLM span instrumentation
- Tool execution span instrumentation

**Files to modify:**
- `shared/jaato_runtime.py` - Add `set_telemetry_plugin()`
- `shared/jaato_session.py` - Add span context managers
- `shared/ai_tool_runner.py` - Add tool span instrumentation

### Phase 3: Provider Integration (Week 3)

**Deliverables:**
- Per-provider attribute population
- Streaming token tracking
- Retry span instrumentation

**Files to modify:**
- `shared/plugins/model_provider/google_genai/provider.py`
- `shared/plugins/model_provider/anthropic/provider.py`
- `shared/plugins/model_provider/github_models/provider.py`
- `shared/utils.py` - `with_retry()` instrumentation

### Phase 4: Advanced Features (Week 4)

**Deliverables:**
- GC span instrumentation
- Permission check spans
- Subagent trace correlation
- Server event trace context propagation

**Files to modify:**
- `shared/jaato_session.py` - GC spans
- `shared/ai_tool_runner.py` - Permission spans
- `server/events.py` - Trace context in events

### Phase 5: Documentation & Polish (Week 5)

**Deliverables:**
- Configuration documentation
- Integration guides for popular backends (Langfuse, Arize Phoenix)
- Performance benchmarks
- CLAUDE.md updates

## 11. Dependencies

### Required (when enabled)

```
# requirements-telemetry.txt
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
```

### Optional Installation

```bash
# Install without telemetry (default)
pip install -r requirements.txt

# Install with telemetry
pip install -r requirements.txt -r requirements-telemetry.txt

# Or via extras
pip install -e ".[telemetry]"
```

## 12. Backend-Specific Notes

### 12.1 Langfuse

```python
telemetry.initialize({
    "endpoint": "https://cloud.langfuse.com",
    "headers": {
        "Authorization": f"Bearer {os.environ['LANGFUSE_SECRET_KEY']}"
    }
})
```

### 12.2 Arize Phoenix

```python
telemetry.initialize({
    "endpoint": "http://localhost:6006/v1/traces"
})
```

### 12.3 Helicone

```python
telemetry.initialize({
    "endpoint": "https://otel.helicone.ai",
    "headers": {
        "Helicone-Auth": f"Bearer {os.environ['HELICONE_API_KEY']}"
    }
})
```

## 13. Future Considerations

### 13.1 Metrics

Beyond traces, OTel metrics could track:
- `jaato.turns.total` - Counter of turns
- `jaato.tokens.input` - Histogram of input tokens
- `jaato.tools.duration` - Histogram of tool execution time
- `jaato.errors.total` - Counter by error type

### 13.2 Logs

OTel logs could replace/augment the file-based trace log:
- Structured logs with trace context
- Automatic correlation with spans
- Same exporter pipeline

### 13.3 Evaluation Integration

Future integration with evaluation frameworks:
- Span attributes for evaluation scores
- Custom span events for evaluation checkpoints
- Trace-based dataset collection
