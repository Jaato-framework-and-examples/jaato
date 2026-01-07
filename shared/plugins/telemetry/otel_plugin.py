"""OpenTelemetry implementation of TelemetryPlugin.

This module provides the OTelPlugin class that implements distributed
tracing using the OpenTelemetry SDK. It follows GenAI semantic conventions
for LLM operations.

Requires:
    opentelemetry-api>=1.20.0
    opentelemetry-sdk>=1.20.0
    opentelemetry-exporter-otlp>=1.20.0 (for OTLP export)
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

# Lazy imports - only loaded when plugin is initialized
_trace = None
_Status = None
_StatusCode = None


def _ensure_imports():
    """Lazily import OpenTelemetry modules."""
    global _trace, _Status, _StatusCode
    if _trace is None:
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import Status, StatusCode
        _trace = otel_trace
        _Status = Status
        _StatusCode = StatusCode


class _SpanWrapper:
    """Wrapper providing consistent interface with content redaction."""

    __slots__ = ("_span", "_redact")

    # Attributes that may contain sensitive content
    _SENSITIVE_ATTRS = frozenset({
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.request.prompt",
        "gen_ai.response.completion",
        "jaato.tool.args",
        "jaato.tool.result",
    })

    def __init__(self, span, redact_content: bool):
        self._span = span
        self._redact = redact_content

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute, redacting sensitive content if configured."""
        if self._redact and key in self._SENSITIVE_ATTRS:
            # Redact but preserve length info for debugging
            if isinstance(value, str):
                value = f"[REDACTED: {len(value)} chars]"
            elif isinstance(value, (dict, list)):
                import json
                try:
                    serialized = json.dumps(value)
                    value = f"[REDACTED: {len(serialized)} chars]"
                except (TypeError, ValueError):
                    value = "[REDACTED]"
            else:
                value = "[REDACTED]"
        self._span.set_attribute(key, value)

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        self._span.record_exception(exception)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self._span.add_event(name, attributes=attributes)

    def set_status_error(self, description: str = "") -> None:
        """Set the span status to error."""
        _ensure_imports()
        self._span.set_status(_Status(_StatusCode.ERROR, description))

    def set_status_ok(self) -> None:
        """Set the span status to OK."""
        _ensure_imports()
        self._span.set_status(_Status(_StatusCode.OK))


class OTelPlugin:
    """OpenTelemetry implementation of TelemetryPlugin.

    Provides distributed tracing using the OpenTelemetry SDK with support
    for OTLP and console exporters. Follows GenAI semantic conventions.
    """

    __slots__ = ("_enabled", "_tracer", "_redact_content", "_provider")

    def __init__(self):
        self._enabled = False
        self._tracer = None
        self._redact_content = True
        self._provider = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize OpenTelemetry with the given configuration.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default True)
                - service_name: str (default "jaato")
                - exporter: str ("otlp", "console", "none")
                - endpoint: str (OTLP endpoint URL)
                - headers: Dict[str, str] (auth headers)
                - batch_export: bool (default True)
                - sample_rate: float (0.0-1.0, default 1.0)
                - redact_content: bool (default True)
        """
        self._enabled = config.get("enabled", True)
        if not self._enabled:
            return

        _ensure_imports()

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME

        self._redact_content = config.get("redact_content", True)

        # Build resource with service name
        service_name = config.get(
            "service_name",
            os.environ.get("OTEL_SERVICE_NAME", "jaato")
        )
        resource = Resource.create({SERVICE_NAME: service_name})

        # Configure sampler if sample_rate specified
        sampler = None
        sample_rate = config.get("sample_rate", 1.0)
        if sample_rate < 1.0:
            from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
            sampler = TraceIdRatioBased(sample_rate)

        # Create tracer provider
        self._provider = TracerProvider(resource=resource, sampler=sampler)

        # Configure exporter
        exporter_type = config.get("exporter", "otlp")
        exporter = self._create_exporter(exporter_type, config)

        if exporter:
            if config.get("batch_export", True):
                processor = BatchSpanProcessor(exporter)
            else:
                processor = SimpleSpanProcessor(exporter)
            self._provider.add_span_processor(processor)

        # Set as global tracer provider
        _trace.set_tracer_provider(self._provider)

        # Get tracer for jaato
        self._tracer = _trace.get_tracer(
            "jaato",
            schema_url="https://opentelemetry.io/schemas/1.21.0"
        )

    def _create_exporter(self, exporter_type: str, config: Dict[str, Any]):
        """Create the appropriate span exporter."""
        if exporter_type == "none":
            return None

        if exporter_type == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()

        if exporter_type == "otlp":
            # Get endpoint from config or environment
            endpoint = config.get(
                "endpoint",
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            )
            if not endpoint:
                # No endpoint configured, skip OTLP export
                return None

            # Parse headers from config or environment
            headers = config.get("headers", {})
            env_headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
            if env_headers:
                for pair in env_headers.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        headers[key.strip()] = value.strip()

            # Try gRPC first, fall back to HTTP
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter as GrpcExporter
                )
                return GrpcExporter(
                    endpoint=endpoint,
                    headers=tuple(headers.items()) if headers else None,
                )
            except ImportError:
                pass

            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter as HttpExporter
                )
                return HttpExporter(
                    endpoint=endpoint,
                    headers=headers if headers else None,
                )
            except ImportError:
                pass

            # No OTLP exporter available
            return None

        # Unknown exporter type
        return None

    def shutdown(self) -> None:
        """Flush pending spans and shutdown."""
        if self._provider:
            self._provider.shutdown()
            self._provider = None
            self._tracer = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if telemetry is enabled and initialized."""
        return self._enabled and self._tracer is not None

    @contextmanager
    def turn_span(
        self,
        session_id: str,
        agent_type: str,
        agent_name: Optional[str] = None,
        turn_index: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_SpanWrapper, None, None]:
        """Create root span for a turn."""
        if not self.enabled:
            from .null_plugin import _NoOpSpan, _NOOP_SPAN
            yield _NOOP_SPAN
            return

        attrs = {
            "jaato.session_id": session_id,
            "jaato.agent_type": agent_type,
        }
        if agent_name:
            attrs["jaato.agent_name"] = agent_name
        if turn_index is not None:
            attrs["jaato.turn_index"] = turn_index
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
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_SpanWrapper, None, None]:
        """Create span for an LLM API call."""
        if not self.enabled:
            from .null_plugin import _NOOP_SPAN
            yield _NOOP_SPAN
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
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_SpanWrapper, None, None]:
        """Create span for tool execution."""
        if not self.enabled:
            from .null_plugin import _NOOP_SPAN
            yield _NOOP_SPAN
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
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_SpanWrapper, None, None]:
        """Create span for a retry attempt."""
        if not self.enabled:
            from .null_plugin import _NOOP_SPAN
            yield _NOOP_SPAN
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
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_SpanWrapper, None, None]:
        """Create span for GC operation."""
        if not self.enabled:
            from .null_plugin import _NOOP_SPAN
            yield _NOOP_SPAN
            return

        attrs = {
            "jaato.gc.trigger_reason": trigger_reason,
            "jaato.gc.strategy": strategy,
        }
        if attributes:
            attrs.update(attributes)

        with self._tracer.start_as_current_span("jaato.gc", attributes=attrs) as span:
            yield _SpanWrapper(span, self._redact_content)

    @contextmanager
    def permission_span(
        self,
        tool_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_SpanWrapper, None, None]:
        """Create span for permission check."""
        if not self.enabled:
            from .null_plugin import _NOOP_SPAN
            yield _NOOP_SPAN
            return

        attrs = {
            "jaato.permission.tool_name": tool_name,
        }
        if attributes:
            attrs.update(attributes)

        with self._tracer.start_as_current_span("jaato.permission", attributes=attrs) as span:
            yield _SpanWrapper(span, self._redact_content)

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID if available."""
        if not self.enabled:
            return None

        _ensure_imports()
        span = _trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, '032x')
        return None

    def get_current_span_id(self) -> Optional[str]:
        """Get the current span ID if available."""
        if not self.enabled:
            return None

        _ensure_imports()
        span = _trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, '016x')
        return None
