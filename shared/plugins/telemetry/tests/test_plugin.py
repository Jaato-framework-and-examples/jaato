"""Tests for telemetry plugin."""

import pytest
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


class TestNullTelemetryPlugin:
    """Tests for NullTelemetryPlugin (no-op implementation)."""

    def test_null_plugin_initialize_is_noop(self):
        """Verify initialize does nothing."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        plugin.initialize({"enabled": True, "exporter": "otlp"})
        assert not plugin.enabled

    def test_null_plugin_enabled_always_false(self):
        """Verify enabled is always False."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        assert not plugin.enabled
        plugin.initialize({})
        assert not plugin.enabled

    def test_null_plugin_turn_span_yields_noop(self):
        """Verify turn_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.turn_span("sess_1", "main") as span:
            # These should all be no-ops
            span.set_attribute("key", "value")
            span.record_exception(ValueError("test"))
            span.add_event("event", {"attr": "val"})
            span.set_status_error("error")
            span.set_status_ok()

    def test_null_plugin_llm_span_yields_noop(self):
        """Verify llm_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.llm_span("model", "provider", streaming=True) as span:
            span.set_attribute("gen_ai.usage.input_tokens", 100)

    def test_null_plugin_tool_span_yields_noop(self):
        """Verify tool_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.tool_span("tool_name", "call_123", "cli") as span:
            span.set_attribute("jaato.tool.success", True)

    def test_null_plugin_retry_span_yields_noop(self):
        """Verify retry_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.retry_span(1, 5, "api_call") as span:
            span.set_attribute("jaato.retry.delay_seconds", 2.5)

    def test_null_plugin_gc_span_yields_noop(self):
        """Verify gc_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.gc_span("threshold", "truncate") as span:
            span.set_attribute("jaato.gc.items_collected", 10)

    def test_null_plugin_permission_span_yields_noop(self):
        """Verify permission_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.permission_span("cli_tool") as span:
            span.set_attribute("jaato.permission.decision", "allowed")

    def test_null_plugin_trace_ids_return_none(self):
        """Verify trace/span IDs return None."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        assert plugin.get_current_trace_id() is None
        assert plugin.get_current_span_id() is None

    def test_null_plugin_shutdown_is_noop(self):
        """Verify shutdown does nothing."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        plugin.shutdown()  # Should not raise

    def test_null_plugin_nested_spans(self):
        """Verify nested spans work correctly."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.turn_span("sess_1", "main") as turn:
            turn.set_attribute("turn", True)
            with plugin.llm_span("model", "provider") as llm:
                llm.set_attribute("llm", True)
                with plugin.tool_span("tool", "call_1") as tool:
                    tool.set_attribute("tool", True)


class TestCreatePlugin:
    """Tests for create_plugin factory function."""

    def test_create_plugin_returns_null_when_disabled(self):
        """Verify create_plugin returns NullTelemetryPlugin when disabled."""
        from shared.plugins.telemetry import create_plugin
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        with patch.dict("os.environ", {"JAATO_TELEMETRY_ENABLED": "false"}):
            plugin = create_plugin()
            assert isinstance(plugin, NullTelemetryPlugin)

    def test_create_plugin_returns_null_when_env_not_set(self):
        """Verify create_plugin returns NullTelemetryPlugin when env not set."""
        from shared.plugins.telemetry import create_plugin
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        with patch.dict("os.environ", {}, clear=True):
            # Remove JAATO_TELEMETRY_ENABLED if present
            import os
            os.environ.pop("JAATO_TELEMETRY_ENABLED", None)
            plugin = create_plugin()
            assert isinstance(plugin, NullTelemetryPlugin)


class TestTelemetryProtocol:
    """Tests for TelemetryPlugin protocol compliance."""

    def test_null_plugin_implements_protocol(self):
        """Verify NullTelemetryPlugin implements TelemetryPlugin protocol."""
        from shared.plugins.telemetry.plugin import TelemetryPlugin
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        assert isinstance(plugin, TelemetryPlugin)

    def test_protocol_has_required_methods(self):
        """Verify protocol defines all required methods."""
        from shared.plugins.telemetry.plugin import TelemetryPlugin
        import inspect

        # Get all abstract methods from the protocol
        methods = [
            "initialize",
            "shutdown",
            "enabled",
            "turn_span",
            "llm_span",
            "tool_span",
            "retry_span",
            "gc_span",
            "permission_span",
            "get_current_trace_id",
            "get_current_span_id",
        ]

        for method in methods:
            assert hasattr(TelemetryPlugin, method), f"Protocol missing {method}"


class TestSpanContext:
    """Tests for SpanContext protocol."""

    def test_noop_span_has_required_methods(self):
        """Verify _NoOpSpan has all required methods."""
        from shared.plugins.telemetry.null_plugin import _NoOpSpan

        span = _NoOpSpan()
        assert hasattr(span, "set_attribute")
        assert hasattr(span, "record_exception")
        assert hasattr(span, "add_event")
        assert hasattr(span, "set_status_error")
        assert hasattr(span, "set_status_ok")


# Conditional tests that require opentelemetry
try:
    import opentelemetry
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestOTelPlugin:
    """Tests for OTelPlugin (requires opentelemetry packages)."""

    def test_otel_plugin_initialize_with_console_exporter(self):
        """Test initialization with console exporter."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin

        plugin = OTelPlugin()
        plugin.initialize({
            "enabled": True,
            "exporter": "console",
            "batch_export": False,
        })
        assert plugin.enabled

        plugin.shutdown()

    def test_otel_plugin_initialize_disabled(self):
        """Test initialization when disabled."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin

        plugin = OTelPlugin()
        plugin.initialize({"enabled": False})
        assert not plugin.enabled

    def test_otel_plugin_turn_span_creates_span(self):
        """Test turn_span creates an actual span."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        # Set up in-memory exporter for testing
        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({
            "enabled": True,
            "exporter": "none",  # We'll add processor manually
        })

        # Add in-memory processor
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.turn_span("sess_123", "main", agent_name="test") as span:
            span.set_attribute("custom_attr", "value")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "jaato.turn"
        assert spans[0].attributes["jaato.session_id"] == "sess_123"
        assert spans[0].attributes["jaato.agent_type"] == "main"
        assert spans[0].attributes["jaato.agent_name"] == "test"
        assert spans[0].attributes["custom_attr"] == "value"

        plugin.shutdown()

    def test_otel_plugin_nested_spans_have_correct_parent(self):
        """Test nested spans have correct parent-child relationships."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({"enabled": True, "exporter": "none"})
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.turn_span("sess_1", "main") as turn:
            with plugin.llm_span("claude-sonnet", "anthropic") as llm:
                with plugin.tool_span("calculator", "call_1") as tool:
                    pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        # Spans are finished in reverse order (innermost first)
        tool_span = spans[0]
        llm_span = spans[1]
        turn_span = spans[2]

        assert tool_span.name == "jaato.tool"
        assert llm_span.name == "gen_ai.chat"
        assert turn_span.name == "jaato.turn"

        # Check parent-child relationships
        assert tool_span.parent.span_id == llm_span.context.span_id
        assert llm_span.parent.span_id == turn_span.context.span_id
        assert turn_span.parent is None

        plugin.shutdown()

    def test_otel_plugin_llm_span_attributes(self):
        """Test llm_span sets correct attributes."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({"enabled": True, "exporter": "none"})
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.llm_span("gemini-2.5-flash", "google_genai", streaming=True) as span:
            span.set_attribute("gen_ai.usage.input_tokens", 500)
            span.set_attribute("gen_ai.usage.output_tokens", 150)
            span.set_attribute("gen_ai.response.finish_reasons", ["stop"])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["gen_ai.system"] == "google_genai"
        assert spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash"
        assert spans[0].attributes["jaato.streaming"] is True
        assert spans[0].attributes["gen_ai.usage.input_tokens"] == 500

        plugin.shutdown()

    def test_otel_plugin_redacts_sensitive_content(self):
        """Test that sensitive content is redacted by default."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({
            "enabled": True,
            "exporter": "none",
            "redact_content": True,
        })
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.llm_span("model", "provider") as span:
            span.set_attribute("gen_ai.prompt", "This is a secret prompt")
            span.set_attribute("gen_ai.completion", "This is a secret response")
            span.set_attribute("gen_ai.usage.input_tokens", 100)  # Not sensitive

        spans = exporter.get_finished_spans()
        assert "[REDACTED:" in spans[0].attributes["gen_ai.prompt"]
        assert "[REDACTED:" in spans[0].attributes["gen_ai.completion"]
        assert spans[0].attributes["gen_ai.usage.input_tokens"] == 100

        plugin.shutdown()

    def test_otel_plugin_no_redaction_when_disabled(self):
        """Test that content is not redacted when redact_content=False."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({
            "enabled": True,
            "exporter": "none",
            "redact_content": False,
        })
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.llm_span("model", "provider") as span:
            span.set_attribute("gen_ai.prompt", "This is a secret prompt")

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["gen_ai.prompt"] == "This is a secret prompt"

        plugin.shutdown()

    def test_otel_plugin_record_exception(self):
        """Test recording exceptions on spans."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({"enabled": True, "exporter": "none"})
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.tool_span("failing_tool", "call_1") as span:
            try:
                raise ValueError("Tool execution failed")
            except ValueError as e:
                span.record_exception(e)
                span.set_status_error("Tool failed")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert len(spans[0].events) == 1
        assert spans[0].events[0].name == "exception"

        plugin.shutdown()

    def test_otel_plugin_get_trace_id(self):
        """Test getting current trace ID."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({"enabled": True, "exporter": "none"})
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        # No trace outside of span
        assert plugin.get_current_trace_id() is None

        with plugin.turn_span("sess_1", "main"):
            trace_id = plugin.get_current_trace_id()
            span_id = plugin.get_current_span_id()

            assert trace_id is not None
            assert len(trace_id) == 32  # 128 bits in hex
            assert span_id is not None
            assert len(span_id) == 16  # 64 bits in hex

        plugin.shutdown()

    def test_otel_plugin_implements_protocol(self):
        """Verify OTelPlugin implements TelemetryPlugin protocol."""
        from shared.plugins.telemetry.plugin import TelemetryPlugin
        from shared.plugins.telemetry.otel_plugin import OTelPlugin

        plugin = OTelPlugin()
        assert isinstance(plugin, TelemetryPlugin)

    def test_otel_plugin_retry_span(self):
        """Test retry_span attributes."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({"enabled": True, "exporter": "none"})
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.retry_span(2, 5, "api_call") as span:
            span.set_attribute("jaato.retry.delay_seconds", 4.5)
            span.set_attribute("jaato.retry.error_type", "rate_limit")

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["jaato.retry.attempt"] == 2
        assert spans[0].attributes["jaato.retry.max_attempts"] == 5
        assert spans[0].attributes["jaato.retry.delay_seconds"] == 4.5

        plugin.shutdown()

    def test_otel_plugin_gc_span(self):
        """Test gc_span attributes."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({"enabled": True, "exporter": "none"})
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.gc_span("threshold", "truncate") as span:
            span.set_attribute("jaato.gc.items_collected", 15)
            span.set_attribute("jaato.gc.tokens_freed", 8500)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["jaato.gc.trigger_reason"] == "threshold"
        assert spans[0].attributes["jaato.gc.strategy"] == "truncate"
        assert spans[0].attributes["jaato.gc.items_collected"] == 15

        plugin.shutdown()

    def test_otel_plugin_add_event(self):
        """Test adding events to spans."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()

        plugin = OTelPlugin()
        plugin.initialize({"enabled": True, "exporter": "none"})
        plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))

        with plugin.turn_span("sess_1", "main") as span:
            span.add_event("checkpoint", {"phase": "start"})
            span.add_event("checkpoint", {"phase": "end"})

        spans = exporter.get_finished_spans()
        events = [e for e in spans[0].events if e.name == "checkpoint"]
        assert len(events) == 2

        plugin.shutdown()
