"""Tests for telemetry plugin with OpenInference semantic conventions."""

import json
import pytest
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
        """Verify turn_span yields no-op span with all methods."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.turn_span("sess_1", "main") as span:
            # These should all be no-ops
            span.set_attribute("key", "value")
            span.record_exception(ValueError("test"))
            span.add_event("event", {"attr": "val"})
            span.set_status_error("error")
            span.set_status_ok()
            span.set_input_messages([{"role": "user", "content": "hi"}])
            span.set_output_messages([{"role": "assistant", "content": "hello"}])
            span.set_metadata({"key": "value"})

    def test_null_plugin_llm_span_yields_noop(self):
        """Verify llm_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.llm_span("model", "provider", streaming=True) as span:
            span.set_attribute("llm.token_count.prompt", 100)

    def test_null_plugin_tool_span_yields_noop(self):
        """Verify tool_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.tool_span("tool_name", "call_123", "cli") as span:
            span.set_attribute("input.value", '{"cmd": "ls"}')

    def test_null_plugin_retry_span_yields_noop(self):
        """Verify retry_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.retry_span(1, 5, "api_call") as span:
            span.set_metadata({"delay_seconds": 2.5})

    def test_null_plugin_gc_span_yields_noop(self):
        """Verify gc_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.gc_span("threshold", "truncate") as span:
            span.set_metadata({"items_collected": 10})

    def test_null_plugin_permission_span_yields_noop(self):
        """Verify permission_span yields no-op span."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.permission_span("cli_tool") as span:
            span.set_metadata({"decision": "allowed"})

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

    def test_null_plugin_turn_span_accepts_parent_session_id(self):
        """Verify turn_span accepts parent_session_id parameter."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        with plugin.turn_span("sess_1", "subagent", parent_session_id="parent_sess") as span:
            span.set_attribute("key", "value")


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
        assert hasattr(span, "set_input_messages")
        assert hasattr(span, "set_output_messages")
        assert hasattr(span, "set_metadata")


# Conditional tests that require opentelemetry
try:
    import opentelemetry
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def _create_test_plugin(redact_content=False):
    """Helper to create an OTelPlugin with in-memory exporter for testing.

    Args:
        redact_content: Whether to redact sensitive content (default False
            for test visibility).
    """
    from shared.plugins.telemetry.otel_plugin import OTelPlugin
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    plugin = OTelPlugin()
    plugin.initialize({
        "enabled": True,
        "exporter": "none",
        "redact_content": redact_content,
    })
    plugin._provider.add_span_processor(SimpleSpanProcessor(exporter))
    return plugin, exporter


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestOTelPlugin:
    """Tests for OTelPlugin with OpenInference semantic conventions."""

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

    def test_turn_span_has_agent_span_kind(self):
        """Test turn_span sets openinference.span.kind = AGENT."""
        plugin, exporter = _create_test_plugin()

        with plugin.turn_span("sess_123", "main", agent_name="test") as span:
            span.set_attribute("custom_attr", "value")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "jaato.test.turn"
        assert spans[0].attributes["openinference.span.kind"] == "AGENT"
        assert spans[0].attributes["session.id"] == "sess_123"
        assert spans[0].attributes["agent.name"] == "test"
        assert spans[0].attributes["custom_attr"] == "value"

        plugin.shutdown()

    def test_turn_span_graph_attributes(self):
        """Test turn_span sets graph.node.* for DAG visualization."""
        plugin, exporter = _create_test_plugin()

        with plugin.turn_span("sess_123", "main", agent_name="planner") as span:
            pass

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["graph.node.id"] == "sess_123"
        assert attrs["graph.node.name"] == "planner"
        assert attrs["graph.node.parent_id"] == ""

        plugin.shutdown()

    def test_turn_span_subagent_graph_parent(self):
        """Test subagent turn_span has graph.node.parent_id pointing to parent."""
        plugin, exporter = _create_test_plugin()

        with plugin.turn_span(
            "child_sess", "subagent",
            agent_name="researcher",
            parent_session_id="parent_sess",
        ) as span:
            pass

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["graph.node.parent_id"] == "parent_sess"
        assert attrs["graph.node.id"] == "child_sess"

        plugin.shutdown()

    def test_turn_span_metadata(self):
        """Test turn_span packs jaato-specific fields into metadata."""
        plugin, exporter = _create_test_plugin()

        with plugin.turn_span("sess_1", "main", turn_index=5) as span:
            pass

        spans = exporter.get_finished_spans()
        metadata = json.loads(spans[0].attributes["metadata"])
        assert metadata["agent_type"] == "main"
        assert metadata["turn_index"] == 5

        plugin.shutdown()

    def test_llm_span_has_llm_span_kind(self):
        """Test llm_span sets openinference.span.kind = LLM."""
        plugin, exporter = _create_test_plugin()

        with plugin.llm_span("gemini-2.5-flash", "google_genai", streaming=True) as span:
            span.set_attribute("llm.token_count.prompt", 500)
            span.set_attribute("llm.token_count.completion", 150)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["openinference.span.kind"] == "LLM"
        assert spans[0].attributes["llm.system"] == "google_genai"
        assert spans[0].attributes["llm.model_name"] == "gemini-2.5-flash"
        assert spans[0].attributes["llm.token_count.prompt"] == 500
        assert spans[0].attributes["llm.token_count.completion"] == 150

        plugin.shutdown()

    def test_llm_span_auto_computes_total_tokens(self):
        """Test that setting prompt + completion auto-computes total."""
        plugin, exporter = _create_test_plugin()

        with plugin.llm_span("model", "provider") as span:
            span.set_attribute("llm.token_count.prompt", 1000)
            span.set_attribute("llm.token_count.completion", 200)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["llm.token_count.total"] == 1200

        plugin.shutdown()

    def test_llm_span_auto_computes_total_reverse_order(self):
        """Test total is computed even when completion is set before prompt."""
        plugin, exporter = _create_test_plugin()

        with plugin.llm_span("model", "provider") as span:
            span.set_attribute("llm.token_count.completion", 300)
            span.set_attribute("llm.token_count.prompt", 700)

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["llm.token_count.total"] == 1000

        plugin.shutdown()

    def test_tool_span_has_tool_span_kind(self):
        """Test tool_span sets openinference.span.kind = TOOL."""
        plugin, exporter = _create_test_plugin()

        with plugin.tool_span("cli", "call_123", "cli") as span:
            span.set_attribute("input.value", '{"command": "ls"}')
            span.set_attribute("output.value", '{"files": ["a.txt"]}')

        spans = exporter.get_finished_spans()
        assert spans[0].name == "jaato.tool.cli"
        assert spans[0].attributes["openinference.span.kind"] == "TOOL"
        assert spans[0].attributes["tool.name"] == "cli"
        assert spans[0].attributes["tool.id"] == "call_123"

        plugin.shutdown()

    def test_retry_span_has_chain_span_kind(self):
        """Test retry_span sets openinference.span.kind = CHAIN."""
        plugin, exporter = _create_test_plugin()

        with plugin.retry_span(2, 5, "api_call") as span:
            pass

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["openinference.span.kind"] == "CHAIN"
        metadata = json.loads(spans[0].attributes["metadata"])
        assert metadata["retry_attempt"] == 2
        assert metadata["retry_max_attempts"] == 5

        plugin.shutdown()

    def test_gc_span_has_chain_span_kind(self):
        """Test gc_span sets openinference.span.kind = CHAIN."""
        plugin, exporter = _create_test_plugin()

        with plugin.gc_span("threshold", "truncate") as span:
            pass

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["openinference.span.kind"] == "CHAIN"
        metadata = json.loads(spans[0].attributes["metadata"])
        assert metadata["gc_trigger_reason"] == "threshold"
        assert metadata["gc_strategy"] == "truncate"

        plugin.shutdown()

    def test_permission_span_has_chain_span_kind(self):
        """Test permission_span sets openinference.span.kind = CHAIN."""
        plugin, exporter = _create_test_plugin()

        with plugin.permission_span("cli") as span:
            pass

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["openinference.span.kind"] == "CHAIN"
        metadata = json.loads(spans[0].attributes["metadata"])
        assert metadata["permission_tool_name"] == "cli"

        plugin.shutdown()

    def test_nested_spans_have_correct_parent(self):
        """Test nested spans have correct parent-child relationships."""
        plugin, exporter = _create_test_plugin()

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

        assert tool_span.name.startswith("jaato.tool.")
        assert llm_span.name == "jaato.sess_1.llm"
        assert turn_span.name == "jaato.unknown.turn"

        # Check parent-child relationships
        assert tool_span.parent.span_id == llm_span.context.span_id
        assert llm_span.parent.span_id == turn_span.context.span_id
        assert turn_span.parent is None

        # All spans have correct span kinds
        assert turn_span.attributes["openinference.span.kind"] == "AGENT"
        assert llm_span.attributes["openinference.span.kind"] == "LLM"
        assert tool_span.attributes["openinference.span.kind"] == "TOOL"

        plugin.shutdown()

    def test_agent_context_cleared_after_turn_span(self):
        """Test that agent context is cleaned up after turn_span exits."""
        plugin, exporter = _create_test_plugin()

        with plugin.turn_span("agent_1", "main", agent_name="planner"):
            pass

        # After turn_span exits, context should be cleared
        assert getattr(plugin._agent_context, "agent_id", None) is None
        assert getattr(plugin._agent_context, "agent_name", None) is None

        # A child span created outside a turn_span should not carry stale context
        with plugin.llm_span("model", "provider") as llm:
            pass

        spans = exporter.get_finished_spans()
        llm_span = spans[-1]
        assert "session.id" not in dict(llm_span.attributes)
        assert "agent.name" not in dict(llm_span.attributes)

        plugin.shutdown()

    def test_plan_step_context_from_bus_events(self):
        """Test that bus STEP_STARTED/STEP_COMPLETED events set plan/step in metadata."""
        from shared.event_bus import EventBus
        from jaato_sdk.event_bus import EventType as BusEventType, Event as BusEvent

        plugin, exporter = _create_test_plugin()
        bus = EventBus()
        plugin.subscribe_to_bus(bus)

        # Simulate STEP_STARTED event
        bus.publish(BusEvent(
            event_id="evt1",
            event_type=BusEventType.STEP_STARTED,
            timestamp="2026-01-01T00:00:00Z",
            source_agent="main",
            payload={"plan_id": "plan_42", "step_id": "step_7"},
        ))

        # Spans created after STEP_STARTED should carry plan/step in metadata
        with plugin.turn_span("sess_1", "main") as turn:
            with plugin.tool_span("web_search", "call_1") as tool:
                pass

        spans = exporter.get_finished_spans()
        for span in spans:
            metadata = json.loads(span.attributes["metadata"])
            assert metadata["plan_id"] == "plan_42"
            assert metadata["step_id"] == "step_7"

        # Simulate STEP_COMPLETED — should clear context
        exporter.clear()
        bus.publish(BusEvent(
            event_id="evt2",
            event_type=BusEventType.STEP_COMPLETED,
            timestamp="2026-01-01T00:01:00Z",
            source_agent="main",
            payload={"plan_id": "plan_42", "step_id": "step_7"},
        ))

        with plugin.llm_span("model", "provider") as llm:
            pass

        spans = exporter.get_finished_spans()
        metadata = json.loads(spans[0].attributes["metadata"])
        assert "plan_id" not in metadata
        assert "step_id" not in metadata

        plugin.shutdown()

    def test_redacts_sensitive_content(self):
        """Test that input.value and output.value are redacted by default."""
        plugin, exporter = _create_test_plugin()

        # Reinitialize with explicit redaction
        plugin.shutdown()
        plugin = None
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

        with plugin.tool_span("cli", "call_1") as span:
            span.set_attribute("input.value", '{"command": "secret"}')
            span.set_attribute("output.value", '{"result": "classified"}')
            span.set_attribute("tool.name", "cli")  # Not sensitive

        spans = exporter.get_finished_spans()
        assert "[REDACTED:" in spans[0].attributes["input.value"]
        assert "[REDACTED:" in spans[0].attributes["output.value"]
        assert spans[0].attributes["tool.name"] == "cli"

        plugin.shutdown()

    def test_no_redaction_when_disabled(self):
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

        with plugin.tool_span("cli", "call_1") as span:
            span.set_attribute("input.value", '{"command": "ls"}')

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["input.value"] == '{"command": "ls"}'

        plugin.shutdown()

    def test_record_exception(self):
        """Test recording exceptions on spans."""
        plugin, exporter = _create_test_plugin()

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

    def test_get_trace_id(self):
        """Test getting current trace ID."""
        plugin, exporter = _create_test_plugin()

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

    def test_llm_span_name_uses_session_id(self):
        """Test llm_span uses session ID in span name when inside a turn."""
        plugin, exporter = _create_test_plugin()

        with plugin.turn_span("sess_42", "main") as turn:
            with plugin.llm_span("model", "provider") as span:
                pass

        spans = exporter.get_finished_spans()
        llm_span = [s for s in spans if ".llm" in s.name][0]
        assert llm_span.name == "jaato.sess_42.llm"

    def test_llm_span_name_fallback_without_session(self):
        """Test llm_span falls back to 'jaato.llm' without session context."""
        plugin, exporter = _create_test_plugin()

        with plugin.llm_span("model", "provider") as span:
            pass

        spans = exporter.get_finished_spans()
        assert spans[0].name == "jaato.llm"

        plugin.shutdown()

    def test_capture_and_attach_context_for_parallel_spans(self):
        """Test that captured context propagates parent to worker threads."""
        from concurrent.futures import ThreadPoolExecutor

        plugin, exporter = _create_test_plugin()

        with plugin.turn_span("sess_1", "main") as turn:
            # Capture context from the parent thread
            ctx = plugin.capture_context()

            def worker():
                with plugin.attach_context(ctx):
                    with plugin.tool_span("parallel_tool", "call_1") as tool:
                        tool.set_attribute("input.value", "{}")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(worker)
                future.result()

        spans = exporter.get_finished_spans()
        # Tool span should be a child of the turn span
        tool_span = [s for s in spans if s.name.startswith("jaato.tool.")][0]
        turn_span = [s for s in spans if "turn" in s.name][0]
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == turn_span.context.span_id

        plugin.shutdown()

    def test_parallel_spans_orphaned_without_context_propagation(self):
        """Test that without attach_context, parallel spans are orphaned."""
        from concurrent.futures import ThreadPoolExecutor

        plugin, exporter = _create_test_plugin()

        with plugin.turn_span("sess_1", "main") as turn:
            def worker():
                # No attach_context — span should be orphaned
                with plugin.tool_span("orphan_tool", "call_1") as tool:
                    tool.set_attribute("input.value", "{}")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(worker)
                future.result()

        spans = exporter.get_finished_spans()
        tool_span = [s for s in spans if s.name.startswith("jaato.tool.")][0]
        turn_span = [s for s in spans if "turn" in s.name][0]
        # Without context propagation, tool span has no parent link to turn
        if tool_span.parent is not None:
            assert tool_span.parent.span_id != turn_span.context.span_id

        plugin.shutdown()

    def test_null_plugin_capture_and_attach_context(self):
        """Test that null plugin capture/attach are no-ops."""
        from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin

        plugin = NullTelemetryPlugin()
        ctx = plugin.capture_context()
        assert ctx is None

        with plugin.attach_context(ctx):
            with plugin.tool_span("tool", "call_1") as span:
                span.set_attribute("key", "value")

    def test_add_event(self):
        """Test adding events to spans."""
        plugin, exporter = _create_test_plugin()

        with plugin.turn_span("sess_1", "main") as span:
            span.add_event("checkpoint", {"phase": "start"})
            span.add_event("checkpoint", {"phase": "end"})

        spans = exporter.get_finished_spans()
        events = [e for e in spans[0].events if e.name == "checkpoint"]
        assert len(events) == 2

        plugin.shutdown()


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestOpenInferenceMessages:
    """Tests for OpenInference message flattening on LLM spans."""

    def test_set_input_messages(self):
        """Test input messages are flattened to indexed attributes."""
        plugin, exporter = _create_test_plugin()

        with plugin.llm_span("model", "provider") as span:
            span.set_input_messages([
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ])

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["llm.input_messages.0.message.role"] == "system"
        assert attrs["llm.input_messages.0.message.content"] == "You are helpful."
        assert attrs["llm.input_messages.1.message.role"] == "user"
        assert attrs["llm.input_messages.1.message.content"] == "Hello"

        plugin.shutdown()

    def test_set_output_messages_with_tool_calls(self):
        """Test output messages with tool calls are flattened correctly."""
        plugin, exporter = _create_test_plugin()

        with plugin.llm_span("model", "provider") as span:
            span.set_output_messages([{
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "cli", "arguments": '{"command": "ls"}'},
                    {"name": "web_search", "arguments": '{"query": "test"}'},
                ],
            }])

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["llm.output_messages.0.message.role"] == "assistant"
        assert attrs["llm.output_messages.0.message.content"] == ""
        assert attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "cli"
        assert attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"] == '{"command": "ls"}'
        assert attrs["llm.output_messages.0.message.tool_calls.1.tool_call.function.name"] == "web_search"

        plugin.shutdown()

    def test_messages_redacted_when_enabled(self):
        """Test message content is redacted when redact_content=True."""
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
            span.set_input_messages([
                {"role": "user", "content": "secret prompt"},
            ])
            span.set_output_messages([{
                "role": "assistant",
                "content": "secret response",
                "tool_calls": [
                    {"name": "cli", "arguments": '{"secret": "data"}'},
                ],
            }])

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)

        # Content should be redacted
        assert "[REDACTED:" in attrs["llm.input_messages.0.message.content"]
        assert "[REDACTED:" in attrs["llm.output_messages.0.message.content"]
        assert "[REDACTED:" in attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]

        # Roles and tool names should NOT be redacted
        assert attrs["llm.input_messages.0.message.role"] == "user"
        assert attrs["llm.output_messages.0.message.role"] == "assistant"
        assert attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "cli"

        plugin.shutdown()

    def test_messages_not_redacted_when_disabled(self):
        """Test message content is preserved when redact_content=False."""
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
            span.set_input_messages([
                {"role": "user", "content": "visible prompt"},
            ])

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["llm.input_messages.0.message.content"] == "visible prompt"

        plugin.shutdown()

    def test_set_metadata(self):
        """Test set_metadata writes JSON string to metadata attribute."""
        plugin, exporter = _create_test_plugin()

        with plugin.tool_span("cli", "call_1") as span:
            span.set_metadata({"duration_seconds": 1.5, "parallel": True})

        spans = exporter.get_finished_spans()
        # The tool_span already sets metadata via the constructor,
        # but set_metadata overwrites it
        metadata = json.loads(spans[0].attributes["metadata"])
        assert metadata["duration_seconds"] == 1.5
        assert metadata["parallel"] is True

        plugin.shutdown()

    def test_telemetry_resource_entry_points_merged(self):
        """Test that jaato.telemetry_resource entry points are discovered
        and merged into the OTel Resource during initialize().

        This is the extension point that allows jaato-premium (or any
        external package) to contribute server identity attributes like
        service.instance.id and host.name to the OTel Resource.
        """
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from unittest.mock import MagicMock

        # Simulate an entry point that returns server identity attrs
        mock_ep = MagicMock()
        mock_ep.name = "gossip_identity"
        mock_ep.load.return_value = lambda: {
            "service.name": "server-a",
            "service.instance.id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "service.version": "0.2.59",
            "service.namespace": "production",
            "host.name": "192.168.1.42",
        }

        with patch(
            "shared.plugins.telemetry.otel_plugin.entry_points",
            return_value=[mock_ep],
        ):
            plugin = OTelPlugin()
            plugin.initialize({
                "enabled": True,
                "exporter": "none",
            })

        # Verify the resource has the contributed attributes
        resource_attrs = dict(plugin._provider.resource.attributes)
        assert resource_attrs["service.name"] == "server-a"
        assert resource_attrs["service.instance.id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert resource_attrs["service.version"] == "0.2.59"
        assert resource_attrs["service.namespace"] == "production"
        assert resource_attrs["host.name"] == "192.168.1.42"

        plugin.shutdown()

    def test_telemetry_resource_entry_point_failure_logged(self):
        """Test that a failing entry point is logged and does not prevent
        initialization from completing."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from unittest.mock import MagicMock

        mock_ep = MagicMock()
        mock_ep.name = "broken_provider"
        mock_ep.load.return_value = MagicMock(side_effect=RuntimeError("boom"))

        with patch(
            "shared.plugins.telemetry.otel_plugin.entry_points",
            return_value=[mock_ep],
        ):
            plugin = OTelPlugin()
            plugin.initialize({
                "enabled": True,
                "exporter": "none",
            })

        # Plugin should still be enabled despite the failed entry point
        assert plugin.enabled

        # Resource falls back to default service.name
        resource_attrs = dict(plugin._provider.resource.attributes)
        assert resource_attrs["service.name"] == "jaato"

        plugin.shutdown()

    def test_telemetry_resource_no_entry_points(self):
        """Test that when no entry points are registered, the Resource
        uses the default service.name (single-server mode)."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin

        with patch(
            "shared.plugins.telemetry.otel_plugin.entry_points",
            return_value=[],
        ):
            plugin = OTelPlugin()
            plugin.initialize({
                "enabled": True,
                "exporter": "none",
            })

        resource_attrs = dict(plugin._provider.resource.attributes)
        assert resource_attrs["service.name"] == "jaato"

        plugin.shutdown()

    def test_telemetry_resource_non_dict_ignored(self):
        """Test that entry points returning non-dict values are ignored."""
        from shared.plugins.telemetry.otel_plugin import OTelPlugin
        from unittest.mock import MagicMock

        mock_ep = MagicMock()
        mock_ep.name = "bad_provider"
        mock_ep.load.return_value = lambda: "not a dict"

        with patch(
            "shared.plugins.telemetry.otel_plugin.entry_points",
            return_value=[mock_ep],
        ):
            plugin = OTelPlugin()
            plugin.initialize({
                "enabled": True,
                "exporter": "none",
            })

        # Should still initialize with defaults
        resource_attrs = dict(plugin._provider.resource.attributes)
        assert resource_attrs["service.name"] == "jaato"

        plugin.shutdown()


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestOTLPProtocolSelection:
    """The OTLP exporter honors a protocol preference.

    Default is gRPC-first (backward compatible). An explicit ``http/protobuf``
    preference (config key or OTEL_EXPORTER_OTLP_PROTOCOL) selects the HTTP
    exporter, which is required for HTTP-only backends like Langfuse.

    Both exporter packages are installed in the telemetry test env, so
    selection is deterministic (no ImportError fallback masks the choice).
    """

    _GRPC_MOD = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    _HTTP_MOD = "opentelemetry.exporter.otlp.proto.http.trace_exporter"

    def _exporter(self, config, monkeypatch, env=None):
        from shared.plugins.telemetry.otel_plugin import OTelPlugin

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
        if env is not None:
            monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", env)
        return OTelPlugin()._create_exporter("otlp", config)

    def test_default_is_grpc(self, monkeypatch):
        exp = self._exporter({"endpoint": "http://localhost:4317"}, monkeypatch)
        assert type(exp).__module__ == self._GRPC_MOD

    def test_config_protocol_http_selects_http(self, monkeypatch):
        exp = self._exporter(
            {"endpoint": "https://cloud.langfuse.com/api/public/otel",
             "protocol": "http/protobuf"},
            monkeypatch,
        )
        assert type(exp).__module__ == self._HTTP_MOD

    def test_env_protocol_http_selects_http(self, monkeypatch):
        exp = self._exporter(
            {"endpoint": "https://cloud.langfuse.com/api/public/otel"},
            monkeypatch,
            env="http/protobuf",
        )
        assert type(exp).__module__ == self._HTTP_MOD

    def test_config_protocol_overrides_env(self, monkeypatch):
        # Explicit config key wins over the environment variable.
        exp = self._exporter(
            {"endpoint": "http://localhost:4317", "protocol": "grpc"},
            monkeypatch,
            env="http/protobuf",
        )
        assert type(exp).__module__ == self._GRPC_MOD


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestSessionIdPropagation:
    """session.id must appear on child llm/tool spans, not just the turn root.

    Langfuse (and other OTLP backends) filter/aggregate per observation, so a
    trace-level attribute like session.id has to be present on every span, not
    only the AGENT root. Langfuse's ingestion reads the OpenInference
    ``session.id`` key directly.
    """

    def _spans_by_kind(self, exporter):
        return {
            s.attributes.get("openinference.span.kind"): s
            for s in exporter.get_finished_spans()
        }

    def test_child_spans_inherit_turn_session_id(self):
        plugin, exporter = _create_test_plugin()
        with plugin.turn_span(session_id="sess-xyz", agent_type="main",
                              agent_name="main"):
            with plugin.llm_span("m", "p"):
                pass
            with plugin.tool_span("cli", "call-1"):
                pass
        plugin.shutdown()

        spans = self._spans_by_kind(exporter)
        assert spans["AGENT"].attributes["session.id"] == "sess-xyz"
        assert spans["LLM"].attributes["session.id"] == "sess-xyz"
        assert spans["TOOL"].attributes["session.id"] == "sess-xyz"

    def test_llm_span_without_turn_context_omits_session_id(self):
        # Outside a turn there is no session id to attach — no crash, no key.
        plugin, exporter = _create_test_plugin()
        with plugin.llm_span("m", "p"):
            pass
        plugin.shutdown()

        llm = self._spans_by_kind(exporter)["LLM"]
        assert "session.id" not in llm.attributes


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestUserIdPropagation:
    """user.id (Langfuse User Tracking) rides the turn span and propagates
    to child llm/tool spans, so per-observation usage/cost attributes to the
    user. Langfuse's ingestion reads the OpenInference ``user.id`` key.
    """

    def _spans_by_kind(self, exporter):
        return {
            s.attributes.get("openinference.span.kind"): s
            for s in exporter.get_finished_spans()
        }

    def test_user_id_stamped_on_turn_and_children(self):
        plugin, exporter = _create_test_plugin()
        with plugin.turn_span(session_id="s", agent_type="main",
                              agent_name="main", user_id="user-42"):
            with plugin.llm_span("m", "p"):
                pass
            with plugin.tool_span("cli", "call-1"):
                pass
        plugin.shutdown()

        spans = self._spans_by_kind(exporter)
        assert spans["AGENT"].attributes["user.id"] == "user-42"
        assert spans["LLM"].attributes["user.id"] == "user-42"
        assert spans["TOOL"].attributes["user.id"] == "user-42"

    def test_no_user_id_when_absent(self):
        plugin, exporter = _create_test_plugin()
        with plugin.turn_span(session_id="s", agent_type="main",
                              agent_name="main"):
            with plugin.llm_span("m", "p"):
                pass
        plugin.shutdown()

        spans = self._spans_by_kind(exporter)
        assert "user.id" not in spans["AGENT"].attributes
        assert "user.id" not in spans["LLM"].attributes
