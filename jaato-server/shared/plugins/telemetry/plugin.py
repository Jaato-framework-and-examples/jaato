"""Telemetry plugin protocol definition.

This module defines the TelemetryPlugin protocol that all telemetry
implementations must follow. The protocol uses context managers for
span creation to ensure proper cleanup and timing.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Protocol, runtime_checkable


class SpanContext(Protocol):
    """Protocol for span context returned by span creation methods.

    Provides methods to add attributes, events, record exceptions,
    and set OpenInference message/metadata attributes on the current span.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span.

        For LLM spans, setting ``llm.token_count.prompt`` and
        ``llm.token_count.completion`` will auto-compute
        ``llm.token_count.total``.

        Args:
            key: Attribute name (e.g., "llm.token_count.prompt")
            value: Attribute value (string, int, float, bool, or list thereof)
        """
        ...

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span.

        Args:
            exception: The exception to record
        """
        ...

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span.

        Args:
            name: Event name
            attributes: Optional event attributes
        """
        ...

    def set_status_error(self, description: str = "") -> None:
        """Set the span status to error.

        Args:
            description: Optional error description
        """
        ...

    def set_status_ok(self) -> None:
        """Set the span status to OK."""
        ...

    def set_input_messages(self, messages: 'List[Dict[str, Any]]') -> None:
        """Set OpenInference input messages on the span.

        Flattens messages to indexed attributes::

            llm.input_messages.{i}.message.role
            llm.input_messages.{i}.message.content

        Each message dict should have ``role`` and ``content`` keys.
        Content is subject to redaction when enabled.

        Args:
            messages: List of message dicts with 'role' and 'content'.
        """
        ...

    def set_output_messages(self, messages: 'List[Dict[str, Any]]') -> None:
        """Set OpenInference output messages on the span.

        Same format as input messages. Tool calls within messages are
        flattened to::

            llm.output_messages.{i}.message.tool_calls.{j}.tool_call.function.name
            llm.output_messages.{i}.message.tool_calls.{j}.tool_call.function.arguments

        Args:
            messages: List of message dicts with 'role', 'content', and
                optional 'tool_calls' list.
        """
        ...

    def set_metadata(self, data: Dict[str, Any]) -> None:
        """Set OpenInference metadata attribute as a JSON string.

        Used for jaato-specific fields that have no OpenInference equivalent
        (e.g., agent_type, turn_index, plugin_type).

        Args:
            data: Dict of arbitrary metadata to serialize as JSON.
        """
        ...


@runtime_checkable
class TelemetryPlugin(Protocol):
    """Protocol for telemetry/tracing plugins.

    Implementations provide distributed tracing capabilities for jaato
    operations. The default NullTelemetryPlugin provides zero overhead
    when telemetry is disabled.

    All span methods are context managers that yield a SpanContext for
    adding attributes and events during the span's lifetime.
    """

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
                - redact_content: bool (redact prompts/responses, default True)
        """
        ...

    def shutdown(self) -> None:
        """Flush pending spans and shutdown telemetry.

        Should be called before application exit to ensure all spans
        are exported.
        """
        ...

    @property
    def enabled(self) -> bool:
        """Check if telemetry is enabled."""
        ...

    def begin_session(
        self,
        session_id: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start the root session span.

        Creates a long-lived span that parents all agent and turn spans
        for this session.  Call ``end_session`` when the session is done.
        """
        ...

    def end_session(self, session_id: str) -> None:
        """End the session span."""
        ...

    def begin_agent(
        self,
        session_id: str,
        agent_id: str,
        agent_name: Optional[str] = None,
        agent_type: str = "main",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start an agent span under the session span.

        Turn spans are automatically parented under the agent span.
        Call ``end_agent`` when the agent is done.
        """
        ...

    def end_agent(self, session_id: str, agent_id: str) -> None:
        """End an agent span."""
        ...

    @contextmanager
    def turn_span(
        self,
        session_id: str,
        agent_type: str,
        agent_name: Optional[str] = None,
        turn_index: Optional[int] = None,
        parent_session_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Create root span for a turn (send_message call).

        This is the top-level span that encompasses an entire turn,
        including all LLM calls, tool executions, and retries.
        Sets ``openinference.span.kind = "AGENT"`` and populates
        ``graph.node.*`` attributes for Phoenix DAG visualization.

        Args:
            session_id: Unique session identifier
            agent_type: Agent type ("main" or "subagent")
            agent_name: Optional agent name
            turn_index: Optional turn number in the session
            parent_session_id: Optional parent session ID for subagents.
                Populates ``graph.node.parent_id`` for DAG visualization.
                Empty string or None means root node.
            attributes: Optional additional attributes

        Yields:
            SpanContext for adding attributes during the turn
        """
        ...

    @contextmanager
    def llm_span(
        self,
        model: str,
        provider: str,
        streaming: bool = False,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Create span for an LLM API call.

        Child of the current turn span. Records model, provider,
        token usage, and finish reason.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514")
            provider: Provider name (e.g., "anthropic", "google_genai")
            streaming: Whether streaming is enabled
            attributes: Optional additional attributes

        Yields:
            SpanContext for adding token usage and other attributes
        """
        ...

    @contextmanager
    def tool_span(
        self,
        tool_name: str,
        call_id: str,
        plugin_type: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Create span for tool execution.

        Child of the current LLM span. Records tool name, duration,
        success/failure, and error details.

        Args:
            tool_name: Name of the tool being executed
            call_id: Unique identifier for this tool call
            plugin_type: Plugin type ("cli", "mcp", "builtin")
            attributes: Optional additional attributes

        Yields:
            SpanContext for adding execution results
        """
        ...

    @contextmanager
    def retry_span(
        self,
        attempt: int,
        max_attempts: int,
        context: str = "api_call",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Create span for a retry attempt.

        Child of the current LLM span. Records attempt number,
        backoff delay, and error classification.

        Args:
            attempt: Current attempt number (1-indexed)
            max_attempts: Maximum number of attempts
            context: Context for the retry (e.g., "api_call", "tool_exec")
            attributes: Optional additional attributes

        Yields:
            SpanContext for adding retry details
        """
        ...

    @contextmanager
    def gc_span(
        self,
        trigger_reason: str,
        strategy: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Create span for garbage collection operation.

        Child of the current turn span. Records GC trigger reason,
        strategy, items collected, and tokens freed.

        Args:
            trigger_reason: Why GC was triggered ("threshold", "manual", "turn_limit")
            strategy: GC strategy used ("truncate", "summarize", "hybrid")
            attributes: Optional additional attributes

        Yields:
            SpanContext for adding GC results
        """
        ...

    @contextmanager
    def permission_span(
        self,
        tool_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Create span for permission check.

        Child of the current tool span. Records permission decision
        and any user interaction required.

        Args:
            tool_name: Tool requiring permission
            attributes: Optional additional attributes

        Yields:
            SpanContext for adding permission decision
        """
        ...

    def capture_context(self) -> Optional[Any]:
        """Capture the current OTel context for propagation to worker threads.

        Call on the parent thread before submitting work to a thread pool.
        Pass the result to ``attach_context()`` on the worker thread.

        Returns:
            Opaque context object, or None if telemetry is disabled.
        """
        ...

    @contextmanager
    def attach_context(self, ctx: Optional[Any]) -> Generator[None, None, None]:
        """Attach a previously captured OTel context on the current thread.

        Use as a context manager in worker threads so child spans are
        parented correctly.

        Args:
            ctx: Context from ``capture_context()``, or None.
        """
        ...

    def subscribe_to_bus(self, bus) -> None:
        """Subscribe to EventBus for plan/step lifecycle context.

        When subscribed, STEP_STARTED events set plan_id/step_id on the
        current thread so all subsequent spans carry them.  STEP_COMPLETED,
        STEP_FAILED, and STEP_SKIPPED clear the context.

        Args:
            bus: A TaskEventBus instance to subscribe to.
        """
        ...

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID if available.

        Returns:
            Hex-encoded trace ID or None if no active trace
        """
        ...

    def get_current_span_id(self) -> Optional[str]:
        """Get the current span ID if available.

        Returns:
            Hex-encoded span ID or None if no active span
        """
        ...
