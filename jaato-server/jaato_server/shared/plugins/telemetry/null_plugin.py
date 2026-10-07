"""Null (no-op) telemetry plugin.

This plugin provides zero-overhead telemetry when tracing is disabled.
All methods are no-ops that return immediately without any OTel imports.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional


class _NoOpSpan:
    """No-op span context that ignores all operations."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def set_status_error(self, description: str = "") -> None:
        pass

    def set_status_ok(self) -> None:
        pass

    def set_input_messages(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def set_output_messages(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def set_metadata(self, data: Dict[str, Any]) -> None:
        pass


# Singleton no-op span to avoid allocations
_NOOP_SPAN = _NoOpSpan()


class NullTelemetryPlugin:
    """No-op telemetry plugin with zero overhead.

    This is the default plugin when telemetry is disabled. All methods
    return immediately without any imports or allocations.
    """

    __slots__ = ("_enabled",)

    def __init__(self) -> None:
        self._enabled = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """No-op initialization."""
        pass

    def shutdown(self) -> None:
        """No-op shutdown."""
        pass

    def reset_for_next_session(self) -> None:
        """No-op (required by the ``TelemetryPlugin`` protocol).

        The null plugin holds no per-session state, so there is nothing
        to clear between cascade sessions."""
        pass

    @property
    def enabled(self) -> bool:
        """Always returns False."""
        return False

    def begin_session(self, session_id, attributes=None):
        pass

    def end_session(self, session_id):
        pass

    def begin_agent(self, session_id, agent_id, agent_name=None, agent_type="main", attributes=None):
        pass

    def end_agent(self, session_id, agent_id):
        pass

    @contextmanager
    def turn_span(
        self,
        session_id: str,
        agent_type: str,
        agent_name: Optional[str] = None,
        turn_index: Optional[int] = None,
        parent_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NOOP_SPAN

    @contextmanager
    def llm_span(
        self,
        model: str,
        provider: str,
        streaming: bool = False,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NOOP_SPAN

    @contextmanager
    def tool_span(
        self,
        tool_name: str,
        call_id: str,
        plugin_type: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NOOP_SPAN

    @contextmanager
    def retry_span(
        self,
        attempt: int,
        max_attempts: int,
        context: str = "api_call",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NOOP_SPAN

    @contextmanager
    def gc_span(
        self,
        trigger_reason: str,
        strategy: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NOOP_SPAN

    @contextmanager
    def permission_span(
        self,
        tool_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NOOP_SPAN

    def capture_context(self) -> Optional[Any]:
        """No-op context capture."""
        return None

    @contextmanager
    def attach_context(self, ctx: Optional[Any]) -> Generator[None, None, None]:
        """No-op context attachment."""
        yield

    def subscribe_to_bus(self, bus) -> None:
        """No-op bus subscription."""
        pass

    def get_current_trace_id(self) -> Optional[str]:
        return None

    def get_current_span_id(self) -> Optional[str]:
        return None

    def register_attribute_redactor(
        self, fn: Callable[[str, Any], Any]
    ) -> None:
        """No-op attribute redactor registration.

        The Null plugin produces no spans, so registered redactors
        would never fire.  Accept and ignore so callers can
        unconditionally register without checking which plugin is
        active.
        """
        pass
