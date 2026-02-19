"""Null (no-op) telemetry plugin.

This plugin provides zero-overhead telemetry when tracing is disabled.
All methods are no-ops that return immediately without any OTel imports.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional


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

    @property
    def enabled(self) -> bool:
        """Always returns False."""
        return False

    @contextmanager
    def turn_span(
        self,
        session_id: str,
        agent_type: str,
        agent_name: Optional[str] = None,
        turn_index: Optional[int] = None,
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

    def get_current_trace_id(self) -> Optional[str]:
        return None

    def get_current_span_id(self) -> Optional[str]:
        return None
