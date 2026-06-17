"""Tests for the AgentErrorEvent recovery contract (ask #5).

Covers the framework side of docs/design/agent-error-recovery-event.md:
- the SDK ``AgentErrorEvent`` (serialization, registry, server→bus bridge)
- ``_extract_provider_request_id`` (best-effort request-id plumbing)
- the server helpers ``_read_spawn_attempt`` / ``_emit_agent_error`` /
  ``_emit_agent_error_from_exc`` (attempt echo + hook routing)
- emit ordering at the bootstrap site (AgentErrorEvent BEFORE
  SessionTerminatedEvent)

The two-counter rule: the event's ``attempt`` is the REACTOR-level re-spawn
count, echoed verbatim from spawn ``agent_params["attempt"]`` — NOT
``with_retry``'s internal per-request attempts.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock


class TestAgentErrorEventWireContract:
    """SDK event: construct, round-trip, registry, server→bus bridge."""

    def test_round_trip_preserves_fields(self):
        from jaato_sdk.events import AgentErrorEvent, deserialize_event
        e = AgentErrorEvent(
            agent_id="host_validator", session_id="s1", error_type="APIError",
            error_summary="500 ...", request_id="req_abc", attempt="2",
            classification="transient_provider", framework_retries_exhausted=5,
            occurred_at=1.0,
        )
        back = deserialize_event(json.dumps(e.to_dict()))
        assert type(back).__name__ == "AgentErrorEvent"
        assert back.agent_id == "host_validator"
        assert back.error_type == "APIError"
        assert back.request_id == "req_abc"
        assert back.attempt == "2"
        assert back.classification == "transient_provider"
        assert back.framework_retries_exhausted == 5

    def test_registered_in_event_classes(self):
        from jaato_sdk.events import _EVENT_CLASSES, EventType, AgentErrorEvent
        assert _EVENT_CLASSES[EventType.AGENT_ERROR.value] is AgentErrorEvent

    def test_defaults(self):
        from jaato_sdk.events import AgentErrorEvent
        e = AgentErrorEvent()
        assert e.attempt == "0"  # first spawn
        assert e.classification is None
        assert e.request_id is None

    def test_server_to_bus_bridge_preserves_attempt(self):
        # The cascade reactor consumes the internal bus — the bridge must map
        # AgentErrorEvent and carry attempt/request_id in the payload.
        from server.core import _server_event_to_bus_event
        from jaato_sdk.events import AgentErrorEvent
        from jaato_sdk.event_bus import EventType as BusET
        be = _server_event_to_bus_event(AgentErrorEvent(
            agent_id="a", session_id="s", error_type="APIError",
            attempt="3", request_id="req_9",
        ))
        assert be is not None
        assert be.event_type == BusET.AGENT_ERROR
        assert be.payload["attempt"] == "3"
        assert be.payload["request_id"] == "req_9"


class TestExtractProviderRequestId:
    """Best-effort request-id extraction across attribute shapes."""

    def test_direct_request_id_attr(self):
        from server.core import _extract_provider_request_id
        exc = ValueError("x")
        exc.request_id = "req_direct"
        assert _extract_provider_request_id(exc) == "req_direct"

    def test_response_headers(self):
        from server.core import _extract_provider_request_id
        exc = ValueError("x")
        exc.response = SimpleNamespace(headers={"x-request-id": "req_hdr"})
        assert _extract_provider_request_id(exc) == "req_hdr"

    def test_cause_chain(self):
        from server.core import _extract_provider_request_id
        inner = ValueError("inner")
        inner.request_id = "req_inner"
        outer = RuntimeError("outer")
        outer.__cause__ = inner
        assert _extract_provider_request_id(outer) == "req_inner"

    def test_none_when_absent(self):
        from server.core import _extract_provider_request_id
        assert _extract_provider_request_id(ValueError("x")) is None
        assert _extract_provider_request_id(None) is None


def _capturing_server(attempt_param=None):
    """An object with the minimal surface the server helpers read, callable as
    unbound JaatoServer methods.  ``_agent_hooks.on_agent_error`` captures."""
    calls: list = []

    class Hooks:
        def on_agent_error(self, **kw):
            calls.append(kw)

    from server.core import JaatoServer
    agent_params = {} if attempt_param is None else {"attempt": attempt_param}
    fake = SimpleNamespace(
        _agent_hooks=Hooks(),
        session_id="s1",
        _main_agent_id="host_validator",
        _jaato=SimpleNamespace(
            get_session=lambda: SimpleNamespace(_agent_params=agent_params)
        ),
    )
    # Bind the real JaatoServer methods the helpers call via ``self.``.
    fake._get_ui_hooks = lambda: JaatoServer._get_ui_hooks(fake)
    fake._read_spawn_attempt = lambda: JaatoServer._read_spawn_attempt(fake)
    fake._emit_agent_error = lambda **kw: JaatoServer._emit_agent_error(fake, **kw)
    return fake, calls


class TestReadSpawnAttempt:
    """attempt echo — reactor-level count from agent_params, default '0'."""

    def test_reads_attempt_from_agent_params(self):
        from server.core import JaatoServer
        fake, _ = _capturing_server(attempt_param="3")
        assert JaatoServer._read_spawn_attempt(fake) == "3"

    def test_defaults_zero_when_absent(self):
        from server.core import JaatoServer
        fake, _ = _capturing_server(attempt_param=None)
        assert JaatoServer._read_spawn_attempt(fake) == "0"

    def test_defaults_zero_when_no_session(self):
        from server.core import JaatoServer
        fake = SimpleNamespace(_jaato=None)
        assert JaatoServer._read_spawn_attempt(fake) == "0"


class TestEmitAgentErrorHelpers:
    """_emit_agent_error / _emit_agent_error_from_exc route through the hook."""

    def test_emit_agent_error_routes_with_echoed_attempt(self):
        from server.core import JaatoServer
        fake, calls = _capturing_server(attempt_param="2")
        JaatoServer._emit_agent_error(
            fake, error_type="APIError", error_summary="boom",
            request_id="req_1", classification="transient_provider",
        )
        assert len(calls) == 1
        kw = calls[0]
        assert kw["error_type"] == "APIError"
        assert kw["error_summary"] == "boom"
        assert kw["request_id"] == "req_1"
        assert kw["attempt"] == "2"            # echoed from agent_params
        assert kw["session_id"] == "s1"
        assert kw["agent_id"] == "host_validator"
        assert kw["classification"] == "transient_provider"

    def test_from_exc_derives_fields(self):
        from server.core import JaatoServer
        fake, calls = _capturing_server(attempt_param="0")

        class APIError(Exception):
            pass
        exc = APIError("upstream 500")
        exc.request_id = "req_x"
        JaatoServer._emit_agent_error_from_exc(fake, exc)
        kw = calls[0]
        assert kw["error_type"] == "APIError"
        assert kw["error_summary"] == "upstream 500"
        assert kw["request_id"] == "req_x"

    def test_overrides_session_and_agent_id(self):
        from server.core import JaatoServer
        fake, calls = _capturing_server()
        JaatoServer._emit_agent_error(
            fake, error_type="X", error_summary="y",
            session_id="boot_sid", agent_id="boot_agent",
        )
        assert calls[0]["session_id"] == "boot_sid"
        assert calls[0]["agent_id"] == "boot_agent"

    def test_no_hooks_is_silent_noop(self):
        from server.core import JaatoServer
        fake = SimpleNamespace()  # no _agent_hooks → _get_ui_hooks None
        # Must not raise.
        JaatoServer._emit_agent_error(fake, error_type="X", error_summary="y")


class TestBootstrapSiteRoutesThroughChokepoint:
    """The bootstrap-failure site delegates to the single error-termination
    chokepoint (which guarantees AgentErrorEvent precedes
    SessionTerminatedEvent — see TestErrorTerminationChokepoint)."""

    def test_delegates_to_chokepoint_with_ids(self):
        from server.runner_spawn import _emit_bootstrap_terminated

        server = MagicMock()
        server._main_agent_id = "build_judge"
        exc = ValueError("bootstrap boom")
        _emit_bootstrap_terminated(server=server, session_id="sid_1", exc=exc)

        server._emit_error_termination_from_exc.assert_called_once()
        call = server._emit_error_termination_from_exc.call_args
        assert call.args[0] is exc
        assert call.kwargs["session_id"] == "sid_1"
        assert call.kwargs["agent_id"] == "build_judge"

    def test_chokepoint_failure_does_not_raise(self):
        """If the chokepoint raises, the bootstrap helper logs but doesn't raise
        — the underlying bootstrap failure isn't masked."""
        from server.runner_spawn import _emit_bootstrap_terminated

        server = MagicMock()
        server._main_agent_id = "main"
        server._emit_error_termination_from_exc = MagicMock(
            side_effect=RuntimeError("emit boom"),
        )
        # Must not raise.
        _emit_bootstrap_terminated(server=server, session_id="sid", exc=ValueError("x"))


class TestErrorTerminationChokepoint:
    """``_emit_error_termination`` is THE single error-terminal emit point:
    AgentErrorEvent (recovery first refusal) THEN
    SessionTerminatedEvent(reason="error"), plus ``_terminal_reason="error"``.
    This is the structural invariant that lets a cascade retire the redundant
    ``on session.terminated reason=error`` abort reactor (which raced the
    recovery reactor's mark_handled)."""

    def _fake(self, order):
        from server.core import JaatoServer

        class Hooks:
            def on_agent_error(self, **kw):
                order.append(("agent_error", kw.get("error_type"), kw.get("error_summary")))

        fake = SimpleNamespace(
            _agent_hooks=Hooks(), session_id="s1", _main_agent_id="stage1",
            _jaato=SimpleNamespace(get_session=lambda: SimpleNamespace(_agent_params={})),
        )
        fake._get_ui_hooks = lambda: JaatoServer._get_ui_hooks(fake)
        fake._read_spawn_attempt = lambda: JaatoServer._read_spawn_attempt(fake)
        fake._emit_agent_error = lambda **kw: JaatoServer._emit_agent_error(fake, **kw)
        fake._emit_error_termination = lambda **kw: JaatoServer._emit_error_termination(fake, **kw)
        fake.emit = lambda ev: order.append(
            ("emit", type(ev).__name__, getattr(ev, "reason", None), getattr(ev, "error_type", None))
        )
        return fake

    def test_orders_agent_error_before_terminated_and_stamps_reason(self):
        from server.core import JaatoServer
        order = []
        fake = self._fake(order)
        JaatoServer._emit_error_termination(fake, error_type="APIError", error_summary="boom")
        assert order[0][:2] == ("agent_error", "APIError")
        assert order[1][:3] == ("emit", "SessionTerminatedEvent", "error")
        assert fake._terminal_reason == "error"

    def test_from_exc_derives_and_routes_in_order(self):
        from server.core import JaatoServer
        order = []
        fake = self._fake(order)

        class APIError(Exception):
            pass
        JaatoServer._emit_error_termination_from_exc(fake, APIError("upstream 500"))
        assert order[0] == ("agent_error", "APIError", "upstream 500")
        assert order[1] == ("emit", "SessionTerminatedEvent", "error", "APIError")

    def test_guard_no_reason_error_emit_outside_chokepoint(self):
        """STRUCTURAL guard: every SessionTerminatedEvent(reason="error") in
        core.py is constructed INSIDE _emit_error_termination, so a future error
        path can't emit a terminal error without first offering recovery."""
        import ast
        import inspect
        from server import core

        tree = ast.parse(inspect.getsource(core))
        choke = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_emit_error_termination"
        )
        span = range(choke.lineno, (choke.end_lineno or choke.lineno) + 1)
        offenders = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "SessionTerminatedEvent"):
                reason = next((kw.value for kw in node.keywords if kw.arg == "reason"), None)
                if isinstance(reason, ast.Constant) and reason.value == "error" \
                        and node.lineno not in span:
                    offenders.append(node.lineno)
        assert not offenders, (
            f"SessionTerminatedEvent(reason='error') constructed outside "
            f"_emit_error_termination at core.py lines {offenders} — that bypasses "
            f"the AgentErrorEvent recovery offer.  Route it through the chokepoint."
        )
