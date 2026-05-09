"""Tests for ``shared.plugins.runner_forwarding`` (Phase 3 §3.4).

The mixin is the single wrap-point that lets every runner-tier
plugin migration ship as a 2-line change.  Pin the contract:
- No runner attached → in-process body runs.
- Runner attached → tool.execute RPC with name/args + thread-local
  callbacks threaded through.
- ok=True dict result → returned unchanged.
- Domain failure dict result → returned unchanged (model sees the
  same shape).
- CancelledException → re-raised locally.
- Transport failure → legacy error-dict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from server.runner.envelope import ErrorPayload, ResponseEnvelope
from shared.plugins.runner_forwarding import (
    RunnerForwardingMixin,
    _forward_via_runner,
)


# ----------------------------------------------------------------------
# Stub RPC client — records call_threadsafe args, returns canned envelopes
# ----------------------------------------------------------------------


class _StubRPC:
    """Minimal RunnerRPCClient stand-in for unit tests."""

    def __init__(self) -> None:
        self.next_response: Optional[ResponseEnvelope] = None
        self.next_raise: Optional[BaseException] = None
        self.calls: List[Dict[str, Any]] = []

    def call_threadsafe(
        self,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        on_output: Optional[Any] = None,
        cancel_token: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> ResponseEnvelope:
        self.calls.append({
            "method": method, "args": args,
            "on_output": on_output, "cancel_token": cancel_token,
            "timeout": timeout,
        })
        if self.next_raise is not None:
            raise self.next_raise
        if self.next_response is None:
            raise AssertionError("test forgot to set next_response")
        return self.next_response


class _DemoPlugin(RunnerForwardingMixin):
    """Tiny plugin with two _execute_* methods to validate the
    wrap-all-executors pattern."""

    def __init__(self) -> None:
        self._plugin_registry: Any = None
        self.in_process_calls: List[str] = []

    def _execute_alpha(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.in_process_calls.append("alpha")
        return {"in_process": "alpha", "args": dict(args)}

    def _execute_beta(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.in_process_calls.append("beta")
        return {"in_process": "beta"}

    def get_executors(self):
        return self.wrap_executors_for_runner_forwarding({
            "alpha": self._execute_alpha,
            "beta": self._execute_beta,
        })


# ----------------------------------------------------------------------
# No-runner fallthrough
# ----------------------------------------------------------------------


def test_no_registry_runs_in_process() -> None:
    """When ``_plugin_registry`` is None (the default), the wrapper
    invokes the underlying in-process executor."""
    plugin = _DemoPlugin()
    executors = plugin.get_executors()
    result = executors["alpha"]({"k": "v"})
    assert result == {"in_process": "alpha", "args": {"k": "v"}}
    assert plugin.in_process_calls == ["alpha"]


def test_registry_without_runner_rpc_runs_in_process() -> None:
    """A registry that doesn't declare ``runner_rpc`` (Phase 2
    pre-runner sessions) falls through to the in-process body."""
    plugin = _DemoPlugin()
    plugin._plugin_registry = MagicMock(spec=[])  # no attributes
    executors = plugin.get_executors()
    result = executors["beta"]({})
    assert result == {"in_process": "beta"}
    assert plugin.in_process_calls == ["beta"]


def test_registry_with_runner_rpc_none_runs_in_process() -> None:
    """Explicit ``runner_rpc=None`` (e.g. between sessions or after
    runner crash) keeps the in-process body."""
    plugin = _DemoPlugin()
    registry = MagicMock()
    registry.runner_rpc = None
    plugin._plugin_registry = registry
    result = plugin.get_executors()["alpha"]({})
    assert result == {"in_process": "alpha", "args": {}}


# ----------------------------------------------------------------------
# Runner-attached forwarding happy path
# ----------------------------------------------------------------------


def test_runner_attached_forwards_via_rpc() -> None:
    plugin = _DemoPlugin()
    rpc = _StubRPC()
    rpc.next_response = ResponseEnvelope(
        id=1, ok=True,
        result={"from": "runner", "alpha": True},
    )
    registry = MagicMock()
    registry.runner_rpc = rpc
    plugin._plugin_registry = registry

    result = plugin.get_executors()["alpha"]({"x": 1})
    assert result == {"from": "runner", "alpha": True}
    assert plugin.in_process_calls == []  # in-process body NOT called
    # RPC call shape correct.
    assert len(rpc.calls) == 1
    assert rpc.calls[0]["method"] == "tool.execute"
    assert rpc.calls[0]["args"] == {"name": "alpha", "args": {"x": 1}}


def test_runner_forward_passes_thread_local_callbacks() -> None:
    """The forwarder reads ``tool_output_callback`` + ``cancel_token``
    from ``shared.ai_tool_runner._thread_local`` and threads them
    through the RPC call."""
    from shared.ai_tool_runner import _thread_local as _ai_tl

    plugin = _DemoPlugin()
    rpc = _StubRPC()
    rpc.next_response = ResponseEnvelope(id=1, ok=True, result={})
    registry = MagicMock()
    registry.runner_rpc = rpc
    plugin._plugin_registry = registry

    sentinel_cb = lambda src, txt, mode: None  # noqa: E731
    sentinel_token = object()

    _ai_tl.tool_output_callback = sentinel_cb
    _ai_tl.cancel_token = sentinel_token
    try:
        plugin.get_executors()["alpha"]({})
    finally:
        _ai_tl.tool_output_callback = None
        _ai_tl.cancel_token = None

    assert rpc.calls[0]["on_output"] is sentinel_cb
    assert rpc.calls[0]["cancel_token"] is sentinel_token


# ----------------------------------------------------------------------
# Result translation
# ----------------------------------------------------------------------


def test_runner_returns_ok_dict_unchanged() -> None:
    plugin = _DemoPlugin()
    rpc = _StubRPC()
    rpc.next_response = ResponseEnvelope(
        id=1, ok=True, result={"complex": {"nested": [1, 2, 3]}},
    )
    plugin._plugin_registry = MagicMock(runner_rpc=rpc)

    result = plugin.get_executors()["alpha"]({})
    assert result == {"complex": {"nested": [1, 2, 3]}}


def test_runner_domain_failure_dict_returned_unchanged() -> None:
    """ok=False with structured result (e.g. exec-not-found) flows
    through to the model unchanged."""
    plugin = _DemoPlugin()
    rpc = _StubRPC()
    rpc.next_response = ResponseEnvelope(
        id=1, ok=False,
        result={
            "error": "alpha: not found",
            "hint": "configure extra_paths",
        },
        error=ErrorPayload(type="ToolError", message="alpha: not found"),
    )
    plugin._plugin_registry = MagicMock(runner_rpc=rpc)

    result = plugin.get_executors()["alpha"]({})
    assert "not found" in result["error"]
    assert result["hint"] == "configure extra_paths"


def test_runner_domain_failure_no_result_synthesises_error_dict() -> None:
    """ok=False with no structured result → synthesise a legacy
    error-dict from the error payload."""
    plugin = _DemoPlugin()
    rpc = _StubRPC()
    rpc.next_response = ResponseEnvelope(
        id=1, ok=False, result=None,
        error=ErrorPayload(type="ValueError", message="bad arg"),
    )
    plugin._plugin_registry = MagicMock(runner_rpc=rpc)

    result = plugin.get_executors()["alpha"]({})
    assert result == {"error": "bad arg"}


def test_runner_cancelled_exception_propagates() -> None:
    """CancelledException is re-raised locally so the in-process
    pipeline classifies the call as cancelled."""
    from jaato_sdk.plugins.model_provider.types import CancelledException

    plugin = _DemoPlugin()
    rpc = _StubRPC()
    rpc.next_response = ResponseEnvelope(
        id=1, ok=False, result=None,
        error=ErrorPayload(
            type="CancelledException",
            message="Cancelled during subprocess output",
        ),
    )
    plugin._plugin_registry = MagicMock(runner_rpc=rpc)

    with pytest.raises(CancelledException, match="Cancelled"):
        plugin.get_executors()["alpha"]({})


def test_runner_transport_failure_translated_to_error_dict() -> None:
    """RunnerCallError / similar transport-level failures translate
    to the legacy error-dict shape so the model sees a clean
    message."""
    from server.runner_rpc_client import RunnerCallError

    plugin = _DemoPlugin()
    rpc = _StubRPC()
    rpc.next_raise = RunnerCallError("runner closed mid-call")
    plugin._plugin_registry = MagicMock(runner_rpc=rpc)

    result = plugin.get_executors()["alpha"]({})
    assert "runner RPC failed" in result["error"]
    assert "RunnerCallError" in result["error"]


# ----------------------------------------------------------------------
# Wrap-all-executors contract
# ----------------------------------------------------------------------


def test_wrap_preserves_iteration_order() -> None:
    """Plugins that rely on executor iteration order (introspection,
    deferred-tool ordering) keep their shape."""
    plugin = _DemoPlugin()
    plugin._plugin_registry = MagicMock(spec=[])
    keys = list(plugin.get_executors().keys())
    assert keys == ["alpha", "beta"]


def test_wrap_preserves_function_name() -> None:
    """The wrapped function carries the underlying executor's
    __name__ for traces / debuggers."""
    plugin = _DemoPlugin()
    wrapped = plugin.get_executors()["alpha"]
    assert wrapped.__name__ == "_execute_alpha"


def test_each_call_re_resolves_runner_rpc() -> None:
    """Setting ``runner_rpc`` AFTER ``get_executors()`` returns is
    legitimate (registry-attribute pattern from §5.4 of plan v5 —
    daemon stashes the handle after the plugin is configured).
    The wrapper re-resolves on every call."""
    plugin = _DemoPlugin()
    plugin._plugin_registry = MagicMock(spec=[])  # no runner_rpc yet
    exec_alpha = plugin.get_executors()["alpha"]

    # First call — no runner; runs in-process.
    r1 = exec_alpha({"phase": 1})
    assert r1["in_process"] == "alpha"

    # Now attach runner.
    rpc = _StubRPC()
    rpc.next_response = ResponseEnvelope(id=1, ok=True, result={"phase": 2})
    plugin._plugin_registry.runner_rpc = rpc

    # Second call — runner attached; forwards.
    r2 = exec_alpha({"phase": 2})
    assert r2 == {"phase": 2}
    assert plugin.in_process_calls == ["alpha"]  # only the first call


# ----------------------------------------------------------------------
# Module-level _forward_via_runner (independent of plugin instances)
# ----------------------------------------------------------------------


def test_module_level_forward_handles_ok_response() -> None:
    rpc = _StubRPC()
    rpc.next_response = ResponseEnvelope(
        id=1, ok=True, result={"sentinel": "module-level"},
    )
    result = _forward_via_runner(rpc, "test_tool", {"k": "v"})
    assert result == {"sentinel": "module-level"}
