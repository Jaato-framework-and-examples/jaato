"""Tests for ``RunnerRPC.tool.execute`` routing through the
bootstrapped session host's executor (Phase 3 §3.3c part 3a).

When ``RunnerRPC._session_host`` is set + the host has a live
session, ``tool.execute`` dispatches through
``host.session._executor.execute(name, args, ...)`` instead of the
cli-only Phase 2 ``execute_fn``.  This unlocks every runner-tier
plugin migration to "just add a daemon-side RPC stub" without
per-plugin runner-side wiring.

Falls through to the Phase 2 surface when no host is bootstrapped
— preserves cli-only runners + tests.
"""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
    )


class _RecordingExecutor:
    """Minimal stand-in for ``ToolExecutor`` from
    ``shared/ai_tool_runner.py``.  Records the args it was called
    with so tests can assert the RPC dispatcher routed correctly."""

    def __init__(
        self,
        *,
        return_value: Tuple[bool, Any] = (True, {"result": "ok"}),
    ) -> None:
        self._return = return_value
        self.calls: List[Dict[str, Any]] = []

    def execute(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        tool_output_callback: Optional[Any] = None,
        call_id: Optional[str] = None,
        cancel_token: Optional[Any] = None,
    ) -> Tuple[bool, Any]:
        self.calls.append({
            "name": name,
            "args": dict(args),
            "has_output_cb": tool_output_callback is not None,
            "call_id": call_id,
            "has_cancel_token": cancel_token is not None,
        })
        return self._return


def _make_lone_runner(execute_fn=None) -> RunnerRPC:
    """Build a RunnerRPC against a closed-side socket; we exercise
    ``_dispatch_method`` directly without driving the serve loop."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _default_exec(name, args):
        return False, {"error": "fallback path; should not be hit"}

    return RunnerRPC(a, execute_fn or _default_exec)


def _set_host_with_executor(rpc: RunnerRPC, executor: Any) -> RunnerSessionHost:
    """Inject a fake host with the given executor so dispatch
    routes through it.  Mirrors what ``bootstrap_session`` would
    set up post-§3.3b."""
    fake_session = SimpleNamespace(_executor=executor)
    host = RunnerSessionHost(
        envelope=_envelope(),
        runtime=None,
        session=fake_session,  # type: ignore[arg-type]
    )
    rpc._session_host = host
    return host


# ----------------------------------------------------------------------
# Default fallthrough — no session host bootstrapped
# ----------------------------------------------------------------------


def test_tool_execute_falls_through_to_execute_fn_when_no_host() -> None:
    """When no session host is bootstrapped, ``tool.execute`` keeps
    the Phase 2 behaviour (routes through the registered
    ``execute_fn``)."""
    captured: List[Tuple[str, Dict[str, Any]]] = []

    def _execute_fn(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        captured.append((name, dict(args)))
        return True, {"sentinel": "execute_fn"}

    rpc = _make_lone_runner(execute_fn=_execute_fn)
    from server.runner.envelope import RequestEnvelope
    ok, result = rpc._dispatch_method(
        RequestEnvelope(
            id=1, method="tool.execute",
            args={"name": "echo", "args": {"k": "v"}},
        ),
    )
    assert ok is True
    assert result == {"sentinel": "execute_fn"}
    assert captured == [("echo", {"k": "v"})]


def test_tool_execute_falls_through_when_host_has_no_session() -> None:
    """A host without a live session (e.g. test-only
    runtime_factory=None path) still falls through."""
    rpc = _make_lone_runner()
    rpc._session_host = RunnerSessionHost(
        envelope=_envelope(), runtime=None, session=None,
    )
    from server.runner.envelope import RequestEnvelope
    ok, result = rpc._dispatch_method(
        RequestEnvelope(
            id=1, method="tool.execute",
            args={"name": "echo", "args": {"k": "v"}},
        ),
    )
    # The default execute_fn returns the fallback sentinel.
    assert ok is False
    assert "fallback" in result.get("error", "")


def test_tool_execute_falls_through_when_session_has_no_executor() -> None:
    """Session attribute exists but has no ``_executor`` — fall
    through to ``execute_fn``."""
    rpc = _make_lone_runner()
    fake_session = SimpleNamespace()  # no _executor attribute
    rpc._session_host = RunnerSessionHost(
        envelope=_envelope(), runtime=None,
        session=fake_session,  # type: ignore[arg-type]
    )
    from server.runner.envelope import RequestEnvelope
    ok, result = rpc._dispatch_method(
        RequestEnvelope(
            id=1, method="tool.execute",
            args={"name": "echo", "args": {}},
        ),
    )
    assert ok is False  # the default fallback


# ----------------------------------------------------------------------
# Route-through happy path
# ----------------------------------------------------------------------


def test_tool_execute_routes_through_session_executor() -> None:
    rpc = _make_lone_runner()
    executor = _RecordingExecutor(
        return_value=(True, {"stdout": "via session executor"}),
    )
    _set_host_with_executor(rpc, executor)
    from server.runner.envelope import RequestEnvelope

    ok, result = rpc._dispatch_method(
        RequestEnvelope(
            id=1, method="tool.execute",
            args={"name": "todo_list", "args": {"plan_id": "p1"}},
        ),
    )
    assert ok is True
    assert result == {"stdout": "via session executor"}
    # The recording executor saw the routed call with the right name.
    assert len(executor.calls) == 1
    assert executor.calls[0]["name"] == "todo_list"
    assert executor.calls[0]["args"] == {"plan_id": "p1"}


def test_tool_execute_passes_streaming_callback_to_executor() -> None:
    """The per-call ``on_output`` adapter is threaded through to the
    session executor's ``tool_output_callback`` parameter."""
    rpc = _make_lone_runner()
    executor = _RecordingExecutor()
    _set_host_with_executor(rpc, executor)
    from server.runner.envelope import RequestEnvelope

    rpc._dispatch_method(
        RequestEnvelope(
            id=42, method="tool.execute",
            args={"name": "fake", "args": {}},
        ),
    )
    assert executor.calls[0]["has_output_cb"] is True


def test_tool_execute_passes_cancel_token_to_executor_when_active() -> None:
    """When a per-call cancel token is registered (RunnerRPC's
    ``_active_calls`` table), it's passed through to the executor."""
    from jaato_sdk.plugins.model_provider.types import CancelToken
    from server.runner.rpc import _ActiveCall

    rpc = _make_lone_runner()
    executor = _RecordingExecutor()
    _set_host_with_executor(rpc, executor)
    token = CancelToken()
    rpc._active_calls[100] = _ActiveCall(cancel_token=token)
    try:
        from server.runner.envelope import RequestEnvelope

        rpc._dispatch_method(
            RequestEnvelope(
                id=100, method="tool.execute",
                args={"name": "fake", "args": {}},
            ),
        )
        assert executor.calls[0]["has_cancel_token"] is True
    finally:
        rpc._active_calls.pop(100, None)


def test_tool_execute_no_active_call_passes_none_token() -> None:
    """When there's no active-call entry for this request_id (which
    can happen when the dispatcher is called outside the worker-pool
    submission path — tests, direct invocation), the cancel token
    is None."""
    rpc = _make_lone_runner()
    executor = _RecordingExecutor()
    _set_host_with_executor(rpc, executor)
    from server.runner.envelope import RequestEnvelope

    rpc._dispatch_method(
        RequestEnvelope(
            id=999, method="tool.execute",
            args={"name": "fake", "args": {}},
        ),
    )
    assert executor.calls[0]["has_cancel_token"] is False


# ----------------------------------------------------------------------
# Thread-local bridge
# ----------------------------------------------------------------------


def test_tool_execute_installs_callbacks_on_shared_thread_local() -> None:
    """The dispatcher installs the streaming callback + cancel token
    on ``shared.ai_tool_runner._thread_local`` so plugin code that
    reads from there (the in-process pattern) sees the runner's
    callbacks.  Restored on exit."""
    from shared.ai_tool_runner import _thread_local as _ai_tl

    rpc = _make_lone_runner()

    seen: Dict[str, Any] = {}

    def _capture_executor_execute(name, args, **kwargs):
        # Read the in-process thread-local mid-call.
        seen["cb"] = getattr(_ai_tl, "tool_output_callback", None)
        return True, {}

    fake_executor = SimpleNamespace(execute=_capture_executor_execute)
    fake_session = SimpleNamespace(_executor=fake_executor)
    rpc._session_host = RunnerSessionHost(
        envelope=_envelope(), runtime=None,
        session=fake_session,  # type: ignore[arg-type]
    )

    # Set a sentinel on the shared TL so we can verify it's restored.
    _ai_tl.tool_output_callback = "sentinel-prior"
    try:
        from server.runner.envelope import RequestEnvelope

        rpc._dispatch_method(
            RequestEnvelope(
                id=1, method="tool.execute",
                args={"name": "fake", "args": {}},
            ),
        )
        # During the call, the runner's own callback was installed.
        assert seen["cb"] is not None
        assert seen["cb"] != "sentinel-prior"
        # On return, the prior value was restored.
        assert _ai_tl.tool_output_callback == "sentinel-prior"
    finally:
        _ai_tl.tool_output_callback = None
