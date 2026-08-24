"""Tests for the session.send_message RPC handler + wrapper
(Phase 3 §7b.2 — the big one).

Streams output chunks via the existing stream-frame channel
(same mechanism ``tool.execute`` uses); cancellation propagates
via the existing cancel-frame mechanism + an ``on_cancel`` hook
into the session's ``request_stop``.

Tests pin:

- happy path: handler invokes session.send_message with the
  prompt + on_output bridge; returns the response string
- streaming: chunks fed to session.send_message's on_output
  callback flow back through the stream-frame channel (verified
  via daemon-side wrapper end-to-end)
- cancellation: cancel-frame triggers session.request_stop via
  the on_cancel hook; handler returns stage="cancelled" error
- exception: send_message raising surfaces as stage="send"
  error
- arg validation: missing prompt rejected
- no_host / no_session error paths
- e2e via daemon-side wrapper: prompt + response round-trip
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient, RunnerCallError
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-sm-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in for JaatoSession exposing send_message + the
    request_stop/cancel surface the cancel-hook touches."""

    def __init__(self) -> None:
        self.received_prompts: List[str] = []
        self.stream_chunks: List[Tuple[str, str, str]] = []
        # Tests configure these to drive paths.
        self.response: str = "model response"
        self.raise_on_send: Optional[Exception] = None
        self.stop_called: bool = False

    def send_message(
        self,
        message: str,
        on_output: Any = None,
        on_usage_update: Any = None,
        on_gc_threshold: Any = None,
        # Mirror the production caller: the runner handler forwards every
        # callback kwarg unconditionally, so a stub missing one fails with a
        # TypeError that reads as a generic send-stage error.
        on_gc_phase: Any = None,
        **_kw: Any,
    ) -> str:
        self.received_prompts.append(message)
        if self.raise_on_send is not None:
            raise self.raise_on_send
        # Stream chunks if a callback is provided.
        if on_output is not None:
            for source, text, mode in self.stream_chunks:
                on_output(source, text, mode)
        return self.response

    def request_stop(self, reason: str = "") -> bool:
        self.stop_called = True
        return True


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _make_active_call(rpc: RunnerRPC, request_id: int) -> Any:
    """Install a fake _ActiveCall for *request_id* so the handler's
    cancel-token lookup finds something."""
    from server.runner.rpc import _ActiveCall
    from jaato_sdk.plugins.model_provider.types import CancelToken

    active = _ActiveCall(cancel_token=CancelToken())
    with rpc._active_lock:
        rpc._active_calls[request_id] = active
    return active


# ----------------------------------------------------------------------
# Direct handler tests
# ----------------------------------------------------------------------


def test_send_message_happy_path() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.response = "the model said hi"
    _install(rpc, session)

    ok, result = rpc._handle_session_send_message(
        {"prompt": "hello"},
        request_id=1,
    )
    assert ok is True
    assert result == {"response": "the model said hi"}
    assert session.received_prompts == ["hello"]


def test_send_message_rejects_missing_prompt() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_send_message({}, request_id=1)
    assert ok is False
    assert result["stage"] == "decode"


def test_send_message_rejects_non_string_prompt() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    for bad in [42, None, ["list"], {"dict": True}]:
        ok, result = rpc._handle_session_send_message(
            {"prompt": bad}, request_id=1,
        )
        assert ok is False, f"should reject prompt={bad!r}"
        assert result["stage"] == "decode"


def test_send_message_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_send_message(
        {"prompt": "hi"}, request_id=1,
    )
    assert ok is False
    assert result["stage"] == "no_host"


def test_send_message_exception_returns_send_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_send = RuntimeError("model API down")
    _install(rpc, session)

    ok, result = rpc._handle_session_send_message(
        {"prompt": "hi"}, request_id=1,
    )
    assert ok is False
    assert result["stage"] == "send"
    assert "model API down" in result["error"]


def test_send_message_cancelled_exception_returns_cancelled_stage() -> None:
    """``CancelledException`` is the typed cancel signal; handler
    must translate it to ``stage="cancelled"`` (matches
    tool.execute's cancel envelope shape)."""
    from jaato_sdk.plugins.model_provider.types import CancelledException

    rpc = _make_lone_runner()
    session = _FakeSession()
    session.raise_on_send = CancelledException("user-cancel")
    _install(rpc, session)

    ok, result = rpc._handle_session_send_message(
        {"prompt": "hi"}, request_id=1,
    )
    assert ok is False
    assert result["stage"] == "cancelled"


def test_send_message_non_string_response_coerced() -> None:
    """Defensive: if a custom session subclass returns non-str,
    the handler coerces to str rather than letting JSON encoder
    fail at the wire boundary."""
    rpc = _make_lone_runner()

    class _BadSession:
        def send_message(self, *args, **kwargs):
            return 42  # int, not str

        def request_stop(self, reason=""):
            return False

    _install(rpc, _BadSession())

    ok, result = rpc._handle_session_send_message(
        {"prompt": "hi"}, request_id=1,
    )
    assert ok is True
    assert result == {"response": "42"}


def test_send_message_none_response_coerced_to_empty_string() -> None:
    rpc = _make_lone_runner()

    class _NoneSession:
        def send_message(self, *args, **kwargs):
            return None

        def request_stop(self, reason=""):
            return False

    _install(rpc, _NoneSession())

    ok, result = rpc._handle_session_send_message(
        {"prompt": "hi"}, request_id=1,
    )
    assert ok is True
    assert result == {"response": ""}


# ----------------------------------------------------------------------
# Cancel-hook propagation
# ----------------------------------------------------------------------


def test_cancel_token_trip_calls_session_request_stop() -> None:
    """When the dispatcher's cancel token fires, the on_cancel
    hook must call session.request_stop so the model loop wakes
    up at the next turn boundary."""
    rpc = _make_lone_runner()
    session = _FakeSession()
    _install(rpc, session)
    active = _make_active_call(rpc, request_id=42)

    # Configure send_message to NOT call on_output; just exit
    # immediately (we're testing the cancel-hook setup, not the
    # body).  We trip the cancel AFTER send_message returns —
    # the on_cancel must have been registered.
    ok, _ = rpc._handle_session_send_message(
        {"prompt": "hi"}, request_id=42,
    )
    assert ok is True

    # Now trip the dispatcher's cancel token.
    active.cancel_token.cancel(reason="user-stop")

    # The on_cancel callback registered by the handler should
    # have called session.request_stop.
    assert session.stop_called is True


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------


def test_dispatch_routes_session_send_message() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession()
    session.response = "routed!"
    _install(rpc, session)

    env = RequestEnvelope(
        id=99, method="session.send_message",
        args={"prompt": "hi"},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"response": "routed!"}


# ----------------------------------------------------------------------
# E2E daemon-side wrapper: prompt → response round-trip
# ----------------------------------------------------------------------


class _Pair:
    def __init__(self, runner_rpc, runner_thread, daemon_client) -> None:
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair() -> _Pair:
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _no_executor(name, args):
        return False, {"error": "no executor"}

    runner_rpc = RunnerRPC(runner_sock, _no_executor)
    thread = threading.Thread(
        target=runner_rpc.serve, name="rpc-serve-test", daemon=True,
    )
    thread.start()

    daemon_client = RunnerRPCClient(daemon_sock, runner_pid=0)
    await daemon_client.start()
    return _Pair(runner_rpc, thread, daemon_client)


async def _teardown(pair: _Pair) -> None:
    pair.runner_rpc.shutdown()
    pair.runner_thread.join(timeout=2)
    pair.daemon_client._closed = True
    if pair.daemon_client._writer is not None:
        try:
            pair.daemon_client._writer.close()
            await pair.daemon_client._writer.wait_closed()
        except Exception:
            pass
    if pair.daemon_client._read_task is not None:
        pair.daemon_client._read_task.cancel()
        try:
            await pair.daemon_client._read_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_e2e_send_message_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.response = "round-trip response"
        _install(pair.runner_rpc, session)

        result = await pair.daemon_client.session_send_message("hi there")
        assert result == "round-trip response"
        assert session.received_prompts == ["hi there"]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_send_message_streams_output_chunks() -> None:
    """Streaming path: chunks fed to session.send_message's
    on_output callback flow back to the daemon-side wrapper's
    on_output callback via the stream-frame channel."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.response = "final"
        session.stream_chunks = [
            ("model", "chunk 1\n", "text"),
            ("model", "chunk 2\n", "text"),
            ("model", "chunk 3\n", "text"),
        ]
        _install(pair.runner_rpc, session)

        received: List[Tuple[str, str, Optional[str]]] = []

        def _on_output(source: str, text: str, mode: Any) -> None:
            received.append((source, text, mode))

        result = await pair.daemon_client.session_send_message(
            "stream me", on_output=_on_output,
        )
        assert result == "final"
        # All three chunks streamed back.
        assert len(received) == 3
        assert received[0][1] == "chunk 1\n"
        assert received[2][1] == "chunk 3\n"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_send_message_exception_raises_runner_call_error() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.raise_on_send = RuntimeError("provider boom")
        _install(pair.runner_rpc, session)

        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_send_message("hi")
        assert "provider boom" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# Post-turn AgentUIHooks forwarding (Path G, 2026-06-07)
# ----------------------------------------------------------------------
#
# Pins that the IPC ``session.send_message`` handler invokes the
# session's ``_ui_hooks`` (the runner-side notification shim) after
# ``session.send_message`` returns, which in turn emits the
# ``agent_turn_completed`` / ``agent_context_updated`` /
# ``agent_history_updated`` notifications.  Without this firing,
# root IPC sessions never emit ``TurnCompletedEvent`` and SDK clients
# subscribed to it hang.  Regression guard for the diagnosis in
# ``project_pr_turn_completed_event_*``.
#
# The IPC handler installs ``_AgentUIHooksNotificationShim`` on
# ``session._ui_hooks`` for the duration of the call (see
# ``_install_session_notification_callbacks``); our forwarding block
# fans calls out through that shim, which translates each into an
# ``emit_notification`` on the rpc.  Tests assert against the rpc's
# emission record because that IS the wire-level contract.


class _SpyRPC:
    """RunnerRPC stand-in that records emit_notification calls.
    Inherits the rest from a real RunnerRPC instance so
    ``_install_session_notification_callbacks`` /
    ``_handle_session_send_message`` work unchanged."""

    def __init__(self) -> None:
        self._inner = _make_lone_runner()
        self.emissions: List[Tuple[int, str, Dict[str, Any]]] = []
        # Replace the inner's emit_notification with our recorder.
        self._inner.emit_notification = (  # type: ignore[method-assign]
            self._record_notification
        )

    def _record_notification(
        self,
        *,
        request_id: int,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        self.emissions.append((request_id, event_type, dict(payload)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SessionWithAccessors(_FakeSession):
    """Extends the bare _FakeSession with the accessors the
    post-turn forwarding block reads.  ``_ui_hooks`` is overwritten
    by the IPC handler's install step; we don't touch it here."""

    def __init__(self) -> None:
        super().__init__()
        self._ui_hooks: Optional[Any] = None
        self._agent_id: str = "main"
        self._turn_accounting: List[Dict[str, Any]] = []
        self._context_usage: Dict[str, Any] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "turns": 0,
            "percent_used": 0,
        }
        self._history: List[Dict[str, Any]] = []

    def get_turn_accounting(self) -> List[Dict[str, Any]]:
        return self._turn_accounting

    def get_context_usage(self) -> Dict[str, Any]:
        return self._context_usage

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history


def _notifications_by_type(
    spy: _SpyRPC, event_type: str,
) -> List[Dict[str, Any]]:
    return [payload for _, et, payload in spy.emissions if et == event_type]


def test_send_message_fires_post_turn_notifications_in_order() -> None:
    """Pre-fix: root IPC sessions never invoked ``_ui_hooks.on_X`` after
    send_message returned (the daemon's pre-§7c-seat-flip
    ``JaatoClient.send_message`` wrapper was bypassed).  Pin the fix:
    the IPC handler MUST fire the three post-turn notifications in the
    same order JaatoClient.send_message did, with the same payload
    shape, by calling through the installed shim."""
    spy = _SpyRPC()
    session = _SessionWithAccessors()
    session.response = "hello"
    session._agent_id = "main"
    session._turn_accounting = [
        {
            "prompt": 1500,
            "output": 200,
            "total": 1700,
            "duration_seconds": 3.5,
            "function_calls": [
                {"name": "list_tools", "duration_seconds": 0.02},
            ],
            "cache_read": 1000,
            "cache_creation": 500,
        },
    ]
    session._context_usage = {
        "total_tokens": 1700,
        "prompt_tokens": 1500,
        "output_tokens": 200,
        "turns": 1,
        "percent_used": 12.5,
    }
    session._history = [{"role": "user", "content": "hi"}]
    _install(spy._inner, session)

    ok, result = spy._inner._handle_session_send_message(
        {"prompt": "hi"}, request_id=42,
    )
    assert ok is True
    assert result == {"response": "hello"}

    # All three notifications fired (alongside whatever streaming-time
    # notifications would have fired — none here since the fake session
    # doesn't drive on_usage_update etc.).
    event_order = [et for _, et, _ in spy.emissions]
    assert "agent_turn_completed" in event_order
    assert "agent_context_updated" in event_order
    assert "agent_history_updated" in event_order

    # Order: turn_completed precedes context_updated precedes
    # history_updated (mirrors JaatoClient.send_message wrapper).
    tc_idx = event_order.index("agent_turn_completed")
    cu_idx = event_order.index("agent_context_updated")
    hu_idx = event_order.index("agent_history_updated")
    assert tc_idx < cu_idx < hu_idx

    # turn_completed payload mirrors JaatoClient.send_message wrapper.
    tc_payloads = _notifications_by_type(spy, "agent_turn_completed")
    assert len(tc_payloads) == 1
    tc = tc_payloads[0]
    assert tc["agent_id"] == "main"
    assert tc["turn_number"] == 0
    assert tc["prompt_tokens"] == 1500
    assert tc["output_tokens"] == 200
    assert tc["total_tokens"] == 1700
    assert tc["duration_seconds"] == 3.5
    # function_calls is a list (the per-call timing entries) — type-bug
    # regression guard for the int-coercion that pre-fix dropped the
    # event for any turn with tool calls.
    assert tc["function_calls"] == [
        {"name": "list_tools", "duration_seconds": 0.02},
    ]
    assert tc["cache_read_tokens"] == 1000
    assert tc["cache_creation_tokens"] == 500

    cu_payloads = _notifications_by_type(spy, "agent_context_updated")
    assert len(cu_payloads) == 1
    cu = cu_payloads[0]
    assert cu["agent_id"] == "main"
    assert cu["total_tokens"] == 1700
    assert cu["turns"] == 1
    assert cu["percent_used"] == 12.5

    hu_payloads = _notifications_by_type(spy, "agent_history_updated")
    assert len(hu_payloads) == 1
    hu = hu_payloads[0]
    assert hu["agent_id"] == "main"
    assert hu["history"] == [{"role": "user", "content": "hi"}]


def test_send_message_post_turn_notifications_not_fired_on_cancellation() -> None:
    """Cancellation short-circuits before the post-turn forwarding
    block.  Pin that the notifications are NOT fired on the cancel
    path — a half-completed turn shouldn't emit TurnCompletedEvent."""
    from jaato_sdk.plugins.model_provider.types import CancelledException

    spy = _SpyRPC()
    session = _SessionWithAccessors()
    session.raise_on_send = CancelledException("user-cancel")
    _install(spy._inner, session)

    ok, result = spy._inner._handle_session_send_message(
        {"prompt": "hi"}, request_id=1,
    )
    assert ok is False
    assert result["stage"] == "cancelled"
    assert _notifications_by_type(spy, "agent_turn_completed") == []
    assert _notifications_by_type(spy, "agent_context_updated") == []
    assert _notifications_by_type(spy, "agent_history_updated") == []


def test_send_message_post_turn_notifications_not_fired_on_exception() -> None:
    """send_message raising a non-cancel exception (e.g. provider
    error) must NOT fire the post-turn notifications — there's no
    completed turn to report."""
    spy = _SpyRPC()
    session = _SessionWithAccessors()
    session.raise_on_send = RuntimeError("provider down")
    _install(spy._inner, session)

    ok, result = spy._inner._handle_session_send_message(
        {"prompt": "hi"}, request_id=1,
    )
    assert ok is False
    assert result["stage"] == "send"
    assert _notifications_by_type(spy, "agent_turn_completed") == []


def test_send_message_post_turn_forwarding_swallows_accessor_errors() -> None:
    """Forwarding is best-effort: a session accessor raising
    (``get_turn_accounting`` etc.) must not corrupt the send_message
    response.  Pins the try/except guard in the forwarding block."""
    spy = _SpyRPC()

    class _BrokenAccessorSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self._ui_hooks = None
            self._agent_id = "main"

        def get_turn_accounting(self) -> List[Dict[str, Any]]:
            raise RuntimeError("accounting broken")

    session = _BrokenAccessorSession()
    session.response = "the response"
    _install(spy._inner, session)

    ok, result = spy._inner._handle_session_send_message(
        {"prompt": "hi"}, request_id=1,
    )
    # send_message still succeeded — accessor failure is swallowed.
    assert ok is True
    assert result == {"response": "the response"}
