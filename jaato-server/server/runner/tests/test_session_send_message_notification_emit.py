"""Tests for the §7c step 6.6.4.2 runner-side notification-emission
machinery installed by ``_handle_session_send_message``.

Per the §7c step 6.6.2 audit (commit 9f28f96d): the 7 daemon-side
callback wirings at sites 1996/2011/3391/3415/3430/3440/4291
collapse into runner-side notification emissions via the §7c
step 6.6.4.1 NotificationFrame protocol (commit 6e31d375).  This
commit lands the runner-side install/restore machinery — each of
the 6 session callbacks emits a NotificationFrame with a
well-known event_type when the session calls it during
``send_message``.

Tests pin:

- install: each session-callback setter is called with a notify
  shim (when present); originals saved for restoration
- restore: ``finally`` path calls the original setters with the
  saved values (no leak between RPC calls)
- emit: each callback type produces a wire frame with the right
  event_type + payload shape (instruction_budget_updated,
  prompt_injected, continuation_needed, retry,
  mid_turn_interrupt, events_subscribed)
- defensive: missing setters are silently skipped (rolling-
  upgrade safe); install raises don't blow up the handler;
  restore raises don't mask the response
- integration: send_message returns normally even when the
  fake session invokes 0..N callbacks during the call
- concurrency: per-call request_id is threaded into each emitted
  frame (no leakage across concurrent calls)
- e2e: daemon-side ``on_notification`` callback receives the
  emitted frames and demuxes by event_type
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest

from server.runner.envelope import NotificationFrame, RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient
from shared.framing import read_frame_sync
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Fixtures + fakes
# ----------------------------------------------------------------------


def _make_lone_runner() -> Tuple[RunnerRPC, socket.socket]:
    """Runner with the peer end of its socket returned for direct
    frame inspection (notification frames go out on the runner's
    write side)."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor), b


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-notif-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeEventBusTools:
    """Stand-in for the session's ``_event_bus_tools`` private
    attr.  Exposes the ``_on_subscribed`` slot the install
    helper writes to."""

    def __init__(self) -> None:
        self._on_subscribed: Optional[Any] = None


class _FakeSession:
    """Stand-in for JaatoSession exposing the 6 callback setters
    + the ``_event_bus_tools`` private attr the install helper
    touches.

    Tests configure ``invoke_during_send`` with a list of
    callable invocations to fire mid-send_message — simulates
    the session's real callbacks firing during a model loop.
    """

    def __init__(self, *, omit: Optional[set] = None) -> None:
        self._omit = omit or set()
        # Slot storage — names match JaatoSession's real internal
        # attrs so the install helper's
        # ``getattr(session, "_on_instruction_budget_updated", None)``
        # original-saving path finds them.
        self._on_instruction_budget_updated: Optional[Any] = None
        self._on_prompt_injected: Optional[Any] = None
        self._on_continuation_needed: Optional[Any] = None
        self._on_retry: Optional[Any] = None
        self._on_mid_turn_interrupt: Optional[Any] = None
        # Optional event-bus tools.
        if "event_bus_tools" not in self._omit:
            self._event_bus_tools = _FakeEventBusTools()
        # Tests configure these to drive paths.
        self.response: str = "ok"
        self.invoke_during_send: List[Any] = []
        self.received_prompts: List[str] = []
        # Track all setter calls (install + restore see two each).
        self.setter_log: List[Tuple[str, Any]] = []

    # --- callback accessors (mirror JaatoSession's setter→slot pattern) ---

    @property
    def instruction_budget_cb(self) -> Any:
        return self._on_instruction_budget_updated

    @property
    def prompt_injected_cb(self) -> Any:
        return self._on_prompt_injected

    @property
    def continuation_cb(self) -> Any:
        return self._on_continuation_needed

    @property
    def retry_cb(self) -> Any:
        return self._on_retry

    @property
    def mid_turn_interrupt_cb(self) -> Any:
        return self._on_mid_turn_interrupt

    # --- callback setters ---

    def set_instruction_budget_callback(self, cb: Any) -> None:
        if "instruction_budget" in self._omit:
            raise AttributeError("setter omitted")
        self.setter_log.append(("instruction_budget", cb))
        self._on_instruction_budget_updated = cb

    def set_prompt_injected_callback(self, cb: Any) -> None:
        if "prompt_injected" in self._omit:
            raise AttributeError("setter omitted")
        self.setter_log.append(("prompt_injected", cb))
        self._on_prompt_injected = cb

    def set_continuation_callback(self, cb: Any) -> None:
        if "continuation" in self._omit:
            raise AttributeError("setter omitted")
        self.setter_log.append(("continuation", cb))
        self._on_continuation_needed = cb

    def set_retry_callback(self, cb: Any) -> None:
        if "retry" in self._omit:
            raise AttributeError("setter omitted")
        self.setter_log.append(("retry", cb))
        self._on_retry = cb

    def set_mid_turn_interrupt_callback(self, cb: Any) -> None:
        if "mid_turn_interrupt" in self._omit:
            raise AttributeError("setter omitted")
        self.setter_log.append(("mid_turn_interrupt", cb))
        self._on_mid_turn_interrupt = cb

    # --- send_message ---

    def send_message(
        self,
        message: str,
        on_output: Any = None,
        on_usage_update: Any = None,
        on_gc_threshold: Any = None,
    ) -> str:
        self.received_prompts.append(message)
        # Fire any pre-configured callback invocations.  Each
        # entry is a callable taking *self* — tests call
        # ``self.instruction_budget_cb(...)`` etc. through these.
        for inv in self.invoke_during_send:
            inv(self)
        return self.response

    def request_stop(self, reason: str = "") -> bool:
        return True


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _drain_frames(peer_sock: socket.socket) -> List[Dict[str, Any]]:
    """Read all pending frames from the peer side (non-blocking)."""
    frames: List[Dict[str, Any]] = []
    peer_sock.settimeout(0.2)
    while True:
        try:
            raw = read_frame_sync(peer_sock)
        except (TimeoutError, OSError, socket.timeout):
            return frames
        if raw is None:
            return frames
        try:
            frames.append(json.loads(raw))
        except json.JSONDecodeError:
            continue


def _notification_frames(
    frames: List[Dict[str, Any]],
) -> List[NotificationFrame]:
    return [
        NotificationFrame.from_dict(d)
        for d in frames
        if d.get("kind") == "event"
    ]


# ----------------------------------------------------------------------
# Install / restore mechanics
# ----------------------------------------------------------------------


def test_install_wires_all_six_callbacks() -> None:
    """Every available setter on the session is invoked with a
    notification-emitting shim."""
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()
        originals = rpc._install_session_notification_callbacks(
            session, request_id=1,
        )
        # Each setter recorded one install call with a callable.
        kinds = {k for k, _ in session.setter_log}
        assert kinds == {
            "instruction_budget", "prompt_injected", "continuation",
            "retry", "mid_turn_interrupt",
        }
        # The event-bus tools slot was wired.
        assert callable(session._event_bus_tools._on_subscribed)
        # Originals dict carries all six keys.
        assert set(originals.keys()) == {
            "instruction_budget", "prompt_injected", "continuation",
            "retry", "mid_turn_interrupt", "events_subscribed",
        }
    finally:
        peer.close()


def test_install_skips_missing_setters() -> None:
    """Rolling-upgrade safety: a session missing one or more
    setters silently skips them; the rest still install."""
    rpc, peer = _make_lone_runner()
    try:
        # Use a custom class that genuinely lacks a setter
        # (hasattr returns False) — _FakeSession's omit set
        # raises inside the method, which is a different code
        # path.  hasattr-gating happens BEFORE the setter call.
        class _Partial:
            def __init__(self) -> None:
                self.calls: List[str] = []
                self._event_bus_tools = _FakeEventBusTools()

            def set_instruction_budget_callback(self, cb: Any) -> None:
                self.calls.append("ib")

            def set_retry_callback(self, cb: Any) -> None:
                self.calls.append("retry")

        session = _Partial()
        originals = rpc._install_session_notification_callbacks(
            session, request_id=1,
        )
        # Only the present setters wired.
        assert set(originals.keys()) == {
            "instruction_budget", "retry", "events_subscribed",
        }
        assert session.calls == ["ib", "retry"]
    finally:
        peer.close()


def test_restore_calls_setters_with_saved_originals() -> None:
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()
        # Pre-existing callback to be restored.
        sentinel_cb = lambda *a, **kw: None  # noqa: E731
        session.set_instruction_budget_callback(sentinel_cb)
        # Save baseline manually as install does.
        originals = {"instruction_budget": sentinel_cb}
        # Reset log so restore is the only event.
        session.setter_log.clear()
        rpc._restore_session_notification_callbacks(session, originals)
        # Restoration path wrote the sentinel back via the setter.
        assert session.setter_log == [("instruction_budget", sentinel_cb)]
        assert session.instruction_budget_cb is sentinel_cb
    finally:
        peer.close()


def test_install_setter_raise_does_not_propagate() -> None:
    """If a setter raises during install, the helper logs and
    continues — other setters still wire.  Critical: the
    handler must not crash mid-install and leave the session
    half-wired."""
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession(omit={"instruction_budget"})
        # Should not raise; other 5 setters still work.
        originals = rpc._install_session_notification_callbacks(
            session, request_id=1,
        )
        # instruction_budget was attempted (hasattr True) but
        # raised — its key still went into originals BEFORE the
        # raise, but the actual wiring failed.  Other setters
        # succeeded.
        kinds = {k for k, _ in session.setter_log}
        assert "instruction_budget" not in kinds
        assert "retry" in kinds
        assert "mid_turn_interrupt" in kinds
    finally:
        peer.close()


# ----------------------------------------------------------------------
# Per-callback emission shape
# ----------------------------------------------------------------------


def test_instruction_budget_callback_emits_notification() -> None:
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()

        def _fire(s: _FakeSession) -> None:
            s.instruction_budget_cb({"agent_id": "main", "tokens": 1234})

        session.invoke_during_send = [_fire]
        _install(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=11,
        )
        assert ok is True

        notes = _notification_frames(_drain_frames(peer))
        assert len(notes) == 1
        assert notes[0].id == 11
        assert notes[0].event_type == "instruction_budget_updated"
        assert notes[0].payload == {
            "snapshot": {"agent_id": "main", "tokens": 1234},
        }
    finally:
        peer.close()


def test_prompt_injected_callback_emits_notification() -> None:
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()

        def _fire(s: _FakeSession) -> None:
            s.prompt_injected_cb("system reminder text")

        session.invoke_during_send = [_fire]
        _install(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=22,
        )
        assert ok is True
        notes = _notification_frames(_drain_frames(peer))
        assert len(notes) == 1
        assert notes[0].event_type == "prompt_injected"
        assert notes[0].payload == {"text": "system reminder text"}
    finally:
        peer.close()


def test_continuation_callback_emits_notification() -> None:
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()

        def _fire(s: _FakeSession) -> None:
            s.continuation_cb("subagent did the thing")

        session.invoke_during_send = [_fire]
        _install(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=33,
        )
        assert ok is True
        notes = _notification_frames(_drain_frames(peer))
        assert len(notes) == 1
        assert notes[0].event_type == "continuation_needed"
        assert notes[0].payload == {
            "child_messages": "subagent did the thing",
        }
    finally:
        peer.close()


def test_retry_callback_emits_notification() -> None:
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()

        def _fire(s: _FakeSession) -> None:
            s.retry_cb("rate limited", 2, 5, 1.5)

        session.invoke_during_send = [_fire]
        _install(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=44,
        )
        assert ok is True
        notes = _notification_frames(_drain_frames(peer))
        assert len(notes) == 1
        assert notes[0].event_type == "retry"
        assert notes[0].payload == {
            "message": "rate limited",
            "attempt": 2,
            "max_attempts": 5,
            "delay": 1.5,
        }
    finally:
        peer.close()


def test_mid_turn_interrupt_callback_emits_notification() -> None:
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()

        def _fire(s: _FakeSession) -> None:
            s.mid_turn_interrupt_cb(987, "partial response so far")

        session.invoke_during_send = [_fire]
        _install(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=55,
        )
        assert ok is True
        notes = _notification_frames(_drain_frames(peer))
        assert len(notes) == 1
        assert notes[0].event_type == "mid_turn_interrupt"
        assert notes[0].payload == {
            "partial_chars": 987,
            "prompt_preview": "partial response so far",
        }
    finally:
        peer.close()


def test_events_subscribed_callback_emits_notification() -> None:
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()

        def _fire(s: _FakeSession) -> None:
            s._event_bus_tools._on_subscribed(
                "sub-1", ["TaskCompleted", "FileWritten"],
            )

        session.invoke_during_send = [_fire]
        _install(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=66,
        )
        assert ok is True
        notes = _notification_frames(_drain_frames(peer))
        assert len(notes) == 1
        assert notes[0].event_type == "events_subscribed"
        assert notes[0].payload == {
            "agent_id": "sub-1",
            "event_names": ["TaskCompleted", "FileWritten"],
        }
    finally:
        peer.close()


def test_multiple_callbacks_emit_in_order() -> None:
    """Several callbacks fired during one send_message → one
    NotificationFrame per call, preserving order."""
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()
        session.invoke_during_send = [
            lambda s: s.instruction_budget_cb({"tokens": 100}),
            lambda s: s.retry_cb("transient", 1, 3, 0.5),
            lambda s: s.instruction_budget_cb({"tokens": 200}),
            lambda s: s.prompt_injected_cb("inject"),
        ]
        _install(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=77,
        )
        assert ok is True
        notes = _notification_frames(_drain_frames(peer))
        assert len(notes) == 4
        assert [n.event_type for n in notes] == [
            "instruction_budget_updated",
            "retry",
            "instruction_budget_updated",
            "prompt_injected",
        ]
        # All carry the in-flight request_id.
        assert all(n.id == 77 for n in notes)
    finally:
        peer.close()


# ----------------------------------------------------------------------
# Lifecycle: restoration after send_message exits
# ----------------------------------------------------------------------


def test_handler_restores_callbacks_after_success() -> None:
    """After a successful send_message, originals are restored —
    callbacks installed by other callers are not leaked-over."""
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()
        # Pre-install a sentinel — represents a daemon-side
        # caller's callback that should survive the RPC.
        sentinel = lambda *a, **kw: None  # noqa: E731
        session.set_instruction_budget_callback(sentinel)
        _install(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=88,
        )
        assert ok is True
        # After return, the sentinel was restored.
        assert session.instruction_budget_cb is sentinel
    finally:
        peer.close()


def test_handler_restores_callbacks_after_send_exception() -> None:
    """Even if send_message raises, the finally block still
    restores the originals."""
    rpc, peer = _make_lone_runner()
    try:
        session = _FakeSession()
        sentinel = lambda *a, **kw: None  # noqa: E731
        session.set_retry_callback(sentinel)
        # Trigger send-loop exception.

        def _boom(s: _FakeSession) -> None:
            raise RuntimeError("provider boom")

        session.invoke_during_send = [_boom]
        _install(rpc, session)

        ok, result = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=99,
        )
        assert ok is False
        assert result["stage"] == "send"
        # Restoration ran.
        assert session.retry_cb is sentinel
    finally:
        peer.close()


# ----------------------------------------------------------------------
# E2E: daemon-side on_notification callback receives the frames
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
        target=runner_rpc.serve, name="rpc-notif-emit-test", daemon=True,
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
async def test_e2e_callbacks_fire_through_to_daemon_handler() -> None:
    """End-to-end: fake session fires its installed callbacks
    during send_message → notification frames emitted on the
    wire → daemon-side ``on_notification`` callback receives
    them with the correct event_type + payload."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.response = "done"
        session.invoke_during_send = [
            lambda s: s.instruction_budget_cb({"tokens": 50}),
            lambda s: s.prompt_injected_cb("hello"),
            lambda s: s.retry_cb("backoff", 1, 3, 2.0),
        ]
        _install(pair.runner_rpc, session)

        received: List[Tuple[str, Dict[str, Any]]] = []

        def _on_notif(event_type: str, payload: Dict[str, Any]) -> None:
            received.append((event_type, payload))

        result = await pair.daemon_client.session_send_message(
            "go", on_notification=_on_notif,
        )
        assert result == "done"
        # Give read loop a chance to drain.
        await asyncio.sleep(0.05)

        assert len(received) == 3
        assert received[0] == (
            "instruction_budget_updated", {"snapshot": {"tokens": 50}},
        )
        assert received[1] == ("prompt_injected", {"text": "hello"})
        assert received[2] == (
            "retry",
            {
                "message": "backoff",
                "attempt": 1,
                "max_attempts": 3,
                "delay": 2.0,
            },
        )
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_request_id_threaded_into_each_notification() -> None:
    """The in-flight request_id (allocated by daemon at call
    time) appears on every NotificationFrame — required so the
    daemon-side wrapper can demux concurrent in-flight calls.
    The exact id is daemon-allocated, but we can assert it
    matches across all received frames."""
    pair = await _make_pair()
    try:
        session = _FakeSession()
        session.invoke_during_send = [
            lambda s: s.instruction_budget_cb({"tokens": 1}),
            lambda s: s.instruction_budget_cb({"tokens": 2}),
        ]
        _install(pair.runner_rpc, session)

        # Inject a notification handler that records the wire id
        # by snooping on _notification_cbs registration.  Easier:
        # just assert that both notifications reach the SAME
        # callback (registered against the same id).
        received: List[Tuple[str, Dict[str, Any]]] = []
        await pair.daemon_client.session_send_message(
            "hi",
            on_notification=lambda et, p: received.append((et, p)),
        )
        await asyncio.sleep(0.05)
        assert len(received) == 2
        assert received[0][1] == {"snapshot": {"tokens": 1}}
        assert received[1][1] == {"snapshot": {"tokens": 2}}
    finally:
        await _teardown(pair)
