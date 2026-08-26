"""Tests for §7c step 6.6.4.3b — atomic seat-flip:

- Runner-side per-call kwargs (``on_usage_update`` /
  ``on_gc_threshold``) wired as NotificationFrame-emitting
  shims (closes audit-caught 7→9 callback miss).
- Daemon-side ``_build_send_message_notification_handler``
  demuxer fans 8 event_types to ``server.emit(<Event>)`` /
  ``server._start_model_thread(...)``.
- ``session_send_message_threadsafe`` plumbs ``on_notification``
  through to the underlying async wrapper.
- ``_start_model_thread`` rewired to runner-RPC.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import (
    AgentStatusChangedEvent,
    ContextUpdatedEvent,
    EventsSubscribedEvent,
    InstructionBudgetEvent,
    MidTurnInterruptEvent,
    MidTurnPromptInjectedEvent,
    RetryEvent,
    SystemMessageEvent,
)
from server.core import JaatoServer
from server.runner.envelope import NotificationFrame
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from shared.framing import read_frame_sync
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Runner-side shim tests: on_usage_update + on_gc_threshold
# ----------------------------------------------------------------------


def _make_lone_runner() -> Tuple[RunnerRPC, socket.socket]:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor), b


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-643b-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeUsage:
    """Stand-in for TokenUsage dataclass — runner-side shim
    reads named attrs via getattr."""

    def __init__(
        self,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.reasoning_tokens = reasoning_tokens
        self.thinking_tokens = thinking_tokens
        self.cost_usd = cost_usd


class _ShimSession:
    """Stand-in that captures the kwargs send_message receives —
    pin the runner handler passes the new shims through."""

    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.kwargs_seen: Dict[str, Any] = {}
        self.fire_usage: Optional[_FakeUsage] = None
        self.fire_gc: Optional[Tuple[float, float]] = None
        self.fire_gc_phase: Optional[Tuple[str, Dict[str, Any]]] = None

    def send_message(
        self,
        message: str,
        on_output: Any = None,
        on_usage_update: Any = None,
        on_gc_threshold: Any = None,
        # GC LIFECYCLE events added ``on_gc_phase``; same story as
        # ``attachments`` below — the runner handler forwards it
        # unconditionally, so a stub missing it fails with a TypeError that
        # surfaces as a generic ``stage: send`` error rather than as "your
        # fixture is out of date".  Third instance of this class: a stub that
        # does not mirror the production caller's shape breaks on the NEXT
        # kwarg, every time.
        on_gc_phase: Any = None,
        # Multimodal v1 (#297-#300) added ``attachments`` to
        # ``JaatoSession.send_message``; the runner handler forwards it
        # unconditionally, so a shim without it fails the call with a
        # TypeError that surfaces as a generic ``stage: send`` error.
        attachments: Any = None,
    ) -> str:
        self.kwargs_seen = {
            "on_output": on_output,
            "on_usage_update": on_usage_update,
            "on_gc_threshold": on_gc_threshold,
            "on_gc_phase": on_gc_phase,
            "attachments": attachments,
        }
        if self.fire_usage is not None and on_usage_update is not None:
            on_usage_update(self.fire_usage)
        if self.fire_gc is not None and on_gc_threshold is not None:
            on_gc_threshold(*self.fire_gc)
        if self.fire_gc_phase is not None and on_gc_phase is not None:
            on_gc_phase(*self.fire_gc_phase)
        return self.response

    def request_stop(self, reason: str = "") -> bool:
        return True


def _install_shim(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


def _drain_notification_frames(
    peer_sock: socket.socket,
) -> List[NotificationFrame]:
    """Read all pending frames from the peer side; return only
    notification-kind ones decoded."""
    peer_sock.settimeout(0.2)
    frames: List[NotificationFrame] = []
    while True:
        try:
            raw = read_frame_sync(peer_sock)
        except (TimeoutError, OSError, socket.timeout):
            return frames
        if raw is None:
            return frames
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if d.get("kind") == "event":
            frames.append(NotificationFrame.from_dict(d))


def test_send_message_passes_usage_and_gc_shims_to_session() -> None:
    """Runner handler now passes ``on_usage_update`` +
    ``on_gc_threshold`` as kwargs to ``session.send_message``
    (closes audit Finding 2)."""
    rpc, peer = _make_lone_runner()
    try:
        session = _ShimSession()
        _install_shim(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=1,
        )
        assert ok is True
        # Both shims threaded through.
        assert callable(session.kwargs_seen["on_usage_update"])
        assert callable(session.kwargs_seen["on_gc_threshold"])
        assert callable(session.kwargs_seen["on_gc_phase"])
    finally:
        peer.close()


def test_usage_update_shim_emits_notification_frame() -> None:
    """Firing the shim with a TokenUsage stand-in produces a
    ``usage_update`` NotificationFrame with serialized fields."""
    rpc, peer = _make_lone_runner()
    try:
        session = _ShimSession()
        session.fire_usage = _FakeUsage(
            prompt_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_read_tokens=20,
            cache_creation_tokens=30,
            reasoning_tokens=5,
            thinking_tokens=10,
            cost_usd=0.0123,
        )
        _install_shim(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=42,
        )
        assert ok is True

        notes = _drain_notification_frames(peer)
        assert len(notes) == 1
        assert notes[0].id == 42
        assert notes[0].event_type == "usage_update"
        # Path E (cycle 6) E.1: shim now batches context_limit +
        # turns into the payload so the daemon-side handler doesn't
        # need in-band RPCs back into the runner.  The _ShimSession
        # stub doesn't expose ``get_context_limit`` /
        # ``_turn_accounting`` — falls back to defaults (0).
        assert notes[0].payload == {
            "prompt_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cache_read_tokens": 20,
            "cache_creation_tokens": 30,
            "reasoning_tokens": 5,
            "thinking_tokens": 10,
            "cost_usd": 0.0123,
            "context_limit": 0,
            "turns": 0,
        }
    finally:
        peer.close()


def test_usage_update_shim_handles_missing_optional_attrs() -> None:
    """Defensive: provider-side TokenUsage subclasses may not
    populate all optional fields — getattr keeps the shim
    resilient."""
    rpc, peer = _make_lone_runner()
    try:
        session = _ShimSession()
        # Bare-minimum usage (only required fields).
        session.fire_usage = _FakeUsage(
            prompt_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
        _install_shim(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=1,
        )
        assert ok is True
        notes = _drain_notification_frames(peer)
        assert len(notes) == 1
        assert notes[0].payload["total_tokens"] == 15
        # Optionals serialize as None.
        assert notes[0].payload["cache_read_tokens"] is None
        assert notes[0].payload["cost_usd"] is None
    finally:
        peer.close()


def test_gc_threshold_shim_emits_notification_frame() -> None:
    rpc, peer = _make_lone_runner()
    try:
        session = _ShimSession()
        session.fire_gc = (85.5, 80.0)
        _install_shim(rpc, session)

        ok, _ = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=99,
        )
        assert ok is True
        notes = _drain_notification_frames(peer)
        assert len(notes) == 1
        assert notes[0].id == 99
        assert notes[0].event_type == "gc_threshold"
        assert notes[0].payload == {
            "percent_used": 85.5,
            "threshold": 80.0,
        }
    finally:
        peer.close()


# ----------------------------------------------------------------------
# Daemon-side notification demuxer tests
# ----------------------------------------------------------------------


def _make_server_for_demuxer() -> JaatoServer:
    """Minimal JaatoServer instance for testing the
    notification demuxer.  Skips full ``__init__``; sets only the
    attributes the demuxer touches."""
    srv = JaatoServer.__new__(JaatoServer)
    srv._on_event = lambda e: None
    # The full emit() chain pulls in pseudonymization + plugin-
    # event aspects we don't need here; replace with a list-
    # appender so tests can read events directly.
    srv._emitted_events: List[Any] = []
    srv.emit = lambda e: srv._emitted_events.append(e)  # type: ignore[method-assign]
    srv._main_agent_id = "main"
    srv._model_running = False
    srv._pending_continuation = None
    # The stash is written from the RPC read loop and read on the model
    # thread, so production guards it; a double without the lock would
    # diverge from the shape the demuxer actually runs against.
    srv._pending_continuation_lock = threading.Lock()
    srv._traces: List[str] = []
    srv._trace = lambda msg: srv._traces.append(msg)  # type: ignore[method-assign]
    srv._start_model_thread_calls: List[str] = []
    srv._start_model_thread = (
        lambda prompt: srv._start_model_thread_calls.append(prompt)  # type: ignore[method-assign]
    )
    srv._jaato = MagicMock()
    srv._jaato.get_context_limit.return_value = 100_000
    srv._jaato.get_turn_accounting.return_value = [object(), object()]
    # Phase 3 §7c step 6.6.4.5b: usage_update demuxer branch now reads
    # via ``server._runner_rpc.session_get_*_threadsafe`` instead of
    # daemon-side ``server._jaato.get_*``.  Mock the runner-RPC handle
    # so the demuxer test exercises the post-migration path.
    srv._runner_rpc = MagicMock()
    srv._runner_rpc.session_get_context_limit_threadsafe.return_value = 100_000
    srv._runner_rpc.session_get_turn_accounting_threadsafe.return_value = (
        [object(), object()]
    )
    return srv


def test_demuxer_instruction_budget_updated_emits_event() -> None:
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("instruction_budget_updated", {
        "snapshot": {"agent_id": "main", "tokens": 1234},
    })

    assert len(srv._emitted_events) == 1
    evt = srv._emitted_events[0]
    assert isinstance(evt, InstructionBudgetEvent)
    assert evt.agent_id == "main"
    assert evt.budget_snapshot == {"agent_id": "main", "tokens": 1234}


def test_demuxer_prompt_injected_emits_event() -> None:
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("prompt_injected", {"text": "hello"})

    assert len(srv._emitted_events) == 1
    assert isinstance(srv._emitted_events[0], MidTurnPromptInjectedEvent)
    assert srv._emitted_events[0].text == "hello"


def test_demuxer_continuation_when_model_idle_starts_thread() -> None:
    srv = _make_server_for_demuxer()
    srv._model_running = False
    handler = srv._build_send_message_notification_handler()

    handler("continuation_needed", {"child_messages": "subagent done"})

    # AgentStatusChangedEvent emitted before recursive start.
    assert len(srv._emitted_events) == 1
    assert isinstance(srv._emitted_events[0], AgentStatusChangedEvent)
    assert srv._emitted_events[0].status == "active"
    # Recursive start_model_thread call.
    assert srv._start_model_thread_calls == ["subagent done"]
    assert "CONTINUATION" in srv._traces[0]


def test_demuxer_continuation_when_model_running_stashes() -> None:
    srv = _make_server_for_demuxer()
    srv._model_running = True
    handler = srv._build_send_message_notification_handler()

    handler("continuation_needed", {"child_messages": "stash me"})

    # Stashed for the model_thread finally block.
    assert srv._pending_continuation == "stash me"
    # No recursive start.
    assert srv._start_model_thread_calls == []
    # No event emitted (status change happens at finally-block boundary).
    assert srv._emitted_events == []


def test_demuxer_continuation_with_empty_text_is_noop() -> None:
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("continuation_needed", {"child_messages": ""})

    assert srv._emitted_events == []
    assert srv._start_model_thread_calls == []
    assert srv._pending_continuation is None


def test_demuxer_retry_emits_event_with_rate_limit_classification() -> None:
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("retry", {
        "message": "rate-limit hit; retrying",
        "attempt": 2,
        "max_attempts": 5,
        "delay": 1.5,
    })

    evt = srv._emitted_events[0]
    assert isinstance(evt, RetryEvent)
    assert evt.message == "rate-limit hit; retrying"
    assert evt.attempt == 2
    assert evt.max_attempts == 5
    assert evt.delay == 1.5
    assert evt.error_type == "rate_limit"


def test_demuxer_retry_classifies_other_messages_as_transient() -> None:
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("retry", {
        "message": "connection reset",
        "attempt": 1,
        "max_attempts": 3,
        "delay": 0.5,
    })

    assert srv._emitted_events[0].error_type == "transient"


def test_demuxer_mid_turn_interrupt_emits_event() -> None:
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("mid_turn_interrupt", {
        "partial_chars": 250,
        "prompt_preview": "user typed something interesting",
    })

    evt = srv._emitted_events[0]
    assert isinstance(evt, MidTurnInterruptEvent)
    assert evt.partial_response_chars == 250
    assert evt.user_prompt_preview == "user typed something interesting"


def test_demuxer_events_subscribed_emits_event() -> None:
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("events_subscribed", {
        "agent_id": "sub-1",
        "event_names": ["TaskCompleted", "FileWritten"],
    })

    evt = srv._emitted_events[0]
    assert isinstance(evt, EventsSubscribedEvent)
    assert evt.agent_id == "sub-1"
    assert evt.event_names == ["TaskCompleted", "FileWritten"]


def test_demuxer_usage_update_emits_context_event() -> None:
    srv = _make_server_for_demuxer()
    # Patch _build_usage to return a real UsageBreakdown
    # (ContextUpdatedEvent validates the field's type).
    from jaato_sdk.events import UsageBreakdown
    real_usage = UsageBreakdown(
        prompt_tokens=100, output_tokens=50, total_tokens=150,
    )
    srv._build_usage = MagicMock(return_value=real_usage)
    handler = srv._build_send_message_notification_handler()

    # Path E (cycle 6) E.1: handler now reads context_limit + turns
    # from the payload (runner-side shim batches them).  Pre-Path-E
    # they came from in-band ``session_get_*_threadsafe`` RPCs.
    handler("usage_update", {
        "prompt_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cache_read_tokens": 20,
        "cache_creation_tokens": None,
        "reasoning_tokens": None,
        "thinking_tokens": None,
        "cost_usd": 0.012,
        "context_limit": 100_000,
        "turns": 2,
    })

    assert len(srv._emitted_events) == 1
    evt = srv._emitted_events[0]
    assert isinstance(evt, ContextUpdatedEvent)
    assert evt.context_limit == 100_000
    # 150/100_000 * 100 = 0.15
    assert evt.percent_used == pytest.approx(0.15)
    assert evt.tokens_remaining == 99_850
    assert evt.turns == 2
    # _build_usage called with payload fields.
    srv._build_usage.assert_called_once_with(
        prompt_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cache_read_tokens=20,
        cache_creation_tokens=None,
        reasoning_tokens=None,
        thinking_tokens=None,
        cost_usd_override=0.012,
    )


def test_demuxer_usage_update_skips_zero_total_tokens() -> None:
    """Pre-§7c behavior: ``if usage.total_tokens == 0: return`` —
    preserved post-migration to avoid spurious ContextUpdatedEvents
    on warm-up calls that report no usage."""
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("usage_update", {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })

    assert srv._emitted_events == []


def test_demuxer_gc_threshold_emits_system_message() -> None:
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("gc_threshold", {"percent_used": 85.5, "threshold": 80.0})

    evt = srv._emitted_events[0]
    assert isinstance(evt, SystemMessageEvent)
    assert evt.style == "warning"
    assert "85.5%" in evt.message
    assert "80.0%" in evt.message


def test_demuxer_unknown_event_type_logged_dropped() -> None:
    """Forward-compat: unknown event types log + drop; don't
    crash the read loop."""
    srv = _make_server_for_demuxer()
    handler = srv._build_send_message_notification_handler()

    handler("future_event_type", {"something": "new"})

    assert srv._emitted_events == []
    assert any(
        "NOTIFICATION_UNKNOWN" in t and "future_event_type" in t
        for t in srv._traces
    )


def test_demuxer_branch_exception_caught() -> None:
    """If an emit/branch raises, the demuxer logs and continues —
    a single bad notification doesn't break the model loop."""
    srv = _make_server_for_demuxer()

    def _boom(_evt: Any) -> None:
        raise RuntimeError("emit boom")

    srv.emit = _boom  # type: ignore[method-assign]
    handler = srv._build_send_message_notification_handler()

    # Should not raise.
    handler("prompt_injected", {"text": "hello"})


# ----------------------------------------------------------------------
# Pin: _start_model_thread no longer wires the 7 daemon-side
# callbacks (regression guard for the seat-flip)
# ----------------------------------------------------------------------


def test_no_set_callback_wirings_remain_in_core() -> None:
    """Pin the 7 deleted ``set_*_callback`` wirings stay
    deleted — guards against accidental re-introduction.

    Counts CALL sites (with ``(`` after the method name); the
    setter-method definitions on JaatoSession itself live in
    ``shared/jaato_session.py`` and stay (they're the runner-
    side install-target for the §7c step 6.6.4.2 machinery)."""
    import server.core as core_mod
    # Read source of the module
    import inspect
    src = inspect.getsource(core_mod)
    # Each setter call site looks like ``session.set_*_callback(...``
    # — the deletions removed all five.  Comments referencing the
    # callback names are fine.
    forbidden = [
        "session.set_instruction_budget_callback(",
        "session.set_prompt_injected_callback(",
        "session.set_continuation_callback(",
        "session.set_retry_callback(",
        "session.set_mid_turn_interrupt_callback(",
        "session._event_bus_tools._on_subscribed =",
    ]
    for pattern in forbidden:
        assert pattern not in src, (
            f"§7c step 6.6.4.3b regression: {pattern!r} re-introduced "
            f"in server/core.py — should route through the runner-side "
            f"NotificationFrame protocol instead"
        )


def test_no_jaato_send_message_in_start_model_thread() -> None:
    """Pin _start_model_thread no longer calls
    ``server._jaato.send_message(...)``.  Both legs (initial +
    formatter-feedback) must use ``session_send_message_threadsafe``.

    Uses AST to walk Call nodes so the docstring's reference
    to the pre-migration call shape doesn't trigger a false
    positive."""
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(JaatoServer._start_model_thread))
    tree = ast.parse(src)

    def _is_jaato_send_message(call: ast.Call) -> bool:
        # Match ``<anything>._jaato.send_message(...)``.
        f = call.func
        if not isinstance(f, ast.Attribute) or f.attr != "send_message":
            return False
        inner = f.value
        return isinstance(inner, ast.Attribute) and inner.attr == "_jaato"

    bad = [n for n in ast.walk(tree)
           if isinstance(n, ast.Call) and _is_jaato_send_message(n)]
    assert not bad, (
        f"§7c step 6.6.4.3b regression: _start_model_thread still "
        f"calls _jaato.send_message ({len(bad)} sites) — must use "
        f"runner-RPC"
    )

    def _is_threadsafe_call(call: ast.Call) -> bool:
        f = call.func
        return (
            isinstance(f, ast.Attribute)
            and f.attr == "session_send_message_threadsafe"
        )

    threadsafe = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and _is_threadsafe_call(n)]
    assert len(threadsafe) >= 2, (
        "§7c step 6.6.4.3b regression: _start_model_thread should "
        "route both legs (initial + formatter-feedback) through "
        "session_send_message_threadsafe"
    )


# ----------------------------------------------------------------------
# session_send_message_threadsafe accepts on_notification
# ----------------------------------------------------------------------


def test_threadsafe_wrapper_accepts_on_notification() -> None:
    """Pin the daemon-side threadsafe wrapper plumbs
    ``on_notification`` through — required for the
    ``_start_model_thread`` rewire."""
    import inspect
    from server.runner_rpc_client import RunnerRPCClient
    sig = inspect.signature(RunnerRPCClient.session_send_message_threadsafe)
    assert "on_notification" in sig.parameters


def test_gc_phase_shim_emits_notification_frame() -> None:
    """Firing ``on_gc_phase`` produces a ``gc_phase`` NotificationFrame.

    The lifecycle counterpart to the ``gc_threshold`` frame above: that one
    carries only the crossing and is rendered as prose daemon-side, this one
    carries the whole pass as branchable values.
    """
    rpc, peer = _make_lone_runner()
    try:
        session = _ShimSession()
        session.fire_gc_phase = ("completed", {
            "trigger_reason": "threshold", "strategy": "gc_hybrid",
            "success": True, "items_collected": 4, "tokens_freed": 600,
        })
        _install_shim(rpc, session)

        ok, _ = rpc._handle_session_send_message({"prompt": "hi"}, request_id=1)
        assert ok is True

        frames = _drain_notification_frames(peer)
        gc_frames = [f for f in frames if f.event_type == "gc_phase"]
        assert len(gc_frames) == 1, (
            f"expected one gc_phase frame, got {[f.event_type for f in frames]}")
        payload = gc_frames[0].payload
        assert payload["phase"] == "completed"
        assert payload["items_collected"] == 4
        assert payload["tokens_freed"] == 600
    finally:
        peer.close()
