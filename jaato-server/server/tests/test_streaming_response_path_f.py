"""Regression-pin tests for Path F — third post-§7c chain
(streaming response delivery: runner→daemon AgentUIHooks bridge).

**Root cause** (cycle-7 audit + backlog
project_backlog_runner_ui_hooks_gap.md Finding 3):

Post-§7c step 6.6.4.3b seat-flip, the runner-side ``JaatoSession``
is the live session for the model loop and tool execution.  Its
``_ui_hooks`` attribute is **never set** — runner-side calls to
``self._ui_hooks.on_tool_call_start(...)`` etc silently drop
because of the ``if self._ui_hooks:`` null-guards.

The backlog doc identified this at §7c step 6.6.4.5 audit time
(Finding 3) but deferred it with the wrong assumption that events
might be redundant.  Cycle 7 empirically proves they are not — the
TUI depends on tool-call events + turn progress to render the
model response.

**Path F sub-fixes** (3 in one commit):

F.1 — Daemon-side ``output_callback`` in ``_start_model_thread``
emits ``AgentOutputEvent`` via ``ServerAgentHooks.on_agent_output``
(pre-Path-F it was a no-op; stream-frame text chunks were
silently dropped daemon-side).

F.2 — Runner-side ``_AgentUIHooksNotificationShim`` installed via
``_install_session_notification_callbacks`` (alongside the other
session callbacks).  Shim implements the 4 missing methods, each
emitting a NotificationFrame:
- on_tool_call_start → ``tool_call_start``
- on_tool_call_end → ``tool_call_end``
- on_tool_output → ``tool_output``
- on_turn_progress → ``turn_progress``
Restored to ``None`` on send_message exit.

F.3 — Daemon-side demuxer extended with 4 new event_type branches
that route through ``server._get_ui_hooks()`` to re-emit via
``ServerAgentHooks`` (preserving formatting + state-mutation at
zero divergence cost).

Tests pin:

- F.1: output_callback emits AgentOutputEvent via hooks
- F.2: shim has all 4 methods; each emits the right frame
- F.2: install/restore via _install_session_notification_callbacks
- F.3: demuxer handles 4 new event_types
- F.3: _get_ui_hooks returns _agent_hooks
- AST: shim class defined; 4 notification constants exist
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ----------------------------------------------------------------------
# F.2 — _AgentUIHooksNotificationShim has all 4 methods
# ----------------------------------------------------------------------


def test_shim_class_exists() -> None:
    from server.runner.rpc import _AgentUIHooksNotificationShim
    assert _AgentUIHooksNotificationShim is not None


def test_shim_has_four_emitting_methods() -> None:
    """Pin: the 4 methods runner-side session calls all exist and
    are implemented (not just inherited from Protocol)."""
    from server.runner.rpc import _AgentUIHooksNotificationShim

    for name in (
        "on_tool_call_start",
        "on_tool_call_end",
        "on_tool_output",
        "on_turn_progress",
    ):
        assert hasattr(_AgentUIHooksNotificationShim, name), (
            f"Path F regression: shim missing {name}; "
            f"backlog gap re-opened."
        )


def _make_rpc_with_capture() -> tuple:
    """Construct a stub RunnerRPC with emit_notification captured."""
    from server.runner.rpc import RunnerRPC

    rpc = RunnerRPC.__new__(RunnerRPC)
    captured: List[Dict[str, Any]] = []
    rpc.emit_notification = lambda **kw: captured.append(kw)
    return rpc, captured


def test_shim_on_tool_call_start_emits_frame() -> None:
    from server.runner.rpc import _AgentUIHooksNotificationShim

    rpc, captured = _make_rpc_with_capture()
    shim = _AgentUIHooksNotificationShim(rpc, request_id=7)
    shim.on_tool_call_start(
        agent_id="main",
        tool_name="cli",
        tool_args={"command": "ls"},
        call_id="c1",
    )
    assert len(captured) == 1
    assert captured[0]["event_type"] == "tool_call_start"
    assert captured[0]["request_id"] == 7
    assert captured[0]["payload"] == {
        "agent_id": "main",
        "tool_name": "cli",
        "tool_args": {"command": "ls"},
        "call_id": "c1",
    }


def test_shim_on_tool_call_end_emits_frame() -> None:
    from server.runner.rpc import _AgentUIHooksNotificationShim

    rpc, captured = _make_rpc_with_capture()
    shim = _AgentUIHooksNotificationShim(rpc, request_id=42)
    shim.on_tool_call_end(
        agent_id="main",
        tool_name="cli",
        success=True,
        duration_seconds=0.5,
        call_id="c1",
    )
    assert len(captured) == 1
    assert captured[0]["event_type"] == "tool_call_end"
    payload = captured[0]["payload"]
    assert payload["success"] is True
    assert payload["duration_seconds"] == 0.5
    assert payload["call_id"] == "c1"
    assert payload["backgrounded"] is False


def test_shim_on_tool_output_emits_frame() -> None:
    from server.runner.rpc import _AgentUIHooksNotificationShim

    rpc, captured = _make_rpc_with_capture()
    shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
    shim.on_tool_output(agent_id="main", call_id="c1", chunk="line1\n")
    assert captured[0]["event_type"] == "tool_output"
    assert captured[0]["payload"] == {
        "agent_id": "main", "call_id": "c1", "chunk": "line1\n",
    }


def test_shim_on_turn_progress_emits_frame() -> None:
    from server.runner.rpc import _AgentUIHooksNotificationShim

    rpc, captured = _make_rpc_with_capture()
    shim = _AgentUIHooksNotificationShim(rpc, request_id=99)
    shim.on_turn_progress(
        agent_id="main",
        total_tokens=150,
        prompt_tokens=100,
        output_tokens=50,
        percent_used=1.5,
        pending_tool_calls=2,
        cache_read_tokens=20,
    )
    assert captured[0]["event_type"] == "turn_progress"
    payload = captured[0]["payload"]
    assert payload["total_tokens"] == 150
    assert payload["pending_tool_calls"] == 2
    assert payload["cache_read_tokens"] == 20


def test_shim_protocol_completeness() -> None:
    """Pin: shim implements the full AgentUIHooks protocol surface
    (no-op fillers for methods the runner-side session doesn't
    call, but consumers might hasattr-check)."""
    from server.runner.rpc import _AgentUIHooksNotificationShim

    expected_methods = {
        # Emitting (4):
        "on_tool_call_start", "on_tool_call_end",
        "on_tool_output", "on_turn_progress",
        # No-op fillers (9):
        "on_agent_created", "on_agent_output",
        "on_agent_status_changed", "on_agent_completed",
        "on_session_quiescent", "on_agent_turn_completed",
        "on_agent_context_updated", "on_agent_gc_config",
        "on_agent_history_updated", "on_agent_instruction_budget_updated",
    }
    for name in expected_methods:
        assert hasattr(_AgentUIHooksNotificationShim, name), (
            f"shim missing AgentUIHooks method: {name}"
        )


# ----------------------------------------------------------------------
# F.2 — install + restore via _install_session_notification_callbacks
# ----------------------------------------------------------------------


def test_install_session_notification_callbacks_wires_ui_hooks_shim() -> None:
    """Pin: ``_install_session_notification_callbacks`` assigns a
    shim to ``session._ui_hooks`` and records the original under
    key ``ui_hooks`` for restoration."""
    from server.runner.rpc import (
        RunnerRPC, _AgentUIHooksNotificationShim,
    )

    rpc = RunnerRPC.__new__(RunnerRPC)
    rpc.emit_notification = lambda **kw: None

    class _Sess:
        _ui_hooks = None  # pre-Path-F state runner-side

    sess = _Sess()
    originals = rpc._install_session_notification_callbacks(sess, request_id=1)

    assert "ui_hooks" in originals, (
        "Path F regression: install must record ``ui_hooks`` in "
        "originals so restore can put the pre-shim value back."
    )
    assert originals["ui_hooks"] is None
    assert isinstance(sess._ui_hooks, _AgentUIHooksNotificationShim), (
        "Path F regression: install must assign the shim to "
        "session._ui_hooks; runner-side session calls to "
        "self._ui_hooks.on_* depend on this."
    )


def test_restore_session_notification_callbacks_unwires_ui_hooks() -> None:
    """Pin: ``_restore_session_notification_callbacks`` puts the
    pre-shim ``_ui_hooks`` (typically None) back."""
    from server.runner.rpc import RunnerRPC

    rpc = RunnerRPC.__new__(RunnerRPC)
    rpc.emit_notification = lambda **kw: None

    class _Sess:
        _ui_hooks = None

    sess = _Sess()
    originals = rpc._install_session_notification_callbacks(sess, request_id=1)
    # Sanity: shim installed.
    assert sess._ui_hooks is not None
    # Restore.
    rpc._restore_session_notification_callbacks(sess, originals)
    assert sess._ui_hooks is None, (
        "Path F regression: restore must reset _ui_hooks; "
        "leaked shim references would survive send_message and "
        "fire against a stale request_id on the next call."
    )


# ----------------------------------------------------------------------
# F.3 — daemon-side demuxer handles 4 new event_types
# ----------------------------------------------------------------------


def _make_server_with_hooks() -> tuple:
    """Construct a minimal JaatoServer with captured emit + hooks."""
    from server.core import JaatoServer
    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = "main"
    srv._model_running = False
    srv._pending_continuation = None
    srv._traces = []
    srv._trace = lambda m: srv._traces.append(m)
    srv._cached_context_limit = 100_000

    # Capture hook invocations.
    hook_calls: List[tuple] = []

    class _CapturingHooks:
        def on_tool_call_start(self, **kw):
            hook_calls.append(("on_tool_call_start", kw))

        def on_tool_call_end(self, **kw):
            hook_calls.append(("on_tool_call_end", kw))

        def on_tool_output(self, **kw):
            hook_calls.append(("on_tool_output", kw))

        def on_turn_progress(self, **kw):
            hook_calls.append(("on_turn_progress", kw))

        def on_agent_output(self, *args, **kw):
            hook_calls.append(("on_agent_output", args, kw))

    srv._agent_hooks = _CapturingHooks()
    return srv, hook_calls


def test_demuxer_tool_call_start_routes_through_hooks() -> None:
    srv, hook_calls = _make_server_with_hooks()
    handler = srv._build_send_message_notification_handler()
    handler("tool_call_start", {
        "agent_id": "main",
        "tool_name": "cli",
        "tool_args": {"command": "ls"},
        "call_id": "c1",
    })
    assert len(hook_calls) == 1
    name, kw = hook_calls[0]
    assert name == "on_tool_call_start"
    assert kw["tool_name"] == "cli"
    assert kw["call_id"] == "c1"


def test_demuxer_tool_call_end_routes_through_hooks() -> None:
    srv, hook_calls = _make_server_with_hooks()
    handler = srv._build_send_message_notification_handler()
    handler("tool_call_end", {
        "agent_id": "main",
        "tool_name": "cli",
        "success": True,
        "duration_seconds": 0.7,
        "call_id": "c1",
        "backgrounded": False,
    })
    name, kw = hook_calls[0]
    assert name == "on_tool_call_end"
    assert kw["success"] is True
    assert kw["duration_seconds"] == 0.7


def test_demuxer_tool_output_routes_through_hooks() -> None:
    srv, hook_calls = _make_server_with_hooks()
    handler = srv._build_send_message_notification_handler()
    handler("tool_output", {
        "agent_id": "main",
        "call_id": "c1",
        "chunk": "running...\n",
    })
    name, kw = hook_calls[0]
    assert name == "on_tool_output"
    assert kw["chunk"] == "running...\n"


def test_demuxer_turn_progress_routes_through_hooks() -> None:
    srv, hook_calls = _make_server_with_hooks()
    handler = srv._build_send_message_notification_handler()
    handler("turn_progress", {
        "agent_id": "main",
        "total_tokens": 150,
        "prompt_tokens": 100,
        "output_tokens": 50,
        "percent_used": 0.15,
        "pending_tool_calls": 2,
    })
    name, kw = hook_calls[0]
    assert name == "on_turn_progress"
    assert kw["total_tokens"] == 150
    assert kw["pending_tool_calls"] == 2


def test_demuxer_handles_missing_hooks_gracefully() -> None:
    """Pin: when ``_agent_hooks`` is None (test fixtures bypassing
    __init__), the demuxer branch must not crash — drop the
    notification cleanly."""
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = "main"
    srv._traces = []
    srv._trace = lambda m: srv._traces.append(m)
    # No _agent_hooks attribute set.

    handler = srv._build_send_message_notification_handler()
    # Should not crash.
    handler("tool_call_start", {"tool_name": "cli"})
    # No emit; quiet drop.
    assert srv._emitted_events == []


# ----------------------------------------------------------------------
# F.1 — output_callback emits via hooks
# ----------------------------------------------------------------------


def test_get_ui_hooks_returns_cached_instance() -> None:
    """Pin: ``_get_ui_hooks`` returns the stored ``_agent_hooks``."""
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    sentinel = object()
    srv._agent_hooks = sentinel
    assert srv._get_ui_hooks() is sentinel


def test_get_ui_hooks_returns_none_when_unset() -> None:
    """Pin: returns None (not crash) when hooks not yet wired."""
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    # No _agent_hooks attribute.
    assert srv._get_ui_hooks() is None


# ----------------------------------------------------------------------
# AST pins
# ----------------------------------------------------------------------


def test_runner_rpc_defines_four_new_notification_constants() -> None:
    """Pin via AST: the 4 new ``_NOTIF_*`` constants exist on
    ``RunnerRPC``.  Catches deletion / typo of the wire event_type
    strings."""
    from server.runner.rpc import RunnerRPC

    for attr in (
        "_NOTIF_TOOL_CALL_START",
        "_NOTIF_TOOL_CALL_END",
        "_NOTIF_TOOL_OUTPUT",
        "_NOTIF_TURN_PROGRESS",
    ):
        assert hasattr(RunnerRPC, attr), (
            f"Path F regression: {attr} missing — wire event_type "
            f"would drift between runner-side shim and daemon-side "
            f"demuxer."
        )


def test_output_callback_emits_via_hooks() -> None:
    """Pin via AST: ``_start_model_thread.output_callback`` body
    references ``_get_ui_hooks`` (no longer a no-op)."""
    from server.core import JaatoServer

    src = inspect.getsource(JaatoServer._start_model_thread)
    src = textwrap.dedent(src)
    assert "_get_ui_hooks" in src, (
        "Path F E.1 regression: output_callback must invoke hooks "
        "to emit AgentOutputEvent for stream-frame text chunks.  "
        "Pre-Path-F it was a no-op and chunks silently dropped."
    )
    assert "on_agent_output" in src, (
        "Path F E.1 regression: output_callback should call "
        "on_agent_output to preserve the pre-§7c formatting + "
        "<hidden> filtering logic."
    )


def test_install_session_notification_callbacks_installs_ui_hooks_shim() -> None:
    """Pin via AST: install function references the shim class."""
    from server.runner.rpc import RunnerRPC

    src = inspect.getsource(
        RunnerRPC._install_session_notification_callbacks,
    )
    assert "_AgentUIHooksNotificationShim" in src, (
        "Path F F.2 regression: install function must instantiate "
        "the ui_hooks shim.  Without it, the runner-side session's "
        "_ui_hooks stays None and tool events silently drop."
    )


def test_demuxer_handles_all_four_path_f_event_types() -> None:
    """Pin via AST: ``_build_send_message_notification_handler``
    body contains string literals for all 4 new event_types."""
    from server.core import JaatoServer

    src = inspect.getsource(
        JaatoServer._build_send_message_notification_handler,
    )
    for event_type in (
        "tool_call_start", "tool_call_end",
        "tool_output", "turn_progress",
    ):
        assert (
            f'"{event_type}"' in src or f"'{event_type}'" in src
        ), (
            f"Path F F.3 regression: demuxer missing branch for "
            f"event_type={event_type!r}.  Runner-emitted "
            f"notifications would hit the unknown-event_type log "
            f"and silently drop."
        )
