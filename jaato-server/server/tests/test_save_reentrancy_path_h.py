"""Regression-pin tests for Path H — session-save re-entrancy fix.

**Root cause** (cycle-10 probe-driven localization):

When a ``ToolCallStartEvent`` enters
``SessionManager._emit_to_session``, ``_handle_turn_tracking_event``
synchronously calls ``_save_session(session)``.  ``_save_session``
then makes 2 blocking runner-RPCs:
- ``session_get_history_threadsafe`` (line 3080)
- ``session_snapshot_conversation_budget_threadsafe`` (line 3158)

These race against the runner's still-active ``send_message``.
After 35s timeout the save fails silently AND the 35s delay
starves the model loop's permission-response window — the model
never receives the permission response, the tool call fails
silently, and the TUI sees nothing.

Architecturally same shape as Path E's Layer 5 race.

**Path H fix** (Option A — deferral):

H.1 ``SessionManager._save_session_async(session)`` helper
    spawns a daemon thread running ``_save_session(session)``.
H.2 Line 1611 changed from synchronous ``_save_session(session)``
    to ``_save_session_async(session)``.
H.3 ``self._async_save_lock`` (threading.Lock) serializes
    concurrent async saves for consistent last-writer-wins.

Tests pin:
- ``_save_session_async`` exists; spawns a daemon thread
- ``_handle_turn_tracking_event`` for ToolCallStartEvent uses
  async save (no synchronous _save_session call)
- The async path returns immediately (does NOT block on RPC)
- Concurrent invocations serialize via the lock
- Other handlers (ToolCallEndEvent, TurnCompletedEvent) do NOT
  trigger save (they only mark dirty)
- AST: ToolCallStartEvent branch references _save_session_async
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ----------------------------------------------------------------------
# H.1 — _save_session_async exists, spawns daemon thread
# ----------------------------------------------------------------------


def test_save_session_async_method_exists() -> None:
    """Pin: SessionManager defines _save_session_async."""
    from server.session_manager import SessionManager
    assert hasattr(SessionManager, "_save_session_async"), (
        "Path H regression: _save_session_async missing.  "
        "Synchronous _save_session is restored at line 1611, "
        "re-entrancy race returns."
    )


def test_save_session_async_spawns_daemon_thread() -> None:
    """Pin: _save_session_async starts a daemon thread + returns
    immediately."""
    from server.session_manager import SessionManager

    sm = SessionManager.__new__(SessionManager)
    sm._async_save_lock = threading.Lock()

    # Capture threads created by the async save.
    created_threads: List[threading.Thread] = []
    original_start = threading.Thread.start

    def _capture(self, *a, **kw):
        created_threads.append(self)
        original_start(self, *a, **kw)

    # Stub _save_session so we don't make real RPCs.
    save_done = threading.Event()

    def _fake_save(session):
        save_done.wait(timeout=2.0)
        return True

    sm._save_session = _fake_save

    session = MagicMock()
    session.session_id = "test-sess"

    with patch.object(threading.Thread, "start", _capture):
        t0 = time.time()
        sm._save_session_async(session)
        elapsed = time.time() - t0

    # Should return immediately (does not wait on save).
    assert elapsed < 0.1, (
        f"Path H regression: _save_session_async blocked for "
        f"{elapsed:.3f}s; expected <0.1s (returns immediately)."
    )
    assert len(created_threads) == 1, (
        "Path H regression: _save_session_async didn't create a "
        "background thread."
    )
    t = created_threads[0]
    assert t.daemon is True, (
        "Path H regression: async-save thread must be daemon=True "
        "so it doesn't block daemon shutdown."
    )
    save_done.set()
    t.join(timeout=2.0)


def test_save_session_async_serializes_saves_of_ONE_session() -> None:
    """Path H's ordering requirement, restated for a per-session guard.

    This used to pin ``_async_save_lock`` -- ONE GLOBAL lock, taken by
    ``_save_session_async`` around its call.  It asserted that saves of THREE
    DIFFERENT sessions ran one at a time, which was an accident of the lock's
    granularity and never a requirement: different sessions write different
    files and their relative order is meaningless.

    Meanwhile eight OTHER call sites reached ``_save_session`` with no lock at
    all, so the guard did not deliver the property even for one session -- an
    explicit ``session.save`` could interleave with an async one, both writing
    ``<id>.json.tmp``, and the loser got ENOENT.

    The guard now lives inside ``_save_session``, keyed on the session.  What
    Path H actually needs -- parallel ToolCallStartEvents, which all belong to
    ONE session, producing consistent last-writer-wins ordering -- still
    holds, and now holds for every caller.

    NOTE the stub below deliberately goes through ``session.save_lock``:
    replacing ``_save_session`` replaces the function that holds the guard, so
    a test that stubs it must take the lock the real one takes.  Serialization
    against the REAL ``_save_session`` is covered in
    ``test_save_serializes_per_session.py``.
    """
    import threading as _threading

    from server.session_manager import Session, SessionManager

    sm = SessionManager.__new__(SessionManager)

    save_order: List[str] = []
    order_lock = _threading.Lock()

    def _fake_save(session):
        with session.save_lock:          # the guard _save_session now holds
            with order_lock:
                save_order.append(f"start-{session.session_id}")
            time.sleep(0.05)
            with order_lock:
                save_order.append(f"end-{session.session_id}")
        return True

    sm._save_session = _fake_save

    # ONE session, three parallel saves -- the real Path H shape.
    session = Session(
        session_id="s0", name="s0",
        server=object(),  # type: ignore[arg-type]
        created_at="2026-08-26T00:00:00Z",
    )
    for _ in range(3):
        sm._save_session_async(session)

    for _ in range(40):
        if len(save_order) == 6:
            break
        time.sleep(0.05)

    assert save_order == ["start-s0", "end-s0"] * 3, (
        f"saves of ONE session interleaved: {save_order}.  Two writers in the "
        f"same window both target <id>.json.tmp; the loser gets ENOENT."
    )


def test_save_session_async_swallows_errors() -> None:
    """Pin: failures in the async save log a warning but don't
    propagate (best-effort semantics)."""
    from server.session_manager import SessionManager

    sm = SessionManager.__new__(SessionManager)
    sm._async_save_lock = threading.Lock()

    def _boom_save(session):
        raise RuntimeError("simulated save failure")

    sm._save_session = _boom_save

    session = MagicMock(session_id="boom")
    # Should not raise.
    sm._save_session_async(session)
    # Give the thread time to fail.
    time.sleep(0.1)
    # No assertion on log content — just that the call didn't crash
    # the caller.


# ----------------------------------------------------------------------
# H.2 — _handle_turn_tracking_event uses async save for ToolCallStartEvent
# ----------------------------------------------------------------------


def test_tool_call_start_uses_async_save() -> None:
    """Pin: ToolCallStartEvent branch calls _save_session_async
    (NOT synchronous _save_session)."""
    from server.session_manager import SessionManager
    from jaato_sdk.events import ToolCallStartEvent

    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._async_save_lock = threading.Lock()

    sync_calls: List[Any] = []
    async_calls: List[Any] = []
    sm._save_session = lambda s: sync_calls.append(s) or True
    sm._save_session_async = lambda s: async_calls.append(s)

    # Construct a session with active interrupted_turn for the right agent.
    session = MagicMock()
    session.session_id = "s1"
    session.interrupted_turn = {
        "agent_id": "main",
        "pending_tool_calls": [],
    }
    session.server = MagicMock()
    session.server.main_agent_id = "main"

    event = ToolCallStartEvent(
        agent_id="main",
        tool_name="cli",
        tool_args={"command": "ls"},
        call_id="c1",
    )

    sm._handle_turn_tracking_event(session, event)

    assert len(sync_calls) == 0, (
        "Path H regression: ToolCallStartEvent path still calls "
        "synchronous _save_session.  The 35s blocking-RPC race "
        "during active turn is back."
    )
    assert len(async_calls) == 1, (
        "Path H regression: ToolCallStartEvent path didn't call "
        "_save_session_async; the incremental crash-recovery save "
        "is missing."
    )
    assert async_calls[0] is session


def test_tool_call_end_does_not_save() -> None:
    """Pin: ToolCallEndEvent only marks dirty (no save).  Defensive
    pin: a hypothetical Path H regression that added save calls to
    end events would re-introduce the same race shape."""
    from server.session_manager import SessionManager
    from jaato_sdk.events import ToolCallEndEvent

    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._async_save_lock = threading.Lock()

    sync_calls: List[Any] = []
    async_calls: List[Any] = []
    sm._save_session = lambda s: sync_calls.append(s) or True
    sm._save_session_async = lambda s: async_calls.append(s)

    session = MagicMock()
    session.session_id = "s1"
    session.interrupted_turn = {
        "agent_id": "main",
        "pending_tool_calls": [{"id": "c1", "name": "cli", "args": {}}],
    }
    session.server = MagicMock()
    session.server.main_agent_id = "main"

    event = ToolCallEndEvent(
        agent_id="main",
        tool_name="cli",
        call_id="c1",
        success=True,
        duration_seconds=0.5,
    )
    sm._handle_turn_tracking_event(session, event)

    assert len(sync_calls) == 0
    assert len(async_calls) == 0


# ----------------------------------------------------------------------
# Behavioral pin: _emit_to_session does NOT block on runner-RPC for tool events
# ----------------------------------------------------------------------


def test_emit_to_session_does_not_block_on_runner_rpc() -> None:
    """Pin: ``_emit_to_session(ToolCallStartEvent)`` returns
    quickly even when the runner-RPC would hang.  Closes the
    cycle-10 35s starvation."""
    from server.session_manager import SessionManager, Session
    from jaato_sdk.events import ToolCallStartEvent

    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._async_save_lock = threading.Lock()
    sm._sessions = {}
    sm._workspace_monitors = {}
    # ``_emit_to_session`` fans out to cascade-driver clients
    # (``_dispatch_to_cascade_clients``, cascade-as-client Phase 2 / #180),
    # which takes this lock unconditionally.  A ``__new__``-built manager
    # must declare both halves or the emit path raises on an object
    # production would never produce.
    sm._cascade_clients = {}
    sm._cascade_clients_lock = threading.Lock()

    # Construct a real Session.
    session = MagicMock()
    session.session_id = "s1"
    session.attached_clients = set()
    session.interrupted_turn = {
        "agent_id": "main",
        "pending_tool_calls": [],
    }
    session.server = MagicMock()
    session.server.main_agent_id = "main"
    sm._sessions["s1"] = session

    # Simulate the cycle-10 race: _save_session blocks for 35s.
    def _hanging_save(_session):
        time.sleep(5.0)  # would be 35s in production
        return True

    sm._save_session = _hanging_save
    sm._handle_workspace_files_changed = lambda *a, **kw: None

    event = ToolCallStartEvent(
        agent_id="main",
        tool_name="cli",
        tool_args={"command": "ls"},
        call_id="c1",
    )

    t0 = time.time()
    sm._emit_to_session("s1", event)
    elapsed = time.time() - t0

    assert elapsed < 0.5, (
        f"Path H regression: _emit_to_session blocked for "
        f"{elapsed:.3f}s while async save was hanging.  Pre-Path-H "
        f"this was 35s, starving the permission-response window."
    )


# ----------------------------------------------------------------------
# AST pin: ToolCallStartEvent branch references _save_session_async
# ----------------------------------------------------------------------


def test_handle_turn_tracking_event_source_uses_async_save() -> None:
    """Pin via AST: ``_handle_turn_tracking_event`` body contains
    a call to ``self._save_session_async`` (not ``self._save_session``)
    in the ToolCallStartEvent path.

    Catches accidental revert of Path H (e.g. someone "cleaning up"
    by inlining the deferred save back to synchronous)."""
    from server.session_manager import SessionManager

    src = inspect.getsource(SessionManager._handle_turn_tracking_event)
    src = textwrap.dedent(src)

    # Async-call must be present (the fix).
    assert "_save_session_async" in src, (
        "Path H regression: _handle_turn_tracking_event body lacks "
        "a call to _save_session_async.  ToolCallStartEvent path "
        "must defer the save off the synchronous _emit_to_session "
        "path."
    )

    # Walk the AST to confirm the ToolCallStartEvent branch
    # specifically uses the async call.
    tree = ast.parse(src)
    async_save_in_tool_start = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Look for `isinstance(event, ToolCallStartEvent)` test.
            test_src = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if "ToolCallStartEvent" in test_src:
                body_src = "\n".join(
                    ast.unparse(stmt) if hasattr(ast, "unparse") else ""
                    for stmt in node.body
                )
                if "_save_session_async" in body_src:
                    async_save_in_tool_start = True
                    break
    assert async_save_in_tool_start, (
        "Path H regression: ToolCallStartEvent branch doesn't call "
        "_save_session_async.  The async deferral applies to other "
        "events but not the one that races; cycle-10 race returns."
    )
