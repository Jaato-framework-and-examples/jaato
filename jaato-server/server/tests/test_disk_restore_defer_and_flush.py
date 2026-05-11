"""Tests for §3.12 disk-restore defer-and-flush infrastructure
(peer-review M5/N1).

Pre-§3.12: a session loaded from disk after a daemon restart had
no way to signal pending in-flight work to the next client.  Any
ASK landing pre-attach was denied (no client to prompt), and the
session's restoration was indistinguishable from a fresh attach.

Post-§3.12 (this commit): a Session record loaded from disk gets
``restored_pending_attach=True``.  The daemon emits
``SessionRestoredEvent(session_id, pending_tool_call_count)`` on
the first client-attach so the client can surface a "review
pending tool calls" prompt.  The flag clears at first attach so
subsequent ASKs revert to normal denial / prompt behaviour.

The actual defer-and-flush queue (held ASKs + drain through the
now-attached ``client.prompt_operator`` channel) lives in the
runner-side permission plugin and lands in a §3.12 follow-on
commit; this commit lays the daemon-side foundation.

Tests pin:

- ``Session.restored_pending_attach`` defaults to False on fresh
  sessions and is True for disk-restored ones.
- First client-attach to a restored session emits
  ``SessionRestoredEvent`` and clears the flag.
- Subsequent attaches do NOT re-emit the event.
- Detach + re-attach does NOT re-trigger restoration semantics
  (the flag is one-shot per restore).
- Fresh sessions never emit ``SessionRestoredEvent`` on attach.
- ``pending_tool_call_count`` placeholder returns 0 (the
  runner-side queue lands in a follow-on; the wire contract is
  stable).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

from jaato_sdk.events import EventType, SessionRestoredEvent
from server.session_manager import Session, SessionManager


# ----------------------------------------------------------------------
# Session dataclass
# ----------------------------------------------------------------------


def test_session_restored_pending_attach_defaults_false() -> None:
    """Fresh / never-restored sessions have the flag False.  The
    field exists for the dataclass shape; defer-and-flush only
    activates when explicitly set on the disk-restore path."""
    sess = Session(
        session_id="s-fresh",
        name="fresh",
        server=MagicMock(),
        created_at="2024-01-01T00:00:00",
    )
    assert sess.restored_pending_attach is False


def test_session_restored_pending_attach_settable() -> None:
    sess = Session(
        session_id="s-restored",
        name="restored",
        server=MagicMock(),
        created_at="2024-01-01T00:00:00",
        restored_pending_attach=True,
    )
    assert sess.restored_pending_attach is True


# ----------------------------------------------------------------------
# attach_session: first-attach to restored session emits + clears
# ----------------------------------------------------------------------


def _make_session_manager_with_restored_session() -> Tuple[SessionManager, Session, List[Tuple[str, Any]]]:
    """Build a SessionManager with a pre-loaded restored session
    and a captured ``_emit_to_client`` so tests can assert on the
    emitted events."""
    sm = SessionManager()
    server_mock = MagicMock()
    server_mock.emit_current_state = MagicMock()
    server_mock.workspace_path = None
    sess = Session(
        session_id="s-restored-1",
        name="restored",
        server=server_mock,
        created_at="2024-01-01T00:00:00",
        workspace_path="/tmp/ws",
        restored_pending_attach=True,
    )
    sm._sessions["s-restored-1"] = sess

    emitted: List[Tuple[str, Any]] = []
    sm._emit_to_client = lambda cid, ev: emitted.append((cid, ev))  # type: ignore[method-assign]
    sm._build_session_info_event = lambda s: MagicMock()  # type: ignore[method-assign]
    sm._apply_client_config_to_server = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return sm, sess, emitted


def test_first_attach_to_restored_session_emits_restored_event() -> None:
    sm, sess, emitted = _make_session_manager_with_restored_session()

    ok = sm.attach_session(
        client_id="c-1",
        session_id="s-restored-1",
    )
    assert ok is True

    # SessionRestoredEvent emitted to the attaching client.
    restored_events = [
        ev for cid, ev in emitted
        if isinstance(ev, SessionRestoredEvent) and cid == "c-1"
    ]
    assert len(restored_events) == 1
    event = restored_events[0]
    assert event.session_id == "s-restored-1"
    assert event.type == EventType.SESSION_RESTORED
    # pending_tool_call_count is 0 for this commit (queue
    # infrastructure ships in §3.12 follow-on).
    assert event.pending_tool_call_count == 0


def test_first_attach_clears_restored_pending_attach_flag() -> None:
    sm, sess, _ = _make_session_manager_with_restored_session()

    sm.attach_session(client_id="c-1", session_id="s-restored-1")
    assert sess.restored_pending_attach is False


def test_second_attach_does_not_re_emit_restored_event() -> None:
    """A second client attaching to the same session (now with the
    first client still attached) must NOT see SessionRestoredEvent —
    it's a one-shot per restore."""
    sm, _, emitted = _make_session_manager_with_restored_session()

    sm.attach_session(client_id="c-1", session_id="s-restored-1")
    emitted.clear()  # focus on the second attach

    sm.attach_session(client_id="c-2", session_id="s-restored-1")

    restored_events = [
        ev for _, ev in emitted if isinstance(ev, SessionRestoredEvent)
    ]
    assert restored_events == []


def test_detach_reattach_does_not_re_trigger_restoration() -> None:
    """Even if the only client detaches and re-attaches, the
    SessionRestoredEvent must not fire again — once flushed, the
    session is "live" and behaves like any other in-memory
    session."""
    sm, sess, emitted = _make_session_manager_with_restored_session()

    sm.attach_session(client_id="c-1", session_id="s-restored-1")
    # Simulate detach by removing the client manually.
    sess.attached_clients.discard("c-1")
    sm._client_to_session.pop("c-1", None)
    emitted.clear()

    sm.attach_session(client_id="c-1", session_id="s-restored-1")

    restored_events = [
        ev for _, ev in emitted if isinstance(ev, SessionRestoredEvent)
    ]
    assert restored_events == []


def test_attach_to_fresh_session_does_not_emit_restored_event() -> None:
    """Sessions created fresh (not loaded from disk) must not emit
    SessionRestoredEvent on attach — they were never "restored"."""
    sm = SessionManager()
    server_mock = MagicMock()
    server_mock.emit_current_state = MagicMock()
    server_mock.workspace_path = None
    sess = Session(
        session_id="s-fresh-1",
        name="fresh",
        server=server_mock,
        created_at="2024-01-01T00:00:00",
        workspace_path="/tmp/ws",
        restored_pending_attach=False,  # fresh session
    )
    sm._sessions["s-fresh-1"] = sess

    emitted: List[Tuple[str, Any]] = []
    sm._emit_to_client = lambda cid, ev: emitted.append((cid, ev))  # type: ignore[method-assign]
    sm._build_session_info_event = lambda s: MagicMock()  # type: ignore[method-assign]
    sm._apply_client_config_to_server = lambda *args, **kwargs: None  # type: ignore[method-assign]

    sm.attach_session(client_id="c-1", session_id="s-fresh-1")

    restored_events = [
        ev for _, ev in emitted if isinstance(ev, SessionRestoredEvent)
    ]
    assert restored_events == []


# ----------------------------------------------------------------------
# Pending-count helper
# ----------------------------------------------------------------------


def test_count_pending_held_tool_calls_placeholder_returns_zero() -> None:
    """The runner-side queue infrastructure ships in a §3.12
    follow-on; for this commit the helper returns 0.  The event
    field is wired end-to-end so the client-side contract is
    stable."""
    sm = SessionManager()
    sess = Session(
        session_id="s-count",
        name="x",
        server=MagicMock(),
        created_at="2024-01-01T00:00:00",
        restored_pending_attach=True,
    )
    assert sm._count_pending_held_tool_calls(sess) == 0
