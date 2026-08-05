"""Phase 2 cascade-as-client — server-side IPC RPC verbs +
disconnect cleanup.

Pins the command_router handlers
(_handle_cascade_register / _handle_cascade_unregister) and the
disconnect-cleanup helper
(SessionManager.unregister_all_cascade_clients_for_connection).

SDK-side IPCClient.cascade_events tests live in
``jaato-sdk/jaato_sdk/tests/test_cascade_events.py``.
"""

from __future__ import annotations

import threading
from typing import List
from unittest.mock import MagicMock

import pytest

from server.command_router import CommandRouter
from server.session_manager import SessionManager


# ----------------------------------------------------------------------
# Lightweight SessionManager + CommandRouter construction
# ----------------------------------------------------------------------


def _make_sm() -> SessionManager:
    """Minimal SessionManager — same pattern as test_cascade_as_client_phase1."""
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._client_to_session = {}
    sm._cascade_clients = {}
    sm._cascade_clients_lock = threading.Lock()
    sm._cascade_client_idle_timeout = 300.0
    sm._cascade_client_sweep_stop = None
    sm._cascade_client_sweep_thread = None
    sm._HEADLESS_CLIENT_ID = "_headless"
    sm._pending_wakes = {}  # wake deferred-turn store (production has it)

    class _NoWakeBindings:
        def has_live_binding_for_cid(self, cid):
            return False
    sm._wake_binding_registry = _NoWakeBindings()
    return sm


class _FakeEventSink:
    def __init__(self):
        self.events_sent: List = []

    def send_event(self, client_id, event):
        self.events_sent.append((client_id, event))


def _make_router(sm: SessionManager) -> CommandRouter:
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = sm
    router._event_sink = _FakeEventSink()
    return router


# ======================================================================
# _handle_cascade_register
# ======================================================================


class TestRegisterHandler:
    def test_register_routes_to_session_manager(self):
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_register(
            client_id="conn-uuid-1",
            args=["cascade-abc", "observer"],
        )
        # Registry has the entry.
        assert "cascade-abc" in sm._cascade_clients
        entries = sm._cascade_clients["cascade-abc"]
        assert len(entries) == 1
        # Namespaced client_id includes both cid + connection.
        assert entries[0].client_id == "_cascade:cascade-abc:conn-uuid-1"
        assert entries[0].role == "observer"

    def test_register_with_event_types(self):
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_register(
            client_id="conn-1",
            args=[
                "cascade-abc", "owner",
                "SessionTerminatedEvent", "AgentCompletedEvent",
            ],
        )
        entry = sm._cascade_clients["cascade-abc"][0]
        assert entry.event_types == {"SessionTerminatedEvent", "AgentCompletedEvent"}
        assert entry.role == "owner"

    def test_register_callback_routes_events_to_event_sink(self):
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_register(
            client_id="conn-1",
            args=["cascade-abc", "observer"],
        )
        # Invoke the registered callback to verify routing.
        entry = sm._cascade_clients["cascade-abc"][0]
        fake_event = MagicMock(name="event")
        entry.callback(fake_event)
        # event_sink received the event for conn-1.
        # (events_sent[0] is the SystemMessageEvent confirmation;
        # events_sent[1] is the dispatched event)
        client_ids = [e[0] for e in router._event_sink.events_sent]
        events = [e[1] for e in router._event_sink.events_sent]
        assert "conn-1" in client_ids
        assert fake_event in events

    def test_register_missing_args_emits_error(self):
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_register(
            client_id="conn-1",
            args=[],  # missing cid + role
        )
        # No registration.
        assert sm._cascade_clients == {}
        # ErrorEvent emitted to client.
        last_event = router._event_sink.events_sent[-1][1]
        assert type(last_event).__name__ == "ErrorEvent"
        assert last_event.error_type == "UsageError"

    def test_register_duplicate_owner_emits_error(self):
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_register(
            client_id="conn-1",
            args=["cascade-abc", "owner"],
        )
        # First registration succeeded.
        assert "cascade-abc" in sm._cascade_clients
        # Second owner from a different connection — rejected.
        router._handle_cascade_register(
            client_id="conn-2",
            args=["cascade-abc", "owner"],
        )
        # Only one owner registered.
        owners = [
            e for e in sm._cascade_clients["cascade-abc"]
            if e.role == "owner"
        ]
        assert len(owners) == 1
        # Error emitted to conn-2.
        last_for_conn2 = next(
            e for cid, e in reversed(router._event_sink.events_sent)
            if cid == "conn-2"
        )
        assert type(last_for_conn2).__name__ == "ErrorEvent"
        assert last_for_conn2.error_type == "CascadeRegistrationError"


# ======================================================================
# _handle_cascade_unregister
# ======================================================================


class TestUnregisterHandler:
    def test_unregister_removes_entry(self):
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_register(
            client_id="conn-1",
            args=["cascade-abc", "observer"],
        )
        router._handle_cascade_unregister(
            client_id="conn-1",
            args=["cascade-abc"],
        )
        assert "cascade-abc" not in sm._cascade_clients

    def test_unregister_idempotent(self):
        sm = _make_sm()
        router = _make_router(sm)
        # Never registered — unregister is a no-op (no error).
        router._handle_cascade_unregister(
            client_id="conn-1",
            args=["never-registered"],
        )
        # SystemMessageEvent with "not-found" surfaced.
        last_event = router._event_sink.events_sent[-1][1]
        assert type(last_event).__name__ == "SystemMessageEvent"

    def test_unregister_missing_args_emits_error(self):
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_unregister(
            client_id="conn-1",
            args=[],
        )
        last_event = router._event_sink.events_sent[-1][1]
        assert type(last_event).__name__ == "ErrorEvent"


# ======================================================================
# unregister_all_cascade_clients_for_connection
# ======================================================================


class TestDisconnectCleanup:
    def test_disconnect_removes_all_entries_for_client(self):
        sm = _make_sm()
        router = _make_router(sm)
        # conn-1 registers for two cascades.
        router._handle_cascade_register(
            client_id="conn-1",
            args=["cascade-A", "observer"],
        )
        router._handle_cascade_register(
            client_id="conn-1",
            args=["cascade-B", "owner"],
        )
        # conn-2 also registers for cascade-A.
        router._handle_cascade_register(
            client_id="conn-2",
            args=["cascade-A", "owner"],
        )
        assert len(sm._cascade_clients["cascade-A"]) == 2
        assert len(sm._cascade_clients["cascade-B"]) == 1

        # Disconnect conn-1.
        removed = sm.unregister_all_cascade_clients_for_connection("conn-1")
        assert removed == 2  # one in cascade-A, one in cascade-B

        # cascade-B was the only one with conn-1's owner — dict
        # entry dropped entirely.
        assert "cascade-B" not in sm._cascade_clients
        # cascade-A still has conn-2's owner.
        assert len(sm._cascade_clients["cascade-A"]) == 1
        assert sm._cascade_clients["cascade-A"][0].client_id.endswith(":conn-2")

    def test_disconnect_idempotent(self):
        sm = _make_sm()
        # No registrations — disconnect cleanup is a no-op.
        removed = sm.unregister_all_cascade_clients_for_connection("conn-never")
        assert removed == 0

    def test_suffix_match_avoids_substring_false_positives(self):
        """Don't accidentally remove ':conn-1abc' when removing ':conn-1'."""
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_register(
            client_id="conn-1",
            args=["cascade-A", "observer"],
        )
        router._handle_cascade_register(
            client_id="conn-1abc",  # similar prefix
            args=["cascade-A", "owner"],
        )
        sm.unregister_all_cascade_clients_for_connection("conn-1")
        # conn-1abc's entry survives.
        assert len(sm._cascade_clients["cascade-A"]) == 1
        assert sm._cascade_clients["cascade-A"][0].client_id.endswith(":conn-1abc")


# ======================================================================
# End-to-end through register + dispatch + unregister
# ======================================================================


class TestEndToEnd:
    def test_register_then_dispatch_then_unregister(self):
        sm = _make_sm()
        router = _make_router(sm)
        router._handle_cascade_register(
            client_id="conn-1",
            args=["cascade-X", "observer", "SessionTerminatedEvent"],
        )
        # Build a session with the matching cid + dispatch an event.
        from jaato_sdk.events import SessionTerminatedEvent
        session = MagicMock()
        session.cascade_driver_id = "cascade-X"
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason="error",
        )
        sm._dispatch_to_cascade_clients(session, event)

        # event_sink received the event addressed to conn-1.
        dispatched = [
            e for cid, e in router._event_sink.events_sent
            if cid == "conn-1" and isinstance(e, SessionTerminatedEvent)
        ]
        assert len(dispatched) == 1
        assert dispatched[0].reason == "error"

        # Unregister.
        router._handle_cascade_unregister(
            client_id="conn-1",
            args=["cascade-X"],
        )

        # Second dispatch — no event for conn-1 this time.
        router._event_sink.events_sent.clear()
        sm._dispatch_to_cascade_clients(session, event)
        assert router._event_sink.events_sent == []
