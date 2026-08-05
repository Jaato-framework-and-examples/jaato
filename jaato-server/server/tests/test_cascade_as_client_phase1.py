"""Phase 1 cascade-as-client — daemon-side registry + dispatch + default policy.

Pins the contract laid out in
``docs/design/cascade-as-client.md`` §4.1-§4.4 (decisions locked
2026-05-21).  Tests use lightweight SessionManager-like stubs to
exercise the registry + dispatch + policy without going through
the full session-creation flow.

Coverage:
  - register_in_process_client API (owner / observer; single-owner
    rule; duplicate-client_id rule)
  - unregister_cascade_client (idempotent; cleans up empty cids)
  - _dispatch_to_cascade_clients (event-type filter; owner-before-
    observer ordering; callback exception isolation;
    last_event_ts stamping)
  - _apply_default_cascade_policy (headless + terminal-error →
    unload triggered; interactive sessions unchanged;
    cascade-owner + terminal-error → unload; non-error reason →
    no-op)
  - GC sweep (stale entries reaped when idle + no active sessions)
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Set
from unittest.mock import MagicMock

import pytest

from server.session_manager import CascadeClientEntry, SessionManager


# ----------------------------------------------------------------------
# Lightweight SessionManager construction without full __init__
# ----------------------------------------------------------------------


def _make_sm() -> SessionManager:
    """Build a SessionManager skeleton sufficient for cascade-client
    tests without invoking the full __init__ (which has heavy deps).

    Sets up only the attributes the cascade-client methods touch.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._client_to_session = {}
    sm._cascade_clients = {}
    sm._cascade_clients_lock = threading.Lock()
    sm._cascade_client_idle_timeout = 300.0
    sm._cascade_client_sweep_stop = None
    sm._cascade_client_sweep_thread = None
    sm._cid_last_session_ts = {}  # Bug B fix, server 0.6.161+
    sm._HEADLESS_CLIENT_ID = "_headless"
    sm._pending_wakes = {}  # wake deferred-turn store (production has it)
    # no wake bindings in these cascade tests → the sweep's wake-durability
    # exemption is a no-op (has_live_binding_for_cid → False).
    sm._wake_binding_registry = _NoWakeBindings()
    return sm


class _NoWakeBindings:
    def has_live_binding_for_cid(self, cid):
        return False


def _make_session(
    session_id: str,
    cid: str = None,
    attached_clients: Set[str] = None,
):
    """Build a Session mock with the attrs cascade-client code reads."""
    session = MagicMock()
    session.session_id = session_id
    session.cascade_driver_id = cid
    session.attached_clients = (
        set(attached_clients) if attached_clients is not None
        else {"_headless"}
    )
    return session


# ======================================================================
# register_in_process_client
# ======================================================================


class TestRegister:
    def test_register_owner_succeeds(self):
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="_cascade:abc",
            callback=cb,
            cascade_driver_id="abc",
            role="owner",
        )
        assert "abc" in sm._cascade_clients
        entries = sm._cascade_clients["abc"]
        assert len(entries) == 1
        assert entries[0].client_id == "_cascade:abc"
        assert entries[0].role == "owner"
        assert entries[0].callback is cb

    def test_register_observer_after_owner(self):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="_cascade:abc", callback=MagicMock(),
            cascade_driver_id="abc", role="owner",
        )
        sm.register_in_process_client(
            client_id="observer-1", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm.register_in_process_client(
            client_id="observer-2", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        assert len(sm._cascade_clients["abc"]) == 3

    def test_second_owner_rejected(self):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="_cascade:abc", callback=MagicMock(),
            cascade_driver_id="abc", role="owner",
        )
        with pytest.raises(ValueError, match="owner already registered"):
            sm.register_in_process_client(
                client_id="_cascade:abc-alt", callback=MagicMock(),
                cascade_driver_id="abc", role="owner",
            )

    def test_duplicate_client_id_idempotent_on_match(self):
        """PR #182 (2026-05-21): re-register with same client_id +
        same config is a silent no-op.  Matches GateRegistry.get_or_create
        semantics so reactor / IPC callers can retry / re-register
        during reconnect cycles without try/except wrapping."""
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="obs", callback=cb,
            cascade_driver_id="abc", role="observer",
            event_types={"X", "Y"},
        )
        # Second call with IDENTICAL args — silent no-op.
        sm.register_in_process_client(
            client_id="obs", callback=cb,  # same callback identity
            cascade_driver_id="abc", role="observer",
            event_types={"X", "Y"},
        )
        # Only one entry persists.
        assert len(sm._cascade_clients["abc"]) == 1
        # First-wins: entry holds the original callback + config.
        assert sm._cascade_clients["abc"][0].callback is cb
        assert sm._cascade_clients["abc"][0].event_types == {"X", "Y"}

    def test_duplicate_client_id_mismatched_config_warns_keeps_first(self, caplog):
        """PR #182: re-register with same client_id but different
        config (callback / event_types / role) warns + keeps first.
        Matches GateRegistry's 'mismatched re-registration → log
        warning, keep original' pattern at registry.py:128."""
        import logging
        sm = _make_sm()
        original_cb = MagicMock(name="original")
        sm.register_in_process_client(
            client_id="obs", callback=original_cb,
            cascade_driver_id="abc", role="observer",
            event_types={"X"},
        )
        # Second call with DIFFERENT callback + event_types — warn + keep first.
        with caplog.at_level(logging.WARNING):
            sm.register_in_process_client(
                client_id="obs", callback=MagicMock(name="different"),
                cascade_driver_id="abc", role="observer",
                event_types={"Y"},
            )
        # Warning emitted.
        assert any(
            "different config" in r.message
            for r in caplog.records
        )
        # First registration preserved.
        assert len(sm._cascade_clients["abc"]) == 1
        assert sm._cascade_clients["abc"][0].callback is original_cb
        assert sm._cascade_clients["abc"][0].event_types == {"X"}

    def test_different_client_id_owner_conflict_still_raises(self):
        """PR #182 preserves the single-owner-per-cid rule for
        DIFFERENT client_ids.  Same-client_id idempotency above
        ONLY applies when client_id matches.  Two DIFFERENT callers
        both trying to own the same cid → real conflict → raise."""
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="_cascade:abc:conn-1", callback=MagicMock(),
            cascade_driver_id="abc", role="owner",
        )
        with pytest.raises(ValueError, match="owner already registered"):
            sm.register_in_process_client(
                client_id="_cascade:abc:conn-2",  # DIFFERENT client_id
                callback=MagicMock(),
                cascade_driver_id="abc", role="owner",
            )

    def test_invalid_role_rejected(self):
        sm = _make_sm()
        with pytest.raises(ValueError, match="must be 'owner' or 'observer'"):
            sm.register_in_process_client(
                client_id="x", callback=MagicMock(),
                cascade_driver_id="abc", role="bogus",
            )

    def test_event_types_filter_stored(self):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
            event_types={"SessionTerminatedEvent"},
        )
        assert sm._cascade_clients["abc"][0].event_types == {"SessionTerminatedEvent"}


# ======================================================================
# unregister_cascade_client
# ======================================================================


class TestUnregister:
    def test_unregister_removes_entry(self):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        assert sm.unregister_cascade_client("abc", "obs") is True
        # Empty cid dropped from registry
        assert "abc" not in sm._cascade_clients

    def test_unregister_idempotent(self):
        sm = _make_sm()
        assert sm.unregister_cascade_client("never-registered", "x") is False
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm.unregister_cascade_client("abc", "obs")
        # Second unregister: idempotent, returns False
        assert sm.unregister_cascade_client("abc", "obs") is False

    def test_unregister_one_of_many_keeps_others(self):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="_cascade:abc", callback=MagicMock(),
            cascade_driver_id="abc", role="owner",
        )
        sm.register_in_process_client(
            client_id="obs-1", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm.unregister_cascade_client("abc", "obs-1")
        assert len(sm._cascade_clients["abc"]) == 1
        assert sm._cascade_clients["abc"][0].client_id == "_cascade:abc"


# ======================================================================
# _dispatch_to_cascade_clients
# ======================================================================


class _FakeEvent:
    pass


class _OtherEvent:
    pass


class TestDispatch:
    def test_no_cascade_id_skips_dispatch(self):
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="obs", callback=cb,
            cascade_driver_id="abc", role="observer",
        )
        session = _make_session("s1", cid=None)
        sm._dispatch_to_cascade_clients(session, _FakeEvent())
        cb.assert_not_called()

    def test_no_match_cid_skips_dispatch(self):
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="obs", callback=cb,
            cascade_driver_id="abc", role="observer",
        )
        session = _make_session("s1", cid="different-cid")
        sm._dispatch_to_cascade_clients(session, _FakeEvent())
        cb.assert_not_called()

    def test_matching_cid_invokes_callback(self):
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="obs", callback=cb,
            cascade_driver_id="abc", role="observer",
        )
        session = _make_session("s1", cid="abc")
        event = _FakeEvent()
        sm._dispatch_to_cascade_clients(session, event)
        cb.assert_called_once_with(event)

    def test_owner_equals_observer_deduped_post_bootstrap(self):
        # owner==observer on one connection: the client is BOTH attached (so it
        # already received the direct _emit_to_client fan-out) AND a cascade
        # observer.  The post-bootstrap cascade dispatch must SKIP its entry
        # (delivery_target_id in session.attached_clients) so post-bootstrap turn
        # events don't double-deliver on the same raw connection.
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="obs", callback=cb,
            cascade_driver_id="abc", role="observer",
            delivery_target_id="conn1",           # raw connection id
        )
        session = _make_session("s1", cid="abc", attached_clients={"conn1"})
        sm._dispatch_to_cascade_clients(session, _FakeEvent())
        cb.assert_not_called()                    # deduped — already got direct emit

    def test_observer_on_separate_connection_still_delivered(self):
        # A cascade observer on a DIFFERENT connection (NOT attached to the
        # session) did not get the direct emit → must still receive via cascade.
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="obs", callback=cb,
            cascade_driver_id="abc", role="observer",
            delivery_target_id="conn2",           # not attached
        )
        session = _make_session("s1", cid="abc", attached_clients={"conn1"})
        event = _FakeEvent()
        sm._dispatch_to_cascade_clients(session, event)
        cb.assert_called_once_with(event)         # delivered (not deduped)

    def test_event_type_filter_drops_non_matches(self):
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="obs", callback=cb,
            cascade_driver_id="abc", role="observer",
            event_types={"_FakeEvent"},  # only this type
        )
        session = _make_session("s1", cid="abc")
        sm._dispatch_to_cascade_clients(session, _OtherEvent())
        cb.assert_not_called()
        sm._dispatch_to_cascade_clients(session, _FakeEvent())
        cb.assert_called_once()

    def test_owner_before_observer_order(self):
        sm = _make_sm()
        call_order: List[str] = []
        sm.register_in_process_client(
            client_id="obs-1",
            callback=lambda e: call_order.append("obs-1"),
            cascade_driver_id="abc", role="observer",
        )
        # Owner registered AFTER observer
        sm.register_in_process_client(
            client_id="_cascade:abc",
            callback=lambda e: call_order.append("owner"),
            cascade_driver_id="abc", role="owner",
        )
        session = _make_session("s1", cid="abc")
        sm._dispatch_to_cascade_clients(session, _FakeEvent())
        # Owner fires FIRST regardless of registration order
        assert call_order == ["owner", "obs-1"]

    def test_callback_exception_isolated(self):
        sm = _make_sm()
        good_cb = MagicMock()
        def bad_cb(event):
            raise RuntimeError("boom")
        sm.register_in_process_client(
            client_id="bad", callback=bad_cb,
            cascade_driver_id="abc", role="observer",
        )
        sm.register_in_process_client(
            client_id="good", callback=good_cb,
            cascade_driver_id="abc", role="observer",
        )
        session = _make_session("s1", cid="abc")
        # Should not raise — bad callback logged + dispatch continues
        sm._dispatch_to_cascade_clients(session, _FakeEvent())
        good_cb.assert_called_once()

    def test_last_event_ts_updated_on_dispatch(self):
        sm = _make_sm()
        cb = MagicMock()
        sm.register_in_process_client(
            client_id="obs", callback=cb,
            cascade_driver_id="abc", role="observer",
        )
        entry = sm._cascade_clients["abc"][0]
        assert entry.last_event_ts is None
        session = _make_session("s1", cid="abc")
        sm._dispatch_to_cascade_clients(session, _FakeEvent())
        assert entry.last_event_ts is not None


# ======================================================================
# _apply_default_cascade_policy
# ======================================================================


class TestDefaultPolicy:
    def _setup_sm_with_unload_stub(self, sm: SessionManager) -> List[str]:
        """Patch _maybe_unload_session to record calls; return the
        recording list for assertions."""
        unload_calls: List[str] = []
        sm._maybe_unload_session = lambda sid: unload_calls.append(sid)
        return unload_calls

    def test_non_terminal_event_no_op(self):
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        session = _make_session("s1", cid="abc")
        sm._apply_default_cascade_policy(session, _FakeEvent())
        assert unload_calls == []

    def test_headless_terminal_error_triggers_unload(self):
        from jaato_sdk.events import SessionTerminatedEvent
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        session = _make_session(
            "s1", cid=None,  # standalone headless, no cid
            attached_clients={"_headless"},
        )
        sm._sessions["s1"] = session
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason="error",
        )
        sm._apply_default_cascade_policy(session, event)
        assert unload_calls == ["s1"]
        # _HEADLESS_CLIENT_ID popped so _maybe_unload_session's gate
        # would see empty attached_clients
        assert "_headless" not in session.attached_clients

    def test_cascade_owner_terminal_error_triggers_unload(self):
        from jaato_sdk.events import SessionTerminatedEvent
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        sm.register_in_process_client(
            client_id="_cascade:abc", callback=MagicMock(),
            cascade_driver_id="abc", role="owner",
        )
        session = _make_session(
            "s1", cid="abc", attached_clients={"_headless"},
        )
        sm._sessions["s1"] = session
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason="error",
        )
        sm._apply_default_cascade_policy(session, event)
        assert unload_calls == ["s1"]

    @pytest.mark.parametrize(
        "reason", ["natural", "client_request", "stopped"],
    )
    def test_headless_terminal_non_error_reasons_trigger_unload(
        self, reason,
    ):
        """Server 0.6.158: natural/client_request/stopped reasons on a
        headless session also trigger unload — closes the cascade-pool
        reuse miss + 1h37min runner-exit lag observed in retry-17.

        ``SessionTerminatedEvent`` is by definition terminal; the old
        ``reason != "error"`` guard let natural-completion sessions
        stay loaded indefinitely, which kept the pool slot busy +
        the runner subprocess alive."""
        from jaato_sdk.events import SessionTerminatedEvent
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        session = _make_session(
            "s1", cid=None, attached_clients={"_headless"},
        )
        sm._sessions["s1"] = session
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason=reason,
        )
        sm._apply_default_cascade_policy(session, event)
        assert unload_calls == ["s1"], (
            f"reason={reason!r} should trigger unload for headless"
        )

    @pytest.mark.parametrize(
        "reason", ["natural", "client_request", "stopped"],
    )
    def test_cascade_owner_terminal_non_error_reasons_trigger_unload(
        self, reason,
    ):
        """Same as the headless case but for cascade-owned sessions —
        any cascade-owner registration + any terminal reason → unload.
        This is the path that fires for ReactorExtension's
        register_in_process_client(role="owner") sessions."""
        from jaato_sdk.events import SessionTerminatedEvent
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        sm.register_in_process_client(
            client_id="_cascade:abc", callback=MagicMock(),
            cascade_driver_id="abc", role="owner",
        )
        session = _make_session(
            "s1", cid="abc", attached_clients={"_headless"},
        )
        sm._sessions["s1"] = session
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason=reason,
        )
        sm._apply_default_cascade_policy(session, event)
        assert unload_calls == ["s1"], (
            f"reason={reason!r} should trigger unload for cascade-owned"
        )

    def test_interactive_session_terminal_error_no_unload(self):
        """Real-client sessions (UI/TUI) NOT unloaded on terminal-
        error — they may want to reconnect to see history."""
        from jaato_sdk.events import SessionTerminatedEvent
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        session = _make_session(
            "s1", cid=None,  # not cascade-driven
            attached_clients={"real-client-uuid"},
        )
        sm._sessions["s1"] = session
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason="error",
        )
        sm._apply_default_cascade_policy(session, event)
        # Real client present, no cascade-owner → no unload
        assert unload_calls == []

    def test_cascade_session_auto_detaches_real_ipc_clients(self):
        """Server 0.6.165+ (γ'): cascade-stamped sessions
        (cascade_driver_id != None) auto-detach IPC clients on
        terminal events so unload proceeds even when the
        cascade-driver's IPCClient is still connected.

        Empirical motivation (peer 7:1, retry-46, 2026-05-28):
        cascade.py driver called client.create_session("discovery",
        cascade_driver_id=...).  The driver's ipc_1 became attached
        to the discovery session.  After natural completion at
        22:27:28, SessionTerminatedEvent fired + policy triggered
        unload — but _maybe_unload_session was a no-op because
        attached_clients still contained ipc_1.  Slot held 6m43s
        until the driver was killed.  γ' makes cascade-stamped
        sessions auto-detach real IPC clients here so the unload
        chain proceeds.

        TUI / interactive sessions don't pass cascade_driver_id so
        they're unaffected — covered by
        test_interactive_session_terminal_error_no_unload above.
        """
        from jaato_sdk.events import SessionTerminatedEvent
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        # In the empirical scenario, the premium reactor extension
        # auto-registers as owner for cascade-driver-id sessions
        # (Phase 3, premium 0.1.187+).  Mirror that so the policy
        # gate passes through to the γ' auto-detach code.
        sm.register_in_process_client(
            client_id="_cascade:abc",
            callback=MagicMock(),
            cascade_driver_id="abc",
            role="owner",
        )
        # Cascade-stamped session with a REAL IPC client attached
        # (NOT just the synthetic _HEADLESS_CLIENT_ID — here ipc_1
        # is the live cascade-driver IPC connection).
        session = _make_session(
            "s1", cid="abc",
            attached_clients={"ipc_1"},
        )
        sm._sessions["s1"] = session
        sm._client_to_session["ipc_1"] = "s1"
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason="natural",
        )
        sm._apply_default_cascade_policy(session, event)
        # Policy must (a) detach ipc_1 from session attached_clients,
        # (b) pop ipc_1 from _client_to_session, and (c) trigger
        # _maybe_unload_session (recorded in unload_calls).
        assert "ipc_1" not in session.attached_clients, (
            "γ' must detach IPC clients from cascade-stamped sessions "
            "on terminal events so unload can proceed"
        )
        assert "ipc_1" not in sm._client_to_session, (
            "γ' must pop the client_to_session reverse mapping"
        )
        assert unload_calls == ["s1"], (
            "γ' must reach _maybe_unload_session for cascade sessions "
            "with real IPC clients attached"
        )

    def test_cascade_session_auto_detach_preserves_other_session_attachments(
        self,
    ):
        """γ' must not pop ``_client_to_session`` entries that
        point at a DIFFERENT session — the client may have attached
        somewhere else since this cascade started.
        """
        from jaato_sdk.events import SessionTerminatedEvent
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        sm.register_in_process_client(
            client_id="_cascade:abc",
            callback=MagicMock(),
            cascade_driver_id="abc",
            role="owner",
        )
        session = _make_session(
            "s1", cid="abc",
            attached_clients={"ipc_1"},
        )
        sm._sessions["s1"] = session
        # ipc_1's _client_to_session points elsewhere (the client
        # re-attached mid-cascade or the bookkeeping diverged).
        sm._client_to_session["ipc_1"] = "other-session"
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="main", reason="natural",
        )
        sm._apply_default_cascade_policy(session, event)
        # ipc_1 still removed from THIS session's attached_clients ...
        assert "ipc_1" not in session.attached_clients
        # ... but client_to_session entry is preserved (it pointed
        # at a different session).
        assert sm._client_to_session["ipc_1"] == "other-session"

    def test_observer_only_discovery_unloads_without_owner(self):
        """γ'-guard fix (server 0.6.166+): a cascade-stamped DISCOVERY
        session whose IPC client registered ONLY as an *observer* (NO
        owner entry) must still detach + unload on SessionTerminated.

        This is the REAL production path (2026-06-11 daemon-log
        evidence): the cascade driver calls
        ``client.create_session("discovery", cascade_driver_id=...)``
        and the SDK harness registers as ``role="observer"``, never
        ``"owner"``.  Before the fix ``is_headless=False`` AND
        ``has_cascade_owner=False``, so the policy early-returned and
        the γ' detach never ran — discovery's pool slot stayed pinned
        2m50s–6m25s until the driver detached on its own, stalling the
        cascade's first handoff while every headless handoff returned
        its slot in ~250ms.

        Distinct from
        ``test_cascade_session_auto_detaches_real_ipc_clients`` above,
        which registers an OWNER and so passes the guard via a
        different disjunct — that test's owner assumption did NOT hold
        in production (no owner is ever registered for these cascades),
        which is exactly why the leak survived it.  This test pins the
        observer-only reality.
        """
        from jaato_sdk.events import SessionTerminatedEvent
        sm = _make_sm()
        unload_calls = self._setup_sm_with_unload_stub(sm)
        # Observer-only registration — the actual driver / SDK-harness
        # role (NOT owner).
        sm.register_in_process_client(
            client_id="driver-observer",
            callback=MagicMock(),
            cascade_driver_id="abc",
            role="observer",
        )
        # Driver-attached discovery session: real IPC client, NOT
        # headless, NO owner entry.
        session = _make_session(
            "s1", cid="abc",
            attached_clients={"ipc_7"},
        )
        sm._sessions["s1"] = session
        sm._client_to_session["ipc_7"] = "s1"
        event = SessionTerminatedEvent(
            session_id="s1", agent_id="discovery", reason="natural",
        )
        sm._apply_default_cascade_policy(session, event)
        assert "ipc_7" not in session.attached_clients, (
            "γ'-guard fix must detach the driver IPC client from an "
            "observer-only cascade session so the slot returns promptly"
        )
        assert "ipc_7" not in sm._client_to_session
        assert unload_calls == ["s1"], (
            "observer-only cascade-stamped session must reach "
            "_maybe_unload_session — without the cid disjunct it "
            "early-returned and the slot leaked for minutes"
        )


# ======================================================================
# GC sweep
# ======================================================================


class TestSweep:
    def test_sweep_reaps_stale_entries_no_active_sessions(self):
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 0.05  # fast for tests
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        # Mark entry as stale.
        entry = sm._cascade_clients["abc"][0]
        entry.registered_at = time.monotonic() - 1.0
        # No session with cid="abc" exists → eligible for GC.
        sm._cascade_client_sweep_once()
        assert "abc" not in sm._cascade_clients

    def test_sweep_exempts_cid_with_live_wake_binding(self):
        # Wake durability (Option 2): a stale, session-less cid is normally
        # reaped — but NOT while a live wake binding carries it (a wake may still
        # arrive; the observer must survive the session going cold).
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 0.05

        class _HasBindingForCid:
            def has_live_binding_for_cid(self, cid):
                return cid == "wake-cid"
        sm._wake_binding_registry = _HasBindingForCid()

        sm.register_in_process_client(
            client_id="bot", callback=MagicMock(),
            cascade_driver_id="wake-cid", role="observer",
        )
        entry = sm._cascade_clients["wake-cid"][0]
        entry.registered_at = time.monotonic() - 1.0  # stale, no session
        sm._cascade_client_sweep_once()
        # observer SURVIVES because a live wake binding keeps the cid alive
        assert "wake-cid" in sm._cascade_clients

    def test_sweep_skips_entries_with_active_sessions(self):
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 0.05
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        entry = sm._cascade_clients["abc"][0]
        entry.registered_at = time.monotonic() - 1.0
        # Add a session WITH cid="abc" — sweep must skip.
        session = _make_session("active-sid", cid="abc")
        sm._sessions["active-sid"] = session
        sm._cascade_client_sweep_once()
        assert "abc" in sm._cascade_clients

    def test_sweep_skips_fresh_entries(self):
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 60.0
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        # Fresh registration — registered_at is now.  No sessions
        # active either.
        sm._cascade_client_sweep_once()
        assert "abc" in sm._cascade_clients

    def test_sweep_partial_reap_keeps_survivors(self):
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 1.0
        sm.register_in_process_client(
            client_id="stale", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm.register_in_process_client(
            client_id="fresh", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm._cascade_clients["abc"][0].registered_at = time.monotonic() - 10.0
        sm._cascade_clients["abc"][1].registered_at = time.monotonic()
        sm._cascade_client_sweep_once()
        # Stale entry reaped; fresh entry survives.
        assert len(sm._cascade_clients.get("abc", [])) == 1
        assert sm._cascade_clients["abc"][0].client_id == "fresh"

    def test_sweep_skips_entries_with_recent_cid_session_activity(self):
        """Server 0.6.161+ (Bug B fix): the sweep must NOT reap an
        observer whose cascade has had a recent session creation,
        even if no session is currently loaded.

        Empirical motivation (peer 7:1 retry-27, 2026-05-26 22:03:57):
        the sweep landed in the 23-second window between codegen
        step 1 unloading (per PR-183 default policy) and codegen
        step 2 spawning.  Pre-fix the observer got reaped because
        the "any currently-loaded session" check failed during that
        gap.  Cascades are SERIAL under PR-183 — these gaps are
        normal, not idle."""
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 1.0
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        # Mark observer's registration as stale by both metrics that
        # the old sweep predicate used (last_event_ts AND
        # registered_at older than timeout).
        entry = sm._cascade_clients["abc"][0]
        entry.registered_at = time.monotonic() - 10.0
        entry.last_event_ts = time.monotonic() - 10.0
        # No session currently loaded with cid="abc" — pre-fix this
        # would reap.
        assert not any(
            getattr(s, "cascade_driver_id", None) == "abc"
            for s in sm._sessions.values()
        )
        # But a session WAS created with cid="abc" recently — the
        # new sweep predicate must keep the observer alive.
        sm._record_cid_session_activity("abc")
        sm._cascade_client_sweep_once()
        assert "abc" in sm._cascade_clients, (
            "Bug B fix: sweep must skip reap when "
            "_cid_last_session_ts[cid] is recent, even if no session "
            "is currently loaded"
        )

    def test_sweep_reaps_when_cid_session_activity_also_stale(self):
        """Bug B fix: when BOTH the registration AND the cid's
        last-session-creation timestamp are older than timeout, the
        sweep still reaps (cascade is genuinely done).

        Also verifies cleanup: ``_cid_last_session_ts[cid]`` is
        removed when the last registration for that cid is reaped.
        """
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 0.05
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        # Both metrics stale.
        entry = sm._cascade_clients["abc"][0]
        entry.registered_at = time.monotonic() - 1.0
        entry.last_event_ts = time.monotonic() - 1.0
        sm._cid_last_session_ts["abc"] = time.monotonic() - 1.0
        sm._cascade_client_sweep_once()
        assert "abc" not in sm._cascade_clients
        # Cleanup: the cid-activity dict was pruned when its last
        # registration was reaped.
        assert "abc" not in sm._cid_last_session_ts


# ======================================================================
# _route_bootstrap_event  (server 0.6.166+)
# ======================================================================


class TestRouteBootstrapEvent:
    """Server 0.6.166+ centralized bootstrap-time event router.

    Closes the gap where events emitted during ``_bootstrap_session``
    bypassed the cascade-dispatch chain because the regular
    ``set_event_callback`` wires AFTER bootstrap.  Empirical motivation:
    peer 7:1 retry-47 — AgentCreatedEvent for reactor-spawned headless
    cascade sessions (context / host_validator / codegen) never reached
    cascade observers because they emit during init via
    ``on_event_during_init`` which previously routed only to
    ``_emit_to_client(_HEADLESS_CLIENT_ID, ...)`` (transport-dropped).
    """

    def _setup_sm_with_emit_stubs(self, sm: SessionManager):
        """Stub _emit_to_client + the cascade dispatch helper to
        record their calls.  Returns the two recording lists.

        ``skip_client_id`` kwarg accepted (server 0.6.177+) so the
        stub mirrors the real signature; recorded as part of the
        tuple alongside cid + event for tests that care about it.
        """
        client_emits: List[tuple] = []  # (client_id, event)
        cascade_dispatches: List[tuple] = []  # (cid, event, skip_client_id)
        sm._emit_to_client = lambda cid, ev: client_emits.append((cid, ev))
        sm._dispatch_to_cascade_clients_by_cid = (
            lambda cid, ev, skip_client_id=None: cascade_dispatches.append(
                (cid, ev, skip_client_id)
            )
        )
        return client_emits, cascade_dispatches

    def test_routes_to_both_client_and_cascade_when_both_set(self):
        """When direct_client_id AND cascade_driver_id are both set,
        the event reaches both the requesting client AND cascade
        observers.  Server 0.6.177+: the cascade-dispatch call also
        passes ``skip_client_id=direct_client_id`` so the same-client
        cascade subscriber doesn't double-deliver — see
        TestRouteBootstrapEventDedup for the empirical pin."""
        sm = _make_sm()
        client_emits, cascade_dispatches = self._setup_sm_with_emit_stubs(sm)
        event = MagicMock()
        sm._route_bootstrap_event("ipc_1", "cid-abc", event)
        assert client_emits == [("ipc_1", event)]
        # 3-tuple shape (cid, event, skip_client_id) per server 0.6.177+.
        assert cascade_dispatches == [("cid-abc", event, "ipc_1")]

    def test_routes_only_to_client_when_no_cascade(self):
        """Standalone (non-cascade) session bootstrap: cascade
        dispatch is a no-op (cascade_driver_id is None), direct
        client gets the event."""
        sm = _make_sm()
        client_emits, cascade_dispatches = self._setup_sm_with_emit_stubs(sm)
        event = MagicMock()
        sm._route_bootstrap_event("real-client", None, event)
        assert client_emits == [("real-client", event)]
        assert cascade_dispatches == []

    def test_routes_only_to_cascade_when_client_id_is_none(self):
        """Restore-without-client path: cascade dispatch fires if
        cid is set, direct client emit is skipped.  ``skip_client_id``
        is None (no direct-attach to dedup against)."""
        sm = _make_sm()
        client_emits, cascade_dispatches = self._setup_sm_with_emit_stubs(sm)
        event = MagicMock()
        sm._route_bootstrap_event(None, "cid-abc", event)
        assert client_emits == []
        assert cascade_dispatches == [("cid-abc", event, None)]

    def test_headless_cascade_bootstrap_reaches_cascade_observers(self):
        """The empirical scenario peer 7:1 hit: reactor-spawned
        downstream cascade session bootstraps with
        client_id=_HEADLESS_CLIENT_ID (transport-dropped).  Before
        0.6.166 the AgentCreatedEvent was lost.  Now the route_helper
        delivers it to cascade observers via the cid path.

        Server 0.6.177+: ``skip_client_id`` is the headless id, but
        the dedup branch is a no-op because no real cascade observer
        has client_id == _HEADLESS_CLIENT_ID (only transport-side
        synthetic).  Load-bearing for the PR-194 cascade-observer
        delivery contract — see TestRouteBootstrapEventDedup
        ``test_headless_direct_plus_real_observer_still_delivers``."""
        sm = _make_sm()
        client_emits, cascade_dispatches = self._setup_sm_with_emit_stubs(sm)
        event = MagicMock()
        sm._route_bootstrap_event(sm._HEADLESS_CLIENT_ID, "cid-abc", event)
        # Direct emit still fires (the transport drops it; the
        # helper doesn't second-guess the transport's job).
        assert client_emits == [(sm._HEADLESS_CLIENT_ID, event)]
        # Critical: cascade observers reached via the cid path.
        # 3-tuple with skip_client_id=_HEADLESS_CLIENT_ID (no-op
        # because no real observer matches that synthetic id).
        assert cascade_dispatches == [
            ("cid-abc", event, sm._HEADLESS_CLIENT_ID),
        ]

    def test_dispatch_by_cid_no_op_on_none(self):
        """``_dispatch_to_cascade_clients_by_cid(None, event)`` is
        a no-op — standalone sessions emit through this path during
        post-bootstrap dispatch + standalone bootstrap, neither
        needs cascade fan-out."""
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="observer", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        cb = sm._cascade_clients["abc"][0].callback
        # No-op: cid is None, no entries should be touched.
        sm._dispatch_to_cascade_clients_by_cid(None, MagicMock())
        cb.assert_not_called()

    def test_dispatch_by_cid_fans_out_to_matching_entries(self):
        """End-to-end through the extracted helper: events fan out
        to all entries registered for the matching cid, owners
        first, observers second.  Compose-check that the helper
        behaves identically to the pre-refactor
        _dispatch_to_cascade_clients on the cid-matching path."""
        sm = _make_sm()
        owner_cb = MagicMock()
        observer_cb = MagicMock()
        sm.register_in_process_client(
            client_id="_cascade:abc", callback=owner_cb,
            cascade_driver_id="abc", role="owner",
        )
        sm.register_in_process_client(
            client_id="obs-1", callback=observer_cb,
            cascade_driver_id="abc", role="observer",
        )
        event = MagicMock()
        sm._dispatch_to_cascade_clients_by_cid("abc", event)
        owner_cb.assert_called_once_with(event)
        observer_cb.assert_called_once_with(event)


# ======================================================================
# Server 0.6.177 — _route_bootstrap_event dedup
# ======================================================================


class TestRouteBootstrapEventDedup:
    """``_route_bootstrap_event`` dedup: when the same IPC client is
    BOTH the direct-attach client AND a cascade observer for the
    same cid, the bootstrap-time event arrives EXACTLY ONCE on the
    SDK queue.

    Surfaced empirically by cascade_develop.py walker against 0.6.176
    (kb-side report 2026-06-03): main-agent stage printed two ``↳
    session <id>`` lines per AgentCreatedEvent because
    cascade_develop's ipc_1 was BOTH the creating client (direct
    attach) AND a cascade observer for its own cid.  PR-207 wired
    the routing-layer skip-branch using ``entry.client_id`` as the
    comparand, but command_router.py stores cascade observers with
    the NAMESPACED registration id (``_cascade:{cid}:{conn}``), not
    the raw connection id — so the dedup never fired empirically
    (kb-side falsification 2026-06-03 against 0.6.177).

    Resolution (server 0.6.178+): ``CascadeClientEntry`` carries a
    separate ``delivery_target_id`` field — the raw connection id
    the callback delivers TO.  The skip-check compares
    ``entry.delivery_target_id == skip_client_id`` instead.  In-
    process callers that don't set ``delivery_target_id`` (extension
    callbacks not tied to IPC) get None → skip never fires →
    extension-callback delivery contract preserved.
    """

    def test_namespaced_observer_with_delivery_target_id_dedups(self):
        """Empirical production scenario (server 0.6.178+):
        cascade_develop creates session (direct-attach as ipc_1)
        AND subscribes via cascade.register → the cascade observer
        is registered with the NAMESPACED ``_cascade:cid:ipc_1``
        client_id, ``delivery_target_id=ipc_1`` (the raw connection).
        ``_route_bootstrap_event`` passes ``skip_client_id=ipc_1``.
        Dedup MUST fire by matching ``delivery_target_id``, not
        ``client_id`` (the namespaced string would never literal-
        match)."""
        sm = _make_sm()

        # Capture direct-IPC deliveries.
        direct_deliveries: List[Any] = []
        sm._emit_to_client = lambda cid, evt: direct_deliveries.append(
            (cid, evt)
        )

        # Register the NAMESPACED cascade observer (matches what
        # command_router.py:548 actually does in production).
        cascade_deliveries: List[Any] = []
        sm.register_in_process_client(
            client_id="_cascade:cid-abc:ipc_1",
            callback=lambda evt: cascade_deliveries.append(evt),
            cascade_driver_id="cid-abc",
            role="observer",
            delivery_target_id="ipc_1",
        )

        event = MagicMock()
        sm._route_bootstrap_event(
            direct_client_id="ipc_1",
            cascade_driver_id="cid-abc",
            event=event,
        )

        # Total deliveries (direct + cascade route-back) for ipc_1:
        # exactly 1.
        assert len(direct_deliveries) == 1, (
            "direct-IPC path must always deliver (load-bearing for "
            "the client that initiated the session)"
        )
        assert direct_deliveries[0] == ("ipc_1", event)
        assert len(cascade_deliveries) == 0, (
            f"cascade route-back must be SKIPPED via "
            f"delivery_target_id match; got {len(cascade_deliveries)} "
            f"unwanted dup delivery"
        )

    def test_inprocess_callback_no_delivery_target_id_still_delivers(self):
        """Load-bearing path: in-process cascade observer (e.g.
        premium reactor extension wiring a callback) registers
        WITHOUT a ``delivery_target_id`` because the callback is not
        tied to a raw IPC connection.  The skip-branch is a no-op
        for such entries — the extension callback STILL receives
        the event.

        Without this guarantee, every extension-based cascade
        observer would silently miss bootstrap events whenever a
        direct-attach client happened to use a client_id that
        matched something (impossible to predict at extension
        author time).  The dedup skip MUST only fire on entries
        with explicit, non-None ``delivery_target_id``."""
        sm = _make_sm()
        sm._emit_to_client = lambda cid, evt: None

        cascade_deliveries: List[Any] = []
        sm.register_in_process_client(
            # Extension-style: no namespaced prefix, no
            # delivery_target_id — pure in-process callback.
            client_id="extension-observer-42",
            callback=lambda evt: cascade_deliveries.append(evt),
            cascade_driver_id="cid-abc",
            role="observer",
            # delivery_target_id deliberately omitted (defaults to None)
        )

        event = MagicMock()
        sm._route_bootstrap_event(
            direct_client_id="ipc_1",  # arbitrary direct-attach
            cascade_driver_id="cid-abc",
            event=event,
        )
        assert len(cascade_deliveries) == 1, (
            f"Extension callback regression: in-process observer "
            f"with delivery_target_id=None MUST still receive the "
            f"event; got {len(cascade_deliveries)} (skip branch "
            f"incorrectly fired for a None delivery_target_id)"
        )

    def test_entry_carries_delivery_target_id_when_provided(self):
        """Pin the dataclass field contract: when
        ``register_in_process_client`` is called with
        ``delivery_target_id=...``, the stored
        ``CascadeClientEntry.delivery_target_id`` reflects it.
        Catches accidental param-drop regressions."""
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="_cascade:cid-x:ipc_99",
            callback=MagicMock(),
            cascade_driver_id="cid-x",
            role="observer",
            delivery_target_id="ipc_99",
        )
        entries = sm._cascade_clients["cid-x"]
        assert len(entries) == 1
        assert entries[0].delivery_target_id == "ipc_99"
        assert entries[0].client_id == "_cascade:cid-x:ipc_99"

    def test_entry_delivery_target_id_defaults_to_none(self):
        """Back-compat pin: callers that don't pass
        ``delivery_target_id`` get the default ``None`` on the
        stored entry.  Preserves the extension-callback
        registration shape that pre-dates 0.6.178."""
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="extension-observer-7",
            callback=MagicMock(),
            cascade_driver_id="cid-x",
            role="observer",
        )
        entries = sm._cascade_clients["cid-x"]
        assert entries[0].delivery_target_id is None

    def test_headless_direct_plus_real_observer_still_delivers(self):
        """Load-bearing path (PR-194 regression guard): when the
        direct_client_id is ``_HEADLESS_CLIENT_ID`` (reactor-spawned
        downstream session like context / host_validator / codegen)
        and a separate real IPC client is registered as cascade
        observer, the observer MUST still receive the event.  The
        skip-branch is a no-op because the headless id never matches
        a real cascade-observer's client_id.

        This is the original cascade-as-client delivery contract;
        regressing it would break the reactor-spawned downstream
        observer fan-out (the v152-retry-47 motivation for PR-194)."""
        sm = _make_sm()

        direct_deliveries: List[Any] = []
        sm._emit_to_client = lambda cid, evt: direct_deliveries.append(
            (cid, evt)
        )

        cascade_deliveries: List[Any] = []
        sm.register_in_process_client(
            client_id="ipc_2",  # a real observer, different from headless
            callback=lambda evt: cascade_deliveries.append(evt),
            cascade_driver_id="cid-abc",
            role="observer",
        )

        event = MagicMock()
        sm._route_bootstrap_event(
            direct_client_id=sm._HEADLESS_CLIENT_ID,
            cascade_driver_id="cid-abc",
            event=event,
        )

        # The headless direct-attach path still fires (existing
        # contract — _emit_to_client called even for headless;
        # transport drops it silently if no real subscriber).
        assert len(direct_deliveries) == 1
        assert direct_deliveries[0] == (sm._HEADLESS_CLIENT_ID, event)
        # The real cascade observer receives it (NOT skipped).
        assert len(cascade_deliveries) == 1, (
            "cascade observer MUST still receive the event when "
            "direct_client_id is headless (the skip branch is a "
            "no-op for non-matching client_ids)"
        )

    def test_direct_attach_only_no_cascade_observer_delivers_once(self):
        """Defensive: when there's no cascade observer registered
        at all, the direct-IPC path still delivers; the cascade
        dispatch is a no-op via the existing ``cid is None`` /
        empty-entries-list check."""
        sm = _make_sm()
        direct_deliveries: List[Any] = []
        sm._emit_to_client = lambda cid, evt: direct_deliveries.append(
            (cid, evt)
        )

        event = MagicMock()
        sm._route_bootstrap_event(
            direct_client_id="ipc_1",
            cascade_driver_id="cid-no-observer",
            event=event,
        )
        assert len(direct_deliveries) == 1

    def test_skip_client_id_param_default_preserves_old_behavior(self):
        """Pin: existing direct callers of
        ``_dispatch_to_cascade_clients_by_cid`` (no
        ``skip_client_id`` arg) get the pre-0.6.177 behavior — all
        registered observers receive the event, including any
        client that happens to also be direct-attached elsewhere.
        Default ``None`` preserves backward compatibility."""
        sm = _make_sm()
        cascade_deliveries: List[Any] = []
        sm.register_in_process_client(
            client_id="ipc_1",
            callback=lambda evt: cascade_deliveries.append(evt),
            cascade_driver_id="cid-abc",
            role="observer",
        )

        event = MagicMock()
        # Call WITHOUT skip_client_id (the post-bootstrap dispatch
        # path's invocation pattern).
        sm._dispatch_to_cascade_clients_by_cid("cid-abc", event)
        assert len(cascade_deliveries) == 1, (
            "default behavior MUST fire every registered observer; "
            "skip_client_id=None is opt-in dedup only"
        )
