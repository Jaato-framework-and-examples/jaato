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
    return sm


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
        record their calls.  Returns the two recording lists."""
        client_emits: List[tuple] = []  # (client_id, event)
        cascade_dispatches: List[tuple] = []  # (cid, event)
        sm._emit_to_client = lambda cid, ev: client_emits.append((cid, ev))
        sm._dispatch_to_cascade_clients_by_cid = (
            lambda cid, ev: cascade_dispatches.append((cid, ev))
        )
        return client_emits, cascade_dispatches

    def test_routes_to_both_client_and_cascade_when_both_set(self):
        """When direct_client_id AND cascade_driver_id are both set,
        the event reaches both the requesting client AND cascade
        observers.  Note: this is the case where duplicate delivery
        can occur for a same-client cascade subscriber — documented
        in _route_bootstrap_event docstring."""
        sm = _make_sm()
        client_emits, cascade_dispatches = self._setup_sm_with_emit_stubs(sm)
        event = MagicMock()
        sm._route_bootstrap_event("ipc_1", "cid-abc", event)
        assert client_emits == [("ipc_1", event)]
        assert cascade_dispatches == [("cid-abc", event)]

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
        cid is set, direct client emit is skipped."""
        sm = _make_sm()
        client_emits, cascade_dispatches = self._setup_sm_with_emit_stubs(sm)
        event = MagicMock()
        sm._route_bootstrap_event(None, "cid-abc", event)
        assert client_emits == []
        assert cascade_dispatches == [("cid-abc", event)]

    def test_headless_cascade_bootstrap_reaches_cascade_observers(self):
        """The empirical scenario peer 7:1 hit: reactor-spawned
        downstream cascade session bootstraps with
        client_id=_HEADLESS_CLIENT_ID (transport-dropped).  Before
        0.6.166 the AgentCreatedEvent was lost.  Now the route_helper
        delivers it to cascade observers via the cid path."""
        sm = _make_sm()
        client_emits, cascade_dispatches = self._setup_sm_with_emit_stubs(sm)
        event = MagicMock()
        sm._route_bootstrap_event(sm._HEADLESS_CLIENT_ID, "cid-abc", event)
        # Direct emit still fires (the transport drops it; the
        # helper doesn't second-guess the transport's job).
        assert client_emits == [(sm._HEADLESS_CLIENT_ID, event)]
        # Critical: cascade observers reached via the cid path.
        assert cascade_dispatches == [("cid-abc", event)]

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
