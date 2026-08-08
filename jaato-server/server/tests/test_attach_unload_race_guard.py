"""Attach-vs-unload race guard (a): attach_session must not attach to a session
whose async unload has committed (runner mid-disposal). It awaits the unload
off-lock, then bails cleanly if the marker is still present (atomic with the
client-add)."""
import threading
import pytest
from server.session_manager import SessionManager


def _bare_sm():
    sm = SessionManager.__new__(SessionManager)   # skip heavy __init__
    sm._unloading = {}
    sm._lock = threading.RLock()
    sm._sessions = {}
    sm._client_to_session = {}
    sm._emitted = []
    sm._emit_to_client = lambda cid, evt: sm._emitted.append((cid, evt))
    return sm


def test_attach_bails_when_session_unloading():
    sm = _bare_sm()
    evt = threading.Event(); evt.set()            # await returns immediately; marker still set
    sm._unloading["s1"] = evt
    ok = sm.attach_session("c1", "s1")
    assert ok is False                            # bailed, did not attach to the corpse
    assert sm._emitted, "expected a SessionError emitted to the client"
    cid, ev = sm._emitted[0]
    assert cid == "c1" and ev.error_type == "SessionError"
    assert "unloaded" in ev.error.lower()


def test_attach_awaits_then_clears(monkeypatch):
    # A background "unload" clears the marker + signals the event shortly after
    # attach starts waiting; attach must wake, see the marker gone, and NOT bail
    # on the unloading-path (it proceeds past the guard).
    sm = _bare_sm()
    evt = threading.Event()
    sm._unloading["s1"] = evt

    def finish_unload():
        with sm._lock:
            sm._unloading.pop("s1", None)
        evt.set()

    # Stub the post-guard work so we isolate the guard: once past the guard,
    # attach hits _sessions.get → None → _load_session. Replace _load_session
    # to return None so attach returns the ordinary "not found", NOT the
    # "being unloaded" bail — proving the guard let it through.
    monkeypatch.setattr(sm, "_load_session", lambda *a, **k: None)
    t = threading.Timer(0.2, finish_unload); t.start()
    ok = sm.attach_session("c1", "s1")
    t.join()
    assert ok is False                            # "not found", not the unload bail
    errs = [ev.error.lower() for _, ev in sm._emitted]
    assert errs and all("being unloaded" not in e for e in errs)
