"""SessionManager.set_session_state_for_session: cross-session state-write
routing primitive (the session-state sibling of inject_prompt_to_session) used
by the reliability T3 gate.released resume to write approved-tools into the
PARKED session — which is not the originating session."""
import threading
from types import SimpleNamespace
from server.session_manager import SessionManager


def _bare_sm(sessions):
    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._sessions = sessions
    return sm


def test_routes_to_target_session_runner():
    calls = []
    rpc = SimpleNamespace(
        session_set_state_threadsafe=lambda k, v, timeout=None: calls.append((k, v)))
    session = SimpleNamespace(server=SimpleNamespace(_runner_rpc=rpc))
    sm = _bare_sm({"s1": session})
    assert sm.set_session_state_for_session("s1", "reliability:approved_tools", ["svc"]) is True
    assert calls == [("reliability:approved_tools", ["svc"])]


def test_unloaded_target_returns_false():
    assert _bare_sm({}).set_session_state_for_session("nope", "k", "v") is False


def test_no_runner_returns_false():
    session = SimpleNamespace(server=SimpleNamespace(_runner_rpc=None))
    sm = _bare_sm({"s1": session})
    assert sm.set_session_state_for_session("s1", "k", "v") is False
