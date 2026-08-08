"""SessionManager.send_message_to_session DRIVES a turn on a loaded session (same id).

The T1 reactor-driven resume primitive: unlike inject_prompt_to_session (which
QUEUES), this dispatches a SendMessageRequest via handle_request — the same
daemon path a client send takes — so an idle headless session actually runs a
turn, keeping the SAME session id (no fork-from-history).
"""
import threading

from server.session_manager import SessionManager


def _bare_sm():
    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.Lock()
    sm._sessions = {}
    sm._HEADLESS_CLIENT_ID = "__headless__"
    sm._calls = []
    sm.handle_request = lambda cid, sid, req: sm._calls.append((cid, sid, req))
    return sm


def test_drives_turn_on_loaded_session_same_id():
    from jaato_sdk.events import SendMessageRequest
    sm = _bare_sm()
    sm._sessions["sid"] = object()   # loaded
    assert sm.send_message_to_session("sid", "retry") is True
    assert len(sm._calls) == 1
    cid, sid, req = sm._calls[0]
    assert cid == sm._HEADLESS_CLIENT_ID
    assert sid == "sid"              # SAME id — no fork
    assert isinstance(req, SendMessageRequest) and req.text == "retry"


def test_returns_false_when_session_not_loaded():
    sm = _bare_sm()
    assert sm.send_message_to_session("nope", "retry") is False
    assert sm._calls == []           # no turn dispatched
