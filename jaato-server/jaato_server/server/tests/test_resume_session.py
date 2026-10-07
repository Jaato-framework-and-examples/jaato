"""SessionManager.resume_session — public same-id reload + presentation restore.

The reliability T2 resume (Daniel's redirect away from fork-from-persisted): a
parked session unloaded to free the runner, then revived on late human approval,
RESUMES under the SAME id with full fidelity (history/state/profile/whitelist) +
the headless API presentation re-applied (presentation is client-config, not
persisted, so the primitive must set it). Pairs with send_message_to_session.
"""
from unittest.mock import MagicMock

from jaato_server.server.session_manager import SessionManager


def _bare_sm():
    sm = SessionManager.__new__(SessionManager)
    sm._HEADLESS_CLIENT_ID = "_headless"
    sm._load_calls = []
    sm._apply_calls = []
    sm._load_return = None
    # ``load_reason`` says WHICH waking this is, and resume_session is the one
    # caller that is a deliberate wake (#1157) -- the announcement record reads
    # it to tell a wake from a grace-expired reattach, so the stub records it
    # rather than swallowing it with **kwargs.
    sm._load_session = lambda sid, client_id=None, workspace_path=None, load_reason="reattach": (
        sm._load_calls.append((sid, client_id, workspace_path, load_reason))
        or sm._load_return)
    sm._apply_client_config_to_server = (
        lambda cid, server: sm._apply_calls.append((cid, server)))
    return sm


def test_resume_reloads_same_id_and_restores_presentation():
    sm = _bare_sm()
    session = MagicMock(); session.server = MagicMock()
    sm._load_return = session
    assert sm.resume_session("sid", "/ws") == "sid"          # SAME id back
    # reloaded headless, right ws, recorded as a WAKE rather than a reattach
    assert sm._load_calls == [("sid", "_headless", "/ws", "wake")]
    assert sm._apply_calls == [("_headless", session.server)]  # API presentation restored


def test_resume_returns_none_when_record_missing():
    sm = _bare_sm()
    sm._load_return = None
    assert sm.resume_session("nope", "/ws") is None
    assert sm._apply_calls == []   # no presentation apply on a failed load
