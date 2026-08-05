"""SessionManager.get_persisted_history reads a persisted session's history off
disk WITHOUT loading the session (no JaatoServer, no runner spawn).

The jaato-server piece of the §9 fork-from-persisted reactor resume (T2): the
reactor reads the persisted history here and feeds it to create_headless_session
to fork a continuation (new id) from an unloaded/ended session.
"""
from unittest.mock import MagicMock

from server.session_manager import SessionManager


def _bare_sm():
    sm = SessionManager.__new__(SessionManager)
    sm._session_plugin = MagicMock()
    sm._session_storage_dir = lambda ws: f"{ws}/.jaato/sessions"
    return sm


def test_returns_persisted_history():
    sm = _bare_sm()
    state = MagicMock(); state.history = [{"role": "user", "content": "hi"}]
    sm._session_plugin.load.return_value = state
    assert sm.get_persisted_history("sid", "/ws") == [{"role": "user", "content": "hi"}]
    sm._session_plugin.load.assert_called_once_with("sid", storage_dir="/ws/.jaato/sessions")


def test_missing_record_returns_none():
    sm = _bare_sm()
    sm._session_plugin.load.side_effect = FileNotFoundError()
    assert sm.get_persisted_history("nope", "/ws") is None


def test_no_workspace_uses_default_storage():
    sm = _bare_sm()
    state = MagicMock(); state.history = []
    sm._session_plugin.load.return_value = state
    assert sm.get_persisted_history("sid") == []
    sm._session_plugin.load.assert_called_once_with("sid", storage_dir=None)
