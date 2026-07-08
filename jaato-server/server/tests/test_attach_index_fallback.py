"""WS workspace-pinless resume: attach + list_sessions fall back to the
session-workspace index.

The gap: the wake path resolves a cold session's workspace server-side by id
(``_wake_impl`` → ``resume_session``), but the client-driven attach /
list_sessions paths trusted the reconnecting client's ``workspace_path``.
An IPC client's ``working_dir`` *is* the session's workspace, so it works;
a browser over WS lives in a server-provisioned ``ws_<hash>`` dir it can't
present, so cold-resume missed. These pin the fix: client-workspace-first,
index-fallback — so ``jaato.session(mode="ws", recovery=True).attach_session
(id)`` cold-resumes with zero client-side workspace knowledge (IPC parity).
"""
from unittest.mock import MagicMock

from server.session_manager import SessionManager


class _FakeIndex:
    def __init__(self, mapping=None):
        self._m = dict(mapping or {})

    def resolve(self, session_id):
        return self._m.get(session_id)

    def workspaces(self):
        return set(self._m.values())


def _bare_sm(index_mapping=None):
    sm = SessionManager.__new__(SessionManager)
    sm._session_workspace_index = _FakeIndex(index_mapping)
    sm._load_calls = []

    # _load_session succeeds ONLY when asked for the workspace the record
    # actually lives in (recorded in _record_ws), mirroring the disk layout.
    sm._record_ws = None

    def _fake_load(sid, client_id=None, workspace_path=None):
        sm._load_calls.append(workspace_path)
        if workspace_path == sm._record_ws:
            s = MagicMock()
            s.session_id = sid
            return s
        return None

    sm._load_session = _fake_load
    return sm


class TestAttachIndexFallback:
    def test_pinned_client_loads_directly_no_index(self):
        # IPC: client working_dir IS the record's workspace → first load wins,
        # index never consulted.
        sm = _bare_sm(index_mapping={"s1": "/ws/ghost"})  # would mislead if used
        sm._record_ws = "/home/proj"
        session = sm._load_persisted_with_index_fallback("s1", "c1", "/home/proj")
        assert session is not None
        assert sm._load_calls == ["/home/proj"]  # exactly one attempt

    def test_pinless_client_falls_back_to_index(self):
        # WS browser: client can't present ws_<hash>; record lives there;
        # index resolves it → second load succeeds.
        WS = "/home/nano/.jaato/workspaces/sessions/ws_deadbeef"
        sm = _bare_sm(index_mapping={"s1": WS})
        sm._record_ws = WS
        session = sm._load_persisted_with_index_fallback(
            "s1", "c1", "/home/nano/nano-chat-workspace")
        assert session is not None
        assert session.session_id == "s1"
        assert sm._load_calls == ["/home/nano/nano-chat-workspace", WS]

    def test_unknown_id_no_fallback_returns_none(self):
        sm = _bare_sm(index_mapping={})  # nothing to resolve
        sm._record_ws = "/somewhere"
        session = sm._load_persisted_with_index_fallback("s1", "c1", "/client/ws")
        assert session is None
        assert sm._load_calls == ["/client/ws"]  # only the client attempt

    def test_ambiguous_id_refused_returns_none(self):
        # Ambiguous ids resolve() to None → no fallback (never guess).
        sm = _bare_sm(index_mapping={})  # _FakeIndex returns None like a refusal
        sm._record_ws = "/real/ws"
        assert sm._load_persisted_with_index_fallback("s1", "c1", "/client/ws") is None

    def test_no_redundant_reload_when_index_equals_client_ws(self):
        # If the index resolves to the SAME workspace the client already tried,
        # don't reload — but here that ws has no record, so still None, one call.
        sm = _bare_sm(index_mapping={"s1": "/client/ws"})
        sm._record_ws = "/elsewhere"
        assert sm._load_persisted_with_index_fallback("s1", "c1", "/client/ws") is None
        assert sm._load_calls == ["/client/ws"]  # index==client → not retried
