"""``session.list`` / ``session.attach`` are scoped to what the caller may see (protocol 1.13).

The transport reports a boundary through ``visible_workspace_paths``: the
paths of the workspaces the connection's user may see, or ``None`` when no
scoping applies (IPC; a WS connection with no identity).  Pinned here:

- ``None`` is the unscoped listing every client always got;
- with a boundary, a session is shown when it runs inside one of those
  workspaces OR when this user created it, and hidden otherwise -- another
  user's session in a shared workspace included;
- ``session.attach`` admits exactly the shown set, and refuses the rest by
  name without touching the session manager;
- a sink predating the method (out of tree) contributes "no scoping".
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from jaato_sdk.events import ErrorEvent, SessionListEvent

from server.command_router import CommandRouter


def _sess(sid, workspace, created_by=None):
    return SimpleNamespace(
        session_id=sid, name=sid, description="", model_provider="", model_name="",
        is_loaded=True, client_count=0, turn_count=0, workspace_path=workspace,
        created_by=created_by, orphaned=False, runner=None,
    )


SESSIONS = [
    _sess("in-mine", "/root/alices"),
    _sess("in-mine-by-bob", "/root/alices", created_by="app:bob"),
    _sess("in-shared", "/root/legacy"),
    _sess("in-bobs", "/root/bobs", created_by="app:bob"),
    _sess("alices-elsewhere", "/elsewhere", created_by="app:alice"),
    _sess("nowhere", None),
]


def _router(paths, user="app:alice", sink_has_method=True):
    sm = MagicMock()
    sm.list_sessions.return_value = SESSIONS
    sm.check_workspace_mismatch.return_value = None
    sm.attach_session.return_value = False
    sink = MagicMock(spec=["send_event", "get_client_user", "get_client_workspace",
                           "set_client_session"] + (["visible_workspace_paths"] if sink_has_method else []))
    sink.get_client_user.return_value = user
    sink.get_client_workspace.return_value = None
    if sink_has_method:
        sink.visible_workspace_paths.return_value = paths
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = sm
    router._event_sink = sink
    router._pending_workspace_mismatch = {}
    return router, sm, sink


def _listed(sink):
    ev = [c[0][1] for c in sink.send_event.call_args_list if isinstance(c[0][1], SessionListEvent)][0]
    return {s["id"] for s in ev.sessions}


def test_no_boundary_is_the_unscoped_listing():
    router, _, sink = _router(None)
    router._handle_session_list("c1", None)
    assert _listed(sink) == {s.session_id for s in SESSIONS}


def test_a_boundary_keeps_the_users_workspaces_and_the_users_own_sessions():
    router, _, sink = _router(["/root/alices", "/root/legacy"])
    router._handle_session_list("c1", None)
    # in a visible workspace: yes, whoever created it -- a workspace is shared
    # by construction.  Elsewhere: only what this user created.
    assert _listed(sink) == {"in-mine", "in-mine-by-bob", "in-shared", "alices-elsewhere"}


def test_an_empty_boundary_is_a_boundary_not_an_absence():
    router, _, sink = _router([])
    router._handle_session_list("c1", None)
    assert _listed(sink) == {"alices-elsewhere"}


def test_a_sink_without_the_method_contributes_no_scoping():
    router, _, sink = _router(None, sink_has_method=False)
    router._handle_session_list("c1", None)
    assert _listed(sink) == {s.session_id for s in SESSIONS}


def test_attach_refuses_a_hidden_session_by_name_and_admits_a_shown_one():
    router, sm, sink = _router(["/root/alices"])
    router._handle_session_attach("c1", None, ["in-bobs"], None)
    err = [c[0][1] for c in sink.send_event.call_args_list if isinstance(c[0][1], ErrorEvent)]
    assert err and "in-bobs" in err[0].error and err[0].error_type == "SessionError"
    sm.attach_session.assert_not_called()

    router._handle_session_attach("c1", None, ["in-mine"], None)
    sm.attach_session.assert_called_once()


def test_attach_is_unguarded_without_a_boundary():
    router, sm, _ = _router(None)
    router._handle_session_attach("c1", None, ["in-bobs"], None)
    sm.attach_session.assert_called_once()


def test_only_a_list_of_paths_is_a_boundary():
    """A sink answering anything else -- a test double's auto-attribute, a
    non-sequence -- contributes no scoping rather than an empty boundary
    that would hide every session."""
    from server.event_sink import client_visible_workspaces
    assert client_visible_workspaces(MagicMock(), "c1") is None
    assert client_visible_workspaces(SimpleNamespace(visible_workspace_paths=lambda c: "/root"), "c1") is None
    assert client_visible_workspaces(SimpleNamespace(visible_workspace_paths=lambda c: [1, 2]), "c1") is None
    assert client_visible_workspaces(SimpleNamespace(visible_workspace_paths=lambda c: ("/a",)), "c1") == ["/a"]
    assert client_visible_workspaces(SimpleNamespace(visible_workspace_paths=lambda c: []), "c1") == []
