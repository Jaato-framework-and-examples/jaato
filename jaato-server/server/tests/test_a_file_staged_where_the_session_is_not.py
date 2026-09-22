"""An attached file was staged into a workspace the session did not run in.

Reported from a deployed client, on a workspace the user had created and
named: every indicator said the file was in the workspace -- the chip went
``staged``, the transcript said *"Staged into the workspace: X.jpg"*, the
user turn carried *"Attached files, staged in the workspace: X.jpg"* -- and
the model could not read it, at the relative path or at the workspace's own
absolute path, while the FILES panel showed only the session's ``.jaato/``
and no new file.

Neither client was lying.  ``staging.ts`` marks a chip ``staged`` only for a
name the daemon echoed back in ``StageFilesEvent.staged``, and the daemon
appends a name only after ``_write_staged_payload`` returned -- so the bytes
really were written, somewhere.

**Two definitions of "this client's workspace", and staging used the wrong
one.**  ``CommandRouter.resolve_caller_workspace`` is the daemon's answer --
the attached session's workspace, else the transport's session, else what
the client declared -- and it already records why staging needs that exact
order: *a session's path outranks a declared one because the session's tree
is what ``WorkspaceMonitor`` watches and what the panel shows.*
``_resolve_staging_workspace`` answered separately, reading
``_client_provisioned`` first: a map stamped by an auto-provisioning
``session.new`` and cleared only on DISCONNECT, never by a later
``workspace.select``.  Measured over the real WS wire, one connection:

===============================  ================  ==============
order                            session runs in   file lands in
===============================  ================  ==============
``select, new``                  named             named
``new, select``                  provisioned       provisioned
``new, select, new``             **named**         **provisioned**
===============================  ================  ==============

Only the third diverges, which is why it reads as intermittent, and it is an
ordinary flow: create a session, end it (in workspace mode that returns to
the workspace list), open a named workspace, create a session, attach.

Reordering the two stores would have fixed that row and broken the second,
so both directions are pinned here:

* ``test_a_stale_provisioned_workspace_does_not_capture_the_attach``
  (the report) fails if the provisioned map is read first;
* ``test_a_client_that_selected_elsewhere_still_stages_into_its_session``
  fails if the client's declaration is read first.

The router's resolver is exercised for real rather than restated, because a
stub of it would assert this file's opinion of the ordering instead of the
daemon's.  Every case is paired with the same call one store apart: a
resolution that would have succeeded anyway proves nothing.
"""
from __future__ import annotations

from typing import Dict, Optional

import pytest

from server.command_router import CommandRouter
from server.websocket import JaatoWSServer
from shared.tests.reversion import Reversion

_WS = "jaato-server/server/websocket.py"

REVERSIONS = [
    Reversion(
        target=_WS,
        find="""        router = getattr(self, "_command_router", None)
        if router is not None and hasattr(router, "resolve_caller_workspace"):
            session_id = adapter._client_sessions.get(client_id) if adapter else None
            current_path, _sources = router.resolve_caller_workspace(
                client_id, declared, session_id,
            )
        else:
            current_path = declared

        if not current_path:
            provisioned = self._client_provisioned.get(client_id)""",
        replace="""        current_path = declared

        if not current_path:
            provisioned = self._client_provisioned.get(client_id)""",
        test="test_a_client_that_selected_elsewhere_still_stages_into_its_session",
        because=(
            "the client's declaration is not where the session runs: after "
            "an auto-provisioned session the user may select another "
            "workspace, and a file attached to that session belongs in the "
            "session's tree, which is the one the files panel watches"
        ),
    ),
    Reversion(
        target=_WS,
        # The pre-fix body, verbatim: a second answer to "which workspace
        # is this client in", reading a map the router never consults.
        find="""        adapter = self._event_sink_adapter
        declared = adapter.get_client_workspace(client_id) if adapter else None
        if not declared and self._workspace_manager is not None:
            selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
            declared = selected.path if selected else None

        router = getattr(self, "_command_router", None)
        if router is not None and hasattr(router, "resolve_caller_workspace"):
            session_id = adapter._client_sessions.get(client_id) if adapter else None
            current_path, _sources = router.resolve_caller_workspace(
                client_id, declared, session_id,
            )
        else:
            current_path = declared

        if not current_path:
            provisioned = self._client_provisioned.get(client_id)
            current_path = provisioned.path if provisioned is not None else None""",
        replace="""        provisioned = self._client_provisioned.get(client_id)
        if provisioned is not None:
            current_path = provisioned.path
        elif self._workspace_manager is not None:
            selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
            current_path = selected.path if selected else None
        else:
            current_path = None""",
        test="test_a_stale_provisioned_workspace_does_not_capture_the_attach",
        because=(
            "the reported defect: _client_provisioned is stamped once by an "
            "auto-provisioning session.new and never cleared, so reading it "
            "first sends every later attach into a directory no session runs "
            "in, with success reported"
        ),
    ),
    Reversion(
        target=_WS,
        find="""        if not current_path:
            provisioned = self._client_provisioned.get(client_id)
            current_path = provisioned.path if provisioned is not None else None""",
        replace="",
        test="test_the_provisioned_map_is_the_last_resort_not_the_first",
        because=(
            "demoting the provisioned map must not DELETE it: "
            "provision_workspace() stamps it without telling the adapter or "
            "the router, so a caller reaching it directly would stage nowhere"
        ),
    ),
]

NAMED = "/srv/workspaces/test_attachment"
PROVISIONED = "/srv/workspaces/sessions/ws_e36320bd"
CLIENT = "client_31"
SESSION = "20260922_193245"


class _Provisioned:
    """Only ``.path`` is read from a :class:`ProvisionedWorkspace` here."""

    def __init__(self, path: str) -> None:
        self.path = path


class _Session:
    def __init__(self, workspace_path: Optional[str]) -> None:
        self.workspace_path = workspace_path


class _SessionManager:
    """What :meth:`CommandRouter.resolve_caller_workspace` reads."""

    def __init__(self, attached: Optional[str]) -> None:
        self._attached = attached

    def get_client_session(self, client_id: str) -> Optional[_Session]:
        return _Session(self._attached) if self._attached and client_id == CLIENT else None

    def get_session(self, session_id: str) -> Optional[_Session]:
        return _Session(self._attached) if self._attached and session_id == SESSION else None


class _Adapter:
    def __init__(self, path: Optional[str], sessions: Dict[str, str]) -> None:
        self._path = path
        self._client_sessions = sessions

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        return self._path if client_id == CLIENT else None


class _Selected:
    def __init__(self, path: str) -> None:
        self.path = path


class _Manager:
    def __init__(self, path: Optional[str]) -> None:
        self._path = path

    def get_selected_workspace(self, client_id: str) -> Optional[_Selected]:
        return _Selected(self._path) if self._path and client_id == CLIENT else None


def _server(
    *,
    session_in: Optional[str] = None,
    declared: Optional[str] = None,
    selected: Optional[str] = None,
    provisioned: Optional[str] = None,
    adapter: bool = True,
) -> JaatoWSServer:
    """A server carrying only what the resolver reads.

    ``__new__`` on both classes deliberately: constructing the real ones
    binds a socket, an event loop and a session manager, and nothing else
    on either object is reachable from ``_resolve_staging_workspace``.
    """
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = _SessionManager(session_in)

    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._command_router = router
    ws._event_sink_adapter = (
        _Adapter(declared, {CLIENT: SESSION} if session_in else {}) if adapter else None
    )
    ws._workspace_manager = _Manager(selected)
    ws._client_provisioned = {CLIENT: _Provisioned(provisioned)} if provisioned else {}
    return ws


# --------------------------------------------------- the two directions

def test_a_stale_provisioned_workspace_does_not_capture_the_attach():
    """``new, select, new``: the session is in the named workspace."""
    stale = _server(session_in=NAMED, declared=NAMED, provisioned=PROVISIONED)
    assert stale._resolve_staging_workspace(CLIENT, "") == NAMED

    clean = _server(session_in=NAMED, declared=NAMED, selected=NAMED)
    assert clean._resolve_staging_workspace(CLIENT, "") == NAMED


def test_a_client_that_selected_elsewhere_still_stages_into_its_session():
    """``new, select``: the session is the auto-provisioned one."""
    diverged = _server(session_in=PROVISIONED, declared=NAMED, provisioned=PROVISIONED)
    assert diverged._resolve_staging_workspace(CLIENT, "") == PROVISIONED

    # The control: with no session, the declaration IS the answer.
    picker = _server(declared=NAMED, provisioned=PROVISIONED)
    assert picker._resolve_staging_workspace(CLIENT, "") == NAMED


# ------------------------------------------------------- the id guard

def test_an_explicit_workspace_id_is_matched_against_the_resolved_workspace():
    """The loud half: a named id was refused for the client's own workspace."""
    stale = _server(session_in=NAMED, declared=NAMED, provisioned=PROVISIONED)
    assert stale._resolve_staging_workspace(CLIENT, "test_attachment") == NAMED

    clean = _server(session_in=NAMED, declared=NAMED)
    assert clean._resolve_staging_workspace(CLIENT, "test_attachment") == NAMED


def test_an_id_naming_another_workspace_is_still_refused():
    ws = _server(session_in=NAMED, declared=NAMED, provisioned=PROVISIONED)
    assert ws._resolve_staging_workspace(CLIENT, "somebody_else") is None


# --------------------------------------------------- the tiers below

def test_a_client_with_no_adapter_answer_falls_back_to_the_selection():
    ws = _server(adapter=False, selected=NAMED)
    assert ws._resolve_staging_workspace(CLIENT, "") == NAMED


def test_the_provisioned_map_is_the_last_resort_not_the_first():
    """``provision_workspace`` stamps it without telling anyone else."""
    ws = _server(adapter=False, provisioned=PROVISIONED)
    assert ws._resolve_staging_workspace(CLIENT, "") == PROVISIONED


def test_a_client_with_no_workspace_at_all_is_refused():
    ws = _server(adapter=False)
    assert ws._resolve_staging_workspace(CLIENT, "") is None


@pytest.mark.parametrize("workspace_id", ["", "ws_e36320bd"])
def test_the_auto_provisioned_session_still_stages_into_its_own_workspace(workspace_id):
    """``new`` alone: the embed flow this map was written for is unaffected."""
    ws = _server(session_in=PROVISIONED, declared=PROVISIONED, provisioned=PROVISIONED)
    assert ws._resolve_staging_workspace(CLIENT, workspace_id) == PROVISIONED
