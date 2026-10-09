"""A ticket client cannot name a path outside its application's root (#1592).

A WS connection authenticated with a user ticket (#1074) belongs to an
application, and #1496 puts that application's workspaces under its own
``workspace_root``.  The workspace verbs enforce that root through
``_workspace_manager_for`` / ``resolve_visible``.  Two routes did not:

* ``ClientConfigRequest.working_dir`` / ``config_root`` / ``env_file`` and
  the two trace paths -- only absoluteness was checked (#742), and the
  ``SO_PEERCRED`` check (#721's "Two Principals") is inert on WS;
* ``set_workspace <path>`` -- the same, one verb over.

Either let ``session.new`` run a session anywhere on the host with the
daemon's credential.  Every case here drives the real WS dispatch
(``_handle_message_daemon``) into a real ``CommandRouter`` and a real
``SessionManager``, with the connection's identity established by the real
ticket bind and connect-time resolver.  What is observed is what the
manager STORED (the config ``session.new`` reads) and what the router would
resolve as the caller's workspace.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import (
    ClientConfigRequest,
    CommandRequest,
    ErrorEvent,
    TicketBindRequest,
    deserialize_event,
    serialize_event,
)
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.event_sink import CompositeEventSink
from jaato_server.server.session_manager import SessionManager
from jaato_server.server.websocket import JaatoWSServer
from jaato_server.server.workspace_manager import WorkspaceManager
from jaato_server.server.ws_tickets import AppCredentialStore, AppWorkspace
from jaato_server.shared.tests.reversion import Reversion

_WS = "jaato-server/jaato_server/server/websocket.py"
_WM = "jaato-server/jaato_server/server/workspace_manager.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"
_CR = "jaato-server/jaato_server/server/command_router.py"
_ES = "jaato-server/jaato_server/server/event_sink.py"

REVERSIONS = [
    Reversion(
        target=_WS,
        find="        if app_ws is None:\n            return None\n",
        replace="        if True:\n            return None\n",
        test="test_a_working_dir_outside_the_root_is_refused",
        because="a ticket client runs a session in any directory on the host",
    ),
    Reversion(
        target=_SM,
        find=(
            "        if self._reject_client_paths_outside_scope(client_id, event):\n"
            "            return\n"
        ),
        replace="",
        test="test_a_working_dir_outside_the_root_is_refused",
        because="the handshake applies whatever the ticket client declared",
    ),
    Reversion(
        target=_CR,
        find="        _lend_path_scope(session_manager, self._event_sink)\n",
        replace="",
        test="test_a_working_dir_outside_the_root_is_refused",
        because="the manager has no way to ask the transport, so nothing is checked",
    ),
    Reversion(
        target=_ES,
        find="            answer = client_path_refusals(sink, client_id, fields)\n",
        replace="            answer = None\n",
        test="test_a_working_dir_outside_the_root_is_refused",
        because="the daemon's composite sink drops the WS transport's verdict",
    ),
    Reversion(
        target=_WM,
        find="        resolved = candidate.resolve()\n",
        replace="        resolved = Path(os.path.abspath(candidate))\n",
        test="test_a_symlink_out_of_the_root_is_refused",
        because="a link planted under the root is judged by its name, not its target",
    ),
    Reversion(
        target=_WM,
        find="    if not WorkspaceManager.visible_to(\n",
        replace="    if False and not WorkspaceManager.visible_to(\n",
        test="test_another_users_workspace_is_refused",
        because="one user of an application runs sessions in another's workspace",
    ),
    Reversion(
        target=_CR,
        find=(
            "        refusals = client_path_refusals(\n"
            "            self._event_sink, client_id, [(\"workspace\", workspace_path)])\n"
        ),
        replace="        refusals = []\n",
        test="test_set_workspace_outside_the_root_is_refused",
        because="set_workspace hands session.new a workspace anywhere on the host",
    ),
]

_APP_TOKEN = "app-credential-0123456789abcdef"
_OTHER_TOKEN = "other-credential-0123456789abcd"
_SHARED_TOKEN = "shared-bearer-token"


def _fake_ws(credential: str) -> MagicMock:
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {credential}"}
    request.path = "/"
    ws = MagicMock()
    ws.request = request
    return ws


def _app_ws(root: Path) -> AppWorkspace:
    return AppWorkspace(account=getpass.getuser(), uid=os.getuid(),
                        gid=os.getgid(), workspace_root=os.path.realpath(root))


class _Daemon:
    """A WS server, router and manager wired as ``JaatoDaemon.start`` wires them.

    Never binds a port.  The manager's client emissions are captured so a
    refusal can be read; everything else is the real code.
    """

    def __init__(self, tmp_path: Path, *, with_manager: bool = True) -> None:
        self.acme_root = tmp_path / "acme-root"
        self.other_root = tmp_path / "other-root"
        self.outside = tmp_path / "elsewhere"
        for d in (self.acme_root, self.other_root, self.outside):
            d.mkdir()
        self.srv = JaatoWSServer(
            host="127.0.0.1", port=0, required_token=_SHARED_TOKEN,
            app_credentials=AppCredentialStore(
                {"acme": _APP_TOKEN, "other": _OTHER_TOKEN},
                {"acme": _app_ws(self.acme_root),
                 "other": _app_ws(self.other_root)},
            ),
        )
        if with_manager:
            self.manager = WorkspaceManager(
                str(self.acme_root), registry_path=tmp_path / "acme.json")
            self.srv._app_managers["acme"] = self.manager
        self.sm = SessionManager()
        self.emitted: List[tuple] = []
        self.sm._emit_to_client = lambda cid, ev: self.emitted.append((cid, ev))
        self.ipc_like = MagicMock(spec=["send_event", "broadcast_event",
                                        "get_client_workspace", "set_client_workspace",
                                        "set_client_session", "get_client_user",
                                        "get_client_peer"])
        self.ipc_like.get_client_peer.return_value = None
        self.ipc_like.get_client_user.return_value = None
        self.ipc_like.get_client_workspace.return_value = None
        sink = CompositeEventSink()
        sink.add_sink(self.ipc_like)
        sink.add_sink(self.srv.get_event_sink_adapter())
        self.router = CommandRouter(
            session_manager=self.sm, event_sink=sink, daemon_plugins={})
        self.srv._command_router = self.router

    def attach(self, client_id: str, credential: str) -> None:
        from jaato_server.server.websocket import ClientConnection
        auth = self.srv._resolve_connection_auth(_fake_ws(credential))
        assert auth is not None
        self.srv._clients[client_id] = ClientConnection(
            websocket=MagicMock(), client_id=client_id,
            connected_at="2026-01-01T00:00:00+00:00", subscriptions=set(),
            user_id=auth.user, app_id=auth.app_id, auth_kind=auth.kind,
        )

    async def ticket(self, client_id: str, user: str = "alice") -> None:
        """Mint a ticket through ``ticket.bind`` and connect with it."""
        self.attach("app-conn", _APP_TOKEN)
        await self.srv._dispatch_client_message(
            "app-conn", serialize_event(TicketBindRequest(request_id="r", user=user)))
        sent = self.srv._clients["app-conn"].websocket.send.call_args_list
        minted = deserialize_event(sent[-1].args[0])
        assert minted.status == "bound", minted
        self.attach(client_id, minted.ticket)
        assert self.srv.get_client_user(client_id) == f"acme:{user}"

    async def config(self, client_id: str, **fields) -> None:
        await self.srv._handle_message_daemon(
            client_id, ClientConfigRequest(**fields))

    def stored(self, client_id: str) -> dict:
        return dict(self.sm._client_config.get(client_id, {}))

    def refusals(self, client_id: str) -> List[ErrorEvent]:
        return [ev for cid, ev in self.emitted
                if cid == client_id and isinstance(ev, ErrorEvent)]


@pytest.fixture(autouse=True)
def _restore_trace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # An ACCEPTED handshake writes the two trace paths into os.environ.
    monkeypatch.setenv("JAATO_TRACE_LOG", "")
    monkeypatch.setenv("JAATO_PROVIDER_TRACE", "")


def _refused(d: _Daemon, client_id: str, field: str) -> None:
    errors = d.refusals(client_id)
    assert errors, f"nothing refused for {client_id}"
    assert errors[-1].error_type == "ClientPathOutsideWorkspaceRoot"
    assert field in errors[-1].error
    assert d.stored(client_id) == {}


# ----------------------------------------------------------------- refusals

@pytest.mark.asyncio
async def test_a_working_dir_outside_the_root_is_refused(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    await d.ticket("user")
    await d.config("user", working_dir=str(d.outside))
    _refused(d, "user", "working_dir")
    # session.new reads its workspace from these two places; neither names it.
    workspace, _ = d.router.resolve_caller_workspace(
        "user", d.router._event_sink.get_client_workspace("user"))
    assert workspace is None


@pytest.mark.asyncio
async def test_the_rule_holds_with_no_manager_for_the_application(
    tmp_path: Path,
) -> None:
    """A daemon with no workspace mode of its own builds no app managers."""
    d = _Daemon(tmp_path, with_manager=False)
    await d.ticket("user")
    await d.config("user", working_dir=str(d.outside))
    _refused(d, "user", "working_dir")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["config_root", "env_file", "trace_log_path", "provider_trace_log"])
async def test_every_path_field_is_held_to_the_root(
    tmp_path: Path, field: str,
) -> None:
    d = _Daemon(tmp_path)
    await d.ticket("user")
    await d.config("user", **{field: str(d.outside / "x")})
    _refused(d, "user", field)


@pytest.mark.asyncio
async def test_a_symlink_out_of_the_root_is_refused(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    (d.acme_root / "planted").symlink_to(d.outside, target_is_directory=True)
    await d.ticket("user")
    await d.config("user", working_dir=str(d.acme_root / "planted"))
    _refused(d, "user", "working_dir")


@pytest.mark.asyncio
async def test_dot_dot_out_of_the_root_is_refused(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    await d.ticket("user")
    await d.config("user", working_dir=f"{d.acme_root}/ws/../../elsewhere")
    _refused(d, "user", "working_dir")


@pytest.mark.asyncio
async def test_the_root_itself_is_not_a_workspace(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    await d.ticket("user")
    await d.config("user", working_dir=str(d.acme_root))
    _refused(d, "user", "working_dir")


@pytest.mark.asyncio
async def test_another_applications_root_is_refused(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    (d.other_root / "theirs").mkdir()
    await d.ticket("user")
    await d.config("user", working_dir=str(d.other_root / "theirs"))
    _refused(d, "user", "working_dir")


@pytest.mark.asyncio
async def test_another_users_workspace_is_refused(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    bobs = d.manager.create_workspace("bobs", owner="acme:bob")
    await d.ticket("user", user="alice")
    await d.config("user", working_dir=bobs.path)
    _refused(d, "user", "working_dir")
    assert "belongs to another user" in d.refusals("user")[-1].error


@pytest.mark.asyncio
async def test_one_bad_path_applies_nothing(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    mine = d.manager.create_workspace("mine", owner="acme:alice")
    await d.ticket("user")
    await d.config("user", working_dir=mine.path,
                   config_root=str(d.outside))
    _refused(d, "user", "config_root")
    assert "working_dir" not in d.refusals("user")[-1].error


@pytest.mark.asyncio
async def test_set_workspace_outside_the_root_is_refused(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    await d.ticket("user")
    await d.srv._handle_message_daemon(
        "user", CommandRequest(command="set_workspace", args=[str(d.outside)]))
    assert d.router._event_sink.get_client_workspace("user") is None
    # The composite fans every send out to each sink; the IPC-shaped one
    # records what the router answered.
    errors = [c.args[1] for c in d.ipc_like.send_event.call_args_list
              if c.args[0] == "user"]
    assert any(getattr(e, "error_type", "") == "ClientPathOutsideWorkspaceRoot"
               for e in errors)


# --------------------------------------------------------------- acceptance

@pytest.mark.asyncio
async def test_own_and_unowned_workspaces_are_accepted(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    mine = d.manager.create_workspace("mine", owner="acme:alice")
    loose = d.acme_root / "loose"
    loose.mkdir()
    await d.ticket("user")
    await d.config("user", working_dir=mine.path,
                   config_root=os.path.join(mine.path, ".jaato"),
                   env_file=os.path.join(mine.path, ".env"),
                   trace_log_path=os.path.join(mine.path, "t.log"))
    assert d.refusals("user") == []
    assert d.stored("user")["working_dir"] == mine.path
    await d.config("user", working_dir=str(loose))
    assert d.refusals("user") == []
    assert d.stored("user")["working_dir"] == str(loose)


@pytest.mark.asyncio
async def test_set_workspace_inside_the_root_is_accepted(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    mine = d.manager.create_workspace("mine", owner="acme:alice")
    await d.ticket("user")
    await d.srv._handle_message_daemon(
        "user", CommandRequest(command="set_workspace", args=[mine.path]))
    assert d.router._event_sink.get_client_workspace("user") == mine.path


# ------------------------------------------------------------ unchanged

@pytest.mark.asyncio
async def test_a_shared_token_connection_is_unchanged(tmp_path: Path) -> None:
    d = _Daemon(tmp_path)
    d.attach("shared", _SHARED_TOKEN)
    await d.config("shared", working_dir=str(d.outside))
    assert d.refusals("shared") == []
    assert d.stored("shared")["working_dir"] == str(d.outside)


def test_an_ipc_client_is_unchanged(tmp_path: Path) -> None:
    """A client the WS transport does not own gets no path scope."""
    d = _Daemon(tmp_path)
    d.router.handle_request(
        "ipc-client", "", ClientConfigRequest(working_dir=str(d.outside)))
    assert d.refusals("ipc-client") == []
    assert d.stored("ipc-client")["working_dir"] == str(d.outside)


def test_a_manager_without_a_lent_scope_checks_nothing() -> None:
    """``SessionManager`` alone (embedded, tests) behaves as before."""
    sm = SessionManager()
    sm._emit_to_client = lambda *a: None
    sm._apply_client_config("c", ClientConfigRequest(working_dir="/somewhere"))
    assert sm._client_config["c"]["working_dir"] == "/somewhere"
