"""A client with no identity is shown its own config root's sessions (#1584).

Reported from production: a Telegram bot connected over WS with the shared
bearer token -- no bound identity -- and asked for its sessions.  The daemon
answered with 2,092 rows (1,203,006 bytes) where 153 were its own.  The same
unscoped listing rode the ``SessionInfoEvent`` snapshot answering
``session.new``; it exceeded the SDK's 1 MiB receive ceiling (#1279), the
client closed with 1009, the confirmation never arrived and every new
conversation timed out.  And each row named another application's session
id, workspace path, model and description.

``CommandRouter._sessions_visible_to`` answered "every session on the
daemon" whenever the transport reported no identity boundary
(``visible_workspace_paths`` -> ``None``): IPC, and every WS connection on a
shared token.  #1113 kept that default deliberately.  It was the defect.

Pinned here, against a REAL ``SessionManager`` listing (cold records on
disk, a loaded session) and the real router:

- an identity-less client (IPC or WS) sees the sessions that ran under ITS
  config root -- the one it declared, else ``<workspace>/.jaato`` -- loaded
  and cold alike, compared resolved;
- a cold record that recorded an explicit config root is scoped by it, not
  by its workspace;
- a row whose config root is unknown matches nobody but its creator;
- a client with neither a config root nor a workspace sees only what it
  created, never the daemon;
- ``session.attach`` / ``session.delete`` refuse outside that set;
- the snapshot answering ``session.new`` is the same set;
- a revived session keeps recording the root it was handed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_sdk.events import ErrorEvent, SessionListEvent
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.session_manager import Session, SessionManager
from jaato_server.shared.tests.reversion import Reversion

_SESSION_MANAGER = "jaato-server/jaato_server/server/session_manager.py"
_COMMAND_ROUTER = "jaato-server/jaato_server/server/command_router.py"
_SERIALIZER = "jaato-server/jaato_server/shared/plugins/session/serializer.py"


class _Sink:
    """An event sink with no identity unless one is given."""

    def __init__(self, workspace=None, user=None, boundary=None):
        self.workspace = workspace
        self.user = user
        self.boundary = boundary
        self.sent = []

    def send_event(self, client_id, event):
        self.sent.append(event)

    def get_client_user(self, client_id):
        return self.user

    def get_client_workspace(self, client_id):
        return self.workspace

    def set_client_workspace(self, client_id, path):
        self.workspace = path

    def set_client_session(self, client_id, session_id):
        pass

    def visible_workspace_paths(self, client_id):
        return self.boundary


def _write_record(sm, workspace: Path, session_id: str, *,
                  config_root=None, created_by=None, record_workspace=True):
    sessions_dir = sm._session_storage_dir(str(workspace))
    sessions_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "version": "2.0", "session_id": session_id, "description": "",
        "created_at": "2026-10-01T12:00:00", "updated_at": "2026-10-01T12:00:00",
        "turn_count": 0, "messages": [],
    }
    if record_workspace:
        data["workspace_path"] = str(workspace)
    if config_root is not None:
        data["config_root"] = config_root
    if created_by is not None:
        data["created_by"] = created_by
    (sessions_dir / f"{session_id}.json").write_text(json.dumps(data))
    sm._session_workspace_index.record(session_id, str(workspace))


def _loaded(sm, session_id, workspace, config_root=None):
    server = SimpleNamespace(model_provider="", model_name="", is_processing=False,
                             get_history=lambda: [], profile_name=None)
    sm._sessions[session_id] = Session(
        session_id=session_id, name=session_id, server=server,  # type: ignore[arg-type]
        created_at="2026-10-01T12:00:00", workspace_path=workspace,
        config_root=config_root)


@pytest.fixture()
def daemon(tmp_path, monkeypatch):
    """A real manager holding two applications' sessions, plus a router."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    sm = SessionManager()
    bot = tmp_path / "bot"
    other = tmp_path / "other"
    repo = tmp_path / "repo"
    run = repo / "tests" / "runs" / "r1"
    for d in (bot, other, run):
        d.mkdir(parents=True)
    # The bot's own sessions: one cold legacy record (no config_root, so
    # derived from its workspace), one cold with the root recorded, one loaded.
    _write_record(sm, bot, "bot_cold_legacy")
    _write_record(sm, bot, "bot_cold", config_root=str(bot / ".jaato"))
    _loaded(sm, "bot_loaded", str(bot), str(bot / ".jaato"))
    # Another application's sessions.
    _write_record(sm, other, "other_cold", created_by="uid:4242")
    _loaded(sm, "other_loaded", str(other), str(other / ".jaato"))
    # A cascade run whose root is the REPO's, not its workspace's.
    _write_record(sm, run, "run_cold", config_root=str(repo / ".jaato"))
    # A record whose config root is unknown: no root and no workspace.
    _write_record(sm, other, "unknown_root", record_workspace=False)
    sink = _Sink()
    router = CommandRouter(session_manager=sm, event_sink=sink, daemon_plugins={})
    return SimpleNamespace(sm=sm, sink=sink, router=router, bot=bot,
                           other=other, repo=repo, run=run)


def _listed(d, client_id="c1"):
    d.sink.sent.clear()
    d.router._handle_session_list(client_id, None)
    [event] = [e for e in d.sink.sent if isinstance(e, SessionListEvent)]
    return {row["id"] for row in event.sessions}


# --------------------------------------------------------------------------

def test_an_identity_less_ipc_client_sees_only_its_config_root(daemon):
    daemon.sm._client_config["c1"] = {"working_dir": str(daemon.bot)}
    assert _listed(daemon) == {"bot_cold_legacy", "bot_cold", "bot_loaded"}


def test_a_ws_client_on_the_shared_token_sees_only_its_config_root(daemon):
    # No bound identity: the WS sink answers ``None`` for the boundary, and
    # the client's workspace is its selection.
    daemon.sink.workspace = str(daemon.bot)
    assert _listed(daemon) == {"bot_cold_legacy", "bot_cold", "bot_loaded"}


def test_cold_rows_carry_the_config_root_they_ran_under(daemon):
    daemon.sm._client_config["c1"] = {"working_dir": str(daemon.bot)}
    rows = {r.session_id: r for r in daemon.sm.list_sessions()}
    assert rows["bot_cold"].config_root == str(daemon.bot / ".jaato")
    assert rows["bot_cold_legacy"].config_root == str(daemon.bot / ".jaato")
    assert rows["unknown_root"].config_root is None
    assert "bot_cold" in _listed(daemon)


def test_a_declared_config_root_is_the_boundary_not_the_workspace(daemon):
    # A cascade driver: workspace is the run directory, root is the repo's.
    daemon.sm._client_config["c1"] = {
        "working_dir": str(daemon.run), "config_root": str(daemon.repo / ".jaato")}
    assert _listed(daemon) == {"run_cold"}


def test_a_client_with_no_root_and_no_workspace_sees_none(daemon):
    assert _listed(daemon) == set()


def test_the_creator_still_sees_its_own_session_in_another_root(daemon):
    daemon.sink.user = "uid:4242"
    assert _listed(daemon) == {"other_cold"}


def test_an_unknown_config_root_matches_nobody(daemon):
    daemon.sm._client_config["c1"] = {"working_dir": str(daemon.other)}
    assert "unknown_root" not in _listed(daemon)


def test_attach_refuses_outside_the_set_and_admits_inside(daemon):
    daemon.sm._client_config["c1"] = {"working_dir": str(daemon.bot)}
    daemon.router._handle_session_attach("c1", None, ["other_loaded"], str(daemon.bot))
    errors = [e for e in daemon.sink.sent if isinstance(e, ErrorEvent)]
    assert errors and "other_loaded" in errors[0].error
    assert "c1" not in daemon.sm._sessions["other_loaded"].attached_clients


def test_delete_refuses_outside_the_set(daemon):
    daemon.sm._client_config["c1"] = {"working_dir": str(daemon.bot)}
    daemon.router._handle_session_delete("c1", ["other_cold"])
    record = daemon.sm._session_storage_dir(str(daemon.other)) / "other_cold.json"
    assert record.exists()
    errors = [e for e in daemon.sink.sent if isinstance(e, ErrorEvent)]
    assert errors and errors[0].error.startswith("session.delete: other_cold")


def test_the_snapshot_answering_session_new_is_the_same_set(daemon):
    daemon.sink.workspace = str(daemon.bot)
    session = SimpleNamespace(session_id="bot_loaded", name="bot_loaded",
                              server=None, user_inputs=[], sandbox_mode=None)
    event = daemon.sm._build_session_info_event(session, client_id="c1")
    assert {row["id"] for row in event.sessions} == {
        "bot_cold_legacy", "bot_cold", "bot_loaded"}


def test_a_revived_session_keeps_recording_its_config_root():
    """The revive builds ``Session(..., workspace_path=state.workspace_path)``
    and must hand it the root the runner was given, or its next save writes
    ``config_root: null`` over the record's value."""
    src = Path(__file__).resolve().parents[1] / "session_manager.py"
    tree = ast.parse(src.read_text())
    found = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Session"):
            continue
        kws = {k.arg: k.value for k in node.keywords}
        ws = kws.get("workspace_path")
        if isinstance(ws, ast.Attribute) and ast.unparse(ws) == "state.workspace_path":
            found = True
            assert "config_root" in kws, "revived Session drops config_root"
    assert found


REVERSIONS = [
    Reversion(
        target=_COMMAND_ROUTER,
        find="            return self._sessions_in_config_root(client_id, sessions, user)",
        replace="            return sessions",
        test="test_an_identity_less_ipc_client_sees_only_its_config_root",
        because=(
            "an identity-less client handed every session on the daemon -- "
            "the 1.2 MB listing that broke the client's receive ceiling"
        ),
    ),
    Reversion(
        target=_SESSION_MANAGER,
        find=(
            "                    config_root=effective_config_root(\n"
            "                        getattr(info, \"config_root\", None), info.workspace_path),\n"
        ),
        replace="",
        test="test_cold_rows_carry_the_config_root_they_ran_under",
        because="cold rows carrying no config root, so the filter has no evidence",
    ),
    Reversion(
        target=_SERIALIZER,
        find="        config_root=data.get('config_root'),  # None on pre-2.4 records\n",
        replace="",
        test="test_a_declared_config_root_is_the_boundary_not_the_workspace",
        because="a cold record scoped by its workspace instead of the root it recorded",
    ),
    Reversion(
        target=_COMMAND_ROUTER,
        find="            if mine is None:\n                return False\n",
        replace="            if mine is None:\n                return True\n",
        test="test_a_client_with_no_root_and_no_workspace_sees_none",
        because="a client that named nothing being shown the whole daemon",
    ),
    Reversion(
        target=_COMMAND_ROUTER,
        find=(
            "        if any(s.session_id == target_session_id\n"
            "               for s in self._sessions_visible_to(client_id)):\n"
        ),
        replace="        if True:\n",
        test="test_delete_refuses_outside_the_set",
        because="session.delete admitting a session the listing does not show",
    ),
    Reversion(
        target=_COMMAND_ROUTER,
        find=(
            "            if user is not None and getattr(s, \"created_by\", None) == user:\n"
            "                return True\n"
        ),
        replace="",
        test="test_the_creator_still_sees_its_own_session_in_another_root",
        because="a client losing sight of the sessions it created",
    ),
    Reversion(
        target=_SESSION_MANAGER,
        find="            config_root=restore_config_root,\n            user_inputs=",
        replace="            user_inputs=",
        test="test_a_revived_session_keeps_recording_its_config_root",
        because="a revived session erasing its recorded config root on the next save",
    ),
]
