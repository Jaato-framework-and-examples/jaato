"""What a root daemon writes INSIDE a workspace is the workspace owner's (#1528).

Under ``--runner-uid-policy workspace-owner`` (or ``peer``) the runner runs
as the workspace's owner, and the root daemon still wrote two things into
that workspace as ``root:root``:

* ``.jaato/sessions/<id>.json`` -- a temp file plus ``os.replace`` on every
  save, so a record the owner held was re-rooted by the next save, and a
  ``chown -R`` was undone by it;
* ``.jaato/logs/session_<id>_client_<c>.log`` -- ``logging.FileHandler``.

The rule now: a file the daemon writes inside a workspace is CREATED owned
by the workspace owner whenever a policy that drops is in effect
(:func:`server.runner_user.workspace_file_owner`).  The temp file is
``fchown``-ed before anything is written, so there is no root-owned window
and no save re-roots it.  Under the default ``daemon`` policy nothing
changes.  Files outside any workspace (the workspace index) stay the
daemon's.

HOW OWNERSHIP IS OBSERVED.  Where the suite runs as root the workspace is
``chown``-ed to ``nobody`` and every assertion reads a real ``stat``.
Elsewhere (CI runs unprivileged) the workspace owner is simulated:
``tree_owner`` names a foreign uid, and ``os.fchown`` / ``os.lchown`` are
recorded per INODE instead of performed.  Keying on the inode is what lets
the simulation see an ``os.replace``: the record that replaced the old one
is a different inode, and it is owned only if it was handed over itself.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Callable, Dict, Tuple
from unittest.mock import MagicMock

import pytest

from jaato_server.server import runner_user
from jaato_server.shared import workspace_ownership
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/session/file_session.py",
        find="                fchown_to(fd, owner)\n",
        replace="                pass\n",
        test="test_a_saved_record_is_the_owners_and_stays_so",
        because="the session record's temp file written as the daemon, so "
                "os.replace installs a root-owned record on every save",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/session_manager.py",
        find="                record_owner = workspace_file_owner(session.workspace_path)\n",
        replace="                record_owner = None\n",
        test="test_a_saved_record_is_the_owners_and_stays_so",
        because="the save path never asking who owns the workspace, so the "
                "record is the daemon's whatever the policy",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/atomic_write.py",
        find="        fchown_to(fd, owner)\n",
        replace="        pass\n",
        test="test_subagent_state_beside_the_record_is_the_owners",
        because="the per-subagent state files under .jaato/sessions/<id>/ "
                "written root-owned beside an owner-owned record",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/session_logging.py",
        find="                _create_owned(log_dir, log_file, workspace)\n",
        replace="                log_dir.mkdir(parents=True, exist_ok=True)\n",
        test="test_a_client_log_is_the_owners_from_its_first_byte",
        because="the per-client session log created by FileHandler as the "
                "daemon, the second writer #1528 names",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/session_logging.py",
        find="        hand_to(str(log_file), owner)\n",
        replace="        pass\n",
        test="test_a_log_a_previous_daemon_left_is_handed_over",
        because="a client log already on disk as the daemon's kept so when "
                "the handler reopens it for append",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_user.py",
        find="    if not workspace_path or current_policy() == POLICY_DAEMON:\n",
        replace="    if not workspace_path:\n",
        test="test_the_daemon_policy_leaves_ownership_alone",
        because="the default policy handing the daemon's files over, where "
                "#1528 promises byte-identical behaviour",
    ),
]

#: The simulated (or, as root, real) workspace owner.
OWNER: Tuple[int, int] = (65534, 65534)
SESSION_ID = "20261004_120000"


@pytest.fixture
def owned_ws(tmp_path, monkeypatch) -> Tuple[str, Callable[[str], int]]:
    """``(workspace, uid_of)``: a workspace owned by :data:`OWNER`.

    ``uid_of(path)`` answers who owns *path* -- a real ``lstat`` as root,
    the inode ledger otherwise (see the module docstring).
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    if os.geteuid() == 0:
        os.chown(ws, *OWNER)
        return str(ws), lambda p: os.lstat(p).st_uid

    ledger: Dict[int, int] = {}

    def fake_fchown(fd, uid, gid):
        ledger[os.fstat(fd).st_ino] = uid

    def fake_lchown(path, uid, gid):
        ledger[os.lstat(path).st_ino] = uid

    monkeypatch.setattr(os, "fchown", fake_fchown)
    monkeypatch.setattr(os, "lchown", fake_lchown)

    def fake_tree_owner(directory):
        return OWNER if os.path.realpath(directory) == os.path.realpath(ws) else None

    monkeypatch.setattr(workspace_ownership, "tree_owner", fake_tree_owner)

    def uid_of(p):
        st = os.lstat(p)
        return ledger.get(st.st_ino, st.st_uid)

    return str(ws), uid_of


@pytest.fixture
def policy(monkeypatch):
    """Set the process-wide runner uid policy for one test."""
    def _set(value: str) -> None:
        monkeypatch.setattr(runner_user, "_policy", value)
    return _set


def _manager(index_path: pathlib.Path):
    """A ``SessionManager`` with just what ``_save_session`` reaches."""
    from jaato_server.server.session_manager import SessionManager
    from jaato_server.server.session_workspace_index import SessionWorkspaceIndex
    from jaato_server.shared.plugins.session.base import SessionConfig
    from jaato_server.shared.plugins.session.file_session import FileSessionPlugin

    sm = SessionManager.__new__(SessionManager)
    sm._session_config = SessionConfig()
    sm._session_workspace_index = SessionWorkspaceIndex(index_path)
    sm._workspace_monitors = {}
    plugin = FileSessionPlugin()
    plugin.initialize({})
    sm._session_plugin = plugin
    return sm


def _session(ws: str):
    from jaato_server.server.session_manager import Session
    return Session(
        session_id=SESSION_ID, name="s", server=None,
        created_at="2026-10-04T12:00:00", workspace_path=ws,
    )


def _record(ws: str) -> str:
    return os.path.join(ws, ".jaato", "sessions", f"{SESSION_ID}.json")


def test_a_saved_record_is_the_owners_and_stays_so(owned_ws, policy, tmp_path):
    """(a) The record, and the directories its save creates, are the owner's
    -- and a second save (a new inode by ``os.replace``) is too."""
    ws, uid_of = owned_ws
    policy("workspace-owner")
    sm = _manager(tmp_path / "index.json")
    session = _session(ws)

    assert sm._save_session(session) is True
    record = _record(ws)
    assert uid_of(record) == OWNER[0]
    assert uid_of(os.path.dirname(record)) == OWNER[0]
    assert uid_of(os.path.join(ws, ".jaato")) == OWNER[0]

    first = os.lstat(record).st_ino
    assert sm._save_session(session) is True
    assert os.lstat(record).st_ino != first, "os.replace installs a new inode"
    assert uid_of(record) == OWNER[0], "the second save re-rooted the record"
    assert not os.path.exists(record + ".tmp")


def test_subagent_state_beside_the_record_is_the_owners(owned_ws, policy, tmp_path):
    """The state files ``_save_session`` writes beside the record follow it."""
    ws, uid_of = owned_ws
    policy("workspace-owner")
    sm = _manager(tmp_path / "index.json")
    plugin = MagicMock()
    plugin.get_agent_full_state.return_value = {"agent_id": "a1"}
    storage = pathlib.Path(ws, ".jaato", "sessions")
    sm._save_subagent_states(
        SESSION_ID, plugin, [{"agent_id": "a1"}], storage_dir=storage,
        owner=runner_user.workspace_file_owner(ws))
    state = storage / SESSION_ID / "subagents" / "a1.json"
    assert state.exists()
    assert uid_of(str(state)) == OWNER[0]
    assert uid_of(str(state.parent)) == OWNER[0]


def _client_log(ws: str, client: str = "client_4") -> Tuple[object, str]:
    from jaato_server.server.session_logging import SessionRoutingHandler
    router = SessionRoutingHandler()
    handler = router._get_or_create_handler(SESSION_ID, client, ws, None)
    path = os.path.join(
        ws, ".jaato", "logs", f"session_{SESSION_ID}_client_{client}.log")
    return handler, path


def _emit(handler) -> None:
    handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None))
    handler.flush()
    handler.close()


def test_a_client_log_is_the_owners_from_its_first_byte(owned_ws, policy):
    """(b) The log exists, owner-owned and EMPTY, before the handler writes."""
    ws, uid_of = owned_ws
    policy("workspace-owner")
    handler, path = _client_log(ws)
    assert handler is not None
    assert os.path.getsize(path) == 0
    assert uid_of(path) == OWNER[0]
    assert uid_of(os.path.dirname(path)) == OWNER[0]
    _emit(handler)
    assert os.path.getsize(path) > 0
    assert uid_of(path) == OWNER[0]


def test_a_log_a_previous_daemon_left_is_handed_over(owned_ws, policy):
    """A client log already on disk as the daemon's is handed over on reopen."""
    ws, uid_of = owned_ws
    logs = os.path.join(ws, ".jaato", "logs")
    os.makedirs(logs)
    path = os.path.join(logs, f"session_{SESSION_ID}_client_client_5.log")
    with open(path, "w") as fh:
        fh.write("old\n")
    assert uid_of(path) != OWNER[0]
    policy("workspace-owner")
    handler, _ = _client_log(ws, "client_5")
    _emit(handler)
    assert uid_of(path) == OWNER[0]


def test_the_daemon_policy_leaves_ownership_alone(owned_ws, policy, tmp_path):
    """(c) Under the default policy the daemon writes as itself, as before."""
    ws, uid_of = owned_ws
    policy("daemon")
    daemon_uid = os.geteuid()
    sm = _manager(tmp_path / "index.json")
    assert sm._save_session(_session(ws)) is True
    assert uid_of(_record(ws)) == daemon_uid
    handler, path = _client_log(ws)
    _emit(handler)
    assert uid_of(path) == daemon_uid


def test_the_index_outside_the_workspace_stays_the_daemons(owned_ws, policy, tmp_path):
    """(d) The workspace index is not inside any workspace: the daemon's."""
    ws, uid_of = owned_ws
    policy("workspace-owner")
    index = tmp_path / "home" / "session_workspace_index.json"
    sm = _manager(index)
    assert sm._save_session(_session(ws)) is True
    assert index.exists()
    assert uid_of(str(index)) == os.geteuid()
    assert uid_of(_record(ws)) == OWNER[0]


def test_a_root_owned_workspace_is_not_handed_anything(tmp_path, policy):
    """No owner to hand to: a root-owned (or the daemon's own) workspace."""
    policy("workspace-owner")
    assert runner_user.workspace_file_owner(str(tmp_path)) is None
    assert runner_user.workspace_file_owner(None) is None
