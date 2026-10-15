"""What the daemon writes into a caller's workspace is the workspace owner's (#1703).

Two writers left ``root:root`` files in a workspace an application account
owns, on a root daemon, so the owner could not remove its own workspace:

* ``scaffold.integration`` (``jaato-scaffold integration <name> --refresh``
  run by the daemon) copied the ``jaato-sdk`` skill to
  ``<ws>/.claude/skills/jaato-sdk`` and created ``.claude/`` and
  ``.claude/skills/`` on the way, all as the daemon.  ``shutil.rmtree`` as
  the owner then failed on ``.claude/skills/jaato-sdk``.
* the daemon's TODO mirror, configured on
  ``<ws>/.jaato/sessions/<id>/plans/``, created that directory (and wrote
  plan files into it) as the daemon, while #1528 had handed the record
  beside it to the owner.

The rules now: the integration's copy and every directory created between
the workspace and it are handed to the owner of the workspace tree (the
#1496 rule every other daemon writer into a workspace follows), on every
refresh, so a copy an older daemon left root-owned is handed over too.
The TODO storage is created owned by
:func:`server.runner_user.workspace_file_owner`, the #1528 rule the record
already follows, so it is the daemon's under the default ``daemon`` policy.

Ownership is observed the way
``test_daemon_workspace_files_owned_1528.py`` observes it: a real
``chown`` to ``nobody`` as root, otherwise ``tree_owner`` names a foreign
uid and ``os.lchown`` / ``os.fchown`` are recorded per inode.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Tuple
from unittest.mock import MagicMock

import pytest

from jaato_server.server import runner_user
from jaato_server.shared import workspace_ownership
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/command_router.py",
        find="            inherit_owner_down(str(dest), workspace)\n",
        replace="            pass\n",
        test="test_an_integration_copy_and_its_parents_are_the_owners",
        because="scaffold.integration leaving the skill and the .claude/ "
                "directories it created root-owned, the reported state",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/workspace_ownership.py",
        find="        _hand(cursor, owner)\n        if os.path.islink(cursor):\n",
        replace="        if os.path.islink(cursor):\n",
        test="test_an_integration_copy_and_its_parents_are_the_owners",
        because="only the skill directory handed over, leaving .claude/ and "
                ".claude/skills/ root-owned, so rmtree still fails",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/session_manager.py",
        find='            "storage_owner": owner,\n',
        replace="",
        test="test_the_todo_storage_is_created_as_the_owner",
        because="the daemon's TODO mirror creating sessions/<id>/plans/ as "
                "root beside an owner-owned record",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/todo/storage.py",
        find="                    os.fchown(f.fileno(), self._owner[0], self._owner[1])\n",
        replace="                    pass\n",
        test="test_the_todo_storage_is_created_as_the_owner",
        because="plan files written into the owned plans/ directory as root",
    ),
]

OWNER: Tuple[int, int] = (65534, 65534)
SESSION_ID = "20261015_150022"


@pytest.fixture
def owned_ws(tmp_path, monkeypatch) -> Tuple[str, Callable[[str], int]]:
    """``(workspace, uid_of)``: a workspace owned by :data:`OWNER`."""
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
        if os.path.realpath(directory) == os.path.realpath(ws):
            return OWNER
        return None

    monkeypatch.setattr(workspace_ownership, "tree_owner", fake_tree_owner)

    def uid_of(p):
        st = os.lstat(p)
        return ledger.get(st.st_ino, st.st_uid)

    return str(ws), uid_of


def _router(ws: str):
    from jaato_server.server.command_router import CommandRouter
    session_manager = MagicMock()
    session = MagicMock()
    session.workspace_path = ws
    session_manager.get_client_session.return_value = session
    session_manager.get_session.return_value = None
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = session_manager
    router._event_sink = MagicMock()
    return router


def _answer(router):
    sent = [c[0][1] for c in router._event_sink.send_event.call_args_list]
    assert len(sent) == 1, sent
    return sent[0]


def _all_paths(top: str):
    yield top
    for dirpath, dirnames, filenames in os.walk(top):
        for name in dirnames + filenames:
            yield os.path.join(dirpath, name)


def test_an_integration_copy_and_its_parents_are_the_owners(owned_ws):
    """The real refresh writes the skill; every path it created is the owner's."""
    ws, uid_of = owned_ws
    router = _router(ws)
    router._handle_scaffold_integration("c1", ["claude-code"], None)
    ans = _answer(router)
    assert ans.ok is True, ans.error
    assert ans.changed is True

    claude = os.path.join(ws, ".claude")
    skill = os.path.join(claude, "skills", "jaato-sdk")
    assert os.path.isfile(os.path.join(skill, "SKILL.md"))
    not_handed = [p for p in _all_paths(claude) if uid_of(p) != OWNER[0]]
    assert not_handed == [], f"left as the daemon's: {not_handed}"


def test_a_copy_an_older_daemon_left_is_handed_over_on_refresh(owned_ws, monkeypatch):
    """A refresh that changes nothing still hands the existing copy over."""
    from jaato_server.shared.scaffold import integrations

    ws, uid_of = owned_ws
    skill = os.path.join(ws, ".claude", "skills", "jaato-sdk")
    os.makedirs(skill)
    with open(os.path.join(skill, "SKILL.md"), "w") as fh:
        fh.write("x\n")
    assert uid_of(skill) != OWNER[0]
    monkeypatch.setattr(integrations, "refresh", lambda name, dest, **kw: {
        "state_before": "current", "state_after": "current", "changed": False,
        "skipped_reason": "", "lines": []})

    router = _router(ws)
    router._handle_scaffold_integration("c1", ["claude-code"], None)
    assert _answer(router).ok is True
    for p in _all_paths(os.path.join(ws, ".claude")):
        assert uid_of(p) == OWNER[0], p


def test_a_path_outside_the_tree_is_not_handed(owned_ws, tmp_path):
    """``inherit_owner_down`` never hands a path that is not under the tree."""
    ws, uid_of = owned_ws
    outside = tmp_path / "elsewhere" / "x"
    outside.mkdir(parents=True)
    workspace_ownership.inherit_owner_down(str(outside), ws)
    assert uid_of(str(outside)) != OWNER[0]
    assert uid_of(str(outside.parent)) != OWNER[0]


def _todo_storage(sm, server, ws):
    session_dir = sm._session_storage_dir(ws) / SESSION_ID
    sm._configure_todo_storage(server, session_dir, ws)
    return session_dir / "plans"


def _manager_and_server():
    from jaato_server.server.session_manager import SessionManager
    from jaato_server.shared.plugins.session.base import SessionConfig
    from jaato_server.shared.plugins.todo.plugin import TodoPlugin

    sm = SessionManager.__new__(SessionManager)
    sm._session_config = SessionConfig()
    todo = TodoPlugin()
    server = MagicMock()
    server.registry.get_plugin.side_effect = (
        lambda name: todo if name == "todo" else None)
    return sm, server, todo


def test_the_todo_storage_is_created_as_the_owner(owned_ws, monkeypatch):
    """Under a policy that drops, ``plans/`` and its plan files are the owner's."""
    from jaato_sdk.plugins.todo.models import TodoPlan

    ws, uid_of = owned_ws
    monkeypatch.setattr(runner_user, "_policy", "workspace-owner")
    sm, server, todo = _manager_and_server()
    plans = _todo_storage(sm, server, ws)
    assert plans.is_dir()
    for p in (plans, plans.parent, plans.parent.parent):
        assert uid_of(str(p)) == OWNER[0], p

    plan = TodoPlan.create("t", ["a"])
    todo._storage.save_plan(plan)
    files = list(plans.iterdir())
    assert files
    assert all(uid_of(str(f)) == OWNER[0] for f in files), files


def test_the_daemon_policy_writes_the_todo_storage_as_the_daemon(owned_ws, monkeypatch):
    """The default policy leaves the TODO storage the daemon's, as before."""
    ws, uid_of = owned_ws
    monkeypatch.setattr(runner_user, "_policy", "daemon")
    sm, server, _ = _manager_and_server()
    plans = _todo_storage(sm, server, ws)
    assert uid_of(str(plans)) == os.geteuid()


def test_a_malformed_storage_owner_is_ignored():
    """The internal key accepts only a pair of non-negative ints."""
    from jaato_server.shared.plugins.todo.plugin import _storage_owner

    assert _storage_owner((1, 2)) == (1, 2)
    assert _storage_owner([1, 2]) == (1, 2)
    for bad in (None, (1,), (True, 2), (-1, 2), ("1", "2"), (1, 2, 3)):
        assert _storage_owner(bad) is None, bad
