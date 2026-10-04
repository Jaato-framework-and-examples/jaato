"""An application's workspaces live under its root and belong to its account (#1496).

A WS connection carries no OS principal, so ``--runner-uid-policy peer``
cannot name a user for it and ``workspace-owner`` read ``root`` for every
workspace the daemon created.  Each application in ``--ws-app-credentials``
now declares ``{credential, account, workspace_root}``: its connections list,
create and open workspaces under that root only, and what the root daemon
writes inside a workspace takes the owner of the tree it lands in.

Three groups, each driving the real code:

* **ownership** (``shared/workspace_ownership.py`` and two of its writers).
  The suite does not run as root, so the module's ``os`` is replaced by a
  view of the world as a root daemon sees it: ``geteuid`` answers 0, a path
  the test created reads as daemon-owned (uid 0) unless it is marked as
  another account's, and ``lchown`` is recorded instead of performed.
  Everything else is the real ``os``.
* **routing** (``JaatoWSServer``): a connection of an application gets that
  application's manager and root, any other connection the daemon's.
* **the credentials file** (``ws_tickets.load_app_credentials``).
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Set, Tuple

import pytest

from jaato_server.server import contained_write
from jaato_server.server.websocket import JaatoWSServer
from jaato_server.server.workspace_manager import WorkspaceManager
from jaato_server.server.ws_tickets import (
    AppCredentialsError,
    AppCredentialStore,
    AppWorkspace,
    load_app_credentials,
)
from jaato_server.shared import workspace_ownership
from jaato_server.shared.tests.reversion import Reversion

_OWN = "jaato-server/jaato_server/shared/workspace_ownership.py"
_CW = "jaato-server/jaato_server/server/contained_write.py"
_WM = "jaato-server/jaato_server/server/workspace_manager.py"
_WS = "jaato-server/jaato_server/server/websocket.py"
_TK = "jaato-server/jaato_server/server/ws_tickets.py"

_CREDENTIAL = "c" * 48

REVERSIONS = [
    Reversion(
        target=_OWN,
        find=(
            "    if os.lstat(path).st_uid == os.geteuid():\n"
            "        os.lchown(path, owner[0], owner[1])"
        ),
        replace="    os.lchown(path, owner[0], owner[1])",
        test="test_a_file_another_account_owns_is_not_taken",
        because="the daemon takes a file another account already owns",
    ),
    Reversion(
        target=_OWN,
        find=(
            "    if st.st_uid == os.geteuid():\n"
            "        return None\n"
        ),
        replace="",
        test="test_a_root_owned_tree_hands_nothing",
        because=(
            "a deployment whose workspaces are root-owned has every file "
            "re-chowned to root, a no-op that is no longer guaranteed one"
        ),
    ),
    Reversion(
        target=_OWN,
        find='    if not hasattr(os, "geteuid") or os.geteuid() != 0:',
        replace='    if not hasattr(os, "geteuid"):',
        test="test_a_daemon_that_is_not_root_hands_nothing",
        because="a non-root daemon attempts a chown it cannot perform",
    ),
    Reversion(
        target=_OWN,
        find="os.walk(top, followlinks=False)",
        replace="os.walk(top, followlinks=True)",
        test="test_a_link_inside_the_tree_is_not_followed",
        because="a link the model planted hands files outside the workspace",
    ),
    Reversion(
        target=_CW,
        find="    inherit_owner(dest, root)\n",
        replace="",
        test="test_a_staged_file_and_its_new_directories_take_the_owner",
        because="a staged upload stays root-owned; the runner cannot rewrite it",
    ),
    Reversion(
        target=_CW,
        find="        inherit_owner(nxt, root)\n",
        replace="",
        test="test_a_staged_file_and_its_new_directories_take_the_owner",
        because="the directories a staging creates stay root-owned",
    ),
    Reversion(
        target=_WM,
        find="        inherit_owner_tree(str(path), str(self.workspace_root))\n",
        replace="",
        test="test_a_created_workspace_belongs_to_the_roots_owner",
        because="workspace.create leaves the whole new workspace root-owned",
    ),
    Reversion(
        target=_WS,
        find="        return app_id if app_id in self._app_managers else None",
        replace="        return None",
        test="test_an_applications_connection_is_served_from_its_root",
        because="every connection lands in the daemon's root again",
    ),
    Reversion(
        target=_WS,
        find=(
            "            if root == own_root or root.startswith(own_root + os.sep)"
            " or own_root.startswith(root + os.sep):"
        ),
        replace="            if False:",
        test="test_an_application_root_inside_the_daemons_is_refused",
        because=(
            "the daemon's manager discovers an application's workspaces as "
            "its own and lists them to every other connection"
        ),
    ),
    Reversion(
        target=_TK,
        find="    if st.st_uid != pw.pw_uid:",
        replace="    if False:",
        test="test_a_root_the_account_does_not_own_is_refused",
        because=(
            "an application is given a root its account does not own, so "
            "workspace-owner runs its sessions as someone else"
        ),
    ),
    Reversion(
        target=_TK,
        find="            if a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep):",
        replace="            if False:",
        test="test_two_applications_sharing_a_root_are_refused",
        because="one workspace would belong to two applications",
    ),
]


# ----------------------------------------------------------------------
# A root daemon's view of the filesystem
# ----------------------------------------------------------------------


class _RootDaemonOs:
    """The real ``os``, seen from a daemon running as root.

    ``foreign`` holds paths another account owns; every other path reads as
    the daemon's (``euid``) when ``lstat``-ed.  ``root_trees`` holds directories
    that read as root-owned when ``stat``-ed; any other tree keeps its real
    owner (the user running the suite).  ``lchown`` is recorded.
    """

    def __init__(self, euid: int = 0) -> None:
        self._euid = euid
        self.foreign: Set[str] = set()
        self.root_trees: Set[str] = set()
        self.handed: List[Tuple[str, int, int]] = []

    def __getattr__(self, name: str):
        return getattr(os, name)

    def geteuid(self) -> int:
        return self._euid

    def stat(self, path, *args, **kwargs):
        st = os.stat(path, *args, **kwargs)
        uid = 0 if os.path.realpath(path) in self.root_trees else st.st_uid
        return SimpleNamespace(st_uid=uid, st_gid=st.st_gid, st_mode=st.st_mode)

    def lstat(self, path):
        st = os.lstat(path)
        uid = st.st_uid if str(path) in self.foreign else self._euid
        return SimpleNamespace(st_uid=uid, st_gid=st.st_gid, st_mode=st.st_mode)

    def lchown(self, path, uid: int, gid: int) -> None:
        self.handed.append((str(path), uid, gid))

    def handed_paths(self) -> Set[str]:
        return {p for p, _, _ in self.handed}


@pytest.fixture()
def daemon(monkeypatch: pytest.MonkeyPatch) -> _RootDaemonOs:
    view = _RootDaemonOs()
    monkeypatch.setattr(workspace_ownership, "os", view)
    return view


def _me() -> Tuple[int, int]:
    return os.getuid(), os.getgid()


# ----------------------------------------------------------------------
# Ownership
# ----------------------------------------------------------------------


def test_a_staged_file_and_its_new_directories_take_the_owner(
    tmp_path: Path, daemon: _RootDaemonOs,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()

    dest = contained_write.write_contained(str(ws), "in/box/note.txt", b"hello")

    real = os.path.realpath(ws)
    expected = {
        os.path.join(real, "in"),
        os.path.join(real, "in", "box"),
        dest,
    }
    assert expected <= daemon.handed_paths()
    assert {(u, g) for p, u, g in daemon.handed if p in expected} == {_me()}


def test_a_file_another_account_owns_is_not_taken(
    tmp_path: Path, daemon: _RootDaemonOs,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    theirs = ws / "theirs.txt"
    theirs.write_text("x")
    daemon.foreign.add(str(theirs))

    workspace_ownership.inherit_owner(str(theirs), str(ws))

    assert daemon.handed == []


def test_a_root_owned_tree_hands_nothing(
    tmp_path: Path, daemon: _RootDaemonOs,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    daemon.root_trees.add(os.path.realpath(ws))

    contained_write.write_contained(str(ws), "a/b.txt", b"x")

    assert daemon.handed == []


def test_a_daemon_that_is_not_root_hands_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Neither root nor the tree's owner, so only the "not root" rule can
    # be what stops the hand-over.
    view = _RootDaemonOs(euid=os.getuid() + 4242)
    monkeypatch.setattr(workspace_ownership, "os", view)
    ws = tmp_path / "ws"
    ws.mkdir()

    contained_write.write_contained(str(ws), "a/b.txt", b"x")

    assert view.handed == []


def test_a_link_inside_the_tree_is_not_followed(
    tmp_path: Path, daemon: _RootDaemonOs,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("x")
    top = tmp_path / "ws" / "clone"
    top.mkdir(parents=True)
    (top / "file").write_text("x")
    (top / "escape").symlink_to(outside, target_is_directory=True)

    workspace_ownership.inherit_owner_tree(str(top), str(tmp_path / "ws"))

    handed = daemon.handed_paths()
    assert str(top / "file") in handed
    assert str(top / "escape") in handed, "the link itself is the daemon's to hand"
    assert not any(p.startswith(str(outside)) for p in handed)
    assert str(top / "escape" / "secret") not in handed


def test_a_created_workspace_belongs_to_the_roots_owner(
    tmp_path: Path, daemon: _RootDaemonOs,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    manager = WorkspaceManager(str(root), registry_path=tmp_path / "registry.json")

    info = manager.create_workspace("proj")

    handed = daemon.handed_paths()
    assert info.path in handed
    assert os.path.join(info.path, ".jaato") in handed
    assert os.path.join(info.path, ".env") in handed
    assert {(u, g) for _, u, g in daemon.handed} == {_me()}


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def _server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_root: Path) -> JaatoWSServer:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    own = tmp_path / "daemon-root"
    own.mkdir(exist_ok=True)
    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._workspace_root = str(own)
    ws._workspace_manager = WorkspaceManager(
        str(own), registry_path=tmp_path / "daemon-registry.json")
    ws._provisioner = None
    ws._default_template = None
    ws._clients = {
        "c-app": SimpleNamespace(app_id="acme"),
        "c-other": SimpleNamespace(app_id=None),
    }
    ws._app_managers = {}
    ws._app_provisioners = {}
    ws._app_credentials = AppCredentialStore(
        {"acme": _CREDENTIAL},
        {"acme": AppWorkspace(account=getpass.getuser(), uid=os.getuid(),
                              gid=os.getgid(),
                              workspace_root=os.path.realpath(app_root))},
    )
    return ws


def test_an_applications_connection_is_served_from_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_root = tmp_path / "acme-root"
    app_root.mkdir()
    ws = _server(tmp_path, monkeypatch, app_root)
    ws._init_app_workspaces()

    app_manager = ws._workspace_manager_for("c-app")
    assert app_manager is not ws._workspace_manager
    created = app_manager.create_workspace("proj")
    assert created.path == os.path.join(os.path.realpath(app_root), "proj")

    assert ws._workspace_manager_for("c-other") is ws._workspace_manager
    assert ws.managed_root_for(created.path) == os.path.realpath(app_root)
    assert ws.managed_root_for(
        str(tmp_path / "daemon-root" / "w")) == os.path.realpath(tmp_path / "daemon-root")
    assert ws.managed_root_for(str(tmp_path / "elsewhere")) is None


def test_an_application_root_inside_the_daemons_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_root = tmp_path / "daemon-root" / "acme"
    app_root.mkdir(parents=True)
    ws = _server(tmp_path, monkeypatch, app_root)

    with pytest.raises(RuntimeError, match="overlaps"):
        ws._init_app_workspaces()


# ----------------------------------------------------------------------
# The credentials file
# ----------------------------------------------------------------------


def _write(tmp_path: Path, entries: dict) -> Path:
    path = tmp_path / "ws-apps.json"
    path.write_text(json.dumps(entries))
    os.chmod(path, 0o600)
    return path


def _entry(root: Path, account: Optional[str] = None, credential: str = _CREDENTIAL) -> dict:
    return {"credential": credential, "account": account or getpass.getuser(),
            "workspace_root": str(root)}


def test_a_well_formed_entry_carries_its_workspace(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    root.mkdir(mode=0o700)

    store = load_app_credentials(_write(tmp_path, {"acme": _entry(root)}))

    assert store.workspace("acme") == AppWorkspace(
        account=getpass.getuser(), uid=os.getuid(), gid=os.getgid(),
        workspace_root=os.path.realpath(root))


@pytest.mark.skipif(os.getuid() == 0, reason="root owns the directory used as foreign")
def test_a_root_the_account_does_not_own_is_refused(tmp_path: Path) -> None:
    # "/" is root's; the account is the user running the suite.
    with pytest.raises(AppCredentialsError, match="owned by uid"):
        load_app_credentials(_write(tmp_path, {"acme": _entry(Path("/"))}))


def test_two_applications_sharing_a_root_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    (root / "inner").mkdir(parents=True)
    entries = {
        "acme": _entry(root),
        "other": _entry(root / "inner", credential="d" * 48),
    }
    with pytest.raises(AppCredentialsError, match="overlapping"):
        load_app_credentials(_write(tmp_path, entries))


@pytest.mark.parametrize("broken", [
    pytest.param({"workspace_root": "relative/root"}, id="relative-root"),
    pytest.param({"account": "no-such-account-1496"}, id="unknown-account"),
    pytest.param({"extra": "x"}, id="unknown-key"),
])
def test_a_malformed_entry_is_refused(tmp_path: Path, broken: dict) -> None:
    root = tmp_path / "acme"
    root.mkdir()
    entry = {**_entry(root), **broken}

    with pytest.raises(AppCredentialsError):
        load_app_credentials(_write(tmp_path, {"acme": entry}))


def test_an_entry_without_a_root_is_refused(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    del entry["workspace_root"]

    with pytest.raises(AppCredentialsError, match="lacks"):
        load_app_credentials(_write(tmp_path, {"acme": entry}))
