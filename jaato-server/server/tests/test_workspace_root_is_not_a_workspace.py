"""The workspace root is not a workspace, and a created one is announced by name.

Seen live, after every ``workspace.create``: a new row named after the
ROOT's own directory (``workspaces``), owner ``-``, timestamped now, that
``select`` and ``delete`` then refused as "does not exist".  Two defects,
one on each side of the wire:

- the WS server answered ``workspace.create`` with a ``workspace=`` dict
  the SDK's ``WorkspaceCreatedEvent`` did not declare, so the client
  received an event with no name and put an unnamed row in its table;
- clicking that row selected ``""``, and ``_resolve_under_root("")`` is
  the root itself, which ``_is_under_root`` accepted.  The root was then
  analysed as a workspace (named by its basename), cached under the key
  ``""`` and written to the registry as a row nothing could act on.

So the event carries the row, and the root is refused wherever a NAME is
resolved -- select, delete, config status, the registry path branch.  Each
refusal is paired with the neighbouring name that still works, because a
refusal that would have happened anyway proves nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from jaato_sdk.events import (
    WorkspaceCreatedEvent,
    WorkspaceListEvent,
    deserialize_event,
)
from server.workspace_manager import WorkspaceContainmentError, WorkspaceManager


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "workspaces"
    (r / "mine" / ".jaato").mkdir(parents=True)
    (r / "mine" / ".env").write_text("")
    (r / ".jaato").mkdir()            # what once made the ROOT look like a workspace
    return r


def _manager(root: Path) -> WorkspaceManager:
    return WorkspaceManager(str(root), registry_path=root.parent / "registry.json")


def _rows(root: Path) -> list:
    reg = root.parent / "registry.json"
    return json.loads(reg.read_text())["workspaces"] if reg.exists() else []


# ------------------------------------------------ the root, by every spelling

@pytest.mark.parametrize("name", ["", ".", "mine/..", "./"])
def test_a_name_resolving_to_the_root_is_refused_and_nothing_is_cached(root, name):
    m = _manager(root)
    with pytest.raises(WorkspaceContainmentError, match="workspace root"):
        m.select_workspace(name, client_id="c1")
    assert "" not in m._workspaces
    assert {ws.name for ws in m.list_workspaces()} == {"mine"}
    assert [r["name"] for r in _rows(root)] == ["mine"]
    assert m.get_selected_workspace(client_id="c1") is None


def test_the_neighbouring_name_still_selects(root):
    m = _manager(root)
    info = m.select_workspace("mine", client_id="c1")
    assert info.name == "mine"
    assert m.get_selected_workspace(client_id="c1") is info


def test_delete_refuses_the_root_before_looking_at_it(root):
    m = _manager(root)
    with pytest.raises(WorkspaceContainmentError, match="workspace root"):
        m.delete_workspace("")
    assert root.is_dir() and (root / "mine").is_dir()


def test_config_status_does_not_analyse_the_root(root):
    m = _manager(root)
    status = m.get_config_status("")
    # An empty name is "no workspace selected", as it always was ...
    assert status["workspace"] is None
    # ... and "." reaches the resolver, which refuses it rather than
    # reading the root's own .env as a workspace's.
    (root / ".env").write_text("JAATO_PROVIDER=anthropic\n")
    status = m.get_config_status(".")
    assert status["configured"] is False
    assert status.get("provider") is None
    assert "." not in m._workspaces


def test_a_registry_row_whose_path_is_the_root_answers_no_path(root):
    """The registry branch of ``get_workspace_path`` shares the comparison:
    a stored path equal to the root is refused, not returned."""
    reg = root.parent / "registry.json"
    reg.write_text(json.dumps({"root": str(root), "workspaces": [
        {"name": "mine", "path": str(root / "mine")},
    ]}))
    m = _manager(root)
    # Poison the CACHE directly, the state a pre-fix select left behind.
    from server.workspace_manager import WorkspaceInfo
    m._workspaces["workspaces"] = WorkspaceInfo(name="workspaces", path=str(root), configured=False)
    assert m.get_workspace_path("workspaces") is None
    assert m.get_workspace_path("mine") == root / "mine"


# ------------------------------------------------ the cache key is the name

def test_a_nested_name_is_refused_by_select_and_delete(root):
    """``a/b`` passed containment and was keyed ``a/b`` while analysed as
    ``b`` -- two names for one row.  The naming rule ``create`` always
    applied now binds the other verbs too."""
    (root / "mine" / "inner" / ".jaato").mkdir(parents=True)
    m = _manager(root)
    with pytest.raises(ValueError, match="Invalid workspace name"):
        m.select_workspace("mine/inner")
    with pytest.raises(ValueError, match="Invalid workspace name"):
        m.delete_workspace("mine/inner")
    assert (root / "mine" / "inner").is_dir()
    assert all(ws.name == key for key, ws in m._workspaces.items())


def test_a_symlinked_entry_is_known_by_the_name_the_client_used(root):
    """Containment resolves the link; the row keeps the client's name, so
    the key and ``WorkspaceInfo.name`` agree and the client's next verb
    finds what the list showed."""
    (root / "target" / ".jaato").mkdir(parents=True)
    (root / "alias").symlink_to(root / "target", target_is_directory=True)
    m = _manager(root)
    info = m.select_workspace("alias", client_id="c1")
    assert info.name == "alias"
    assert m._workspaces["alias"] is info
    assert Path(info.path) == (root / "target").resolve()


# ------------------------------------------------ the wire carries the row

def _ws_server(root: Path):
    """A server that never binds a port, wearing the manager ``start()``
    would have built over ``root``."""
    from server.websocket import ClientConnection, JaatoWSServer

    srv = JaatoWSServer(host="127.0.0.1", port=0, workspace_root=str(root),
                        required_token="t" * 32)
    srv._workspace_manager = _manager(root)
    ws = MagicMock()
    ws.send = AsyncMock()
    srv._clients["c1"] = ClientConnection(
        websocket=ws, client_id="c1",
        connected_at="2026-01-01T00:00:00+00:00", subscriptions=set(),
    )
    return srv


def _sent(srv, client_id="c1"):
    ws = srv._clients[client_id].websocket
    return [deserialize_event(call.args[0]) for call in ws.send.call_args_list]


@pytest.mark.asyncio
async def test_workspace_created_reaches_the_client_with_its_name_and_row(root):
    srv = _ws_server(root)
    assert srv._workspace_manager is not None
    await srv._handle_workspace_create("c1", "fresh")
    ev = _sent(srv)[-1]
    # Decoded through the SDK model -- the path that used to DROP the dict.
    assert isinstance(ev, WorkspaceCreatedEvent)
    assert ev.name == "fresh"
    assert ev.path == str((root / "fresh").resolve())
    assert ev.workspace["name"] == "fresh"
    assert ev.workspace["path"] == ev.path
    assert ev.workspace["configured"] is False
    assert ev.workspace["last_accessed"]


@pytest.mark.asyncio
async def test_workspace_list_names_the_root(root):
    srv = _ws_server(root)
    await srv._handle_workspace_list("c1")
    ev = _sent(srv)[-1]
    assert isinstance(ev, WorkspaceListEvent)
    assert ev.root == str(root.resolve())
    assert {w["name"] for w in ev.workspaces} == {"mine"}
