"""A workspace the list shows is one the verbs can act on.

``~/.jaato/workspaces.json`` is one file per daemon user, keyed by
workspace NAME, and it survives a change of ``--workspace-root``.  A row
written under one root was loaded verbatim under the next, so it sat in
every ``workspace.list`` while ``select`` and ``delete`` -- which resolve
the NAME under the CURRENT root -- answered "does not exist".  Seen live:
a row named ``workspaces`` (the new root's own directory, recorded when
the root was one level up) that nobody could open or remove.

Two reconciliations, at the two points that know something:

- ``_load_registry`` keeps a row only when the path it records is
  ``<root>/<name>`` -- the same resolution the verbs perform;
- ``discover_workspaces`` forgets a cached row whose directory is gone,
  since it is the one point that looks at the disk.

Each case is paired with the row that must SURVIVE the same pass (owner
and last-opened time included), because a prune that empties the list
proves nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_server.server.workspace_manager import WorkspaceManager


@pytest.fixture
def roots(tmp_path: Path):
    """Two nested roots, the shape the live defect had: ``old/workspaces`` is
    both a workspace under ``old`` and the whole root once it moves down."""
    old = tmp_path / "old"
    new = old / "workspaces"
    (new / "mine" / ".jaato").mkdir(parents=True)
    (new / "mine" / ".env").write_text("")
    (new / ".jaato").mkdir()          # what made ``workspaces`` look like a workspace of ``old``
    return old, new


def _registry(path: Path, root: Path, rows: list[dict]) -> Path:
    reg = path / "registry.json"
    reg.write_text(json.dumps({"root": str(root), "workspaces": rows}))
    return reg


class TestARowFromAnotherRootIsNotListed:
    def test_the_old_roots_row_is_dropped_and_the_current_one_kept(self, roots, tmp_path):
        old, new = roots
        reg = _registry(tmp_path, old, [
            {"name": "workspaces", "path": str(new), "configured": False, "last_accessed": "2026-09-16T19:43:00+00:00"},
            {"name": "mine", "path": str(new / "mine"), "configured": False, "owner": "app:alice", "last_accessed": "2026-09-15T09:00:00+00:00"},
        ])
        m = WorkspaceManager(str(new), registry_path=reg)
        names = {ws.name for ws in m.list_workspaces()}
        assert names == {"mine"}
        # The surviving row keeps what the directory cannot tell us.
        mine = next(ws for ws in m.list_workspaces() if ws.name == "mine")
        assert mine.owner == "app:alice"
        assert mine.last_accessed == "2026-09-15T09:00:00+00:00"

    def test_the_verbs_and_the_list_now_agree(self, roots, tmp_path):
        old, new = roots
        reg = _registry(tmp_path, old, [{"name": "workspaces", "path": str(new)}])
        m = WorkspaceManager(str(new), registry_path=reg)
        assert "workspaces" not in {ws.name for ws in m.list_workspaces()}
        with pytest.raises(ValueError, match="does not exist"):
            m.delete_workspace("workspaces")

    def test_the_file_is_rewritten_without_the_stale_row(self, roots, tmp_path):
        old, new = roots
        reg = _registry(tmp_path, old, [{"name": "workspaces", "path": str(new)}])
        m = WorkspaceManager(str(new), registry_path=reg)
        m.list_workspaces()
        rows = json.loads(reg.read_text())
        assert [r["name"] for r in rows["workspaces"]] == ["mine"]
        assert rows["root"] == str(new.resolve())

    def test_a_row_at_the_same_root_is_kept_as_before(self, roots, tmp_path):
        _old, new = roots
        reg = _registry(tmp_path, new, [{"name": "mine", "path": str(new / "mine"), "owner": "app:bob"}])
        m = WorkspaceManager(str(new), registry_path=reg)
        assert {ws.name: ws.owner for ws in m.list_workspaces()} == {"mine": "app:bob"}

    def test_a_row_with_no_recorded_path_is_kept_on_its_name(self, roots, tmp_path):
        _old, new = roots
        reg = _registry(tmp_path, new, [{"name": "mine", "owner": "app:bob"}])
        m = WorkspaceManager(str(new), registry_path=reg)
        assert {ws.name: ws.owner for ws in m.list_workspaces()} == {"mine": "app:bob"}


class TestADirectoryRemovedOutOfBandLeavesTheList:
    def test_discovery_forgets_the_row(self, roots, tmp_path):
        _old, new = roots
        reg = _registry(tmp_path, new, [])
        m = WorkspaceManager(str(new), registry_path=reg)
        m.create_workspace("gone", owner="app:alice")
        assert {ws.name for ws in m.list_workspaces()} == {"mine", "gone"}
        import shutil
        shutil.rmtree(new / "gone")
        assert {ws.name for ws in m.list_workspaces()} == {"mine"}
        assert [r["name"] for r in json.loads(reg.read_text())["workspaces"]] == ["mine"]
