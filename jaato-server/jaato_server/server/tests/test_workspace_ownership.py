"""A workspace belongs to whoever created it, and the verbs agree with the list.

Every WS client used to see every workspace under the root, and the
containment guard (``test_workspace_name_containment.py``) says so in its
own docstring: it stops a name leaving the root and says nothing about
which workspace inside it a client may select.  With #1074 a connection
can carry an identity, so:

- ``create`` stamps the creator as ``owner``; the registry persists it and
  re-discovery -- which rebuilds every other field from the directory --
  preserves it;
- ``list`` shows a user their own and the UNOWNED workspaces, never another
  user's; a connection with no identity sees everything, as before;
- ``select`` and ``delete`` refuse another user's workspace, so the list's
  rule is the verbs' rule and not decoration;
- ``delete`` removes the directory and the registry row, and refuses a
  workspace something is still using.

Each refusal is paired with the same call one fact apart (a different
user, no user, nothing running), because a refusal that would have
happened anyway proves nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_server.server.workspace_manager import (
    WorkspaceContainmentError,
    WorkspaceManager,
    WorkspaceOwnershipError,
)


@pytest.fixture
def root(tmp_path: Path):
    r = tmp_path / "ws-root"
    (r / "legacy" / ".jaato").mkdir(parents=True)
    (r / "legacy" / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-x\n")
    return r


def _manager(root: Path) -> WorkspaceManager:
    return WorkspaceManager(str(root), registry_path=root.parent / "registry.json")


class TestOwnerIsRecorded:
    def test_create_stamps_the_owner_and_the_registry_keeps_it(self, root):
        m = _manager(root)
        info = m.create_workspace("mine", owner="app:alice")
        assert info.owner == "app:alice"
        rows = json.loads((root.parent / "registry.json").read_text())["workspaces"]
        assert {r["name"]: r.get("owner") for r in rows}["mine"] == "app:alice"

    def test_rediscovery_preserves_the_owner(self, root):
        m = _manager(root)
        m.create_workspace("mine", owner="app:alice")
        # A fresh manager over the same registry: discovery rebuilds every
        # field from the directory except the two the directory cannot know.
        again = _manager(root)
        by_name = {w.name: w for w in again.list_workspaces()}
        assert by_name["mine"].owner == "app:alice"
        assert by_name["legacy"].owner is None

    def test_a_workspace_created_with_no_identity_is_unowned(self, root):
        m = _manager(root)
        assert m.create_workspace("shared").owner is None


class TestVisibility:
    def test_a_user_sees_their_own_and_the_unowned_never_anothers(self, root):
        m = _manager(root)
        m.create_workspace("alices", owner="app:alice")
        m.create_workspace("bobs", owner="app:bob")
        names = {w.name for w in m.list_workspaces(for_user="app:alice")}
        assert names == {"legacy", "alices"}

    def test_no_identity_sees_everything_as_before(self, root):
        m = _manager(root)
        m.create_workspace("alices", owner="app:alice")
        assert {w.name for w in m.list_workspaces()} == {"legacy", "alices"}
        assert {w.name for w in m.list_workspaces(for_user=None)} == {"legacy", "alices"}


class TestTheVerbsAgreeWithTheList:
    def test_select_refuses_anothers_and_admits_own_and_unowned(self, root):
        m = _manager(root)
        m.create_workspace("alices", owner="app:alice")
        with pytest.raises(WorkspaceOwnershipError):
            m.select_workspace("alices", client_id="c-bob", user="app:bob")
        assert m.select_workspace("alices", client_id="c-alice", user="app:alice").owner == "app:alice"
        assert m.select_workspace("legacy", client_id="c-bob", user="app:bob").owner is None
        # the control: no identity, no refusal
        assert m.select_workspace("alices", client_id="c-anon").name == "alices"

    def test_ownership_refusal_is_not_worded_as_missing(self, root):
        m = _manager(root)
        m.create_workspace("alices", owner="app:alice")
        with pytest.raises(WorkspaceOwnershipError) as exc:
            m.select_workspace("alices", user="app:bob")
        assert "does not exist" not in str(exc.value)


class TestDelete:
    def test_deletes_the_directory_and_the_registry_row(self, root):
        m = _manager(root)
        m.create_workspace("gone", owner="app:alice")
        path = root / "gone"
        (path / "notes.txt").write_text("x")
        m.select_workspace("gone", client_id="c-alice", user="app:alice")
        info = m.delete_workspace("gone", user="app:alice", client_id="c-alice")
        assert info.name == "gone"
        assert not path.exists()
        assert "gone" not in {w.name for w in m.list_workspaces()}
        rows = json.loads((root.parent / "registry.json").read_text())["workspaces"]
        assert all(r["name"] != "gone" for r in rows)
        # the deleting client's own selection is cleared
        assert m.get_selected_workspace(client_id="c-alice") is None

    def test_refuses_anothers_workspace(self, root):
        m = _manager(root)
        m.create_workspace("alices", owner="app:alice")
        with pytest.raises(WorkspaceOwnershipError):
            m.delete_workspace("alices", user="app:bob")
        assert (root / "alices").exists()
        m.delete_workspace("alices", user="app:alice")  # the control
        assert not (root / "alices").exists()

    def test_refuses_a_workspace_with_loaded_sessions(self, root):
        m = _manager(root)
        m.create_workspace("busy")
        with pytest.raises(ValueError, match="loaded session"):
            m.delete_workspace("busy", in_use_by=["20260916_1"])
        assert (root / "busy").exists()
        m.delete_workspace("busy", in_use_by=[])  # the control
        assert not (root / "busy").exists()

    def test_refuses_a_workspace_another_client_has_selected(self, root):
        m = _manager(root)
        m.create_workspace("shared")
        m.select_workspace("shared", client_id="c-other")
        with pytest.raises(ValueError, match="other client"):
            m.delete_workspace("shared")
        m.remove_client("c-other")
        m.delete_workspace("shared")  # the control
        assert not (root / "shared").exists()

    @pytest.mark.parametrize("bad", ["..", "../outside", "/etc"])
    def test_refuses_a_name_that_leaves_the_root_and_the_root_itself(self, root, bad):
        (root.parent / "outside").mkdir(exist_ok=True)
        m = _manager(root)
        with pytest.raises(WorkspaceContainmentError):
            m.delete_workspace(bad)
        assert root.exists() and (root.parent / "outside").exists()

    def test_refuses_a_missing_workspace(self, root):
        m = _manager(root)
        with pytest.raises(ValueError, match="does not exist"):
            m.delete_workspace("nope")
