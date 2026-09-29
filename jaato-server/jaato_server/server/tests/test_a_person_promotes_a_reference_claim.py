"""A person promotes an agent's reference claim; the daemon writes the catalog.

``proposeReference`` writes a CLAIM and never a catalog entry, because every
AppArmor body denies the runner writes to ``.jaato/references/**``.  The
other half is ``reference.promote`` / ``reference.dismiss`` (protocol 1.32),
run by the daemon (:mod:`jaato_server.server.reference_curation`).  What it
must get right, each its own failure if dropped:

1. **Only the workspace owner curates.**  The memory rail's rule
   (``may_curate``), with the identity read from the TRANSPORT.
2. **``created_by`` is re-derived, never copied.**  The claims directory is
   model-writable, so a claim's ``created_by`` is whatever something wrote
   there.  The promoted entry's is the daemon's record of who the named
   session was created for, and only when that session ran in this
   workspace.
3. **``curated_by`` names the person who promoted it.**
4. **The claim is re-validated.**  A claim edited after it was proposed --
   a ``path`` pointing out of the workspace -- is refused, not copied.
5. **No link is followed and nothing is overwritten.**  A symlinked claim
   file and an existing catalog file of the same name are refused.
6. **A promoted local reference loads.**  Its ``path`` is re-anchored to
   the catalog file, so the catalog loader resolves the document the claim
   named.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.events import ReferenceCurationResultEvent
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.reference_curation import curate_claim
from jaato_server.server.session_manager import SessionManager
from jaato_server.server.session_workspace_index import SessionWorkspaceIndex
from jaato_server.shared.plugins.references.claims import (
    build_proposed_reference,
    claims_dir,
    new_claim,
    write_claim,
)
from jaato_server.shared.plugins.references.config_loader import discover_references
from jaato_server.shared.tests.reversion import Reversion

_CURATION = "jaato-server/jaato_server/server/reference_curation.py"
_ROUTER = "jaato-server/jaato_server/server/command_router.py"
_MANAGER = "jaato-server/jaato_server/server/session_manager.py"

REVERSIONS = [
    Reversion(
        target=_CURATION,
        find="    if not may_curate(owner, user_id):\n",
        replace="    if False:\n",
        because="anyone on the connection could write the owner's catalog",
        test="TestOnlyTheOwnerCurates::test_another_user_is_refused",
    ),
    Reversion(
        target=_CURATION,
        find="        created_by = creator_in_workspace(session_id, root)\n",
        replace="        created_by = recorded.created_by\n",
        because=(
            "the claim file is model-writable; copying its created_by lets a "
            "model name any user as the author of a catalog entry"
        ),
        test="TestCreatedByIsDerived::test_the_claims_own_created_by_is_not_kept",
    ),
    Reversion(
        target=_CURATION,
        find="    entry, errors = build_proposed_reference(claim_as_args(claim), workspace=root,\n"
             "                                             catalog_ids=ids)\n",
        replace="    entry, errors = dict(claim[\"reference\"]), []\n",
        because=(
            "a claim edited after it was proposed would be copied as is, "
            "path out of the workspace included"
        ),
        test="TestTheClaimIsRevalidated::test_a_path_out_of_the_workspace_is_refused",
    ),
    Reversion(
        target=_CURATION,
        find="    if os.path.islink(path) or not os.path.isfile(path):\n",
        replace="    if not os.path.isfile(path):\n",
        because=(
            "a claim file planted as a link would point the daemon's read, "
            "and its unlink, at a file outside the workspace"
        ),
        test="TestNoLinkIsFollowed::test_a_symlinked_claim_is_refused",
    ),
    Reversion(
        target=_CURATION,
        find="    if ref_id in ids or os.path.lexists(os.path.join(root, rel_file)):\n",
        replace="    if ref_id in ids:\n",
        because="a promotion would overwrite a catalog file someone curated",
        test="TestNothingIsOverwritten::test_an_existing_file_of_that_name_is_kept",
    ),
    Reversion(
        target=_CURATION,
        find='        out["path"] = os.path.relpath(target, os.path.join(root, dest_rel)).replace(os.sep, "/")\n',
        replace="        pass\n",
        because=(
            "the catalog loader resolves a relative path against the "
            "reference file's directory, so the promoted entry would name a "
            "document that is not there"
        ),
        test="TestAPromotedReferenceLoads::test_the_path_resolves_to_the_proposed_document",
    ),
    Reversion(
        target=_ROUTER,
        find="        user_id = self._event_sink.get_client_user(client_id)\n"
             "        outcome = curate_claim(",
        replace="        user_id = None\n"
                "        outcome = curate_claim(",
        because="the owner gate would never see who is asking",
        test="TestTheRouter::test_the_transport_identity_reaches_the_gate",
    ),
    Reversion(
        target=_MANAGER,
        find="        if not where or os.path.realpath(where) != want:\n            return None\n"
             "        row = index.membership(session_id) or {}\n",
        replace="        row = index.membership(session_id) or {}\n",
        because=(
            "a claim naming another workspace's session would borrow that "
            "session's creator"
        ),
        test="TestCreatorInWorkspace::test_a_cold_session_in_another_workspace_answers_none",
    ),
]


class _Session:
    """The two attributes ``proposing_origin`` reads."""

    def __init__(self, user: Optional[str] = "acme:alice") -> None:
        self._client_user_id = user

    def _model_provenance(self) -> Dict[str, Any]:
        return {"kind": "ai", "provider": "p", "model": "m",
                "session_id": "S1", "agent_id": "doc"}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "docs").mkdir(parents=True)
    (ws / "docs" / "runbook.md").write_text("# Runbook\n", encoding="utf-8")
    (ws / ".jaato" / "references").mkdir(parents=True)
    return ws


def _propose(ws: Path, *, ref_id: str = "runbook", **extra: Any) -> Dict[str, Any]:
    args = {"id": ref_id, "name": "Runbook", "description": "how to deploy",
            "tags": ["ops"], "path": "docs/runbook.md", **extra}
    entry, errors = build_proposed_reference(args, workspace=str(ws), catalog_ids=[])
    assert entry is not None, errors
    claim = new_claim(entry, _Session())
    write_claim(str(ws), claim)
    return claim


def _creator(mapping: Optional[Dict[str, str]] = None):
    def lookup(session_id: str, workspace: str) -> Optional[str]:
        return (mapping or {}).get(session_id)
    return lookup


def _promote(ws: Path, claim_id: str, *, owner: Optional[str] = None,
             user: Optional[str] = None, creators: Optional[Dict[str, str]] = None):
    return curate_claim(str(ws), "promote", claim_id, owner=owner, user_id=user,
                        creator_in_workspace=_creator(creators))


def _catalog_file(ws: Path, ref_id: str = "runbook") -> Path:
    return ws / ".jaato" / "references" / f"{ref_id}.json"


class TestOnlyTheOwnerCurates:
    def test_another_user_is_refused(self, workspace: Path) -> None:
        claim = _propose(workspace)
        out = _promote(workspace, claim["claim_id"], owner="acme:alice", user="acme:bob")
        assert (out.ok, out.category) == (False, "not_owner")
        assert not _catalog_file(workspace).exists()
        assert (claims_dir(str(workspace)) / f"{claim['claim_id']}.json").exists()

    def test_the_owner_promotes(self, workspace: Path) -> None:
        claim = _propose(workspace)
        out = _promote(workspace, claim["claim_id"], owner="acme:alice", user="acme:alice")
        assert out.ok, out.error
        assert out.reference_file == ".jaato/references/runbook.json"
        assert _catalog_file(workspace).is_file()
        assert not (claims_dir(str(workspace)) / f"{claim['claim_id']}.json").exists()

    def test_an_identity_less_connection_may_not_on_an_owned_workspace(
            self, workspace: Path) -> None:
        claim = _propose(workspace)
        out = _promote(workspace, claim["claim_id"], owner="acme:alice", user=None)
        assert out.category == "not_owner"

    def test_an_unowned_workspace_is_curated_by_anyone(self, workspace: Path) -> None:
        claim = _propose(workspace)
        assert _promote(workspace, claim["claim_id"]).ok


class TestCreatedByIsDerived:
    def test_the_claims_own_created_by_is_not_kept(self, workspace: Path) -> None:
        claim = _propose(workspace)
        path = claims_dir(str(workspace)) / f"{claim['claim_id']}.json"
        data = json.loads(path.read_text())
        data["origin"]["created_by"] = "acme:mallory"
        path.write_text(json.dumps(data))
        out = _promote(workspace, claim["claim_id"], creators={"S1": "acme:alice"})
        assert out.ok, out.error
        origin = json.loads(_catalog_file(workspace).read_text())["origin"]
        assert origin["created_by"] == "acme:alice"

    def test_an_unplaceable_session_leaves_it_absent(self, workspace: Path) -> None:
        claim = _propose(workspace)
        assert _promote(workspace, claim["claim_id"], creators={}).ok
        origin = json.loads(_catalog_file(workspace).read_text())["origin"]
        assert "created_by" not in origin
        assert origin["generated_by"]["session_id"] == "S1"
        assert origin["claim_id"] == claim["claim_id"]


class TestCuratedByIsStamped:
    def test_the_promoting_person_is_recorded(self, workspace: Path) -> None:
        claim = _propose(workspace)
        _promote(workspace, claim["claim_id"], owner="acme:alice", user="acme:alice")
        origin = json.loads(_catalog_file(workspace).read_text())["origin"]
        assert origin["curated_by"] == {"kind": "human", "via": "reference.promote",
                                        "user": "acme:alice"}

    def test_no_identity_claims_no_user(self, workspace: Path) -> None:
        claim = _propose(workspace)
        _promote(workspace, claim["claim_id"])
        origin = json.loads(_catalog_file(workspace).read_text())["origin"]
        assert origin["curated_by"] == {"kind": "human", "via": "reference.promote"}


class TestTheClaimIsRevalidated:
    def test_a_path_out_of_the_workspace_is_refused(self, workspace: Path, tmp_path: Path) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("x")
        claim = _propose(workspace)
        path = claims_dir(str(workspace)) / f"{claim['claim_id']}.json"
        data = json.loads(path.read_text())
        data["reference"]["path"] = "../secret.txt"
        path.write_text(json.dumps(data))
        out = _promote(workspace, claim["claim_id"])
        assert (out.ok, out.category) == (False, "invalid_claim")
        assert not _catalog_file(workspace).exists()

    def test_an_unknown_claim_is_not_found(self, workspace: Path) -> None:
        assert _promote(workspace, "20260101T000000Z-deadbeef").category == "not_found"

    def test_a_malformed_claim_id_is_refused_before_any_path_is_built(
            self, workspace: Path) -> None:
        assert _promote(workspace, "../../etc/passwd").category == "invalid_request"


class TestNoLinkIsFollowed:
    def test_a_symlinked_claim_is_refused(self, workspace: Path, tmp_path: Path) -> None:
        claim = _propose(workspace)
        path = claims_dir(str(workspace)) / f"{claim['claim_id']}.json"
        outside = tmp_path / "planted.json"
        outside.write_text(path.read_text())
        path.unlink()
        path.symlink_to(outside)
        out = _promote(workspace, claim["claim_id"])
        assert (out.ok, out.category) == (False, "unsafe_path")
        assert outside.exists()

    def test_a_catalog_directory_linked_out_is_refused(self, workspace: Path,
                                                       tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (workspace / ".jaato" / "references").rmdir()
        (workspace / ".jaato" / "references").symlink_to(elsewhere)
        claim = _propose(workspace)
        out = _promote(workspace, claim["claim_id"])
        assert (out.ok, out.category) == (False, "unsafe_path")
        assert list(elsewhere.iterdir()) == []


class TestNothingIsOverwritten:
    def test_a_catalog_id_collides(self, workspace: Path) -> None:
        claim = _propose(workspace)
        (workspace / ".jaato" / "references" / "other.json").write_text(
            json.dumps({"id": "runbook", "name": "curated"}))
        out = _promote(workspace, claim["claim_id"])
        assert (out.ok, out.category) == (False, "collision")
        assert (claims_dir(str(workspace)) / f"{claim['claim_id']}.json").exists()

    def test_an_existing_file_of_that_name_is_kept(self, workspace: Path) -> None:
        claim = _propose(workspace)
        existing = _catalog_file(workspace)
        existing.write_text(json.dumps({"id": "curated-elsewhere", "name": "keep me"}))
        out = _promote(workspace, claim["claim_id"])
        assert out.category == "collision"
        assert json.loads(existing.read_text())["name"] == "keep me"


class TestAPromotedReferenceLoads:
    def test_the_path_resolves_to_the_proposed_document(self, workspace: Path) -> None:
        claim = _propose(workspace)
        assert _promote(workspace, claim["claim_id"], creators={"S1": "acme:alice"}).ok
        [source] = discover_references(".jaato/references", base_path=str(workspace))
        assert source.id == "runbook"
        assert source.resolved_path == "docs/runbook.md"
        assert source.origin is not None and source.origin.curated_by is not None
        assert "promoted" in source.origin.describe()

    def test_an_inline_claim_promotes_as_is(self, workspace: Path) -> None:
        args = {"id": "note", "name": "Note", "content": "remember this"}
        entry, _ = build_proposed_reference(args, workspace=str(workspace), catalog_ids=[])
        claim = new_claim(entry, _Session())
        write_claim(str(workspace), claim)
        assert _promote(workspace, claim["claim_id"]).ok
        data = json.loads(_catalog_file(workspace, "note").read_text())
        assert (data["type"], data["content"], data["mode"]) == ("inline", "remember this",
                                                                 "selectable")


class TestDismiss:
    def test_dismiss_removes_the_claim_and_writes_nothing(self, workspace: Path) -> None:
        claim = _propose(workspace)
        out = curate_claim(str(workspace), "dismiss", claim["claim_id"], owner=None,
                           user_id=None, creator_in_workspace=_creator())
        assert out.ok
        assert not (claims_dir(str(workspace)) / f"{claim['claim_id']}.json").exists()
        assert not _catalog_file(workspace).exists()


class _Sink:
    def __init__(self, user: Optional[str]) -> None:
        self.user = user
        self.sent: List[Any] = []

    def get_client_user(self, client_id: str) -> Optional[str]:
        return self.user

    def send_event(self, client_id: str, event: Any) -> None:
        self.sent.append(event)


class _Manager:
    def __init__(self, ws: Path, owner: Optional[str]) -> None:
        self.ws, self.owner = ws, owner

    def get_client_session(self, client_id: str) -> Any:
        return SimpleNamespace(workspace_path=str(self.ws))

    def get_session(self, session_id: str) -> Any:
        return None

    def _workspace_owner_of(self, workspace_path: Optional[str]) -> Optional[str]:
        return self.owner

    def creator_in_workspace(self, session_id: str, workspace: str) -> Optional[str]:
        return None


def _router(ws: Path, *, owner: Optional[str], user: Optional[str]) -> CommandRouter:
    router = object.__new__(CommandRouter)
    router._event_sink = _Sink(user)
    router._session_manager = _Manager(ws, owner)
    return router


class TestTheRouter:
    def test_the_transport_identity_reaches_the_gate(self, workspace: Path) -> None:
        claim = _propose(workspace)
        router = _router(workspace, owner="acme:alice", user="acme:alice")
        assert router._dispatch_prefixed_command(
            "reference.promote", "c1", [claim["claim_id"]], None, None)
        [event] = router._event_sink.sent
        assert isinstance(event, ReferenceCurationResultEvent)
        assert (event.ok, event.action, event.reference_id) == (True, "promote", "runbook")

    def test_a_refusal_is_answered_too(self, workspace: Path) -> None:
        claim = _propose(workspace)
        router = _router(workspace, owner="acme:alice", user="acme:bob")
        router._dispatch_prefixed_command("reference.dismiss", "c1",
                                          [claim["claim_id"]], None, None)
        [event] = router._event_sink.sent
        assert (event.ok, event.category, event.action) == (False, "not_owner", "dismiss")


class TestCreatorInWorkspace:
    def _manager(self, tmp_path: Path, loaded: Any = None) -> SessionManager:
        manager = object.__new__(SessionManager)
        manager._session_workspace_index = SessionWorkspaceIndex(tmp_path / "index.json")
        manager.get_session = lambda session_id: loaded  # type: ignore[method-assign]
        return manager

    def test_a_cold_session_in_this_workspace_answers_its_creator(
            self, tmp_path: Path, workspace: Path) -> None:
        manager = self._manager(tmp_path)
        manager._session_workspace_index.record("S1", str(workspace))
        manager._session_workspace_index.record_membership(
            "S1", created_by="acme:alice", cascade_driver_id=None, sibling_name=None)
        assert manager.creator_in_workspace("S1", str(workspace)) == "acme:alice"

    def test_a_cold_session_in_another_workspace_answers_none(
            self, tmp_path: Path, workspace: Path) -> None:
        manager = self._manager(tmp_path)
        manager._session_workspace_index.record("S1", str(tmp_path / "other"))
        manager._session_workspace_index.record_membership(
            "S1", created_by="acme:alice", cascade_driver_id=None, sibling_name=None)
        assert manager.creator_in_workspace("S1", str(workspace)) is None

    def test_a_loaded_session_elsewhere_answers_none(self, tmp_path: Path,
                                                     workspace: Path) -> None:
        loaded = SimpleNamespace(workspace_path=str(tmp_path / "other"),
                                 created_by="acme:alice")
        assert self._manager(tmp_path, loaded).creator_in_workspace(
            "S1", str(workspace)) is None
