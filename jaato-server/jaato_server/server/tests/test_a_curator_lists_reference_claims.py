"""A curator can SEE the reference claims agents proposed (protocol 1.32).

``reference.promote`` / ``reference.dismiss`` turned a claim into a catalog
entry or dropped it, and nothing listed claims to a client: the only
listing was ``listReferences``, which is the MODEL's tool.  A person had to
know a claim id from somewhere to act on it.  ``ReferenceClaimsRequest`` ->
``ReferenceClaimsEvent`` is that listing, and ``ReferenceCurationRequest``
is the correlated form of the two verbs, so a client can await each
answer by ``request_id``.

Properties, each a way the listing could mislead a curator:

1. **It reads what a promotion would read**: the claims directory resolved
   inside the workspace, a symlinked claim file never followed, every
   record re-checked -- the directory is model-writable.
2. **What cannot be shown is named**, never dropped, and a directory that
   cannot be read is ``ok=False``, never "nothing proposed".
3. **A claim that promotion would refuse says why** (``problems``), so the
   curator is not told after pressing Promote.
4. **``may_curate`` is the promotion gate's answer**, from the transport's
   identity.
5. **The correlated answers carry the caller's ``request_id``.**
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.events import (
    ReferenceClaimsEvent,
    ReferenceClaimsRequest,
    ReferenceCurationRequest,
    ReferenceCurationResultEvent,
    deserialize_event,
    serialize_event,
)
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.reference_curation import list_claims
from jaato_server.shared.plugins.references.claims import (
    build_proposed_reference,
    claims_dir,
    new_claim,
    write_claim,
)
from jaato_server.shared.tests.reversion import Reversion

_CURATION = "jaato-server/jaato_server/server/reference_curation.py"
_ROUTER = "jaato-server/jaato_server/server/command_router.py"

REVERSIONS = [
    Reversion(
        target=_CURATION,
        find="    if not os.path.isfile(path) or os.path.islink(path):\n        return None\n",
        replace="    if not os.path.isfile(path):\n        return None\n",
        because=(
            "the listing follows a claim file planted as a link, so the "
            "daemon reads -- and shows a curator -- a file outside the workspace"
        ),
        test="TestTheListingReadsLikeAPromotion::test_a_symlinked_claim_is_not_followed",
    ),
    Reversion(
        target=_CURATION,
        find="            listing.unreadable.append(name)\n",
        replace="            pass\n",
        because="a file that is not a claim vanishes from the listing in silence",
        test="TestTheListingReadsLikeAPromotion::test_a_non_claim_is_named_not_dropped",
    ),
    Reversion(
        target=_CURATION,
        find="    row[\"problems\"] = problems\n",
        replace="    row[\"problems\"] = []\n",
        because=(
            "a claim promotion would refuse looks promotable, and the "
            "curator learns why only after pressing Promote"
        ),
        test="TestProblemsAreShownBeforePromote::test_a_colliding_id_is_a_problem",
    ),
    Reversion(
        target=_ROUTER,
        find="            may_curate=may_curate(self._workspace_owner(workspace), user_id)))\n",
        replace="            may_curate=True))\n",
        because=(
            "every connection is told it may curate, so a client offers "
            "Promote to a user the daemon then refuses"
        ),
        test="TestTheRouter::test_may_curate_is_the_owner_gate",
    ),
    Reversion(
        target=_ROUTER,
        find="        answer = functools.partial(\n"
             "            ReferenceCurationResultEvent, request_id=request_id,\n",
        replace="        answer = functools.partial(\n"
                "            ReferenceCurationResultEvent, request_id=\"\",\n",
        because="a client awaiting its promote by request_id never gets an answer",
        test="TestTheRouter::test_a_curation_request_is_answered_by_its_request_id",
    ),
]


class _Session:
    _client_user_id = "acme:alice"

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


class TestTheListingReadsLikeAPromotion:
    def test_a_claim_is_listed_with_what_a_curator_needs(self, workspace: Path) -> None:
        claim = _propose(workspace)
        listing = list_claims(str(workspace))
        assert listing.ok and listing.unreadable == []
        [row] = listing.claims
        assert row["claim_id"] == claim["claim_id"]
        assert (row["id"], row["name"], row["description"]) == (
            "runbook", "Runbook", "how to deploy")
        assert (row["type"], row["path"], row["tags"]) == ("local", "docs/runbook.md", ["ops"])
        assert row["origin"]["created_by"] == "acme:alice"
        assert row["problems"] == []

    def test_inline_content_is_shown(self, workspace: Path) -> None:
        _propose(workspace, ref_id="note", path=None, content="the steps")
        [row] = list_claims(str(workspace)).claims
        assert (row["type"], row["content"]) == ("inline", "the steps")
        assert "path" not in row

    def test_no_claims_directory_lists_nothing(self, workspace: Path) -> None:
        listing = list_claims(str(workspace))
        assert listing.ok and listing.claims == [] and listing.unreadable == []

    def test_a_symlinked_claim_is_not_followed(self, workspace: Path, tmp_path: Path) -> None:
        claim = _propose(workspace)
        outside = tmp_path / "planted.json"
        outside.write_text(json.dumps({**claim, "claim_id": "planted"}), encoding="utf-8")
        os.symlink(outside, claims_dir(str(workspace)) / "planted.json")
        listing = list_claims(str(workspace))
        assert [r["claim_id"] for r in listing.claims] == [claim["claim_id"]]
        assert listing.unreadable == ["planted.json"]

    def test_a_non_claim_is_named_not_dropped(self, workspace: Path) -> None:
        _propose(workspace)
        (claims_dir(str(workspace)) / "junk.json").write_text("{\"x\": 1}", encoding="utf-8")
        (claims_dir(str(workspace)) / "broken.json").write_text("{", encoding="utf-8")
        listing = list_claims(str(workspace))
        assert len(listing.claims) == 1
        assert sorted(listing.unreadable) == ["broken.json", "junk.json"]

    def test_a_claim_file_under_another_name_is_not_listed(self, workspace: Path) -> None:
        claim = _propose(workspace)
        src = claims_dir(str(workspace)) / f"{claim['claim_id']}.json"
        src.rename(src.with_name("renamed.json"))
        listing = list_claims(str(workspace))
        assert listing.claims == [] and listing.unreadable == ["renamed.json"]

    def test_a_claims_directory_linked_out_is_unsafe(self, workspace: Path, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        os.symlink(elsewhere, workspace / ".jaato" / "references-claims")
        listing = list_claims(str(workspace))
        assert (listing.ok, listing.category) == (False, "unsafe_path")


class TestProblemsAreShownBeforePromote:
    def test_a_colliding_id_is_a_problem(self, workspace: Path) -> None:
        _propose(workspace)
        (workspace / ".jaato" / "references" / "runbook.json").write_text(
            json.dumps({"id": "runbook", "name": "x", "type": "inline", "content": "y",
                        "mode": "selectable"}), encoding="utf-8")
        [row] = list_claims(str(workspace)).claims
        assert any("already in the catalog" in p for p in row["problems"])

    def test_a_document_that_is_gone_is_a_problem(self, workspace: Path) -> None:
        _propose(workspace)
        (workspace / "docs" / "runbook.md").unlink()
        [row] = list_claims(str(workspace)).claims
        assert row["problems"]


class _Sink:
    def __init__(self, user: Optional[str], workspace: Optional[str] = None) -> None:
        self.user, self.workspace = user, workspace
        self.sent: List[Any] = []

    def get_client_user(self, client_id: str) -> Optional[str]:
        return self.user

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        return self.workspace

    def send_event(self, client_id: str, event: Any) -> None:
        self.sent.append(event)


class _Manager:
    def __init__(self, ws: Optional[Path], owner: Optional[str]) -> None:
        self.ws, self.owner = ws, owner

    def get_client_session(self, client_id: str) -> Any:
        return SimpleNamespace(workspace_path=str(self.ws)) if self.ws else None

    def get_session(self, session_id: str) -> Any:
        return None

    def _workspace_owner_of(self, workspace_path: Optional[str]) -> Optional[str]:
        return self.owner

    def creator_in_workspace(self, session_id: str, workspace: str) -> Optional[str]:
        return None


def _router(ws: Optional[Path], *, owner: Optional[str], user: Optional[str]) -> CommandRouter:
    router = object.__new__(CommandRouter)
    router._event_sink = _Sink(user)
    router._session_manager = _Manager(ws, owner)
    return router


class TestTheRouter:
    def test_a_listing_is_answered_by_its_request_id(self, workspace: Path) -> None:
        _propose(workspace)
        router = _router(workspace, owner=None, user=None)
        router._dispatch("c1", "", ReferenceClaimsRequest(request_id="rq-1"))
        [event] = router._event_sink.sent
        assert isinstance(event, ReferenceClaimsEvent)
        assert (event.request_id, event.ok, len(event.claims)) == ("rq-1", True, 1)

    def test_may_curate_is_the_owner_gate(self, workspace: Path) -> None:
        owner = _router(workspace, owner="acme:alice", user="acme:alice")
        owner._dispatch("c1", "", ReferenceClaimsRequest(request_id="a"))
        other = _router(workspace, owner="acme:alice", user="acme:bob")
        other._dispatch("c1", "", ReferenceClaimsRequest(request_id="b"))
        assert owner._event_sink.sent[0].may_curate is True
        assert other._event_sink.sent[0].may_curate is False

    def test_no_workspace_is_said(self) -> None:
        router = _router(None, owner=None, user=None)
        router._dispatch("c1", "", ReferenceClaimsRequest(request_id="rq"))
        [event] = router._event_sink.sent
        assert (event.ok, event.category, event.request_id) == (False, "no_workspace", "rq")

    def test_a_curation_request_is_answered_by_its_request_id(self, workspace: Path) -> None:
        claim = _propose(workspace)
        router = _router(workspace, owner=None, user=None)
        router._dispatch("c1", "", ReferenceCurationRequest(
            request_id="rq-2", action="promote", claim_id=claim["claim_id"]))
        [event] = router._event_sink.sent
        assert isinstance(event, ReferenceCurationResultEvent)
        assert (event.request_id, event.ok, event.reference_id) == ("rq-2", True, "runbook")

    def test_an_unknown_action_is_refused_not_guessed(self, workspace: Path) -> None:
        claim = _propose(workspace)
        router = _router(workspace, owner=None, user=None)
        router._dispatch("c1", "", ReferenceCurationRequest(
            request_id="rq", action="delete", claim_id=claim["claim_id"]))
        [event] = router._event_sink.sent
        assert (event.ok, event.category) == (False, "invalid_request")
        assert (claims_dir(str(workspace)) / f"{claim['claim_id']}.json").exists()


def test_the_requests_cross_the_wire() -> None:
    for event in (ReferenceClaimsRequest(request_id="x"),
                  ReferenceCurationRequest(request_id="y", action="dismiss", claim_id="c")):
        assert deserialize_event(serialize_event(event)) == event
