"""A person reads the reference catalog and edits a reference's typed links.

After a reference is in the catalog -- written by hand, or promoted from an
agent's claim -- its ``links`` could be changed only by editing the file on
the host: a confined runner is write-denied on ``.jaato/references/**``.
``reference.catalog`` and ``reference.links`` (protocol 1.33,
:mod:`jaato_server.server.reference_catalog`) are the daemon's half.  What
each must get right:

1. **Only the workspace owner edits** (``may_curate``, the identity read
   from the transport).  A ``supersedes`` reroutes every request for its
   target, so an edge is a curator's act.
2. **The links are validated before anything is written**: an unknown
   ``rel`` or an edge to the reference itself is refused, the file intact.
3. **Only the ``links`` key changes**: name, description, tags and origin
   are written back as they were.
4. **An id declared in two files is not edited**: which one the loader
   keeps is not this verb's to guess.
5. **The listing reads what the loader reads**: the root and each
   sub-bundle with a ``bundle.json``, never a link, never an unrelated
   subdirectory -- and it marks a dangling edge instead of dropping it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from jaato_sdk.events import (
    ReferenceCatalogEvent,
    ReferenceCatalogRequest,
    ReferenceLinksUpdateRequest,
    ReferenceLinksUpdateResultEvent,
    deserialize_event,
    serialize_event,
)
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.reference_catalog import list_catalog, update_links
from jaato_server.shared.plugins.references.config_loader import discover_references
from jaato_server.shared.tests.reversion import Reversion

_CATALOG = "jaato-server/jaato_server/server/reference_catalog.py"
_ROUTER = "jaato-server/jaato_server/server/command_router.py"

REVERSIONS = [
    Reversion(
        target=_CATALOG,
        find="    if not may_curate(owner, user_id):\n"
             "        return _fail(outcome, \"not_owner\",\n"
             "                     \"only the workspace owner may change its references' links\")\n",
        replace="",
        because="anyone on the connection could reroute the owner's catalog",
        test="TestOnlyTheOwnerEdits::test_another_user_is_refused",
    ),
    Reversion(
        target=_CATALOG,
        find="    errors = link_errors(links, source_id=reference_id)\n",
        replace="    errors = []\n",
        because=(
            "a malformed edge would be written, then silently dropped by the "
            "lenient loader -- saved, and not there"
        ),
        test="TestTheLinksAreValidated::test_an_unknown_rel_is_refused",
    ),
    Reversion(
        target=_CATALOG,
        find="    updated = dict(data)\n",
        replace="    updated = {\"id\": data[\"id\"]}\n",
        because="editing an edge would wipe the rest of the reference",
        test="TestOnlyTheLinksChange::test_the_rest_of_the_file_is_kept",
    ),
    Reversion(
        target=_CATALOG,
        find="    if len(matches) > 1:\n",
        replace="    if False:\n",
        because="an id in two files would be edited in whichever came first",
        test="TestAnAmbiguousIdIsNotEdited::test_a_duplicate_id_is_refused",
    ),
    Reversion(
        target=_CATALOG,
        find="                and os.path.isfile(os.path.join(path, BUNDLE_MANIFEST_FILENAME))):\n",
        replace="                ):\n",
        because=(
            "the view would list files the loader never reads, and let a "
            "person edit a reference no session sees"
        ),
        test="TestTheListing::test_a_directory_without_a_manifest_is_not_the_catalog",
    ),
    Reversion(
        target=_ROUTER,
        find="        outcome = update_links(\n"
             "            workspace, event.reference_id, event.links,\n"
             "            owner=self._workspace_owner(workspace), user_id=user_id)\n",
        replace="        outcome = update_links(\n"
                "            workspace, event.reference_id, event.links,\n"
                "            owner=None, user_id=user_id)\n",
        because="the gate would never see who owns the workspace",
        test="TestTheRouter::test_the_owner_reaches_the_gate",
    ),
]


def _ref(ref_id: str, **extra: Any) -> dict:
    data = {"id": ref_id, "name": ref_id.upper(), "description": f"about {ref_id}",
            "type": "inline", "content": "x", "mode": "selectable", "tags": ["t"]}
    data.update(extra)
    return data


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"
    _write(refs / "adr-1.json", _ref("adr-1"))
    _write(refs / "adr-2.json", _ref("adr-2", origin={"kind": "imported", "bundle": "x"}))
    _write(refs / "ops" / "bundle.json", {"name": "ops"})
    _write(refs / "ops" / "runbook.json", _ref("runbook", links=[
        {"to": "adr-1", "rel": "depends-on"}, {"to": "gone", "rel": "elaborates"}]))
    _write(refs / "scratch" / "stray.json", _ref("stray"))  # no manifest: not a bundle
    return ws


def _file(ws: Path, rel: str) -> dict:
    return json.loads((ws / rel).read_text(encoding="utf-8"))


class TestTheListing:
    def test_rows_carry_their_bundle_and_links_both_ways(self, workspace: Path) -> None:
        rows = {r["id"]: r for r in list_catalog(str(workspace)).references}
        assert set(rows) == {"adr-1", "adr-2", "runbook"}
        assert rows["runbook"]["bundle"] == "ops"
        assert rows["adr-1"]["bundle"] == ""
        assert rows["adr-1"]["linked_from"] == [{"from": "runbook", "rel": "depends-on"}]

    def test_a_dangling_edge_is_marked_not_dropped(self, workspace: Path) -> None:
        [runbook] = [r for r in list_catalog(str(workspace)).references
                     if r["id"] == "runbook"]
        assert {"to": "gone", "rel": "elaborates", "dangling": True} in runbook["links"]

    def test_a_directory_without_a_manifest_is_not_the_catalog(self, workspace: Path) -> None:
        ids = {r["id"] for r in list_catalog(str(workspace)).references}
        assert "stray" not in ids

    def test_it_agrees_with_the_loader(self, workspace: Path) -> None:
        refs = workspace / ".jaato" / "references"
        loaded = {s.id for s in discover_references(str(refs), base_path=str(workspace))}
        loaded |= {s.id for s in discover_references(str(refs / "ops"),
                                                     base_path=str(workspace))}
        assert loaded == {r["id"] for r in list_catalog(str(workspace)).references}

    def test_a_linked_file_is_not_followed(self, workspace: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.json"
        _write(outside, _ref("planted"))
        os.symlink(outside, workspace / ".jaato" / "references" / "planted.json")
        listing = list_catalog(str(workspace))
        assert "planted" not in {r["id"] for r in listing.references}
        assert ".jaato/references/planted.json" in listing.unreadable


class TestOnlyTheOwnerEdits:
    def test_the_owner_may(self, workspace: Path) -> None:
        out = update_links(str(workspace), "adr-2", [{"to": "adr-1", "rel": "supersedes"}],
                           owner="acme:alice", user_id="acme:alice")
        assert out.ok, out.error

    def test_another_user_is_refused(self, workspace: Path) -> None:
        before = (workspace / ".jaato/references/adr-2.json").read_bytes()
        out = update_links(str(workspace), "adr-2", [{"to": "adr-1", "rel": "supersedes"}],
                           owner="acme:alice", user_id="acme:bob")
        assert (out.ok, out.category) == (False, "not_owner")
        assert (workspace / ".jaato/references/adr-2.json").read_bytes() == before

    def test_an_unowned_workspace_is_anyones(self, workspace: Path) -> None:
        assert update_links(str(workspace), "adr-2", [], owner=None, user_id=None).ok


class TestTheLinksAreValidated:
    def test_an_unknown_rel_is_refused(self, workspace: Path) -> None:
        before = (workspace / ".jaato/references/adr-2.json").read_bytes()
        out = update_links(str(workspace), "adr-2", [{"to": "adr-1", "rel": "replaces"}],
                           owner=None, user_id=None)
        assert (out.ok, out.category) == (False, "invalid_links")
        assert (workspace / ".jaato/references/adr-2.json").read_bytes() == before

    def test_an_edge_to_itself_is_refused(self, workspace: Path) -> None:
        out = update_links(str(workspace), "adr-2", [{"to": "adr-2", "rel": "elaborates"}],
                           owner=None, user_id=None)
        assert out.category == "invalid_links"

    def test_a_dangling_target_warns_and_is_kept(self, workspace: Path) -> None:
        out = update_links(str(workspace), "adr-2", [{"to": "later", "rel": "elaborates"}],
                           owner=None, user_id=None)
        assert out.ok and any("later" in w for w in out.warnings)
        assert _file(workspace, ".jaato/references/adr-2.json")["links"] == [
            {"to": "later", "rel": "elaborates"}]

    def test_a_second_successor_warns(self, workspace: Path) -> None:
        update_links(str(workspace), "adr-2", [{"to": "adr-1", "rel": "supersedes"}],
                     owner=None, user_id=None)
        out = update_links(str(workspace), "runbook", [{"to": "adr-1", "rel": "supersedes"}],
                           owner=None, user_id=None)
        assert out.ok and any("also supersedes" in w for w in out.warnings)

    def test_a_non_list_is_a_usage_error(self, workspace: Path) -> None:
        out = update_links(str(workspace), "adr-2", {"to": "adr-1"}, owner=None, user_id=None)
        assert out.category == "invalid_request"


class TestOnlyTheLinksChange:
    def test_the_rest_of_the_file_is_kept(self, workspace: Path) -> None:
        before = _file(workspace, ".jaato/references/adr-2.json")
        out = update_links(str(workspace), "adr-2",
                           [{"to": "adr-1", "rel": "supersedes", "note": "ADR-2 replaces it"}],
                           owner=None, user_id=None)
        after = _file(workspace, ".jaato/references/adr-2.json")
        assert out.reference_file == ".jaato/references/adr-2.json"
        assert after.pop("links") == [
            {"to": "adr-1", "rel": "supersedes", "note": "ADR-2 replaces it"}]
        assert after == before

    def test_an_empty_list_removes_the_key(self, workspace: Path) -> None:
        assert update_links(str(workspace), "runbook", [], owner=None, user_id=None).ok
        assert "links" not in _file(workspace, ".jaato/references/ops/runbook.json")

    def test_a_sub_bundle_reference_is_edited_in_place(self, workspace: Path) -> None:
        out = update_links(str(workspace), "runbook", [{"to": "adr-2", "rel": "depends-on"}],
                           owner=None, user_id=None)
        assert out.reference_file == ".jaato/references/ops/runbook.json"


class TestAnAmbiguousIdIsNotEdited:
    def test_a_duplicate_id_is_refused(self, workspace: Path) -> None:
        _write(workspace / ".jaato/references/ops/adr-1.json", _ref("adr-1"))
        out = update_links(str(workspace), "adr-1", [{"to": "adr-2", "rel": "elaborates"}],
                           owner=None, user_id=None)
        assert (out.ok, out.category) == (False, "ambiguous")
        assert "links" not in _file(workspace, ".jaato/references/adr-1.json")

    def test_the_listing_marks_it(self, workspace: Path) -> None:
        _write(workspace / ".jaato/references/ops/adr-1.json", _ref("adr-1"))
        rows = [r for r in list_catalog(str(workspace)).references if r["id"] == "adr-1"]
        assert len(rows) == 2 and all(r.get("duplicate_id") for r in rows)

    def test_an_unknown_id_is_not_found(self, workspace: Path) -> None:
        out = update_links(str(workspace), "nope", [], owner=None, user_id=None)
        assert out.category == "not_found"


class _Sink:
    def __init__(self, user: Optional[str]) -> None:
        self.user = user
        self.sent: List[Any] = []

    def get_client_user(self, client_id: str) -> Optional[str]:
        return self.user

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        return None

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


def _router(ws: Optional[Path], *, owner: Optional[str], user: Optional[str]) -> CommandRouter:
    router = object.__new__(CommandRouter)
    router._event_sink = _Sink(user)
    router._session_manager = _Manager(ws, owner)
    return router


class TestTheRouter:
    def test_a_listing_is_answered_by_its_request_id(self, workspace: Path) -> None:
        router = _router(workspace, owner="acme:alice", user="acme:alice")
        router._dispatch("c1", "", ReferenceCatalogRequest(request_id="rk-1"))
        [event] = router._event_sink.sent
        assert isinstance(event, ReferenceCatalogEvent)
        assert (event.request_id, event.ok, event.may_curate) == ("rk-1", True, True)
        assert len(event.references) == 3

    def test_the_owner_reaches_the_gate(self, workspace: Path) -> None:
        router = _router(workspace, owner="acme:alice", user="acme:bob")
        router._dispatch("c1", "", ReferenceLinksUpdateRequest(
            request_id="rk-2", reference_id="adr-2",
            links=[{"to": "adr-1", "rel": "supersedes"}]))
        [event] = router._event_sink.sent
        assert isinstance(event, ReferenceLinksUpdateResultEvent)
        assert (event.request_id, event.ok, event.category) == ("rk-2", False, "not_owner")

    def test_an_update_is_answered_with_what_was_written(self, workspace: Path) -> None:
        router = _router(workspace, owner=None, user=None)
        router._dispatch("c1", "", ReferenceLinksUpdateRequest(
            request_id="rk-3", reference_id="adr-2",
            links=[{"to": "adr-1", "rel": "supersedes"}]))
        [event] = router._event_sink.sent
        assert (event.ok, event.links) == (True, [{"to": "adr-1", "rel": "supersedes"}])

    def test_no_workspace_is_said(self) -> None:
        router = _router(None, owner=None, user=None)
        router._dispatch("c1", "", ReferenceLinksUpdateRequest(request_id="rk", reference_id="x"))
        [event] = router._event_sink.sent
        assert (event.ok, event.category) == (False, "no_workspace")


def test_the_requests_cross_the_wire() -> None:
    for event in (ReferenceCatalogRequest(request_id="x"),
                  ReferenceLinksUpdateRequest(request_id="y", reference_id="r",
                                              links=[{"to": "a", "rel": "elaborates"}])):
        assert deserialize_event(serialize_event(event)) == event
