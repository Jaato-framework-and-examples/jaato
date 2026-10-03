"""A claim revises a reference already in the catalog, and the daemon replaces it (#1437).

``proposeReference`` with ``revises: <id>`` writes a REVISION claim: the new
version of a catalog reference, plus the sha256 of the catalog file it was
written against.  ``reference.promote`` on it replaces that file in place.

Properties, each a way it could go wrong:

1. **Replaced in place, origin kept.**  The name, description, tags and
   document change; the id, ``origin`` (where the reference arrived),
   ``mode`` and every other key stay as the file has them.  A
   ``revisions`` record is appended, carrying the daemon's own
   ``curated_by``.
2. **A stale revision changes nothing.**  When the file's bytes no longer
   have the recorded digest -- another revision promoted first, an edge
   edited, a hand edit -- the promotion is refused ``stale``.
3. **The id cannot change, and neither can the origin.**  A claim whose
   entry id was edited away from what it revises is not a claim the
   listing can show; an ``origin`` in the claim's entry is not written.
4. **Written where the reference lives**, and that bundle's index is
   reconciled, because the embedding text may have changed.
5. **The owner gate** is promotion's.
6. **The curator sees a diff**: a revision row carries ``revises``, the
   current fields and ``stale``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from jaato_server.server.reference_curation import curate_claim, list_claims
from jaato_server.shared.plugins.bundle_common.bundle import write_bundle_manifest
from jaato_server.shared.plugins.references.bundle import EMBEDDING_CONFIG_FILENAME
from jaato_server.shared.plugins.references.claims import (
    build_revision,
    new_claim,
    revision_record,
    revision_target,
    write_claim,
)
from jaato_server.shared.tests.reversion import Reversion

_CURATION = "jaato-server/jaato_server/server/reference_curation.py"
_CLAIMS = "jaato-server/jaato_server/shared/plugins/references/claims.py"

REVERSIONS = [
    Reversion(
        target=_CURATION,
        find="    if stale:\n        return _fail(outcome, \"stale\", reason)\n",
        replace="",
        because="a revision written against an older version would undo what changed it",
        test="TestAStaleRevision::test_a_changed_file_is_refused_and_left_as_it_is",
    ),
    Reversion(
        target=_CURATION,
        find="    out = {k: v for k, v in current.items() if k not in _DOCUMENT_KEYS}\n",
        replace="    out = {k: v for k, v in current.items() if k not in _DOCUMENT_KEYS + (\"origin\",)}\n",
        because="a revision would erase where the reference arrived from",
        test="TestReplacedInPlace::test_origin_mode_and_other_keys_are_kept",
    ),
    Reversion(
        target=_CURATION,
        find="    out[\"revisions\"] = (list(prior) if isinstance(prior, list) else []) + [stamp]\n",
        replace="",
        because="nothing would record who revised the reference, from which claim",
        test="TestReplacedInPlace::test_a_revision_record_is_appended",
    ),
    Reversion(
        target=_CLAIMS,
        find="    if args.get(\"id\") not in (None, \"\", ref_id):\n",
        replace="    if False:\n",
        because="a revision could rename a reference, which is a new one plus supersedes",
        test="TestIdAndOriginCannotChange::test_build_revision_refuses_another_id",
    ),
    Reversion(
        target=_CLAIMS,
        find="    return (isinstance(revises, dict) and revises.get(\"id\") == ref[\"id\"]\n",
        replace="    return (isinstance(revises, dict)\n",
        because="a claim file edited to revise one id with another's entry would be shown as a claim",
        test="TestIdAndOriginCannotChange::test_an_edited_id_is_not_a_claim",
    ),
    Reversion(
        target=_CURATION,
        find="    outcome.bundle = target[\"bundle\"]\n",
        replace="",
        because="the revised bundle's index would not be reconciled",
        test="TestWhereItLives::test_the_indexed_bundle_is_reconciled",
    ),
    Reversion(
        target=_CURATION,
        find="    if is_revision(claim):\n        row.update(_revision_fields(claim, root))\n",
        replace="",
        because="the curator would see a revision as a new page, with no diff and no stale mark",
        test="TestTheCuratorsView::test_a_revision_row_carries_the_current_entry",
    ),
    Reversion(
        target=_CURATION,
        find="    if not may_curate(owner, user_id):\n",
        replace="    if False:\n",
        because="anyone could replace a reference in an owned workspace",
        test="test_a_non_owner_is_refused",
    ),
]


def _refs(ws: Path) -> Path:
    return ws / ".jaato" / "references"


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "deploy.md").write_text("# Deploy v1\n")
    (root / "docs" / "deploy-v2.md").write_text("# Deploy v2\n")
    _refs(root).mkdir(parents=True)
    return root


def _catalog(directory: Path, ref_id: str = "runbook", **extra: Any) -> Path:
    data = {"id": ref_id, "name": "Runbook", "description": "how to deploy, old",
            "type": "inline", "mode": "auto", "tags": ["ops"], "content": "step 1\n",
            "origin": {"kind": "imported", "bundle": "teammate", "source_id": "rb",
                       "at": "2026-09-01T00:00:00+00:00"},
            "fetchHint": "read before a release", **extra}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ref_id}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _revision(ws: Path, ref_id: str = "runbook", **args: Any) -> Dict[str, Any]:
    full = {"revises": ref_id, "name": "Runbook", "description": "how to deploy, current",
            "tags": ["ops", "deploy"], "content": "step 1\nstep 2\n", **args}
    entry, errors = build_revision(full, workspace=str(ws), catalog_ids=[ref_id])
    assert entry is not None, errors
    target, _c, error = revision_target(str(ws), ref_id)
    assert target is not None, error
    claim = new_claim(entry, None)
    claim["revises"] = revision_record(target)
    write_claim(str(ws), claim)
    return claim


def _promote(ws: Path, claim: Dict[str, Any], *, bundle: str = "", embed=None,
             owner=None, user_id=None):
    return curate_claim(str(ws), "promote", claim["claim_id"], owner=owner, user_id=user_id,
                        creator_in_workspace=lambda _s, _w: None, bundle=bundle, embed=embed)


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class TestReplacedInPlace:
    def test_the_fields_are_replaced(self, ws):
        path = _catalog(_refs(ws))
        outcome = _promote(ws, _revision(ws))
        assert outcome.ok, outcome.error
        assert (outcome.revised, outcome.reference_file) == (True, ".jaato/references/runbook.json")
        data = _read(path)
        assert data["description"] == "how to deploy, current"
        assert data["tags"] == ["ops", "deploy"]
        assert data["content"] == "step 1\nstep 2\n"

    def test_origin_mode_and_other_keys_are_kept(self, ws):
        path = _catalog(_refs(ws))
        before = _read(path)
        assert _promote(ws, _revision(ws)).ok
        data = _read(path)
        assert data["origin"] == before["origin"]
        assert (data["id"], data["mode"], data["fetchHint"]) == ("runbook", "auto",
                                                                 "read before a release")

    def test_a_revision_record_is_appended(self, ws):
        path = _catalog(_refs(ws))
        first = _revision(ws)
        assert _promote(ws, first, user_id="app:alice").ok
        second = _revision(ws, description="again")
        assert _promote(ws, second, user_id="app:alice").ok
        records = _read(path)["revisions"]
        assert [r["claim_id"] for r in records] == [first["claim_id"], second["claim_id"]]
        assert records[0]["curated_by"] == {"kind": "human", "via": "reference.promote",
                                            "user": "app:alice"}
        assert "kind" not in records[0]

    def test_a_document_becomes_a_path_reanchored_to_the_file(self, ws):
        path = _catalog(_refs(ws) / "ops", type="local", path="../../../docs/deploy.md",
                        content=None)
        write_bundle_manifest(_refs(ws) / "ops", name="ops")
        claim = _revision(ws, content=None, path="docs/deploy-v2.md")
        assert _promote(ws, claim).ok
        data = _read(path)
        assert data["type"] == "local" and "content" not in data
        assert (path.parent / data["path"]).resolve() == (ws / "docs" / "deploy-v2.md").resolve()

    def test_links_are_kept_unless_given(self, ws):
        path = _catalog(_refs(ws), links=[{"to": "glossary", "rel": "see-also"}])
        assert _promote(ws, _revision(ws)).ok
        assert _read(path)["links"] == [{"to": "glossary", "rel": "see-also"}]
        assert _promote(ws, _revision(ws, links=[])).ok
        assert "links" not in _read(path)

    def test_the_claim_is_removed(self, ws):
        _catalog(_refs(ws))
        claim = _revision(ws)
        assert _promote(ws, claim).ok
        assert list_claims(str(ws)).claims == []


class TestAStaleRevision:
    def test_a_changed_file_is_refused_and_left_as_it_is(self, ws):
        path = _catalog(_refs(ws))
        claim = _revision(ws)
        path.write_text(path.read_text().replace("old", "edited by hand"))
        edited = path.read_bytes()
        outcome = _promote(ws, claim)
        assert (outcome.ok, outcome.category) == (False, "stale")
        assert path.read_bytes() == edited
        assert [c["claim_id"] for c in list_claims(str(ws)).claims] == [claim["claim_id"]]

    def test_of_two_revisions_the_first_promoted_wins(self, ws):
        path = _catalog(_refs(ws))
        a = _revision(ws, description="A")
        b = _revision(ws, description="B")
        assert _promote(ws, b).ok
        outcome = _promote(ws, a)
        assert outcome.category == "stale"
        assert _read(path)["description"] == "B"

    def test_a_reference_that_is_gone_is_stale(self, ws):
        path = _catalog(_refs(ws))
        claim = _revision(ws)
        path.unlink()
        assert _promote(ws, claim).category == "stale"


class TestIdAndOriginCannotChange:
    def test_build_revision_refuses_another_id(self, ws):
        _catalog(_refs(ws))
        entry, errors = build_revision({"revises": "runbook", "id": "runbook-2",
                                        "name": "x", "content": "y"},
                                       workspace=str(ws), catalog_ids=["runbook"])
        assert entry is None and "cannot change the id" in errors[0]

    def test_build_revision_refuses_an_origin(self, ws):
        entry, errors = build_revision({"revises": "runbook", "name": "x", "content": "y",
                                        "origin": {"kind": "imported"}},
                                       workspace=str(ws), catalog_ids=["runbook"])
        assert entry is None and "origin" in errors[0]

    def test_an_edited_id_is_not_a_claim(self, ws):
        _catalog(_refs(ws))
        claim = _revision(ws)
        claim_file = ws / ".jaato" / "references-claims" / f"{claim['claim_id']}.json"
        data = json.loads(claim_file.read_text())
        data["reference"]["id"] = "something-else"
        claim_file.write_text(json.dumps(data))
        listing = list_claims(str(ws))
        assert listing.claims == [] and listing.unreadable == [claim_file.name]
        assert _promote(ws, claim).category == "invalid_claim"

    def test_an_origin_planted_in_the_claim_is_not_written(self, ws):
        path = _catalog(_refs(ws))
        before = _read(path)["origin"]
        claim = _revision(ws)
        claim_file = ws / ".jaato" / "references-claims" / f"{claim['claim_id']}.json"
        data = json.loads(claim_file.read_text())
        data["reference"]["origin"] = {"kind": "local", "at": "x"}
        claim_file.write_text(json.dumps(data))
        assert _promote(ws, claim).ok
        assert _read(path)["origin"] == before


MODEL, DIM = "mock-embed", 4


def _session(texts: List[str]) -> Dict[str, Any]:
    return {"ok": True, "model": MODEL, "dimensions": DIM,
            "vectors": [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]}


def _indexed(ws: Path, name: str = "ops") -> Path:
    directory = _refs(ws) / name
    directory.mkdir(parents=True, exist_ok=True)
    write_bundle_manifest(directory, name=name)
    (directory / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": MODEL, "embedding_dimensions": DIM,
        "embedding_sidecar": "references.embeddings.npy", "rows": []}))
    return directory


class TestWhereItLives:
    def test_the_indexed_bundle_is_reconciled(self, ws):
        _catalog(_indexed(ws))
        outcome = _promote(ws, _revision(ws), embed=None)
        assert outcome.ok and outcome.bundle == "ops"
        assert outcome.reconcile == "unavailable", "the ops index, not the root's (none)"

    def test_an_unindexed_bundle_reports_none(self, ws):
        _catalog(_refs(ws))
        assert _promote(ws, _revision(ws)).reconcile == "none"

    def test_the_index_holds_the_new_embedding(self, ws):
        pytest.importorskip("numpy")
        bundle = _indexed(ws)
        path = _catalog(bundle)
        texts: List[List[str]] = []

        def embed(batch):
            texts.append(list(batch))
            return _session(batch)

        outcome = _promote(ws, _revision(ws), embed=embed)
        assert (outcome.ok, outcome.reconcile) == (True, "updated"), outcome.reconcile_detail
        assert any("how to deploy, current" in t for batch in texts for t in batch)
        config = json.loads((bundle / EMBEDDING_CONFIG_FILENAME).read_text())
        assert config["rows"] == ["runbook"]
        assert "source_hash" in _read(path)["embedding"]

    def test_another_bundle_is_refused(self, ws):
        _catalog(_refs(ws))
        _indexed(ws, "elsewhere")
        claim = _revision(ws)
        outcome = _promote(ws, claim, bundle="elsewhere")
        assert outcome.category == "invalid_request"
        assert not (_refs(ws) / "elsewhere" / "runbook.json").exists()

    def test_an_id_in_two_files_is_ambiguous(self, ws):
        _catalog(_refs(ws))
        claim = _revision(ws)
        sub = _refs(ws) / "ops"
        _catalog(sub)
        write_bundle_manifest(sub, name="ops")
        assert _promote(ws, claim).category == "ambiguous"


class TestTheCuratorsView:
    def test_a_revision_row_carries_the_current_entry(self, ws):
        _catalog(_refs(ws))
        claim = _revision(ws)
        row = list_claims(str(ws)).claims[0]
        assert row["claim_id"] == claim["claim_id"]
        assert (row["revises"], row["stale"], row["problems"]) == ("runbook", False, [])
        assert row["current"]["description"] == "how to deploy, old"
        assert row["current"]["content"] == "step 1\n"
        assert row["revises_file"] == ".jaato/references/runbook.json"
        assert row["links_replaced"] is False

    def test_a_stale_row_says_why(self, ws):
        path = _catalog(_refs(ws))
        _revision(ws)
        path.write_text(path.read_text().replace("old", "new"))
        row = list_claims(str(ws)).claims[0]
        assert row["stale"] is True and "changed since" in row["stale_reason"]

    def test_the_recorded_digest_is_the_files(self, ws):
        path = _catalog(_refs(ws))
        claim = _revision(ws)
        assert claim["revises"] == {"id": "runbook", "file": ".jaato/references/runbook.json",
                                    "digest": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_a_non_owner_is_refused(ws):
    path = _catalog(_refs(ws))
    before = path.read_bytes()
    outcome = _promote(ws, _revision(ws), owner="app:alice", user_id="app:bob")
    assert outcome.category == "not_owner"
    assert path.read_bytes() == before
