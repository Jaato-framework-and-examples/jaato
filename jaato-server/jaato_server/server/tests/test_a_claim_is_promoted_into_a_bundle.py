"""A person promotes a reference claim into a named bundle, and its index is reconciled.

``reference.promote <claim_id> --bundle <name>`` (and the correlated
``ReferenceCurationRequest.bundle``) writes the entry into a workspace-tier
sub-bundle instead of the catalog root.  When the destination bundle has a
vector index, the new entry has no row in it, so the DAEMON reconciles it:
on a confined host it is the only process that may write
``.jaato/references/**`` (every runner body denies it, the base profile
included).  The embedding model is in the runner, so the vectors come from
the caller's session (``session.embed_texts``) through
``RunnerEmbeddingProvider``.

Properties, each a way it could go wrong:

1. **The entry lands in the bundle**, with a local ``path`` re-anchored to
   the bundle directory so the catalog loader still finds the document.
2. **An unknown bundle is refused** before anything is written, and the
   claim is left for another try; a name that is not one path component is
   a usage error.
3. **Vectors from another model never enter an index**: a session whose
   model differs from the index's is ``unavailable``, reported.
4. **The outcome is reported, and the reference is placed either way**:
   ``none`` for a bundle with no index, ``unavailable`` with no session to
   embed with.
5. **The typed command takes ``--bundle`` and nothing else** after the id.
6. **The listing offers the sub-bundles**, marking the indexed ones.

The reconcile itself (``reconcile_bundle``, unchanged) needs numpy, which is
optional; the tests that drive it through to a written sidecar skip without
it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from jaato_server.server.command_router import _curation_args
from jaato_server.server.reference_curation import (
    RunnerEmbeddingProvider,
    curate_claim,
    list_claims,
)
from jaato_server.shared.plugins.bundle_common.bundle import write_bundle_manifest
from jaato_server.shared.plugins.references.bundle import EMBEDDING_CONFIG_FILENAME
from jaato_server.shared.plugins.references.claims import (
    build_proposed_reference,
    new_claim,
    write_claim,
)
from jaato_server.shared.plugins.references.config_loader import discover_references
from jaato_server.shared.tests.reversion import Reversion

_CURATION = "jaato-server/jaato_server/server/reference_curation.py"
_ROUTER = "jaato-server/jaato_server/server/command_router.py"

REVERSIONS = [
    Reversion(
        target=_CURATION,
        find="    dest_rel = destination_rel(outcome.bundle)\n",
        replace="    dest_rel = CATALOG_REL\n",
        because="the bundle a person chose would be ignored and the entry written to the root",
        test="TestIntoTheBundle::test_the_entry_lands_in_the_bundle_and_loads",
    ),
    Reversion(
        target=_CURATION,
        find="    if outcome.bundle and _destination_bundle(root, outcome.bundle) is None:\n",
        replace="    if False:\n",
        because="a mistyped bundle would be created as a directory nobody declared",
        test="TestIntoTheBundle::test_an_unknown_bundle_is_refused_and_the_claim_kept",
    ),
    Reversion(
        target=_CURATION,
        find="    if probe.get(\"model\") != dest.embedding_model:\n",
        replace="    if False:\n",
        because="vectors from another model would be written into the index",
        test="TestTheIndex::test_a_different_model_is_unavailable_and_reported",
    ),
    Reversion(
        target=_CURATION,
        find="    outcome.reconcile, outcome.reconcile_detail = reconcile_destination(\n"
             "        root, outcome.bundle, embed, ref_id)\n",
        replace="    outcome.reconcile, outcome.reconcile_detail = \"none\", \"\"\n",
        because="an index left without the new row would be reported as having none",
        test="TestTheIndex::test_no_session_to_embed_with_is_reported_and_the_entry_placed",
    ),
    Reversion(
        target=_ROUTER,
        find="    elif rest:\n        return \"\", \"\"\n",
        replace="",
        because="a typo after the claim id would be ignored and the claim promoted anyway",
        test="test_the_typed_command_takes_bundle_and_nothing_else",
    ),
]

MODEL = "mock-embed"
DIM = 4


def _bundle(ws: Path, name: str, *, indexed: bool, rows=()) -> Path:
    directory = ws / ".jaato" / "references" / name
    directory.mkdir(parents=True, exist_ok=True)
    write_bundle_manifest(directory, name=name)
    if indexed:
        (directory / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": MODEL, "embedding_dimensions": DIM,
            "embedding_sidecar": "references.embeddings.npy", "rows": list(rows),
        }))
    return directory


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "deploy.md").write_text("# Deploy\n")
    (root / ".jaato" / "references").mkdir(parents=True)
    return root


def _claim(ws: Path, ref_id: str = "runbook") -> Dict[str, Any]:
    entry, errors = build_proposed_reference(
        {"id": ref_id, "name": "Runbook", "description": "how to deploy", "path": "docs/deploy.md"},
        workspace=str(ws), catalog_ids=[],
    )
    assert entry is not None, errors
    claim = new_claim(entry, None)
    write_claim(str(ws), claim)
    return claim


def _promote(ws: Path, claim: Dict[str, Any], bundle: str = "", embed=None):
    return curate_claim(str(ws), "promote", claim["claim_id"], owner=None, user_id=None,
                        creator_in_workspace=lambda _s, _w: None, bundle=bundle, embed=embed)


class _Session:
    """A stand-in for ``JaatoServer.embed_texts``: one vector per text."""

    def __init__(self, model: str = MODEL, ok: bool = True) -> None:
        self.model, self.ok, self.calls = model, ok, []

    def __call__(self, texts: List[str]) -> Dict[str, Any]:
        self.calls.append(list(texts))
        if not self.ok:
            return {"ok": False, "category": "no_provider", "error": "no embedding provider"}
        return {"ok": True, "model": self.model, "dimensions": DIM,
                "vectors": [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]}


class TestIntoTheBundle:
    def test_the_entry_lands_in_the_bundle_and_loads(self, ws):
        bundle = _bundle(ws, "ops", indexed=False)
        outcome = _promote(ws, _claim(ws), "ops")
        assert outcome.ok, outcome.error
        assert outcome.reference_file == ".jaato/references/ops/runbook.json"
        assert (bundle / "runbook.json").is_file()
        assert not (ws / ".jaato" / "references" / "runbook.json").exists()
        assert outcome.reconcile == "none"
        loaded = discover_references(str(bundle), base_path=str(bundle.parent),
                                     project_root=str(ws))
        assert [s.resolved_path for s in loaded] == ["docs/deploy.md"]

    def test_an_unknown_bundle_is_refused_and_the_claim_kept(self, ws):
        claim = _claim(ws)
        outcome = _promote(ws, claim, "opps")
        assert (outcome.ok, outcome.category) == (False, "unknown_bundle")
        assert not (ws / ".jaato" / "references" / "opps").exists()
        assert [r["claim_id"] for r in list_claims(str(ws)).claims] == [claim["claim_id"]]

    @pytest.mark.parametrize("name", ["../x", "a/b", ".."])
    def test_a_bundle_name_that_is_not_one_component_is_a_usage_error(self, ws, name):
        outcome = _promote(ws, _claim(ws), name)
        assert outcome.category == "invalid_request"

    def test_the_root_still_works_and_reports_no_index(self, ws):
        outcome = _promote(ws, _claim(ws))
        assert outcome.ok and outcome.reference_file == ".jaato/references/runbook.json"
        assert (outcome.bundle, outcome.reconcile) == ("", "none")


class TestTheIndex:
    def test_no_session_to_embed_with_is_reported_and_the_entry_placed(self, ws):
        _bundle(ws, "ops", indexed=True)
        outcome = _promote(ws, _claim(ws), "ops", embed=None)
        assert outcome.ok
        assert outcome.reconcile == "unavailable"
        assert "no session" in outcome.reconcile_detail
        assert any("vector index was not updated" in w for w in outcome.warnings)

    def test_a_session_without_a_provider_is_unavailable(self, ws):
        _bundle(ws, "ops", indexed=True)
        outcome = _promote(ws, _claim(ws), "ops", embed=_Session(ok=False))
        assert outcome.ok and outcome.reconcile == "unavailable"
        assert "no embedding provider" in outcome.reconcile_detail

    def test_a_different_model_is_unavailable_and_reported(self, ws):
        _bundle(ws, "ops", indexed=True)
        session = _Session(model="other-model")
        outcome = _promote(ws, _claim(ws), "ops", embed=session)
        assert outcome.ok and outcome.reconcile == "unavailable"
        assert "other-model" in outcome.reconcile_detail
        assert session.calls == [[]], "only the model probe; nothing embedded"

    def test_the_adapter_refuses_an_answer_from_another_model(self):
        provider = RunnerEmbeddingProvider(_Session(model="other"), MODEL, DIM)
        with pytest.raises(RuntimeError):
            provider.embed_batch(["x"])

    def test_the_index_gains_the_row(self, ws):
        np = pytest.importorskip("numpy")
        bundle = _bundle(ws, "ops", indexed=True)
        session = _Session()
        outcome = _promote(ws, _claim(ws), "ops", embed=session)
        assert (outcome.ok, outcome.reconcile) == (True, "updated"), outcome.reconcile_detail
        config = json.loads((bundle / EMBEDDING_CONFIG_FILENAME).read_text())
        assert config["rows"] == ["runbook"]
        assert np.load(bundle / "references.embeddings.npy").shape == (1, DIM)
        assert "source_hash" in json.loads((bundle / "runbook.json").read_text())["embedding"]

    def test_a_skipped_row_is_an_error_not_updated(self, ws):
        pytest.importorskip("numpy")
        _bundle(ws, "ops", indexed=True)

        def no_vector(texts):
            answer = _Session()(texts)
            answer["vectors"] = [None for _ in texts]
            return answer

        outcome = _promote(ws, _claim(ws), "ops", embed=no_vector)
        assert outcome.ok and outcome.reconcile == "error"
        assert "runbook" in outcome.reconcile_detail


def test_the_typed_command_takes_bundle_and_nothing_else():
    assert _curation_args(["c1"]) == ("c1", "")
    assert _curation_args(["c1", "--bundle", "ops"]) == ("c1", "ops")
    assert _curation_args(["c1", "--bundle=ops"]) == ("c1", "ops")
    assert _curation_args(["c1", "--bundel", "ops"]) == ("", "")
    assert _curation_args(["c1", "ops"]) == ("", "")


def test_the_listing_offers_the_sub_bundles(ws):
    _bundle(ws, "ops", indexed=True)
    _bundle(ws, "notes", indexed=False)
    assert list_claims(str(ws)).bundles == [
        {"name": "notes", "indexed": False},
        {"name": "ops", "indexed": True, "model": MODEL},
    ]


class _Provider:
    model_name, dimensions, available = MODEL, DIM, True

    def load_model(self):
        return True

    def embed_batch(self, texts):
        from jaato_server.shared.plugins.references.embedding_types import EmbeddingResult
        return [EmbeddingResult(embedding=[1.0] * DIM, model=MODEL, dimensions=DIM) for _ in texts]


class TestTheRunnerHalf:
    """``ReferencesPlugin.embed_texts`` -- what ``session.embed_texts`` serves."""

    def _plugin(self, provider=None):
        from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
        plugin = ReferencesPlugin()
        plugin._embedding_provider = provider
        return plugin

    def test_vectors_and_the_model_come_back(self):
        answer = self._plugin(_Provider()).embed_texts(["a", "b"])
        assert answer == {"ok": True, "model": MODEL, "dimensions": DIM,
                          "vectors": [[1.0] * DIM, [1.0] * DIM]}

    def test_an_empty_probe_names_the_model_and_embeds_nothing(self):
        answer = self._plugin(_Provider()).embed_texts([])
        assert (answer["ok"], answer["model"], answer["vectors"]) == (True, MODEL, [])

    def test_no_provider_is_said(self):
        assert self._plugin(None).embed_texts(["a"])["category"] == "no_provider"

    @pytest.mark.parametrize("texts", ["a", [1], ["x" * (32 * 1024 + 1)], ["a"] * 257])
    def test_bad_input_is_refused(self, texts):
        assert self._plugin(_Provider()).embed_texts(texts)["category"] == "invalid"
