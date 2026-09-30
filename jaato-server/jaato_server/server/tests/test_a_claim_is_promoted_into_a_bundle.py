"""A person promotes a reference claim into a named bundle, and its index is reconciled.

``reference.promote <claim_id> --bundle <name>`` (and the correlated
``ReferenceCurationRequest.bundle``) writes the entry into a workspace-tier
sub-bundle instead of the catalog root.  When the destination bundle has a
vector index, the new entry has no row in it, so the index is reconciled.

Since #1422 the RUNNER does both: the daemon gates, re-validates and stamps
(``reference_curation``), then hands the bytes to ``session.write_reference``,
which writes in the runner's base profile (template v44 lets base write the
catalog) and reconciles with the references plugin's own provider
(``reference_catalog_write``).  With no session in the workspace the daemon
writes itself and an indexed bundle is ``unavailable``.

Properties, each a way it could go wrong:

1. **The entry lands in the bundle**, with a local ``path`` re-anchored to
   the bundle directory so the catalog loader still finds the document.
2. **An unknown bundle is refused** before anything is written, and the
   claim is left for another try; a name that is not one path component is
   a usage error.
3. **Vectors from another model never enter an index**: a provider whose
   model differs from the index's is ``unavailable``, reported.
4. **The outcome is reported, and the reference is placed either way**:
   ``none`` for a bundle with no index, ``unavailable`` with no provider.
5. **The typed command takes ``--bundle`` and nothing else** after the id.
6. **The listing offers the sub-bundles**, marking the indexed ones.
7. **The runner writes only its own workspace's catalog.**

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
from jaato_server.server.reference_catalog_write import write_catalog_file
from jaato_server.server.reference_curation import (
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
_WRITE = "jaato-server/jaato_server/server/reference_catalog_write.py"
_ROUTER = "jaato-server/jaato_server/server/command_router.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"

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
        find="    if outcome.bundle and destination_bundle(root, outcome.bundle) is None:\n",
        replace="    if False:\n",
        because="a mistyped bundle would be created as a directory nobody declared",
        test="TestIntoTheBundle::test_an_unknown_bundle_is_refused_and_the_claim_kept",
    ),
    Reversion(
        target=_WRITE,
        find="    if provider.model_name != dest.embedding_model:\n",
        replace="    if False:\n",
        because="vectors from another model would be written into the index",
        test="TestTheIndex::test_a_different_model_is_unavailable_and_reported",
    ),
    Reversion(
        target=_CURATION,
        find="    outcome.reconcile = written.get(\"reconcile\") or \"none\"\n",
        replace="    outcome.reconcile = \"none\"\n",
        because="an index left without the new row would be reported as having none",
        test="TestTheIndex::test_no_session_to_embed_with_is_reported_and_the_entry_placed",
    ),
    Reversion(
        target=_RPC,
        find="        if not root or os.path.realpath(str(args.get(\"workspace\") or \"\")) != root:\n",
        replace="        if not root:\n",
        because="a runner would write another workspace's catalog",
        test="TestTheRunnerHalf::test_another_workspace_is_refused",
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


class _Provider:
    """The references plugin's embedding provider, as the runner holds it."""

    dimensions = DIM

    def __init__(self, model: str = MODEL, vectors: bool = True) -> None:
        self.model_name, self.vectors, self.available = model, vectors, True
        self.calls: List[List[str]] = []

    def load_model(self) -> bool:
        return True

    def embed_batch(self, texts):
        from jaato_server.shared.plugins.references.embedding_types import EmbeddingResult
        self.calls.append(list(texts))
        return [EmbeddingResult(embedding=[float(len(t)), 1.0, 0.0, 0.0], model=self.model_name,
                                dimensions=DIM) if self.vectors else None for t in texts]

    def embed_text(self, text):
        return self.embed_batch([text])[0]


def _runner_writer(ws: Path, provider):
    """What ``session.write_reference`` does in the runner, minus the wire."""
    return lambda args: write_catalog_file(str(ws), provider=provider, **args)


def _promote(ws: Path, claim: Dict[str, Any], bundle: str = "", provider=None, runner=True):
    write = _runner_writer(ws, provider) if runner else None
    return curate_claim(str(ws), "promote", claim["claim_id"], owner=None, user_id=None,
                        creator_in_workspace=lambda _s, _w: None, bundle=bundle, write=write)


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

    def test_a_failed_write_keeps_the_claim(self, ws):
        claim = _claim(ws)
        outcome = curate_claim(
            str(ws), "promote", claim["claim_id"], owner=None, user_id=None,
            creator_in_workspace=lambda _s, _w: None,
            write=lambda _a: {"ok": False, "category": "runner_unreachable",
                              "error": "no answer"})
        assert (outcome.ok, outcome.category) == (False, "runner_unreachable")
        assert [r["claim_id"] for r in list_claims(str(ws)).claims] == [claim["claim_id"]]


class TestTheIndex:
    def test_no_session_to_embed_with_is_reported_and_the_entry_placed(self, ws):
        _bundle(ws, "ops", indexed=True)
        outcome = _promote(ws, _claim(ws), "ops", runner=False)
        assert outcome.ok
        assert outcome.reconcile == "unavailable"
        assert "no session" in outcome.reconcile_detail
        assert any("vector index was not updated" in w for w in outcome.warnings)

    def test_a_session_without_a_provider_is_unavailable(self, ws):
        _bundle(ws, "ops", indexed=True)
        outcome = _promote(ws, _claim(ws), "ops", provider=None)
        assert outcome.ok and outcome.reconcile == "unavailable"
        assert "no embedding provider" in outcome.reconcile_detail

    def test_a_different_model_is_unavailable_and_reported(self, ws):
        _bundle(ws, "ops", indexed=True)
        provider = _Provider(model="other-model")
        outcome = _promote(ws, _claim(ws), "ops", provider=provider)
        assert outcome.ok and outcome.reconcile == "unavailable"
        assert "other-model" in outcome.reconcile_detail
        assert provider.calls == [], "nothing embedded"

    def test_the_index_gains_the_row(self, ws):
        np = pytest.importorskip("numpy")
        bundle = _bundle(ws, "ops", indexed=True)
        outcome = _promote(ws, _claim(ws), "ops", provider=_Provider())
        assert (outcome.ok, outcome.reconcile) == (True, "updated"), outcome.reconcile_detail
        config = json.loads((bundle / EMBEDDING_CONFIG_FILENAME).read_text())
        assert config["rows"] == ["runbook"]
        assert np.load(bundle / "references.embeddings.npy").shape == (1, DIM)
        assert "source_hash" in json.loads((bundle / "runbook.json").read_text())["embedding"]

    def test_a_skipped_row_is_an_error_not_updated(self, ws):
        pytest.importorskip("numpy")
        _bundle(ws, "ops", indexed=True)
        outcome = _promote(ws, _claim(ws), "ops", provider=_Provider(vectors=False))
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


class TestTheRunnerHalf:
    """``session.write_reference`` -- the runner writes its own catalog (#1422)."""

    def _rpc(self, ws: Path, provider=None):
        import socket
        from types import SimpleNamespace

        from jaato_server.server.runner.rpc import RunnerRPC
        from jaato_server.server.runner.session import RunnerSessionHost
        from jaato_server.shared.session_envelope import SessionInitEnvelope

        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        b.close()
        rpc = RunnerRPC(a, lambda _n, _a: (False, {}))
        plugin = SimpleNamespace(reconcile_provider=lambda: provider)
        registry = SimpleNamespace(get_plugin=lambda n: plugin if n == "references" else None)
        session = SimpleNamespace(_runtime=SimpleNamespace(registry=registry))
        rpc._session_host = RunnerSessionHost(
            envelope=SessionInitEnvelope(session_id="s", workspace_path=str(ws),
                                         profile_name="", provider_name="p",
                                         model_name="m", plugins=[]),
            runtime=None, session=session)
        return rpc

    def _args(self, ws: Path, **over):
        args = {"workspace": str(ws), "rel_file": ".jaato/references/r.json",
                "data": '{"id": "r"}\n', "replace": False, "bundle": "",
                "ref_id": "r", "reconcile": True}
        args.update(over)
        return args

    def test_the_runner_writes_and_reconciles_with_its_own_provider(self, ws):
        pytest.importorskip("numpy")
        _bundle(ws, "ops", indexed=True)
        provider = _Provider()
        ok, answer = self._rpc(ws, provider)._handle_session_write_reference(self._args(
            ws, rel_file=".jaato/references/ops/r.json", bundle="ops",
            data=json.dumps({"id": "r", "name": "R", "description": "d",
                             "type": "inline", "content": "x"})))
        assert ok and answer["ok"], answer
        assert answer["reconcile"] == "updated", answer
        assert provider.calls

    def test_another_workspace_is_refused(self, ws, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        ok, answer = self._rpc(ws)._handle_session_write_reference(
            self._args(ws, workspace=str(other)))
        assert ok and answer["category"] == "wrong_workspace"
        assert not (ws / ".jaato" / "references" / "r.json").exists()

    def test_only_the_catalog_is_writable_and_existing_files_collide(self, ws):
        rpc = self._rpc(ws)
        _ok, outside = rpc._handle_session_write_reference(
            self._args(ws, rel_file=".jaato/profiles/p.yaml"))
        assert outside["category"] == "invalid_request"
        _ok, first = rpc._handle_session_write_reference(self._args(ws))
        _ok, again = rpc._handle_session_write_reference(self._args(ws))
        assert first["ok"] and again["category"] == "collision"
