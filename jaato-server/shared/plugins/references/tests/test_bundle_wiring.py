"""Integration tests for bundle wiring on ``ReferencesPlugin``.

Covers:
    * Sub-bundle discovery at ``initialize()`` time (references in
      ``.jaato/references/<subdir>/`` are loaded with ``bundle_name`` set).
    * Cross-bundle ``_semantic_score_sources`` and ``_semantic_embed_and_match``
      against fake matcher stubs.
    * The ``references bundles`` and ``references reconcile`` subcommands.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

from shared.plugins.references.bundle import (
    EMBEDDING_CONFIG_FILENAME,
    ROOT_BUNDLE_NAME,
    Bundle,
    metadata_hash,
)
from shared.plugins.references.embedding_types import (
    EmbeddingResult,
    SemanticMatch,
)
from shared.plugins.references.models import (
    EmbeddingMetadata,
    InjectionMode,
    ReferenceSource,
    SourceType,
)
from shared.plugins.references.plugin import ReferencesPlugin


def _write_ref(directory: Path, sid: str, *, tags=None, hash_=None, mode="selectable"):
    """Drop a reference JSON file under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "id": sid,
        "name": sid.replace("-", " ").title(),
        "description": f"Description for {sid}",
        "type": "local",
        "path": f"./dummy/{sid}.md",
        "mode": mode,
    }
    if tags is not None:
        payload["tags"] = list(tags)
    if hash_ is not None:
        payload["embedding"] = {"source_hash": hash_}
    (directory / f"{sid}.json").write_text(json.dumps(payload, indent=2))


def _write_manifest(directory: Path, *, rows, model="mock-model", dim=4):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": model,
        "embedding_dimensions": dim,
        "embedding_sidecar": "references.embeddings.npy",
        "rows": list(rows),
    }))


class _FakeMatcher:
    """Fake ``SemanticMatcherProtocol`` for testing cross-bundle merge logic."""

    def __init__(self, model="mock-model"):
        self._model = model
        self.available = True
        self._by_id: Dict[str, float] = {}
        self._provider = None

    def validate_model(self, provider_model: str) -> bool:
        return provider_model == self._model

    def load_index(self, *, sidecar_path, embedding_model, embedding_dimensions,
                   index_to_source_id):
        self._ids = set(index_to_source_id.values())
        return True

    def set_provider(self, provider):
        self._provider = provider

    def score_sources(self, query_vec, source_ids: Set[str]) -> Dict[str, float]:
        return {sid: self._by_id.get(sid, 0.0) for sid in source_ids if sid in self._ids}

    def find_matches(self, query_vec, threshold: float, top_k: int,
                     exclude_ids: Optional[Set[str]] = None) -> List[SemanticMatch]:
        exclude_ids = exclude_ids or set()
        hits = [
            SemanticMatch(source_id=sid, score=score, embedding_index=i)
            for i, (sid, score) in enumerate(self._by_id.items())
            if sid in self._ids and sid not in exclude_ids and score >= threshold
        ]
        hits.sort(key=lambda m: m.score, reverse=True)
        return hits[:top_k]

    def embed_and_match(self, *, content, threshold, top_k, exclude_ids=None):
        return self.find_matches(None, threshold, top_k, exclude_ids)


class _FakeProvider:
    """Fake ``EmbeddingProviderProtocol`` for plugin wiring tests."""

    def __init__(self, *, model="mock-model"):
        self.model_name = model
        self.dimensions = 4
        self.available = True

    def load_model(self):
        self.available = True
        return True

    def embed_text(self, text):
        return EmbeddingResult(embedding=[0.0] * 4, model=self.model_name, dimensions=4)

    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]

    def embed_text_as_array(self, text):
        return [0.0] * 4


@pytest.fixture
def workspace(tmp_path):
    """A workspace with root + one sub-bundle, each containing one ref."""
    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"

    # Root bundle ref.
    _write_ref(refs, "root-a", hash_="PLACEHOLDER")
    _write_manifest(refs, rows=["root-a"])

    # Sub-bundle 'teammate' with its own ref.
    sub = refs / "teammate"
    _write_ref(sub, "teammate-x", hash_="PLACEHOLDER")
    _write_manifest(sub, rows=["teammate-x"])

    # Stamp each JSON with the hash the plugin will compute so reconcile
    # doesn't see drift at init.
    for ref_dir, sid in [(refs, "root-a"), (sub, "teammate-x")]:
        raw = json.loads((ref_dir / f"{sid}.json").read_text())
        source = ReferenceSource.from_dict(raw)
        raw["embedding"] = {"source_hash": metadata_hash(source)}
        (ref_dir / f"{sid}.json").write_text(json.dumps(raw, indent=2))

    return ws


class TestSubBundleDiscovery:
    """Sub-bundle refs are loaded with their bundle_name stamped."""

    def test_root_and_sub_bundle_sources_discovered(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})

        ids_by_bundle = {
            s.bundle_name: s.id for s in plugin._sources
        }
        assert ids_by_bundle == {"": "root-a", "teammate": "teammate-x"}
        assert [b.name for b in plugin._bundles] == [ROOT_BUNDLE_NAME, "teammate"]

    def test_sub_bundle_with_duplicate_id_is_skipped(self, tmp_path):
        """Cross-bundle id collision: the first one wins (root first by order)."""
        ws = tmp_path / "ws"
        refs = ws / ".jaato" / "references"
        _write_ref(refs, "dup", hash_="h")
        _write_manifest(refs, rows=["dup"])
        sub = refs / "other"
        _write_ref(sub, "dup", hash_="h")
        _write_manifest(sub, rows=["dup"])

        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(ws))
        plugin.initialize({"lookup_strategy": "tags_only"})

        ids = [s.id for s in plugin._sources]
        # Duplicate id is kept only once (the root bundle's version).
        assert ids.count("dup") == 1
        root_dup = next(s for s in plugin._sources if s.id == "dup")
        assert root_dup.bundle_name == ""


class TestCrossBundleSemantic:
    """The cross-bundle matching helpers merge per-bundle hits correctly."""

    def _wire_bundles(self, plugin, sub_matcher_scores=None, root_matcher_scores=None):
        """Attach fake matchers to the plugin's already-discovered bundles."""
        plugin._embedding_provider = _FakeProvider()
        for bundle in plugin._bundles:
            fake = _FakeMatcher()
            fake.load_index(
                sidecar_path=str(bundle.sidecar_path),
                embedding_model=bundle.embedding_model,
                embedding_dimensions=bundle.embedding_dimensions,
                index_to_source_id={
                    i: sid for i, sid in enumerate(bundle.embedding_rows)
                },
            )
            if bundle.name == ROOT_BUNDLE_NAME:
                fake._by_id = dict(root_matcher_scores or {})
            else:
                fake._by_id = dict(sub_matcher_scores or {})
            bundle.matcher = fake

    def test_score_sources_partitions_by_bundle(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})
        self._wire_bundles(
            plugin,
            root_matcher_scores={"root-a": 0.91},
            sub_matcher_scores={"teammate-x": 0.42},
        )

        scores = plugin._semantic_score_sources(
            query_vec=None, source_ids={"root-a", "teammate-x"},
        )

        assert scores == {"root-a": 0.91, "teammate-x": 0.42}

    def test_score_sources_skips_sources_not_in_any_bundle(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})
        self._wire_bundles(plugin, root_matcher_scores={"root-a": 0.5})

        scores = plugin._semantic_score_sources(
            query_vec=None, source_ids={"root-a", "orphan-id"},
        )

        assert "orphan-id" not in scores

    def test_embed_and_match_merges_and_truncates_global_top_k(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})
        self._wire_bundles(
            plugin,
            root_matcher_scores={"root-a": 0.90},
            sub_matcher_scores={"teammate-x": 0.95},
        )

        matches = plugin._semantic_embed_and_match(
            content="whatever", threshold=0.0, top_k=1,
        )

        assert len(matches) == 1
        # The teammate match scores higher so it wins global top-1 even
        # though root has its own bundle.
        assert matches[0].source_id == "teammate-x"

    def test_embed_and_match_skips_bundles_without_matcher(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})
        self._wire_bundles(
            plugin,
            root_matcher_scores={"root-a": 0.99},
            sub_matcher_scores={"teammate-x": 0.80},
        )
        # Detach the sub-bundle's matcher — simulating a model mismatch.
        for bundle in plugin._bundles:
            if bundle.name != ROOT_BUNDLE_NAME:
                bundle.matcher = None

        matches = plugin._semantic_embed_and_match(
            content="whatever", threshold=0.0, top_k=5,
        )

        assert [m.source_id for m in matches] == ["root-a"]


class TestBundleListSubcommand:
    """``bundle list`` shows every loaded bundle with its tier."""

    def test_bundle_list_lists_each_bundle(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_bundle_cmd(
            {"subcommand": "list", "target": ""}
        )

        # HelpLines object — flatten line text for inspection.
        text = "\n".join(line for line, _tag in result.lines)
        assert "(root)" in text
        assert "teammate" in text
        assert "mock-model" in text

    def test_bundle_list_with_no_bundles(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / ".jaato" / "references").mkdir(parents=True)
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(ws))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_bundle_cmd(
            {"subcommand": "list", "target": ""}
        )

        text = "\n".join(line for line, _tag in result.lines)
        assert "no bundles" in text.lower()


class TestBundleReconcileSubcommand:
    def test_reconcile_unknown_bundle_errors(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_bundle_cmd(
            {"subcommand": "reconcile", "target": "ghost-bundle"}
        )

        assert "error" in result
        assert "ghost-bundle" in result["error"]

    def test_reconcile_root_alias(self, workspace):
        """'bundle reconcile root' resolves to the root bundle."""
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_bundle_cmd(
            {"subcommand": "reconcile", "target": "root"}
        )

        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        assert result["results"][0]["bundle"] == "(root)"

    def test_reconcile_workspace_default_when_no_target(self, workspace):
        """No-arg reconcile defaults to workspace tier (matches the
        write-default-to-workspace contract)."""
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_bundle_cmd(
            {"subcommand": "reconcile", "target": ""}
        )

        assert result["status"] == "ok"
        bundles_touched = {r["bundle"] for r in result["results"]}
        assert bundles_touched == {"(root)", "teammate"}


class TestReferencesUnknownSubcommand:
    """The references command errors mention reference-level ops only;
    bundle-level ops are redirected to the bundle command."""

    def test_unknown_subcommand_points_at_references_surface(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_references_cmd(
            {"subcommand": "nonsense", "target": ""}
        )

        assert "error" in result
        assert "list" in result["error"]
        assert "bundle help" in result["error"]

    def test_old_bundle_subcommands_redirect(self, workspace):
        """``references bundles|reconcile|merge|pack|unpack`` are gone —
        the dispatcher hints at the new home so users discover ``bundle``."""
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})

        for old in ("bundles", "reconcile", "merge", "pack", "unpack"):
            result = plugin._execute_references_cmd(
                {"subcommand": old, "target": ""}
            )
            assert "error" in result, f"expected error for 'references {old}'"
            assert "bundle" in result["error"].lower()


class TestBundleUnknownSubcommand:
    """The bundle command's unknown-subcommand error names the full surface."""

    def test_unknown_bundle_subcommand_lists_known_verbs(self, workspace):
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(workspace))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_bundle_cmd(
            {"subcommand": "nonsense", "target": ""}
        )

        assert "error" in result
        assert "list" in result["error"]
        assert "create" in result["error"]
        assert "pack" in result["error"]
