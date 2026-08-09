"""Tests for the merge module and the ``references merge`` subcommand.

Covers:
    * ``parse_merge_args`` — flag parser edge cases
    * ``merge_bundle`` — compatible-model hot path, re-embed path,
      collision strategies (reject/prefix/newer), dry-run
    * ``ReferencesPlugin._cmd_references_merge`` — the user-facing wiring
      that resolves source (loaded bundle or filesystem path), stitches
      the merged refs into the live catalog, and re-attaches matchers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from shared.plugins.references.bundle import (
    EMBEDDING_CONFIG_FILENAME,
    Bundle,
    metadata_hash,
)
from shared.plugins.references.embedding_types import EmbeddingResult
from shared.plugins.references.merge import (
    MergeOptions,
    MergeStatus,
    _read_bundle_manifest,
    _load_sources_from_dir,
    merge_bundle,
    parse_merge_args,
)
from shared.plugins.references.models import (
    EmbeddingMetadata,
    InjectionMode,
    ReferenceSource,
    SourceType,
)
from shared.plugins.references.plugin import ReferencesPlugin


# Skip the heavy merge tests when numpy is unavailable — they all need it.
np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_ref(directory: Path, sid: str, *, extra=None):
    directory.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "id": sid,
        "name": sid.replace("-", " ").title(),
        "description": f"Description for {sid}",
        "type": "local",
        "path": f"./dummy/{sid}.md",
        "mode": "selectable",
    }
    if extra:
        payload.update(extra)
    (directory / f"{sid}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )


def _build_bundle(
    directory: Path,
    *,
    rows,
    vectors,
    model="mock-model",
    dim=4,
    sidecar="references.embeddings.npy",
) -> Bundle:
    """Create an on-disk bundle with a pre-populated sidecar."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": model,
        "embedding_dimensions": dim,
        "embedding_sidecar": sidecar,
        "rows": list(rows),
    }))
    matrix = np.asarray(vectors, dtype=np.float32)
    np.save(directory / sidecar, matrix, allow_pickle=False)
    return Bundle(
        name=directory.name,
        directory=directory.resolve(),
        embedding_model=model,
        embedding_dimensions=dim,
        embedding_sidecar=sidecar,
        embedding_rows=list(rows),
        owned_source_ids=set(rows),
    )


def _load_refs_from_dir(directory: Path) -> List[ReferenceSource]:
    return _load_sources_from_dir(directory)


class _FakeProvider:
    """Embedding provider stub for --re-embed tests."""

    def __init__(self, *, model="mock-model", dim=4, vectors=None):
        self.model_name = model
        self.dimensions = dim
        self.available = True
        self._vectors = vectors or {}

    def load_model(self):
        return True

    def embed_text(self, text):
        vec = self._vectors.get(text, [0.5, 0.5, 0.5, 0.5])
        return EmbeddingResult(
            embedding=list(vec),
            model=self.model_name,
            dimensions=self.dimensions,
        )

    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]

    def embed_text_as_array(self, text):  # pragma: no cover — not used here
        return None


# ---------------------------------------------------------------------------
# parse_merge_args
# ---------------------------------------------------------------------------


class TestParseMergeArgs:
    def test_plain_source(self):
        source, target, opts, err = parse_merge_args(["teammate"])
        assert err is None
        assert source == "teammate"
        assert target is None
        assert opts.on_conflict == "reject"
        assert opts.re_embed is False
        assert opts.dry_run is False

    def test_into_flag(self):
        source, target, opts, err = parse_merge_args(["src", "--into", "root"])
        assert err is None
        assert source == "src"
        assert target == "root"

    def test_on_conflict_flag(self):
        source, target, opts, err = parse_merge_args(
            ["src", "--on-conflict", "prefix"]
        )
        assert err is None
        assert opts.on_conflict == "prefix"

    def test_re_embed_and_dry_run(self):
        source, target, opts, err = parse_merge_args(
            ["src", "--re-embed", "--dry-run"]
        )
        assert err is None
        assert opts.re_embed is True
        assert opts.dry_run is True

    def test_missing_flag_value(self):
        source, target, opts, err = parse_merge_args(["src", "--into"])
        assert err == "Flag --into requires a value"

    def test_unknown_flag(self):
        source, target, opts, err = parse_merge_args(["src", "--weird"])
        assert err is not None and "--weird" in err

    def test_extra_positional_rejected(self):
        source, target, opts, err = parse_merge_args(["src", "other", "extra"])
        assert err is not None

    def test_options_validate_catches_bad_on_conflict(self):
        opts = MergeOptions(on_conflict="bogus")
        msg = opts.validate()
        assert msg is not None and "bogus" in msg


# ---------------------------------------------------------------------------
# merge_bundle — compatible-model hot path
# ---------------------------------------------------------------------------


class TestMergeCompatible:
    def test_simple_append(self, tmp_path):
        """Same model + dim + no collisions: source rows appended to target."""
        target_dir = tmp_path / "target"
        src_dir = tmp_path / "src"
        target = _build_bundle(
            target_dir, rows=["t1"], vectors=[[1, 0, 0, 0]],
        )
        source = _build_bundle(
            src_dir, rows=["s1", "s2"], vectors=[[0, 1, 0, 0], [0, 0, 1, 0]],
        )
        _write_ref(target_dir, "t1")
        _write_ref(src_dir, "s1")
        _write_ref(src_dir, "s2")

        result = merge_bundle(
            target=target,
            source_bundle=source,
            source_sources=_load_refs_from_dir(src_dir),
            target_sources=_load_refs_from_dir(target_dir),
            provider=None,
            options=MergeOptions(),
        )

        assert result.status == MergeStatus.OK
        assert result.added == ["s1", "s2"]
        assert result.final_row_count == 3

        matrix = np.load(target.sidecar_path)
        assert matrix.shape == (3, 4)
        # Target row preserved, source rows appended in order.
        np.testing.assert_array_equal(matrix[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(matrix[1], [0, 1, 0, 0])
        np.testing.assert_array_equal(matrix[2], [0, 0, 1, 0])

        # Source JSONs copied into target dir.
        assert (target_dir / "s1.json").is_file()
        assert (target_dir / "s2.json").is_file()

        # Manifest updated.
        manifest = json.loads(target.manifest_path.read_text())
        assert manifest["rows"] == ["t1", "s1", "s2"]

    def test_dry_run_does_not_touch_disk(self, tmp_path):
        target_dir = tmp_path / "target"
        src_dir = tmp_path / "src"
        target = _build_bundle(target_dir, rows=["t1"], vectors=[[1, 0, 0, 0]])
        source = _build_bundle(src_dir, rows=["s1"], vectors=[[0, 1, 0, 0]])
        _write_ref(target_dir, "t1")
        _write_ref(src_dir, "s1")

        original_matrix = np.load(target.sidecar_path).copy()

        result = merge_bundle(
            target=target,
            source_bundle=source,
            source_sources=_load_refs_from_dir(src_dir),
            target_sources=_load_refs_from_dir(target_dir),
            provider=None,
            options=MergeOptions(dry_run=True),
        )

        assert result.status == MergeStatus.DRY_RUN
        assert result.added == ["s1"]
        # Sidecar unchanged.
        np.testing.assert_array_equal(np.load(target.sidecar_path), original_matrix)
        # Source JSON not copied.
        assert not (target_dir / "s1.json").is_file()
        # Manifest unchanged.
        manifest = json.loads(target.manifest_path.read_text())
        assert manifest["rows"] == ["t1"]


# ---------------------------------------------------------------------------
# Conflict strategies
# ---------------------------------------------------------------------------


class TestMergeConflicts:
    def _setup_collision(self, tmp_path):
        target_dir = tmp_path / "target"
        src_dir = tmp_path / "src"
        target = _build_bundle(target_dir, rows=["shared"], vectors=[[1, 0, 0, 0]])
        source = _build_bundle(src_dir, rows=["shared"], vectors=[[0, 1, 0, 0]])
        _write_ref(target_dir, "shared", extra={
            "description": "Target version",
            "embedding": {"source_hash": "sha256:target-hash"},
        })
        _write_ref(src_dir, "shared", extra={
            "description": "Source version",
            "embedding": {"source_hash": "sha256:source-hash"},
        })
        return target, source, target_dir, src_dir

    def test_reject_mode_errors(self, tmp_path):
        target, source, target_dir, src_dir = self._setup_collision(tmp_path)

        result = merge_bundle(
            target=target,
            source_bundle=source,
            source_sources=_load_refs_from_dir(src_dir),
            target_sources=_load_refs_from_dir(target_dir),
            provider=None,
            options=MergeOptions(on_conflict="reject"),
        )

        assert result.status == MergeStatus.ERROR
        assert result.conflicts == ["shared"]
        # On error, target stays untouched.
        matrix = np.load(target.sidecar_path)
        assert matrix.shape == (1, 4)

    def test_prefix_mode_renames_and_appends(self, tmp_path):
        target, source, target_dir, src_dir = self._setup_collision(tmp_path)

        result = merge_bundle(
            target=target,
            source_bundle=source,
            source_sources=_load_refs_from_dir(src_dir),
            target_sources=_load_refs_from_dir(target_dir),
            provider=None,
            options=MergeOptions(on_conflict="prefix"),
        )

        assert result.status == MergeStatus.OK
        assert result.renamed == [("shared", "src_shared")]
        assert "src_shared" in result.added

        # Both the target's original row and the renamed source row live on.
        matrix = np.load(target.sidecar_path)
        assert matrix.shape == (2, 4)
        np.testing.assert_array_equal(matrix[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(matrix[1], [0, 1, 0, 0])

        # The renamed JSON file exists with the rewritten id.
        renamed_path = target_dir / "src_shared.json"
        assert renamed_path.is_file()
        raw = json.loads(renamed_path.read_text())
        assert raw["id"] == "src_shared"

        # Target's original shared.json is untouched.
        original = json.loads((target_dir / "shared.json").read_text())
        assert original["description"] == "Target version"

    def test_newer_mode_drops_same_hash(self, tmp_path):
        """Same source_hash → skip (it's already effectively in target)."""
        target_dir = tmp_path / "target"
        src_dir = tmp_path / "src"
        target = _build_bundle(target_dir, rows=["shared"], vectors=[[1, 0, 0, 0]])
        source = _build_bundle(src_dir, rows=["shared"], vectors=[[9, 9, 9, 9]])
        shared_payload = {
            "embedding": {"source_hash": "sha256:identical"},
        }
        _write_ref(target_dir, "shared", extra=shared_payload)
        _write_ref(src_dir, "shared", extra=shared_payload)

        result = merge_bundle(
            target=target,
            source_bundle=source,
            source_sources=_load_refs_from_dir(src_dir),
            target_sources=_load_refs_from_dir(target_dir),
            provider=None,
            options=MergeOptions(on_conflict="newer"),
        )

        assert result.status == MergeStatus.OK
        assert result.added == []
        assert any(sid == "shared" for sid, _reason in result.skipped)
        # Target kept its original row.
        matrix = np.load(target.sidecar_path)
        np.testing.assert_array_equal(matrix[0], [1, 0, 0, 0])


# ---------------------------------------------------------------------------
# Re-embed path
# ---------------------------------------------------------------------------


class TestMergeReEmbed:
    def test_model_mismatch_without_reembed_errors(self, tmp_path):
        target_dir = tmp_path / "target"
        src_dir = tmp_path / "src"
        target = _build_bundle(target_dir, rows=[], vectors=[], dim=4)
        # Source uses a different model: same dim but different model name
        # means vectors are not directly comparable.
        source = _build_bundle(
            src_dir, rows=["s1"], vectors=[[0, 1, 0, 0]],
            model="other-model", dim=4,
        )
        _write_ref(src_dir, "s1")

        result = merge_bundle(
            target=target,
            source_bundle=source,
            source_sources=_load_refs_from_dir(src_dir),
            target_sources=[],
            provider=None,
            options=MergeOptions(),
        )

        assert result.status == MergeStatus.ERROR
        assert "--re-embed" in (result.error or "")

    def test_re_embed_across_models(self, tmp_path):
        """--re-embed: source vectors are regenerated against target's model."""
        target_dir = tmp_path / "target"
        src_dir = tmp_path / "src"
        target = _build_bundle(
            target_dir,
            rows=[],
            vectors=np.zeros((0, 4), dtype=np.float32).tolist() or [[]],
            model="target-model",
            dim=4,
        )
        # Need to overwrite with a truly empty matrix (the helper above
        # handles rows=[] but np.asarray([[]]) yields shape (1, 0)).
        np.save(
            target.sidecar_path,
            np.zeros((0, 4), dtype=np.float32),
            allow_pickle=False,
        )

        source = _build_bundle(
            src_dir, rows=["s1"], vectors=[[9, 9, 9, 9]],
            model="source-model",
        )
        _write_ref(src_dir, "s1")

        # Provider returns predictable vectors keyed on embed text.
        source_ref = ReferenceSource.from_dict(
            json.loads((src_dir / "s1.json").read_text())
        )
        provider = _FakeProvider(model="target-model", vectors={
            "\n".join([
                source_ref.name,
                source_ref.description,
                "",  # no tags
                "",  # no fetch_hint
            ]): [0.1, 0.2, 0.3, 0.4],
        })

        result = merge_bundle(
            target=target,
            source_bundle=source,
            source_sources=_load_refs_from_dir(src_dir),
            target_sources=[],
            provider=provider,
            options=MergeOptions(re_embed=True),
        )

        assert result.status == MergeStatus.OK, result.error
        assert result.added == ["s1"]

        matrix = np.load(target.sidecar_path)
        assert matrix.shape == (1, 4)
        np.testing.assert_allclose(matrix[0], [0.1, 0.2, 0.3, 0.4], atol=1e-6)

        # Copied JSON carries a fresh sha256: hash stamped by merge.
        copied = json.loads((target_dir / "s1.json").read_text())
        assert copied["embedding"]["source_hash"].startswith("sha256:")
        # Matches the canonical metadata hash of the source.
        assert copied["embedding"]["source_hash"] == metadata_hash(source_ref)

    def test_re_embed_without_provider_is_unavailable(self, tmp_path):
        target_dir = tmp_path / "target"
        src_dir = tmp_path / "src"
        target = _build_bundle(target_dir, rows=[], vectors=[], model="a")
        np.save(
            target.sidecar_path,
            np.zeros((0, 4), dtype=np.float32),
            allow_pickle=False,
        )
        source = _build_bundle(src_dir, rows=["s1"], vectors=[[1, 1, 1, 1]], model="b")
        _write_ref(src_dir, "s1")

        result = merge_bundle(
            target=target,
            source_bundle=source,
            source_sources=_load_refs_from_dir(src_dir),
            target_sources=[],
            provider=None,
            options=MergeOptions(re_embed=True),
        )

        assert result.status == MergeStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Plugin subcommand wiring
# ---------------------------------------------------------------------------


class TestMergeSubcommand:
    def test_missing_source_shows_usage(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / ".jaato" / "references").mkdir(parents=True)
        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(ws))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_references_cmd(
            {"subcommand": "bundle", "target": "merge ".strip()}
        )

        assert "error" in result
        assert "Usage" in result["error"]

    def test_unknown_source_errors(self, tmp_path):
        ws = tmp_path / "ws"
        refs = ws / ".jaato" / "references"
        refs.mkdir(parents=True)
        _build_bundle(refs, rows=[], vectors=[])
        np.save(refs / "references.embeddings.npy",
                np.zeros((0, 4), dtype=np.float32))

        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(ws))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_references_cmd(
            {"subcommand": "bundle", "target": "merge ghost-bundle".strip()}
        )

        assert "error" in result
        assert "ghost-bundle" in result["error"]

    def test_merge_sub_bundle_into_root(self, tmp_path):
        """End-to-end: merge a loaded sub-bundle into the root bundle."""
        ws = tmp_path / "ws"
        refs = ws / ".jaato" / "references"

        # Root bundle with one ref.
        _write_ref(refs, "root-a", extra={
            "embedding": {"source_hash": "sha256:root-a-hash"},
        })
        _build_bundle(refs, rows=["root-a"], vectors=[[1, 0, 0, 0]])

        # Sub-bundle 'teammate' with its own ref.
        sub = refs / "teammate"
        _write_ref(sub, "teammate-x", extra={
            "embedding": {"source_hash": "sha256:teammate-x-hash"},
        })
        _build_bundle(sub, rows=["teammate-x"], vectors=[[0, 1, 0, 0]])

        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(ws))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_references_cmd(
            {"subcommand": "bundle", "target": "merge teammate".strip()}
        )

        assert result["status"] == "ok"
        assert result["added"] == ["teammate-x"]
        # Bundle names are SCOPE-QUALIFIED now: the root bundle reports as
        # "workspace:(root)" (see plugin.py:3345, which documents that as the
        # --into default).
        assert result["target"] == "workspace:(root)"

        # Root's manifest now has both rows.
        root_manifest = json.loads((refs / EMBEDDING_CONFIG_FILENAME).read_text())
        assert root_manifest["rows"] == ["root-a", "teammate-x"]

        # Root's sidecar has 2 rows.
        matrix = np.load(refs / "references.embeddings.npy")
        assert matrix.shape == (2, 4)

        # Root directory now has teammate-x.json too.
        assert (refs / "teammate-x.json").is_file()

        # Live catalog reflects the merge — teammate-x is in root now.
        merged_in_catalog = [
            s for s in plugin._sources
            if s.id == "teammate-x" and s.bundle_name == ""
        ]
        assert len(merged_in_catalog) == 1

    def test_dry_run_does_not_mutate_catalog(self, tmp_path):
        ws = tmp_path / "ws"
        refs = ws / ".jaato" / "references"

        _write_ref(refs, "root-a", extra={
            "embedding": {"source_hash": "sha256:root-a-hash"},
        })
        _build_bundle(refs, rows=["root-a"], vectors=[[1, 0, 0, 0]])

        sub = refs / "teammate"
        _write_ref(sub, "teammate-x", extra={
            "embedding": {"source_hash": "sha256:teammate-x-hash"},
        })
        _build_bundle(sub, rows=["teammate-x"], vectors=[[0, 1, 0, 0]])

        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(ws))
        plugin.initialize({"lookup_strategy": "tags_only"})

        result = plugin._execute_references_cmd(
            {"subcommand": "bundle", "target": "merge teammate --dry-run".strip()}
        )

        assert result["status"] == "dry_run"
        # Catalog still has teammate-x only under the 'teammate' bundle.
        in_root = [
            s for s in plugin._sources
            if s.id == "teammate-x" and s.bundle_name == ""
        ]
        assert in_root == []

    def test_merge_same_bundle_into_self_errors(self, tmp_path):
        ws = tmp_path / "ws"
        refs = ws / ".jaato" / "references"
        sub = refs / "teammate"
        sub.mkdir(parents=True)
        _write_ref(sub, "teammate-x")
        _build_bundle(sub, rows=["teammate-x"], vectors=[[0, 1, 0, 0]])
        # Root also has a manifest (even if empty) so both bundles load.
        refs.mkdir(parents=True, exist_ok=True)
        _build_bundle(refs, rows=[], vectors=[])
        np.save(refs / "references.embeddings.npy",
                np.zeros((0, 4), dtype=np.float32))

        plugin = ReferencesPlugin()
        plugin.set_workspace_path(str(ws))
        plugin.initialize({"lookup_strategy": "tags_only"})

        # Merge teammate into teammate (explicit --into target == source).
        result = plugin._execute_references_cmd(
            {"subcommand": "bundle", "target": "merge teammate --into teammate".strip()}
        )

        assert "error" in result
        assert "same bundle" in result["error"]
