"""Tests for the per-bundle reconcile pass.

The numpy-free paths (CLEAN, UNAVAILABLE, SKIPPED_BUSY) are covered
unconditionally. The UPDATED path needs numpy and is marked to skip when
the embedding stack is not installed in the test environment.
"""

import json
from pathlib import Path

import pytest

from shared.plugins.references.bundle import (
    EMBEDDING_CONFIG_FILENAME,
    Bundle,
    metadata_hash,
)
from shared.plugins.references.models import (
    EmbeddingMetadata,
    InjectionMode,
    ReferenceSource,
    SourceType,
)
from shared.plugins.references.reconcile import (
    ReconcileStatus,
    reconcile_bundle,
)


def _source(sid, *, bundle_name="", hash_=None, name=None):
    src = ReferenceSource(
        id=sid,
        name=name or sid,
        description=f"Description for {sid}",
        type=SourceType.LOCAL,
        mode=InjectionMode.SELECTABLE,
        path=f"/tmp/{sid}",
        bundle_name=bundle_name,
    )
    if hash_ is not None:
        src.embedding = EmbeddingMetadata(source_hash=hash_)
    return src


def _bundle(tmp_path: Path, rows):
    refs = tmp_path / "references"
    refs.mkdir(exist_ok=True)
    (refs / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": "mock-model",
        "embedding_dimensions": 4,
        "embedding_sidecar": "references.embeddings.npy",
        "rows": list(rows),
    }))
    return Bundle(
        name="",
        directory=refs.resolve(),
        embedding_model="mock-model",
        embedding_dimensions=4,
        embedding_sidecar="references.embeddings.npy",
        embedding_rows=list(rows),
        owned_source_ids=set(rows),
    )


class _FakeEmbeddingResult:
    def __init__(self, vector, dim=4, model="mock-model"):
        self.embedding = list(vector)
        self.dimensions = dim
        self.model = model
        self.input_tokens = 0


class _FakeProvider:
    """Minimal embedding provider stub for reconcile tests."""

    def __init__(self, *, vectors=None, fail_batch=False, available=True):
        self.model_name = "mock-model"
        self.dimensions = 4
        self.available = available
        self._vectors = vectors or {}  # text → list[float]
        self._fail_batch = fail_batch

    def load_model(self):
        self.available = True
        return True

    def embed_text(self, text):
        vec = self._vectors.get(text, [0.1, 0.2, 0.3, 0.4])
        return _FakeEmbeddingResult(vec)

    def embed_batch(self, texts):
        if self._fail_batch:
            raise RuntimeError("batch failed")
        return [self.embed_text(t) for t in texts]

    def embed_text_as_array(self, text):  # pragma: no cover — not used here
        return None


class TestReconcileCleanPath:
    def test_no_drift_returns_clean_without_numpy(self, tmp_path):
        """A fully-synced bundle never touches disk or numpy."""
        src = _source("a")
        src.embedding = EmbeddingMetadata(source_hash=metadata_hash(src))
        bundle = _bundle(tmp_path, rows=["a"])

        result = reconcile_bundle(bundle, [src], provider=None)

        assert result.status == ReconcileStatus.CLEAN
        assert result.added == []
        assert result.final_row_count == 1

    def test_no_provider_returns_unavailable(self, tmp_path):
        """Drift + no provider → UNAVAILABLE, no write."""
        src = _source("a")
        bundle = _bundle(tmp_path, rows=[])

        result = reconcile_bundle(bundle, [src], provider=None)

        assert result.status == ReconcileStatus.UNAVAILABLE
        assert bundle.embedding_rows == []

    def test_provider_unavailable_returns_unavailable(self, tmp_path):
        src = _source("a")
        bundle = _bundle(tmp_path, rows=[])
        provider = _FakeProvider(available=False)

        result = reconcile_bundle(bundle, [src], provider=provider)

        assert result.status == ReconcileStatus.UNAVAILABLE


@pytest.fixture
def np():
    """numpy is optional in CI; skip the UPDATED-path tests when it's missing."""
    return pytest.importorskip("numpy")


class TestReconcileUpdatePath:
    """The UPDATED path rewrites sidecar + manifest atomically. Needs numpy."""

    def test_adds_new_source_to_sidecar(self, tmp_path, np):
        """A missing source gets a vector and a row."""
        # Start with an empty sidecar (rows=[], matrix shape (0, 4)).
        bundle = _bundle(tmp_path, rows=[])
        np.save(
            bundle.sidecar_path,
            np.zeros((0, 4), dtype=np.float32),
            allow_pickle=False,
        )
        src = _source("a")
        # Write the reference JSON so source_hash can be stamped back.
        (bundle.directory / "a.json").write_text(json.dumps({
            "id": "a",
            "name": "a",
            "description": "Description for a",
            "type": "local",
            "path": "/tmp/a",
            "mode": "selectable",
        }))

        provider = _FakeProvider(vectors={
            # reconcile hashes name+description+tags+fetch_hint
            "a\nDescription for a\n\n": [1.0, 0.0, 0.0, 0.0],
        })

        result = reconcile_bundle(bundle, [src], provider=provider)

        assert result.status == ReconcileStatus.UPDATED
        assert result.added == ["a"]
        assert bundle.embedding_rows == ["a"]

        # Sidecar now has one row.
        matrix = np.load(bundle.sidecar_path)
        assert matrix.shape == (1, 4)

        # Manifest has been rewritten.
        raw = json.loads(bundle.manifest_path.read_text())
        assert raw["rows"] == ["a"]

        # Reference JSON stamped with the new source_hash.
        ref_json = json.loads((bundle.directory / "a.json").read_text())
        assert ref_json["embedding"]["source_hash"] == metadata_hash(src)

    def test_prunes_orphan_rows(self, tmp_path, np):
        """A row pointing at a missing source gets dropped."""
        bundle = _bundle(tmp_path, rows=["ghost", "real"])
        initial = np.array(
            [[9, 9, 9, 9], [1, 2, 3, 4]],
            dtype=np.float32,
        )
        np.save(bundle.sidecar_path, initial, allow_pickle=False)
        real = _source("real")
        real.embedding = EmbeddingMetadata(source_hash=metadata_hash(real))

        provider = _FakeProvider()  # shouldn't be called — only prune

        result = reconcile_bundle(bundle, [real], provider=provider)

        assert result.status == ReconcileStatus.UPDATED
        assert result.pruned == ["ghost"]
        assert bundle.embedding_rows == ["real"]

        matrix = np.load(bundle.sidecar_path)
        assert matrix.shape == (1, 4)
        # The real row's vector survives.
        assert list(matrix[0]) == [1.0, 2.0, 3.0, 4.0]

    def test_refreshes_stale_row(self, tmp_path, np):
        """A row whose source metadata drifted gets re-embedded."""
        bundle = _bundle(tmp_path, rows=["a"])
        initial = np.array([[0, 0, 0, 0]], dtype=np.float32)
        np.save(bundle.sidecar_path, initial, allow_pickle=False)

        src = _source("a", hash_="sha256:old-and-wrong")
        (bundle.directory / "a.json").write_text(json.dumps({
            "id": "a",
            "name": "a",
            "description": "Description for a",
            "type": "local",
            "path": "/tmp/a",
            "mode": "selectable",
            "embedding": {"source_hash": "sha256:old-and-wrong"},
        }))
        provider = _FakeProvider(vectors={
            "a\nDescription for a\n\n": [5.0, 5.0, 5.0, 5.0],
        })

        result = reconcile_bundle(bundle, [src], provider=provider)

        assert result.status == ReconcileStatus.UPDATED
        assert result.refreshed == ["a"]

        matrix = np.load(bundle.sidecar_path)
        assert list(matrix[0]) == [5.0, 5.0, 5.0, 5.0]

        # JSON now carries the fresh hash.
        ref_json = json.loads((bundle.directory / "a.json").read_text())
        assert ref_json["embedding"]["source_hash"] == metadata_hash(src)
