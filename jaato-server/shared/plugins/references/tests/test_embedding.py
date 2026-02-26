"""Tests for embedding types, provider, and semantic matching."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared.plugins.references.embedding_types import EmbeddingResult, SemanticMatch


class TestEmbeddingResult:
    """Tests for the EmbeddingResult dataclass."""

    def test_to_dict(self):
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
            input_tokens=10,
        )
        d = result.to_dict()
        assert d == {
            "embedding": [0.1, 0.2, 0.3],
            "model": "test-model",
            "dimensions": 3,
            "input_tokens": 10,
        }

    def test_default_input_tokens(self):
        result = EmbeddingResult(
            embedding=[], model="m", dimensions=0
        )
        assert result.input_tokens == 0


class TestSemanticMatch:
    """Tests for the SemanticMatch dataclass."""

    def test_fields(self):
        m = SemanticMatch(source_id="ref-1", score=0.92, embedding_index=5)
        assert m.source_id == "ref-1"
        assert m.score == 0.92
        assert m.embedding_index == 5


class TestLocalEmbeddingProvider:
    """Tests for LocalEmbeddingProvider."""

    def test_unavailable_when_model_load_fails(self):
        """Provider reports unavailable when model fails to load."""
        from shared.plugins.references.embedding_provider import LocalEmbeddingProvider
        with patch(
            "shared.plugins.references.embedding_provider.SentenceTransformer",
            side_effect=RuntimeError("model not found"),
        ):
            provider = LocalEmbeddingProvider(eager_load=True)
            assert not provider.available

    def test_embed_text_returns_none_when_unavailable(self):
        """embed_text returns None when provider model failed to load."""
        from shared.plugins.references.embedding_provider import LocalEmbeddingProvider
        with patch(
            "shared.plugins.references.embedding_provider.SentenceTransformer",
            side_effect=RuntimeError("model not found"),
        ):
            provider = LocalEmbeddingProvider(eager_load=False)
            assert provider.embed_text("hello") is None

    def test_embed_batch_returns_empty_when_unavailable(self):
        """embed_batch returns empty list when provider model failed to load."""
        from shared.plugins.references.embedding_provider import LocalEmbeddingProvider
        with patch(
            "shared.plugins.references.embedding_provider.SentenceTransformer",
            side_effect=RuntimeError("model not found"),
        ):
            provider = LocalEmbeddingProvider(eager_load=False)
            assert provider.embed_batch(["hello", "world"]) == []

    def test_embed_text_as_array_returns_none_when_unavailable(self):
        """embed_text_as_array returns None when provider model failed to load."""
        from shared.plugins.references.embedding_provider import LocalEmbeddingProvider
        with patch(
            "shared.plugins.references.embedding_provider.SentenceTransformer",
            side_effect=RuntimeError("model not found"),
        ):
            provider = LocalEmbeddingProvider(eager_load=False)
            assert provider.embed_text_as_array("hello") is None


class TestSemanticMatcher:
    """Tests for SemanticMatcher with mocked numpy operations."""

    def test_not_available_initially(self):
        from shared.plugins.references.semantic_matching import SemanticMatcher
        matcher = SemanticMatcher()
        assert not matcher.available

    def test_load_index_fails_with_missing_file(self):
        """load_index returns False when sidecar file doesn't exist."""
        from shared.plugins.references.semantic_matching import SemanticMatcher
        matcher = SemanticMatcher()
        result = matcher.load_index(
            sidecar_path="/nonexistent_path_12345.npy",
            embedding_model="test",
            embedding_dimensions=3,
            index_to_source_id={0: "ref-1"},
        )
        assert result is False

    def test_load_index_succeeds_with_valid_npy(self):
        """load_index succeeds with a valid .npy file."""
        from shared.plugins.references.semantic_matching import SemanticMatcher

        # Create a temporary .npy file with 2 vectors of dimension 3
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, matrix)
            npy_path = f.name

        try:
            matcher = SemanticMatcher()
            result = matcher.load_index(
                sidecar_path=npy_path,
                embedding_model="test-model",
                embedding_dimensions=3,
                index_to_source_id={0: "ref-a", 1: "ref-b"},
            )
            assert result is True
            assert matcher.embedding_model == "test-model"
            assert matcher.embedding_dimensions == 3
        finally:
            os.unlink(npy_path)

    def test_load_index_rejects_dimension_mismatch(self):
        """load_index returns False when matrix dimensions don't match config."""
        from shared.plugins.references.semantic_matching import SemanticMatcher

        matrix = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, matrix)
            npy_path = f.name

        try:
            matcher = SemanticMatcher()
            result = matcher.load_index(
                sidecar_path=npy_path,
                embedding_model="test",
                embedding_dimensions=5,  # mismatch: matrix has 3, config says 5
                index_to_source_id={0: "ref-a"},
            )
            assert result is False
        finally:
            os.unlink(npy_path)

    def test_find_matches_with_loaded_index(self):
        """find_matches returns matches above threshold."""
        from shared.plugins.references.semantic_matching import SemanticMatcher

        # Two orthogonal unit vectors
        matrix = np.array([
            [1.0, 0.0, 0.0],  # ref-a: points along x
            [0.0, 1.0, 0.0],  # ref-b: points along y
        ], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, matrix)
            npy_path = f.name

        try:
            matcher = SemanticMatcher()
            matcher.load_index(
                sidecar_path=npy_path,
                embedding_model="test",
                embedding_dimensions=3,
                index_to_source_id={0: "ref-a", 1: "ref-b"},
            )

            # Query vector close to ref-a (x-axis)
            query = np.array([0.95, 0.05, 0.0], dtype=np.float32)
            query = query / np.linalg.norm(query)

            matches = matcher.find_matches(query, threshold=0.5, top_k=5)
            assert len(matches) == 1
            assert matches[0].source_id == "ref-a"
            assert matches[0].score > 0.9
        finally:
            os.unlink(npy_path)

    def test_find_matches_excludes_ids(self):
        """find_matches respects exclude_ids."""
        from shared.plugins.references.semantic_matching import SemanticMatcher

        # Two similar vectors
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, matrix)
            npy_path = f.name

        try:
            matcher = SemanticMatcher()
            matcher.load_index(
                sidecar_path=npy_path,
                embedding_model="test",
                embedding_dimensions=3,
                index_to_source_id={0: "ref-a", 1: "ref-b"},
            )

            query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            matches = matcher.find_matches(
                query, threshold=0.5, top_k=5, exclude_ids={"ref-a"}
            )
            # ref-a excluded, only ref-b should appear
            source_ids = [m.source_id for m in matches]
            assert "ref-a" not in source_ids
            assert "ref-b" in source_ids
        finally:
            os.unlink(npy_path)

    def test_validate_model_mismatch(self):
        """validate_model returns False on model mismatch."""
        from shared.plugins.references.semantic_matching import SemanticMatcher
        matcher = SemanticMatcher()
        matcher.embedding_model = "model-a"
        assert matcher.validate_model("model-b") is False

    def test_validate_model_match(self):
        """validate_model returns True when models match."""
        from shared.plugins.references.semantic_matching import SemanticMatcher
        matcher = SemanticMatcher()
        matcher.embedding_model = "model-a"
        assert matcher.validate_model("model-a") is True


class TestComputeEmbeddingExecutor:
    """Tests for the compute_embedding tool executor on ReferencesPlugin."""

    def _make_plugin(self, provider_available=False):
        """Create a ReferencesPlugin with a mocked embedding provider."""
        from shared.plugins.references.plugin import ReferencesPlugin
        plugin = ReferencesPlugin()
        plugin._initialized = True

        if provider_available:
            mock_provider = MagicMock()
            mock_provider.available = True
            mock_provider.embed_text.return_value = EmbeddingResult(
                embedding=[0.1, 0.2, 0.3],
                model="test-model",
                dimensions=3,
                input_tokens=5,
            )
            plugin._embedding_provider = mock_provider
        return plugin

    def test_error_when_both_input_and_file(self):
        plugin = self._make_plugin(provider_available=True)
        result = plugin._execute_compute_embedding({"input": "hi", "file": "/tmp/x"})
        assert "error" in result

    def test_error_when_neither_input_nor_file(self):
        plugin = self._make_plugin(provider_available=True)
        result = plugin._execute_compute_embedding({})
        assert "error" in result

    def test_error_when_provider_unavailable(self):
        plugin = self._make_plugin(provider_available=False)
        result = plugin._execute_compute_embedding({"input": "hello"})
        assert "error" in result

    def test_success_with_text_input(self):
        plugin = self._make_plugin(provider_available=True)
        result = plugin._execute_compute_embedding({"input": "hello world"})
        assert "error" not in result
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["model"] == "test-model"
        assert result["dimensions"] == 3

    def test_success_with_file_input(self):
        plugin = self._make_plugin(provider_available=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("file content here")
            file_path = f.name

        try:
            result = plugin._execute_compute_embedding({"file": file_path})
            assert "error" not in result
            assert result["embedding"] == [0.1, 0.2, 0.3]
            # Verify the provider was called with file contents
            plugin._embedding_provider.embed_text.assert_called_once_with(
                "file content here"
            )
        finally:
            os.unlink(file_path)

    def test_error_with_nonexistent_file(self):
        plugin = self._make_plugin(provider_available=True)
        result = plugin._execute_compute_embedding(
            {"file": "/nonexistent/path/12345.txt"}
        )
        assert "error" in result
