"""Tests for embedding types and protocol-based embedding integration.

Concrete implementation tests (LocalEmbeddingProvider, SemanticMatcher) live
in the package that provides the implementation (e.g. jaato-premium).
"""

import os
import tempfile
from unittest.mock import MagicMock

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
