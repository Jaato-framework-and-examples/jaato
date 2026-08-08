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


class TestLazyEmbeddingProviderInit:
    """Tests for the server 0.6.111+ lazy embedding-provider load.

    ``initialize()`` skips ``_init_embedding_provider`` when the
    workspace has no bundles — saving the premium SentenceTransformer
    module import cost on workspaces that do not run semantic queries.
    First ``compute_embedding`` call lazy-loads the provider via
    ``_cached_init_config``.
    """

    def _make_plugin_with_no_bundles(self, tmp_path):
        """Create a freshly-initialized ReferencesPlugin in an empty workspace.

        The temp workspace has no ``.jaato/references/`` directory so
        bundle discovery returns an empty list.
        """
        from shared.plugins.references.plugin import ReferencesPlugin
        plugin = ReferencesPlugin()
        plugin.initialize({"workspace_path": str(tmp_path)})
        return plugin

    def test_no_bundles_skips_provider_init(self, tmp_path):
        """Workspace without bundles → provider stays None, config cached."""
        plugin = self._make_plugin_with_no_bundles(tmp_path)
        assert plugin._embedding_provider is None
        assert plugin._cached_init_config is not None
        # _bundles must be empty for the skip path to apply.
        assert plugin._bundles == []

    def test_no_bundles_compute_embedding_triggers_lazy_init(self, tmp_path):
        """compute_embedding call attempts lazy-load when no bundles existed at init.

        We patch ``_init_embedding_provider`` to assert the lazy call
        happens with the cached config; the patched method does not
        install a real provider, so the error path still fires
        (matching the existing ``test_error_when_provider_unavailable``
        behaviour).
        """
        from unittest.mock import patch
        plugin = self._make_plugin_with_no_bundles(tmp_path)
        assert plugin._embedding_provider is None
        with patch.object(plugin, "_init_embedding_provider") as mock_init:
            result = plugin._execute_compute_embedding({"input": "hello"})
        # Lazy-init was attempted with the cached config.
        mock_init.assert_called_once_with(plugin._cached_init_config)
        # Provider still None (mock didn't install one) → error returned.
        assert "error" in result

    def test_with_bundles_eager_init(self, tmp_path):
        """When bundles exist at init time, provider load happens eagerly.

        We seed a workspace with an ``embedding_config.json`` so
        ``_discover_and_load_bundles`` returns a non-empty list, then
        verify ``_init_embedding_provider`` is invoked during
        ``initialize()`` rather than deferred.
        """
        import json
        from unittest.mock import patch
        from shared.plugins.references.plugin import ReferencesPlugin

        # Seed a minimal embedding_config.json so the workspace has a root bundle.
        # Required fields per bundle_common/bundle.py:120-146.
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        (refs_dir / "embedding_config.json").write_text(
            json.dumps({
                "embedding_model": "test-model",
                "embedding_dimensions": 3,
                "embedding_sidecar": "refs.npy",
                "rows": [],
                "reconcile_mode": "off",
            })
        )

        plugin = ReferencesPlugin()
        with patch.object(plugin, "_init_embedding_provider") as mock_init:
            plugin.initialize({"workspace_path": str(tmp_path)})
        # Eager init happened (called at least once during initialize()).
        assert mock_init.called, "Expected eager _init_embedding_provider call when bundles exist"
