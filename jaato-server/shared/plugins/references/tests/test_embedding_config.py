"""Tests for embedding_config.json discovery in separate mode."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from shared.plugins.references.config_loader import (
    EMBEDDING_CONFIG_FILENAME,
    ReferencesConfig,
    _load_embedding_config,
    discover_references,
    load_config,
)


class TestEmbeddingConfigFilename:
    """The well-known filename constant."""

    def test_value(self):
        assert EMBEDDING_CONFIG_FILENAME == "embedding_config.json"


class TestLoadEmbeddingConfig:
    """Tests for the _load_embedding_config helper."""

    def test_loads_valid_config(self, tmp_path):
        refs_dir = tmp_path / "references"
        refs_dir.mkdir()
        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "embedding_sidecar": "references.embeddings.npy",
        }))

        config = ReferencesConfig(references_dir=str(refs_dir))
        _load_embedding_config(config, str(tmp_path))

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_dimensions == 384
        assert config.embedding_sidecar == "references.embeddings.npy"
        assert config.config_base_path == str(refs_dir.resolve())

    def test_no_op_when_file_missing(self, tmp_path):
        refs_dir = tmp_path / "references"
        refs_dir.mkdir()

        config = ReferencesConfig(references_dir=str(refs_dir))
        _load_embedding_config(config, str(tmp_path))

        assert config.embedding_model is None
        assert config.embedding_dimensions is None
        assert config.embedding_sidecar is None

    def test_no_op_when_dir_missing(self, tmp_path):
        config = ReferencesConfig(references_dir=str(tmp_path / "nonexistent"))
        _load_embedding_config(config, str(tmp_path))

        assert config.embedding_model is None

    def test_skips_invalid_json(self, tmp_path):
        refs_dir = tmp_path / "references"
        refs_dir.mkdir()
        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text("not json {{{")

        config = ReferencesConfig(references_dir=str(refs_dir))
        _load_embedding_config(config, str(tmp_path))

        assert config.embedding_model is None

    def test_skips_non_object(self, tmp_path):
        refs_dir = tmp_path / "references"
        refs_dir.mkdir()
        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps([1, 2, 3]))

        config = ReferencesConfig(references_dir=str(refs_dir))
        _load_embedding_config(config, str(tmp_path))

        assert config.embedding_model is None

    def test_skips_missing_required_fields(self, tmp_path):
        """All three fields must be present."""
        refs_dir = tmp_path / "references"
        refs_dir.mkdir()
        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "all-MiniLM-L6-v2",
            # missing embedding_dimensions and embedding_sidecar
        }))

        config = ReferencesConfig(references_dir=str(refs_dir))
        _load_embedding_config(config, str(tmp_path))

        assert config.embedding_model is None

    def test_does_not_overwrite_config_base_path(self, tmp_path):
        """If config_base_path is already set (merged mode), leave it alone."""
        refs_dir = tmp_path / "references"
        refs_dir.mkdir()
        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "embedding_sidecar": "references.embeddings.npy",
        }))

        config = ReferencesConfig(
            references_dir=str(refs_dir),
            config_base_path="/already/set",
        )
        _load_embedding_config(config, str(tmp_path))

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.config_base_path == "/already/set"

    def test_resolves_relative_references_dir(self, tmp_path):
        """references_dir can be relative to workspace_path."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "test-model",
            "embedding_dimensions": 128,
            "embedding_sidecar": "sidecar.npy",
        }))

        config = ReferencesConfig(references_dir=".jaato/references")
        _load_embedding_config(config, str(tmp_path))

        assert config.embedding_model == "test-model"
        assert config.embedding_dimensions == 128


class TestDiscoverReferencesSkipsEmbeddingConfig:
    """discover_references() should skip embedding_config.json."""

    def test_skips_embedding_config_file(self, tmp_path):
        refs_dir = tmp_path / "references"
        refs_dir.mkdir()

        # Write a valid reference
        (refs_dir / "my-ref.json").write_text(json.dumps({
            "id": "my-ref",
            "name": "My Reference",
            "type": "local",
            "path": "/tmp",
        }))

        # Write embedding config (should be skipped)
        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "embedding_sidecar": "references.embeddings.npy",
        }))

        sources = discover_references(str(refs_dir), base_path=str(tmp_path))
        ids = [s.id for s in sources]

        assert "my-ref" in ids
        assert len(sources) == 1  # Only the reference, not the config


class TestLoadConfigEmbeddingIntegration:
    """Integration: load_config reads embedding_config.json in separate mode."""

    def test_separate_mode_loads_embedding_config(self, tmp_path):
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)

        (refs_dir / "ref-a.json").write_text(json.dumps({
            "id": "ref-a",
            "name": "Reference A",
            "type": "local",
            "path": "/tmp",
            "embedding": {"index": 0, "source_hash": "sha256:abc123"},
        }))

        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "embedding_sidecar": "references.embeddings.npy",
        }))

        config = load_config(workspace_path=str(tmp_path))

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_dimensions == 384
        assert config.embedding_sidecar == "references.embeddings.npy"
        assert len(config.sources) == 1
        assert config.sources[0].id == "ref-a"

    def test_merged_mode_takes_precedence(self, tmp_path):
        """When a references.json exists with embedding fields, those win."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)

        # Write merged config with embedding fields
        config_file = tmp_path / "references.json"
        config_file.write_text(json.dumps({
            "version": "1.0",
            "sources": [{
                "id": "ref-merged",
                "name": "Merged Ref",
                "type": "local",
                "path": "/tmp",
            }],
            "embedding_model": "merged-model",
            "embedding_dimensions": 768,
            "embedding_sidecar": "merged.npy",
        }))

        # Also write separate embedding config (should be ignored)
        (refs_dir / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
            "embedding_model": "separate-model",
            "embedding_dimensions": 384,
            "embedding_sidecar": "separate.npy",
        }))

        config = load_config(path=str(config_file), workspace_path=str(tmp_path))

        # Merged config wins
        assert config.embedding_model == "merged-model"
        assert config.embedding_dimensions == 768

    def test_no_embedding_config_is_fine(self, tmp_path):
        """No embedding_config.json — embedding fields stay None."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)

        (refs_dir / "ref-a.json").write_text(json.dumps({
            "id": "ref-a",
            "name": "Reference A",
            "type": "local",
            "path": "/tmp",
        }))

        config = load_config(workspace_path=str(tmp_path))

        assert config.embedding_model is None
        assert config.embedding_dimensions is None
        assert config.embedding_sidecar is None
