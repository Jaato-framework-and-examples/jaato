"""Tests for validate_reference_file() in config_loader."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from shared.plugins.references.config_loader import validate_reference_file


class TestValidateReferenceFile:
    """Tests for the standalone reference file validator."""

    def test_valid_local_reference(self):
        data = {
            "id": "my-ref",
            "name": "My Reference",
            "description": "A test reference",
            "type": "local",
            "path": "/some/path/to/docs",
            "mode": "selectable",
            "tags": ["test", "docs"],
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    def test_valid_url_reference(self):
        data = {
            "id": "url-ref",
            "name": "URL Reference",
            "type": "url",
            "url": "https://example.com/docs",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    def test_valid_mcp_reference(self):
        data = {
            "id": "mcp-ref",
            "name": "MCP Reference",
            "type": "mcp",
            "server": "my-server",
            "tool": "fetch_docs",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    def test_valid_inline_reference(self):
        data = {
            "id": "inline-ref",
            "name": "Inline Reference",
            "type": "inline",
            "content": "Some inline content here.",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    def test_missing_id(self):
        data = {"name": "No ID", "type": "local", "path": "/tmp"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'id' is required" in e for e in errors)

    def test_missing_name(self):
        data = {"id": "no-name", "type": "local", "path": "/tmp"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'name' is required" in e for e in errors)

    def test_invalid_type(self):
        data = {"id": "bad-type", "name": "Bad Type", "type": "ftp"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("Invalid type 'ftp'" in e for e in errors)

    def test_invalid_mode(self):
        data = {"id": "bad-mode", "name": "Bad Mode", "type": "local", "path": "/tmp", "mode": "manual"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("Invalid mode 'manual'" in e for e in errors)

    def test_local_missing_path(self):
        data = {"id": "no-path", "name": "No Path", "type": "local"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'path' is required for local type" in e for e in errors)

    def test_url_missing_url(self):
        data = {"id": "no-url", "name": "No URL", "type": "url"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'url' is required for url type" in e for e in errors)

    def test_mcp_missing_server(self):
        data = {"id": "no-srv", "name": "No Server", "type": "mcp", "tool": "fetch"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'server' is required" in e for e in errors)

    def test_mcp_missing_tool(self):
        data = {"id": "no-tool", "name": "No Tool", "type": "mcp", "server": "srv"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'tool' is required" in e for e in errors)

    def test_inline_missing_content(self):
        data = {"id": "no-content", "name": "No Content", "type": "inline"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'content' is required" in e for e in errors)

    def test_tags_not_list(self):
        data = {"id": "bad-tags", "name": "Bad Tags", "type": "local", "path": "/tmp", "tags": "not-a-list"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'tags' must be an array" in e for e in errors)

    def test_tags_not_strings(self):
        data = {"id": "bad-tags", "name": "Bad Tags", "type": "local", "path": "/tmp", "tags": ["ok", 123]}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'tags' must contain only strings" in e for e in errors)

    def test_not_a_dict(self):
        is_valid, errors, warnings = validate_reference_file([1, 2, 3])
        assert is_valid is False
        assert any("JSON object" in e for e in errors)

    def test_warning_nonexistent_absolute_path(self):
        data = {
            "id": "missing-path",
            "name": "Missing Path",
            "type": "local",
            "path": "/nonexistent/absolutely/fake/path/12345",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True  # Valid structure, just a warning
        assert len(warnings) == 1
        assert "does not exist" in warnings[0]

    def test_no_warning_for_relative_path(self):
        """Relative paths are not checked for existence (resolved later)."""
        data = {
            "id": "rel-path",
            "name": "Relative Path",
            "type": "local",
            "path": "relative/path/to/docs",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert warnings == []

    def test_defaults_for_missing_optional_fields(self):
        """Minimal valid reference uses defaults for type and mode."""
        data = {"id": "minimal", "name": "Minimal", "path": "/tmp"}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True  # type defaults to "local", mode defaults to "selectable"

    def test_multiple_errors(self):
        """Multiple errors are all reported."""
        data = {"type": "ftp", "mode": "manual", "tags": 42}
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert len(errors) >= 4  # id, name, type, mode, tags

    # ==================== contents validation ====================

    def test_valid_contents_all_set(self):
        """Valid contents with all subfolder types declared."""
        data = {
            "id": "mod-001",
            "name": "Module 001",
            "type": "local",
            "path": "/tmp/mod-001",
            "contents": {
                "templates": "templates/",
                "validation": "validation/",
                "policies": "policies/",
                "scripts": "scripts/",
            },
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    def test_valid_contents_sparse(self):
        """Valid contents with only some types declared."""
        data = {
            "id": "mod-002",
            "name": "Module 002",
            "type": "local",
            "path": "/tmp/mod-002",
            "contents": {
                "templates": "templates/",
                "validation": None,
                "policies": None,
                "scripts": None,
            },
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    def test_contents_not_object(self):
        """contents must be an object, not a string."""
        data = {
            "id": "bad-contents",
            "name": "Bad Contents",
            "type": "local",
            "path": "/tmp",
            "contents": "templates/",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'contents' must be an object" in e for e in errors)

    def test_contents_value_not_string(self):
        """contents values must be strings or null."""
        data = {
            "id": "bad-val",
            "name": "Bad Value",
            "type": "local",
            "path": "/tmp",
            "contents": {
                "templates": 42,
                "validation": None,
                "policies": None,
                "scripts": None,
            },
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'contents.templates' must be a string" in e for e in errors)

    def test_contents_unknown_keys_warning(self):
        """Unknown keys in contents produce a warning, not an error."""
        data = {
            "id": "extra-keys",
            "name": "Extra Keys",
            "type": "local",
            "path": "/tmp",
            "contents": {
                "templates": "templates/",
                "validation": None,
                "policies": None,
                "scripts": None,
                "examples": "examples/",
            },
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert any("unknown keys" in w for w in warnings)

    def test_no_contents_is_valid(self):
        """Omitting contents entirely is valid."""
        data = {
            "id": "no-contents",
            "name": "No Contents",
            "type": "local",
            "path": "/tmp",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    # ==================== embedding validation ====================

    def test_valid_embedding(self):
        """Valid embedding with index and source_hash."""
        data = {
            "id": "emb-ref",
            "name": "Embedded Ref",
            "type": "local",
            "path": "/tmp",
            "embedding": {
                "index": 0,
                "source_hash": "sha256:a1b2c3d4e5f6",
            },
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []
        assert warnings == []

    def test_valid_embedding_higher_index(self):
        """Embedding with a non-zero index is valid."""
        data = {
            "id": "emb-ref-2",
            "name": "Embedded Ref 2",
            "type": "local",
            "path": "/tmp/docs",
            "embedding": {"index": 42, "source_hash": "sha256:deadbeef"},
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    def test_no_embedding_is_valid(self):
        """Omitting embedding entirely is valid (not all refs are embedded)."""
        data = {
            "id": "no-emb",
            "name": "No Embedding",
            "type": "local",
            "path": "/tmp",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []

    def test_embedding_not_object(self):
        """embedding must be an object, not a string."""
        data = {
            "id": "bad-emb",
            "name": "Bad Embedding",
            "type": "local",
            "path": "/tmp",
            "embedding": "not-an-object",
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'embedding' must be an object" in e for e in errors)

    def test_embedding_missing_index(self):
        """embedding.index is required."""
        data = {
            "id": "no-idx",
            "name": "No Index",
            "type": "local",
            "path": "/tmp",
            "embedding": {"source_hash": "sha256:abc123"},
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'embedding.index' is required" in e for e in errors)

    def test_embedding_missing_source_hash(self):
        """embedding.source_hash is required."""
        data = {
            "id": "no-hash",
            "name": "No Hash",
            "type": "local",
            "path": "/tmp",
            "embedding": {"index": 0},
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'embedding.source_hash' is required" in e for e in errors)

    def test_embedding_index_not_integer(self):
        """embedding.index must be an integer."""
        data = {
            "id": "bad-idx",
            "name": "Bad Index",
            "type": "local",
            "path": "/tmp",
            "embedding": {"index": "zero", "source_hash": "sha256:abc"},
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'embedding.index' must be an integer" in e for e in errors)

    def test_embedding_index_negative(self):
        """embedding.index must be non-negative."""
        data = {
            "id": "neg-idx",
            "name": "Negative Index",
            "type": "local",
            "path": "/tmp",
            "embedding": {"index": -1, "source_hash": "sha256:abc"},
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'embedding.index' must be non-negative" in e for e in errors)

    def test_embedding_source_hash_not_string(self):
        """embedding.source_hash must be a string."""
        data = {
            "id": "bad-hash",
            "name": "Bad Hash",
            "type": "local",
            "path": "/tmp",
            "embedding": {"index": 0, "source_hash": 12345},
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is False
        assert any("'embedding.source_hash' must be a string" in e for e in errors)

    def test_embedding_source_hash_missing_prefix_warning(self):
        """source_hash without sha256: prefix produces a warning."""
        data = {
            "id": "no-prefix",
            "name": "No Prefix",
            "type": "local",
            "path": "/tmp",
            "embedding": {"index": 0, "source_hash": "a1b2c3d4"},
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True  # Warning, not error
        assert any("sha256:" in w for w in warnings)

    def test_embedding_with_contents(self):
        """Both embedding and contents can coexist."""
        data = {
            "id": "full-ref",
            "name": "Full Reference",
            "type": "local",
            "path": "/tmp/mod",
            "contents": {
                "templates": "templates/",
                "validation": None,
                "policies": None,
                "scripts": None,
            },
            "embedding": {
                "index": 5,
                "source_hash": "sha256:deadbeefcafe",
            },
        }
        is_valid, errors, warnings = validate_reference_file(data)
        assert is_valid is True
        assert errors == []
