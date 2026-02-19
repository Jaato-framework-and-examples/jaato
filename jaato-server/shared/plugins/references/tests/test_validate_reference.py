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
