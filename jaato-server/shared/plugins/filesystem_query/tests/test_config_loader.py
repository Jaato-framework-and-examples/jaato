"""Tests for the filesystem_query config loader."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from ..config_loader import (
    FilesystemQueryConfig,
    ConfigValidationError,
    load_config,
    validate_config,
    create_default_config,
    get_default_excludes,
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_MAX_RESULTS,
)


class TestFilesystemQueryConfig:
    """Tests for FilesystemQueryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FilesystemQueryConfig()

        assert config.version == "1.0"
        assert config.exclude_patterns == []
        assert config.exclude_mode == "extend"
        assert config.include_patterns == []
        assert config.max_results == DEFAULT_MAX_RESULTS
        assert config.max_file_size_kb == 1024
        assert config.timeout_seconds == 30
        assert config.context_lines == 2

    def test_get_effective_excludes_extend_mode(self):
        """Test effective excludes in extend mode."""
        config = FilesystemQueryConfig(
            exclude_patterns=["custom_dir", "vendor"],
            exclude_mode="extend",
        )

        effective = config.get_effective_excludes()

        # Should include defaults
        assert ".git" in effective
        assert "node_modules" in effective
        # Should include custom patterns
        assert "custom_dir" in effective
        assert "vendor" in effective

    def test_get_effective_excludes_replace_mode(self):
        """Test effective excludes in replace mode."""
        config = FilesystemQueryConfig(
            exclude_patterns=["only_this", "and_this"],
            exclude_mode="replace",
        )

        effective = config.get_effective_excludes()

        # Should NOT include defaults
        assert ".git" not in effective
        assert "node_modules" not in effective
        # Should only include custom patterns
        assert effective == ["and_this", "only_this"]

    def test_should_include(self):
        """Test force-include pattern matching."""
        config = FilesystemQueryConfig(
            include_patterns=["node_modules/@company", "vendor/internal"],
        )

        assert config.should_include("node_modules/@company/lib") is True
        assert config.should_include("vendor/internal/util") is True
        assert config.should_include("node_modules/lodash") is False
        assert config.should_include("src/main.py") is False

    def test_should_exclude_basic(self):
        """Test basic exclusion matching."""
        config = FilesystemQueryConfig()

        assert config.should_exclude("node_modules/lodash/index.js") is True
        assert config.should_exclude("__pycache__/module.cpython-310.pyc") is True
        assert config.should_exclude("src/main.py") is False

    def test_should_exclude_with_include_override(self):
        """Test that include patterns override exclusions."""
        config = FilesystemQueryConfig(
            include_patterns=["node_modules/@company"],
        )

        # Normally excluded, but force-included
        assert config.should_exclude("node_modules/@company/lib.js") is False
        # Still excluded (not in include patterns)
        assert config.should_exclude("node_modules/lodash/index.js") is True


class TestValidateConfig:
    """Tests for config validation."""

    def test_valid_config(self):
        """Test validation of a valid config."""
        config = {
            "version": "1.0",
            "exclude_patterns": [".cache"],
            "exclude_mode": "extend",
            "include_patterns": [],
            "max_results": 100,
        }

        is_valid, errors = validate_config(config)
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_version(self):
        """Test validation fails for invalid version."""
        config = {"version": "2.0"}

        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any("version" in e for e in errors)

    def test_invalid_exclude_patterns(self):
        """Test validation fails for invalid exclude_patterns."""
        config = {"exclude_patterns": "not_a_list"}

        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any("exclude_patterns" in e for e in errors)

    def test_invalid_exclude_mode(self):
        """Test validation fails for invalid exclude_mode."""
        config = {"exclude_mode": "invalid"}

        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any("exclude_mode" in e for e in errors)

    def test_invalid_max_results(self):
        """Test validation fails for invalid max_results."""
        config = {"max_results": -1}

        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any("max_results" in e for e in errors)

    def test_invalid_max_file_size_kb(self):
        """Test validation fails for invalid max_file_size_kb."""
        config = {"max_file_size_kb": "not_an_int"}

        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any("max_file_size_kb" in e for e in errors)

    def test_invalid_context_lines(self):
        """Test validation fails for negative context_lines."""
        config = {"context_lines": -1}

        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any("context_lines" in e for e in errors)


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_default_config(self):
        """Test loading default config when no file exists."""
        config = load_config()

        assert config.max_results == DEFAULT_MAX_RESULTS
        assert config.exclude_mode == "extend"

    def test_load_from_file(self):
        """Test loading config from a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({
                "version": "1.0",
                "max_results": 200,
                "exclude_patterns": ["custom"],
            }))

            config = load_config(path=str(config_path))

            assert config.max_results == 200
            assert "custom" in config.exclude_patterns

    def test_load_with_runtime_override(self):
        """Test runtime config overrides file config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({
                "max_results": 200,
                "timeout_seconds": 45,
            }))

            config = load_config(
                path=str(config_path),
                runtime_config={"max_results": 100},
            )

            # Runtime overrides file
            assert config.max_results == 100
            # File config still applies where not overridden
            assert config.timeout_seconds == 45

    def test_load_from_env_var(self):
        """Test loading config from environment variable path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({
                "max_results": 300,
            }))

            env_var = "TEST_FILESYSTEM_QUERY_CONFIG"
            os.environ[env_var] = str(config_path)

            try:
                config = load_config(env_var=env_var)
                assert config.max_results == 300
            finally:
                del os.environ[env_var]

    def test_load_from_default_location(self):
        """Test loading config from default .jaato location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .jaato directory with config
            jaato_dir = Path(tmpdir) / ".jaato"
            jaato_dir.mkdir()
            config_path = jaato_dir / "filesystem_query.json"
            config_path.write_text(json.dumps({
                "max_results": 400,
            }))

            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                config = load_config()
                assert config.max_results == 400
            finally:
                os.chdir(original_cwd)

    def test_load_nonexistent_explicit_path_raises(self):
        """Test that explicit nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(path="/nonexistent/config.json")

    def test_load_invalid_config_raises(self):
        """Test that invalid config raises validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({
                "exclude_mode": "invalid_mode",
            }))

            with pytest.raises(ConfigValidationError):
                load_config(path=str(config_path))


class TestCreateDefaultConfig:
    """Tests for creating default config file."""

    def test_create_default_config(self):
        """Test creating a default config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            create_default_config(str(config_path))

            assert config_path.exists()

            content = json.loads(config_path.read_text())
            assert content["version"] == "1.0"
            assert content["exclude_mode"] == "extend"
            assert content["max_results"] == DEFAULT_MAX_RESULTS

    def test_create_default_config_creates_parent_dirs(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nested" / "dir" / "config.json"

            create_default_config(str(config_path))

            assert config_path.exists()


class TestGetDefaultExcludes:
    """Tests for get_default_excludes function."""

    def test_returns_copy(self):
        """Test that a copy is returned, not the original."""
        excludes = get_default_excludes()
        original_len = len(excludes)

        excludes.append("new_pattern")

        # Original should be unchanged
        assert len(get_default_excludes()) == original_len

    def test_contains_expected_patterns(self):
        """Test that expected patterns are included."""
        excludes = get_default_excludes()

        assert ".git" in excludes
        assert "node_modules" in excludes
        assert "__pycache__" in excludes
        assert ".venv" in excludes
