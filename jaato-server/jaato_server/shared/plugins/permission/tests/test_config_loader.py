"""Tests for the config_loader module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from ..config_loader import (
    PermissionConfig,
    ConfigValidationError,
    validate_config,
    load_config,
    create_default_config,
)


class TestPermissionConfig:
    """Tests for PermissionConfig dataclass."""

    def test_default_values(self):
        config = PermissionConfig()
        assert config.version == "1.0"
        assert config.default_policy == "deny"
        assert config.blacklist_tools == []
        assert config.blacklist_patterns == []
        assert config.blacklist_arguments == {}
        assert config.whitelist_tools == []
        assert config.whitelist_patterns == []
        assert config.whitelist_arguments == {}
        assert config.channel_type == "console"
        assert config.channel_endpoint is None
        assert config.channel_timeout == 30

    def test_custom_values(self):
        config = PermissionConfig(
            default_policy="allow",
            blacklist_tools=["dangerous"],
            whitelist_patterns=["git *"],
            channel_type="webhook",
            channel_endpoint="http://example.com",
            channel_timeout=60,
        )
        assert config.default_policy == "allow"
        assert config.blacklist_tools == ["dangerous"]
        assert config.whitelist_patterns == ["git *"]
        assert config.channel_type == "webhook"
        assert config.channel_endpoint == "http://example.com"
        assert config.channel_timeout == 60

    def test_to_policy_dict(self):
        config = PermissionConfig(
            default_policy="ask",
            blacklist_tools=["tool1"],
            blacklist_patterns=["rm *"],
            blacklist_arguments={"cli": {"cmd": ["sudo"]}},
            whitelist_tools=["tool2"],
            whitelist_patterns=["git *"],
            whitelist_arguments={"cli": {"cmd": ["npm"]}},
        )

        policy_dict = config.to_policy_dict()

        assert policy_dict["defaultPolicy"] == "ask"
        assert policy_dict["blacklist"]["tools"] == ["tool1"]
        assert policy_dict["blacklist"]["patterns"] == ["rm *"]
        assert policy_dict["blacklist"]["arguments"] == {"cli": {"cmd": ["sudo"]}}
        assert policy_dict["whitelist"]["tools"] == ["tool2"]
        assert policy_dict["whitelist"]["patterns"] == ["git *"]
        assert policy_dict["whitelist"]["arguments"] == {"cli": {"cmd": ["npm"]}}


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_empty_config_valid(self):
        is_valid, errors = validate_config({})
        # Filter out warnings
        actual_errors = [e for e in errors if not e.startswith("Warning:")]
        assert is_valid or len(actual_errors) == 0

    def test_valid_full_config(self):
        config = {
            "version": "1.0",
            "defaultPolicy": "ask",
            "blacklist": {
                "tools": ["dangerous"],
                "patterns": ["rm *"],
                "arguments": {
                    "cli_based_tool": {"command": ["sudo"]}
                }
            },
            "whitelist": {
                "tools": ["safe"],
                "patterns": ["git *"],
                "arguments": {}
            },
            "channel": {
                "type": "console",
                "timeout": 30
            }
        }
        is_valid, errors = validate_config(config)
        actual_errors = [e for e in errors if not e.startswith("Warning:")]
        assert is_valid or len(actual_errors) == 0

    def test_invalid_version(self):
        config = {"version": "2.0"}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("version" in e.lower() for e in errors)

    def test_invalid_default_policy(self):
        config = {"defaultPolicy": "maybe"}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("defaultPolicy" in e for e in errors)

    def test_blacklist_not_object(self):
        config = {"blacklist": "invalid"}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("blacklist" in e for e in errors)

    def test_blacklist_tools_not_array(self):
        config = {"blacklist": {"tools": "not-array"}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("tools" in e and "array" in e for e in errors)

    def test_blacklist_tools_non_strings(self):
        config = {"blacklist": {"tools": [123, "valid"]}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("tools" in e and "strings" in e for e in errors)

    def test_blacklist_patterns_not_array(self):
        config = {"blacklist": {"patterns": {"not": "array"}}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("patterns" in e and "array" in e for e in errors)

    def test_blacklist_arguments_not_object(self):
        config = {"blacklist": {"arguments": ["invalid"]}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("arguments" in e and "object" in e for e in errors)

    def test_blacklist_arguments_tool_not_object(self):
        config = {"blacklist": {"arguments": {"tool": "invalid"}}}
        is_valid, errors = validate_config(config)
        assert not is_valid

    def test_blacklist_arguments_values_not_array(self):
        config = {"blacklist": {"arguments": {"tool": {"arg": "not-array"}}}}
        is_valid, errors = validate_config(config)
        assert not is_valid

    def test_whitelist_validation(self):
        config = {"whitelist": {"tools": 123}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("whitelist" in e for e in errors)

    def test_invalid_channel_type(self):
        config = {"channel": {"type": "invalid"}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("channel type" in e.lower() for e in errors)

    def test_webhook_requires_endpoint(self):
        config = {"channel": {"type": "webhook"}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("endpoint" in e.lower() for e in errors)

    def test_invalid_timeout(self):
        config = {"channel": {"timeout": -5}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("timeout" in e.lower() for e in errors)

    def test_invalid_timeout_type(self):
        config = {"channel": {"timeout": "thirty"}}
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("timeout" in e.lower() for e in errors)

    def test_conflict_warning(self):
        config = {
            "blacklist": {"tools": ["conflict_tool"]},
            "whitelist": {"tools": ["conflict_tool"]}
        }
        is_valid, errors = validate_config(config)
        # Should have a warning about conflict
        assert any("Warning" in e and "conflict_tool" in e for e in errors)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_path(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "version": "1.0",
                "defaultPolicy": "allow",
                "blacklist": {"tools": ["blocked"]},
                "whitelist": {"patterns": ["git *"]},
            }
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.default_policy == "allow"
                assert config.blacklist_tools == ["blocked"]
                assert config.whitelist_patterns == ["git *"]
            finally:
                os.unlink(f.name)

    def test_load_from_env_var(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"defaultPolicy": "ask"}
            json.dump(config_data, f)
            f.flush()

            try:
                os.environ["TEST_PERMISSION_CONFIG"] = f.name
                config = load_config(env_var="TEST_PERMISSION_CONFIG")
                assert config.default_policy == "ask"
            finally:
                os.unlink(f.name)
                del os.environ["TEST_PERMISSION_CONFIG"]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {")
            f.flush()

            try:
                with pytest.raises(json.JSONDecodeError):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_validation_error(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"defaultPolicy": "invalid_policy"}
            json.dump(config_data, f)
            f.flush()

            try:
                with pytest.raises(ConfigValidationError) as exc_info:
                    load_config(f.name)
                assert "defaultPolicy" in str(exc_info.value)
            finally:
                os.unlink(f.name)

    def test_returns_default_when_no_file(self):
        # Clear env var if set
        if "PERMISSION_CONFIG_PATH" in os.environ:
            del os.environ["PERMISSION_CONFIG_PATH"]

        # Use a non-existent directory as CWD context
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                config = load_config()
                assert isinstance(config, PermissionConfig)
                assert config.default_policy == "deny"  # Default
            finally:
                os.chdir(original_cwd)

    def test_load_with_channel_config(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "channel": {
                    "type": "webhook",
                    "endpoint": "http://example.com/hook",
                    "timeout": 60
                }
            }
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.channel_type == "webhook"
                assert config.channel_endpoint == "http://example.com/hook"
                assert config.channel_timeout == 60
            finally:
                os.unlink(f.name)

    def test_load_with_arguments(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "blacklist": {
                    "arguments": {
                        "cli_based_tool": {"command": ["sudo", "rm -rf"]}
                    }
                },
                "whitelist": {
                    "arguments": {
                        "cli_based_tool": {"command": ["git", "npm"]}
                    }
                }
            }
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert "cli_based_tool" in config.blacklist_arguments
                assert "sudo" in config.blacklist_arguments["cli_based_tool"]["command"]
                assert "cli_based_tool" in config.whitelist_arguments
                assert "git" in config.whitelist_arguments["cli_based_tool"]["command"]
            finally:
                os.unlink(f.name)


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "permissions.json")
            create_default_config(config_path)

            assert os.path.exists(config_path)

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "subdir", "deep", "permissions.json")
            create_default_config(config_path)

            assert os.path.exists(config_path)

    def test_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "permissions.json")
            create_default_config(config_path)

            with open(config_path) as f:
                config = json.load(f)

            assert "version" in config
            assert "defaultPolicy" in config
            assert "blacklist" in config
            assert "whitelist" in config

    def test_created_config_is_loadable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "permissions.json")
            create_default_config(config_path)

            config = load_config(config_path)
            assert isinstance(config, PermissionConfig)

    def test_default_config_has_sensible_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "permissions.json")
            create_default_config(config_path)

            with open(config_path) as f:
                config = json.load(f)

            # Should have some default blacklist patterns
            assert len(config.get("blacklist", {}).get("patterns", [])) > 0

            # Should have some default whitelist patterns
            assert len(config.get("whitelist", {}).get("patterns", [])) > 0

            # Default policy should be ask
            assert config.get("defaultPolicy") == "ask"


class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_error_message(self):
        errors = ["Error 1", "Error 2"]
        exc = ConfigValidationError(errors)
        assert "Error 1" in str(exc)
        assert "Error 2" in str(exc)

    def test_errors_attribute(self):
        errors = ["Error 1", "Error 2"]
        exc = ConfigValidationError(errors)
        assert exc.errors == errors
