"""Tests for webhook config loading, merging, and validation."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from shared.plugins.webhook.config import (
    RouteConfig,
    WebhookConfig,
    _deep_merge,
    _expand_env_vars,
    load_config,
    validate_config,
)


class TestRouteConfig:
    """Tests for RouteConfig dataclass."""

    def test_from_dict_minimal(self):
        route = RouteConfig.from_dict({"path": "/webhook"})
        assert route.path == "/webhook"
        assert route.secret_header is None
        assert route.secret_algo is None
        assert route.event_type_header is None
        assert route.metadata == {}

    def test_from_dict_full(self):
        route = RouteConfig.from_dict({
            "path": "/webhook/github",
            "secret_header": "X-Hub-Signature-256",
            "secret_algo": "hmac-sha256",
            "event_type_header": "X-GitHub-Event",
            "metadata": {"source": "github"},
        })
        assert route.path == "/webhook/github"
        assert route.secret_header == "X-Hub-Signature-256"
        assert route.secret_algo == "hmac-sha256"
        assert route.event_type_header == "X-GitHub-Event"
        assert route.metadata == {"source": "github"}


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_defaults(self):
        config = WebhookConfig()
        assert config.port == 9100
        assert config.host == "127.0.0.1"
        assert config.secret is None
        assert "generic" in config.routes
        assert config.routes["generic"].path == "/webhook"
        assert config.max_body_size == 1048576
        assert config.response_timeout == 5.0

    def test_from_dict(self):
        config = WebhookConfig.from_dict({
            "port": 8080,
            "host": "0.0.0.0",
            "secret": "mysecret",
            "routes": {
                "github": {"path": "/webhook/github"},
            },
            "max_body_size": 2097152,
            "response_timeout": 10.0,
        })
        assert config.port == 8080
        assert config.host == "0.0.0.0"
        assert config.secret == "mysecret"
        assert "github" in config.routes
        assert config.routes["github"].path == "/webhook/github"
        assert config.max_body_size == 2097152
        assert config.response_timeout == 10.0

    def test_from_dict_empty_routes_gets_default(self):
        config = WebhookConfig.from_dict({})
        assert "generic" in config.routes


class TestDeepMerge:
    """Tests for _deep_merge helper."""

    def test_simple_override(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_add_key(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_merge(self):
        result = _deep_merge(
            {"routes": {"github": {"path": "/gh"}}},
            {"routes": {"slack": {"path": "/slack"}}},
        )
        assert result == {
            "routes": {
                "github": {"path": "/gh"},
                "slack": {"path": "/slack"},
            }
        }

    def test_nested_override(self):
        result = _deep_merge(
            {"routes": {"github": {"path": "/old"}}},
            {"routes": {"github": {"path": "/new"}}},
        )
        assert result["routes"]["github"]["path"] == "/new"

    def test_does_not_mutate_base(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert "b" not in base


class TestExpandEnvVars:
    """Tests for _expand_env_vars helper."""

    def test_expand_string(self):
        with mock.patch.dict(os.environ, {"MY_SECRET": "s3cret"}):
            assert _expand_env_vars("${MY_SECRET}") == "s3cret"

    def test_expand_in_dict(self):
        with mock.patch.dict(os.environ, {"PORT": "8080"}):
            result = _expand_env_vars({"port": "${PORT}"})
            assert result == {"port": "8080"}

    def test_expand_in_list(self):
        with mock.patch.dict(os.environ, {"X": "hello"}):
            result = _expand_env_vars(["${X}", "world"])
            assert result == ["hello", "world"]

    def test_no_expansion_needed(self):
        assert _expand_env_vars("plain text") == "plain text"

    def test_unknown_var_kept(self):
        result = _expand_env_vars("${UNLIKELY_VAR_NAME_12345}")
        assert result == "${UNLIKELY_VAR_NAME_12345}"

    def test_non_string_passthrough(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None


class TestLoadConfig:
    """Tests for load_config with precedence merging."""

    def test_defaults_when_no_files(self):
        config = load_config()
        assert config.port == 9100
        assert config.host == "127.0.0.1"

    def test_explicit_config_wins(self):
        config = load_config(explicit_config={"port": 7777})
        assert config.port == 7777

    def test_workspace_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jaato_dir = Path(tmpdir) / ".jaato"
            jaato_dir.mkdir()
            config_file = jaato_dir / "webhook.json"
            config_file.write_text(json.dumps({"port": 8888}))

            config = load_config(workspace_path=tmpdir)
            assert config.port == 8888

    def test_explicit_overrides_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jaato_dir = Path(tmpdir) / ".jaato"
            jaato_dir.mkdir()
            config_file = jaato_dir / "webhook.json"
            config_file.write_text(json.dumps({"port": 8888, "host": "0.0.0.0"}))

            config = load_config(
                explicit_config={"port": 7777},
                workspace_path=tmpdir,
            )
            # Explicit port wins, workspace host is inherited
            assert config.port == 7777
            assert config.host == "0.0.0.0"

    def test_routes_merged_across_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jaato_dir = Path(tmpdir) / ".jaato"
            jaato_dir.mkdir()
            config_file = jaato_dir / "webhook.json"
            config_file.write_text(json.dumps({
                "routes": {"github": {"path": "/webhook/github"}},
            }))

            config = load_config(
                explicit_config={
                    "routes": {"slack": {"path": "/webhook/slack"}},
                },
                workspace_path=tmpdir,
            )
            assert "github" in config.routes
            assert "slack" in config.routes


class TestValidateConfig:
    """Tests for validate_config."""

    def test_valid_minimal(self):
        is_valid, errors = validate_config({})
        assert is_valid
        assert errors == []

    def test_valid_full(self):
        is_valid, errors = validate_config({
            "port": 9100,
            "host": "127.0.0.1",
            "routes": {
                "github": {
                    "path": "/webhook/github",
                    "secret_header": "X-Hub-Signature-256",
                    "secret_algo": "hmac-sha256",
                },
            },
        })
        assert is_valid

    def test_invalid_port_type(self):
        is_valid, errors = validate_config({"port": "not_a_number"})
        assert not is_valid
        assert any("port" in e for e in errors)

    def test_invalid_port_range(self):
        is_valid, errors = validate_config({"port": 99999})
        assert not is_valid

    def test_invalid_route_path_no_slash(self):
        is_valid, errors = validate_config({
            "routes": {"bad": {"path": "no-slash"}},
        })
        assert not is_valid
        assert any("start with '/'" in e for e in errors)

    def test_invalid_secret_algo(self):
        is_valid, errors = validate_config({
            "routes": {"x": {"path": "/x", "secret_algo": "hmac-md5"}},
        })
        assert not is_valid
        assert any("hmac-sha256" in e for e in errors)

    def test_invalid_max_body_size(self):
        is_valid, errors = validate_config({"max_body_size": -1})
        assert not is_valid

    def test_invalid_response_timeout(self):
        is_valid, errors = validate_config({"response_timeout": 0})
        assert not is_valid
