"""Tests for webhook config loading, merging, and validation."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from shared.plugins.webhook.config import (
    RouteConfig,
    TLSConfig,
    WebhookConfig,
    _deep_merge,
    load_config,
    validate_config,
)
from shared.plugins.subagent.config import expand_variables


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
        # No auto-created default route: a zero-config route would be an open,
        # unauthenticated ingestion endpoint. Empty means the listener 404s
        # everything until the operator declares routes.
        assert config.routes == {}
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

    def test_from_dict_empty_routes_stays_empty(self):
        # Fail-closed: no auto default route (was the fail-open bug).
        config = WebhookConfig.from_dict({})
        assert config.routes == {}

    def test_route_allow_unauthenticated_parses(self):
        config = WebhookConfig.from_dict({
            "routes": {"open": {"path": "/hook", "allow_unauthenticated": True}},
        })
        assert config.routes["open"].allow_unauthenticated is True
        # Defaults to False (fail-closed) when omitted.
        config2 = WebhookConfig.from_dict({
            "routes": {"g": {"path": "/g"}},
        })
        assert config2.routes["g"].allow_unauthenticated is False

    def test_allow_unauthenticated_string_false_fails_closed(self):
        # A quoted/expanded string must NOT enable via Python truthiness
        # (bool("false") is True). Only affirmative tokens count.
        for falsey in ("false", "False", "", "0", "no", "off", "nope"):
            config = WebhookConfig.from_dict({
                "routes": {"g": {"path": "/g", "allow_unauthenticated": falsey}},
            })
            assert config.routes["g"].allow_unauthenticated is False, falsey
        for truthy in ("true", "True", "1", "yes", "on"):
            config = WebhookConfig.from_dict({
                "routes": {"g": {"path": "/g", "allow_unauthenticated": truthy}},
            })
            assert config.routes["g"].allow_unauthenticated is True, truthy


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
    """Tests for expand_variables (shared utility from subagent.config).

    Webhook config uses expand_variables for ${VAR} expansion.
    These tests verify the integration works for webhook-relevant cases.
    Exhaustive expand_variables tests live in subagent/tests/test_config.py.
    """

    def test_expand_string(self):
        with mock.patch.dict(os.environ, {"MY_SECRET": "s3cret"}):
            assert expand_variables("${MY_SECRET}") == "s3cret"

    def test_expand_in_dict(self):
        with mock.patch.dict(os.environ, {"PORT": "8080"}):
            result = expand_variables({"port": "${PORT}"})
            assert result == {"port": "8080"}

    def test_expand_in_list(self):
        with mock.patch.dict(os.environ, {"X": "hello"}):
            result = expand_variables(["${X}", "world"])
            assert result == ["hello", "world"]

    def test_no_expansion_needed(self):
        assert expand_variables("plain text") == "plain text"

    def test_unknown_var_kept(self):
        result = expand_variables("${UNLIKELY_VAR_NAME_12345}")
        assert result == "${UNLIKELY_VAR_NAME_12345}"

    def test_non_string_passthrough(self):
        assert expand_variables(42) == 42
        assert expand_variables(True) is True
        assert expand_variables(None) is None


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

    def test_valid_tls_config(self):
        is_valid, errors = validate_config({
            "tls": {
                "enabled": True,
                "certfile": "/path/to/cert.pem",
                "keyfile": "/path/to/key.pem",
            },
        })
        assert is_valid

    def test_tls_missing_certfile(self):
        is_valid, errors = validate_config({
            "tls": {"enabled": True, "keyfile": "/key.pem"},
        })
        assert not is_valid
        assert any("certfile" in e for e in errors)

    def test_tls_missing_keyfile(self):
        is_valid, errors = validate_config({
            "tls": {"enabled": True, "certfile": "/cert.pem"},
        })
        assert not is_valid
        assert any("keyfile" in e for e in errors)

    def test_tls_disabled_no_files_required(self):
        is_valid, errors = validate_config({
            "tls": {"enabled": False},
        })
        assert is_valid

    def test_valid_allowed_ips(self):
        is_valid, errors = validate_config({
            "allowed_ips": ["192.168.1.0/24", "10.0.0.5", "::1"],
        })
        assert is_valid

    def test_invalid_allowed_ip(self):
        is_valid, errors = validate_config({
            "allowed_ips": ["not-an-ip"],
        })
        assert not is_valid
        assert any("not a valid IP" in e for e in errors)

    def test_valid_rate_limit(self):
        is_valid, errors = validate_config({"rate_limit_per_second": 10})
        assert is_valid

    def test_invalid_rate_limit_negative(self):
        is_valid, errors = validate_config({"rate_limit_per_second": -1})
        assert not is_valid


class TestTLSConfig:
    """Tests for TLSConfig dataclass."""

    def test_defaults(self):
        tls = TLSConfig()
        assert tls.enabled is False
        assert tls.certfile is None

    def test_from_dict(self):
        tls = TLSConfig.from_dict({
            "enabled": True,
            "certfile": "/cert.pem",
            "keyfile": "/key.pem",
            "ca_certfile": "/ca.pem",
        })
        assert tls.enabled is True
        assert tls.certfile == "/cert.pem"
        assert tls.ca_certfile == "/ca.pem"


class TestWebhookConfigSecurityFields:
    """Tests for security-related config fields."""

    def test_defaults_no_tls(self):
        config = WebhookConfig()
        assert config.tls.enabled is False
        assert config.allowed_ips == []
        assert config.rate_limit_per_second == 0

    def test_from_dict_with_tls(self):
        config = WebhookConfig.from_dict({
            "tls": {
                "enabled": True,
                "certfile": "/cert.pem",
                "keyfile": "/key.pem",
            },
        })
        assert config.tls.enabled is True
        assert config.tls.certfile == "/cert.pem"

    def test_from_dict_with_allowed_ips(self):
        config = WebhookConfig.from_dict({
            "allowed_ips": ["10.0.0.0/8"],
        })
        assert config.allowed_ips == ["10.0.0.0/8"]

    def test_from_dict_with_rate_limit(self):
        config = WebhookConfig.from_dict({
            "rate_limit_per_second": 50,
        })
        assert config.rate_limit_per_second == 50
