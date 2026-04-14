"""Tests for the LM Studio provider.

Covers:
- Environment resolution (host default, context override, optional token)
- Passive mode: no ``load`` in config → no POST to /api/v1/models/load
- Active mode: ``load`` present → POST with expected body
- Connectivity check failures surface as LMStudioConnectionError
- Unknown model surfaces as LMStudioModelNotFoundError with catalog
- Load failure (non-2xx) surfaces as LMStudioLoadError
- ``get_context_limit()`` priority: override > discovered > default
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.lmstudio.env import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_HOST,
    resolve_api_token,
    resolve_context_length,
    resolve_host,
)
from shared.plugins.model_provider.lmstudio.errors import (
    LMStudioConnectionError,
    LMStudioLoadError,
    LMStudioModelNotFoundError,
)
from shared.plugins.model_provider.lmstudio.provider import (
    LMStudioProvider,
    create_provider,
)


# ============================================================
# Fixtures
# ============================================================


def _catalog_response(model_id: str = "gpt-oss-20b", max_ctx: int = 16384):
    """Build a fake /api/v0/models response for one model."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "type": "llm",
                "state": "not-loaded",
                "max_context_length": max_ctx,
            }
        ],
    }
    return resp


def _load_response(status: int = 200, body: str = '{"status": "loaded"}'):
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    return resp


# ============================================================
# env.py
# ============================================================


class TestEnv:

    def test_host_default_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("LMSTUDIO_HOST", raising=False)
        assert resolve_host() == DEFAULT_HOST

    def test_host_from_env(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_HOST", "http://gpu-box:9000")
        assert resolve_host() == "http://gpu-box:9000"

    def test_context_length_parsed(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_CONTEXT_LENGTH", "32768")
        assert resolve_context_length() == 32768

    def test_context_length_invalid_is_none(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_CONTEXT_LENGTH", "not-a-number")
        assert resolve_context_length() is None

    def test_api_token_absent_returns_none(self, monkeypatch):
        monkeypatch.delenv("LMSTUDIO_API_TOKEN", raising=False)
        assert resolve_api_token() is None

    def test_api_token_from_env(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_API_TOKEN", "secret-123")
        assert resolve_api_token() == "secret-123"


# ============================================================
# create_provider factory
# ============================================================


class TestFactory:

    def test_create_provider_returns_instance(self):
        provider = create_provider()
        assert isinstance(provider, LMStudioProvider)
        assert provider.name == "lmstudio"
        assert provider.is_connected is False


# ============================================================
# initialize()
# ============================================================


class TestInitialize:

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_initialize_reads_host_from_extra(self, mock_get, monkeypatch):
        monkeypatch.delenv("LMSTUDIO_HOST", raising=False)
        mock_get.return_value = _catalog_response()

        provider = LMStudioProvider()
        provider.initialize(ProviderConfig(extra={"host": "http://my-rig:4200"}))

        assert provider._host == "http://my-rig:4200"
        # Trailing slash stripped
        assert not provider._host.endswith("/")

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_initialize_connection_failure_raises(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        provider = LMStudioProvider()
        with pytest.raises(LMStudioConnectionError):
            provider.initialize(ProviderConfig(extra={"host": "http://offline:1234"}))

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_initialize_rejects_non_dict_load(self, mock_get):
        mock_get.return_value = _catalog_response()
        provider = LMStudioProvider()
        with pytest.raises(ValueError, match="load config must be a dict"):
            provider.initialize(ProviderConfig(extra={"load": "not a dict"}))

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_initialize_stores_load_config(self, mock_get):
        mock_get.return_value = _catalog_response()
        provider = LMStudioProvider()
        provider.initialize(
            ProviderConfig(extra={"load": {"context_length": 8192}})
        )
        assert provider._load_config == {"context_length": 8192}

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_initialize_passive_mode_load_is_none(self, mock_get):
        mock_get.return_value = _catalog_response()
        provider = LMStudioProvider()
        provider.initialize(ProviderConfig())
        assert provider._load_config is None


# ============================================================
# connect() — passive mode
# ============================================================


class TestConnectPassive:

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.post")
    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_connect_without_load_does_not_post(self, mock_get, mock_post):
        mock_get.return_value = _catalog_response()
        provider = LMStudioProvider()
        provider.initialize(ProviderConfig())
        provider.connect("gpt-oss-20b")

        # The load endpoint must NOT be hit in passive mode
        mock_post.assert_not_called()
        assert provider.model_name == "gpt-oss-20b"

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_connect_records_discovered_context_length(self, mock_get):
        mock_get.return_value = _catalog_response(max_ctx=131072)
        provider = LMStudioProvider()
        provider.initialize(ProviderConfig())
        provider.connect("gpt-oss-20b")
        assert provider._discovered_context_length == 131072

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_connect_unknown_model_raises(self, mock_get):
        mock_get.return_value = _catalog_response(model_id="different-model")
        provider = LMStudioProvider()
        provider.initialize(ProviderConfig())
        with pytest.raises(LMStudioModelNotFoundError):
            provider.connect("gpt-oss-20b")


# ============================================================
# connect() — active mode (load-control)
# ============================================================


class TestConnectActive:

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.post")
    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_connect_with_load_posts_expected_body(self, mock_get, mock_post):
        mock_get.return_value = _catalog_response()
        mock_post.return_value = _load_response(status=200)

        provider = LMStudioProvider()
        provider.initialize(
            ProviderConfig(
                extra={
                    "load": {
                        "context_length": 16384,
                        "flash_attention": True,
                        "eval_batch_size": 512,
                    }
                }
            )
        )
        provider.connect("gpt-oss-20b")

        # POST happened once to the load endpoint with the expected URL
        assert mock_post.call_count == 1
        args, kwargs = mock_post.call_args
        assert args[0].endswith("/api/v1/models/load")
        body = kwargs["json"]
        assert body["model"] == "gpt-oss-20b"
        assert body["context_length"] == 16384
        assert body["flash_attention"] is True
        assert body["eval_batch_size"] == 512

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.post")
    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_connect_active_overrides_any_model_in_load_dict(self, mock_get, mock_post):
        """A stray ``model`` key in load config must be overridden by connect()'s arg."""
        mock_get.return_value = _catalog_response()
        mock_post.return_value = _load_response(status=200)

        provider = LMStudioProvider()
        provider.initialize(
            ProviderConfig(extra={"load": {"model": "wrong-model", "context_length": 4096}})
        )
        provider.connect("gpt-oss-20b")

        body = mock_post.call_args[1]["json"]
        assert body["model"] == "gpt-oss-20b"

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.post")
    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_connect_with_load_failure_raises(self, mock_get, mock_post):
        mock_get.return_value = _catalog_response()
        mock_post.return_value = _load_response(
            status=400, body='{"error": "context too large"}'
        )

        provider = LMStudioProvider()
        provider.initialize(ProviderConfig(extra={"load": {"context_length": 999_999}}))
        with pytest.raises(LMStudioLoadError) as excinfo:
            provider.connect("gpt-oss-20b")
        assert excinfo.value.status_code == 400
        assert excinfo.value.load_config == {"context_length": 999_999}

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.post")
    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_connect_active_sends_bearer_when_token_configured(self, mock_get, mock_post):
        mock_get.return_value = _catalog_response()
        mock_post.return_value = _load_response(status=200)

        provider = LMStudioProvider()
        provider.initialize(
            ProviderConfig(
                extra={
                    "api_token": "tok-abc",
                    "load": {"context_length": 8192},
                }
            )
        )
        provider.connect("gpt-oss-20b")

        post_headers = mock_post.call_args[1]["headers"]
        assert post_headers["Authorization"] == "Bearer tok-abc"


# ============================================================
# get_context_limit priority
# ============================================================


class TestContextLimit:

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_override_wins(self, mock_get):
        mock_get.return_value = _catalog_response(max_ctx=131072)
        provider = LMStudioProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 4096}))
        provider.connect("gpt-oss-20b")
        assert provider.get_context_limit() == 4096

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_discovered_used_when_no_override(self, mock_get):
        mock_get.return_value = _catalog_response(max_ctx=32768)
        provider = LMStudioProvider()
        provider.initialize(ProviderConfig())
        provider.connect("gpt-oss-20b")
        assert provider.get_context_limit() == 32768

    def test_default_used_before_connect(self):
        provider = LMStudioProvider()
        assert provider.get_context_limit() == DEFAULT_CONTEXT_LENGTH


# ============================================================
# list_models
# ============================================================


class TestListModels:

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_list_models_returns_ids(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "object": "list",
            "data": [
                {"id": "alpha", "max_context_length": 4096},
                {"id": "beta", "max_context_length": 8192},
            ],
        }
        mock_get.return_value = resp

        provider = LMStudioProvider()
        provider.initialize(ProviderConfig())
        names = provider.list_models()
        assert names == ["alpha", "beta"]

    @patch("shared.plugins.model_provider.lmstudio.provider.httpx.get")
    def test_list_models_prefix_filter(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "object": "list",
            "data": [
                {"id": "openai/gpt-oss-20b"},
                {"id": "meta/llama-3.1-8b"},
                {"id": "openai/gpt-oss-120b"},
            ],
        }
        mock_get.return_value = resp

        provider = LMStudioProvider()
        provider.initialize(ProviderConfig())
        names = provider.list_models(prefix="openai/")
        assert names == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
