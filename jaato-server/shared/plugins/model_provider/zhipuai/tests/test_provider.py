"""Tests for ZhipuAIProvider."""

import json
import pytest
from unittest.mock import MagicMock, patch

from ..provider import (
    ZhipuAIProvider,
    ZhipuAIAPIKeyNotFoundError,
    ZhipuAIConnectionError,
    DEFAULT_CONTEXT_LIMIT,
    MODEL_CONTEXT_LIMITS,
    KNOWN_MODELS,
    THINKING_CAPABLE_MODELS,
    _openai_models_url,
)
from ..env import DEFAULT_ZHIPUAI_BASE_URL
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import ThinkingConfig


class TestInitialization:
    """Tests for initialization."""

    @patch('anthropic.Anthropic')
    def test_initialize_with_api_key(self, mock_anthropic):
        """Should initialize with API key from config."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        assert provider._api_key == "test-key"
        mock_anthropic.assert_called_once()
        call_kwargs = mock_anthropic.call_args.kwargs
        assert call_kwargs["base_url"] == DEFAULT_ZHIPUAI_BASE_URL
        assert call_kwargs["api_key"] == "test-key"

    @patch('anthropic.Anthropic')
    @patch.dict('os.environ', {'ZHIPUAI_API_KEY': 'env-key'})
    def test_initialize_from_env(self, mock_anthropic):
        """Should use API key from environment."""
        provider = ZhipuAIProvider()
        provider.initialize()

        assert provider._api_key == "env-key"

    def test_initialize_no_api_key(self):
        """Should raise error when no API key found."""
        with patch.dict('os.environ', {}, clear=True):
            provider = ZhipuAIProvider()
            with pytest.raises(ZhipuAIAPIKeyNotFoundError) as exc_info:
                provider.initialize()

            assert "ZHIPUAI_API_KEY" in str(exc_info.value)
            assert "open.bigmodel.cn" in str(exc_info.value)

    @patch('anthropic.Anthropic')
    def test_initialize_custom_base_url(self, mock_anthropic):
        """Should use custom base URL from config."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            extra={"base_url": "https://custom.api.com"}
        ))

        assert provider._base_url == "https://custom.api.com"
        call_kwargs = mock_anthropic.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom.api.com"

    @patch('anthropic.Anthropic')
    @patch.dict('os.environ', {
        'ZHIPUAI_API_KEY': 'key',
        'ZHIPUAI_BASE_URL': 'https://env-url.com'
    })
    def test_initialize_base_url_from_env(self, mock_anthropic):
        """Should use base URL from environment."""
        provider = ZhipuAIProvider()
        provider.initialize()

        assert provider._base_url == "https://env-url.com"

    @patch('anthropic.Anthropic')
    def test_no_cache_plugin_by_default(self, mock_anthropic):
        """Cache plugin is not attached by default (wired by session)."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        # No cache plugin attached directly on provider
        assert provider._cache_plugin is None

    @patch('anthropic.Anthropic')
    def test_thinking_default_disabled(self, mock_anthropic):
        """Should have thinking disabled by default."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        assert provider._enable_thinking is False

    @patch('anthropic.Anthropic')
    def test_thinking_enabled_via_config(self, mock_anthropic):
        """Should allow enabling thinking via config."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            extra={"enable_thinking": True}
        ))

        assert provider._enable_thinking is True

    @patch('anthropic.Anthropic')
    def test_strips_trailing_slash_from_base_url(self, mock_anthropic):
        """Should strip trailing slash from base URL."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            extra={"base_url": "https://api.example.com/"}
        ))

        assert provider._base_url == "https://api.example.com"


class TestConnection:
    """Tests for model connection."""

    @patch('anthropic.Anthropic')
    def test_connect_sets_model(self, mock_anthropic):
        """Should set model name on connect."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4.7")

        assert provider.model_name == "glm-4.7"
        assert provider.is_connected is True

    @patch('anthropic.Anthropic')
    def test_connect_flash_model(self, mock_anthropic):
        """Should connect to flash model."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4.7-flash")

        assert provider.model_name == "glm-4.7-flash"


class TestModelListing:
    """Tests for model listing.

    ``list_models()`` first attempts dynamic discovery via Z.AI's
    OpenAI-compatible ``GET /models`` endpoint, then falls back to the
    static ``KNOWN_MODELS`` list.
    """

    @patch('anthropic.Anthropic')
    def test_list_models_fallback_to_static(self, mock_anthropic):
        """Should fall back to static list when remote fetch fails."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        with patch.object(provider, '_fetch_remote_models', return_value=[]):
            models = provider.list_models()

        assert len(models) == len(KNOWN_MODELS)
        assert "glm-5" in models
        assert "glm-4.7" in models
        assert "glm-4.7-flash" in models

    @patch('anthropic.Anthropic')
    def test_list_models_uses_remote(self, mock_anthropic):
        """Should use remote model list when available."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        remote = ["glm-5", "glm-4.7", "glm-new-model"]
        with patch.object(provider, '_fetch_remote_models', return_value=remote):
            models = provider.list_models()

        assert models == sorted(remote)
        assert "glm-new-model" in models

    @patch('anthropic.Anthropic')
    def test_list_models_with_prefix(self, mock_anthropic):
        """Should filter models by prefix."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        with patch.object(provider, '_fetch_remote_models', return_value=[]):
            models = provider.list_models(prefix="glm-4.7")

        assert len(models) == 3  # glm-4.7, glm-4.7-flash, glm-4.7-flashx
        assert all(m.startswith("glm-4.7") for m in models)

    @patch('anthropic.Anthropic')
    def test_list_models_prefix_on_remote(self, mock_anthropic):
        """Should apply prefix filter on dynamically fetched models."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        remote = ["glm-5", "glm-5-vision", "glm-4.7"]
        with patch.object(provider, '_fetch_remote_models', return_value=remote):
            models = provider.list_models(prefix="glm-5")

        assert models == ["glm-5", "glm-5-vision"]

    @patch('anthropic.Anthropic')
    def test_glm5_in_known_models(self, mock_anthropic):
        """GLM-5 should be present in the static model list."""
        assert "glm-5" in KNOWN_MODELS
        assert MODEL_CONTEXT_LIMITS["glm-5"] == 204800


class TestContextLimit:
    """Tests for context limit handling."""

    @patch('anthropic.Anthropic')
    def test_default_context_limit(self, mock_anthropic):
        """Should return fallback 128K when no model connected."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        assert provider.get_context_limit() == DEFAULT_CONTEXT_LIMIT
        assert provider.get_context_limit() == 131072

    @patch('anthropic.Anthropic')
    def test_model_specific_context_limit(self, mock_anthropic):
        """Should return per-model context limit after connect."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        provider.connect("glm-4.7")
        assert provider.get_context_limit() == 204800

        provider.connect("glm-4.5")
        assert provider.get_context_limit() == 131072

    @patch('anthropic.Anthropic')
    def test_custom_context_limit(self, mock_anthropic):
        """Should use custom context limit from config (overrides per-model)."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            extra={"context_length": 65536}
        ))
        provider.connect("glm-4.7")

        # Override takes precedence even though glm-4.7 is 200K
        assert provider.get_context_limit() == 65536


class TestVerifyAuth:
    """Tests for auth verification.

    verify_auth() must work BEFORE initialize() is called — it only checks
    whether credentials are available, not whether the client can connect.
    """

    @patch.dict('os.environ', {'ZHIPUAI_API_KEY': 'env-key'})
    def test_verify_auth_with_env_key(self):
        """Should return True when API key is in environment."""
        provider = ZhipuAIProvider()  # NOT initialized

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is True
        assert any("Found" in m for m in messages)

    @patch.dict('os.environ', {}, clear=True)
    @patch('shared.plugins.model_provider.zhipuai.provider.get_stored_api_key', return_value='stored-key')
    def test_verify_auth_with_stored_key(self, mock_stored):
        """Should return True when API key is stored."""
        provider = ZhipuAIProvider()  # NOT initialized

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is True

    @patch.dict('os.environ', {}, clear=True)
    @patch('shared.plugins.model_provider.zhipuai.provider.get_stored_api_key', return_value=None)
    def test_verify_auth_no_credentials(self, mock_stored):
        """Should return False when no credentials are available."""
        provider = ZhipuAIProvider()  # NOT initialized

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is False
        assert any("No" in m for m in messages)


class TestErrorHandling:
    """Tests for error handling."""

    @patch('anthropic.Anthropic')
    def test_handle_auth_error(self, mock_anthropic):
        """Should raise helpful error for authentication failures."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider._model_name = "glm-4.7"

        with pytest.raises(ZhipuAIConnectionError) as exc_info:
            provider._handle_api_error(Exception("401 Unauthorized"))

        assert "Invalid API key" in str(exc_info.value)

    @patch('anthropic.Anthropic')
    def test_handle_rate_limit_error(self, mock_anthropic):
        """Should raise helpful error for rate limiting."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider._model_name = "glm-4.7"

        with pytest.raises(RuntimeError) as exc_info:
            provider._handle_api_error(Exception("429 rate limit exceeded"))

        assert "rate limit" in str(exc_info.value).lower()

    @patch('anthropic.Anthropic')
    def test_handle_model_not_found(self, mock_anthropic):
        """Should raise helpful error when model not found."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider._model_name = "unknown-model"

        with pytest.raises(RuntimeError) as exc_info:
            provider._handle_api_error(Exception("404 model not found"))

        assert "not found" in str(exc_info.value).lower()
        assert "glm-4.7" in str(exc_info.value)


class TestProviderName:
    """Tests for provider identification."""

    def test_provider_name(self):
        """Should return 'zhipuai' as provider name."""
        provider = ZhipuAIProvider()
        assert provider.name == "zhipuai"


class TestLogin:
    """Tests for login method."""

    def test_login_provides_guidance(self):
        """Login should provide guidance for API key setup."""
        messages = []
        ZhipuAIProvider.login(on_message=messages.append)

        assert any("ZHIPUAI_API_KEY" in m for m in messages)
        assert any("open.bigmodel.cn" in m for m in messages)


class TestThinkingSupport:
    """Tests for extended thinking / chain-of-thought support."""

    @patch('anthropic.Anthropic')
    def test_thinking_capable_glm47(self, mock_anthropic):
        """GLM-4.7 should be thinking-capable."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4.7")

        assert provider._is_thinking_capable() is True
        assert provider.supports_thinking() is True

    @patch('anthropic.Anthropic')
    def test_thinking_capable_glm5(self, mock_anthropic):
        """GLM-5 should be thinking-capable."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-5")

        assert provider._is_thinking_capable() is True
        assert provider.supports_thinking() is True

    @patch('anthropic.Anthropic')
    def test_thinking_not_capable_flash(self, mock_anthropic):
        """GLM-4.7-flash should NOT be thinking-capable."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4.7-flash")

        assert provider._is_thinking_capable() is False
        assert provider.supports_thinking() is False

    @patch('anthropic.Anthropic')
    def test_thinking_not_capable_glm4(self, mock_anthropic):
        """GLM-4 should NOT be thinking-capable."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4")

        assert provider._is_thinking_capable() is False

    @patch('anthropic.Anthropic')
    def test_thinking_not_capable_glm4v(self, mock_anthropic):
        """GLM-4V should NOT be thinking-capable."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4v")

        assert provider._is_thinking_capable() is False

    @patch('anthropic.Anthropic')
    def test_thinking_capable_dated_variant(self, mock_anthropic):
        """Dated GLM-4.7 variants should be thinking-capable."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4.7-20250601")

        assert provider._is_thinking_capable() is True

    @patch('anthropic.Anthropic')
    def test_thinking_not_capable_no_model(self, mock_anthropic):
        """Should return False when no model is connected."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        assert provider._is_thinking_capable() is False
        assert provider.supports_thinking() is False

    @patch('anthropic.Anthropic')
    def test_set_thinking_config(self, mock_anthropic):
        """Should accept ThinkingConfig to enable/disable thinking."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4.7")

        # Enable thinking
        provider.set_thinking_config(ThinkingConfig(enabled=True, budget=5000))
        assert provider._enable_thinking is True
        assert provider._thinking_budget == 5000

        # Disable thinking
        provider.set_thinking_config(ThinkingConfig(enabled=False, budget=0))
        assert provider._enable_thinking is False

    @patch('anthropic.Anthropic')
    def test_thinking_budget_from_config(self, mock_anthropic):
        """Should use thinking budget from config."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            extra={"enable_thinking": True, "thinking_budget": 20000}
        ))

        assert provider._enable_thinking is True
        assert provider._thinking_budget == 20000

    @patch('anthropic.Anthropic')
    @patch.dict('os.environ', {
        'ZHIPUAI_API_KEY': 'key',
        'ZHIPUAI_ENABLE_THINKING': 'true',
        'ZHIPUAI_THINKING_BUDGET': '15000',
    })
    def test_thinking_from_env(self, mock_anthropic):
        """Should use thinking config from environment variables."""
        provider = ZhipuAIProvider()
        provider.initialize()

        assert provider._enable_thinking is True
        assert provider._thinking_budget == 15000


class TestOpenAIModelsURL:
    """Tests for _openai_models_url helper.

    This function derives the OpenAI-compatible ``/models`` URL from the
    Anthropic base URL so that both pay-per-token and coding-plan
    endpoints are handled correctly.
    """

    def test_default_anthropic_url(self):
        """Should map default Anthropic URL to /api/paas/v4/models."""
        url = _openai_models_url("https://api.z.ai/api/anthropic")
        assert url == "https://api.z.ai/api/paas/v4/models"

    def test_coding_plan_anthropic_url(self):
        """Should map coding plan Anthropic URL to /api/coding/paas/v4/models."""
        url = _openai_models_url("https://api.z.ai/api/coding/anthropic")
        assert url == "https://api.z.ai/api/coding/paas/v4/models"

    def test_trailing_slash_stripped(self):
        """Should handle trailing slash on input URL."""
        url = _openai_models_url("https://api.z.ai/api/anthropic/")
        assert url == "https://api.z.ai/api/paas/v4/models"

    def test_china_endpoint(self):
        """Should handle open.bigmodel.cn domain."""
        url = _openai_models_url("https://open.bigmodel.cn/api/anthropic")
        assert url == "https://open.bigmodel.cn/api/paas/v4/models"

    def test_unknown_path_fallback(self):
        """Should fall back to sibling /paas/v4 for unknown paths."""
        url = _openai_models_url("https://custom.example.com/v1")
        assert url == "https://custom.example.com/paas/v4/models"


class TestFetchRemoteModels:
    """Tests for dynamic model discovery via ``_fetch_remote_models()``.

    Verifies that the provider correctly queries Z.AI's OpenAI-compatible
    ``GET /models`` endpoint and parses the response.  Uses the project's
    corporate-ready httpx client via ``shared.http.proxy.get_httpx_client``.
    """

    @patch('anthropic.Anthropic')
    def test_fetch_parses_openai_format(self, mock_anthropic):
        """Should parse standard OpenAI /models response format."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        resp_data = {
            "object": "list",
            "data": [
                {"id": "glm-5", "object": "model"},
                {"id": "glm-4.7", "object": "model"},
                {"id": "glm-4.7-flash", "object": "model"},
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch('shared.http.proxy.get_httpx_client', return_value=mock_client):
            models = provider._fetch_remote_models()

        assert "glm-5" in models
        assert "glm-4.7" in models
        assert len(models) == 3

        # Verify the correct URL was called
        call_args = mock_client.get.call_args
        assert "/paas/v4/models" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    @patch('anthropic.Anthropic')
    def test_fetch_returns_empty_on_network_error(self, mock_anthropic):
        """Should return empty list on network errors (graceful fallback)."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection refused")

        with patch('shared.http.proxy.get_httpx_client', return_value=mock_client):
            models = provider._fetch_remote_models()

        assert models == []

    @patch('anthropic.Anthropic')
    def test_fetch_returns_empty_without_api_key(self, mock_anthropic):
        """Should return empty list when no API key is available."""
        provider = ZhipuAIProvider()
        # Not initialized — no API key set
        provider._api_key = None

        with patch(
            'shared.plugins.model_provider.zhipuai.provider.resolve_api_key',
            return_value=None,
        ), patch(
            'shared.plugins.model_provider.zhipuai.provider.get_stored_api_key',
            return_value=None,
        ):
            models = provider._fetch_remote_models()

        assert models == []

    @patch('anthropic.Anthropic')
    def test_fetch_skips_entries_without_id(self, mock_anthropic):
        """Should skip malformed entries in /models response."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        resp_data = {
            "object": "list",
            "data": [
                {"id": "glm-5"},
                {"name": "no-id-field"},  # Missing 'id'
                {"id": "glm-4.7"},
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = resp_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch('shared.http.proxy.get_httpx_client', return_value=mock_client):
            models = provider._fetch_remote_models()

        assert models == ["glm-5", "glm-4.7"]


class TestCreateProvider:
    """Tests for factory function."""

    def test_create_provider(self):
        """Factory function should return ZhipuAIProvider instance."""
        from ..provider import create_provider

        provider = create_provider()
        assert isinstance(provider, ZhipuAIProvider)
        assert provider.name == "zhipuai"
