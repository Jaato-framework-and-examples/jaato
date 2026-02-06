"""Tests for ZhipuAIProvider."""

import pytest
from unittest.mock import MagicMock, patch

from ..provider import (
    ZhipuAIProvider,
    ZhipuAIAPIKeyNotFoundError,
    ZhipuAIConnectionError,
    DEFAULT_CONTEXT_LIMIT,
    KNOWN_MODELS,
    THINKING_CAPABLE_MODELS,
)
from ..env import DEFAULT_ZHIPUAI_BASE_URL
from ...base import ProviderConfig
from ...types import ThinkingConfig


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
    def test_caching_disabled(self, mock_anthropic):
        """Should have caching disabled (may not be supported)."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            extra={"enable_caching": True}
        ))

        # Should still be disabled
        assert provider._enable_caching is False

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
    """Tests for model listing."""

    @patch('anthropic.Anthropic')
    def test_list_models(self, mock_anthropic):
        """Should list known GLM models."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        models = provider.list_models()

        assert len(models) == len(KNOWN_MODELS)
        assert "glm-4.7" in models
        assert "glm-4.7-flash" in models

    @patch('anthropic.Anthropic')
    def test_list_models_with_prefix(self, mock_anthropic):
        """Should filter models by prefix."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        models = provider.list_models(prefix="glm-4.7")

        assert len(models) == 2
        assert all(m.startswith("glm-4.7") for m in models)


class TestContextLimit:
    """Tests for context limit handling."""

    @patch('anthropic.Anthropic')
    def test_default_context_limit(self, mock_anthropic):
        """Should return default context limit (128K for GLM-4.7)."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        assert provider.get_context_limit() == DEFAULT_CONTEXT_LIMIT
        assert provider.get_context_limit() == 131072

    @patch('anthropic.Anthropic')
    def test_custom_context_limit(self, mock_anthropic):
        """Should use custom context limit from config."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            extra={"context_length": 65536}
        ))

        assert provider.get_context_limit() == 65536


class TestVerifyAuth:
    """Tests for auth verification."""

    @patch('anthropic.Anthropic')
    def test_verify_auth_success(self, mock_anthropic):
        """Should return True when API key is valid."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock()
        mock_anthropic.return_value = mock_client

        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is True
        assert any("Connected" in m for m in messages)

    @patch('anthropic.Anthropic')
    def test_verify_auth_failure(self, mock_anthropic):
        """Should return False when API key is invalid."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("401 Unauthorized")
        mock_anthropic.return_value = mock_client

        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="bad-key"))

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is False
        assert any("Cannot connect" in m for m in messages)


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


class TestCreateProvider:
    """Tests for factory function."""

    def test_create_provider(self):
        """Factory function should return ZhipuAIProvider instance."""
        from ..provider import create_provider

        provider = create_provider()
        assert isinstance(provider, ZhipuAIProvider)
        assert provider.name == "zhipuai"
