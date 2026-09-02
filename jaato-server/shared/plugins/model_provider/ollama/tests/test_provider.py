"""Tests for OllamaProvider."""

import pytest
from unittest.mock import MagicMock, patch

from ..provider import (
    OllamaProvider,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from ..env import DEFAULT_OLLAMA_HOST
from shared.plugins.model_provider.base import ProviderConfig


@pytest.fixture(autouse=True)
def _mock_api_show(monkeypatch):
    """Default mock for POST /api/show (tier-1 context auto-detection at
    connect()) so connect tests resolve a context window deterministically
    instead of hitting a real local Ollama."""
    def _post(url, *args, **kwargs):
        resp = MagicMock(status_code=200)
        resp.raise_for_status = MagicMock()
        if "/api/show" in url:
            resp.json.return_value = {
                "model_info": {"general.context_length": 8192}
            }
        else:
            resp.json.return_value = {}
        return resp
    monkeypatch.setattr("httpx.post", _post)


class TestInitialization:
    """Tests for initialization."""

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_initialize_default_host(self, mock_anthropic, mock_get):
        """Should use default host when not configured."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}

        provider = OllamaProvider()
        provider.initialize()

        assert provider._host == DEFAULT_OLLAMA_HOST
        mock_anthropic.assert_called_once()
        call_kwargs = mock_anthropic.call_args.kwargs
        # SDK adds /v1/messages to base_url, so we pass host directly
        assert call_kwargs["base_url"] == DEFAULT_OLLAMA_HOST
        assert call_kwargs["api_key"] == "ollama"

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_initialize_custom_host(self, mock_anthropic, mock_get):
        """Should use custom host from config."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}

        provider = OllamaProvider()
        provider.initialize(ProviderConfig(extra={"host": "http://remote:11434"}))

        assert provider._host == "http://remote:11434"
        call_kwargs = mock_anthropic.call_args.kwargs
        # SDK adds /v1/messages to base_url, so we pass host directly
        assert call_kwargs["base_url"] == "http://remote:11434"

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    @patch.dict('os.environ', {'OLLAMA_HOST': 'http://envhost:11434'})
    def test_initialize_host_from_env(self, mock_anthropic, mock_get):
        """Should use host from OLLAMA_HOST env var."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}

        provider = OllamaProvider()
        provider.initialize()

        assert provider._host == "http://envhost:11434"

    @patch('httpx.get')
    def test_initialize_connection_error(self, mock_get):
        """Should raise OllamaConnectionError if server not running."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        provider = OllamaProvider()
        with pytest.raises(OllamaConnectionError) as exc_info:
            provider.initialize()

        assert DEFAULT_OLLAMA_HOST in str(exc_info.value)
        assert "ollama serve" in str(exc_info.value)

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_no_cache_plugin_by_default(self, mock_anthropic, mock_get):
        """Cache plugin is not attached by default (no cache plugin matches 'ollama')."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}

        provider = OllamaProvider()
        provider.initialize()

        # No cache plugin attached directly on provider
        assert provider._cache_plugin is None

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_thinking_disabled(self, mock_anthropic, mock_get):
        """Should have thinking disabled (not supported by Ollama)."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}

        provider = OllamaProvider()
        provider.initialize(ProviderConfig(extra={"enable_thinking": True}))

        # Should still be disabled
        assert provider._enable_thinking is False


class TestConnection:
    """Tests for model connection."""

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_connect_existing_model(self, mock_anthropic, mock_get):
        """Should connect to an existing model."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "qwen3:32b"},
                {"name": "llama3.3:70b"},
            ]
        }

        provider = OllamaProvider()
        provider.initialize()
        provider.connect("qwen3:32b")

        assert provider.model_name == "qwen3:32b"
        assert provider.is_connected is True

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_connect_nonexistent_model(self, mock_anthropic, mock_get):
        """Should raise OllamaModelNotFoundError for missing model."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen3:32b"}]
        }

        provider = OllamaProvider()
        provider.initialize()

        with pytest.raises(OllamaModelNotFoundError) as exc_info:
            provider.connect("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "ollama pull" in str(exc_info.value)

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_connect_with_latest_tag(self, mock_anthropic, mock_get):
        """Should connect when model has :latest tag in Ollama."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }

        provider = OllamaProvider()
        provider.initialize()
        provider.connect("llama3")  # Without :latest

        assert provider.model_name == "llama3"

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_connect_memory_error(self, mock_anthropic, mock_get):
        """Should raise RuntimeError with helpful message on memory error."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen3:32b"}]
        }

        # Mock the client to raise a memory error
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception(
            "Error code: 500 - model requires more system memory (20.0 GiB)"
        )
        mock_anthropic.return_value = mock_client

        provider = OllamaProvider()
        provider.initialize()

        with pytest.raises(RuntimeError) as exc_info:
            provider.connect("qwen3:32b")

        assert "Not enough memory" in str(exc_info.value)
        assert "qwen3:32b" in str(exc_info.value)


class TestModelListing:
    """Tests for model listing."""

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_list_models(self, mock_anthropic, mock_get):
        """Should list all available models."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "qwen3:32b"},
                {"name": "llama3.3:70b"},
                {"name": "mistral:7b"},
            ]
        }

        provider = OllamaProvider()
        provider.initialize()
        models = provider.list_models()

        assert len(models) == 3
        assert "qwen3:32b" in models

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_list_models_with_prefix(self, mock_anthropic, mock_get):
        """Should filter models by prefix."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "qwen3:32b"},
                {"name": "qwen3:14b"},
                {"name": "llama3.3:70b"},
            ]
        }

        provider = OllamaProvider()
        provider.initialize()
        models = provider.list_models(prefix="qwen")

        assert len(models) == 2
        assert all(m.startswith("qwen") for m in models)


class TestContextLimit:
    """Tests for context limit handling."""

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_context_limit_unknown_before_connect_no_hardcoded_default(
        self, mock_anthropic, mock_get, monkeypatch,
    ):
        """With no manual override and before connect() (no model yet, so
        no /api/show auto-detect), get_context_limit() reports 0 ("unknown")
        — NOT a hardcoded default (no-fallback rule)."""
        monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}

        provider = OllamaProvider()
        provider.initialize()

        assert provider.get_context_limit() == 0

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_custom_context_limit(self, mock_anthropic, mock_get):
        """Should use custom context limit from config."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}

        provider = OllamaProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 65536}))

        assert provider.get_context_limit() == 65536


class TestVerifyAuth:
    """Tests for auth verification."""

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_verify_auth_success(self, mock_anthropic, mock_get):
        """Should return True when Ollama is accessible."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}

        provider = OllamaProvider()
        provider.initialize()

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is True
        assert any("Connected" in m for m in messages)

    @patch('httpx.get')
    def test_verify_auth_failure(self, mock_get):
        """Should return False when Ollama is not accessible."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        provider = OllamaProvider()
        provider._host = DEFAULT_OLLAMA_HOST

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is False
        assert any("Cannot connect" in m for m in messages)


class TestProviderName:
    """Tests for provider identification."""

    def test_provider_name(self):
        """Should return 'ollama' as provider name."""
        provider = OllamaProvider()
        assert provider.name == "ollama"


class TestLogin:
    """Tests for login method."""

    def test_login_no_op(self):
        """Login should be a no-op for Ollama."""
        messages = []
        OllamaProvider.login(on_message=messages.append)

        assert any("doesn't require authentication" in m for m in messages)


class TestContextCapacityAutodetect:
    """Tier-1 context-window auto-detection via POST /api/show."""

    def _provider(self):
        p = OllamaProvider()
        p._host = "http://localhost:11434"
        p._model_name = "qwen3:32b"
        return p

    def test_detect_reads_arch_prefixed_context_length(self):
        p = self._provider()
        resp = MagicMock(status_code=200)
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"qwen3.context_length": 40960}}
        with patch("httpx.post", return_value=resp):
            assert p._detect_context_capacity() == 40960

    def test_detect_none_when_field_absent(self):
        p = self._provider()
        resp = MagicMock(status_code=200)
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"model_info": {"general.parameter_count": 32_000_000_000}}
        with patch("httpx.post", return_value=resp):
            assert p._detect_context_capacity() is None

    def test_detect_none_when_no_model(self):
        p = OllamaProvider()
        p._model_name = None
        assert p._detect_context_capacity() is None

    def test_detect_none_on_http_error(self):
        import httpx
        p = self._provider()
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            assert p._detect_context_capacity() is None

    @patch('httpx.get')
    @patch('anthropic.Anthropic')
    def test_connect_uses_detected_context_over_no_knob(
        self, mock_anthropic, mock_get, monkeypatch,
    ):
        """connect() resolves the detected /api/show window when no manual
        override is set (detect PRIMARY)."""
        monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "qwen3:32b"}]}
        show = MagicMock(status_code=200)
        show.raise_for_status = MagicMock()
        show.json.return_value = {"model_info": {"qwen3.context_length": 40960}}
        provider = OllamaProvider()
        provider.initialize()
        with patch("httpx.post", return_value=show):
            provider.connect("qwen3:32b", skip_model_test=True)
        assert provider.get_context_limit() == 40960
