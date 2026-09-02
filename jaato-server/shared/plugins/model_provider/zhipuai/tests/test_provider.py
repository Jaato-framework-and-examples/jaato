"""Tests for ZhipuAIProvider."""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from ..provider import (
    ZhipuAIProvider,
    ZhipuAIAPIKeyNotFoundError,
    ZhipuAIConnectionError,
    ZhipuAIRateLimitError,
    MODEL_CONTEXT_LIMITS,
    KNOWN_MODELS,
    THINKING_CAPABLE_MODELS,
    ZHIPUAI_MODELS_URL,
)
from ..env import DEFAULT_ZHIPUAI_BASE_URL
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import ThinkingConfig


@pytest.fixture(autouse=True)
def _isolate_home_credentials(monkeypatch, tmp_path):
    """Keep the developer's real ~/.jaato credentials out of these tests.

    Credential resolution falls through to a HOME tier, so "no key
    configured" tests found a REAL stored key on a machine where the
    developer has actually authenticated -- one asserted a
    ZhipuAIAPIKeyNotFoundError that never came.  Clean CI has no such file,
    which is why it passed there while failing locally.
    """
    empty_home = tmp_path / "home"
    (empty_home / ".jaato").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty_home))
    # BOTH tiers leak, not just home: the project tier resolves to
    # ``<cwd>/.jaato/`` and pytest runs from the repo root, which carries real
    # stored credentials.  Move cwd somewhere empty and clear the workspace
    # env so neither tier can reach them.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)


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


class TestConfigNamespacing:
    """Tests for the four-layer config namespacing introduced in 0.6.24.

    Layers (under ``plugin_configs.zhipuai``):
      - Top-level: api_key (auth identity)
      - api_params: temperature / top_p / top_k / max_tokens /
                    enable_thinking / thinking_budget
      - framework_overrides: base_url, context_length

    Sampling params reach ``messages.create()`` via the inherited
    ``complete()`` method in AnthropicProvider, which reads
    ``self._temperature`` etc.

    Backward compatibility: same keys are also read from the legacy
    flat position with a deprecation warning per key.
    """

    @patch('anthropic.Anthropic')
    def test_new_shape_no_deprecation_warning(self, mock_anthropic, caplog):
        provider = ZhipuAIProvider()
        with caplog.at_level("WARNING"):
            provider.initialize(ProviderConfig(
                api_key="zhipuai-test",
                extra={
                    "api_params": {
                        "temperature": 0.0,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_tokens": 4096,
                        "enable_thinking": False,
                        "thinking_budget": 3000,
                    },
                    "framework_overrides": {
                        "context_length": 131072,
                        "base_url": "https://api.z.ai/api/anthropic",
                    },
                },
            ))
        assert provider._temperature == 0.0
        assert provider._top_p == 0.95
        assert provider._top_k == 40
        assert provider._max_tokens_override == 4096
        assert provider._enable_thinking is False
        assert provider._thinking_budget == 3000
        assert provider._context_length_override == 131072
        assert provider._base_url == "https://api.z.ai/api/anthropic"
        legacy_warnings = [r for r in caplog.records if "legacy" in r.getMessage().lower()]
        assert legacy_warnings == [], (
            f"new shape should not emit deprecation warnings, got: "
            f"{[r.getMessage() for r in legacy_warnings]}"
        )

    @patch('anthropic.Anthropic')
    def test_legacy_flat_shape_still_works_with_warnings(
        self, mock_anthropic, caplog,
    ):
        provider = ZhipuAIProvider()
        with caplog.at_level("WARNING"):
            provider.initialize(ProviderConfig(
                api_key="zhipuai-test",
                extra={
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "enable_thinking": False,
                    "thinking_budget": 3000,
                    "context_length": 131072,
                },
            ))
        assert provider._temperature == 0.0
        assert provider._top_p == 0.95
        assert provider._enable_thinking is False
        assert provider._thinking_budget == 3000
        assert provider._context_length_override == 131072
        legacy_warnings = [r for r in caplog.records if "legacy" in r.getMessage().lower()]
        assert len(legacy_warnings) >= 5, (
            f"expected ≥5 deprecation warnings, got {len(legacy_warnings)}"
        )

    @patch('anthropic.Anthropic')
    def test_no_sampling_config_means_none(self, mock_anthropic):
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="zhipuai-test"))
        assert provider._temperature is None
        assert provider._top_p is None
        assert provider._top_k is None
        assert provider._max_tokens_override is None


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
    def test_no_model_no_override_raises(self, mock_anthropic):
        """No hardcoded fallback: no model connected + no override → raise."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))

        with pytest.raises(ValueError, match="no known context window"):
            provider.get_context_limit()

    @patch('anthropic.Anthropic')
    def test_dated_variant_resolves_via_longest_prefix(self, mock_anthropic):
        """A dated variant lands on its family (glm-4.7), not a shorter prefix."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4.7-20250601")
        assert provider.get_context_limit() == 204800  # glm-4.7 family, not glm-4

    @patch('anthropic.Anthropic')
    def test_legacy_glm4_family_resolved(self, mock_anthropic):
        """Legacy GLM-4 generation (128K) is in the table, not fail-loud."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(api_key="test-key"))
        provider.connect("glm-4")
        assert provider.get_context_limit() == 131072
        provider.connect("glm-4v")
        assert provider.get_context_limit() == 131072

    @patch('anthropic.Anthropic')
    def test_override_knob_covers_unknown_model(self, mock_anthropic):
        """context_length override is the escape hatch for an unlisted model."""
        provider = ZhipuAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            extra={"framework_overrides": {"context_length": 65536}},
        ))
        provider.connect("glm-99-future")
        assert provider.get_context_limit() == 65536

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
    @patch('shared.plugins.model_provider.zhipuai.provider.try_load_credentials_with_reason')
    def test_verify_auth_with_stored_key(self, mock_try_load):
        """Should return True when API key is stored."""
        from ..auth import ZhipuAICredentials
        mock_try_load.return_value = (
            ZhipuAICredentials(api_key='stored-key', created_at=0.0),
            None,
        )
        provider = ZhipuAIProvider()  # NOT initialized

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is True

    @patch.dict('os.environ', {}, clear=True)
    @patch(
        'shared.plugins.model_provider.zhipuai.provider.try_load_credentials_with_reason',
        return_value=(None, None),
    )
    def test_verify_auth_no_credentials(self, mock_try_load):
        """Should return False with "No credentials" when nothing is configured."""
        provider = ZhipuAIProvider()  # NOT initialized

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is False
        assert any("No Zhipu AI credentials found" in m for m in messages)

    @patch.dict('os.environ', {}, clear=True)
    @patch('shared.plugins.model_provider.zhipuai.provider.try_load_credentials_with_reason')
    def test_verify_auth_broken_credentials_surfaces_reason(self, mock_try_load):
        """Broken credential file must surface the reason, not a generic
        "No credentials found" message.

        This is the bug the user flagged: a provider error (in this case a
        credential file that couldn't be loaded) was being masked as a
        missing token.  The fix exposes the real reason through
        ``on_message`` so the user can fix the actual problem.
        """
        mock_try_load.return_value = (
            None,
            "invalid JSON at /tmp/zhipuai_auth.json: Expecting value (line 1, col 1)",
        )
        provider = ZhipuAIProvider()  # NOT initialized

        messages = []
        result = provider.verify_auth(on_message=messages.append)

        assert result is False
        joined = "\n".join(messages)
        assert "could not be loaded" in joined
        assert "invalid JSON" in joined
        # Must NOT emit the old misleading "No credentials found" message.
        assert "No Zhipu AI credentials found" not in joined


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

        with pytest.raises(ZhipuAIRateLimitError) as exc_info:
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
        # login() gained an ``on_input`` callback and falls back to builtin
        # input() without one -- which raises under pytest's output capture.
        # Supply one so the guidance path runs without touching stdin.
        ZhipuAIProvider.login(
            on_message=messages.append,
            on_input=lambda _prompt="": "",
        )

        # The flow changed from "set ZHIPUAI_API_KEY" to an interactive paste,
        # so it no longer names the env var.  The intent -- actionable guidance
        # for obtaining a key, and a clear outcome -- is what gets pinned.
        blob = "\n".join(messages)
        assert "z.ai/model-api" in blob        # international portal
        assert "open.bigmodel.cn" in blob      # China portal
        assert any("cancelled" in m.lower() for m in messages), (
            "supplying no key must report the outcome, not fail silently"
        )


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


class TestModelsURL:
    """Tests for ZHIPUAI_MODELS_URL constant."""

    def test_models_url_points_to_bigmodel(self):
        """Models endpoint is on open.bigmodel.cn, not api.z.ai."""
        assert ZHIPUAI_MODELS_URL == "https://open.bigmodel.cn/api/paas/v4/models"


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

        assert models == ["glm-4.7", "glm-5"]


class TestCreateProvider:
    """Tests for factory function."""

    def test_create_provider(self):
        """Factory function should return ZhipuAIProvider instance."""
        from ..provider import create_provider

        provider = create_provider()
        assert isinstance(provider, ZhipuAIProvider)
        assert provider.name == "zhipuai"
