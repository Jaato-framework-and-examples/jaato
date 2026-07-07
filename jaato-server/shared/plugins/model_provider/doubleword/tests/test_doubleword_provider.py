"""Tests for the Doubleword provider."""

import pytest
from unittest.mock import MagicMock, patch

from ..provider import DoublewordProvider, create_provider
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import (
    Message,
    Role,
    ToolSchema,
)
from ..errors import (
    APIKeyNotFoundError,
    RateLimitError,
    InfrastructureError,
)
from ..env import (
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_service_tier,
    is_self_hosted,
    DEFAULT_BASE_URL,
)


# ==================== Helpers ====================

def create_mock_response(
    text="Hello!",
    tool_calls=None,
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=20,
    reasoning_content=None,
):
    """Create a mock OpenAI ChatCompletion response."""
    mock_response = MagicMock()

    mock_choice = MagicMock()
    mock_choice.finish_reason = finish_reason
    mock_choice.message = MagicMock()
    mock_choice.message.content = text
    mock_choice.message.tool_calls = tool_calls or []
    mock_choice.message.reasoning_content = reasoning_content

    mock_response.choices = [mock_choice]

    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = prompt_tokens + completion_tokens

    return mock_response


def create_mock_tool_call(name="test_tool", args='{"key": "value"}', call_id="call_123"):
    """Create a mock tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return tc


# ==================== Environment Tests ====================

class TestEnvironment:
    """Tests for environment variable resolution."""

    def test_resolve_api_key_from_env(self):
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_API_KEY": "dw-test-test123"}):
            assert resolve_api_key() == "dw-test-test123"

    def test_resolve_api_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.doubleword.auth.get_stored_api_key",
                return_value=None,
            ):
                assert resolve_api_key() is None

    def test_resolve_base_url_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL

    def test_resolve_base_url_from_env(self):
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_BASE_URL": "http://localhost:8000/v1"}):
            assert resolve_base_url() == "http://localhost:8000/v1"

    def test_resolve_context_length_unset_is_none(self):
        # No hardcoded fallback (no-fallback rule): unset env → None, so the
        # provider raises a "not configured" error rather than guessing a default.
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_context_length() is None

    def test_resolve_context_length_from_env(self):
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_CONTEXT_LENGTH": "131072"}):
            assert resolve_context_length() == 131072

    def test_resolve_context_length_invalid_is_none(self):
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_CONTEXT_LENGTH": "not-a-number"}):
            assert resolve_context_length() is None

    def test_resolve_service_tier_unset_is_none(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_service_tier() is None

    def test_resolve_service_tier_from_env(self):
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_SERVICE_TIER": "flex"}):
            assert resolve_service_tier() == "flex"

    def test_resolve_service_tier_blank_is_none(self):
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_SERVICE_TIER": "  "}):
            assert resolve_service_tier() is None

    def test_is_self_hosted_localhost(self):
        assert is_self_hosted("http://localhost:8000/v1") is True
        assert is_self_hosted("http://127.0.0.1:8000/v1") is True

    def test_is_self_hosted_private_network(self):
        assert is_self_hosted("http://192.168.1.100:8000/v1") is True
        assert is_self_hosted("http://10.0.0.5:8000/v1") is True

    def test_is_self_hosted_public(self):
        assert is_self_hosted(DEFAULT_BASE_URL) is False
        assert is_self_hosted("https://custom-gw.example.com/v1") is False


# ==================== Provider Tests ====================

class TestAuthentication:
    """Tests for authentication and initialization."""

    def test_initialize_without_key_raises(self):
        """Should raise APIKeyNotFoundError if no key and not self-hosted."""
        provider = DoublewordProvider()

        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.doubleword.auth.get_stored_api_key",
                return_value=None,
            ):
                with pytest.raises(APIKeyNotFoundError) as exc_info:
                    provider.initialize(ProviderConfig())

        assert "JAATO_DOUBLEWORD_API_KEY" in str(exc_info.value)
        assert "app.doubleword.ai/api-keys" in str(exc_info.value)

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_with_api_key(self, mock_client_class):
        """Should initialize with key from config.api_key."""
        mock_client_class.return_value = MagicMock()

        provider = DoublewordProvider()
        provider.initialize(ProviderConfig(
            api_key="dw-test-test", extra={"context_length": 131072},
        ))

        assert provider._api_key == "dw-test-test"
        assert provider._client is not None

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_DOUBLEWORD_API_KEY": "dw-test-env"}, clear=True)
    def test_initialize_from_env(self, mock_client_class):
        """Should auto-detect key from JAATO_DOUBLEWORD_API_KEY env var."""
        mock_client_class.return_value = MagicMock()

        provider = DoublewordProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 131072}))

        assert provider._api_key == "dw-test-env"

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_DOUBLEWORD_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_initialize_self_hosted_no_key(self, mock_client_class):
        """Should initialize without key for self-hosted proxy endpoints."""
        mock_client_class.return_value = MagicMock()

        provider = DoublewordProvider()
        with patch(
            "shared.plugins.model_provider.doubleword.auth.get_stored_api_key",
            return_value=None,
        ):
            provider.initialize(ProviderConfig(extra={"context_length": 131072}))

        assert provider._api_key is None
        assert provider._client is not None

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_connect_raises_when_context_unresolved(self, mock_client_class):
        """No hardcoded fallback: with the catalog empty (no per-model entry)
        AND no manual override, connect() raises rather than guessing a
        default context window.  init() stays cheap and does NOT raise (the
        window is bootstrapped at connect from the catalog)."""
        mock_client_class.return_value = MagicMock()
        provider = DoublewordProvider()
        provider.initialize(ProviderConfig(api_key="dw-test-test"))  # no raise
        provider._catalog_cache = []  # empty catalog -> detect returns None
        with pytest.raises(ValueError, match="context_length could not be resolved"):
            provider.connect("some-model")

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_DOUBLEWORD_CONTEXT_LENGTH": "200000"}, clear=True)
    def test_env_knob_stashed_at_init_resolves_at_connect(self, mock_client_class):
        """The env override is stashed at init and used at connect when the
        catalog has no entry for the model."""
        mock_client_class.return_value = MagicMock()
        provider = DoublewordProvider()
        provider.initialize(ProviderConfig(api_key="dw-test-test"))
        assert provider._context_length_knob == 200000
        assert provider._context_length == 0  # not resolved until connect
        provider._catalog_cache = []
        provider.connect("some-model")
        assert provider._context_length == 200000

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_catalog_context_beats_knob_at_connect(self, mock_client_class):
        """Catalog auto-detect is PRIMARY: the per-model context length from
        GET /v1/models wins over the manual knob."""
        mock_client_class.return_value = MagicMock()
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_CONTEXT_LENGTH": "200000"}):
            provider = DoublewordProvider()
            provider.initialize(ProviderConfig(api_key="dw-test-test"))
            provider._catalog_cache = [
                {"id": "deepseek-ai/DeepSeek-V4-Pro", "context_length": 131072},
            ]
            provider.connect("deepseek-ai/DeepSeek-V4-Pro")
            assert provider._context_length == 131072  # catalog wins over 200000

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_stashes_profile_knob_over_env(self, mock_client_class):
        """Profile knob (config.extra.context_length) wins over the env tier
        when stashed at init."""
        mock_client_class.return_value = MagicMock()
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_CONTEXT_LENGTH": "200000"}):
            provider = DoublewordProvider()
            provider.initialize(ProviderConfig(
                api_key="dw-test-test", extra={"context_length": 131072},
            ))
            assert provider._context_length_knob == 131072

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_custom_base_url(self, mock_client_class):
        """Should use custom base_url from config.extra."""
        mock_client_class.return_value = MagicMock()

        provider = DoublewordProvider()
        provider.initialize(ProviderConfig(
            api_key="dw-test-test",
            extra={"base_url": "http://gw.internal:8080/v1", "context_length": 131072},
        ))

        assert provider._base_url == "http://gw.internal:8080/v1"

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_bad_modalities_knob_raises(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = DoublewordProvider()
        with pytest.raises(TypeError, match="modalities"):
            provider.initialize(ProviderConfig(
                api_key="dw-test-test",
                extra={"modalities": "image"},  # must be a list, not a string
            ))


class TestVerifyAuth:
    """Tests for verify_auth (must work before initialize)."""

    def test_verify_auth_with_key(self):
        provider = DoublewordProvider()
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_API_KEY": "dw-test-test"}):
            assert provider.verify_auth() is True

    def test_verify_auth_honors_profile_extra_api_key(self):
        """The pre-init gate must accept a profile-supplied key in
        config.extra['api_key'] — the shape the runtime builds for
        verify_auth (ProviderConfig(extra=plugin_configs[doubleword])) when a
        profile sets plugin_configs.doubleword.api_key: pass://... ."""
        provider = DoublewordProvider()
        cfg = ProviderConfig(extra={"api_key": "dw-test-from-profile"})
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.doubleword.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),  # no stored creds
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_honors_profile_top_level_api_key(self):
        """config.api_key (top-level) is also honored."""
        provider = DoublewordProvider()
        cfg = ProviderConfig(api_key="dw-test-top-level")
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.doubleword.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_self_hosted(self):
        provider = DoublewordProvider()
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            with patch(
                "shared.plugins.model_provider.doubleword.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth() is True

    def test_verify_auth_no_key_raises(self):
        provider = DoublewordProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.doubleword.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                with pytest.raises(APIKeyNotFoundError):
                    provider.verify_auth(allow_interactive=False)

    def test_verify_auth_no_key_returns_false(self):
        provider = DoublewordProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.doubleword.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(allow_interactive=True) is False

    def test_verify_auth_surfaces_broken_credentials(self):
        """Broken credential file must surface the reason, not be
        swallowed into a generic "no key" error."""
        provider = DoublewordProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.doubleword.auth.try_load_credentials_with_reason",
                return_value=(None, "invalid JSON at /tmp/doubleword_auth.json: Expecting value"),
            ):
                messages = []
                # Not self-hosted, no key -> should raise, but first emit the reason
                with pytest.raises(APIKeyNotFoundError):
                    provider.verify_auth(on_message=messages.append)
                joined = "\n".join(messages)
                assert "could not be loaded" in joined
                assert "invalid JSON" in joined


class TestConnection:
    """Tests for connect and model bootstrap."""

    def test_connect_sets_model_and_bootstraps_context(self):
        provider = DoublewordProvider()
        provider._client = MagicMock()
        provider._catalog_cache = [
            {"id": "deepseek-ai/DeepSeek-V4-Pro", "context_length": 131072},
        ]
        provider.connect("deepseek-ai/DeepSeek-V4-Pro")
        assert provider.model_name == "deepseek-ai/DeepSeek-V4-Pro"
        assert provider.get_context_limit() == 131072

    def test_is_connected(self):
        provider = DoublewordProvider()
        assert provider.is_connected is False

        provider._client = MagicMock()
        assert provider.is_connected is False

        provider._model_name = "test-model"
        assert provider.is_connected is True

    def test_stateless_no_session_state(self):
        """Provider should not hold any conversation state."""
        provider = DoublewordProvider()
        assert not hasattr(provider, '_system_instruction')
        assert not hasattr(provider, '_tools')
        assert not hasattr(provider, '_history')


class TestCapabilities:
    """Tests for capability queries."""

    def test_supports_streaming(self):
        assert DoublewordProvider().supports_streaming() is True

    def test_supports_structured_output(self):
        assert DoublewordProvider().supports_structured_output() is True

    def test_supports_stop(self):
        assert DoublewordProvider().supports_stop() is True

    def test_supports_thinking_default(self):
        assert DoublewordProvider().supports_thinking() is False

    def test_supports_thinking_reasoning_models(self):
        provider = DoublewordProvider()
        for model in (
            "deepseek-ai/DeepSeek-V4-Pro",
            "gpt-oss-120b",
            "Qwen/Qwen3.5-35B-A3B",
            "zai-org/GLM-5.1",
            "moonshotai/Kimi-K2.6",
        ):
            provider._model_name = model
            assert provider.supports_thinking() is True, model

    def test_name(self):
        assert DoublewordProvider().name == "doubleword"


class TestTokenManagement:
    """Tests for token counting and context limits."""

    def test_count_tokens(self):
        provider = DoublewordProvider()
        assert provider.count_tokens("abcd") == 1
        assert provider.count_tokens("a" * 100) == 25

    def test_get_context_limit(self):
        provider = DoublewordProvider()
        provider._context_length = 131072
        assert provider.get_context_limit() == 131072

    def test_get_token_usage(self):
        provider = DoublewordProvider()
        assert provider.get_token_usage().total_tokens == 0


class TestErrorClassification:
    """Tests for error classification and retry logic."""

    def test_classify_rate_limit(self):
        provider = DoublewordProvider()
        exc = RateLimitError(original_error="429")
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": True, "infra": False}

    def test_classify_infrastructure(self):
        provider = DoublewordProvider()
        exc = InfrastructureError(status_code=500)
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": False, "infra": True}

    def test_classify_unknown(self):
        provider = DoublewordProvider()
        result = provider.classify_error(ValueError("unknown"))
        assert result is None

    def test_retry_after_rate_limit(self):
        provider = DoublewordProvider()
        exc = RateLimitError(retry_after=30.0)
        assert provider.get_retry_after(exc) == 30.0

    def test_retry_after_other(self):
        provider = DoublewordProvider()
        assert provider.get_retry_after(ValueError("x")) is None


class TestCreateProvider:
    """Tests for factory function."""

    def test_create_provider(self):
        provider = create_provider()
        assert isinstance(provider, DoublewordProvider)
        assert provider.name == "doubleword"


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_clears_state(self):
        provider = DoublewordProvider()
        provider._client = MagicMock()
        provider._model_name = "test"

        provider.shutdown()

        assert provider._client is None
        assert provider._model_name is None


# ==================== Catalog + Modality Detection ====================

class TestCatalogAndModalities:
    """The provider bootstraps context + INPUT modalities from the
    ``GET /v1/models`` catalog when it reports them, degrading to the
    manual knobs / text floor otherwise (Doubleword's OpenAI-compat catalog
    schema is not pinned to one shape).
    """

    def _provider_with_catalog(self, catalog):
        provider = DoublewordProvider()
        provider._catalog_cache = catalog  # seed cache; detect wins
        return provider

    def test_lookup_context_length(self):
        p = self._provider_with_catalog([
            {"id": "m1", "context_length": 131072},
        ])
        assert p._lookup_context_length("m1") == 131072
        assert p._lookup_context_length("absent") is None

    def test_lookup_context_length_alternate_keys(self):
        """The lookup tolerates the common catalog key spellings."""
        p = self._provider_with_catalog([
            {"id": "m1", "max_model_len": 32768},
            {"id": "m2", "max_context_length": 65536},
        ])
        assert p._lookup_context_length("m1") == 32768
        assert p._lookup_context_length("m2") == 65536

    def test_lookup_context_length_no_metadata_is_none(self):
        """A bare OpenAI-shape entry (id/object/owned_by only) yields None so
        resolution falls through to the manual override tiers."""
        p = self._provider_with_catalog([
            {"id": "m1", "object": "model", "owned_by": "doubleword"},
        ])
        assert p._lookup_context_length("m1") is None

    def test_modality_text_only(self):
        p = self._provider_with_catalog([
            {"id": "glm", "context_length": 131072,
             "architecture": {"modality": "text->text"}},
        ])
        p._model_name = "glm"
        assert p.modalities() == {"text"}
        assert p.supports_modality("image") is False

    def test_modality_vision_input(self):
        p = self._provider_with_catalog([
            {"id": "qwen-vl", "context_length": 32768,
             "architecture": {"modality": "text+image->text"}},
        ])
        assert p.modalities(model="qwen-vl") == {"text", "image"}

    def test_modality_knob_when_model_absent(self):
        p = self._provider_with_catalog([])  # empty catalog -> detect None
        p._model_name = "uncatalogued"
        p._modalities_knob = ["text", "image"]
        assert p.modalities() == {"text", "image"}

    def test_modality_text_floor_when_unknown(self):
        p = self._provider_with_catalog([])
        p._model_name = "uncatalogued"
        assert p.modalities() == {"text"}  # never a false image claim

    def test_modality_missing_architecture_falls_through(self):
        p = self._provider_with_catalog([{"id": "m", "context_length": 8192}])
        p._model_name = "m"
        p._modalities_knob = ["text", "image"]
        assert p.modalities() == {"text", "image"}  # knob, since no arch field

    def test_list_models_from_catalog(self):
        p = self._provider_with_catalog([
            {"id": "gpt-oss-20b"}, {"id": "deepseek-ai/DeepSeek-V4-Pro"},
            {"id": "Qwen/Qwen3.5-4B"},
        ])
        assert p.list_models() == [
            "Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-V4-Pro", "gpt-oss-20b",
        ]
        assert p.list_models(prefix="gpt-") == ["gpt-oss-20b"]

    def test_fetch_catalog_parses_and_caches(self):
        p = DoublewordProvider()
        p._api_key = "dw-test-key"
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"data": [{"id": "x", "context_length": 1}]}
        fake_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            first = p._fetch_catalog()
            second = p._fetch_catalog()  # cached: no second network call
        assert first == [{"id": "x", "context_length": 1}]
        assert second is first
        assert mock_get.call_count == 1
        args, kwargs = mock_get.call_args
        assert args[0] == f"{DEFAULT_BASE_URL}/models"

    def test_fetch_catalog_sends_bearer_auth(self):
        """The catalog GET carries the Bearer key.

        Doubleword's ``/v1/models`` requires auth and the listing is
        account-scoped (it returns the models *your* key can access —
        401 without one), unlike OVHcloud's public catalog.  Regression
        guard against copying the anonymous-fetch behavior here.
        """
        p = DoublewordProvider()
        p._api_key = "dw-test-key"
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"data": []}
        fake_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            p._fetch_catalog()
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer dw-test-key"

    def test_fetch_catalog_network_failure_returns_empty_not_cached(self):
        p = DoublewordProvider()
        with patch("httpx.get", side_effect=Exception("boom")):
            assert p._fetch_catalog() == []
        assert p._catalog_cache is None  # not cached -> next call retries


# ==================== api_params Passthrough ====================

class TestApiParams:
    """plugin_configs.doubleword.api_params forwarded onto
    chat.completions.create (the determinism + tool_choice knob; parity
    with the rest of the OpenAI-compat fleet)."""

    def _provider(self, api_params, env=None):
        with patch(
            "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = DoublewordProvider()
            with patch.dict("os.environ", env or {}, clear=env is not None):
                p.initialize(ProviderConfig(
                    api_key="dw-test", extra={"api_params": api_params},
                ))
        return p

    def test_parsed_and_filtered_at_init(self):
        p = self._provider({
            "temperature": 0.0, "tool_choice": "required", "max_tokens": 4096,
            "top_k": 40, "bogus": 1,  # unsupported -> dropped (warning)
        })
        assert p._api_params == {
            "temperature": 0.0, "tool_choice": "required", "max_tokens": 4096,
        }

    def test_temperature_zero_survives(self):
        # The determinism case: 0.0 must NOT be falsy-dropped.
        p = self._provider({"temperature": 0.0})
        kwargs = {}
        p._apply_api_params(kwargs, tool_choice=None)
        assert kwargs["temperature"] == 0.0

    def test_tool_choice_forwarded_with_tools(self):
        p = self._provider({"tool_choice": "required"})
        kwargs = {"tools": [{"type": "function"}]}
        p._apply_api_params(kwargs, tool_choice=None)
        assert kwargs["tool_choice"] == "required"

    def test_tool_choice_dropped_without_tools(self):
        # OpenAI rejects tool_choice without tools; profile-wide "required"
        # must not break a tool-less call (e.g. GC summarization).
        p = self._provider({"tool_choice": "required", "temperature": 0.0})
        kwargs = {}
        p._apply_api_params(kwargs, tool_choice=None)
        assert "tool_choice" not in kwargs
        assert kwargs["temperature"] == 0.0

    def test_non_dict_api_params_raises(self):
        with patch(
            "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = DoublewordProvider()
            with pytest.raises(TypeError):
                p.initialize(ProviderConfig(
                    api_key="dw-test", extra={"api_params": "required"},
                ))

    def test_extra_body_forwarded_to_create_kwargs(self):
        p = DoublewordProvider()
        p._extra_body = {"guided_json": {"type": "object"}}
        kwargs = {"tools": [object()]}
        p._apply_api_params(kwargs, None)
        assert kwargs["extra_body"] == {"guided_json": {"type": "object"}}

    def test_complete_batch_forwards_api_params_to_create(self):
        # End-to-end: complete() must hand api_params to chat.completions.create.
        p = self._provider({"temperature": 0.0, "tool_choice": "required",
                            "max_tokens": 256})
        p._model_name = "deepseek-ai/DeepSeek-V4-Pro"
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return create_mock_response(text=None,
                                        tool_calls=[create_mock_tool_call()],
                                        finish_reason="tool_calls")

        p._client.chat.completions.create = fake_create
        schema = ToolSchema(name="discovery_result", description="d", parameters={
            "type": "object", "properties": {}})
        p.complete([Message.from_text(Role.USER, "go")], tools=[schema])
        assert captured["temperature"] == 0.0
        assert captured["tool_choice"] == "required"
        assert captured["max_tokens"] == 256
        assert "tools" in captured


# ==================== service_tier Knob ====================

class TestServiceTier:
    """``api_params.service_tier`` selects Doubleword's inference tier on
    the same chat endpoint: ``"flex"`` = the discounted async tier (queued
    work, minutes-level latency), ``"priority"`` = realtime.  The env var
    ``JAATO_DOUBLEWORD_SERVICE_TIER`` is the profile-free fallback; the
    profile knob wins when both are set."""

    def _provider(self, api_params=None, env=None):
        with patch(
            "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = DoublewordProvider()
            extra = {"context_length": 131072}
            if api_params is not None:
                extra["api_params"] = api_params
            with patch.dict("os.environ", env or {}, clear=True):
                p.initialize(ProviderConfig(api_key="dw-test", extra=extra))
        return p

    def test_unset_by_default(self):
        p = self._provider()
        assert "service_tier" not in p._api_params
        kwargs = {}
        p._apply_api_params(kwargs, tool_choice=None)
        assert "service_tier" not in kwargs

    def test_profile_knob_forwarded(self):
        p = self._provider(api_params={"service_tier": "flex"})
        kwargs = {}
        p._apply_api_params(kwargs, tool_choice=None)
        assert kwargs["service_tier"] == "flex"

    def test_env_fallback(self):
        p = self._provider(env={"JAATO_DOUBLEWORD_SERVICE_TIER": "flex"})
        assert p._api_params["service_tier"] == "flex"

    def test_profile_knob_beats_env(self):
        p = self._provider(
            api_params={"service_tier": "priority"},
            env={"JAATO_DOUBLEWORD_SERVICE_TIER": "flex"},
        )
        assert p._api_params["service_tier"] == "priority"

    def test_non_string_tier_raises(self):
        with patch(
            "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = DoublewordProvider()
            with pytest.raises(TypeError, match="service_tier"):
                p.initialize(ProviderConfig(
                    api_key="dw-test",
                    extra={"api_params": {"service_tier": 1}},
                ))

    def test_complete_forwards_service_tier_to_create(self):
        # End-to-end: the tier must reach chat.completions.create.
        p = self._provider(api_params={"service_tier": "flex"})
        p._model_name = "deepseek-ai/DeepSeek-V4-Pro"
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return create_mock_response(text="ok")

        p._client.chat.completions.create = fake_create
        p.complete([Message.from_text(Role.USER, "go")])
        assert captured["service_tier"] == "flex"


# ==================== Auth info ====================

class TestGetAuthInfo:
    """get_auth_info describes the credential source actually in use."""

    def test_env_key(self):
        p = DoublewordProvider()
        p._api_key = "dw-test"
        with patch.dict("os.environ", {"JAATO_DOUBLEWORD_API_KEY": "dw-test"}):
            assert "JAATO_DOUBLEWORD_API_KEY" in p.get_auth_info()

    def test_self_hosted(self):
        p = DoublewordProvider()
        p._base_url = "http://localhost:8000/v1"
        with patch.dict("os.environ", {}, clear=True):
            assert "Self-hosted" in p.get_auth_info()
