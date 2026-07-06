"""Tests for OVHcloud AI Endpoints provider."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from ..provider import OVHcloudProvider, create_provider
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
    ToolSchema,
)
from ..errors import (
    APIKeyNotFoundError,
    RateLimitError,
    InfrastructureError,
)
from ..._openai_compat.converters import (
    sanitize_tool_name,
    message_to_openai,
)
from ..env import (
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_allow_anonymous,
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
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_API_KEY": "ovh-test-test123"}):
            assert resolve_api_key() == "ovh-test-test123"

    def test_resolve_api_key_from_vendor_env(self):
        """The vendor's own documented variable is honored when the jaato
        namespace var is unset — users who already export it for OVHcloud's
        OpenAI SDK examples work with no extra config."""
        with patch.dict(
            "os.environ",
            {"OVH_AI_ENDPOINTS_ACCESS_TOKEN": "ovh-test-vendor"},
            clear=True,
        ):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth.get_stored_api_key",
                return_value=None,
            ):
                assert resolve_api_key() == "ovh-test-vendor"

    def test_jaato_namespace_beats_vendor_env(self):
        with patch.dict("os.environ", {
            "JAATO_OVHCLOUD_API_KEY": "ovh-test-jaato",
            "OVH_AI_ENDPOINTS_ACCESS_TOKEN": "ovh-test-vendor",
        }):
            assert resolve_api_key() == "ovh-test-jaato"

    def test_resolve_api_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_api_key() is None

    def test_resolve_base_url_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL

    def test_resolve_base_url_from_env(self):
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_BASE_URL": "http://localhost:8000/v1"}):
            assert resolve_base_url() == "http://localhost:8000/v1"

    def test_resolve_context_length_unset_is_none(self):
        # No hardcoded fallback (no-fallback rule): unset env → None, so the
        # provider raises a "not configured" error rather than guessing a default.
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_context_length() is None

    def test_resolve_context_length_from_env(self):
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_CONTEXT_LENGTH": "131072"}):
            assert resolve_context_length() == 131072

    def test_resolve_context_length_invalid_is_none(self):
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_CONTEXT_LENGTH": "not-a-number"}):
            assert resolve_context_length() is None

    def test_resolve_allow_anonymous_default_false(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_allow_anonymous() is False

    def test_resolve_allow_anonymous_truthy_values(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            with patch.dict("os.environ", {"JAATO_OVHCLOUD_ALLOW_ANONYMOUS": value}):
                assert resolve_allow_anonymous() is True

    def test_resolve_allow_anonymous_falsy_values(self):
        for value in ("0", "false", "no", "off", ""):
            with patch.dict("os.environ", {"JAATO_OVHCLOUD_ALLOW_ANONYMOUS": value}):
                assert resolve_allow_anonymous() is False

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
        """Should raise APIKeyNotFoundError if no key, no anonymous opt-in,
        and not self-hosted."""
        provider = OVHcloudProvider()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIKeyNotFoundError) as exc_info:
                provider.initialize(ProviderConfig())

        assert "JAATO_OVHCLOUD_API_KEY" in str(exc_info.value)
        assert "JAATO_OVHCLOUD_ALLOW_ANONYMOUS" in str(exc_info.value)

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_with_api_key(self, mock_client_class):
        """Should initialize with key from config.api_key."""
        mock_client_class.return_value = MagicMock()

        provider = OVHcloudProvider()
        provider.initialize(ProviderConfig(
            api_key="ovh-test-test", extra={"context_length": 131072},
        ))

        assert provider._api_key == "ovh-test-test"
        assert provider._client is not None

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_OVHCLOUD_API_KEY": "ovh-test-env"}, clear=True)
    def test_initialize_from_env(self, mock_client_class):
        """Should auto-detect key from JAATO_OVHCLOUD_API_KEY env var."""
        mock_client_class.return_value = MagicMock()

        provider = OVHcloudProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 131072}))

        assert provider._api_key == "ovh-test-env"

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"OVH_AI_ENDPOINTS_ACCESS_TOKEN": "ovh-test-vendor"}, clear=True)
    def test_initialize_from_vendor_env(self, mock_client_class):
        """Should auto-detect key from the vendor's own env var."""
        mock_client_class.return_value = MagicMock()

        provider = OVHcloudProvider()
        with patch(
            "shared.plugins.model_provider.ovhcloud.auth.get_stored_api_key",
            return_value=None,
        ):
            provider.initialize(ProviderConfig(extra={"context_length": 131072}))

        assert provider._api_key == "ovh-test-vendor"

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_OVHCLOUD_ALLOW_ANONYMOUS": "true"}, clear=True)
    def test_initialize_anonymous_env_opt_in(self, mock_client_class):
        """Keyless init succeeds when the free tier is opted in via env."""
        mock_client_class.return_value = MagicMock()

        provider = OVHcloudProvider()
        with patch(
            "shared.plugins.model_provider.ovhcloud.env.resolve_api_key",
            return_value=None,
        ):
            provider.initialize(ProviderConfig(extra={"context_length": 131072}))

        assert provider._api_key is None
        assert provider._allow_anonymous is True
        assert provider._client is not None

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_initialize_anonymous_knob_opt_in(self, mock_client_class):
        """Keyless init succeeds via the allow_anonymous profile knob."""
        mock_client_class.return_value = MagicMock()

        provider = OVHcloudProvider()
        provider.initialize(ProviderConfig(extra={
            "allow_anonymous": True, "context_length": 131072,
        }))

        assert provider._api_key is None
        assert provider._allow_anonymous is True

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_OVHCLOUD_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_initialize_self_hosted_no_key(self, mock_client_class):
        """Should initialize without key for self-hosted proxy endpoints."""
        mock_client_class.return_value = MagicMock()

        provider = OVHcloudProvider()
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
        provider = OVHcloudProvider()
        provider.initialize(ProviderConfig(api_key="ovh-test-test"))  # no raise
        provider._catalog_cache = []  # empty catalog -> detect returns None
        with pytest.raises(ValueError, match="context_length could not be resolved"):
            provider.connect("some-model")

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_OVHCLOUD_CONTEXT_LENGTH": "200000"}, clear=True)
    def test_env_knob_stashed_at_init_resolves_at_connect(self, mock_client_class):
        """The env override is stashed at init and used at connect when the
        catalog has no entry for the model."""
        mock_client_class.return_value = MagicMock()
        provider = OVHcloudProvider()
        provider.initialize(ProviderConfig(api_key="ovh-test-test"))
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
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_CONTEXT_LENGTH": "200000"}):
            provider = OVHcloudProvider()
            provider.initialize(ProviderConfig(api_key="ovh-test-test"))
            provider._catalog_cache = [
                {"id": "Meta-Llama-3_3-70B-Instruct", "context_length": 131072},
            ]
            provider.connect("Meta-Llama-3_3-70B-Instruct")
            assert provider._context_length == 131072  # catalog wins over 200000

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_stashes_profile_knob_over_env(self, mock_client_class):
        """Profile knob (config.extra.context_length) wins over the env tier
        when stashed at init."""
        mock_client_class.return_value = MagicMock()
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_CONTEXT_LENGTH": "200000"}):
            provider = OVHcloudProvider()
            provider.initialize(ProviderConfig(
                api_key="ovh-test-test", extra={"context_length": 131072},
            ))
            assert provider._context_length_knob == 131072

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_custom_base_url(self, mock_client_class):
        """Should use custom base_url from config.extra."""
        mock_client_class.return_value = MagicMock()

        provider = OVHcloudProvider()
        provider.initialize(ProviderConfig(
            api_key="ovh-test-test",
            extra={"base_url": "http://gw.internal:8080/v1", "context_length": 131072},
        ))

        assert provider._base_url == "http://gw.internal:8080/v1"

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_bad_modalities_knob_raises(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OVHcloudProvider()
        with pytest.raises(TypeError, match="modalities"):
            provider.initialize(ProviderConfig(
                api_key="ovh-test-test",
                extra={"modalities": "image"},  # must be a list, not a string
            ))


class TestVerifyAuth:
    """Tests for verify_auth (must work before initialize)."""

    def test_verify_auth_with_key(self):
        provider = OVHcloudProvider()
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_API_KEY": "ovh-test-test"}):
            assert provider.verify_auth() is True

    def test_verify_auth_with_vendor_key(self):
        provider = OVHcloudProvider()
        with patch.dict(
            "os.environ",
            {"OVH_AI_ENDPOINTS_ACCESS_TOKEN": "ovh-test-vendor"},
            clear=True,
        ):
            assert provider.verify_auth() is True

    def test_verify_auth_honors_profile_extra_api_key(self):
        """The pre-init gate must accept a profile-supplied key in
        config.extra['api_key'] — the shape the runtime builds for
        verify_auth (ProviderConfig(extra=plugin_configs[ovhcloud])) when a
        profile sets plugin_configs.ovhcloud.api_key: pass://... ."""
        provider = OVHcloudProvider()
        cfg = ProviderConfig(extra={"api_key": "ovh-test-from-profile"})
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),  # no stored creds
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_honors_profile_top_level_api_key(self):
        """config.api_key (top-level) is also honored."""
        provider = OVHcloudProvider()
        cfg = ProviderConfig(api_key="ovh-test-top-level")
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_anonymous_env_opt_in(self):
        """The keyless free tier passes the pre-init gate with a notice."""
        provider = OVHcloudProvider()
        with patch.dict(
            "os.environ", {"JAATO_OVHCLOUD_ALLOW_ANONYMOUS": "true"}, clear=True,
        ):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                messages = []
                assert provider.verify_auth(on_message=messages.append) is True
                assert any("anonymous" in m for m in messages)

    def test_verify_auth_anonymous_knob_opt_in(self):
        provider = OVHcloudProvider()
        cfg = ProviderConfig(extra={"allow_anonymous": True})
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_self_hosted(self):
        provider = OVHcloudProvider()
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth() is True

    def test_verify_auth_no_key_raises(self):
        provider = OVHcloudProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                with pytest.raises(APIKeyNotFoundError):
                    provider.verify_auth(allow_interactive=False)

    def test_verify_auth_no_key_returns_false(self):
        provider = OVHcloudProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(allow_interactive=True) is False

    def test_verify_auth_surfaces_broken_credentials(self):
        """Broken credential file must surface the reason, not be
        swallowed into a generic "no key" error."""
        provider = OVHcloudProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.ovhcloud.auth.try_load_credentials_with_reason",
                return_value=(None, "invalid JSON at /tmp/ovhcloud_auth.json: Expecting value"),
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
        provider = OVHcloudProvider()
        provider._client = MagicMock()
        provider._catalog_cache = [
            {"id": "gpt-oss-120b", "context_length": 131072},
        ]
        provider.connect("gpt-oss-120b")
        assert provider.model_name == "gpt-oss-120b"
        assert provider.get_context_limit() == 131072

    def test_is_connected(self):
        provider = OVHcloudProvider()
        assert provider.is_connected is False

        provider._client = MagicMock()
        assert provider.is_connected is False

        provider._model_name = "test-model"
        assert provider.is_connected is True

    def test_stateless_no_session_state(self):
        """Provider should not hold any conversation state."""
        provider = OVHcloudProvider()
        assert not hasattr(provider, '_system_instruction')
        assert not hasattr(provider, '_tools')
        assert not hasattr(provider, '_history')


class TestCapabilities:
    """Tests for capability queries."""

    def test_supports_streaming(self):
        assert OVHcloudProvider().supports_streaming() is True

    def test_supports_structured_output(self):
        assert OVHcloudProvider().supports_structured_output() is True

    def test_supports_stop(self):
        assert OVHcloudProvider().supports_stop() is True

    def test_supports_thinking_default(self):
        assert OVHcloudProvider().supports_thinking() is False

    def test_supports_thinking_reasoning_models(self):
        provider = OVHcloudProvider()
        for model in ("DeepSeek-R1-Distill-Llama-70B", "gpt-oss-120b", "QwQ-32B"):
            provider._model_name = model
            assert provider.supports_thinking() is True, model

    def test_name(self):
        assert OVHcloudProvider().name == "ovhcloud"


class TestTokenManagement:
    """Tests for token counting and context limits."""

    def test_count_tokens(self):
        provider = OVHcloudProvider()
        assert provider.count_tokens("abcd") == 1
        assert provider.count_tokens("a" * 100) == 25

    def test_get_context_limit(self):
        provider = OVHcloudProvider()
        provider._context_length = 131072
        assert provider.get_context_limit() == 131072

    def test_get_token_usage(self):
        provider = OVHcloudProvider()
        assert provider.get_token_usage().total_tokens == 0


class TestErrorClassification:
    """Tests for error classification and retry logic."""

    def test_classify_rate_limit(self):
        provider = OVHcloudProvider()
        exc = RateLimitError(original_error="429")
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": True, "infra": False}

    def test_classify_infrastructure(self):
        provider = OVHcloudProvider()
        exc = InfrastructureError(status_code=500)
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": False, "infra": True}

    def test_classify_unknown(self):
        provider = OVHcloudProvider()
        result = provider.classify_error(ValueError("unknown"))
        assert result is None

    def test_retry_after_rate_limit(self):
        provider = OVHcloudProvider()
        exc = RateLimitError(retry_after=30.0)
        assert provider.get_retry_after(exc) == 30.0

    def test_retry_after_other(self):
        provider = OVHcloudProvider()
        assert provider.get_retry_after(ValueError("x")) is None


class TestCreateProvider:
    """Tests for factory function."""

    def test_create_provider(self):
        provider = create_provider()
        assert isinstance(provider, OVHcloudProvider)
        assert provider.name == "ovhcloud"


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_clears_state(self):
        provider = OVHcloudProvider()
        provider._client = MagicMock()
        provider._model_name = "test"

        provider.shutdown()

        assert provider._client is None
        assert provider._model_name is None


# ==================== Catalog + Modality Detection ====================

class TestCatalogAndModalities:
    """The provider bootstraps context + INPUT modalities from the
    ``GET /v1/models`` catalog when it reports them, degrading to the
    manual knobs / text floor otherwise (OVHcloud's OpenAI-compat catalog
    schema is not pinned to one shape).
    """

    def _provider_with_catalog(self, catalog):
        provider = OVHcloudProvider()
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
            {"id": "m1", "object": "model", "owned_by": "ovhcloud"},
        ])
        assert p._lookup_context_length("m1") is None

    def test_modality_text_only(self):
        p = self._provider_with_catalog([
            {"id": "llama", "context_length": 131072,
             "architecture": {"modality": "text->text"}},
        ])
        p._model_name = "llama"
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
            {"id": "gpt-oss-20b"}, {"id": "Meta-Llama-3_3-70B-Instruct"},
            {"id": "Mistral-7B-Instruct-v0.3"},
        ])
        assert p.list_models() == [
            "Meta-Llama-3_3-70B-Instruct", "Mistral-7B-Instruct-v0.3",
            "gpt-oss-20b",
        ]
        assert p.list_models(prefix="gpt-") == ["gpt-oss-20b"]

    def test_fetch_catalog_parses_and_caches(self):
        p = OVHcloudProvider()
        p._api_key = "ovh-test-key"
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"data": [{"id": "x", "context_length": 1}]}
        fake_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            first = p._fetch_catalog()
            second = p._fetch_catalog()  # cached: no second network call
        assert first == [{"id": "x", "context_length": 1}]
        assert second is first
        assert mock_get.call_count == 1
        # Bearer auth header carried on the catalog GET.
        args, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer ovh-test-key"
        assert args[0] == f"{DEFAULT_BASE_URL}/models"

    def test_fetch_catalog_anonymous_sends_no_auth_header(self):
        """The keyless free tier fetches the public catalog with no header."""
        p = OVHcloudProvider()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"data": []}
        fake_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            p._fetch_catalog()
        _, kwargs = mock_get.call_args
        assert "Authorization" not in kwargs["headers"]

    def test_fetch_catalog_network_failure_returns_empty_not_cached(self):
        p = OVHcloudProvider()
        with patch("httpx.get", side_effect=Exception("boom")):
            assert p._fetch_catalog() == []
        assert p._catalog_cache is None  # not cached -> next call retries


# ==================== api_params Passthrough ====================

class TestApiParams:
    """plugin_configs.ovhcloud.api_params forwarded onto
    chat.completions.create (the determinism + tool_choice knob; parity
    with the rest of the OpenAI-compat fleet)."""

    def _provider(self, api_params):
        with patch(
            "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = OVHcloudProvider()
            p.initialize(ProviderConfig(
                api_key="ovh-test", extra={"api_params": api_params},
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

    def test_per_call_tool_choice_overrides_profile(self):
        from shared.tool_id_map import name_to_id
        p = self._provider({"tool_choice": "required"})
        kwargs = {"tools": [1]}
        named = {"type": "function", "function": {"name": "x"}}
        p._apply_api_params(kwargs, tool_choice=named)
        # Per-call wins over the profile's "required"; the function name is
        # mapped to its wire id (name_to_id) like the tools array.
        assert kwargs["tool_choice"] == {
            "type": "function", "function": {"name": name_to_id("x")}}

    def test_non_dict_api_params_raises(self):
        with patch(
            "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = OVHcloudProvider()
            with pytest.raises(TypeError):
                p.initialize(ProviderConfig(
                    api_key="ovh-test", extra={"api_params": "required"},
                ))

    def test_extra_body_forwarded_to_create_kwargs(self):
        p = OVHcloudProvider()
        p._extra_body = {"guided_json": {"type": "object"}}
        kwargs = {"tools": [object()]}
        p._apply_api_params(kwargs, None)
        assert kwargs["extra_body"] == {"guided_json": {"type": "object"}}

    def test_complete_batch_forwards_api_params_to_create(self):
        # End-to-end: complete() must hand api_params to chat.completions.create.
        p = self._provider({"temperature": 0.0, "tool_choice": "required",
                            "max_tokens": 256})
        p._model_name = "gpt-oss-120b"
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


# ==================== Auth info ====================

class TestGetAuthInfo:
    """get_auth_info describes the credential source actually in use."""

    def test_anonymous_tier(self):
        p = OVHcloudProvider()
        p._allow_anonymous = True
        with patch.dict("os.environ", {}, clear=True):
            assert "anonymous" in p.get_auth_info().lower()

    def test_env_key(self):
        p = OVHcloudProvider()
        p._api_key = "ovh-test"
        with patch.dict("os.environ", {"JAATO_OVHCLOUD_API_KEY": "ovh-test"}):
            assert "JAATO_OVHCLOUD_API_KEY" in p.get_auth_info()

    def test_vendor_env_key(self):
        p = OVHcloudProvider()
        p._api_key = "ovh-test"
        with patch.dict(
            "os.environ",
            {"OVH_AI_ENDPOINTS_ACCESS_TOKEN": "ovh-test"},
            clear=True,
        ):
            assert "OVH_AI_ENDPOINTS_ACCESS_TOKEN" in p.get_auth_info()
