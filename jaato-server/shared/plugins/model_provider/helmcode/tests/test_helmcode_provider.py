"""Tests for the Helmcode provider."""

import pytest
from unittest.mock import MagicMock, patch

from ..provider import HelmcodeProvider, create_provider
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import (
    Message,
    Role,
    ToolSchema,
)
from ..errors import (
    APIKeyNotFoundError,
    CreditsExhaustedError,
    RateLimitError,
    InfrastructureError,
)
from ..env import (
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
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
        with patch.dict("os.environ", {"JAATO_HELMCODE_API_KEY": "sk-test-test123"}):
            assert resolve_api_key() == "sk-test-test123"

    def test_resolve_api_key_from_vendor_env(self):
        """Helmcode's own docs export HELMCODE_API_KEY; honor it so a user
        already set up for the vendor's SDK examples needs no extra config."""
        with patch.dict("os.environ", {"HELMCODE_API_KEY": "sk-vendor"}, clear=True):
            assert resolve_api_key() == "sk-vendor"

    def test_jaato_env_wins_over_vendor_env(self):
        """The jaato-namespaced variable is the higher-priority source."""
        with patch.dict("os.environ", {
            "JAATO_HELMCODE_API_KEY": "sk-jaato",
            "HELMCODE_API_KEY": "sk-vendor",
        }, clear=True):
            assert resolve_api_key() == "sk-jaato"

    def test_resolve_api_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.helmcode.auth.get_stored_api_key",
                return_value=None,
            ):
                assert resolve_api_key() is None

    def test_resolve_base_url_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL
            assert DEFAULT_BASE_URL == "https://api.helmcode.com/v1"

    def test_resolve_base_url_from_env(self):
        with patch.dict("os.environ", {"JAATO_HELMCODE_BASE_URL": "http://localhost:8000/v1"}):
            assert resolve_base_url() == "http://localhost:8000/v1"

    def test_resolve_context_length_unset_is_none(self):
        # No hardcoded fallback (no-fallback rule): unset env → None, so the
        # provider raises a "not configured" error rather than guessing a default.
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_context_length() is None

    def test_resolve_context_length_from_env(self):
        with patch.dict("os.environ", {"JAATO_HELMCODE_CONTEXT_LENGTH": "262144"}):
            assert resolve_context_length() == 262144

    def test_resolve_context_length_invalid_is_none(self):
        with patch.dict("os.environ", {"JAATO_HELMCODE_CONTEXT_LENGTH": "not-a-number"}):
            assert resolve_context_length() is None

    def test_is_self_hosted_localhost(self):
        assert is_self_hosted("http://localhost:8000/v1") is True
        assert is_self_hosted("http://127.0.0.1:8000/v1") is True

    def test_is_self_hosted_private_network(self):
        assert is_self_hosted("http://192.168.1.100:8000/v1") is True
        assert is_self_hosted("http://10.0.0.5:8000/v1") is True

    def test_is_self_hosted_public(self):
        assert is_self_hosted(DEFAULT_BASE_URL) is False
        # An On-premise deployment on a routable corporate host still needs a
        # key — only unmistakably local addresses waive the credential check.
        assert is_self_hosted("https://helmcode.corp.example/v1") is False


# ==================== Provider Tests ====================

class TestAuthentication:
    """Tests for authentication and initialization."""

    def test_initialize_without_key_raises(self):
        """Should raise APIKeyNotFoundError if no key and not self-hosted."""
        provider = HelmcodeProvider()

        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.helmcode.auth.get_stored_api_key",
                return_value=None,
            ):
                with pytest.raises(APIKeyNotFoundError) as exc_info:
                    provider.initialize(ProviderConfig())

        message = str(exc_info.value)
        assert "JAATO_HELMCODE_API_KEY" in message
        assert "HELMCODE_API_KEY" in message

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_with_api_key(self, mock_client_class):
        """Should initialize with key from config.api_key."""
        mock_client_class.return_value = MagicMock()

        provider = HelmcodeProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-test-test", extra={"context_length": 262144},
        ))

        assert provider._api_key == "sk-test-test"
        assert provider._client is not None

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_HELMCODE_API_KEY": "sk-test-env"}, clear=True)
    def test_initialize_from_env(self, mock_client_class):
        """Should auto-detect key from JAATO_HELMCODE_API_KEY env var."""
        mock_client_class.return_value = MagicMock()

        provider = HelmcodeProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 262144}))

        assert provider._api_key == "sk-test-env"

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"HELMCODE_API_KEY": "sk-test-vendor"}, clear=True)
    def test_initialize_from_vendor_env(self, mock_client_class):
        """The vendor's own variable also initializes the provider."""
        mock_client_class.return_value = MagicMock()

        provider = HelmcodeProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 262144}))

        assert provider._api_key == "sk-test-vendor"

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_HELMCODE_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_initialize_self_hosted_no_key(self, mock_client_class):
        """Should initialize without key for self-hosted proxy endpoints."""
        mock_client_class.return_value = MagicMock()

        provider = HelmcodeProvider()
        with patch(
            "shared.plugins.model_provider.helmcode.auth.get_stored_api_key",
            return_value=None,
        ):
            provider.initialize(ProviderConfig(extra={"context_length": 262144}))

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
        provider = HelmcodeProvider()
        provider.initialize(ProviderConfig(api_key="sk-test-test"))  # no raise
        provider._catalog_cache = []  # empty catalog -> detect returns None
        with pytest.raises(ValueError, match="context_length could not be resolved"):
            provider.connect("qwen3.6")

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_connect_error_names_both_override_tiers(self, mock_client_class):
        """The fail-loud message must name the profile knob AND the env var,
        so the operator can act on it without reading the source."""
        mock_client_class.return_value = MagicMock()
        provider = HelmcodeProvider()
        provider.initialize(ProviderConfig(api_key="sk-test-test"))
        provider._catalog_cache = []
        with pytest.raises(ValueError) as exc_info:
            provider.connect("glm-5.3")
        message = str(exc_info.value)
        assert "plugin_configs.helmcode.context_length" in message
        assert "JAATO_HELMCODE_CONTEXT_LENGTH" in message
        assert "helmcode.com/docs/models" in message

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_HELMCODE_CONTEXT_LENGTH": "262144"}, clear=True)
    def test_env_knob_stashed_at_init_resolves_at_connect(self, mock_client_class):
        """The env override is stashed at init and used at connect when the
        catalog has no entry for the model."""
        mock_client_class.return_value = MagicMock()
        provider = HelmcodeProvider()
        provider.initialize(ProviderConfig(api_key="sk-test-test"))
        assert provider._context_length_knob == 262144
        assert provider._context_length == 0  # not resolved until connect
        provider._catalog_cache = []
        provider.connect("qwen3.6")
        assert provider._context_length == 262144

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_catalog_context_beats_knob_at_connect(self, mock_client_class):
        """Catalog auto-detect is PRIMARY: the per-model context length from
        GET /v1/models wins over the manual knob."""
        mock_client_class.return_value = MagicMock()
        with patch.dict("os.environ", {"JAATO_HELMCODE_CONTEXT_LENGTH": "200000"}):
            provider = HelmcodeProvider()
            provider.initialize(ProviderConfig(api_key="sk-test-test"))
            provider._catalog_cache = [
                {"id": "qwen3.6", "context_length": 262144},
            ]
            provider.connect("qwen3.6")
            assert provider._context_length == 262144  # catalog wins over 200000

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_stashes_profile_knob_over_env(self, mock_client_class):
        """Profile knob (config.extra.context_length) wins over the env tier
        when stashed at init."""
        mock_client_class.return_value = MagicMock()
        with patch.dict("os.environ", {"JAATO_HELMCODE_CONTEXT_LENGTH": "200000"}):
            provider = HelmcodeProvider()
            provider.initialize(ProviderConfig(
                api_key="sk-test-test", extra={"context_length": 262144},
            ))
            assert provider._context_length_knob == 262144

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_initialize_custom_base_url(self, mock_client_class):
        """Should use custom base_url from config.extra.

        This is also how an On-premise Helmcode deployment is reached: the
        same API, served from the customer's own hardware.
        """
        mock_client_class.return_value = MagicMock()

        provider = HelmcodeProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-test-test",
            extra={"base_url": "https://helmcode.corp.example/v1",
                   "context_length": 262144},
        ))

        assert provider._base_url == "https://helmcode.corp.example/v1"

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    def test_bad_modalities_knob_raises(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = HelmcodeProvider()
        with pytest.raises(TypeError, match="modalities"):
            provider.initialize(ProviderConfig(
                api_key="sk-test-test",
                extra={"modalities": "image"},  # must be a list, not a string
            ))


class TestVerifyAuth:
    """Tests for verify_auth (must work before initialize)."""

    def test_verify_auth_with_key(self):
        provider = HelmcodeProvider()
        with patch.dict("os.environ", {"JAATO_HELMCODE_API_KEY": "sk-test-test"}):
            assert provider.verify_auth() is True

    def test_verify_auth_with_vendor_key(self):
        provider = HelmcodeProvider()
        messages = []
        with patch.dict("os.environ", {"HELMCODE_API_KEY": "sk-vendor"}, clear=True):
            assert provider.verify_auth(on_message=messages.append) is True
        assert "HELMCODE_API_KEY" in "\n".join(messages)

    def test_verify_auth_honors_profile_extra_api_key(self):
        """The pre-init gate must accept a profile-supplied key in
        config.extra['api_key'] — the shape the runtime builds for
        verify_auth (ProviderConfig(extra=plugin_configs[helmcode])) when a
        profile sets plugin_configs.helmcode.api_key: pass://... ."""
        provider = HelmcodeProvider()
        cfg = ProviderConfig(extra={"api_key": "sk-test-from-profile"})
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.helmcode.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),  # no stored creds
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_honors_profile_top_level_api_key(self):
        """config.api_key (top-level) is also honored."""
        provider = HelmcodeProvider()
        cfg = ProviderConfig(api_key="sk-test-top-level")
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.helmcode.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_self_hosted(self):
        provider = HelmcodeProvider()
        with patch.dict("os.environ", {"JAATO_HELMCODE_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            with patch(
                "shared.plugins.model_provider.helmcode.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth() is True

    def test_verify_auth_no_key_raises(self):
        provider = HelmcodeProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.helmcode.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                with pytest.raises(APIKeyNotFoundError):
                    provider.verify_auth(allow_interactive=False)

    def test_verify_auth_no_key_returns_false(self):
        provider = HelmcodeProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.helmcode.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(allow_interactive=True) is False

    def test_verify_auth_surfaces_broken_credentials(self):
        """Broken credential file must surface the reason, not be
        swallowed into a generic "no key" error."""
        provider = HelmcodeProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.helmcode.auth.try_load_credentials_with_reason",
                return_value=(None, "invalid JSON at /tmp/helmcode_auth.json: Expecting value"),
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
        provider = HelmcodeProvider()
        provider._client = MagicMock()
        provider._catalog_cache = [
            {"id": "deepseek-v4-flash", "context_length": 1000000},
        ]
        provider.connect("deepseek-v4-flash")
        assert provider.model_name == "deepseek-v4-flash"
        assert provider.get_context_limit() == 1000000

    def test_is_connected(self):
        provider = HelmcodeProvider()
        assert provider.is_connected is False

        provider._client = MagicMock()
        assert provider.is_connected is False

        provider._model_name = "qwen3.6"
        assert provider.is_connected is True

    def test_stateless_no_session_state(self):
        """Provider should not hold any conversation state."""
        provider = HelmcodeProvider()
        assert not hasattr(provider, '_system_instruction')
        assert not hasattr(provider, '_tools')
        assert not hasattr(provider, '_history')


class TestCapabilities:
    """Tests for capability queries."""

    def test_supports_streaming(self):
        assert HelmcodeProvider().supports_streaming() is True

    def test_supports_structured_output(self):
        assert HelmcodeProvider().supports_structured_output() is True

    def test_supports_stop(self):
        assert HelmcodeProvider().supports_stop() is True

    def test_supports_thinking_default(self):
        assert HelmcodeProvider().supports_thinking() is False

    def test_supports_thinking_reasoning_models(self):
        """Every language model in Helmcode's own catalogue is documented as
        reasoning-capable."""
        provider = HelmcodeProvider()
        for model in ("glm-5.3", "deepseek-v4-flash", "qwen3.6", "gemma4"):
            provider._model_name = model
            assert provider.supports_thinking() is True, model

    def test_name(self):
        assert HelmcodeProvider().name == "helmcode"


class TestTokenManagement:
    """Tests for token counting and context limits."""

    def test_count_tokens(self):
        provider = HelmcodeProvider()
        assert provider.count_tokens("abcd") == 1
        assert provider.count_tokens("a" * 100) == 25

    def test_get_context_limit(self):
        provider = HelmcodeProvider()
        provider._context_length = 262144
        assert provider.get_context_limit() == 262144

    def test_get_token_usage(self):
        provider = HelmcodeProvider()
        assert provider.get_token_usage().total_tokens == 0


class TestErrorClassification:
    """Tests for error classification and retry logic."""

    def test_classify_rate_limit(self):
        provider = HelmcodeProvider()
        exc = RateLimitError(original_error="429")
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": True, "infra": False}

    def test_classify_infrastructure(self):
        provider = HelmcodeProvider()
        exc = InfrastructureError(status_code=500)
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": False, "infra": True}

    def test_classify_unknown(self):
        provider = HelmcodeProvider()
        result = provider.classify_error(ValueError("unknown"))
        assert result is None

    def test_credits_exhausted_is_not_transient(self):
        """A 402 cannot succeed on retry — the balance does not refill on its
        own.  Leaving it unclassified is what makes the turn fail fast with
        the actionable message instead of burning the backoff budget."""
        provider = HelmcodeProvider()
        exc = CreditsExhaustedError(model="claude-sonnet-5")
        assert provider.classify_error(exc) is None
        assert provider.get_retry_after(exc) is None

    def test_retry_after_rate_limit(self):
        provider = HelmcodeProvider()
        exc = RateLimitError(retry_after=30.0)
        assert provider.get_retry_after(exc) == 30.0

    def test_retry_after_other(self):
        provider = HelmcodeProvider()
        assert provider.get_retry_after(ValueError("x")) is None


# ==================== 402 credits_exhausted mapping ====================

class _FakeStatusError(Exception):
    """Stand-in for an SDK error carrying an HTTP status code.

    Not an ``openai`` exception subclass on purpose: it exercises the
    Helmcode-specific 402 branch, and anything it does NOT match falls
    through the base's isinstance chain without raising — which is
    exactly the "unmapped" behavior the fall-through tests assert.
    """

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class TestCreditsExhausted:
    """Helmcode's ``402 credits_exhausted`` is mapped to its own class.

    It can only come from the resold frontier models (billed per token
    from prepaid credit); everything the monthly plan covers keeps
    answering.  The base's mapping leaves 402 unhandled, so without this
    override the turn would fail with a raw SDK error and no remedy.
    """

    def _provider(self, model="claude-sonnet-5"):
        p = HelmcodeProvider()
        p._model_name = model
        return p

    def test_402_credits_exhausted_maps_to_credits_error(self):
        p = self._provider()
        err = _FakeStatusError(402, "Error code: 402 - {'error': "
                                    "{'code': 'credits_exhausted'}}")
        with pytest.raises(CreditsExhaustedError) as exc_info:
            p._handle_api_error(err)
        assert exc_info.value.model == "claude-sonnet-5"

    def test_402_message_names_the_remedy(self):
        p = self._provider()
        err = _FakeStatusError(402, "credits_exhausted")
        with pytest.raises(CreditsExhaustedError) as exc_info:
            p._handle_api_error(err)
        message = str(exc_info.value)
        assert "claude-sonnet-5" in message
        assert "Credits" in message          # where to top up
        assert "deepseek-v4-flash" in message  # the plan-covered alternative

    def test_402_matches_a_reworded_credit_body(self):
        """The documented code is ``credits_exhausted``; a reworded body that
        still says "credit" is matched too, so a copy change upstream does
        not silently demote this to an unmapped error."""
        p = self._provider()
        err = _FakeStatusError(402, "Payment required: prepaid credit balance is 0")
        with pytest.raises(CreditsExhaustedError):
            p._handle_api_error(err)

    def test_unrelated_402_falls_through_to_base(self):
        """A 402 that is not about credit is NOT claimed by this branch — it
        falls through to the base mapping (which leaves it unmapped rather
        than mislabelling it)."""
        p = self._provider()
        err = _FakeStatusError(402, "Payment required: subscription lapsed")
        p._handle_api_error(err)  # returns without raising

    def test_non_402_falls_through_to_base(self):
        p = self._provider()
        err = _FakeStatusError(403, "credits_exhausted")  # wrong status
        p._handle_api_error(err)  # returns without raising


class TestCreateProvider:
    """Tests for factory function."""

    def test_create_provider(self):
        provider = create_provider()
        assert isinstance(provider, HelmcodeProvider)
        assert provider.name == "helmcode"


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_clears_state(self):
        provider = HelmcodeProvider()
        provider._client = MagicMock()
        provider._model_name = "qwen3.6"

        provider.shutdown()

        assert provider._client is None
        assert provider._model_name is None


# ==================== Catalog + Modality Detection ====================

class TestCatalogAndModalities:
    """The provider bootstraps context + INPUT modalities from the
    ``GET /v1/models`` catalog when it reports them, degrading to the
    manual knobs / text floor otherwise (Helmcode's OpenAI-compat catalog
    schema is not pinned to one shape, and requires a key to inspect).
    """

    def _provider_with_catalog(self, catalog):
        provider = HelmcodeProvider()
        provider._catalog_cache = catalog  # seed cache; detect wins
        return provider

    def test_lookup_context_length(self):
        p = self._provider_with_catalog([
            {"id": "qwen3.6", "context_length": 262144},
        ])
        assert p._lookup_context_length("qwen3.6") == 262144
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
            {"id": "qwen3.6", "object": "model", "owned_by": "helmcode"},
        ])
        assert p._lookup_context_length("qwen3.6") is None

    def test_modality_text_only(self):
        """``glm-5.3`` reads and writes text and does not take images."""
        p = self._provider_with_catalog([
            {"id": "glm-5.3", "context_length": 1000000,
             "architecture": {"modality": "text->text"}},
        ])
        p._model_name = "glm-5.3"
        assert p.modalities() == {"text"}
        assert p.supports_modality("image") is False

    def test_modality_vision_input(self):
        p = self._provider_with_catalog([
            {"id": "qwen3.6", "context_length": 262144,
             "architecture": {"modality": "text+image->text"}},
        ])
        assert p.modalities(model="qwen3.6") == {"text", "image"}

    def test_modality_knob_when_model_absent(self):
        p = self._provider_with_catalog([])  # empty catalog -> detect None
        p._model_name = "gemma4"
        p._modalities_knob = ["text", "image"]
        assert p.modalities() == {"text", "image"}

    def test_modality_text_floor_when_unknown(self):
        p = self._provider_with_catalog([])
        p._model_name = "uncatalogued"
        assert p.modalities() == {"text"}  # never a false image claim

    def test_modality_missing_architecture_falls_through(self):
        p = self._provider_with_catalog([{"id": "gemma4", "context_length": 262144}])
        p._model_name = "gemma4"
        p._modalities_knob = ["text", "image"]
        assert p.modalities() == {"text", "image"}  # knob, since no arch field

    def test_list_models_from_catalog(self):
        """The account-scoped listing carries the resold frontier models
        alongside Helmcode's own."""
        p = self._provider_with_catalog([
            {"id": "qwen3.6"}, {"id": "deepseek-v4-flash"},
            {"id": "claude-sonnet-5"}, {"id": "gemma4"},
        ])
        assert p.list_models() == [
            "claude-sonnet-5", "deepseek-v4-flash", "gemma4", "qwen3.6",
        ]
        assert p.list_models(prefix="claude-") == ["claude-sonnet-5"]

    def test_fetch_catalog_parses_and_caches(self):
        p = HelmcodeProvider()
        p._api_key = "sk-test-key"
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"data": [{"id": "qwen3.6", "context_length": 1}]}
        fake_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            first = p._fetch_catalog()
            second = p._fetch_catalog()  # cached: no second network call
        assert first == [{"id": "qwen3.6", "context_length": 1}]
        assert second is first
        assert mock_get.call_count == 1
        args, kwargs = mock_get.call_args
        assert args[0] == f"{DEFAULT_BASE_URL}/models"

    def test_fetch_catalog_sends_bearer_auth(self):
        """The catalog GET carries the Bearer key.

        Helmcode's ``/v1/models`` requires auth — an unkeyed request is
        answered ``401 auth_error`` (verified live 2026-09-05) — and the
        listing is account-scoped, reflecting the key's entitlements.
        Regression guard against copying OVHcloud's anonymous fetch here.
        """
        p = HelmcodeProvider()
        p._api_key = "sk-test-key"
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"data": []}
        fake_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            p._fetch_catalog()
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"

    def test_fetch_catalog_network_failure_returns_empty_not_cached(self):
        p = HelmcodeProvider()
        with patch("httpx.get", side_effect=Exception("boom")):
            assert p._fetch_catalog() == []
        assert p._catalog_cache is None  # not cached -> next call retries


# ==================== api_params Passthrough ====================

class TestApiParams:
    """plugin_configs.helmcode.api_params forwarded onto
    chat.completions.create (the determinism + tool_choice knob; parity
    with the rest of the OpenAI-compat fleet)."""

    def _provider(self, api_params, env=None):
        with patch(
            "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = HelmcodeProvider()
            with patch.dict("os.environ", env or {}, clear=env is not None):
                p.initialize(ProviderConfig(
                    api_key="sk-test", extra={"api_params": api_params},
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
            p = HelmcodeProvider()
            with pytest.raises(TypeError):
                p.initialize(ProviderConfig(
                    api_key="sk-test", extra={"api_params": "required"},
                ))

    def test_extra_body_forwarded_to_create_kwargs(self):
        p = HelmcodeProvider()
        p._extra_body = {"guided_json": {"type": "object"}}
        kwargs = {"tools": [object()]}
        p._apply_api_params(kwargs, None)
        assert kwargs["extra_body"] == {"guided_json": {"type": "object"}}

    def test_complete_batch_forwards_api_params_to_create(self):
        # End-to-end: complete() must hand api_params to chat.completions.create.
        p = self._provider({"temperature": 0.0, "tool_choice": "required",
                            "max_tokens": 256})
        p._model_name = "deepseek-v4-flash"
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

    def test_env_key(self):
        p = HelmcodeProvider()
        p._api_key = "sk-test"
        with patch.dict("os.environ", {"JAATO_HELMCODE_API_KEY": "sk-test"}):
            assert "JAATO_HELMCODE_API_KEY" in p.get_auth_info()

    def test_vendor_env_key(self):
        p = HelmcodeProvider()
        p._api_key = "sk-test"
        with patch.dict("os.environ", {"HELMCODE_API_KEY": "sk-test"}, clear=True):
            assert "HELMCODE_API_KEY" in p.get_auth_info()

    def test_self_hosted(self):
        p = HelmcodeProvider()
        p._base_url = "http://localhost:8000/v1"
        with patch.dict("os.environ", {}, clear=True):
            assert "Self-hosted" in p.get_auth_info()


# ==================== Bare-catalog shape (fail-loud path) ====================

# Helmcode's `/v1/models` requires a key, so its entry schema could not be
# captured without an account (unlike the Doubleword/OVHcloud fixtures,
# which are verbatim live samples).  This fixture therefore asserts the
# BEHAVIOUR under the bare OpenAI shape a LiteLLM-style gateway commonly
# serves — the shape the provider must survive — without claiming it is
# what Helmcode returns.  Model ids and windows are the documented ones
# (https://helmcode.com/docs/models, 2026-09).
BARE_CATALOG_SAMPLE = [
    {"id": "qwen3.6", "object": "model", "created": 1771510633,
     "owned_by": "helmcode"},
    {"id": "deepseek-v4-flash", "object": "model", "created": 1771510633,
     "owned_by": "helmcode"},
    {"id": "claude-sonnet-5", "object": "model", "created": 1771510633,
     "owned_by": "helmcode"},
]


class TestBareCatalogShape:
    """When the catalog carries no metadata, the manual tier must take over.

    Both catalog lookups return None for a bare entry, so resolution falls
    through to the knob / text floor — and, with no knob, to a loud error.
    If Helmcode does enrich its listing, these stay true (they seed a bare
    fixture explicitly) while the auto-detect tier simply starts winning
    for real models.
    """

    def _provider(self):
        p = HelmcodeProvider()
        p._catalog_cache = BARE_CATALOG_SAMPLE
        return p

    def test_bare_catalog_reports_no_context_length(self):
        p = self._provider()
        for entry in BARE_CATALOG_SAMPLE:
            assert p._lookup_context_length(entry["id"]) is None

    def test_bare_catalog_classifies_no_modalities(self):
        """Even ``qwen3.6``, which does take image input, is unclassified —
        the knob is then the only way to declare it."""
        p = self._provider()
        for entry in BARE_CATALOG_SAMPLE:
            assert p._lookup_modalities(entry["id"]) is None

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_bare_catalog_fails_loud_without_knob(self, mock_client_class):
        """The model IS in the catalog, but the entry carries no window — so
        connect() must fail loud, not guess.

        Distinct from `test_connect_raises_when_context_unresolved`, which
        seeds an EMPTY catalog (model absent).  Absent-model and
        present-but-bare are different paths into the same fail-loud tier.
        """
        mock_client_class.return_value = MagicMock()
        p = HelmcodeProvider()
        p.initialize(ProviderConfig(api_key="sk-test-test"))
        p._catalog_cache = BARE_CATALOG_SAMPLE
        with pytest.raises(ValueError, match="context_length could not be resolved"):
            p.connect("deepseek-v4-flash")

    @patch("shared.plugins.model_provider._openai_compat.base.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_bare_catalog_resolves_via_knob(self, mock_client_class):
        """With the knob set, connect() succeeds and the manual window is
        what the session sees."""
        mock_client_class.return_value = MagicMock()
        p = HelmcodeProvider()
        p.initialize(ProviderConfig(
            api_key="sk-test-test", extra={"context_length": 1000000},
        ))
        p._catalog_cache = BARE_CATALOG_SAMPLE
        p.connect("deepseek-v4-flash")
        assert p.get_context_limit() == 1000000
