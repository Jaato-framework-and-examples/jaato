"""Tests for the Azure OpenAI provider.

The transport is the shared OpenAI-compat one and is covered by its own
suites; what is Azure's own — and what these cases defend — is everything
around the request: the two required configuration values that have no
defensible default, deployment-name routing, the two credential kinds, and
the modality table that deliberately refuses to key on a deployment name.
"""

import pytest
from unittest.mock import MagicMock, patch

from shared.plugins.model_provider.base import ProviderConfig

from ..env import (
    AUTH_AAD,
    AUTH_KEY,
    ENV_AZURE_API_KEY,
    ENV_VENDOR_API_KEY,
    get_checked_credential_locations,
    resolve_api_version,
    resolve_auth_method,
    resolve_context_length,
    resolve_deployment,
    resolve_endpoint,
)
from ..errors import APIKeyNotFoundError, ConfigurationError
from ..provider import AzureOpenAIProvider, create_provider

_ENDPOINT = "https://my-resource.openai.azure.com"
_VERSION = "2024-10-21"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "JAATO_AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
        "JAATO_AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT",
        "JAATO_AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_VERSION",
        "JAATO_AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT",
        "JAATO_AZURE_OPENAI_CONTEXT_LENGTH", "JAATO_AZURE_OPENAI_AUTH",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_stored_credentials(monkeypatch):
    monkeypatch.setattr(
        "shared.plugins.model_provider.azure_openai.auth.get_stored_api_key",
        lambda **_kw: None,
    )
    monkeypatch.setattr(
        "shared.plugins.model_provider.azure_openai.auth."
        "try_load_credentials_with_reason",
        lambda **_kw: (None, None),
    )


def _extra(**overrides):
    base = {
        "endpoint": _ENDPOINT,
        "api_version": _VERSION,
        "context_length": 128000,
    }
    base.update(overrides)
    return base


def _initialized(extra=None, api_key="azure-key-abcdef123456"):
    provider = AzureOpenAIProvider()
    with patch.object(AzureOpenAIProvider, "_create_client",
                      return_value=MagicMock()):
        provider.initialize(
            ProviderConfig(api_key=api_key, extra=extra or _extra()))
    return provider


# ==================== Identity ====================

class TestIdentity:
    def test_name_is_azure_openai(self):
        assert create_provider().name == "azure_openai"

    def test_it_shares_the_openai_compat_transport(self):
        from shared.plugins.model_provider._openai_compat.base import (
            OpenAICompatProvider,
        )
        assert isinstance(create_provider(), OpenAICompatProvider)


# ==================== Required configuration ====================

class TestRequiredConfiguration:
    def test_a_missing_endpoint_fails_at_init_naming_the_key(self):
        """Left to the first request this becomes a connection failure to a
        URL nobody meant to build."""
        provider = AzureOpenAIProvider()
        with pytest.raises(ConfigurationError) as caught:
            provider.initialize(ProviderConfig(
                api_key="k", extra={"api_version": _VERSION}))
        assert "endpoint" in str(caught.value)

    def test_a_missing_api_version_fails_at_init(self):
        provider = AzureOpenAIProvider()
        with pytest.raises(ConfigurationError) as caught:
            provider.initialize(ProviderConfig(
                api_key="k", extra={"endpoint": _ENDPOINT}))
        assert "api_version" in str(caught.value)

    def test_there_is_no_default_api_version(self):
        """A framework-chosen date would decide, invisibly, which request
        fields a user's deployment accepts."""
        assert resolve_api_version() is None

    def test_there_is_no_default_endpoint(self):
        assert resolve_endpoint() is None

    def test_the_vendors_own_env_vars_are_honored(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", _ENDPOINT)
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", _VERSION)
        assert resolve_endpoint() == _ENDPOINT
        assert resolve_api_version() == _VERSION

    def test_the_jaato_namespaced_vars_win(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://vendor.example")
        monkeypatch.setenv("JAATO_AZURE_OPENAI_ENDPOINT", _ENDPOINT)
        assert resolve_endpoint() == _ENDPOINT


# ==================== Credentials ====================

class TestCredentials:
    def test_key_auth_is_the_default(self):
        assert _initialized().auth_method == AUTH_KEY

    def test_a_missing_key_raises_with_the_places_checked(self):
        provider = AzureOpenAIProvider()
        with pytest.raises(APIKeyNotFoundError) as caught:
            provider.initialize(ProviderConfig(extra=_extra()))
        assert ENV_AZURE_API_KEY in str(caught.value)

    def test_entra_needs_no_key_at_all(self):
        """The correct configuration for many deployments has no key
        anywhere, so "no key found" cannot end the credential search."""
        provider = AzureOpenAIProvider()
        with patch.object(AzureOpenAIProvider, "_create_client",
                          return_value=MagicMock()):
            provider.initialize(ProviderConfig(extra=_extra(auth="aad")))
        assert provider.auth_method == AUTH_AAD
        assert provider._api_key is None

    def test_env_selects_entra(self, monkeypatch):
        monkeypatch.setenv("JAATO_AZURE_OPENAI_AUTH", "aad")
        assert resolve_auth_method() == "aad"
        provider = AzureOpenAIProvider()
        with patch.object(AzureOpenAIProvider, "_create_client",
                          return_value=MagicMock()):
            provider.initialize(ProviderConfig(extra=_extra()))
        assert provider.auth_method == AUTH_AAD

    def test_an_unknown_auth_kind_is_refused_not_defaulted(self):
        """Falling back to key auth would report a missing secret for a
        profile that deliberately asked for one that needs none."""
        provider = AzureOpenAIProvider()
        with pytest.raises(ValueError, match="auth"):
            provider.initialize(ProviderConfig(
                api_key="k", extra=_extra(auth="entra")))

    def test_checked_locations_lead_with_the_profile_knob(self):
        locations = get_checked_credential_locations(
            config=ProviderConfig(extra={}))
        assert locations[0] == \
            "plugin_configs.azure_openai.api_key (profile): not set"


class TestVerifyAuth:
    """Runs on an UNINITIALIZED instance; may never touch ``self._client``."""

    def test_profile_key_satisfies_the_pre_init_gate(self):
        provider = AzureOpenAIProvider()
        assert provider.verify_auth(config=ProviderConfig(api_key="k")) is True
        assert provider._client is None

    def test_env_key_satisfies_it(self, monkeypatch):
        monkeypatch.setenv(ENV_VENDOR_API_KEY, "k")
        assert AzureOpenAIProvider().verify_auth() is True

    def test_nothing_configured_raises(self):
        with pytest.raises(APIKeyNotFoundError):
            AzureOpenAIProvider().verify_auth()

    def test_entra_passes_without_a_key_when_azure_identity_is_present(self):
        provider = AzureOpenAIProvider()
        with patch("shared.plugins.model_provider.azure_openai.auth."
                   "azure_identity_available", return_value=True):
            assert provider.verify_auth(
                config=ProviderConfig(extra={"auth": "aad"})) is True

    def test_entra_without_azure_identity_says_which_package_is_missing(self):
        provider = AzureOpenAIProvider()
        said = []
        with patch("shared.plugins.model_provider.azure_openai.auth."
                   "azure_identity_available", return_value=False):
            result = provider.verify_auth(
                config=ProviderConfig(extra={"auth": "aad"}),
                on_message=said.append)
        assert result is False
        assert any("azure-identity" in line for line in said)

    def test_entra_does_not_mint_a_token_to_answer(self):
        """Acquiring one can mean an IMDS round trip or a browser flow;
        neither belongs in a pre-flight check."""
        provider = AzureOpenAIProvider()
        with patch("shared.plugins.model_provider.azure_openai.auth."
                   "build_token_provider") as build:
            with patch("shared.plugins.model_provider.azure_openai.auth."
                       "azure_identity_available", return_value=True):
                provider.verify_auth(config=ProviderConfig(
                    extra={"auth": "aad"}))
        build.assert_not_called()


# ==================== Client construction ====================

class TestClientConstruction:
    def _capture(self, extra, api_key="k"):
        captured = {}

        def fake_azure(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        module = MagicMock()
        module.AzureOpenAI = fake_azure
        provider = AzureOpenAIProvider()
        with patch("shared.plugins.model_provider._openai_compat._lazy."
                   "get_openai_module", return_value=module):
            provider.initialize(ProviderConfig(api_key=api_key, extra=extra))
        return provider, captured

    def test_the_azure_client_gets_the_endpoint_and_version(self):
        """The Azure client is what builds ``/openai/deployments/<name>/...``
        and attaches ``api-version`` to every request."""
        _, captured = self._capture(_extra())
        assert captured["azure_endpoint"] == _ENDPOINT
        assert captured["api_version"] == _VERSION
        assert captured["api_key"] == "k"

    def test_entra_passes_a_token_PROVIDER_not_a_token(self):
        """A header set once works until the first hour boundary and then
        401s for the rest of the session."""
        with patch("shared.plugins.model_provider.azure_openai.auth."
                   "azure_identity_available", return_value=True):
            with patch("shared.plugins.model_provider.azure_openai.auth."
                       "build_token_provider",
                       return_value=lambda: "token") as build:
                _, captured = self._capture(_extra(auth="aad"), api_key=None)
        assert callable(captured["azure_ad_token_provider"])
        assert "api_key" not in captured
        build.assert_called_once()

    def test_entra_without_azure_identity_names_the_extra_to_install(self):
        provider = AzureOpenAIProvider()
        with patch("shared.plugins.model_provider.azure_openai.auth."
                   "azure_identity_available", return_value=False):
            with pytest.raises(ConfigurationError) as caught:
                provider.initialize(ProviderConfig(extra=_extra(auth="aad")))
        assert "azure-identity" in str(caught.value)


# ==================== Deployment routing ====================

class TestDeploymentRouting:
    def test_the_model_field_carries_the_deployment_name(self):
        provider = _initialized()
        provider.connect("my-gpt4o-deployment")
        assert provider.model_name == "my-gpt4o-deployment"

    def test_the_env_default_fills_in_when_none_is_named(self, monkeypatch):
        monkeypatch.setenv("JAATO_AZURE_OPENAI_DEPLOYMENT", "shared-deploy")
        assert resolve_deployment() == "shared-deploy"
        provider = _initialized()
        provider.connect("")
        assert provider.model_name == "shared-deploy"

    def test_no_deployment_anywhere_fails_loud(self):
        provider = _initialized()
        with pytest.raises(ConfigurationError) as caught:
            provider.connect("")
        assert "DEPLOYMENT" in str(caught.value)


# ==================== Context window ====================

class TestContextWindow:
    def test_the_knob_resolves_it(self):
        provider = _initialized()
        provider.connect("my-deploy")
        assert provider.get_context_limit() == 128000

    def test_env_resolves_it(self, monkeypatch):
        monkeypatch.setenv("JAATO_AZURE_OPENAI_CONTEXT_LENGTH", "32768")
        assert resolve_context_length() == 32768
        provider = _initialized(_extra(context_length=None))
        provider.connect("my-deploy")
        assert provider.get_context_limit() == 32768

    def test_unconfigured_fails_loud_naming_the_deployment(self):
        """Azure lists deployments, not capacities, and the name says nothing
        about the model version behind it."""
        provider = _initialized(_extra(context_length=None))
        with pytest.raises(ValueError) as caught:
            provider.connect("my-deploy")
        assert "my-deploy" in str(caught.value)
        assert "no-fallback" in str(caught.value)


# ==================== Modalities ====================

class TestModalities:
    def test_a_deployment_name_alone_yields_the_text_floor(self):
        """Matching the table against a deployment name would be reading
        meaning into a string its owner was free to choose."""
        provider = _initialized()
        provider.connect("gpt-4o")   # a deployment that HAPPENS to be named so
        assert provider.modalities() == {"text"}

    def test_a_declared_model_name_makes_the_table_apply(self):
        provider = _initialized(_extra(model_name="gpt-4o"))
        provider.connect("prod-vision")
        assert "image" in provider.modalities()

    def test_the_knob_asserts_it_directly(self):
        provider = _initialized(_extra(modalities=["text", "image"]))
        provider.connect("prod-vision")
        assert provider.modalities() == {"text", "image"}

    def test_a_non_list_modalities_knob_is_refused(self):
        provider = AzureOpenAIProvider()
        with pytest.raises(TypeError, match="modalities"):
            provider.initialize(ProviderConfig(
                api_key="k", extra=_extra(modalities="image")))


# ==================== Catalog ====================

class TestListModels:
    def test_a_failed_listing_is_empty_not_fatal(self):
        """A catalog that cannot be read must not stop a working session."""
        provider = _initialized()
        provider._client.models.list.side_effect = RuntimeError("403")
        assert provider.list_models() == []

    def test_ids_come_back_sorted_and_filterable(self):
        provider = _initialized()
        provider._client.models.list.return_value = [
            MagicMock(id="gpt-4o"), MagicMock(id="gpt-35-turbo"),
            MagicMock(id="text-embedding-3-large"),
        ]
        assert provider.list_models() == [
            "gpt-35-turbo", "gpt-4o", "text-embedding-3-large"]
        assert provider.list_models(prefix="gpt-4") == ["gpt-4o"]

    def test_no_client_yet_means_no_models(self):
        assert AzureOpenAIProvider().list_models() == []


# ==================== Wire policy ====================

class TestWirePolicy:
    def test_the_conservative_extensions_are_not_claimed(self):
        """PDFs and audio are gated on Azure by api-version and by what the
        deployment points at, so carrying them would be a guess about
        somebody else's resource."""
        assert AzureOpenAIProvider.WIRE_PDF_AS_FILE is False
        assert AzureOpenAIProvider.WIRE_AUDIO_AS_INPUT_AUDIO is False

    def test_the_declaration_agrees_with_the_wire_policy(self):
        from .. import PROVIDER_CAPABILITIES
        assert PROVIDER_CAPABILITIES.pdf_input is \
            AzureOpenAIProvider.WIRE_PDF_AS_FILE
        assert PROVIDER_CAPABILITIES.audio_input is \
            AzureOpenAIProvider.WIRE_AUDIO_AS_INPUT_AUDIO
