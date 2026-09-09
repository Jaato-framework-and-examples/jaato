"""Tests for the native OpenAI provider (identity, config, credentials).

The Responses wire has its own suite (``test_responses_wire.py``); this one
covers what is true whichever wire is selected: credential and endpoint
resolution, the wire selector itself, the fail-loud context window, the
modality table, and the client construction that carries the billing
headers.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from shared.plugins.model_provider.base import ProviderConfig

from ..env import (
    API_CHAT,
    API_RESPONSES,
    DEFAULT_BASE_URL,
    ENV_OPENAI_API_KEY,
    ENV_VENDOR_API_KEY,
    get_checked_credential_locations,
    is_self_hosted,
    resolve_api,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
)
from ..errors import APIKeyNotFoundError
from ..provider import MODEL_INPUT_MODALITIES, OpenAIProvider, create_provider
from ..responses import RESPONSES_API_PARAMS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No ambient OpenAI configuration leaks into these tests.

    ``OPENAI_API_KEY`` in particular is very likely to be set on a
    developer's machine, and a test that silently passed because of it
    would be asserting nothing.
    """
    for var in (
        "JAATO_OPENAI_API_KEY", "OPENAI_API_KEY",
        "JAATO_OPENAI_BASE_URL", "OPENAI_BASE_URL",
        "JAATO_OPENAI_MODEL", "JAATO_OPENAI_CONTEXT_LENGTH",
        "JAATO_OPENAI_ORG_ID", "OPENAI_ORG_ID",
        "JAATO_OPENAI_PROJECT_ID", "OPENAI_PROJECT_ID",
        "JAATO_OPENAI_API",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_stored_credentials(monkeypatch):
    """The credential file on the developer's box is not this test's input."""
    monkeypatch.setattr(
        "shared.plugins.model_provider.openai.auth.get_stored_api_key",
        lambda **_kw: None,
    )
    monkeypatch.setattr(
        "shared.plugins.model_provider.openai.auth.try_load_credentials_with_reason",
        lambda **_kw: (None, None),
    )


def _initialized(extra=None, api_key="sk-test-key-123456"):
    """A provider through ``initialize()`` with a stubbed client."""
    provider = OpenAIProvider()
    with patch.object(OpenAIProvider, "_create_client", return_value=MagicMock()):
        provider.initialize(ProviderConfig(api_key=api_key, extra=extra or {}))
    return provider


# ==================== Identity ====================

class TestIdentity:
    def test_name_is_openai(self):
        assert create_provider().name == "openai"

    def test_factory_returns_a_fresh_instance(self):
        assert create_provider() is not create_provider()

    def test_default_base_url_is_openais_own(self):
        assert resolve_base_url() == DEFAULT_BASE_URL


# ==================== Credentials ====================

class TestCredentialResolution:
    def test_jaato_namespaced_key_is_found(self, monkeypatch):
        monkeypatch.setenv(ENV_OPENAI_API_KEY, "sk-jaato")
        assert resolve_api_key() == "sk-jaato"

    def test_vendor_key_is_honored(self, monkeypatch):
        """``OPENAI_API_KEY`` is what every SDK sample already sets."""
        monkeypatch.setenv(ENV_VENDOR_API_KEY, "sk-vendor")
        assert resolve_api_key() == "sk-vendor"

    def test_jaato_key_wins_over_the_vendors(self, monkeypatch):
        """A workspace that deliberately points elsewhere is not overridden."""
        monkeypatch.setenv(ENV_VENDOR_API_KEY, "sk-vendor")
        monkeypatch.setenv(ENV_OPENAI_API_KEY, "sk-jaato")
        assert resolve_api_key() == "sk-jaato"

    def test_profile_key_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv(ENV_OPENAI_API_KEY, "sk-env")
        provider = _initialized(api_key="sk-profile")
        assert provider._api_key == "sk-profile"

    def test_missing_key_raises_with_the_places_checked(self):
        provider = OpenAIProvider()
        with pytest.raises(APIKeyNotFoundError) as caught:
            provider.initialize(ProviderConfig(extra={}))
        message = str(caught.value)
        assert ENV_OPENAI_API_KEY in message
        assert ENV_VENDOR_API_KEY in message

    def test_a_local_proxy_needs_no_key(self):
        """The escape hatch for fronting the API through a local gateway."""
        provider = OpenAIProvider()
        with patch.object(OpenAIProvider, "_create_client",
                          return_value=MagicMock()):
            provider.initialize(ProviderConfig(extra={
                "base_url": "http://localhost:9999/v1",
                "context_length": 8192,
            }))
        assert provider._api_key is None

    def test_checked_locations_lead_with_the_profile_knob(self):
        locations = get_checked_credential_locations(
            config=ProviderConfig(extra={}))
        assert locations[0] == \
            "plugin_configs.openai.api_key (profile): not set"

    @pytest.mark.parametrize("url,expected", [
        ("http://localhost:1234/v1", True),
        ("http://127.0.0.1:1234/v1", True),
        ("https://api.openai.com/v1", False),
    ])
    def test_self_hosted_detection(self, url, expected):
        assert is_self_hosted(url) is expected


class TestVerifyAuth:
    """``verify_auth`` runs on an UNINITIALIZED instance (see the plugin
    guide): it may never touch ``self._client``."""

    def test_profile_key_satisfies_the_pre_init_gate(self):
        provider = OpenAIProvider()
        assert provider.verify_auth(config=ProviderConfig(api_key="sk-x")) is True
        assert provider._client is None

    def test_env_key_satisfies_it(self, monkeypatch):
        monkeypatch.setenv(ENV_VENDOR_API_KEY, "sk-x")
        assert OpenAIProvider().verify_auth() is True

    def test_nothing_configured_raises(self):
        with pytest.raises(APIKeyNotFoundError):
            OpenAIProvider().verify_auth()

    def test_a_broken_credential_file_says_so(self, monkeypatch):
        """"Present but corrupt" must not report as "not configured"."""
        monkeypatch.setattr(
            "shared.plugins.model_provider.openai.auth."
            "try_load_credentials_with_reason",
            lambda **_kw: (None, "invalid JSON at ~/.jaato/openai_auth.json"),
        )
        said = []
        with pytest.raises(APIKeyNotFoundError):
            OpenAIProvider().verify_auth(on_message=said.append)
        assert any("could not be loaded" in line for line in said)


# ==================== Wire selection ====================

class TestWireSelection:
    def test_chat_is_the_default(self):
        assert _initialized().api_mode == API_CHAT

    def test_profile_knob_selects_responses(self):
        assert _initialized({"api": "responses"}).api_mode == API_RESPONSES

    def test_env_selects_responses(self, monkeypatch):
        monkeypatch.setenv("JAATO_OPENAI_API", "responses")
        assert resolve_api() == "responses"
        assert _initialized().api_mode == API_RESPONSES

    def test_profile_knob_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("JAATO_OPENAI_API", "responses")
        assert _initialized({"api": "chat"}).api_mode == API_CHAT

    def test_an_unknown_wire_is_refused_not_defaulted(self):
        """Falling back to chat would answer a profile that asked for the
        Responses API with a session that never touches it."""
        provider = OpenAIProvider()
        with pytest.raises(ValueError, match="api"):
            provider.initialize(ProviderConfig(
                api_key="sk-x", extra={"api": "responzes"}))


class TestApiParamAllowList:
    def test_chat_wire_forwards_max_tokens(self):
        provider = _initialized({"api_params": {"max_tokens": 500}})
        assert provider._api_params["max_tokens"] == 500

    def test_responses_wire_forwards_max_output_tokens(self):
        provider = _initialized(
            {"api": "responses", "api_params": {"max_output_tokens": 500}})
        assert provider._api_params["max_output_tokens"] == 500

    def test_the_wrong_wires_knob_is_dropped_not_sent(self):
        """``max_tokens`` on the Responses wire is a 400; dropping it with a
        warning is what makes the mistake visible instead of opaque."""
        provider = _initialized(
            {"api": "responses", "api_params": {"max_tokens": 500}})
        assert "max_tokens" not in provider._api_params

    def test_the_allow_lists_actually_differ(self):
        assert "max_output_tokens" in RESPONSES_API_PARAMS
        assert "max_tokens" not in RESPONSES_API_PARAMS


# ==================== Context window ====================

class TestContextWindow:
    def test_profile_knob_resolves_it(self):
        provider = _initialized({"context_length": 400_000})
        provider.connect("gpt-5.1")
        assert provider.get_context_limit() == 400_000

    def test_env_resolves_it(self, monkeypatch):
        monkeypatch.setenv("JAATO_OPENAI_CONTEXT_LENGTH", "128000")
        assert resolve_context_length() == 128000
        provider = _initialized()
        provider.connect("gpt-4.1")
        assert provider.get_context_limit() == 128000

    def test_unconfigured_fails_loud_and_names_the_model(self):
        """OpenAI's catalog reports no window, so there is nothing to detect
        and nothing safe to assume."""
        provider = _initialized()
        with pytest.raises(ValueError) as caught:
            provider.connect("gpt-5.1")
        assert "gpt-5.1" in str(caught.value)
        assert "no-fallback" in str(caught.value)

    def test_a_bad_env_value_is_ignored_rather_than_crashing(self, monkeypatch):
        monkeypatch.setenv("JAATO_OPENAI_CONTEXT_LENGTH", "lots")
        assert resolve_context_length() is None


# ==================== Modalities ====================

class TestModalities:
    def test_text_floor_for_an_unknown_model(self):
        provider = _initialized({"context_length": 8192})
        provider.connect("some-future-model")
        assert provider.modalities() == {"text"}

    def test_vision_model_from_the_table(self):
        provider = _initialized({"context_length": 8192})
        provider.connect("gpt-4o")
        assert "image" in provider.modalities()

    def test_longest_prefix_wins(self):
        """``gpt-4o-audio`` must not inherit ``gpt-4o``'s vision profile — a
        shortest-match walk gives the audio model eyes and no ears."""
        provider = _initialized({"context_length": 8192})
        provider.connect("gpt-4o-audio-preview")
        mods = provider.modalities()
        assert "audio" in mods
        assert "image" not in mods

    def test_the_knob_overrides_the_table(self):
        provider = _initialized({
            "context_length": 8192,
            "modalities": ["text", "image", "audio"],
        })
        provider.connect("gpt-4o")
        assert provider.modalities() == {"text", "image", "audio"}

    def test_a_non_list_modalities_knob_is_refused(self):
        provider = OpenAIProvider()
        with pytest.raises(TypeError, match="modalities"):
            provider.initialize(ProviderConfig(
                api_key="sk-x", extra={"modalities": "image"}))

    def test_every_table_entry_includes_text(self):
        """Every text-completion model accepts text; an entry that forgot it
        would make the floor a ceiling."""
        for prefix, mods in MODEL_INPUT_MODALITIES.items():
            assert "text" in mods, prefix


# ==================== Client construction ====================

class TestClientConstruction:
    def test_billing_headers_are_passed_when_set(self):
        provider = OpenAIProvider()
        captured = {}

        def fake_class(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("shared.plugins.model_provider._openai_compat._lazy."
                   "get_openai_client_class", return_value=fake_class):
            provider.initialize(ProviderConfig(api_key="sk-x", extra={
                "organization": "org-abc",
                "project": "proj-xyz",
                "context_length": 8192,
            }))
        assert captured["organization"] == "org-abc"
        assert captured["project"] == "proj-xyz"
        assert captured["base_url"] == DEFAULT_BASE_URL

    def test_absent_headers_are_omitted_not_sent_empty(self):
        """An empty ``OpenAI-Project`` header is not the same as none."""
        provider = OpenAIProvider()
        captured = {}

        def fake_class(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("shared.plugins.model_provider._openai_compat._lazy."
                   "get_openai_client_class", return_value=fake_class):
            provider.initialize(ProviderConfig(api_key="sk-x", extra={}))
        assert "organization" not in captured
        assert "project" not in captured


# ==================== Thinking ====================

class TestThinking:
    def test_reasoning_is_claimed_only_on_the_responses_wire(self):
        """Chat Completions bills for the reasoning and returns none of it."""
        chat = _initialized({"context_length": 8192})
        chat.connect("o3-mini")
        assert chat.supports_thinking() is False

        responses = _initialized({"api": "responses", "context_length": 8192})
        responses.connect("o3-mini")
        assert responses.supports_thinking() is True

    def test_a_non_reasoning_model_claims_nothing_on_either_wire(self):
        provider = _initialized({"api": "responses", "context_length": 8192})
        provider.connect("gpt-4.1-mini")
        assert provider.supports_thinking() is False


# ==================== Wire policy ====================

class TestWirePolicy:
    def test_the_provider_declares_openais_extensions(self):
        """The class attributes and ``converters.py`` must agree — they are
        what the capability declaration is read against."""
        from .. import converters
        assert OpenAIProvider.WIRE_PDF_AS_FILE is converters.PDF_AS_FILE
        assert OpenAIProvider.WIRE_AUDIO_AS_INPUT_AUDIO is \
            converters.AUDIO_AS_INPUT_AUDIO

    def test_a_pdf_reaches_the_chat_wire_through_complete(self):
        """The end-to-end version of the conformance guard: the flags have to
        be threaded from the class attribute into the actual request."""
        provider = _initialized({"context_length": 8192})
        provider.connect("gpt-4.1")
        captured = {}
        provider._client.chat.completions.create = (
            lambda **kw: captured.update(kw) or _stub_chat_response())

        provider.complete([Message(role=Role.USER, parts=[
            Part(text="summarise"),
            Part(inline_data={"mime_type": "application/pdf",
                              "data": b"%PDF-1.4 x", "display_name": "d.pdf"}),
        ])])
        blocks = captured["messages"][-1]["content"]
        assert any(b.get("type") == "file" for b in blocks)


def _stub_chat_response():
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = MagicMock()
    choice.message.content = "ok"
    choice.message.tool_calls = []
    choice.message.reasoning_content = None
    response.choices = [choice]
    response.usage = MagicMock(
        prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response.usage.prompt_tokens_details = None
    return response
