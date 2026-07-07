"""Lifecycle, connection, and contract tests for ChromeAIProvider."""

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
)

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.chrome_ai import env as env_mod
from shared.plugins.model_provider.chrome_ai.errors import (
    ChromeAIUnavailableError,
)
from shared.plugins.model_provider.chrome_ai.provider import ChromeAIProvider


class TestIdentity:
    def test_name(self):
        assert ChromeAIProvider().name == "chrome_ai"

    def test_list_models(self, connected_provider):
        assert connected_provider.list_models() == ["gemini-nano"]
        assert connected_provider.list_models(prefix="gem") == ["gemini-nano"]
        assert connected_provider.list_models(prefix="zzz") == []


class TestInitialize:
    def test_knobs_reach_the_bridge(self, fake_bridge, make_provider):
        provider = make_provider({
            "binary": "/opt/chrome",
            "user_data_dir": "/tmp/profile",
            "headless": False,
            "page_url": "https://example.com",
            "extra_args": ["--foo"],
            "connect_timeout": 7,
        })
        # find_browser_binary would reject a nonexistent explicit binary;
        # bypass filesystem checks for this wiring test.
        fake_bridge.quota = 4096
        import shared.plugins.model_provider.chrome_ai.provider as prov_mod
        original = prov_mod.find_browser_binary
        prov_mod.find_browser_binary = lambda explicit=None: explicit or "chrome"
        try:
            provider.connect("gemini-nano")
        finally:
            prov_mod.find_browser_binary = original
        assert fake_bridge.ctor_kwargs["binary"] == "/opt/chrome"
        assert fake_bridge.ctor_kwargs["user_data_dir"] == "/tmp/profile"
        assert fake_bridge.ctor_kwargs["headless"] is False
        assert fake_bridge.ctor_kwargs["page_url"] == "https://example.com"
        assert fake_bridge.ctor_kwargs["extra_args"] == ["--foo"]
        assert fake_bridge.ctor_kwargs["connect_timeout"] == 7

    def test_cdp_url_skips_binary_lookup(self, fake_bridge, make_provider):
        provider = make_provider({"cdp_url": "http://localhost:9222"})
        provider.connect("gemini-nano")
        assert fake_bridge.ctor_kwargs["cdp_url"] == "http://localhost:9222"
        assert fake_bridge.ctor_kwargs["binary"] is None


class TestConnect:
    @pytest.fixture(autouse=True)
    def _attach_mode(self, make_provider):
        """Connect via cdp_url so no binary lookup touches the host."""
        self.make = lambda extra=None: make_provider(
            {"cdp_url": "http://localhost:9222", **(extra or {})})

    def test_happy_path_detects_quota(self, fake_bridge):
        fake_bridge.quota = 9216
        provider = self.make()
        provider.connect("gemini-nano")
        assert provider.is_connected
        assert provider.model_name == "gemini-nano"
        assert provider.get_context_limit() == 9216

    def test_detection_beats_knob(self, fake_bridge):
        fake_bridge.quota = 9216
        provider = self.make({"context_length": 4096})
        provider.connect("gemini-nano")
        assert provider.get_context_limit() == 9216

    def test_knob_fallback_when_no_quota(self, fake_bridge):
        fake_bridge.quota = None
        provider = self.make({"context_length": 6144})
        provider.connect("gemini-nano")
        assert provider.get_context_limit() == 6144

    def test_env_fallback_when_no_quota(self, fake_bridge, monkeypatch):
        fake_bridge.quota = None
        monkeypatch.setenv("JAATO_CHROME_AI_CONTEXT_LENGTH", "5000")
        provider = self.make()
        provider.connect("gemini-nano")
        assert provider.get_context_limit() == 5000

    def test_unresolved_context_fails_loud(self, fake_bridge):
        fake_bridge.quota = None
        provider = self.make()
        with pytest.raises(ChromeAIUnavailableError, match="context_length"):
            provider.connect("gemini-nano")

    def test_skip_model_test_uses_manual_context(self, fake_bridge):
        fake_bridge.availability_value = "unavailable"  # would fail if checked
        provider = self.make({"context_length": 6144})
        provider.connect("gemini-nano", skip_model_test=True)
        assert provider.get_context_limit() == 6144

    def test_no_api_reports_requirements(self, fake_bridge):
        fake_bridge.availability_value = "no-api"
        provider = self.make()
        with pytest.raises(ChromeAIUnavailableError, match="Chrome 138"):
            provider.connect("gemini-nano")

    def test_unavailable_reports_hardware_gating(self, fake_bridge):
        fake_bridge.availability_value = "unavailable"
        provider = self.make()
        with pytest.raises(ChromeAIUnavailableError,
                           match="on-device-internals"):
            provider.connect("gemini-nano")

    def test_downloadable_without_auto_download(self, fake_bridge):
        fake_bridge.availability_value = "downloadable"
        provider = self.make()
        with pytest.raises(ChromeAIUnavailableError, match="auto_download"):
            provider.connect("gemini-nano")
        assert fake_bridge.download_calls == 0

    def test_downloadable_with_auto_download(self, fake_bridge):
        fake_bridge.availability_value = "downloadable"
        provider = self.make({"auto_download": True})
        provider.connect("gemini-nano")
        assert fake_bridge.download_calls == 1
        assert provider.is_connected


class TestVerifyAuth:
    def test_true_with_cdp_url_env(self, monkeypatch):
        monkeypatch.setenv("JAATO_CHROME_AI_CDP_URL", "http://localhost:9222")
        assert ChromeAIProvider().verify_auth() is True

    def test_true_with_cdp_url_config(self):
        config = ProviderConfig(extra={"cdp_url": "http://localhost:9222"})
        assert ChromeAIProvider().verify_auth(config=config) is True

    def test_reflects_binary_discovery(self, monkeypatch):
        messages = []
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(env_mod.os.path, "isfile", lambda path: False)
        assert ChromeAIProvider().verify_auth(
            on_message=messages.append) is False
        assert any("JAATO_CHROME_AI_BINARY" in m for m in messages)

        monkeypatch.setattr(
            env_mod.shutil, "which",
            lambda name: "/usr/bin/google-chrome"
            if name == "google-chrome" else None)
        assert ChromeAIProvider().verify_auth() is True

    def test_works_uninitialized(self, monkeypatch):
        # The runtime calls verify_auth on a fresh instance before
        # initialize(); it must not rely on initialize()-set state.
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(env_mod.os.path, "isfile", lambda path: False)
        provider = ChromeAIProvider()
        assert provider.verify_auth() is False  # no crash


class TestShutdown:
    def test_closes_bridge(self, fake_bridge, connected_provider):
        connected_provider.shutdown()
        assert fake_bridge.close_calls == 1
        assert connected_provider.is_connected is False


class TestTokensAndHistory:
    def test_count_tokens_heuristic(self):
        provider = ChromeAIProvider()
        assert provider.count_tokens("") == 0
        assert provider.count_tokens("abcd" * 10) == 10
        assert provider.count_tokens("ab") == 1

    def test_history_round_trip(self):
        provider = ChromeAIProvider()
        history = [
            Message.from_text(Role.USER, "hi"),
            Message(role=Role.MODEL, parts=[
                Part(text="calling"),
                Part(function_call=FunctionCall(id="c1", name="lookup",
                                                args={"q": "x"})),
            ]),
            Message(role=Role.TOOL, parts=[
                Part(function_response=ToolResult(
                    call_id="c1", name="lookup", result={"answer": 42})),
            ]),
        ]
        restored = provider.deserialize_history(
            provider.serialize_history(history))
        assert len(restored) == 3
        assert restored[0].text == "hi"
        assert restored[1].function_calls[0].name == "lookup"
        assert restored[2].parts[0].function_response.result == {"answer": 42}


class TestCapabilities:
    def test_flags(self, connected_provider):
        provider = connected_provider
        assert provider.supports_streaming() is True
        assert provider.supports_stop() is True
        assert provider.supports_structured_output() is True
        assert provider.supports_thinking() is False
        assert provider.modalities() == {"text"}

    def test_capabilities_constant(self):
        caps = ChromeAIProvider().capabilities()
        assert caps.streaming is True
        assert caps.cancellation is True
        assert caps.user_message_images is False

    def test_classify_error(self):
        from shared.plugins.model_provider.chrome_ai.errors import (
            ChromeAIConnectionError,
        )
        provider = ChromeAIProvider()
        verdict = provider.classify_error(ChromeAIConnectionError("boom"))
        assert verdict == {"transient": True, "rate_limit": False,
                           "infra": True}
        assert provider.classify_error(ValueError("x")) is None
