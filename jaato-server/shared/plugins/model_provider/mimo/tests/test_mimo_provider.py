"""Tests for the Xiaomi MiMo provider (docs/design/minimax-kimi-mimo-providers.md §6)."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    ToolSchema,
)
from shared.plugins.model_provider.base import ProviderConfig
from shared.tool_id_map import name_to_id

from ..provider import MODEL_CONTEXT_LIMITS, MiMoProvider, create_provider
from ..errors import (
    APIKeyNotFoundError,
    ContentFilteredError,
    InfrastructureError,
    QuotaExhaustedError,
    RateLimitError,
    RegionDeniedError,
)
from ..env import (
    DEFAULT_BASE_URL,
    is_self_hosted,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
)

CLIENT_CLASS = "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
STORED_KEY = "shared.plugins.model_provider.mimo.auth.get_stored_api_key"
STORED_CREDS = "shared.plugins.model_provider.mimo.auth.try_load_credentials_with_reason"


def _response(text="Hello!", tool_calls=None, finish_reason="stop", reasoning=None, usage=True):
    resp = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = tool_calls or []
    choice.message.reasoning_content = reasoning
    resp.choices = [choice]
    if usage:
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 20
        resp.usage.total_tokens = 30
        resp.usage.prompt_tokens_details = None
        resp.usage.completion_tokens_details = MagicMock(reasoning_tokens=7)
    else:
        resp.usage = None
    return resp


def _tool_call(name="test_tool", args='{"key": "value"}', call_id="call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return tc


def _connected(extra=None, model="mimo-v2.5-pro"):
    with patch(CLIENT_CLASS) as mc:
        mc.return_value = MagicMock()
        p = MiMoProvider()
        p.initialize(ProviderConfig(api_key="sk-test", extra=extra or {}))
    p._catalog_cache = []
    p.connect(model)
    return p


# ==================== Environment ====================

class TestEnvironment:
    def test_resolve_api_key_jaato_namespace_first(self):
        with patch.dict("os.environ", {"JAATO_MIMO_API_KEY": "sk-jaato", "MIMO_API_KEY": "sk-vendor"}):
            assert resolve_api_key() == "sk-jaato"

    def test_resolve_api_key_honours_the_vendor_variable(self):
        with patch.dict("os.environ", {"MIMO_API_KEY": "sk-vendor"}, clear=True):
            assert resolve_api_key() == "sk-vendor"

    def test_resolve_api_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(STORED_KEY, return_value=None):
                assert resolve_api_key() is None

    def test_resolve_base_url(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL == "https://api.xiaomimimo.com/v1"
        with patch.dict("os.environ", {"JAATO_MIMO_BASE_URL": "https://token-plan-sgp.xiaomimimo.com/v1"}):
            assert resolve_base_url().startswith("https://token-plan-sgp")

    def test_resolve_context_length(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_context_length() is None
        with patch.dict("os.environ", {"JAATO_MIMO_CONTEXT_LENGTH": "4096"}):
            assert resolve_context_length() == 4096
        with patch.dict("os.environ", {"JAATO_MIMO_CONTEXT_LENGTH": "lots"}):
            assert resolve_context_length() is None

    def test_is_self_hosted(self):
        assert is_self_hosted("http://localhost:8000/v1")
        assert is_self_hosted("http://10.1.2.3/v1")
        assert not is_self_hosted(DEFAULT_BASE_URL)


# ==================== Init / context ====================

class TestInitialization:
    def test_initialize_without_key_raises(self):
        with patch.dict("os.environ", {}, clear=True), patch(STORED_KEY, return_value=None):
            with pytest.raises(APIKeyNotFoundError) as exc:
                MiMoProvider().initialize(ProviderConfig())
        assert "JAATO_MIMO_API_KEY" in str(exc.value)
        assert "MIMO_API_KEY" in str(exc.value)

    def test_initialize_self_hosted_without_key(self):
        with patch(CLIENT_CLASS) as mc, patch.dict(
                "os.environ", {"JAATO_MIMO_BASE_URL": "http://localhost:8000/v1"}, clear=True), \
                patch(STORED_KEY, return_value=None):
            mc.return_value = MagicMock()
            p = MiMoProvider()
            p.initialize(ProviderConfig())
        assert p._api_key is None and p._client is not None

    def test_context_from_the_table_by_longest_prefix(self):
        assert _connected(model="mimo-v2.5-pro").get_context_limit() == 1_048_576
        assert _connected(model="MiMo-V2.5").get_context_limit() == 1_048_576

    def test_unknown_model_fails_loud_naming_the_knob(self):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            p = MiMoProvider()
            p.initialize(ProviderConfig(api_key="sk-test"))
        p._catalog_cache = []
        with pytest.raises(ValueError) as exc:
            p.connect("mimo-v9")
        assert "plugin_configs.mimo.context_length" in str(exc.value)
        assert "JAATO_MIMO_CONTEXT_LENGTH" in str(exc.value)

    def test_knob_beats_the_table(self):
        assert _connected({"context_length": 4096}).get_context_limit() == 4096

    def test_catalog_window_beats_the_table_when_reported(self):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            p = MiMoProvider()
            p.initialize(ProviderConfig(api_key="sk-test"))
        p._catalog_cache = [{"id": "mimo-v2.5-pro", "max_model_len": 262144}]
        p.connect("mimo-v2.5-pro")
        assert p.get_context_limit() == 262144

    def test_deprecated_v2_series_is_not_in_the_table(self):
        assert not any(k.startswith("mimo-v2-") for k in MODEL_CONTEXT_LIMITS)

    def test_bad_modalities_knob_raises(self):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            with pytest.raises(TypeError):
                MiMoProvider().initialize(ProviderConfig(api_key="sk", extra={"modalities": "image"}))


class TestModalities:
    def test_pro_is_text_only(self):
        assert _connected(model="mimo-v2.5-pro").modalities() == {"text"}

    def test_omni_model_accepts_images(self):
        assert _connected(model="mimo-v2.5").modalities() == {"text", "image"}

    def test_knob_asserts_for_an_unlisted_model(self):
        p = _connected({"modalities": ["text", "image"], "context_length": 1}, model="mimo-next")
        assert p.modalities() == {"text", "image"}

    def test_audio_is_never_declared(self):
        """The model listens; this wire has not been probed for input_audio."""
        assert "audio" not in _connected(model="mimo-v2.5").modalities()


# ==================== Thinking ====================

class TestThinking:
    def test_toggle_is_sent_explicitly_and_defaults_to_enabled(self):
        p = _connected()
        kw = {}
        p._apply_api_params(kw, None)
        assert kw["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_enable_thinking_false_disables(self):
        p = _connected({"api_params": {"enable_thinking": False}})
        kw = {}
        p._apply_api_params(kw, None)
        assert kw["extra_body"]["thinking"] == {"type": "disabled"}

    def test_profile_extra_body_wins_on_a_collision(self):
        p = _connected({"extra_body": {"thinking": {"type": "disabled"}, "x": 1}})
        kw = {}
        p._apply_api_params(kw, None)
        assert kw["extra_body"] == {"thinking": {"type": "disabled"}, "x": 1}

    @pytest.mark.parametrize("knob", ["thinking_level", "thinking_budget"])
    def test_framework_knobs_without_an_equivalent_are_refused(self, knob):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            with pytest.raises(ValueError) as exc:
                MiMoProvider().initialize(ProviderConfig(
                    api_key="sk", extra={"api_params": {knob: 3}}))
        assert knob in str(exc.value) and "enable_thinking" in str(exc.value)

    def test_enable_thinking_is_not_reported_as_unsupported(self, caplog):
        with caplog.at_level(logging.WARNING):
            _connected({"api_params": {"enable_thinking": True}})
        assert "unsupported" not in caplog.text

    def test_supports_thinking(self):
        assert _connected().supports_thinking() is True

    def test_set_thinking_config_flips_the_toggle(self):
        from jaato_sdk.plugins.model_provider.types import ThinkingConfig
        p = _connected()
        p.set_thinking_config(ThinkingConfig(enabled=False))
        assert p._thinking_request_fields() == {"thinking": {"type": "disabled"}}


# ==================== Tool choice / params ====================

class TestToolChoice:
    def test_auto_passes(self):
        p = _connected()
        kw = {"tools": [object()]}
        p._apply_api_params(kw, "auto")
        assert kw["tool_choice"] == "auto"

    @pytest.mark.parametrize("choice", ["required", "none",
                                        {"type": "function", "function": {"name": "signal_completion"}}])
    def test_anything_else_is_folded_to_auto_with_a_warning(self, choice, caplog):
        p = _connected()
        kw = {"tools": [object()]}
        with caplog.at_level(logging.WARNING):
            p._apply_api_params(kw, choice)
        assert kw["tool_choice"] == "auto"
        assert "mimo-v2.5-pro" in caplog.text and "auto" in caplog.text

    def test_dropped_without_tools(self):
        p = _connected()
        kw = {}
        p._apply_api_params(kw, "auto")
        assert "tool_choice" not in kw


class TestApiParams:
    def test_max_tokens_is_renamed_on_the_wire(self):
        p = _connected({"api_params": {"max_tokens": 4096, "temperature": 0.7}})
        kw = {}
        p._apply_api_params(kw, None)
        assert kw["max_completion_tokens"] == 4096 and "max_tokens" not in kw
        assert kw["temperature"] == 0.7

    def test_unsupported_sampling_keys_are_dropped(self):
        p = _connected({"api_params": {"frequency_penalty": 1, "seed": 3, "response_format": {"type": "json_object"}}})
        assert p._api_params == {"response_format": {"type": "json_object"}}


# ==================== Finish reasons / errors ====================

class TestFinishAndErrors:
    def test_repetition_truncation_is_max_tokens(self):
        p = _connected()
        assert p._map_finish_reason("repetition_truncation") == FinishReason.MAX_TOKENS
        assert p._map_finish_reason("stop") == FinishReason.STOP

    def test_batch_response_uses_the_vendor_finish_label(self):
        p = _connected()
        p._client.chat.completions.create = lambda **kw: _response(finish_reason="repetition_truncation")
        result = p.complete([Message.from_text(Role.USER, "go")])
        assert result.response.finish_reason == FinishReason.MAX_TOKENS
        assert result.response.usage.reasoning_tokens == 7

    def _status_error(self, status):
        import openai
        resp = MagicMock()
        resp.status_code = status
        resp.headers = {}
        return openai.APIStatusError(f"http {status}", response=resp, body=None)

    @pytest.mark.parametrize("status, cls", [
        (402, QuotaExhaustedError), (403, RegionDeniedError), (421, ContentFilteredError),
    ])
    def test_vendor_statuses_are_not_transient(self, status, cls):
        p = _connected()
        with pytest.raises(cls):
            p._handle_api_error(self._status_error(status))
        assert p.classify_error(cls()) is None

    def test_5xx_is_still_infrastructure(self):
        p = _connected()
        with pytest.raises(InfrastructureError):
            p._handle_api_error(self._status_error(503))

    def test_rate_limit_is_transient(self):
        p = _connected()
        assert p.classify_error(RateLimitError(retry_after=2))["transient"] is True
        assert p.get_retry_after(RateLimitError(retry_after=2)) == 2.0


# ==================== Reasoning replay ====================

class TestReasoningReplay:
    def test_second_request_of_a_tool_loop_carries_reasoning_content(self):
        """The whole reason this provider needs the seam: without
        reasoning_content next to tool_calls the vendor answers 400."""
        p = _connected()
        sent = {}

        def create(**kw):
            sent.update(kw)
            return _response(text="done")

        p._client.chat.completions.create = create
        history = [
            Message.from_text(Role.USER, "read x"),
            Message(role=Role.MODEL, parts=[
                Part(thought="I should read the file"),
                Part(function_call=FunctionCall(id="c1", name="readFile", args={"path": "x"})),
            ]),
            Message(role=Role.TOOL, parts=[Part(function_response=MagicMock(
                call_id="c1", name="readFile", result={"content": "hi"},
                model_suffix=None, untrusted=False, untrusted_source=None, attachments=None))]),
        ]
        p.complete(history, tools=[ToolSchema(name="readFile", description="d",
                                              parameters={"type": "object", "properties": {}})])
        assistant = sent["messages"][1]
        assert assistant["reasoning_content"] == "I should read the file"
        assert assistant["content"] == ""
        assert assistant["tool_calls"][0]["function"]["name"] == name_to_id("readFile")

    def test_batch_turn_reasoning_becomes_a_thought_part(self):
        p = _connected()
        p._client.chat.completions.create = lambda **kw: _response(
            text=None, tool_calls=[_tool_call()], finish_reason="tool_calls",
            reasoning="plan first")
        result = p.complete([Message.from_text(Role.USER, "go")])
        assert result.response.parts[0].thought == "plan first"
        assert result.response.parts[1].function_call is not None


# ==================== Auth introspection / catalog ====================

class TestVerifyAuth:
    def test_env_key(self):
        with patch.dict("os.environ", {"JAATO_MIMO_API_KEY": "sk"}):
            assert MiMoProvider().verify_auth() is True

    def test_vendor_env_key(self):
        with patch.dict("os.environ", {"MIMO_API_KEY": "sk"}, clear=True):
            assert MiMoProvider().verify_auth() is True

    def test_profile_key(self):
        with patch.dict("os.environ", {}, clear=True), patch(STORED_CREDS, return_value=(None, None)):
            assert MiMoProvider().verify_auth(config=ProviderConfig(extra={"api_key": "sk"})) is True
            assert MiMoProvider().verify_auth(config=ProviderConfig(api_key="sk")) is True

    def test_no_key_raises_or_returns_false(self):
        with patch.dict("os.environ", {}, clear=True), patch(STORED_CREDS, return_value=(None, None)):
            with pytest.raises(APIKeyNotFoundError):
                MiMoProvider().verify_auth()
            assert MiMoProvider().verify_auth(allow_interactive=True) is False

    def test_broken_credentials_surface_the_reason(self):
        with patch.dict("os.environ", {}, clear=True), \
                patch(STORED_CREDS, return_value=(None, "invalid JSON at x")):
            messages = []
            with pytest.raises(APIKeyNotFoundError):
                MiMoProvider().verify_auth(on_message=messages.append)
        assert any("invalid JSON" in m for m in messages)

    def test_self_hosted(self):
        with patch.dict("os.environ", {"JAATO_MIMO_BASE_URL": "http://localhost:1/v1"}, clear=True), \
                patch(STORED_CREDS, return_value=(None, None)):
            assert MiMoProvider().verify_auth() is True


class TestCatalog:
    def test_list_models_from_catalog(self):
        p = _connected()
        p._catalog_cache = [{"id": "mimo-v2.5"}, {"id": "mimo-v2.5-pro"}]
        assert p.list_models() == ["mimo-v2.5", "mimo-v2.5-pro"]
        assert p.list_models(prefix="mimo-v2.5-p") == ["mimo-v2.5-pro"]

    def test_fetch_sends_bearer_and_caches(self):
        p = _connected()
        p._catalog_cache = None
        resp = MagicMock()
        resp.json.return_value = {"data": [{"id": "mimo-v2.5"}]}
        with patch("httpx.get", return_value=resp) as get:
            assert p._fetch_catalog() == [{"id": "mimo-v2.5"}]
            assert p._fetch_catalog() == [{"id": "mimo-v2.5"}]
        assert get.call_count == 1
        assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"

    def test_fetch_failure_is_empty_and_not_cached(self):
        p = _connected()
        p._catalog_cache = None
        with patch("httpx.get", side_effect=OSError("down")):
            assert p._fetch_catalog() == []
        assert p._catalog_cache is None


class TestIdentity:
    def test_create_provider(self):
        assert isinstance(create_provider(), MiMoProvider)
        assert create_provider().name == "mimo"

    def test_auth_info(self):
        p = _connected()
        with patch.dict("os.environ", {"MIMO_API_KEY": "sk"}, clear=True):
            assert "MIMO_API_KEY" in p.get_auth_info()

    def test_shutdown(self):
        p = _connected()
        p.shutdown()
        assert p._client is None and not p.is_connected
