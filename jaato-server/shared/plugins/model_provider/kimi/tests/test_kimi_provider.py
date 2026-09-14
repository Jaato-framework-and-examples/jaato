"""Tests for the Moonshot AI Kimi provider (docs/design/minimax-kimi-mimo-providers.md §5)."""

import logging
from unittest.mock import MagicMock, patch

import openai
import pytest

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ThinkingConfig,
    ToolSchema,
)
from shared.plugins.model_provider.base import ProviderConfig
from shared.tool_id_map import name_to_id

from ..provider import KimiProvider, _family, create_provider
from ..errors import (
    APIKeyNotFoundError,
    ContextLimitError,
    QuotaExhaustedError,
    RateLimitError,
)
from ..env import DEFAULT_BASE_URL, resolve_api_key, resolve_base_url

CLIENT_CLASS = "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
STORED_KEY = "shared.plugins.model_provider.kimi.auth.get_stored_api_key"
STORED_CREDS = "shared.plugins.model_provider.kimi.auth.try_load_credentials_with_reason"

CATALOG = [
    {"id": "kimi-k3", "object": "model", "owned_by": "moonshot", "context_length": 1048576,
     "supports_image_in": True, "supports_video_in": True, "supports_reasoning": True},
    {"id": "kimi-k2.7-code", "object": "model", "context_length": 262144,
     "supports_image_in": True, "supports_reasoning": True},
    {"id": "kimi-k2.6", "object": "model", "context_length": 262144,
     "supports_image_in": True, "supports_reasoning": True},
]


def _connected(extra=None, model="kimi-k3", catalog=CATALOG):
    with patch(CLIENT_CLASS) as mc:
        mc.return_value = MagicMock()
        p = KimiProvider()
        p.initialize(ProviderConfig(api_key="sk-test", extra=extra or {}))
    p._catalog_cache = list(catalog)
    p.connect(model)
    return p


def _response(text="ok", reasoning=None, finish="stop", cached=None):
    resp = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish
    choice.message = MagicMock(content=text, tool_calls=[], reasoning_content=reasoning)
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                           prompt_tokens_details=None, completion_tokens_details=None)
    resp.usage.cached_tokens = cached
    return resp


class TestEnvironment:
    def test_jaato_key_then_vendor_key(self):
        with patch.dict("os.environ", {"JAATO_KIMI_API_KEY": "a", "MOONSHOT_API_KEY": "b"}):
            assert resolve_api_key() == "a"
        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "b"}, clear=True):
            assert resolve_api_key() == "b"

    def test_default_base_url(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL == "https://api.moonshot.ai/v1"


class TestFamilies:
    @pytest.mark.parametrize("model, family", [
        ("kimi-k3", "k3"), ("kimi-k3[1m]", "k3"), ("k3-256k", "k3"),
        ("kimi-k2.7-code", "k2.7-code"), ("kimi-k2.7-code-highspeed", "k2.7-code"),
        ("kimi-k2.6", "k2.6"), ("kimi-latest", ""), (None, ""),
    ])
    def test_family(self, model, family):
        assert _family(model) == family


class TestContextAndModalities:
    def test_catalog_is_primary(self):
        assert _connected(model="kimi-k3").get_context_limit() == 1048576
        assert _connected(model="kimi-k2.6").get_context_limit() == 262144

    def test_catalog_beats_the_knob(self):
        assert _connected({"context_length": 4096}).get_context_limit() == 1048576

    def test_knob_when_the_catalog_lacks_the_model(self):
        assert _connected({"context_length": 4096}, model="k3-256k").get_context_limit() == 4096

    def test_no_table_retired_ids_fail_loud(self):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            p = KimiProvider()
            p.initialize(ProviderConfig(api_key="sk"))
        p._catalog_cache = list(CATALOG)
        with pytest.raises(ValueError) as exc:
            p.connect("kimi-k2-0905")
        assert "plugin_configs.kimi.context_length" in str(exc.value)
        assert "2026-08-31" in str(exc.value)

    def test_modalities_from_the_catalog_flags(self):
        assert _connected(model="kimi-k3").modalities() == {"text", "image"}

    def test_video_is_not_declared(self):
        assert "video" not in _connected(model="kimi-k3").modalities()

    def test_text_only_when_the_catalog_says_so(self):
        catalog = [{"id": "kimi-text", "context_length": 1, "supports_image_in": False}]
        assert _connected(model="kimi-text", catalog=catalog).modalities() == {"text"}

    def test_knob_for_an_unlisted_model(self):
        p = _connected({"context_length": 1, "modalities": ["text", "image"]}, model="k3-256k")
        assert p.modalities() == {"text", "image"}

    def test_supports_thinking_reads_the_catalog(self):
        assert _connected(model="kimi-k3").supports_thinking() is True
        catalog = [{"id": "kimi-plain", "context_length": 1, "supports_reasoning": False}]
        assert _connected(model="kimi-plain", catalog=catalog).supports_thinking() is False


class TestThinkingDialects:
    def _fields(self, extra, model):
        p = _connected(extra, model=model)
        kw = {}
        p._apply_api_params(kw, None)
        return kw.get("extra_body", {})

    def test_k3_sends_reasoning_effort_from_thinking_level(self):
        assert self._fields({"api_params": {"thinking_level": "high"}}, "kimi-k3") == {"reasoning_effort": "high"}

    def test_k3_sends_nothing_when_unset(self):
        assert self._fields({}, "kimi-k3") == {}

    def test_k3_level_vocabulary_is_validated(self):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            with pytest.raises(ValueError) as exc:
                KimiProvider().initialize(ProviderConfig(
                    api_key="sk", extra={"api_params": {"thinking_level": "medium"}}))
        assert "low" in str(exc.value) and "max" in str(exc.value)

    def test_k3_cannot_be_disabled(self, caplog):
        with caplog.at_level(logging.INFO):
            assert self._fields({"api_params": {"enable_thinking": False}}, "kimi-k3") == {}
        assert "cannot be disabled" in caplog.text

    def test_k27_code_sends_the_only_legal_shape(self):
        assert self._fields({"api_params": {"enable_thinking": False}}, "kimi-k2.7-code") == {
            "thinking": {"type": "enabled", "keep": "all"}}

    def test_k26_type_and_keep(self):
        assert self._fields({"api_params": {"enable_thinking": False}}, "kimi-k2.6") == {
            "thinking": {"type": "disabled"}}
        assert self._fields({"api_params": {"enable_thinking": True, "thinking_keep": "all"}}, "kimi-k2.6") == {
            "thinking": {"type": "enabled", "keep": "all"}}

    def test_k26_unset_is_the_vendor_default(self):
        assert self._fields({}, "kimi-k2.6") == {}

    def test_thinking_budget_is_refused(self):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            with pytest.raises(ValueError) as exc:
                KimiProvider().initialize(ProviderConfig(
                    api_key="sk", extra={"api_params": {"thinking_budget": 1000}}))
        assert "thinking_level" in str(exc.value)

    def test_bad_keep_is_refused(self):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            with pytest.raises(ValueError):
                KimiProvider().initialize(ProviderConfig(
                    api_key="sk", extra={"api_params": {"thinking_keep": "some"}}))

    def test_runtime_toggle_reaches_k26(self):
        p = _connected(model="kimi-k2.6")
        p.set_thinking_config(ThinkingConfig(enabled=False))
        assert p._thinking_request_fields() == {"thinking": {"type": "disabled"}}


class TestParamsAndTools:
    def test_sampling_parameters_never_reach_the_wire(self, caplog):
        with caplog.at_level(logging.WARNING):
            p = _connected({"api_params": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 32000,
                                           "prompt_cache_key": "task-1"}})
        kw = {}
        p._apply_api_params(kw, None)
        assert "temperature" not in kw and "top_p" not in kw
        assert kw["max_completion_tokens"] == 32000 and "max_tokens" not in kw
        assert kw["prompt_cache_key"] == "task-1"
        assert "temperature" in caplog.text

    def test_tool_choice_full_set_on_k3(self):
        p = _connected(model="kimi-k3")
        kw = {"tools": [object()]}
        p._apply_api_params(kw, "required")
        assert kw["tool_choice"] == "required"
        kw = {"tools": [object()]}
        p._apply_api_params(kw, {"type": "function", "function": {"name": "signal_completion"}})
        assert kw["tool_choice"]["function"]["name"] == name_to_id("signal_completion")

    def test_tool_choice_narrowed_on_k2x(self, caplog):
        p = _connected(model="kimi-k2.6")
        kw = {"tools": [object()]}
        with caplog.at_level(logging.WARNING):
            p._apply_api_params(kw, "required")
        assert kw["tool_choice"] == "auto" and "kimi-k2.6" in caplog.text
        kw = {"tools": [object()]}
        p._apply_api_params(kw, "none")
        assert kw["tool_choice"] == "none"

    def test_tools_are_stamped_strict_false_by_default(self):
        p = _connected()
        wire = p._wire_tools([ToolSchema(name="t", description="d",
                                         parameters={"type": "object", "properties": {}})])
        assert wire[0]["function"]["strict"] is False

    def test_strict_tools_knob(self):
        p = _connected({"api_params": {"strict_tools": True}})
        wire = p._wire_tools([ToolSchema(name="t", description="d",
                                         parameters={"type": "object", "properties": {}})])
        assert wire[0]["function"]["strict"] is True
        assert "strict_tools" not in p._api_params


class TestCacheAccounting:
    def test_top_level_cached_tokens(self):
        usage = MagicMock(cached_tokens=64, prompt_tokens_details=None)
        assert KimiProvider._extract_cache_tokens(usage) == 64

    def test_falls_back_to_the_openai_location(self):
        usage = MagicMock(cached_tokens=None)
        usage.prompt_tokens_details = MagicMock(cached_tokens=8)
        assert KimiProvider._extract_cache_tokens(usage) == 8

    def test_batch_response_normalises_the_hit(self):
        p = _connected()
        p._client.chat.completions.create = lambda **kw: _response(cached=40)
        result = p.complete([Message.from_text(Role.USER, "hi")])
        assert result.response.usage.cache_read_tokens == 40
        assert result.response.usage.prompt_tokens == 60


class TestErrors:
    def _err(self, cls, status, body, headers=None):
        resp = MagicMock()
        resp.status_code = status
        resp.headers = headers or {}
        return cls("boom", response=resp, body=body)

    def test_quota_exhausted_is_not_transient(self):
        p = _connected()
        err = self._err(openai.RateLimitError, 429, {"type": "exceeded_current_quota_error", "message": "x"})
        with pytest.raises(QuotaExhaustedError):
            p._handle_api_error(err)
        assert p.classify_error(QuotaExhaustedError()) is None

    def test_quota_type_under_a_nested_error_key(self):
        p = _connected()
        err = self._err(openai.RateLimitError, 429, {"error": {"type": "exceeded_current_quota_error"}})
        with pytest.raises(QuotaExhaustedError):
            p._handle_api_error(err)

    @pytest.mark.parametrize("kind", ["engine_overloaded_error", "rate_limit_reached_error"])
    def test_other_429s_are_transient_with_retry_after(self, kind):
        p = _connected()
        err = self._err(openai.RateLimitError, 429, {"type": kind}, headers={"retry-after": "3"})
        with pytest.raises(RateLimitError) as exc:
            p._handle_api_error(err)
        assert exc.value.retry_after == 3.0
        assert p.classify_error(exc.value)["transient"] is True

    def test_context_phrasing_maps_to_context_limit(self):
        p = _connected()
        err = self._err(openai.BadRequestError, 400, {"type": "invalid_request_error",
                                                      "message": "Input token length too long"})
        err.args = ("Input token length too long",)
        with pytest.raises(ContextLimitError):
            p._handle_api_error(err)


class TestReasoningReplay:
    def test_history_carries_reasoning_content(self):
        p = _connected()
        sent = {}

        def create(**kw):
            sent.update(kw)
            return _response()

        p._client.chat.completions.create = create
        p.complete([
            Message.from_text(Role.USER, "q"),
            Message(role=Role.MODEL, parts=[
                Part(thought="think"),
                Part(function_call=FunctionCall(id="c1", name="t", args={})),
            ]),
        ])
        assert sent["messages"][1]["reasoning_content"] == "think"
        assert sent["messages"][1]["content"] == ""

    def test_new_turn_reasoning_becomes_a_thought_part(self):
        p = _connected()
        p._client.chat.completions.create = lambda **kw: _response(reasoning="deep")
        result = p.complete([Message.from_text(Role.USER, "q")])
        assert result.response.parts[0].thought == "deep"


class TestVerifyAuthAndIdentity:
    def test_env_and_vendor_env(self):
        with patch.dict("os.environ", {"JAATO_KIMI_API_KEY": "sk"}):
            assert KimiProvider().verify_auth() is True
        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "sk"}, clear=True):
            assert KimiProvider().verify_auth() is True

    def test_profile_key(self):
        with patch.dict("os.environ", {}, clear=True), patch(STORED_CREDS, return_value=(None, None)):
            assert KimiProvider().verify_auth(config=ProviderConfig(extra={"api_key": "sk"})) is True

    def test_no_key(self):
        with patch.dict("os.environ", {}, clear=True), patch(STORED_CREDS, return_value=(None, None)):
            with pytest.raises(APIKeyNotFoundError):
                KimiProvider().verify_auth()
            assert KimiProvider().verify_auth(allow_interactive=True) is False

    def test_initialize_without_key_names_both_vars(self):
        with patch.dict("os.environ", {}, clear=True), patch(STORED_KEY, return_value=None):
            with pytest.raises(APIKeyNotFoundError) as exc:
                KimiProvider().initialize(ProviderConfig())
        assert "MOONSHOT_API_KEY" in str(exc.value)

    def test_identity(self):
        assert create_provider().name == "kimi"
        assert create_provider().replay_reasoning is True

    def test_list_models(self):
        assert _connected().list_models(prefix="kimi-k2") == ["kimi-k2.6", "kimi-k2.7-code"]
