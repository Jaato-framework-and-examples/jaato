"""Tests for the MiniMax provider (docs/design/minimax-kimi-mimo-providers.md §4)."""

import logging
from unittest.mock import MagicMock, patch

import openai
import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
)
from shared.plugins.model_provider.base import ProviderConfig

from ..provider import MODEL_CONTEXT_LIMITS, MiniMaxProvider, create_provider
from ..errors import (
    APIKeyNotFoundError,
    ContentFilteredError,
    QuotaExhaustedError,
    RateLimitError,
)
from ..env import DEFAULT_BASE_URL, resolve_api_key, resolve_base_url

CLIENT_CLASS = "shared.plugins.model_provider._openai_compat.base.get_openai_client_class"
STORED_KEY = "shared.plugins.model_provider.minimax.auth.get_stored_api_key"
STORED_CREDS = "shared.plugins.model_provider.minimax.auth.try_load_credentials_with_reason"


def _connected(extra=None, model="MiniMax-M3"):
    with patch(CLIENT_CLASS) as mc:
        mc.return_value = MagicMock()
        p = MiniMaxProvider()
        p.initialize(ProviderConfig(api_key="sk-test", extra=extra or {}))
    p._catalog_cache = []
    p.connect(model)
    return p


def _response(text="ok", reasoning=None, finish="stop"):
    resp = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish
    choice.message = MagicMock(content=text, tool_calls=[], reasoning_content=reasoning)
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                           prompt_tokens_details=None, completion_tokens_details=None)
    return resp


def _chunk(content=None, reasoning=None, details=None, finish=None):
    chunk = MagicMock()
    chunk.usage = None
    choice = MagicMock()
    choice.finish_reason = finish
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning
    delta.reasoning_details = details
    delta.tool_calls = None
    delta.audio = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


class TestEnvironment:
    def test_jaato_key_then_vendor_key(self):
        with patch.dict("os.environ", {"JAATO_MINIMAX_API_KEY": "a", "MINIMAX_API_KEY": "b"}):
            assert resolve_api_key() == "a"
        with patch.dict("os.environ", {"MINIMAX_API_KEY": "b"}, clear=True):
            assert resolve_api_key() == "b"

    def test_default_base_url(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL == "https://api.minimax.io/v1"


class TestContextAndModalities:
    @pytest.mark.parametrize("model, window", [
        ("MiniMax-M3", 1_000_000), ("MiniMax-M2.7", 204_800),
        ("MiniMax-M2.7-highspeed", 204_800), ("MiniMax-M2", 204_800), ("minimax-m2.1", 204_800),
    ])
    def test_table_by_longest_prefix_case_insensitive(self, model, window):
        assert _connected(model=model).get_context_limit() == window

    def test_knob_beats_the_table(self):
        assert _connected({"context_length": 4096}).get_context_limit() == 4096

    def test_unknown_model_fails_loud(self):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            p = MiniMaxProvider()
            p.initialize(ProviderConfig(api_key="sk"))
        p._catalog_cache = []
        with pytest.raises(ValueError) as exc:
            p.connect("MiniMax-Text-01")
        assert "JAATO_MINIMAX_CONTEXT_LENGTH" in str(exc.value)

    def test_m3_accepts_images_m2_does_not(self):
        assert _connected(model="MiniMax-M3").modalities() == {"text", "image"}
        assert _connected(model="MiniMax-M2.7").modalities() == {"text"}

    def test_every_table_model_is_reasoning_capable(self):
        for model in MODEL_CONTEXT_LIMITS:
            assert _connected(model=model).supports_thinking() is True


class TestThinking:
    def _fields(self, extra, model="MiniMax-M3"):
        p = _connected(extra, model=model)
        kw = {}
        p._apply_api_params(kw, None)
        return kw["extra_body"]

    def test_reasoning_split_is_always_requested(self):
        assert self._fields({}) == {"reasoning_split": True}

    def test_m3_toggle(self):
        assert self._fields({"api_params": {"enable_thinking": False}})["thinking"] == {"type": "disabled"}
        assert self._fields({"api_params": {"enable_thinking": True}})["thinking"] == {"type": "adaptive"}

    def test_m2_cannot_be_disabled(self, caplog):
        with caplog.at_level(logging.INFO):
            fields = self._fields({"api_params": {"enable_thinking": False}}, model="MiniMax-M2.7")
        assert "thinking" not in fields
        assert "cannot be disabled" in caplog.text

    @pytest.mark.parametrize("knob", ["thinking_level", "thinking_budget"])
    def test_knobs_without_an_equivalent_are_refused(self, knob):
        with patch(CLIENT_CLASS) as mc:
            mc.return_value = MagicMock()
            with pytest.raises(ValueError) as exc:
                MiniMaxProvider().initialize(ProviderConfig(api_key="sk", extra={"api_params": {knob: 1}}))
        assert knob in str(exc.value)

    def test_profile_extra_body_wins(self):
        assert self._fields({"extra_body": {"reasoning_split": False}}) == {"reasoning_split": False}


class TestReasoningChannels:
    def test_stream_reads_reasoning_details(self):
        p = _connected()
        stream = MagicMock()
        stream.__iter__ = lambda self: iter([
            _chunk(details=[{"type": "reasoning.text", "text": "think "}]),
            _chunk(details=[{"type": "reasoning.text", "text": "hard"}]),
            _chunk(content="answer", finish="stop"),
        ])
        p._client.chat.completions.create = lambda **kw: stream
        r = p._stream_response(messages=[], kwargs={}, on_chunk=lambda _t: None)
        assert r.thinking == "think hard"
        assert r.parts[0].thought == "think hard" and r.parts[1].text == "answer"

    def test_stream_prefers_reasoning_content_when_both_arrive(self):
        p = _connected()
        stream = MagicMock()
        stream.__iter__ = lambda self: iter([
            _chunk(reasoning="rc", details=[{"text": "rd"}]), _chunk(content="a", finish="stop")])
        p._client.chat.completions.create = lambda **kw: stream
        assert p._stream_response(messages=[], kwargs={}, on_chunk=lambda _t: None).thinking == "rc"

    def test_replay_echoes_both_spellings(self):
        p = _connected()
        sent = {}

        def create(**kw):
            sent.update(kw)
            return _response()

        p._client.chat.completions.create = create
        p.complete([
            Message.from_text(Role.USER, "q"),
            Message(role=Role.MODEL, parts=[
                Part(thought="why"), Part(function_call=FunctionCall(id="c1", name="t", args={}))]),
        ])
        assistant = sent["messages"][1]
        assert assistant["reasoning_content"] == "why"
        assert assistant["reasoning_details"] == [{
            "type": "reasoning.text", "id": "reasoning-text-1",
            "format": "MiniMax-response-v1", "index": 0, "text": "why"}]
        assert assistant["content"] == ""

    def test_leaked_think_block_is_moved_to_the_reasoning_channel(self):
        p = _connected(model="MiniMax-M2.7")
        p._client.chat.completions.create = lambda **kw: _response(
            text="<think>plan\nmore</think>\nThe answer.")
        result = p.complete([Message.from_text(Role.USER, "q")])
        assert result.response.thinking == "plan\nmore"
        assert result.response.parts[0].thought == "plan\nmore"
        assert result.response.get_text() == "The answer."

    def test_no_second_copy_when_split_delivered_reasoning(self):
        p = _connected()
        p._client.chat.completions.create = lambda **kw: _response(
            text="<think>x</think>y", reasoning="real")
        result = p.complete([Message.from_text(Role.USER, "q")])
        assert result.response.thinking == "real"
        assert [pt.thought for pt in result.response.parts] == ["real", None]
        assert result.response.get_text() == "y"


class TestParams:
    def test_max_tokens_renamed_and_defaulted(self):
        p = _connected(model="MiniMax-M3")
        kw = {}
        p._apply_api_params(kw, None)
        assert kw["max_completion_tokens"] == 131072
        p = _connected({"api_params": {"max_tokens": 4096}}, model="MiniMax-M2.7")
        kw = {}
        p._apply_api_params(kw, None)
        assert kw["max_completion_tokens"] == 4096 and "max_tokens" not in kw
        kw = {}
        _connected(model="MiniMax-M2.7")._apply_api_params(kw, None)
        assert kw["max_completion_tokens"] == 65536

    def test_penalties_and_json_mode_are_dropped(self):
        p = _connected({"api_params": {"presence_penalty": 1, "frequency_penalty": 1,
                                       "response_format": {"type": "json_object"},
                                       "service_tier": "priority", "temperature": 1.0}})
        assert p._api_params == {"service_tier": "priority", "temperature": 1.0}

    def test_tool_choice_vocabulary(self, caplog):
        p = _connected()
        kw = {"tools": [object()]}
        p._apply_api_params(kw, "none")
        assert kw["tool_choice"] == "none"
        kw = {"tools": [object()]}
        with caplog.at_level(logging.WARNING):
            p._apply_api_params(kw, "required")
        assert kw["tool_choice"] == "auto" and "MiniMax-M3" in caplog.text


class TestErrors:
    def _err(self, cls, status, message):
        resp = MagicMock()
        resp.status_code = status
        resp.headers = {}
        return cls(message, response=resp, body={"type": "x", "message": message})

    def test_2056_plan_window_is_quota_with_reset_time(self):
        p = _connected()
        err = self._err(openai.RateLimitError, 429,
                        "usage limit exceeded, 5-hour usage limit reached for Token Plan, "
                        "resets at 2026-09-08T17:00:00Z (2056)")
        with pytest.raises(QuotaExhaustedError) as exc:
            p._handle_api_error(err)
        assert exc.value.resets_at == "2026-09-08T17:00:00Z"
        assert p.classify_error(exc.value) is None

    def test_1008_balance_is_quota(self):
        p = _connected()
        with pytest.raises(QuotaExhaustedError):
            p._handle_api_error(self._err(openai.APIStatusError, 403, "insufficient balance (1008)"))

    def test_sensitive_content_is_filtered(self):
        p = _connected()
        with pytest.raises(ContentFilteredError):
            p._handle_api_error(self._err(openai.BadRequestError, 400, "output sensitive (1027)"))

    def test_plain_429_stays_transient(self):
        p = _connected()
        with pytest.raises(RateLimitError) as exc:
            p._handle_api_error(self._err(openai.RateLimitError, 429, "rate limit (1002)"))
        assert p.classify_error(exc.value)["transient"] is True


class TestVerifyAuthAndIdentity:
    def test_env_and_vendor_env(self):
        with patch.dict("os.environ", {"JAATO_MINIMAX_API_KEY": "sk"}):
            assert MiniMaxProvider().verify_auth() is True
        with patch.dict("os.environ", {"MINIMAX_API_KEY": "sk"}, clear=True):
            assert MiniMaxProvider().verify_auth() is True

    def test_no_key(self):
        with patch.dict("os.environ", {}, clear=True), patch(STORED_CREDS, return_value=(None, None)):
            with pytest.raises(APIKeyNotFoundError):
                MiniMaxProvider().verify_auth()
            assert MiniMaxProvider().verify_auth(allow_interactive=True) is False

    def test_initialize_without_key_names_both_vars(self):
        with patch.dict("os.environ", {}, clear=True), patch(STORED_KEY, return_value=None):
            with pytest.raises(APIKeyNotFoundError) as exc:
                MiniMaxProvider().initialize(ProviderConfig())
        assert "MINIMAX_API_KEY" in str(exc.value)

    def test_identity(self):
        p = create_provider()
        assert p.name == "minimax" and p.replay_reasoning is True

    def test_batch_finish_reason(self):
        p = _connected()
        p._client.chat.completions.create = lambda **kw: _response(finish="length")
        assert p.complete([Message.from_text(Role.USER, "q")]).response.finish_reason == FinishReason.MAX_TOKENS
