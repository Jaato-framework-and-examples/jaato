"""The provider trace records the resolved tool name beside the wire id (#873).

Every hashing provider wrote a tool-call trace record naming the call only
by ``name=<wire id>`` — ``t_75386892`` — and the reverse map that resolves
it (``shared.tool_id_map._reverse``) is populated as a side effect of
hashing in the issuing process, so a later reader of the journal could
not recover the name at all.  These tests drive each provider's real
streaming loop with a hashed tool id on the wire and assert the record
now carries ``tool_name=<human name>`` beside the unchanged ``name``.

Two properties the issue asked for are pinned here rather than assumed:

* ``name`` keeps meaning "what the wire said" (existing readers do not
  change meaning); the resolution is the *additional* ``tool_name`` field;
* resolution is not destructive — a hallucinated id the process never
  issued is recorded as ``tool_name='t_deadbeef'``, not hidden.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.tool_id_map import name_to_id


REAL_NAME = "renderTemplateToFile"


def _records(lines, *prefixes):
    return [ln for ln in lines if ln.startswith(prefixes)]


def _assert_names(record, wire, human):
    assert f"name={wire!r}" in record, record
    assert f"tool_name={human!r}" in record, record


# ==================== OpenAI-compat base (nim, nebius, ovhcloud, ...) ======


def _oai_delta(index=0, call_id="call_abc", name="", args=""):
    return SimpleNamespace(
        index=index, id=call_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


def _oai_chunk(*, tool_calls=None, finish_reason=None):
    chunk = MagicMock()
    chunk.usage = None
    chunk.error = None
    chunk.model_extra = None
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.native_finish_reason = None
    choice.model_extra = None
    delta = MagicMock()
    delta.content = None
    delta.tool_calls = tool_calls
    delta.reasoning = None
    delta.reasoning_content = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _oai_stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    stream.response = None
    return stream


def _run_openai_shaped(provider, chunks):
    lines = []
    provider._trace = lines.append
    provider._client = MagicMock()
    provider._client.chat.completions.create = lambda **kw: _oai_stream(chunks)
    provider._enable_thinking = False
    provider._stream_response(messages=[], kwargs={}, on_chunk=lambda _t: None)
    return lines


def _nim():
    from shared.plugins.model_provider.nim.provider import NIMProvider
    p = NIMProvider()
    p._model_name = "meta/llama-3.3-70b-instruct"
    return p


def _openrouter():
    from shared.plugins.model_provider.openrouter.provider import OpenRouterProvider
    p = OpenRouterProvider()
    p._model_name = "openai/gpt-5-mini"
    return p


@pytest.mark.parametrize("build", [_nim, _openrouter], ids=["openai_compat", "openrouter"])
class TestOpenAIShapedStreams:

    def test_start_record_carries_the_resolved_name(self, build):
        wire = name_to_id(REAL_NAME)
        lines = _run_openai_shaped(build(), [
            _oai_chunk(tool_calls=[_oai_delta(name=wire, args='{"a": 1}')]),
            _oai_chunk(finish_reason="tool_calls"),
        ])
        (start,) = _records(lines, "TOOL_CALL_START")
        assert "id='call_abc'" in start
        _assert_names(start, wire, REAL_NAME)

    def test_end_record_carries_the_resolved_name(self, build):
        wire = name_to_id(REAL_NAME)
        lines = _run_openai_shaped(build(), [
            _oai_chunk(tool_calls=[_oai_delta(name=wire, args="{}")]),
            _oai_chunk(finish_reason="tool_calls"),
        ])
        (end,) = _records(lines, "TOOL_CALL_END")
        assert "idx=0 id='call_abc'" in end
        _assert_names(end, wire, REAL_NAME)

    def test_name_arriving_on_a_later_delta_is_still_recorded(self, build):
        """The START record is honest about an upstream that opens the call
        without a name; the END record is where the name is guaranteed."""
        wire = name_to_id(REAL_NAME)
        lines = _run_openai_shaped(build(), [
            _oai_chunk(tool_calls=[_oai_delta(name="")]),
            _oai_chunk(tool_calls=[_oai_delta(call_id=None, name=wire, args="{}")]),
            _oai_chunk(finish_reason="tool_calls"),
        ])
        (start,) = _records(lines, "TOOL_CALL_START")
        (end,) = _records(lines, "TOOL_CALL_END")
        _assert_names(start, "", "")
        _assert_names(end, wire, REAL_NAME)

    def test_hallucinated_id_is_recorded_as_itself(self, build):
        lines = _run_openai_shaped(build(), [
            _oai_chunk(tool_calls=[_oai_delta(name="t_deadbeef", args="{}")]),
            _oai_chunk(finish_reason="tool_calls"),
        ])
        for rec in _records(lines, "TOOL_CALL_START", "TOOL_CALL_END"):
            _assert_names(rec, "t_deadbeef", "t_deadbeef")


# ==================== GitHub Models (Copilot streaming path) ==============


def _copilot_choice(*, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=None, tool_calls=tool_calls)
    return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _run_copilot(choices):
    from shared.plugins.model_provider.github_models.provider import (
        GitHubModelsProvider,
    )
    p = GitHubModelsProvider()
    p._model_name = "gpt-4o"
    p._enable_thinking = False
    lines = []
    p._trace = lines.append
    p._copilot_client = MagicMock()
    p._copilot_client.complete_stream = lambda **kw: iter(choices)
    p._copilot_streaming_response(
        messages=[], tools=None, on_chunk=lambda _t: None)
    return lines


class TestGitHubModelsCopilotStream:

    def test_start_and_end_records_carry_the_resolved_name(self):
        wire = name_to_id(REAL_NAME)
        lines = _run_copilot([
            _copilot_choice(tool_calls=[{
                "index": 0, "id": "call_gh",
                "function": {"name": wire, "arguments": "{}"},
            }]),
            _copilot_choice(finish_reason="tool_calls"),
        ])
        (start,) = _records(lines, "TOOL_CALL_START")
        (end,) = _records(lines, "TOOL_CALL_END")
        assert "id='call_gh'" in start and "id='call_gh'" in end
        _assert_names(start, wire, REAL_NAME)
        _assert_names(end, wire, REAL_NAME)

    def test_hallucinated_id_is_recorded_as_itself(self):
        lines = _run_copilot([
            _copilot_choice(tool_calls=[{
                "index": 0, "id": "call_gh",
                "function": {"name": "t_deadbeef", "arguments": "{}"},
            }]),
            _copilot_choice(finish_reason="tool_calls"),
        ])
        for rec in _records(lines, "TOOL_CALL_START", "TOOL_CALL_END"):
            _assert_names(rec, "t_deadbeef", "t_deadbeef")


# ==================== Anthropic (also hashes on the wire) =================


def _anthropic_events(wire, call_id="toolu_01"):
    return [
        SimpleNamespace(
            type="content_block_start", index=0,
            content_block=SimpleNamespace(type="tool_use", id=call_id, name=wire),
        ),
        SimpleNamespace(
            type="content_block_delta", index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json="{}"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"), usage=None,
        ),
        SimpleNamespace(type="message_stop"),
    ]


def _run_anthropic(events):
    from shared.plugins.model_provider.anthropic.provider import AnthropicProvider
    p = AnthropicProvider()
    p._model_name = "claude-sonnet-4-20250514"
    lines = []
    p._trace = lines.append
    cm = MagicMock()
    cm.__enter__.return_value = events
    p._client = MagicMock()
    p._client.messages.stream = lambda **kw: cm
    p._stream_response(messages=[], kwargs={}, on_chunk=lambda _t: None)
    return lines


class TestAnthropicStream:
    """The issue counted Anthropic among the providers that write the real
    name — true of ``STREAM_FUNC_CALL``, but ``STREAM_TOOL_START`` logged
    the block's wire name, which ``anthropic/converters`` hashes too."""

    def test_tool_start_record_carries_the_resolved_name(self):
        wire = name_to_id(REAL_NAME)
        lines = _run_anthropic(_anthropic_events(wire))
        (start,) = _records(lines, "STREAM_TOOL_START")
        assert "id='toolu_01'" in start
        _assert_names(start, wire, REAL_NAME)

    def test_hallucinated_id_is_recorded_as_itself(self):
        lines = _run_anthropic(_anthropic_events("t_deadbeef"))
        (start,) = _records(lines, "STREAM_TOOL_START")
        _assert_names(start, "t_deadbeef", "t_deadbeef")
