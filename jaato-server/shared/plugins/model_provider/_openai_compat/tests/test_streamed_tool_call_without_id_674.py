"""A streamed tool call that carries no id still reaches history usable (#674).

The OpenAI-shaped streaming loop accumulates tool-call deltas keyed by
``index`` and records the id only when the upstream sends one:

    "id": tc_delta.id,        # may be None at TOOL_CALL_START
    ...
    if tc_delta.id:
        acc["id"] = tc_delta.id

A third-party OpenAI-compatible endpoint that never sends one left the
accumulated id ``None``, and the loop built ``FunctionCall(id=None)``
anyway — one comment in the tree said so outright ("May be None - will
cause API error, which is correct").  It is not correct: the call reaches
history, the result the framework sends back cannot name it, and the
**next** turn is rejected with a pairing 400 several steps from the cause.
Claude Code hit the same input as a render error; this tree hits it as a
400.

The provider set on this path is large and several of its members front
self-hosted servers whose OpenAI compatibility is approximate — nim,
nebius, ovhcloud, doubleword, zhipuai_openai via ``_openai_compat``, plus
openrouter, vllm and the models-endpoint provider, each owning a copy of
this loop.  So the id is minted at the seam, and
:func:`shared.history_invariant.repair_history` remains the backstop for
histories written before that.

These tests drive the **real** streaming loop through a mock SDK stream,
so they fail if the mint is removed from it.
"""

from unittest.mock import MagicMock

import pytest

from shared.history_invariant import SYNTHETIC_CALL_ID_PREFIX
from shared.plugins.model_provider.nim.provider import NIMProvider
from shared.plugins.model_provider.openrouter.provider import OpenRouterProvider
from shared.plugins.model_provider.vllm.provider import VLLMProvider


def _tool_call_delta(index, *, call_id, name, arguments):
    """One streamed ``tool_calls`` delta, with ``id`` possibly ``None``."""
    tc = MagicMock()
    tc.index = index
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _chunk(tool_calls=None, finish_reason=None):
    chunk = MagicMock()
    chunk.usage = None
    choice = MagicMock()
    choice.finish_reason = finish_reason
    delta = MagicMock()
    delta.content = None
    delta.tool_calls = tool_calls
    delta.reasoning_content = None
    delta.audio = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    return stream


def _idless_stream(count=1):
    """A stream whose tool calls never carry an id — the defective wire."""
    deltas = [
        _tool_call_delta(i, call_id=None, name=f"tool_{i}",
                         arguments='{"x": 1}')
        for i in range(count)
    ]
    return [_chunk(tool_calls=deltas), _chunk(finish_reason="tool_calls")]


def _build(cls, model):
    provider = cls()
    provider._client = MagicMock()
    provider._model_name = model
    provider._enable_thinking = False
    provider._trace = lambda _msg: None
    return provider


PROVIDERS = [
    pytest.param(NIMProvider, "meta/llama-3.3-70b-instruct", id="openai_compat"),
    pytest.param(OpenRouterProvider, "meta-llama/llama-3.3-70b-instruct",
                 id="openrouter"),
    pytest.param(VLLMProvider, "Qwen/Qwen2.5-7B-Instruct", id="vllm"),
]


def _complete(provider, chunks):
    """Drive one turn through the provider's **streaming** loop.

    ``on_chunk`` is what selects it — without a chunk callback ``complete``
    takes the batched path, which is a different accumulator and not the
    one this issue is about.
    """
    provider._client.chat.completions.create = (
        lambda **kwargs: _stream(chunks))
    return provider.complete([], on_chunk=lambda _chunk: None)


def _calls_of(turn_result):
    """The ``FunctionCall``s of a completed turn, read off its parts.

    ``complete`` returns a ``TurnResult`` wrapping a ``ProviderResponse``
    whose ``parts`` are the ordered content; the calls are the parts
    carrying ``function_call``, which is exactly what the session appends
    to history.
    """
    response = getattr(turn_result, "response", turn_result)
    return [p.function_call for p in response.parts if p.function_call]


@pytest.mark.parametrize("cls,model", PROVIDERS)
class TestIdIsMintedAtTheSeam:
    """Every streaming loop that accumulates by index mints a usable id."""

    def test_a_single_id_less_call_gets_an_id(self, cls, model):
        calls = _calls_of(_complete(_build(cls, model), _idless_stream(1)))
        assert len(calls) == 1
        assert calls[0].id
        assert calls[0].id.startswith(SYNTHETIC_CALL_ID_PREFIX)

    def test_a_parallel_id_less_batch_gets_distinct_ids(self, cls, model):
        """Eight-wide is the default parallel width; ids must not collide."""
        calls = _calls_of(_complete(_build(cls, model), _idless_stream(8)))
        assert len(calls) == 8
        assert len({c.id for c in calls}) == 8

    def test_two_responses_do_not_reuse_the_same_id(self, cls, model):
        """``index`` repeats every turn; the per-response nonce separates them.

        Without it both turns mint ``..._0`` and one history holds two
        different calls under one id — the defect wearing the fix as a
        disguise.
        """
        first = _calls_of(_complete(_build(cls, model), _idless_stream(2)))
        second = _calls_of(_complete(_build(cls, model), _idless_stream(2)))
        assert not ({c.id for c in first} & {c.id for c in second})

    def test_an_upstream_supplied_id_is_never_replaced(self, cls, model):
        """Minting is a fallback, not a rewrite."""
        chunks = [
            _chunk(tool_calls=[_tool_call_delta(
                0, call_id="call_upstream_1", name="tool_0",
                arguments='{"x": 1}')]),
            _chunk(finish_reason="tool_calls"),
        ]
        calls = _calls_of(_complete(_build(cls, model), chunks))
        assert [c.id for c in calls] == ["call_upstream_1"]

    def test_an_id_arriving_on_a_later_delta_still_wins(self, cls, model):
        """The opening delta may be nameless AND idless; a later one fills in."""
        chunks = [
            _chunk(tool_calls=[_tool_call_delta(
                0, call_id=None, name="tool_0", arguments='{"x":')]),
            _chunk(tool_calls=[_tool_call_delta(
                0, call_id="call_late", name=None, arguments=' 1}')]),
            _chunk(finish_reason="tool_calls"),
        ]
        calls = _calls_of(_complete(_build(cls, model), chunks))
        assert [c.id for c in calls] == ["call_late"]
