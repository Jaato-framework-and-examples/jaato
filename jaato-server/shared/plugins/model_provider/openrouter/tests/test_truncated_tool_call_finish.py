"""Guards for issue #745: a truncated turn must not present as tool-use.

A streamed turn that hits the model's output cap while it is
serializing a tool call carries the *fragments* of that call — the name
survives, ``arguments`` is severed part-way.  The provider used to end
every such turn with an unconditional::

    if function_calls and not was_cancelled:
        finish_reason = FinishReason.TOOL_USE

which computed ``MAX_TOKENS`` from the wire and then threw it away
precisely in the case where truncation matters most.  Downstream, that
is a silent wrong answer:

* :func:`shared.rewind.detect_truncated_tool_call` keys on
  ``MAX_TOKENS`` **together with** function calls, so the override made
  rewind-with-hint recovery structurally unreachable for streamed
  turns; and
* ``JaatoSession._classify_finish_reason`` never raised the
  abnormal-finish banner (#544), so an operator saw a turn that looked
  like it wanted a tool run.

Observed cost (OpenRouter activity export, gpt-5-mini, 2026-08-31): two
turns ran to a 65,536-token cap over 12-14 minutes each and were
reported as ``tool_calls``; together they were 44% of the arm's spend
and produced nothing.

The second half of the bug is vocabulary.  OpenRouter normalises
``finish_reason`` into the OpenAI four but reports the upstream's own
word in ``native_finish_reason``, and the export shows the pair
disagreeing: ``native_finish_reason="max_output_tokens"`` (OpenAI's
Responses API spelling, which the gpt-5 family is served over) against
``finish_reason="tool_calls"``.  Because it was not knowable from
outside which of the two crosses the SSE socket, both are handled: the
mapping accepts every truncation spelling, and the native field is
consulted whenever the normalised one is non-terminal.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    TERMINAL_FINISH_REASONS,
    resolve_tool_use_finish,
)

from ..converters import (
    TRUNCATION_FINISH_REASONS,
    extract_finish_reason,
    map_finish_reason,
    read_native_finish_reason,
    resolve_choice_finish_reason,
)
from ..provider import OpenRouterProvider


# ==================== Fixtures ====================


def _tool_call_delta(index=0, call_id="call_abc", name="write_file", args=""):
    """A streaming ``tool_calls`` delta entry as the SDK shapes it."""
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


def _make_chunk(
    *,
    content=None,
    finish_reason=None,
    native_finish_reason=None,
    tool_calls=None,
):
    """Build a streaming chunk carrying an explicit native finish reason.

    Mirrors ``test_openrouter_provider._make_chunk`` but adds the
    ``native_finish_reason`` sibling that OpenRouter puts on every
    choice.  ``model_extra`` is set to ``None`` unless a test wants the
    Pydantic-extras placement specifically, so the two read paths in
    :func:`read_native_finish_reason` can be exercised apart.
    """
    chunk = MagicMock()
    chunk.error = None
    chunk.model_extra = None
    chunk.usage = None

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.native_finish_reason = native_finish_reason
    choice.model_extra = None
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    delta.reasoning = None
    delta.reasoning_content = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _make_stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    stream.response = None
    return stream


def _build_provider():
    provider = OpenRouterProvider()
    provider._client = MagicMock()
    provider._model_name = "openai/gpt-5-mini"
    provider._enable_thinking = False
    return provider


def _stream(chunks):
    """Run ``_stream_response`` over ``chunks`` and return the response."""
    provider = _build_provider()
    provider._client.chat.completions.create = lambda **kw: _make_stream(chunks)
    return provider._stream_response(
        messages=[],
        kwargs={},
        on_chunk=lambda _t: None,
    )


# ==================== The shared resolution helper ====================


class TestResolveToolUseFinish:
    """``TOOL_USE`` is a fallback, not an override."""

    def test_fills_in_an_unreported_finish(self):
        assert resolve_tool_use_finish(
            FinishReason.UNKNOWN, has_function_calls=True,
        ) == FinishReason.TOOL_USE

    def test_upgrades_a_bare_stop(self):
        # Several upstreams report ``stop`` on a turn that did emit tool
        # calls; the accumulated calls are the better evidence.
        assert resolve_tool_use_finish(
            FinishReason.STOP, has_function_calls=True,
        ) == FinishReason.TOOL_USE

    def test_leaves_a_reason_alone_when_no_calls_accumulated(self):
        assert resolve_tool_use_finish(
            FinishReason.STOP, has_function_calls=False,
        ) == FinishReason.STOP

    @pytest.mark.parametrize("terminal", sorted(TERMINAL_FINISH_REASONS))
    def test_never_displaces_a_terminal_reason(self, terminal):
        assert resolve_tool_use_finish(
            terminal, has_function_calls=True,
        ) == terminal

    def test_the_regression_itself(self):
        # The one assertion the pre-fix code fails.
        assert resolve_tool_use_finish(
            FinishReason.MAX_TOKENS, has_function_calls=True,
        ) is not FinishReason.TOOL_USE


# ==================== Truncation vocabulary ====================


class TestTruncationSpellings:
    """Every upstream word for "the output cap was hit" maps to MAX_TOKENS."""

    @pytest.mark.parametrize("spelling", sorted(TRUNCATION_FINISH_REASONS))
    def test_spelling_maps_to_max_tokens(self, spelling):
        assert map_finish_reason(spelling) == FinishReason.MAX_TOKENS

    def test_max_output_tokens_is_no_longer_unknown(self):
        # The spelling the #745 export actually carried.  Before the fix
        # it wasn't in the mapping at all and fell through to UNKNOWN.
        assert map_finish_reason("max_output_tokens") == FinishReason.MAX_TOKENS

    def test_google_uppercase_folds_in(self):
        assert map_finish_reason("MAX_TOKENS") == FinishReason.MAX_TOKENS

    def test_an_unrecognised_word_is_still_unknown(self):
        # Not guessed at: an unfamiliar reason is a reason to look.
        assert map_finish_reason("banana") == FinishReason.UNKNOWN


# ==================== native_finish_reason ====================


class TestReadNativeFinishReason:
    def test_none_choice(self):
        assert read_native_finish_reason(None) is None

    def test_direct_attribute(self):
        choice = SimpleNamespace(native_finish_reason="max_output_tokens")
        assert read_native_finish_reason(choice) == "max_output_tokens"

    def test_pydantic_model_extra(self):
        # Where it actually lands on a real SDK ``Choice``, which
        # doesn't declare the field.
        choice = SimpleNamespace(
            model_extra={"native_finish_reason": "model_length"},
        )
        assert read_native_finish_reason(choice) == "model_length"

    def test_magicmock_is_not_mistaken_for_a_reported_value(self):
        # MagicMock auto-vivifies every attribute, so the helper must
        # require a genuine ``str`` before believing it.
        assert read_native_finish_reason(MagicMock()) is None

    def test_absent_field(self):
        assert read_native_finish_reason(SimpleNamespace()) is None


class TestResolveChoiceFinishReason:
    def test_native_truncation_beats_a_normalised_tool_calls(self):
        # The exact #745 pair from the activity export.
        choice = SimpleNamespace(
            finish_reason="tool_calls",
            native_finish_reason="max_output_tokens",
        )
        assert resolve_choice_finish_reason(choice) == FinishReason.MAX_TOKENS

    def test_native_truncation_beats_a_normalised_stop(self):
        choice = SimpleNamespace(
            finish_reason="stop",
            native_finish_reason="length",
        )
        assert resolve_choice_finish_reason(choice) == FinishReason.MAX_TOKENS

    def test_a_terminal_normalised_reason_is_not_second_guessed(self):
        # ``error`` is precise (OpenRouter's mid-stream disconnect); a
        # native truncation word must not overwrite it.
        choice = SimpleNamespace(
            finish_reason="error",
            native_finish_reason="max_output_tokens",
        )
        assert resolve_choice_finish_reason(choice) == FinishReason.ERROR

    def test_clean_tool_calls_still_resolves_to_tool_use(self):
        choice = SimpleNamespace(
            finish_reason="tool_calls",
            native_finish_reason="tool_calls",
        )
        assert resolve_choice_finish_reason(choice) == FinishReason.TOOL_USE

    def test_no_native_field_falls_back_to_the_normalised_one(self):
        choice = SimpleNamespace(finish_reason="length")
        assert resolve_choice_finish_reason(choice) == FinishReason.MAX_TOKENS


# ==================== End-to-end over the streaming path ====================


class TestTruncatedStreamDoesNotReportToolUse:
    """The guard the issue asked for, exercised through ``_stream_response``.

    A test that only streams a clean ``tool_calls`` finish passes both
    before and after the regression — which is how this survived — so
    each case here ends the stream on a truncation *while* tool-call
    deltas have been accumulated.
    """

    def test_length_finish_with_accumulated_calls(self):
        result = _stream([
            _make_chunk(content="Writing the file now."),
            _make_chunk(tool_calls=[
                _tool_call_delta(args='{"path": "a.py", "content": "de'),
            ]),
            _make_chunk(finish_reason="length"),
        ])
        assert result.finish_reason is not FinishReason.TOOL_USE
        assert result.finish_reason == FinishReason.MAX_TOKENS

    def test_native_max_output_tokens_under_a_tool_calls_finish(self):
        # The wire shape the #745 export implies, if OpenRouter sends
        # its normalised value down the SSE stream.
        result = _stream([
            _make_chunk(tool_calls=[
                _tool_call_delta(args='{"path": "a.py"'),
            ]),
            _make_chunk(
                finish_reason="tool_calls",
                native_finish_reason="max_output_tokens",
            ),
        ])
        assert result.finish_reason is not FinishReason.TOOL_USE
        assert result.finish_reason == FinishReason.MAX_TOKENS

    def test_raw_max_output_tokens_finish(self):
        # The other branch of the same uncertainty: if the raw value
        # crosses the socket, the widened mapping catches it.
        result = _stream([
            _make_chunk(tool_calls=[_tool_call_delta()]),
            _make_chunk(finish_reason="max_output_tokens"),
        ])
        assert result.finish_reason == FinishReason.MAX_TOKENS

    def test_content_filter_with_accumulated_calls(self):
        result = _stream([
            _make_chunk(tool_calls=[_tool_call_delta()]),
            _make_chunk(finish_reason="content_filter"),
        ])
        assert result.finish_reason == FinishReason.SAFETY

    def test_the_fragments_are_still_handed_back(self):
        # The turn failed, but whatever made it out stays on the
        # response — ``shared.rewind`` needs the partial call to name
        # the tool in its hint.
        result = _stream([
            _make_chunk(content="Writing it."),
            _make_chunk(tool_calls=[
                _tool_call_delta(name="write_file", args='{"path": "a.py'),
            ]),
            _make_chunk(finish_reason="length"),
        ])
        calls = [p.function_call for p in result.parts if p.function_call]
        assert [c.name for c in calls] == ["write_file"]

    def test_a_clean_tool_call_turn_is_unaffected(self):
        result = _stream([
            _make_chunk(tool_calls=[
                _tool_call_delta(args='{"path": "a.py", "content": "x"}'),
            ]),
            _make_chunk(finish_reason="tool_calls"),
        ])
        assert result.finish_reason == FinishReason.TOOL_USE

    def test_an_unreported_finish_with_calls_is_still_tool_use(self):
        # The case the override existed to serve: some upstreams report
        # nothing, and the accumulated calls are the only evidence.
        result = _stream([
            _make_chunk(tool_calls=[
                _tool_call_delta(args='{"path": "a.py"}'),
            ]),
        ])
        assert result.finish_reason == FinishReason.TOOL_USE

    def test_a_bare_stop_with_calls_is_still_tool_use(self):
        result = _stream([
            _make_chunk(tool_calls=[
                _tool_call_delta(args='{"path": "a.py"}'),
            ]),
            _make_chunk(finish_reason="stop"),
        ])
        assert result.finish_reason == FinishReason.TOOL_USE


# ==================== The batch path ====================


class TestBatchFinishReason:
    """A non-streamed turn can hit the cap mid-call just as easily."""

    def _response(self, finish_reason, native=None):
        choice = SimpleNamespace(
            finish_reason=finish_reason,
            native_finish_reason=native,
        )
        return SimpleNamespace(choices=[choice])

    def test_length_maps_to_max_tokens(self):
        assert extract_finish_reason(
            self._response("length")
        ) == FinishReason.MAX_TOKENS

    def test_native_truncation_is_seen(self):
        assert extract_finish_reason(
            self._response("tool_calls", native="max_output_tokens")
        ) == FinishReason.MAX_TOKENS

    def test_clean_tool_calls_unaffected(self):
        assert extract_finish_reason(
            self._response("tool_calls")
        ) == FinishReason.TOOL_USE

    def test_error_reason_now_reaches_the_framework(self):
        # ``extract_finish_reason`` previously had its own inlined
        # mapping that omitted ``error`` entirely; sharing
        # ``resolve_choice_finish_reason`` with the streaming path
        # closes that gap too.
        assert extract_finish_reason(
            self._response("error")
        ) == FinishReason.ERROR

    def test_no_choices(self):
        assert extract_finish_reason(
            SimpleNamespace(choices=[])
        ) == FinishReason.UNKNOWN
