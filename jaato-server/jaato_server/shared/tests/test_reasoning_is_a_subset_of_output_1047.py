"""Reasoning tokens are one quantity, inside the output, and they reach the report (#1047).

Before this change ``TokenUsage`` carried two reasoning-shaped fields,
``reasoning_tokens`` ("OpenAI o-series") and ``thinking_tokens``
("Anthropic/Gemini"), for what is ONE quantity under different vendors'
names.  Nothing downstream read them as one:

* ``reasoning_tokens`` was extracted at the OpenAI-compatible seam and then
  reached no consumer — the consumption report and the THINKING budget
  entry read ``thinking_tokens`` only;
* Gemini's ``thoughts_token_count`` was never read, and Gemini reports it
  BESIDE ``candidates_token_count``, so ``output_tokens`` was the answer
  alone and every cost computed from it omitted reasoning Google bills at
  the output rate;
* a reported zero was folded into ``None`` at the OpenAI seam, and an
  ABSENT Gemini count (antigravity) was recorded as zero — both directions
  of the distinction ``reported_cache_count`` exists to keep;
* an Anthropic/Bedrock estimate from the thinking text sat in the same
  field as a measurement, with nothing saying which it was.

THE CONVENTION NOW.  ``reasoning_tokens`` is the one field
(``thinking_tokens`` is a deprecated alias, not a field), and it is a
SUBSET of ``output_tokens``.  A wire that reports it outside the output
converts at the seam (:func:`fold_exclusive_reasoning`), the #758 posture
applied to the output side.

PRICING.  Not separated: every vendor here bills reasoning at the output
rate and it is inside ``output_tokens``, so ``PricingTable`` already prices
it.  The report states the count and the answer share; it does not invent
a reasoning price.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    ProviderResponse,
    TokenUsage,
    fold_exclusive_reasoning,
    reported_reasoning_count,
)
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.model_provider._openai_compat.base import (
    OpenAICompatProvider,
)
from jaato_server.shared.plugins.model_provider.anthropic.converters import (
    _extract_thinking_tokens,
)
from jaato_server.shared.plugins.model_provider.antigravity.converters import (
    extract_usage_from_stream_chunk,
)
from jaato_server.shared.plugins.model_provider.google_genai.converters import (
    extract_usage_from_response,
)
from jaato_server.shared.plugins.model_provider.openai.responses_converters import (
    usage_from_responses,
)
from jaato_server.shared.plugins.model_provider.openrouter.converters import (
    apply_cache_usage,
)
from jaato_server.shared.session_consumption import (
    REASONING_SOURCE_ESTIMATE,
    REASONING_SOURCE_PROVIDER,
    ConsumptionLedger,
)
from jaato_server.shared.tests.reversion import Reversion

_TYPES = "jaato-sdk/jaato_sdk/plugins/model_provider/types.py"
_MP = "jaato-server/jaato_server/shared/plugins/model_provider/"
_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"
_CONSUMPTION = "jaato-server/jaato_server/shared/session_consumption.py"

REVERSIONS = [
    Reversion(
        target=_TYPES,
        find="TokenUsage.thinking_tokens = property(",
        replace="_thinking_tokens_alias_removed = property(",
        because="thinking_tokens stops being an alias, so writing it no "
                "longer reaches reasoning_tokens and the two drift apart",
        test="TestOneField::test_the_alias_writes_through",
    ),
    Reversion(
        target=_MP + "google_genai/converters.py",
        find="        fold_exclusive_reasoning(\n"
             "            usage, getattr(metadata, 'thoughts_token_count', None))\n",
        replace="",
        because="Gemini's thoughts go unread and output_tokens is the "
                "answer alone, so cost omits reasoning billed as output",
        test="TestSeams::test_gemini_thoughts_are_folded_into_the_output",
    ),
    Reversion(
        target=_MP + "_openai_compat/base.py",
        find="        return reported_reasoning_count(count)",
        replace="        return count if isinstance(count, int) and count else None",
        because="a reported zero reasoning count collapses into 'reports "
                "no reasoning' at the OpenAI-compatible seam",
        test="TestSeams::test_openai_compat_keeps_a_reported_zero",
    ),
    Reversion(
        target=_MP + "antigravity/converters.py",
        find='usage_metadata.get("thoughtsTokenCount"))',
        replace='usage_metadata.get("thoughtsTokenCount", 0))',
        because="a model that reports no thoughts is recorded as one that "
                "reasoned for zero tokens",
        test="TestSeams::test_antigravity_absent_thoughts_stay_unreported",
    ),
    Reversion(
        target=_MP + "openrouter/converters.py",
        find="        usage.reasoning_tokens = reasoning\n",
        replace="        pass\n",
        because="OpenRouter's completion_tokens_details.reasoning_tokens "
                "is dropped and the gateway reports one output figure",
        test="TestSeams::test_openrouter_reads_reasoning",
    ),
    Reversion(
        target=_SESSION,
        find="                reasoning_tokens=_reasoning_count(usage),\n",
        replace="",
        because="the consumption report stops receiving the reasoning "
                "count — the gap the issue was filed about",
        test="TestTheReport::test_openai_reasoning_reaches_the_report",
    ),
    Reversion(
        target=_CONSUMPTION,
        find="        if self.reasoning_responses < self.responses:\n"
             "            return None\n",
        replace="",
        because="answer_tokens is published over output whose reasoning "
                "was partly unreported, passing that reasoning off as answer",
        test="TestTheReport::test_answer_share_is_withheld_when_partly_unmeasured",
    ),
    Reversion(
        target=_MP + "anthropic/converters.py",
        find="            usage.reasoning_tokens_estimated = True\n",
        replace="",
        because="an estimate from the thinking text reads as a measurement",
        test="TestSeams::test_anthropic_estimate_says_it_is_one",
    ),
]


def _session():
    runtime = MagicMock()
    runtime.ledger = None
    runtime.registry = MagicMock()
    session = JaatoSession(runtime, "model-a")
    session._model_name = "model-a"
    session._active_provider_name = "prov-a"
    session._budget_tracker = None
    return session


def _response(usage):
    return ProviderResponse(parts=[], usage=usage,
                            finish_reason=FinishReason.STOP)


class TestOneField:

    def test_thinking_tokens_is_not_a_second_field(self):
        names = {f.name for f in dataclasses.fields(TokenUsage)}
        assert "reasoning_tokens" in names
        assert "thinking_tokens" not in names

    def test_the_old_constructor_keyword_still_works(self):
        assert TokenUsage(thinking_tokens=7).reasoning_tokens == 7

    def test_the_alias_writes_through(self):
        usage = TokenUsage()
        usage.thinking_tokens = 11
        assert usage.reasoning_tokens == 11
        assert usage.thinking_tokens == 11

    def test_a_reported_zero_is_a_measurement(self):
        assert reported_reasoning_count(0) == 0
        assert reported_reasoning_count(None) is None
        assert reported_reasoning_count(True) is None
        assert reported_reasoning_count(-3) is None

    def test_fold_puts_reasoning_inside_the_output(self):
        usage = fold_exclusive_reasoning(TokenUsage(output_tokens=10), 5)
        assert (usage.output_tokens, usage.reasoning_tokens) == (15, 5)
        untouched = fold_exclusive_reasoning(TokenUsage(output_tokens=10), None)
        assert (untouched.output_tokens, untouched.reasoning_tokens) == (10, None)


class TestSeams:

    def test_gemini_thoughts_are_folded_into_the_output(self):
        response = SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20,
            total_token_count=170, thoughts_token_count=50,
            cached_content_token_count=None))
        usage = extract_usage_from_response(response)
        assert usage.output_tokens == 70
        assert usage.reasoning_tokens == 50
        assert usage.total_tokens == 170

    def test_openai_compat_keeps_a_reported_zero(self):
        usage = SimpleNamespace(
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
        assert OpenAICompatProvider._extract_reasoning_tokens(usage) == 0
        assert OpenAICompatProvider._extract_reasoning_tokens(
            SimpleNamespace()) is None

    def test_antigravity_absent_thoughts_stay_unreported(self):
        chunk = {"usageMetadata": {"promptTokenCount": 10,
                                   "candidatesTokenCount": 5,
                                   "totalTokenCount": 15}}
        assert extract_usage_from_stream_chunk(chunk).reasoning_tokens is None
        chunk["usageMetadata"]["thoughtsTokenCount"] = 4
        usage = extract_usage_from_stream_chunk(chunk)
        assert (usage.output_tokens, usage.reasoning_tokens) == (9, 4)

    def test_openrouter_reads_reasoning(self):
        raw = SimpleNamespace(
            prompt_tokens_details=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30))
        usage = TokenUsage(prompt_tokens=10, output_tokens=40, total_tokens=50)
        apply_cache_usage(raw, usage)
        assert usage.reasoning_tokens == 30
        assert usage.output_tokens == 40   # already inclusive on this wire

    def test_responses_api_reads_reasoning(self):
        usage = usage_from_responses({
            "input_tokens": 10, "output_tokens": 40, "total_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 25}})
        assert (usage.output_tokens, usage.reasoning_tokens) == (40, 25)

    def test_anthropic_estimate_says_it_is_one(self):
        block = SimpleNamespace(type="thinking", thinking="x" * 400)
        usage = TokenUsage(output_tokens=200)
        _extract_thinking_tokens(SimpleNamespace(), SimpleNamespace(content=[block]),
                                 usage)
        assert usage.reasoning_tokens == 100
        assert usage.reasoning_tokens_estimated is True

    def test_anthropic_reported_count_is_not_an_estimate(self):
        usage = TokenUsage(output_tokens=200)
        _extract_thinking_tokens(
            SimpleNamespace(output_tokens_details=SimpleNamespace(thinking_tokens=0)),
            None, usage)
        assert usage.reasoning_tokens == 0
        assert usage.reasoning_tokens_estimated is False


class TestTheReport:

    def test_openai_reasoning_reaches_the_report(self):
        session = _session()
        session._accumulate_turn_tokens(_response(TokenUsage(
            prompt_tokens=100, output_tokens=40, total_tokens=140,
            reasoning_tokens=30)), {})
        row = session._consumption.bindings()[0].as_dict()
        assert row["reasoning_tokens"] == 30
        assert row["reasoning_source"] == REASONING_SOURCE_PROVIDER
        assert row["answer_tokens"] == 10

    def test_openai_reasoning_feeds_the_turn_too(self):
        session = _session()
        turn = {}
        session._accumulate_turn_tokens(_response(TokenUsage(
            prompt_tokens=100, output_tokens=40, total_tokens=140,
            reasoning_tokens=30)), turn)
        assert turn["thinking"] == 30

    def test_an_estimate_is_labelled_in_the_report(self):
        session = _session()
        session._accumulate_turn_tokens(_response(TokenUsage(
            prompt_tokens=100, output_tokens=40, total_tokens=140,
            reasoning_tokens=30, reasoning_tokens_estimated=True)), {})
        row = session._consumption.bindings()[0].as_dict()
        assert row["reasoning_source"] == REASONING_SOURCE_ESTIMATE

    def test_no_reasoning_reported_means_no_reasoning_keys(self):
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="m", uncached_input_tokens=10,
                       output_tokens=5, total_tokens=15)
        row = ledger.bindings()[0].as_dict()
        assert "reasoning_tokens" not in row
        assert "answer_tokens" not in row

    def test_a_reported_zero_is_published(self):
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="m", uncached_input_tokens=10,
                       output_tokens=5, total_tokens=15, reasoning_tokens=0)
        row = ledger.bindings()[0].as_dict()
        assert row["reasoning_tokens"] == 0
        assert row["answer_tokens"] == 5

    def test_answer_share_is_withheld_when_partly_unmeasured(self):
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="m", uncached_input_tokens=10,
                       output_tokens=40, total_tokens=50, reasoning_tokens=30)
        ledger.observe(provider="p", model="m", uncached_input_tokens=10,
                       output_tokens=40, total_tokens=50)
        row = ledger.bindings()[0].as_dict()
        assert row["reasoning_tokens"] == 30
        assert "answer_tokens" not in row

    def test_totals_mix_sources_and_never_add_reasoning_to_output(self):
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="a", uncached_input_tokens=10,
                       output_tokens=40, total_tokens=50, reasoning_tokens=30,
                       reasoning_source=REASONING_SOURCE_PROVIDER)
        ledger.observe(provider="p", model="b", uncached_input_tokens=10,
                       output_tokens=20, total_tokens=30, reasoning_tokens=5,
                       reasoning_source=REASONING_SOURCE_ESTIMATE)
        totals = ledger.totals().as_dict()
        assert totals["output_tokens"] == 60
        assert totals["reasoning_tokens"] == 35
        assert totals["answer_tokens"] == 25
        assert totals["reasoning_source"] == "mixed"
