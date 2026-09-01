"""Cache-hit % and cost, reconciled against BILLS — not against themselves.

WHY THIS FILE IS SHAPED LIKE THIS.  The bug it guards (issue #758) was
invisible to every self-consistent test.  ``compute_cache_hit_percent``
divides ``cache_read`` by ``cache_read + prompt``; a test that feeds it
``cache_read=X, prompt=Y`` and asserts ``X/(X+Y)`` passes whether or not
``Y`` already contains ``X``, because both sides of the assertion share
the assumption.  Such a test does not check the formula — it restates
it.

So every case here starts from numbers a provider actually PUT ON A
WIRE and ends at a number a provider actually CHARGED.  The bridge
between them is the code under test.  If the provider seam stops
converting the token convention, the reconstructed cost stops matching
the invoice, and these fail.

THE INVOICE.  A live GLM-5.3 turn through OpenRouter, from the issue:

    TUI footer:          130,755 in / 952 out, "cache hit: 50%", $0.0391
    OpenRouter console:  subtotal $0.187, cache read discount -$0.148,
                         final $0.0391
    z-ai/glm-5.3 rates:  prompt $1.4/M, completion $4.4/M,
                         input_cache_read $0.26/M

The discount fixes the cached count: $0.148 / ($1.4/M - $0.26/M) =
129,825 tokens, against a 130,755-token prompt — a 99.3% hit, reported
as 50%.  The reported figure was not merely wrong, it was structurally
incapable of exceeding 50%, because the wire's ``prompt_tokens``
already contained the cached tokens the formula added to it.
"""

from types import SimpleNamespace

import pytest

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

from jaato_sdk.helpers import compute_cache_hit_percent
from jaato_sdk.events import TurnCompletedEvent, UsageBreakdown
from jaato_sdk.plugins.model_provider.types import (
    TokenUsage,
    normalize_inclusive_usage,
    uncached_prompt_tokens,
)
from shared.pricing import PricingTable


#: The defect, put back — three ways, because it had three shapes: the
#: arithmetic itself, and a seam that stops calling it.  A seam is worth
#: its own reversion precisely because the shared helper can be perfect
#: while a provider simply never invokes it, which is what the whole
#: family did before #758.
REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/plugins/model_provider/types.py",
        find="""    cached = (cache_read_tokens or 0) + (cache_creation_tokens or 0)
    if cached <= 0:
        return prompt_tokens
    return max(0, prompt_tokens - cached)""",
        replace="""    return prompt_tokens""",
        test=("TestGLM53TurnAgainstItsInvoice::"
              "test_reported_hit_matches_the_discount_the_console_applied"),
        because="cached tokens landing on both sides of the cache-hit "
                "denominator again, which caps the reported rate at a "
                "structural 50% and makes a perfect cache read as half "
                "wasted",
    ),
    Reversion(
        target=(
            "jaato-server/shared/plugins/model_provider/"
            "openrouter/converters.py"
        ),
        find="""    # LAST, once both cached counts are known: take them out of
    # ``prompt_tokens``.  Order matters — reading the writes above is what
    # makes the cold-arrival turn (all input written, none read) normalize
    # correctly rather than reporting a whole cold prefix as new input.
    normalize_inclusive_usage(usage)""",
        replace="""    return""",
        test=("TestOpenRouterCacheWritesAreInsideThePromptTotal::"
              "test_writes_come_out_of_the_prompt_total"),
        because="the seam on the provider that produced the reported "
                "invoice going back to passing the wire's inclusive "
                "prompt count straight through",
    ),
    Reversion(
        target=(
            "jaato-server/shared/plugins/model_provider/"
            "google_genai/converters.py"
        ),
        find="""            usage.cache_read_tokens = cached_tokens
            normalize_inclusive_usage(usage)""",
        replace="""            usage.cache_read_tokens = cached_tokens""",
        test=("TestEveryInclusiveSeamConverts::"
              "test_google_genai_converter"),
        because="a second inclusive provider silently dropping the "
                "conversion — the failure mode is one seam at a time, "
                "not the helper",
    ),
]


# ---------------------------------------------------------------- fixtures

#: The z-ai/glm-5.3 catalog entry, in Litellm's per-token shape.
GLM_53_PRICES = {
    "z-ai/glm-5.3": {
        "input_cost_per_token": 1.4e-06,
        "output_cost_per_token": 4.4e-06,
        "cache_read_input_token_cost": 0.26e-06,
    }
}

#: What the wire said.  ``prompt_tokens`` is the WHOLE input on this
#: wire and ``cached_tokens`` is a subset of it — the OpenAI
#: convention.
GLM_53_WIRE_PROMPT = 130_755
GLM_53_WIRE_CACHED = 129_825
GLM_53_WIRE_COMPLETION = 952

#: What the console charged, after the cache-read discount.
GLM_53_INVOICE_USD = 0.0391


def _openai_usage(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
):
    """An OpenAI-shaped ``usage`` block, as the SDK hands it over.

    ``prompt_tokens_details`` arrives as a nested object; the counts in
    it are SUBSETS of the sibling ``prompt_tokens``.  That containment
    is the whole subject of this module, so the fixture models it
    literally rather than passing the three numbers side by side.
    """
    details = SimpleNamespace(
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=details,
    )


def _turn_event(usage: TokenUsage) -> TurnCompletedEvent:
    """Wrap a provider ``TokenUsage`` in the wire event clients read.

    Deliberately goes through :class:`UsageBreakdown` rather than
    handing ``compute_cache_hit_percent`` a hand-built stub: the number
    the user complained about was rendered from this event, so the test
    reads the same path the TUI footer does.
    """
    return TurnCompletedEvent(
        usage=UsageBreakdown(
            prompt_tokens=usage.prompt_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
        )
    )


def _glm_53_usage() -> TokenUsage:
    """The invoiced turn, carried through the OpenRouter seam."""
    from shared.plugins.model_provider.openrouter.converters import extract_usage

    response = SimpleNamespace(
        usage=_openai_usage(
            GLM_53_WIRE_PROMPT,
            GLM_53_WIRE_COMPLETION,
            cached_tokens=GLM_53_WIRE_CACHED,
        )
    )
    return extract_usage(response)


# ------------------------------------------------- the invoiced GLM-5.3 turn

class TestGLM53TurnAgainstItsInvoice:
    """One real turn, reconciled at both ends: the % and the dollars."""

    def test_reported_hit_matches_the_discount_the_console_applied(self):
        # The console's cache-read discount implies 129,825 of 130,755
        # prompt tokens were served from cache: 99.3%.  The TUI said 50%.
        pct = compute_cache_hit_percent(_turn_event(_glm_53_usage()))
        assert pct == pytest.approx(99.3, abs=0.1)

    def test_the_reported_hit_can_now_exceed_the_old_structural_ceiling(self):
        # Before the seam converted the convention, the metric could not
        # cross 50% no matter how good the cache was — so "50%" was the
        # reading for a PERFECT hit, and nothing ever looked anomalous.
        # Anything above 50 proves the double-count is gone.
        pct = compute_cache_hit_percent(_turn_event(_glm_53_usage()))
        assert pct > 50.0

    def test_the_token_counts_reconstruct_the_charged_cost(self):
        # The end-to-end claim: catalog rates + the counts this code
        # produces == what the provider billed.  Tolerance is 1%: the
        # cached count is itself derived from a discount rounded to the
        # tenth of a cent on the console.
        usage = _glm_53_usage()
        cost = PricingTable(GLM_53_PRICES).cost_for_usage(
            "z-ai/glm-5.3",
            prompt_tokens=usage.prompt_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
        )
        assert cost == pytest.approx(GLM_53_INVOICE_USD, rel=0.01)

    def test_the_raw_wire_counts_would_have_billed_multiples_of_the_invoice(self):
        # The negative control, and the reason the two footer numbers
        # disagreed with each other.  Feeding the wire's own
        # ``prompt_tokens`` straight to the price table charges the
        # cached tokens twice — full input rate AND cache-read rate.
        # This is what the pricing-table fallback did on every
        # OpenAI-compatible provider (OpenRouter itself reports a cost,
        # so the fallback was dormant there and the damage showed up in
        # the hit rate instead).
        cost = PricingTable(GLM_53_PRICES).cost_for_usage(
            "z-ai/glm-5.3",
            prompt_tokens=GLM_53_WIRE_PROMPT,
            output_tokens=GLM_53_WIRE_COMPLETION,
            cache_read_tokens=GLM_53_WIRE_CACHED,
        )
        assert cost > GLM_53_INVOICE_USD * 4

    def test_the_raw_wire_counts_produce_the_reported_fifty_percent(self):
        # Ties the arithmetic to the symptom: the un-converted counts
        # reproduce the exact figure the user saw, so this file's other
        # assertions are measuring the right defect.
        raw = TokenUsage(
            prompt_tokens=GLM_53_WIRE_PROMPT,
            output_tokens=GLM_53_WIRE_COMPLETION,
            total_tokens=GLM_53_WIRE_PROMPT + GLM_53_WIRE_COMPLETION,
            cache_read_tokens=GLM_53_WIRE_CACHED,
        )
        assert compute_cache_hit_percent(_turn_event(raw)) == pytest.approx(
            49.8, abs=0.1)


# ------------------------------------------- OpenRouter counts writes too

class TestOpenRouterCacheWritesAreInsideThePromptTotal:
    """A cold ARRIVAL, from the measurements in
    ``docs/design/model-tier-prompt-cache.md`` §6.0.1.

    That table records a per-response row of ``prompt=28,278`` beside
    ``cache_write=27,503`` and no reads, billed $0.035179.  Read
    inclusively that is 775 new tokens plus a 27,503-token write; on an
    Anthropic-family price family (write = 1.25x input, read = 0.1x
    input) it reconstructs at an input rate of $1.00/Mtok — a real
    catalog price — to within 0.1%.  Read exclusively the same row
    demands an input rate of $0.56/Mtok, which fits no catalog and
    contradicts the warm rows on the same table (28,294 prompt against
    a $0.003566 bill is 1/9 of what 28,294 new tokens could cost at any
    rate that also explains the arrival).

    So on OpenRouter BOTH cached quantities sit inside
    ``prompt_tokens``, and both come back out at the seam.

    A SECOND CASE SETTLES IT WITHOUT ANY PRICE LIST — which matters,
    because the rates above were fitted to the row they explain, and a
    fitted constant is weak evidence on its own.
    ``test_cache_spend_survives_a_tier_switch`` carries a wire shape
    captured live from a COLD Sonnet call: ``prompt_tokens=4412`` beside
    ``cache_write_tokens=4403`` and no reads.  A write is by definition a
    subset of what was sent — you can only cache content you transmitted
    — so the exclusive reading makes that turn's input 4412 new tokens
    PLUS 4403 written, 8,815 tokens of input for a prompt the same
    object calls 4,412.  It contradicts itself.  The inclusive reading
    gives 4,403 written and 9 new, which sums to exactly the reported
    4,412.  No cost, no rate, no fit.
    """

    RATES = {
        "cold-arrival-model": {
            "input_cost_per_token": 1.00e-06,
            "output_cost_per_token": 0.0,
            "cache_creation_input_token_cost": 1.25e-06,
            "cache_read_input_token_cost": 0.10e-06,
        }
    }

    def _arrival_usage(self) -> TokenUsage:
        from shared.plugins.model_provider.openrouter.converters import extract_usage

        return extract_usage(SimpleNamespace(usage=_openai_usage(
            28_278, 0, cache_write_tokens=27_503)))

    def test_writes_come_out_of_the_prompt_total(self):
        assert self._arrival_usage().prompt_tokens == 775

    def test_the_arrival_reconstructs_its_measured_cost(self):
        usage = self._arrival_usage()
        cost = PricingTable(self.RATES).cost_for_usage(
            "cold-arrival-model",
            prompt_tokens=usage.prompt_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
        )
        assert cost == pytest.approx(0.035179, rel=0.01)

    def test_a_cold_call_only_adds_up_under_the_inclusive_reading(self):
        # The price-free proof, from the live capture reused by
        # ``test_cache_spend_survives_a_tier_switch``.  A write is a
        # subset of what was sent, so the parts must sum to the whole:
        # 4,403 written + 9 new == the 4,412 reported.  Read the other
        # way the same object claims 8,815 tokens of input for a
        # 4,412-token prompt.
        from shared.plugins.model_provider.openrouter.converters import (
            extract_usage,
        )

        usage = extract_usage(SimpleNamespace(usage=_openai_usage(
            4412, 0, cache_write_tokens=4403)))
        assert usage.cache_creation_tokens == 4403
        assert usage.prompt_tokens == 9
        assert usage.prompt_tokens + usage.cache_creation_tokens == 4412

    def test_leaving_writes_in_would_overcharge_the_arrival(self):
        # The control for the write side specifically: a seam that
        # subtracted only reads would still bill a whole cold prefix at
        # the full input rate on top of the write rate.
        cost = PricingTable(self.RATES).cost_for_usage(
            "cold-arrival-model",
            prompt_tokens=28_278,
            output_tokens=0,
            cache_creation_tokens=27_503,
        )
        assert cost > 0.035179 * 1.5


# ------------------------------------------------------ the other seams

class TestEveryInclusiveSeamConverts:
    """Each provider whose wire counts cached tokens inside the prompt
    total must convert on the way out.  One test per seam, because the
    conversion lives in the provider and a shared helper test would not
    notice a seam that simply never calls it.
    """

    def test_openai_compatible_batch_path(self):
        # nim / lmstudio / tensorrt_llm / triton / doubleword / ovhcloud /
        # zhipuai_openai / nebius all inherit this one.
        from shared.plugins.model_provider._openai_compat.converters import (
            extract_usage,
        )

        raw = _openai_usage(10_000, 200, cached_tokens=9_500)
        usage = extract_usage(SimpleNamespace(usage=raw, choices=[]))
        # The shared converter does not read the cache block; the
        # provider sets it and normalizes.  Emulate exactly that pair.
        usage.cache_read_tokens = 9_500
        normalize_inclusive_usage(usage)
        assert usage.prompt_tokens == 500
        assert compute_cache_hit_percent(_turn_event(usage)) == pytest.approx(95.0)

    def test_nebius_converter(self):
        from shared.plugins.model_provider.nebius.converters import extract_usage

        usage = extract_usage(SimpleNamespace(
            usage=_openai_usage(10_000, 200, cached_tokens=9_500)))
        assert usage.prompt_tokens == 500
        assert usage.cache_read_tokens == 9_500

    def test_google_genai_converter(self):
        # Gemini's ``prompt_token_count`` is the whole prompt and
        # ``cached_content_token_count`` is the cached part of it.
        from shared.plugins.model_provider.google_genai.converters import (
            extract_usage_from_response,
        )

        response = SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=10_000,
            candidates_token_count=200,
            total_token_count=10_200,
            cached_content_token_count=9_500,
        ))
        usage = extract_usage_from_response(response)
        assert usage.prompt_tokens == 500
        assert usage.cache_read_tokens == 9_500
        # ``total_token_count`` is left as reported — it is the
        # end-of-turn context size, which GC's provider-path denominator
        # reads, and collapsing it on a cache-warm turn would stop GC
        # from ever firing.
        assert usage.total_tokens == 10_200

    def test_anthropic_converter_is_left_alone(self):
        # The control.  Anthropic's ``input_tokens`` already excludes
        # both cache counts — it IS the framework convention — so a
        # conversion here would create the mirror-image bug.
        from shared.plugins.model_provider.anthropic.converters import (
            extract_usage_from_response,
        )

        response = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=500,
            output_tokens=200,
            cache_read_input_tokens=9_500,
            cache_creation_input_tokens=0,
        ))
        usage = extract_usage_from_response(response)
        assert usage.prompt_tokens == 500
        assert usage.cache_read_tokens == 9_500
        assert compute_cache_hit_percent(_turn_event(usage)) == pytest.approx(95.0)


# --------------------------------------------------------- the arithmetic

class TestUncachedPromptTokens:
    """Edge behaviour of the conversion itself."""

    def test_no_cache_reported_is_a_no_op(self):
        assert uncached_prompt_tokens(1_000, None, None) == 1_000

    def test_a_reported_zero_is_a_no_op(self):
        assert uncached_prompt_tokens(1_000, 0, 0) == 1_000

    def test_reads_and_writes_both_come_out(self):
        assert uncached_prompt_tokens(1_000, 600, 300) == 100

    def test_a_perfect_hit_leaves_no_new_input(self):
        assert uncached_prompt_tokens(1_000, 1_000, None) == 0

    def test_an_impossible_report_clamps_instead_of_going_negative(self):
        # A subset cannot exceed its superset; a provider that says so
        # has a bug of its own, and a negative token count three layers
        # downstream is a worse way to find out.
        assert uncached_prompt_tokens(100, 900, None) == 0

    def test_the_helper_returns_the_object_it_mutated(self):
        usage = TokenUsage(prompt_tokens=1_000, cache_read_tokens=900)
        assert normalize_inclusive_usage(usage) is usage
        assert usage.prompt_tokens == 100
