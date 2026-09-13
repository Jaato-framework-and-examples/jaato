"""A session says what it spent, and which model it spent it on.

``_turn_accounting`` carries no model stamp and neither does the token
ledger's ``response`` record, so before the consumption ledger "what did
the voice tier cost me" was not a question the stored data could answer.
These tests pin the three properties that make the answer trustworthy:

* spend is attributed to the ``(provider, model, tier)`` that served the
  response, at the moment it served it;
* a dimension nobody reported stays ``None`` rather than becoming a zero;
* the per-chunk streaming hook contributes nothing, exactly as it
  contributes no ``spend_`` key -- accumulating there would count one
  response once per usage chunk.
"""

from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    ProviderResponse,
    TokenUsage,
)
from shared.jaato_session import JaatoSession
from shared.session_consumption import (
    COST_SOURCE_PRICING_TABLE,
    COST_SOURCE_PROVIDER,
    BindingUsage,
    ConsumptionLedger,
)


def _response(
    prompt=100, output=10, total=110, cache_read=None, cache_creation=None,
    thinking=None, cost=None, finish=FinishReason.STOP,
):
    return ProviderResponse(
        parts=[],
        usage=TokenUsage(
            prompt_tokens=prompt,
            output_tokens=output,
            total_tokens=total,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            thinking_tokens=thinking,
            cost_usd=cost,
        ),
        finish_reason=finish,
    )


def _session(model="model-a", provider="prov-a"):
    """A session with just enough runtime to accumulate token usage."""
    runtime = MagicMock()
    runtime.ledger = None
    runtime.registry = MagicMock()
    session = JaatoSession(runtime, model)
    session._model_name = model
    session._active_provider_name = provider
    session._budget_tracker = None
    return session


# --------------------------------------------------------------------
# The ledger, as pure logic
# --------------------------------------------------------------------

class TestConsumptionLedger:

    def test_bindings_are_keyed_on_provider_model_and_tier(self):
        """One tier, two models = two rows.

        This is the case a tier-keyed report gets wrong: a budget-control
        degrade rung rebinds a tier's model in place, so both responses
        are ``tier='planner'`` and they are not the same bill.
        """
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="opus", tier="planner",
                       uncached_input_tokens=10, output_tokens=1, total_tokens=11)
        ledger.observe(provider="p", model="flash", tier="planner",
                       uncached_input_tokens=20, output_tokens=2, total_tokens=22)
        assert len(ledger) == 2
        assert {b.model for b in ledger.bindings()} == {"opus", "flash"}

    def test_same_model_on_two_providers_is_two_bindings(self):
        ledger = ConsumptionLedger()
        for provider in ("openrouter", "anthropic"):
            ledger.observe(provider=provider, model="claude", tier=None,
                           uncached_input_tokens=10, output_tokens=1,
                           total_tokens=11)
        assert len(ledger) == 2

    def test_single_model_mode_is_one_row_with_a_null_tier(self):
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="m", tier=None,
                       uncached_input_tokens=10, output_tokens=1, total_tokens=11)
        rows = [b.as_dict() for b in ledger.bindings()]
        assert len(rows) == 1
        assert rows[0]["tier"] is None

    def test_an_unreported_dimension_stays_absent(self):
        """A provider with no cache must not read as a cache that never hits."""
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="m", uncached_input_tokens=10,
                       output_tokens=1, total_tokens=11)
        row = ledger.bindings()[0]
        assert row.cache_read_tokens is None
        assert row.cache_hit_percent is None
        assert "cache_read_tokens" not in row.as_dict()
        assert "cost_usd" not in row.as_dict()

    def test_a_reported_zero_is_not_an_absence(self):
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="m", uncached_input_tokens=10,
                       output_tokens=1, total_tokens=11, cache_read_tokens=0)
        assert ledger.bindings()[0].as_dict()["cache_read_tokens"] == 0

    def test_input_total_sums_the_three_disjoint_buckets(self):
        """``prompt_tokens`` EXCLUDES both cache buckets (the #758 convention)."""
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="m", uncached_input_tokens=100,
                       output_tokens=5, total_tokens=1105,
                       cache_read_tokens=900, cache_creation_tokens=100)
        row = ledger.bindings()[0]
        assert row.input_tokens_total == 1100
        # Hit rate excludes cache CREATION -- new content written, not a hit.
        assert row.cache_hit_percent == pytest.approx(90.0)

    def test_turns_counts_distinct_turns_per_binding(self):
        ledger = ConsumptionLedger()
        for turn in (0, 0, 1):
            ledger.observe(provider="p", model="m", uncached_input_tokens=1,
                           output_tokens=1, total_tokens=2, turn_index=turn)
        assert ledger.bindings()[0].responses == 3
        assert ledger.bindings()[0].turns == 2

    def test_total_turns_is_exact_across_bindings(self):
        """Neither summing nor maxing the per-binding counts is right.

        Turn 0 is served by two bindings (a mid-turn tier switch) and turn
        1 by one.  Summing gives 3, max gives 2; the session had 2 turns.
        """
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="a", tier="x",
                       uncached_input_tokens=1, output_tokens=1,
                       total_tokens=2, turn_index=0)
        ledger.observe(provider="p", model="b", tier="y",
                       uncached_input_tokens=1, output_tokens=1,
                       total_tokens=2, turn_index=0)
        ledger.observe(provider="p", model="b", tier="y",
                       uncached_input_tokens=1, output_tokens=1,
                       total_tokens=2, turn_index=1)
        assert ledger.totals().turns == 2
        assert ledger.totals().responses == 3

    def test_two_cost_sources_report_as_mixed(self):
        """A total that silently spoke with the first source's voice would
        present half an estimate as a bill."""
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="a", uncached_input_tokens=1,
                       output_tokens=1, total_tokens=2, cost_usd=0.5,
                       cost_source=COST_SOURCE_PROVIDER)
        ledger.observe(provider="p", model="b", uncached_input_tokens=1,
                       output_tokens=1, total_tokens=2, cost_usd=0.25,
                       cost_source=COST_SOURCE_PRICING_TABLE)
        totals = ledger.totals()
        assert totals.cost_usd == pytest.approx(0.75)
        assert totals.cost_source == "mixed"

    def test_totals_name_no_particular_model(self):
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="a", uncached_input_tokens=1,
                       output_tokens=1, total_tokens=2)
        assert ledger.totals().as_dict()["model"] == "*"

    def test_finish_reasons_accumulate(self):
        ledger = ConsumptionLedger()
        for reason in ("stop", "max_tokens", "max_tokens"):
            ledger.observe(provider="p", model="m", uncached_input_tokens=1,
                           output_tokens=1, total_tokens=2,
                           finish_reason=reason)
        assert ledger.bindings()[0].finish_reasons == {
            "stop": 1, "max_tokens": 2}

    def test_empty_ledger_reports_empty_totals(self):
        totals = ConsumptionLedger().totals()
        assert totals.responses == 0
        assert totals.turns == 0
        assert totals.as_dict()["total_tokens"] == 0


# --------------------------------------------------------------------
# The POOLED cache figure, where "absent is not zero" is easiest to break
# --------------------------------------------------------------------

def _mixed_pool():
    """The measured two-tier voice session: one caching tier, one not.

    Numbers are the ones a live session reported (executor
    ``google/gemini-2.5-flash`` at 76.41%, voz ``openai/gpt-audio``
    reporting no cache dimension at all), because the figure they pooled
    to -- 54.66% -- is the regression this section exists to prevent.
    """
    ledger = ConsumptionLedger()
    ledger.observe(provider="openrouter", model="google/gemini-2.5-flash",
                   tier="executor", uncached_input_tokens=133_044,
                   output_tokens=475, total_tokens=564_461,
                   cache_read_tokens=430_942, turn_index=0)
    ledger.observe(provider="openrouter", model="openai/gpt-audio", tier="voz",
                   uncached_input_tokens=224_423, output_tokens=1_476,
                   total_tokens=225_899, turn_index=1)
    return ledger


class TestPooledCacheHitRate:
    """A session-wide rate is published only where its premise holds.

    ``BindingUsage.cache_hit_percent`` divides by "input that COULD have
    been a hit", which one binding establishes by its provider reporting
    the dimension.  Summing across a binding that reported nothing keeps
    the formula and drops the premise: the unmeasured input lands in the
    denominator and nothing lands in the numerator, which is
    absent-read-as-zero reached by arithmetic instead of by a literal --
    the one thing this module's docstring forbids.
    """

    def test_a_mixed_pool_withholds_the_session_wide_rate(self):
        totals = _mixed_pool().totals()
        assert totals.cache_hit_percent is None
        assert "cache_hit_percent" not in totals.as_dict()

    def test_the_withheld_rate_is_the_one_that_would_have_misled(self):
        """Pin the number, so a reversion fails loudly rather than quietly."""
        totals = _mixed_pool().totals()
        pooled = 100.0 * totals.cache_read_tokens / (
            totals.cache_read_tokens + totals.uncached_input_tokens)
        assert pooled == pytest.approx(54.66, abs=0.01)
        # ... and no binding had it.
        assert all(b.cache_hit_percent != pytest.approx(pooled, abs=0.01)
                   for b in _mixed_pool().bindings())

    def test_the_basis_names_what_could_not_be_measured(self):
        basis = _mixed_pool().totals().as_dict()["cache_hit_basis"]
        assert basis["measured_bindings"] == 1
        assert basis["unmeasured_bindings"] == 1
        # Exactly what a pooled denominator would have absorbed -- the
        # unmeasured binding's UNCACHED input, not its input_tokens_total.
        assert basis["unmeasured_input_tokens"] == 224_423
        assert basis["measured_input_tokens"] == 563_986

    def test_the_measured_subset_still_reports_its_rate(self):
        """Withholding must not cost the reader the defensible figure."""
        basis = _mixed_pool().totals().as_dict()["cache_hit_basis"]
        assert basis["cache_hit_percent"] == pytest.approx(76.41)

    def test_the_token_sums_are_untouched(self):
        """Only the derived rate is withheld; sums of measurements stay."""
        row = _mixed_pool().totals().as_dict()
        assert row["cache_read_tokens"] == 430_942
        assert row["input_tokens_total"] == 788_409

    def test_a_homogeneous_pool_is_unchanged(self):
        """The common case must not pay for the mixed one."""
        ledger = ConsumptionLedger()
        for model in ("a", "b"):
            ledger.observe(provider="p", model=model, uncached_input_tokens=100,
                           output_tokens=5, total_tokens=1_005,
                           cache_read_tokens=900, turn_index=0)
        row = ledger.totals().as_dict()
        assert row["cache_hit_percent"] == pytest.approx(90.0)
        assert "cache_hit_basis" not in row

    def test_a_reported_zero_makes_the_pool_homogeneous(self):
        """The payoff of not collapsing a reported 0 at the provider seam.

        The same two bindings as the mixed pool in shape, except the
        second says "I cache, and I hit nothing".  That is a measurement,
        so the pooled denominator is fully established and the rate is
        published -- which is why the provider-seam half of this fix is
        what turns a withheld figure back into a real one.
        """
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="a", uncached_input_tokens=100,
                       output_tokens=5, total_tokens=1_005,
                       cache_read_tokens=900, turn_index=0)
        ledger.observe(provider="p", model="b", uncached_input_tokens=200,
                       output_tokens=5, total_tokens=205,
                       cache_read_tokens=0, turn_index=1)
        row = ledger.totals().as_dict()
        assert row["cache_hit_percent"] == pytest.approx(75.0)
        assert "cache_hit_basis" not in row

    def test_a_pool_measuring_nothing_explains_nothing(self):
        """``cache_read_tokens`` is already absent -- that reads correctly.

        A basis here would explain the absence of a number no reader was
        expecting, and would be the only place in the payload where an
        all-uncached session says the word "cache".
        """
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="m", uncached_input_tokens=100,
                       output_tokens=5, total_tokens=105, turn_index=0)
        row = ledger.totals().as_dict()
        assert "cache_hit_percent" not in row
        assert "cache_hit_basis" not in row
        assert "cache_read_tokens" not in row

    def test_an_unmeasured_binding_with_no_uncached_input_is_not_mixed(self):
        """It could not have moved the denominator, so nothing is withheld."""
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="a", uncached_input_tokens=100,
                       output_tokens=5, total_tokens=1_005,
                       cache_read_tokens=900, turn_index=0)
        ledger.observe(provider="p", model="b", uncached_input_tokens=0,
                       output_tokens=5, total_tokens=5, turn_index=1)
        row = ledger.totals().as_dict()
        assert row["cache_hit_percent"] == pytest.approx(90.0)
        assert "cache_hit_basis" not in row

    def test_a_real_binding_never_carries_a_basis(self):
        """The field is the totals row's alone; a binding is not a pool."""
        for binding in _mixed_pool().bindings():
            assert binding.cache_basis is None
            assert "cache_hit_basis" not in binding.as_dict()

    def test_the_note_promises_no_figure_it_does_not_publish(self):
        """A measured subset whose own denominator is empty says less."""
        ledger = ConsumptionLedger()
        ledger.observe(provider="p", model="a", uncached_input_tokens=0,
                       output_tokens=5, total_tokens=5,
                       cache_read_tokens=0, turn_index=0)
        ledger.observe(provider="p", model="b", uncached_input_tokens=200,
                       output_tokens=5, total_tokens=205, turn_index=1)
        basis = ledger.totals().as_dict()["cache_hit_basis"]
        assert "cache_hit_percent" not in basis
        assert "figure" not in basis["note"]
        assert "cache_hit_percent here" not in basis["note"]


# --------------------------------------------------------------------
# The session, wiring the ledger to real responses
# --------------------------------------------------------------------

class TestSessionAttribution:

    def test_a_response_is_attributed_to_the_live_binding(self):
        session = _session(model="model-a", provider="prov-a")
        session._accumulate_turn_tokens(_response(), {})
        rows = session._consumption.bindings()
        assert len(rows) == 1
        assert (rows[0].provider, rows[0].model) == ("prov-a", "model-a")

    def test_a_tier_switch_splits_the_spend(self):
        """The point of the whole feature: one history, two bills."""
        session = _session(model="planner-model", provider="prov")
        session._active_tier = "planner"
        session._accumulate_turn_tokens(_response(prompt=100, total=110), {})

        # What ``_connect_tier_entry`` / ``switch_tier`` do before the
        # next request goes out.
        session._active_tier = "voice"
        session._model_name = "voice-model"
        session._accumulate_turn_tokens(_response(prompt=500, total=510), {})

        by_tier = {b.tier: b for b in session._consumption.bindings()}
        assert set(by_tier) == {"planner", "voice"}
        assert by_tier["planner"].uncached_input_tokens == 100
        assert by_tier["voice"].uncached_input_tokens == 500

    def test_the_streaming_chunk_hook_records_nothing(self):
        """Mirrors the ``spend_`` rule, and for the same reason.

        A provider may emit several usage chunks per response; observing
        there would bill one response as many.  Asserted by EFFECT rather
        than by reading the function body, so a write that hides in a
        helper is still caught.
        """
        session = _session()
        usage = TokenUsage(prompt_tokens=100, output_tokens=10, total_tokens=110)
        session._track_streaming_usage({}, usage)
        session._track_streaming_usage({}, usage)
        assert len(session._consumption) == 0

    def test_a_response_reporting_nothing_invents_no_binding(self):
        """A cancelled stream can settle with zero usage."""
        session = _session()
        session._accumulate_turn_tokens(_response(prompt=0, output=0, total=0), {})
        assert len(session._consumption) == 0

    def test_observation_never_raises_into_the_turn(self):
        session = _session()
        session._consumption = MagicMock()
        session._consumption.observe.side_effect = RuntimeError("boom")
        # Must not propagate: accounting is an observation about a turn,
        # not part of its contract.
        session._accumulate_turn_tokens(_response(), {})

    def test_provider_cost_beats_the_pricing_table(self):
        session = _session()
        cost, source = session._resolve_cost_with_source(
            TokenUsage(prompt_tokens=1, output_tokens=1, total_tokens=2,
                       cost_usd=0.42))
        assert (cost, source) == (0.42, COST_SOURCE_PROVIDER)

    def test_cost_and_its_source_agree_with_the_span_cost(self):
        """One ladder, walked once -- the two answers cannot disagree."""
        session = _session()
        usage = TokenUsage(prompt_tokens=1, output_tokens=1, total_tokens=2,
                           cost_usd=0.42)
        assert session._resolve_span_cost(usage) == \
            session._resolve_cost_with_source(usage)[0]


# --------------------------------------------------------------------
# The report
# --------------------------------------------------------------------

class TestGetConsumption:

    def test_summary_withholds_the_binding_list(self):
        session = _session()
        session._accumulate_turn_tokens(_response(), {})
        report = session.get_consumption("summary")
        assert "bindings" not in report
        assert report["binding_count"] == 1
        assert report["totals"]["total_tokens"] == 110

    def test_full_reports_every_binding(self):
        session = _session()
        session._active_tier = "planner"
        session._accumulate_turn_tokens(_response(), {})
        session._active_tier = "voice"
        session._model_name = "voice-model"
        session._accumulate_turn_tokens(_response(), {})
        report = session.get_consumption("full")
        assert [b["tier"] for b in report["bindings"]] == ["planner", "voice"]

    def test_summary_says_a_breakdown_exists(self):
        session = _session()
        session._active_tier = "planner"
        session._accumulate_turn_tokens(_response(), {})
        session._active_tier = "voice"
        session._model_name = "voice-model"
        session._accumulate_turn_tokens(_response(), {})
        assert "detail='full'" in session.get_consumption("summary")["hint"]

    def test_an_unknown_detail_degrades_rather_than_raising(self):
        session = _session()
        assert "totals" in session.get_consumption("wildly-invalid")

    def test_the_active_binding_carries_occupancy_not_spend(self):
        """``percent_used`` belongs to the history, not to a model."""
        session = _session()
        session._accumulate_turn_tokens(_response(), {})
        active = session.get_consumption()["active"]
        assert "context_percent_used" in active
        assert "total_tokens" not in active

    def test_no_budget_block_when_none_is_declared(self):
        session = _session()
        assert "budget" not in session.get_consumption()

    def test_no_completion_block_without_a_gate(self):
        session = _session()
        assert "completion" not in session.get_consumption()

    def test_a_fresh_session_reports_zeroes_not_an_error(self):
        report = _session().get_consumption()
        assert report["binding_count"] == 0
        assert report["totals"]["total_tokens"] == 0


class TestBudgetView:

    def _budgeted(self, limits, degrade=()):
        from shared.budget_control import BudgetControlConfig, BudgetTracker
        session = _session()
        config = BudgetControlConfig.from_dict(
            {"limits": limits, "degrade": list(degrade)})
        session._budget_tracker = BudgetTracker(config)
        return session

    def test_only_declared_dimensions_are_reported(self):
        """The tracker accumulates all five; an undeclared one reading 0.0
        beside a ceiling it does not have is a phantom headroom reading."""
        session = self._budgeted({"tool_calls": 100})
        session._budget_tracker.observe(tool_calls=60)
        budget = session.get_consumption()["budget"]
        assert set(budget["used"]) == {"tool_calls"}
        assert budget["used"]["tool_calls"] == 60.0
        assert budget["fraction_used"] == pytest.approx(0.6)

    def test_next_rung_is_the_lowest_one_not_yet_passed(self):
        session = self._budgeted(
            {"tool_calls": 100},
            [{"at": 50, "action": "finalize"}, {"at": 95, "action": "abort"}])
        session._budget_tracker.observe(tool_calls=60)
        assert session.get_consumption()["budget"]["next_rung"] == {
            "at_percent": 95.0, "action": "abort"}

    def test_no_next_rung_once_the_ladder_is_spent(self):
        session = self._budgeted(
            {"tool_calls": 100}, [{"at": 50, "action": "abort"}])
        session._budget_tracker.observe(tool_calls=99)
        assert "next_rung" not in session.get_consumption()["budget"]

    def test_limits_without_a_ladder_still_report(self):
        """#947's case: ceilings nobody enforces are still worth seeing."""
        session = self._budgeted({"usd": 5.0})
        budget = session.get_consumption()["budget"]
        assert budget["limits"] == {"usd": 5.0}
        assert "next_rung" not in budget


class TestCompletionView:

    def _gated(self, hidden=False, schema=None):
        session = _session()
        lifecycle = MagicMock()
        lifecycle._should_hide_signal_completion.return_value = hidden
        session._lifecycle_tools = lifecycle
        session._completion_payload_schema = schema
        return session

    def test_absent_when_signal_completion_is_hidden(self):
        assert "completion" not in self._gated(hidden=True).get_consumption()

    def test_reports_the_framework_default_until_a_nudge_is_considered(self):
        from shared.completion_nudge import DEFAULT_MAX_COMPLETION_NUDGES
        completion = self._gated().get_consumption()["completion"]
        assert completion["max_nudges_per_turn"] == DEFAULT_MAX_COMPLETION_NUDGES
        assert completion["max_nudges_source"] == "framework_default"

    def test_the_observed_budget_replaces_the_default(self):
        """A profile that raised ``max_completion_nudges`` reaches the
        session only through the caller that nudges."""
        session = self._gated()
        session.try_completion_nudge(4)
        completion = session.get_consumption()["completion"]
        assert completion["max_nudges_per_turn"] == 4
        assert completion["max_nudges_source"] == "observed"

    def test_remaining_falls_as_nudges_fire(self):
        session = self._gated()
        session.try_completion_nudge(2)
        completion = session.get_consumption()["completion"]
        assert completion["nudges_fired_this_turn"] == 1
        assert completion["nudges_remaining_this_turn"] == 1

    def test_the_lifetime_count_survives_a_turn_reset(self):
        """``_completion_nudges_fired`` is per TURN by design (#934), so a
        session-lifetime figure needs its own counter."""
        session = self._gated()
        session.try_completion_nudge(2)
        session._begin_turn_completion_state()   # consumes the nudge latch
        session._begin_turn_completion_state()   # a caller-started turn
        completion = session.get_consumption()["completion"]
        assert completion["nudges_fired_this_turn"] == 0
        assert completion["nudges_fired_total"] == 1

    def test_the_lifetime_counter_decides_nothing(self):
        """It must never become a second budget: the guard reads only the
        per-turn counter, or #767's unbounded nudge loop returns."""
        session = self._gated()
        session.try_completion_nudge(1)
        session._begin_turn_completion_state()
        session._begin_turn_completion_state()
        # Lifetime total is 1, but a fresh turn has its full budget back.
        should_nudge, _ = session.try_completion_nudge(1)
        assert should_nudge is True

    def test_schema_presence_is_reported(self):
        assert self._gated(schema={"type": "object"}).get_consumption()[
            "completion"]["payload_schema_declared"] is True
        assert self._gated().get_consumption()[
            "completion"]["payload_schema_declared"] is False


class TestLedgerLifecycle:
    """Two ways the ledger and the turn list can fall out of step."""

    def test_reset_clears_the_ledger(self):
        """Not tidiness: the turn index is derived from the turn list's
        length, so a reset restarts it at 0 and every re-used index would
        read as the SAME turn, under-counting ``turns`` thereafter."""
        session = _session()
        session._accumulate_turn_tokens(_response(), {})
        session._turn_accounting.append({"total": 110})
        assert len(session._consumption) == 1

        session.reset_session()
        assert len(session._consumption) == 0
        assert session.get_consumption()["binding_count"] == 0

    def test_turns_keep_counting_after_a_reset(self):
        session = _session()
        session._accumulate_turn_tokens(_response(), {})
        session._turn_accounting.append({"total": 110})
        session.reset_session()
        # Turn index is 0 again -- with a stale ledger this second turn
        # would merge into the first and report turns == 1.
        session._accumulate_turn_tokens(_response(), {})
        assert session.get_consumption()["totals"]["turns"] == 1

    def test_a_revived_session_names_what_it_cannot_account_for(self):
        """The ledger is in-memory and not part of the session snapshot."""
        session = _session()
        session.restore_turn_accounting([
            {"total": 100, "duration_seconds": 1.0},
            {"total": 200, "duration_seconds": 1.0},
        ])
        report = session.get_consumption()
        assert report["turns_not_attributed"] == 2
        assert "revived session" in report["measurement_note"]

    def test_no_note_when_everything_is_attributed(self):
        session = _session()
        session._accumulate_turn_tokens(_response(), {})
        session._turn_accounting.append({"total": 110})
        report = session.get_consumption()
        assert "turns_not_attributed" not in report
        assert "measurement_note" not in report


class TestObservationIsTotal:
    """``_observe_binding_usage`` must not be able to fail a turn.

    The shape that broke it: several guard tests drive
    ``_accumulate_turn_tokens`` on a ``JaatoSession`` built WITHOUT
    ``__init__``, so ``_model_name`` does not exist.  The cost lookup
    raised, the handler caught it — and then ``self._trace`` raised too,
    because it reads ``_agent_type``.  A handler that can raise is not a
    handler.
    """

    def _uninitialised(self):
        return JaatoSession.__new__(JaatoSession)

    def test_a_session_without_init_is_left_alone(self):
        session = self._uninitialised()
        session._accumulate_turn_tokens(_response(), {})   # must not raise

    def test_the_turn_still_accumulates_on_such_a_session(self):
        """The observation is an addition to that hook, never a condition
        of it: the spend keys it was bolted onto must still be written."""
        session = self._uninitialised()
        turn_tokens = {}
        session._accumulate_turn_tokens(_response(prompt=100, total=110), {})
        session._accumulate_turn_tokens(
            _response(prompt=100, total=110), turn_tokens)
        assert turn_tokens["spend_total"] == 110

    def test_unusable_usage_values_do_not_escape(self):
        """A response whose token counts are real but whose cost is not a
        number: the arithmetic inside the ledger raises, inside the guard,
        where it belongs.

        (A *bare* MagicMock never reaches the ledger at all — the
        pre-existing ``response.usage.total_tokens > 0`` comparison at the
        top of the hook rejects it first.)
        """
        session = _session()
        response = _response()
        response.usage.cost_usd = MagicMock()
        session._accumulate_turn_tokens(response, {})   # must not raise

        # The row is written; only the unusable figure is dropped.  The
        # alternative -- letting the addition raise -- would abandon a row
        # whose token fields had already been written.
        row = session._consumption.bindings()[0]
        assert row.total_tokens == 110
        assert row.cost_usd is None
