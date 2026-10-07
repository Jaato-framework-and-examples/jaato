"""Per-binding consumption accounting: what this session spent, and on what.

WHAT A BINDING IS.  A ``(provider, model, tier)`` triple — the thing a
request was actually served by.  It is the unit of segregation here, and
the tier name alone is deliberately NOT:

* A budget-control degrade rung **rebinds a tier's model in place**
  (``planner: opus -> flash``) without changing the tier's name.  A
  tier-keyed total would silently merge two models' spend under one row,
  and the session where that happens is by definition the session someone
  is reading the numbers because of.
  :meth:`shared.jaato_session.JaatoSession.switch_tier` short-circuits on
  the resolved entry rather than the name for the same reason.
* Tiers may name different providers, so ``model`` alone is not a key
  either — the same model served by two providers is two prices.

A single-model session has exactly one binding with ``tier=None``, so the
shape does not special-case the common case; it reports a list of one.

WHY THIS MODULE EXISTS.  Nothing in the tree attributed consumption to a
model.  ``_turn_accounting`` carries no model or tier stamp and neither
does the token ledger's ``response`` record, so "what did the voice tier
cost me" was not a question the stored data could answer — a measurement
gap, not a reporting one.

LEVEL VS SPEND.  Every figure here is SPEND: the sum over the responses
the binding served.  It is never the end-of-turn context size, which is a
property of the history rather than of any binding, and which
``_turn_accounting`` already tracks separately under its un-prefixed keys.
Conflating the two undercounted a real 3-turn run by 41% (see
``_accumulate_turn_tokens``), and this module exists downstream of that
lesson, not upstream of it.

THE PROMPT-TOKEN CONVENTION.  ``TokenUsage.prompt_tokens`` is the NEW,
uncached input — it excludes both cache buckets (see that dataclass's
docstring, and issue #758).  This module keeps the three disjoint and
names them for what they are, because the accumulated totals are read by
a *model* and ``prompt_tokens`` sitting beside ``cache_read_tokens``
invites it to add the same tokens twice.  :meth:`BindingUsage.as_dict`
publishes the derived ``input_tokens_total`` so nobody has to.

ABSENT IS NOT ZERO.  ``cache_read_tokens``, ``cache_creation_tokens``,
``reasoning_tokens`` and ``cost_usd`` stay ``None`` until something reports
one.  A provider with no prompt cache must not read as a provider whose
cache never hits, and a session with no pricing table must not read as
free.  ``cost_source`` says which source supplied a cost, so a consumer
can tell a billed figure from an estimated one.

AND THE TOTAL IS WHERE THAT RULE IS EASIEST TO BREAK.  A pooled
cache-hit rate sums the numerator over the bindings that HAVE a cache
and the denominator over all of them, so a binding reporting no cache
dimension contributes zero hits and its whole uncached input — which is
absent-read-as-zero, arrived at by arithmetic instead of by a literal.
Measured on a two-tier voice session: a caching ``executor`` binding at
76.41% pooled with a non-reporting ``voz`` binding to 54.66%, a
session-wide "efficiency" figure no binding had and nothing in the
payload explained.  So :meth:`ConsumptionLedger.totals` WITHHOLDS the
pooled percentage whenever part of its denominator is unmeasured, and
publishes :class:`CacheMeasurementBasis` in its place — which says how
much input could not be measured, and what the rate is over the subset
that could.  Nothing is lost; the number that was never defensible is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: ``(provider, model, tier)``.  ``tier`` is ``None`` in single-model mode.
BindingKey = Tuple[str, str, Optional[str]]

#: ``cost_usd`` came from the provider's own report — fiscal truth.
COST_SOURCE_PROVIDER = "provider"
#: ``cost_usd`` was computed from ``.jaato/pricing.json`` — an estimate.
COST_SOURCE_PRICING_TABLE = "pricing_table"

#: ``reasoning_tokens`` was reported by the upstream — a measurement.
REASONING_SOURCE_PROVIDER = "provider"
#: ``reasoning_tokens`` was estimated from the reasoning text by the
#: provider seam (Anthropic, Bedrock report no count).
REASONING_SOURCE_ESTIMATE = "estimate"

#: Accepted ``detail`` levels for a consumption report.
DETAIL_SUMMARY = "summary"
DETAIL_FULL = "full"
VALID_DETAIL_LEVELS = (DETAIL_SUMMARY, DETAIL_FULL)


def _merge_source(current: Optional[str], incoming: str) -> str:
    """Combine two provenance labels: equal stays, different is ``"mixed"``.

    One number fed by two sources says so, rather than letting the first
    source speak for the whole -- the rule ``cost_source`` follows.
    """
    if current is None or current == incoming:
        return incoming
    return "mixed"


def _add_optional(current: Optional[int], delta: Optional[int]) -> Optional[int]:
    """Sum that keeps ``None`` meaning "nothing reported".

    ``None + None`` is ``None`` (still nothing reported); ``None + 0`` is
    ``0`` (a provider reported a zero, which is a measurement).  Written
    out rather than inlined because getting it wrong in either direction
    erases the distinction the whole module rests on.
    """
    if delta is None:
        return current
    return delta if current is None else current + delta


@dataclass
class CacheMeasurementBasis:
    """Which bindings a POOLED cache-hit figure could actually be taken over.

    Attached to the totals row only (:meth:`ConsumptionLedger.totals`);
    an ordinary binding leaves it ``None``, because a single binding
    either reports the dimension or does not and there is no mixture to
    describe.

    It exists because the pooled rate's premise is stronger than the
    per-binding one.  ``BindingUsage.cache_hit_percent`` divides by "what
    this binding paid for that COULD have been a hit" — established, for
    one binding, by the provider having reported the dimension at all.
    Pooled across bindings that premise is no longer established: a
    binding whose provider reports nothing contributes its uncached input
    to the denominator and nothing to the numerator, and we cannot say
    whether that input was cacheable-and-missed or not cacheable at all.

    Attributes:
        measured_bindings: Bindings that reported ``cache_read_tokens``.
        unmeasured_bindings: Bindings that reported none.
        measured_input_tokens: ``cache_read + uncached`` over the measured
            bindings — the denominator the subset rate is taken over.
        unmeasured_input_tokens: UNCACHED input from the unmeasured
            bindings.  Not their ``input_tokens_total``: this is precisely
            the quantity a pooled denominator would have absorbed, and
            naming it after a larger number would repeat the error this
            class exists to report.
        cache_hit_percent: The rate over the measured subset, or ``None``
            when nothing was measured.  Defensible where the pooled figure
            is not, because every binding behind it reported the dimension.
    """

    measured_bindings: int = 0
    unmeasured_bindings: int = 0
    measured_input_tokens: int = 0
    unmeasured_input_tokens: int = 0
    cache_hit_percent: Optional[float] = None

    @property
    def is_mixed(self) -> bool:
        """True when a pooled rate would silently absorb unmeasured input.

        All three clauses are load-bearing, and each excludes a pool that
        needs no explanation:

        * ``measured_bindings`` — with nothing measured there is no pooled
          rate to withhold.  ``cache_read_tokens`` is already absent from
          the row, which reads correctly on its own, and a basis beside it
          would explain the absence of a number nobody expected.
        * ``unmeasured_bindings`` — a homogeneous pool's rate is sound.
        * ``unmeasured_input_tokens`` — a binding contributing no uncached
          input could not have moved the pooled denominator, so the figure
          it would have produced is the one we would publish anyway.
        """
        return (
            self.measured_bindings > 0
            and self.unmeasured_bindings > 0
            and self.unmeasured_input_tokens > 0
        )

    def as_dict(self) -> Dict[str, Any]:
        """The reportable view, with the note that says why it is here.

        The note's second sentence is conditional: a measured subset whose
        own denominator came out empty publishes no rate, and promising a
        "figure below" that is not below it would be the same shape of
        defect at one remove.
        """
        note = (
            f"No session-wide cache_hit_percent is reported: "
            f"{self.unmeasured_bindings} binding(s) contributed "
            f"{self.unmeasured_input_tokens} uncached input tokens while "
            f"reporting no cache dimension, so a pooled rate would count them "
            f"as misses without knowing they were cacheable."
        )
        out: Dict[str, Any] = {
            "measured_bindings": self.measured_bindings,
            "unmeasured_bindings": self.unmeasured_bindings,
            "measured_input_tokens": self.measured_input_tokens,
            "unmeasured_input_tokens": self.unmeasured_input_tokens,
        }
        if self.cache_hit_percent is not None:
            out["cache_hit_percent"] = self.cache_hit_percent
            note += (
                " cache_hit_percent here is over the measured bindings only;"
                " see the per-binding rows (detail='full') for each one's own"
                " rate."
            )
        out["note"] = note
        return out


@dataclass
class BindingUsage:
    """What one ``(provider, model, tier)`` binding consumed.

    Every token field is a SUM over the responses this binding served —
    see the module docstring on level vs spend.  Optional fields are
    ``None`` until a provider reports the dimension at least once.

    Attributes:
        provider: Provider plugin name (``"openrouter"``, ``"anthropic"``).
        model: Model id as sent on the wire.
        tier: Tier name that was active, or ``None`` in single-model mode.
        responses: Provider responses billed to this binding.  A turn with
            a tool call produces several, which is exactly why this is not
            a turn count.
        turns: Turns in which this binding served at least one response.
            A turn that crosses an ``enter_tier`` counts for both bindings;
            summing this column across bindings can therefore exceed the
            session's turn count, and that is the honest answer.
        uncached_input_tokens: NEW input — excludes both cache buckets.
        output_tokens: Generated tokens, cache-independent.
        cache_read_tokens: Input served from cache, or ``None``.
        cache_creation_tokens: Input written to cache, or ``None``.
        reasoning_tokens: Output tokens spent reasoning — a SUBSET of
            ``output_tokens``, never beside it (the ``TokenUsage``
            output-token convention, #1047) — or ``None`` when the
            provider reported none.  A reported ``0`` is kept.
        reasoning_source: :data:`REASONING_SOURCE_PROVIDER`,
            :data:`REASONING_SOURCE_ESTIMATE`, or ``"mixed"`` — the
            ``cost_source`` idea applied to reasoning, because an estimate
            from text beside a row of measurements has to say so.
        reasoning_responses: How many responses contributed a reasoning
            count.  When it equals ``responses`` the whole of
            ``output_tokens`` is covered and ``answer_tokens`` is
            published; otherwise some output's reasoning is unknown and
            the derived figure is withheld (the cache-basis argument,
            one bucket over).
        total_tokens: The provider's own ``total_tokens``, summed.  Kept
            as reported rather than recomputed, matching
            :class:`~jaato_sdk.plugins.model_provider.types.TokenUsage`.
        cost_usd: Accumulated cost, or ``None`` when no source knew.
        cost_source: :data:`COST_SOURCE_PROVIDER` or
            :data:`COST_SOURCE_PRICING_TABLE` — which source supplied the
            cost.  ``"mixed"`` if both contributed, which happens when a
            tier's provider reports cost and another's does not.
        finish_reasons: How each response ended, by reason.  A binding
            accumulating ``max_tokens`` is one whose output cap is wrong.
        first_used: ISO timestamp of this binding's first response.
        last_used: ISO timestamp of its most recent one.
        cache_basis: Set on the TOTALS row only, by
            :meth:`ConsumptionLedger.totals` — see
            :class:`CacheMeasurementBasis`.  ``None`` on every real
            binding, which is what keeps a single-model session's report
            byte-identical to what it was before the totals row learned to
            withhold.
    """

    provider: str
    model: str
    tier: Optional[str] = None
    responses: int = 0
    turns: int = 0
    uncached_input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    reasoning_source: Optional[str] = None
    reasoning_responses: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None
    cost_source: Optional[str] = None
    finish_reasons: Dict[str, int] = field(default_factory=dict)
    first_used: Optional[str] = None
    last_used: Optional[str] = None
    cache_basis: Optional[CacheMeasurementBasis] = None
    # Which turn index last contributed, so ``turns`` counts distinct turns
    # in O(1) without holding the set.  Not reported.
    _last_turn_index: Optional[int] = field(default=None, repr=False)

    @property
    def input_tokens_total(self) -> int:
        """All input paid for: uncached + cache reads + cache creation.

        The three are disjoint by the ``TokenUsage`` convention, so this
        is a plain sum.  Published because a model reading the buckets
        should never have to know that.
        """
        return (
            self.uncached_input_tokens
            + (self.cache_read_tokens or 0)
            + (self.cache_creation_tokens or 0)
        )

    @property
    def answer_tokens(self) -> Optional[int]:
        """Output that was ANSWER rather than reasoning, or ``None``.

        ``output_tokens - reasoning_tokens`` — subtraction, because
        reasoning is a subset of output.  Withheld unless every response
        reported a reasoning count: a response that reported none may
        still have reasoned, and subtracting nothing for it would pass its
        reasoning off as answer.
        """
        if self.reasoning_tokens is None:
            return None
        if self.reasoning_responses < self.responses:
            return None
        return max(0, self.output_tokens - self.reasoning_tokens)

    @property
    def cache_hit_percent(self) -> Optional[float]:
        """Share of cacheable input that was served from cache.

        Denominator is ``cache_read + uncached_input`` — what the binding
        paid for that COULD have been a hit.  ``cache_creation`` is
        excluded: it is new content being written, not a hit on prior
        content.  Same contract as
        :func:`jaato_sdk.helpers.compute_cache_hit_percent`, including its
        dependence on the prompt-token convention.

        ``None`` when the provider reports no cache at all, so "uncached
        provider" stays distinct from "cache never hit".

        Also ``None`` on a TOTALS row whose :attr:`cache_basis` reports a
        mixed pool.  The denominator's premise — "input that could have
        been a hit" — is established per binding by its provider having
        reported the dimension, and pooling across a binding that reported
        nothing silently drops that premise while keeping the formula.
        The measured subset's rate is published on the basis instead.
        """
        if self.cache_read_tokens is None:
            return None
        if self.cache_basis is not None and self.cache_basis.is_mixed:
            return None
        denominator = self.cache_read_tokens + self.uncached_input_tokens
        if denominator <= 0:
            return None
        return round(100.0 * self.cache_read_tokens / denominator, 2)

    def observe(
        self,
        *,
        uncached_input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        reasoning_source: Optional[str] = None,
        cost_usd: Optional[float] = None,
        cost_source: Optional[str] = None,
        finish_reason: Optional[str] = None,
        turn_index: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Fold one provider response into this binding.

        Called once per response.  Every argument is a DELTA for this
        response, never a running total — the caller is the same
        once-per-response hook that accumulates ``spend_*`` onto the turn,
        and feeding it a level reading would double-count exactly as it
        would there.
        """
        self.responses += 1
        self.uncached_input_tokens += uncached_input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.cache_read_tokens = _add_optional(
            self.cache_read_tokens, cache_read_tokens)
        self.cache_creation_tokens = _add_optional(
            self.cache_creation_tokens, cache_creation_tokens)
        self.reasoning_tokens = _add_optional(
            self.reasoning_tokens, reasoning_tokens)
        if reasoning_tokens is not None:
            self.reasoning_responses += 1
            self.reasoning_source = _merge_source(
                self.reasoning_source,
                reasoning_source or REASONING_SOURCE_PROVIDER)

        if cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + cost_usd
            if cost_source is not None:
                if self.cost_source is None:
                    self.cost_source = cost_source
                elif self.cost_source != cost_source:
                    # Two sources contributed to one number.  Saying so
                    # beats letting the first one speak for the total.
                    self.cost_source = "mixed"

        if finish_reason:
            self.finish_reasons[finish_reason] = (
                self.finish_reasons.get(finish_reason, 0) + 1)

        if turn_index is not None and turn_index != self._last_turn_index:
            self.turns += 1
            self._last_turn_index = turn_index

        if timestamp:
            if self.first_used is None:
                self.first_used = timestamp
            self.last_used = timestamp

    def as_dict(self) -> Dict[str, Any]:
        """The reportable view, with the derived figures filled in.

        Optional dimensions nothing reported are OMITTED rather than
        rendered as ``null``: an absent key reads as "this provider does
        not report it", which is the truth, where ``null`` beside a row of
        numbers reads as a failed measurement.
        """
        out: Dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "responses": self.responses,
            "turns": self.turns,
            "uncached_input_tokens": self.uncached_input_tokens,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cache_read_tokens is not None:
            out["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_creation_tokens is not None:
            out["cache_creation_tokens"] = self.cache_creation_tokens
        hit = self.cache_hit_percent
        if hit is not None:
            out["cache_hit_percent"] = hit
        if self.cache_basis is not None and self.cache_basis.is_mixed:
            # Emitted ONLY in the mixed case.  On a homogeneous pool the
            # percentage above is sound and a basis beside it is noise; on
            # a pool where nothing was measured ``cache_read_tokens`` is
            # already absent, which reads correctly on its own.
            out["cache_hit_basis"] = self.cache_basis.as_dict()
        if self.reasoning_tokens is not None:
            out["reasoning_tokens"] = self.reasoning_tokens
            out["reasoning_source"] = self.reasoning_source
            answer = self.answer_tokens
            if answer is not None:
                out["answer_tokens"] = answer
        if self.cost_usd is not None:
            out["cost_usd"] = round(self.cost_usd, 6)
            out["cost_source"] = self.cost_source
        if self.finish_reasons:
            out["finish_reasons"] = dict(self.finish_reasons)
        if self.first_used:
            out["first_used"] = self.first_used
        if self.last_used:
            out["last_used"] = self.last_used
        return out


class ConsumptionLedger:
    """Per-binding spend for ONE session.

    Deliberately pure logic holding no session reference — the session
    feeds it responses and asks it for a report, which keeps the
    interesting part (the ``None``-preserving sums, the binding key, the
    turn counting) testable without standing up a session.  Same posture
    as :class:`shared.budget_control.BudgetTracker`.

    Own-session only.  A subagent runs its own :class:`JaatoSession` and
    therefore its own ledger; a parent's report never silently absorbs a
    child's spend.  Aggregating a cascade is
    :class:`~shared.budget_control.CascadeBudgetPool`'s job, and a rollup
    here would quietly compete with it.

    Insertion order is preserved, so ``bindings()`` reads chronologically
    by first use — the order a tier ladder was actually walked in.
    """

    def __init__(self) -> None:
        self._bindings: Dict[BindingKey, BindingUsage] = {}
        # Distinct turn indices seen, so the TOTAL turn count is exact.
        # It cannot be derived from the per-binding counts: summing them
        # double-counts a turn that crossed an ``enter_tier`` (both
        # bindings served it), and taking the max under-counts when two
        # bindings served DIFFERENT turns.  Bounded by the session's turn
        # count, which the turn-accounting list already holds in full.
        self._turn_indices: set = set()

    def __len__(self) -> int:
        return len(self._bindings)

    def observe(
        self,
        *,
        provider: str,
        model: str,
        tier: Optional[str] = None,
        **fields: Any,
    ) -> BindingUsage:
        """Fold one response into the binding that served it.

        Args:
            provider: Active provider plugin name.
            model: Active model id.
            tier: Active tier name, or ``None`` in single-model mode.
            **fields: Passed through to :meth:`BindingUsage.observe`.

        Returns:
            The binding, after the fold — handy for tests and tracing.
        """
        key: BindingKey = (provider, model, tier)
        binding = self._bindings.get(key)
        if binding is None:
            binding = BindingUsage(provider=provider, model=model, tier=tier)
            self._bindings[key] = binding
        binding.observe(**fields)
        turn_index = fields.get("turn_index")
        if turn_index is not None:
            self._turn_indices.add(turn_index)
        return binding

    def bindings(self) -> List[BindingUsage]:
        """Every binding this session used, in first-use order."""
        return list(self._bindings.values())

    def totals(self) -> BindingUsage:
        """One row summing every binding.

        ``provider`` / ``model`` are reported as ``"*"`` rather than as
        the active binding's: a total that names one model is read as that
        model's total.

        ``turns`` comes from the ledger's own set of turn indices, not
        from the per-binding counts, because neither way of combining
        those is right: summing double-counts a turn that crossed an
        ``enter_tier`` (each binding it touched counted it), and taking
        the max under-counts two bindings that served different turns.

        ``cache_hit_percent`` is WITHHELD here when the bindings disagree
        about whether their provider reports a cache at all — see
        :class:`CacheMeasurementBasis`, which is attached in its place.
        The token sums are unaffected: they are sums of measurements, and
        a dimension nothing reported still contributes nothing to them.
        """
        total = BindingUsage(provider="*", model="*", tier=None)
        basis = CacheMeasurementBasis()
        for binding in self._bindings.values():
            if binding.cache_read_tokens is None:
                basis.unmeasured_bindings += 1
                basis.unmeasured_input_tokens += binding.uncached_input_tokens
            else:
                basis.measured_bindings += 1
                basis.measured_input_tokens += (
                    binding.cache_read_tokens + binding.uncached_input_tokens)
            total.responses += binding.responses
            total.uncached_input_tokens += binding.uncached_input_tokens
            total.output_tokens += binding.output_tokens
            total.total_tokens += binding.total_tokens
            total.cache_read_tokens = _add_optional(
                total.cache_read_tokens, binding.cache_read_tokens)
            total.cache_creation_tokens = _add_optional(
                total.cache_creation_tokens, binding.cache_creation_tokens)
            total.reasoning_tokens = _add_optional(
                total.reasoning_tokens, binding.reasoning_tokens)
            total.reasoning_responses += binding.reasoning_responses
            if binding.reasoning_source is not None:
                total.reasoning_source = _merge_source(
                    total.reasoning_source, binding.reasoning_source)
            if binding.cost_usd is not None:
                total.cost_usd = (total.cost_usd or 0.0) + binding.cost_usd
                if total.cost_source is None:
                    total.cost_source = binding.cost_source
                elif total.cost_source != binding.cost_source:
                    total.cost_source = "mixed"
            for reason, count in binding.finish_reasons.items():
                total.finish_reasons[reason] = (
                    total.finish_reasons.get(reason, 0) + count)
            for stamp, attr in (
                (binding.first_used, "first_used"),
                (binding.last_used, "last_used"),
            ):
                if not stamp:
                    continue
                current = getattr(total, attr)
                if current is None:
                    setattr(total, attr, stamp)
                elif attr == "first_used":
                    setattr(total, attr, min(current, stamp))
                else:
                    setattr(total, attr, max(current, stamp))
        if total.cache_read_tokens is not None and basis.measured_input_tokens > 0:
            basis.cache_hit_percent = round(
                100.0 * total.cache_read_tokens / basis.measured_input_tokens, 2)
        total.cache_basis = basis
        total.turns = len(self._turn_indices)
        return total
