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
``thinking_tokens`` and ``cost_usd`` stay ``None`` until something reports
one.  A provider with no prompt cache must not read as a provider whose
cache never hits, and a session with no pricing table must not read as
free.  ``cost_source`` says which source supplied a cost, so a consumer
can tell a billed figure from an estimated one.
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

#: Accepted ``detail`` levels for a consumption report.
DETAIL_SUMMARY = "summary"
DETAIL_FULL = "full"
VALID_DETAIL_LEVELS = (DETAIL_SUMMARY, DETAIL_FULL)


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
        thinking_tokens: Reasoning tokens, a SUBSET of ``output_tokens``,
            or ``None`` when the provider reports none.
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
    thinking_tokens: Optional[int] = None
    total_tokens: int = 0
    cost_usd: Optional[float] = None
    cost_source: Optional[str] = None
    finish_reasons: Dict[str, int] = field(default_factory=dict)
    first_used: Optional[str] = None
    last_used: Optional[str] = None
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
        """
        if self.cache_read_tokens is None:
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
        thinking_tokens: Optional[int] = None,
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
        self.thinking_tokens = _add_optional(
            self.thinking_tokens, thinking_tokens)

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
        if self.thinking_tokens is not None:
            out["thinking_tokens"] = self.thinking_tokens
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
        """
        total = BindingUsage(provider="*", model="*", tier=None)
        for binding in self._bindings.values():
            total.responses += binding.responses
            total.uncached_input_tokens += binding.uncached_input_tokens
            total.output_tokens += binding.output_tokens
            total.total_tokens += binding.total_tokens
            total.cache_read_tokens = _add_optional(
                total.cache_read_tokens, binding.cache_read_tokens)
            total.cache_creation_tokens = _add_optional(
                total.cache_creation_tokens, binding.cache_creation_tokens)
            total.thinking_tokens = _add_optional(
                total.thinking_tokens, binding.thinking_tokens)
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
        total.turns = len(self._turn_indices)
        return total
