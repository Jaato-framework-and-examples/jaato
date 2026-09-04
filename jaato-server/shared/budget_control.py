"""Budget control — multi-dimensional resource ceilings + graceful degradation.

The value type plumbed through profiles as ``SubagentProfile.budget_control``.
Lives in ``shared/`` as a free-standing module for the same reason
:mod:`shared.runtime_limits` does: the profile dataclass references it
statically, and the runtime consumers must be able to import it without
pulling in the subagent plugin package.

What it is
----------

Two halves, declared together under one profile key:

* **``limits``** — multi-dimensional ceilings (``usd`` / ``tokens`` /
  ``seconds`` / ``tool_calls`` / ``turns``).  Any dimension may be
  omitted, meaning "unbounded".  Distinct from
  :class:`~shared.runtime_limits.RuntimeLimits`, which caps *host*
  resources (memory / pids / cpu) — this caps *agent economics*.
* **``degrade``** — an ordered ladder of rungs.  A rung fires when
  **any** declared dimension crosses its ``at`` percentage, and applies
  a sparse **overlay** onto the session's ``model_tiers`` table: the
  tier vocabulary and the model's cognitive role are untouched; only the
  model each tier *points at* changes.  A rung may instead (or also)
  carry a terminal ``action``.

Why degradation rebinds tiers rather than switching them
--------------------------------------------------------

Tier labels (``planner`` / ``dispatcher`` / ``executor`` / ``vision``)
are a **cognitive/role** axis, not a cost axis — :mod:`shared.model_tiers`
says so explicitly ("order is conceptual … operators are free to wire
them however the provider's pricing makes sense").  An operator may map
``planner`` to a cheap model and ``executor`` to an expensive one.  So a
budget layer must not infer a cost ordering from the labels, and must not
move the agent between tiers (that would also yank its role identity).

Instead each rung declares, per tier, the exact replacement binding.  A
brownout, not a blackout: every room dims to a cheaper bulb, none are
switched off.  This needs no cost ordering, introduces no new tier
labels, and never fights the model's own ``enter_tier`` choices.

See ``docs/design/budget-control-degradation.md`` for the full design,
including the runtime tracker and the ``switch_tier`` re-resolve path
that applies an overlay to the *currently active* tier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

# Deliberate reuse of the tier-entry normalizer: a ``degrade[].model_tiers``
# overlay IS a tier table, so it must accept the byte-identical grammar
# (``"model"`` | ``{model, provider}``) the base ``model_tiers`` accepts.
# Re-implementing it here would let the two grammars drift apart.
from .model_tiers import (
    RESERVED_KEYS,
    VALID_TIER_NAMES,
    TierEntry,
    _normalize_tier_entry,
)

logger = logging.getLogger(__name__)


# The dimensions a budget can cap.  Each maps to a measurement jaato
# already emits on the event bus (see the design note §2):
#   usd        -> UsageBreakdown.cost_usd (provider-reported -> pricing.json)
#   tokens     -> running total token count
#   seconds    -> summed turn / tool wall-clock
#   tool_calls -> count of tool.call_completed
#   turns      -> turn counter
VALID_DIMENSIONS: frozenset = frozenset(
    {"usd", "tokens", "seconds", "tool_calls", "turns"}
)

# Terminal actions a rung may take instead of / alongside an overlay.
#   finalize -> inject "wrap up and answer with what you have" (graceful)
#   abort    -> hard stop (ungraceful; for when a partial answer is worthless)
#   escalate -> hand off to the cascade owner / a human
ACTION_FINALIZE = "finalize"
ACTION_ABORT = "abort"
ACTION_ESCALATE = "escalate"
VALID_ACTIONS: frozenset = frozenset(
    {ACTION_FINALIZE, ACTION_ABORT, ACTION_ESCALATE}
)


class CascadeExhaustedError(RuntimeError):
    """A child was to be spawned into a cascade with no headroom left.

    Raised by :meth:`CascadeBudgetPool.child_config` instead of handing back
    a zero-valued limit.  Zero is not a budget a session can run under — the
    first turn would immediately breach it — and returning ``None``
    ("unbudgeted") would be catastrophically wrong here, since the reason
    there is no budget is that the cascade is OUT of budget.

    **Why refuse rather than spawn-and-abort.** Spawning a doomed child
    SPENDS the resource the cascade has just run out of: a bootstrap plus —
    because ``abort`` lets the in-flight turn finish (§5.1) — up to one full
    turn of a model you cannot afford, multiplied by N under fan-out.  A
    ceiling that must spend to enforce itself is not a ceiling; that is the
    Finding-B defect one layer up.  Refusing the spawn is the only option
    that costs nothing.

    **Carries evidence, not just a message.**  The refusal has to be
    demonstrable by whoever reads the trace afterwards, at the same fidelity
    as a session-level refusal — otherwise the only proof is that the
    caller's own ``except`` block ran, which is self-reported.  So the
    structured fields below are framework-generated and are what the owner
    surfaces; ``as_payload()`` is the renderable form.  Note that a refused
    spawn recorded as such is materially different from an ABSENT one:
    "absent" and "never attempted" are indistinguishable in a trace.

    Attributes:
        cascade_driver_id: The cascade with no headroom.
        limits: The :class:`EffectiveLimits` that would have been produced —
            both inputs and the clamp, not just the outcome.
        exhausted: Dimensions with no headroom left.
    """

    def __init__(
        self,
        cascade_driver_id: str,
        limits: "EffectiveLimits",
        message: Optional[str] = None,
    ) -> None:
        self.cascade_driver_id = cascade_driver_id
        self.limits = limits
        self.exhausted = limits.exhausted
        super().__init__(message or (
            f"cascade {cascade_driver_id} has no headroom left on "
            f"{', '.join(self.exhausted)} — refusing to spawn a child that "
            f"could not run a single turn ({limits.describe()})"
        ))

    def as_payload(self) -> Dict[str, Any]:
        """Structured form for an event / cascade record / log line."""
        return {
            "cascade_driver_id": self.cascade_driver_id,
            "reason": "cascade_budget_exhausted",
            "exhausted_dimensions": list(self.exhausted),
            "effective": dict(self.limits.effective),
            "profile_limits": dict(self.limits.profile_limits),
            "cascade_remaining": dict(self.limits.cascade_remaining),
            "clamped": list(self.limits.clamped),
            "detail": self.limits.describe(),
        }

    def render(self) -> str:
        """One-line client-facing form, matching the session-level refusals."""
        return (
            f"[cascade_budget_exhausted ({', '.join(self.exhausted)}) — "
            f"refusing to spawn: {self.limits.describe()}]"
        )


class BudgetControlConfigError(ValueError):
    """Raised when a ``budget_control`` block can't be parsed or is
    internally inconsistent.  Mirrors
    :class:`~shared.model_tiers.ModelTierConfigError` so profile loaders
    can treat both structured blocks the same way."""


def _parse_at(raw: object) -> float:
    """Coerce a rung's ``at`` into a percentage in ``(0, 100]``.

    Accepts the two spellings an author naturally writes, with ONE
    unambiguous rule — the value is always a **percentage**, never a
    fraction:

        at: 70      -> 70.0
        at: "70%"   -> 70.0

    A fraction-looking value (``0.7``) is therefore 0.7 percent, not
    70 — deliberate, because guessing the author's intent from
    magnitude would be a silent mis-read of a safety ceiling.
    """
    if isinstance(raw, bool):  # bool is an int subclass — reject explicitly
        raise BudgetControlConfigError(f"'at' must be a number or percent string, got {raw!r}")
    value: Any = raw
    if isinstance(raw, str):
        text = raw.strip().rstrip("%").strip()
        if not text:
            raise BudgetControlConfigError(f"'at' is empty: {raw!r}")
        try:
            value = float(text)
        except ValueError:
            raise BudgetControlConfigError(
                f"'at' must be a number or a percent string (e.g. '70%'), got {raw!r}"
            ) from None
    if not isinstance(value, (int, float)):
        raise BudgetControlConfigError(
            f"'at' must be a number or a percent string, got {type(raw).__name__}"
        )
    value = float(value)
    if not 0.0 < value <= 100.0:
        raise BudgetControlConfigError(
            f"'at' must be a percentage in (0, 100], got {value}"
        )
    return value


def _parse_degrade_overlay(
    raw_overlay: Any, *, index: int
) -> Dict[str, TierEntry]:
    """Parse one rung's ``model_tiers`` overlay into tier entries.

    Split out of :meth:`DegradeRung.from_dict` to keep that method under
    the complexity ceiling — the per-key validation is a self-contained
    concern (it enforces what an OVERLAY may say, as opposed to what a
    base tier table may say).

    Args:
        raw_overlay: The rung's raw ``model_tiers`` value.  Falsy yields
            an empty overlay.
        index: The rung's position, used only in error messages.

    Returns:
        Tier name → :class:`~shared.model_tiers.TierEntry`.  Entries never
        carry a ``description`` — see the refusal below.

    Raises:
        BudgetControlConfigError: Not an object; a reserved control key; a
            name outside :data:`~shared.model_tiers.VALID_TIER_NAMES`; a
            ``description`` (a rung rebinds a tier's model, not its role);
            or an entry the shared tier-entry normalizer rejects.
    """
    overlay: Dict[str, TierEntry] = {}
    if not raw_overlay:
        return overlay
    if not isinstance(raw_overlay, Mapping):
        raise BudgetControlConfigError(
            f"degrade[{index}].model_tiers: expected an object, "
            f"got {type(raw_overlay).__name__}"
        )
    for tier_name, entry in raw_overlay.items():
        key = str(tier_name)
        # An overlay rebinds EXISTING tiers; the control keys (initial /
        # fallback) select a tier rather than bind a model, so they are
        # meaningless in an overlay.
        if key in RESERVED_KEYS:
            raise BudgetControlConfigError(
                f"degrade[{index}].model_tiers: control key '{key}' is "
                f"not valid in an overlay (an overlay rebinds tier→model "
                f"only; set initial/fallback on the base model_tiers)"
            )
        if key not in VALID_TIER_NAMES:
            raise BudgetControlConfigError(
                f"degrade[{index}].model_tiers: '{key}' is not a tier name "
                f"({', '.join(sorted(VALID_TIER_NAMES))})"
            )
        # A rung rebinds a tier's MODEL; the tier's role — and so the prose
        # describing it — is untouched by a brownout.  Accepting a
        # description here would also be a lie: the ``enter_tier`` schema is
        # built once at session configure time and lives in the prompt-cache
        # prefix, so a rung firing mid-session could never change what the
        # model reads.  Refuse it rather than silently ignore it.
        if isinstance(entry, Mapping) and "description" in entry:
            raise BudgetControlConfigError(
                f"degrade[{index}].model_tiers.{key}: 'description' is "
                f"not valid in an overlay (a rung rebinds a tier's model, "
                f"not its role; describe the tier on the base model_tiers "
                f"instead)"
            )
        try:
            overlay[key] = _normalize_tier_entry(key, entry)
        except Exception as exc:
            raise BudgetControlConfigError(
                f"degrade[{index}].model_tiers.{key}: {exc}"
            ) from None
    return overlay


@dataclass(frozen=True)
class DegradeRung:
    """One rung of the degradation ladder.

    Attributes:
        at_percent: Threshold in ``(0, 100]``.  The rung fires when ANY
            declared dimension's usage reaches this percentage of its
            limit ("first dimension wins" — see the design note §5.1).
        model_tiers: Sparse overlay onto the session's tier table, keyed
            by tier name (a subset of
            :data:`~shared.model_tiers.VALID_TIER_NAMES`).  Tiers absent
            from the overlay keep their current binding.  Empty when the
            rung only carries an ``action``.
        action: Optional terminal action from :data:`VALID_ACTIONS`.

    Rungs are **latched** (fire once, never un-fire) and **cumulative**
    (later rungs stack on earlier ones) at runtime; both properties are
    enforced by the tracker, not by this value type.
    """

    at_percent: float
    model_tiers: Mapping[str, TierEntry] = field(default_factory=dict)
    action: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, index: int) -> "DegradeRung":
        """Parse one rung.  ``index`` is used only for error messages."""
        if not isinstance(raw, Mapping):
            raise BudgetControlConfigError(
                f"degrade[{index}]: expected an object, got {type(raw).__name__}"
            )
        if "at" not in raw:
            raise BudgetControlConfigError(f"degrade[{index}] is missing 'at'")
        try:
            at_percent = _parse_at(raw["at"])
        except BudgetControlConfigError as exc:
            raise BudgetControlConfigError(f"degrade[{index}]: {exc}") from None

        overlay = _parse_degrade_overlay(raw.get("model_tiers"), index=index)

        action = raw.get("action")
        if action is not None:
            if not isinstance(action, str) or action.strip() not in VALID_ACTIONS:
                raise BudgetControlConfigError(
                    f"degrade[{index}].action: '{action}' is not a valid action "
                    f"({', '.join(sorted(VALID_ACTIONS))})"
                )
            action = action.strip()

        if not overlay and action is None:
            raise BudgetControlConfigError(
                f"degrade[{index}] does nothing — declare 'model_tiers' (an "
                f"overlay), 'action', or both"
            )
        return cls(at_percent=at_percent, model_tiers=overlay, action=action)

    def to_dict(self) -> Dict[str, Any]:
        """Round-trip form for the session envelope (daemon -> runner)."""
        out: Dict[str, Any] = {"at": self.at_percent}
        if self.model_tiers:
            out["model_tiers"] = {
                name: {"model": e.model, "provider": e.provider}
                if e.provider else {"model": e.model}
                for name, e in self.model_tiers.items()
            }
        if self.action:
            out["action"] = self.action
        return out


@dataclass(frozen=True)
class BudgetControlConfig:
    """Resolved ``budget_control`` block for one profile / session.

    Held on :class:`~shared.plugins.subagent.config.SubagentProfile` as
    ``budget_control``.  A profile with neither ``limits`` nor
    ``degrade`` parses to ``None`` (no budget control), so consumers can
    treat "absent" and "empty" identically.

    Attributes:
        limits: Dimension → positive ceiling.  Keys are a subset of
            :data:`VALID_DIMENSIONS`; an absent dimension is unbounded.
        degrade: Rungs ordered by strictly increasing ``at_percent``.
    """

    limits: Mapping[str, float] = field(default_factory=dict)
    degrade: Tuple[DegradeRung, ...] = ()

    def __post_init__(self) -> None:
        for dim, value in self.limits.items():
            if dim not in VALID_DIMENSIONS:
                raise BudgetControlConfigError(
                    f"limits: unknown dimension '{dim}' "
                    f"(valid: {', '.join(sorted(VALID_DIMENSIONS))})"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BudgetControlConfigError(
                    f"limits.{dim}: must be a number, got {type(value).__name__}"
                )
            if value <= 0:
                raise BudgetControlConfigError(
                    f"limits.{dim}: must be > 0, got {value}"
                )
        # Strictly increasing thresholds: two rungs at the same 'at' would
        # make "which fires first" arbitrary, and is nearly always a
        # copy-paste slip rather than intent.
        previous: Optional[float] = None
        for rung in self.degrade:
            if previous is not None and rung.at_percent <= previous:
                raise BudgetControlConfigError(
                    f"degrade thresholds must strictly increase; "
                    f"{rung.at_percent} follows {previous}"
                )
            previous = rung.at_percent
        # A ladder with no ceilings can never fire — 'at' is a percentage
        # OF a limit, so without limits there is nothing to take a
        # percentage of.  Fail loud rather than sit inert.
        if self.degrade and not self.limits:
            raise BudgetControlConfigError(
                "degrade declared without limits — 'at' is a percentage of a "
                "limit, so no rung could ever fire; declare at least one of: "
                + ", ".join(sorted(VALID_DIMENSIONS))
            )

    @property
    def has_tier_overlays(self) -> bool:
        """True if any rung rebinds a tier.

        Consumers (the profile validator) use this to reject a ladder
        that overlays tiers on a profile which declares no
        ``model_tiers`` — the overlay would have no table to patch.
        """
        return any(rung.model_tiers for rung in self.degrade)

    @classmethod
    def from_dict(
        cls, data: Optional[Mapping[str, Any]]
    ) -> Optional["BudgetControlConfig"]:
        """Build from the profile dict shape, or ``None`` when absent/empty.

        Raises:
            BudgetControlConfigError: On any malformed field.  Profile
                loaders surface this the same way they surface a bad
                ``runtime_limits`` — at load time, not at session start.
        """
        if not data:
            return None
        if not isinstance(data, Mapping):
            raise BudgetControlConfigError(
                f"budget_control must be an object, got {type(data).__name__}"
            )
        unknown = set(data) - {"limits", "degrade"}
        if unknown:
            raise BudgetControlConfigError(
                f"budget_control: unknown key(s) {sorted(unknown)} "
                f"(valid: degrade, limits)"
            )

        raw_limits = data.get("limits") or {}
        if raw_limits and not isinstance(raw_limits, Mapping):
            raise BudgetControlConfigError(
                f"budget_control.limits must be an object, "
                f"got {type(raw_limits).__name__}"
            )
        limits = {str(k): v for k, v in raw_limits.items()}

        raw_degrade = data.get("degrade") or []
        if raw_degrade and not isinstance(raw_degrade, (list, tuple)):
            raise BudgetControlConfigError(
                f"budget_control.degrade must be a list, "
                f"got {type(raw_degrade).__name__}"
            )
        degrade = tuple(
            DegradeRung.from_dict(rung, index=i)
            for i, rung in enumerate(raw_degrade)
        )

        if not limits and not degrade:
            return None
        return cls(limits=limits, degrade=degrade)

    def to_dict(self) -> Dict[str, Any]:
        """Round-trip form for the session envelope (daemon -> runner).

        The profile parses ``budget_control`` EAGERLY into this object so a
        malformed block fails at profile load.  The wire therefore carries a
        re-serialised dict (rather than the author's raw YAML), which the
        runner re-parses via :meth:`from_dict` — validating a second time,
        cheaply, on the far side of the process boundary.
        """
        out: Dict[str, Any] = {}
        if self.limits:
            out["limits"] = dict(self.limits)
        if self.degrade:
            out["degrade"] = [r.to_dict() for r in self.degrade]
        return out


def merge_limits(
    parent_limits: Mapping[str, float],
    child_limits: Mapping[str, float],
) -> Dict[str, float]:
    """Min-wins merge of two limit maps — a child may only TIGHTEN.

    ``effective[dim] = min(child[dim], parent[dim])`` when both declare a
    dimension; otherwise whichever one declares it wins (an absent
    dimension is unbounded, so the declared one is strictly tighter).

    Min-wins (rather than the child-replaces-parent used by most scalar
    profile fields) is the safety direction: a child profile must never
    be able to grant itself a larger ceiling than the parent that spawned
    it.  Mirrors how ``max_turns`` already takes the most restrictive
    value across parents.
    """
    merged: Dict[str, float] = dict(parent_limits)
    for dim, value in child_limits.items():
        if dim in merged:
            merged[dim] = min(merged[dim], value)
        else:
            merged[dim] = value
    return merged


@dataclass
class BudgetUsage:
    """Running consumption per dimension.  Mutable — the tracker owns one."""

    usd: float = 0.0
    tokens: float = 0.0
    seconds: float = 0.0
    tool_calls: float = 0.0
    turns: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "usd": self.usd,
            "tokens": self.tokens,
            "seconds": self.seconds,
            "tool_calls": self.tool_calls,
            "turns": self.turns,
        }


class BudgetTracker:
    """Accumulates consumption and decides which degrade rungs have fired.

    Deliberately **pure logic** — it holds no session, provider, or event-bus
    reference.  The session feeds it numbers via :meth:`observe` and acts on
    the rungs it returns.  That keeps the firing policy (the part with the
    interesting invariants: first-dimension-wins, latching, cumulative
    overlays) unit-testable without standing up a session.

    Semantics (design note §5):

    * **First dimension wins** — :meth:`usage_fraction` is the MAX over the
      declared dimensions, so blowing the dollar budget fires a rung even
      when tokens are nowhere near their ceiling.
    * **Latched** — a rung fires at most once, tracked in ``_fired``.  This
      is load-bearing: GC *lowers* token usage, so an unlatched token rung
      would flap the model between an expensive and a cheap binding on every
      GC cycle.
    * **Cumulative** — rungs are not undone; the session keeps every overlay
      already applied and layers later ones on top.

    A dimension the profile did not declare is unbounded and contributes
    nothing to the fraction.  ``usd`` only advances when a cost is actually
    known (provider-reported or priced) — see :meth:`observe`.
    """

    def __init__(self, config: BudgetControlConfig) -> None:
        self._config = config
        self._usage = BudgetUsage()
        self._fired: set = set()  # indices into config.degrade

    @property
    def usage(self) -> BudgetUsage:
        return self._usage

    @property
    def config(self) -> BudgetControlConfig:
        return self._config

    def observe(
        self,
        *,
        usd: Optional[float] = None,
        tokens: Optional[float] = None,
        seconds: Optional[float] = None,
        tool_calls: Optional[float] = None,
        turns: Optional[float] = None,
    ) -> Tuple["DegradeRung", ...]:
        """Add consumption, then return the rungs that fire as a result.

        Every argument is a **delta**, not an absolute.  ``None`` means "no
        news about this dimension" and leaves it untouched — importantly,
        that is how an unknown cost is handled: when neither the provider nor
        the pricing table can supply a number, ``usd`` stays put rather than
        advancing on a guess (a budget must never hard-stop on an estimate it
        invented).

        Returns rungs in ladder order, each at most once across the tracker's
        lifetime.  An empty tuple is the overwhelmingly common case.
        """
        for name, delta in (
            ("usd", usd), ("tokens", tokens), ("seconds", seconds),
            ("tool_calls", tool_calls), ("turns", turns),
        ):
            if delta is None:
                continue
            try:
                value = float(delta)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            setattr(self._usage, name, getattr(self._usage, name) + value)
        return self._newly_fired()

    def restore_usage(self, usage: Dict[str, float]) -> None:
        """Re-seed accumulated usage after a session reload.

        ``BudgetTracker`` accumulates in memory only, so an unload/reload
        rebuilt it with ZEROED usage and every cross-turn dimension silently
        restarted.  For a suspend/resume cascade that is the whole ceiling: a
        session evicted during a wait came back with ``turns`` at 0, so a
        ``turns: 2`` limit never fired however many resumes ran.  Confirmed
        live 2026-08-23 -- the session unloaded during every wait, and only a
        ``usd`` limit small enough to cross WITHIN one turn ever tripped.

        Absolute, not a delta: this restores a snapshot, it does not
        re-observe.  Unknown keys are ignored and non-numeric values are
        skipped with a warning, so an older or newer snapshot loads rather
        than raising -- a budget that refuses to restore is worse than one
        that restores what it understands.

        Rungs are deliberately NOT re-latched: :meth:`observe` re-evaluates
        the ladder on the next turn against the restored totals, so a rung
        already crossed fires once more after a reload instead of being
        skipped.  Firing again is the safe direction -- a rebind is
        idempotent and an abort must re-assert.
        """
        for name in VALID_DIMENSIONS:
            value = usage.get(name)
            if value is None:
                continue
            try:
                setattr(self._usage, name, float(value))
            except (TypeError, ValueError):
                logger.warning(
                    "budget: ignoring non-numeric restored %s=%r", name, value)


    def usage_fraction(self) -> float:
        """Highest ``used / limit`` ratio across the DECLARED dimensions.

        Returns ``0.0`` when nothing is capped.  Not clamped — a run that
        overshoots reports > 1.0, which is meaningful (it says by how much).
        """
        usage = self._usage.as_dict()
        fractions = [
            usage[dim] / limit
            for dim, limit in self._config.limits.items()
            if limit
        ]
        return max(fractions, default=0.0)

    def exceeded_dimensions(self) -> Dict[str, float]:
        """Dimensions at or past 100% of their ceiling → their fraction.

        Reported alongside a fired rung so an operator can see WHICH ceiling
        drove the degradation, not merely that one did.
        """
        usage = self._usage.as_dict()
        return {
            dim: usage[dim] / limit
            for dim, limit in self._config.limits.items()
            if limit and usage[dim] >= limit
        }

    def describe_pressure(self) -> str:
        """Human-readable "which ceiling is driving this" string.

        Names the DRIVING dimension (the one with the highest fraction), not
        only the exhausted ones.  An earlier version listed
        :meth:`exceeded_dimensions`, which is empty below 100% — so a rung
        firing at 50% reported the useless "50% of budget" and never said
        WHICH of several declared dimensions had moved.
        """
        usage = self._usage.as_dict()
        ranked = sorted(
            ((usage[d] / lim, d) for d, lim in self._config.limits.items() if lim),
            reverse=True,
        )
        if not ranked:
            return "no limits declared"
        return ", ".join(f"{d} {f * 100:.0f}%" for f, d in ranked[:3])

    def _newly_fired(self) -> Tuple["DegradeRung", ...]:
        """Rungs whose threshold is now crossed and which have not fired."""
        fraction = self.usage_fraction() * 100.0
        fired = []
        for index, rung in enumerate(self._config.degrade):
            if index in self._fired:
                continue
            if fraction >= rung.at_percent:
                self._fired.add(index)
                fired.append(rung)
        return tuple(fired)


def overlay_tier_table(
    tiers: Dict[str, TierEntry],
    overlay: Mapping[str, TierEntry],
) -> Dict[str, str]:
    """Apply a degrade overlay onto a live tier table, in place.

    Mutates ``tiers`` (the dict held by the session's ``ModelTierConfig``)
    because the config object is frozen while its tier mapping is not — the
    binding is what changes, never the tier vocabulary.

    Returns a ``{tier: "old-model -> new-model"}`` map of what actually
    changed, for logging and for the emitted event.  A rung that rebinds a
    tier to the model it already had contributes nothing.
    """
    changes: Dict[str, str] = {}
    for tier_name, entry in overlay.items():
        current = tiers.get(tier_name)
        if current is not None and (
            current.model == entry.model and current.provider == entry.provider
        ):
            continue
        was = current.model if current is not None else "(undeclared)"
        # Carry the base tier's description forward.  A rung cannot set one
        # (DegradeRung.from_dict refuses it), so the overlay entry's is
        # always None — replacing the entry wholesale would drop the
        # profile's prose, contradicting "only the model each tier points
        # at changes".  Harmless for the tool schema, which was already
        # built, but ``describe_tier`` is also read by diagnostics.
        if current is not None and current.description:
            entry = _dc_replace(entry, description=current.description)
        tiers[tier_name] = entry
        changes[tier_name] = f"{was} -> {entry.model}"
    return changes


@dataclass(frozen=True)
class EffectiveLimits:
    """Result of clamping a child's own ceiling against a cascade's remainder.

    Carries BOTH inputs, not just the outcome, so an operator can see that
    ``min()`` actually clamped rather than having to infer it from a single
    number — the observability the first cascade PoC asked for.

    Attributes:
        effective: The ceiling the child session will actually run under.
        profile_limits: What the child's own profile asked for.
        cascade_remaining: What was left in the cascade pool at spawn time.
        clamped: Dimensions where the cascade remainder won (i.e. the child
            asked for more than the cascade had left).
    """

    effective: Mapping[str, float]
    profile_limits: Mapping[str, float]
    cascade_remaining: Mapping[str, float]
    clamped: Tuple[str, ...] = ()

    @property
    def exhausted(self) -> Tuple[str, ...]:
        """Dimensions with no headroom left (effective <= 0).

        Non-empty means the cascade cannot afford this child at all.
        """
        return tuple(sorted(d for d, v in self.effective.items() if v <= 0))

    def describe(self) -> str:
        """One-line, per-dimension account of the clamp for logs/events."""
        if not self.effective:
            return "unbudgeted"
        parts = []
        for dim in sorted(self.effective):
            prof = self.profile_limits.get(dim)
            rem = self.cascade_remaining.get(dim)
            mark = " CLAMPED" if dim in self.clamped else ""
            parts.append(
                f"{dim}: profile={prof if prof is not None else '-'} "
                f"cascade_remaining={rem if rem is not None else '-'} "
                f"-> {self.effective[dim]}{mark}"
            )
        return "; ".join(parts)


class ReconcileResult(NamedTuple):
    """Outcome of :meth:`CascadeBudgetPool.reconcile_session`.

    Both halves are needed by the caller: ``deltas`` says what moved (for
    logging), ``fired`` says which POOL rungs those deltas crossed and must
    therefore be pushed to the cascade's live children.  An earlier version
    returned only the deltas and silently dropped ``fired`` — which meant
    cascade degrade ladders could never fire, invisible until a cascade
    declared one.
    """

    deltas: Dict[str, float]
    fired: Tuple["DegradeRung", ...]


class CascadeBudgetPool:
    """The AGGREGATE ceiling for one cascade (cid) — a single shared counter.

    Distinct from :class:`BudgetTracker`, which is per-session.  The pool is
    what makes a cascade cap a real ceiling rather than a per-child
    suggestion:

    * **Atomic, not snapshot.** Spend is applied under a lock to ONE counter.
      Were each child instead handed a snapshot of the remainder at spawn,
      N concurrent children could each pass their own check and collectively
      overshoot — min-wins would silently stop holding under fan-out.
    * **Two mechanisms, not two alternatives.** :meth:`effective_limits_for`
      still clamps each child's own ceiling at spawn (``min(profile,
      remaining)``) — that is a genuine per-child guarantee and the operator-
      visible one.  The pool is the aggregate on top.

    Granularity: spend is reported at TURN boundaries (the daemon already
    receives per-turn usage), so a fan-out can overshoot by at most one
    in-flight turn per concurrent child between the pool crossing and the
    degrade reaching them.  Bounded and explainable — not a silent
    unbounded overshoot.  Tightening it would need per-response reporting
    from runner to daemon, a heavier channel deliberately not built.
    """

    def __init__(self, cascade_driver_id: str, config: BudgetControlConfig) -> None:
        self.cascade_driver_id = cascade_driver_id
        self._tracker = BudgetTracker(config)
        self._lock = __import__("threading").Lock()
        # session_id -> its absolute contribution per dimension.  Lets
        # reconcile_session apply deltas idempotently from repeated
        # absolute readings, instead of trusting a stream of increments.
        self._per_session: Dict[str, Dict[str, float]] = {}

    @property
    def config(self) -> BudgetControlConfig:
        return self._tracker.config

    def spend(self, **deltas: Optional[float]) -> Tuple["DegradeRung", ...]:
        """Apply spend atomically; return rungs the POOL just crossed."""
        with self._lock:
            return self._tracker.observe(**deltas)

    def remaining(self) -> Dict[str, float]:
        """Per-dimension headroom left in the pool (never negative)."""
        with self._lock:
            usage = self._tracker.usage.as_dict()
            return {
                dim: max(0.0, limit - usage[dim])
                for dim, limit in self._tracker.config.limits.items()
            }

    def usage_fraction(self) -> float:
        with self._lock:
            return self._tracker.usage_fraction()

    def describe_pressure(self) -> str:
        with self._lock:
            return self._tracker.describe_pressure()

    def reconcile_session(
        self, session_id: str, absolute: Mapping[str, float],
    ) -> "ReconcileResult":
        """Set a session's TOTAL contribution to the pool, applying the delta.

        Idempotent and monotonic: call it repeatedly with the session's
        running absolute usage and the pool only ever moves by the
        difference.  Never decreases a contribution (a lower reading is
        treated as stale and ignored) so a late or out-of-order report
        cannot refund spend.

        This exists because accumulating the pool from the EVENT STREAM
        proved unsound twice: turn.progress re-emits (inflating), and a
        cancelled turn's TurnCompletedEvent can go missing (leaking).  The
        invariant that actually matters is "the pool must see every token
        the per-session TRACKER saw" — so the pool is reconciled against the
        tracker's own absolute totals rather than derived from a stream of
        increments that may be duplicated or dropped.

        Returns a :class:`ReconcileResult` — the deltas actually applied
        (for logging) AND any pool rungs those deltas crossed.  The rungs
        matter: reconciliation is the pool's real accumulation path, so
        discarding what ``observe`` fired here would mean cascade-level
        degrade ladders never fire at all.
        """
        with self._lock:
            prior = self._per_session.setdefault(session_id, {})
            deltas: Dict[str, float] = {}
            for dim, total in (absolute or {}).items():
                if dim not in VALID_DIMENSIONS:
                    continue
                try:
                    total_f = float(total)
                except (TypeError, ValueError):
                    continue
                seen = prior.get(dim, 0.0)
                if total_f <= seen:
                    continue
                deltas[dim] = total_f - seen
                prior[dim] = total_f
            fired: Tuple["DegradeRung", ...] = ()
            if deltas:
                fired = self._tracker.observe(**deltas)
            return ReconcileResult(deltas=deltas, fired=fired)

    def session_contribution(self, session_id: str) -> Dict[str, float]:
        """What this session has contributed to the pool so far."""
        with self._lock:
            return dict(self._per_session.get(session_id, {}))

    def effective_limits_for(
        self, profile_limits: Optional[Mapping[str, float]]
    ) -> EffectiveLimits:
        """Clamp a child's declared ceiling against what the cascade has left.

        ``effective[dim] = min(profile[dim], remaining[dim])`` where both
        declare the dimension; otherwise whichever declares it wins (an
        undeclared dimension is unbounded, so the declared one is strictly
        tighter).  A child can therefore never widen its parent's ceiling,
        and never exceed what the cascade still has.
        """
        remaining = self.remaining()
        profile = dict(profile_limits or {})
        effective: Dict[str, float] = {}
        clamped = []
        for dim in set(profile) | set(remaining):
            p = profile.get(dim)
            r = remaining.get(dim)
            if p is not None and r is not None:
                effective[dim] = min(p, r)
                if r < p:
                    clamped.append(dim)
            elif p is not None:
                effective[dim] = p
            else:
                effective[dim] = r
                clamped.append(dim)
        return EffectiveLimits(
            effective=effective, profile_limits=profile,
            cascade_remaining=remaining, clamped=tuple(sorted(clamped)),
        )

    def child_config(
        self, profile_budget: Optional[BudgetControlConfig]
    ) -> Tuple[Optional[BudgetControlConfig], EffectiveLimits]:
        """Decide the budget a child spawned into this scope runs under.

        **A child that declared its own budget keeps it, untouched.**  A
        subagent is a delegation to another department with its own budget:
        the author wrote a number, and the parent is not entitled to rewrite
        it down to whatever happens to be left in the shared pot.  Its
        spending is accounted on its own books, so it is neither clamped at
        spawn nor refused when the shared pot is dry.

        **A child that declared nothing draws on the parent's budget** —
        both the limits (whatever remains) and the degradation ladder.
        Choosing to spawn without a declared budget, whether by omitting a
        profile or by an inline spec that does not mention one, IS the
        author delegating that policy to the parent.

        The test is therefore "did this spawn carry a ``budget_control``",
        from a profile file or an inline spec alike — not "was a profile
        referenced".  The spawn tool accepts both, so the mechanism the
        author chose is itself the expression of intent.

        ``EffectiveLimits`` is still returned in both cases, now purely as
        OBSERVABILITY: it shows what the child asked for beside what the
        scope had, so an operator can see the relationship even when
        nothing was clamped.
        """
        eff = self.effective_limits_for(
            profile_budget.limits if profile_budget else None)
        if profile_budget is not None:
            # Its own books.  Not clamped, not refused.
            return profile_budget, eff
        if not eff.effective:
            return None, eff
        if eff.exhausted:
            raise CascadeExhaustedError(self.cascade_driver_id, eff)
        return BudgetControlConfig(
            limits=eff.effective, degrade=self._tracker.config.degrade
        ), eff
