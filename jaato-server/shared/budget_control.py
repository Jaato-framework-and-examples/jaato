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
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

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

        overlay: Dict[str, TierEntry] = {}
        raw_overlay = raw.get("model_tiers") or {}
        if raw_overlay:
            if not isinstance(raw_overlay, Mapping):
                raise BudgetControlConfigError(
                    f"degrade[{index}].model_tiers: expected an object, "
                    f"got {type(raw_overlay).__name__}"
                )
            for tier_name, entry in raw_overlay.items():
                key = str(tier_name)
                # An overlay rebinds EXISTING tiers; the control keys
                # (initial / fallback) select a tier rather than bind a
                # model, so they are meaningless in an overlay.
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
                try:
                    overlay[key] = _normalize_tier_entry(key, entry)
                except Exception as exc:
                    raise BudgetControlConfigError(
                        f"degrade[{index}].model_tiers.{key}: {exc}"
                    ) from None

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
