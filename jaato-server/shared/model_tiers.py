"""Model-tier configuration for per-turn model switching.

Named tiers the agent transitions between via the ``enter_tier(name)``
lifecycle tool: three *cognitive* tiers (``planner`` / ``dispatcher`` /
``executor``) plus the optional *modality* role ``vision`` (a tier whose
model accepts image input — multimodal-by-composition, see
docs/design/multimodal-model-support.md).  The tier set is opt-in:
profiles that don't declare tiers (and aren't backed by tier env vars)
run in single-model mode unchanged — no ``enter_tier`` tool registered,
no system-prompt augmentation, no provider model switching.

Resolution order:

1. Profile declares a non-empty ``tiers`` map → that profile's full
   tier config wins (profile.model is silently ignored if both are set;
   a warning is logged at profile-load time).
2. Profile lacks ``tiers`` (or no profile at all) → check env vars
   ``JAATO_TIER_PLANNER``, ``JAATO_TIER_DISPATCHER``,
   ``JAATO_TIER_EXECUTOR``, ``JAATO_TIER_INITIAL``,
   ``JAATO_TIER_FALLBACK``.  If at least one tier model env var is set,
   build the config from env.
3. Neither set → ``None`` returned, single-model mode.

**Cross-provider tiers**: a tier may declare its own ``provider``, and
tiers are free to disagree — the historical same-provider gate
(``_validate_same_provider_v1``) is gone.  When the tier being entered
names a provider other than the active one,
``JaatoSession.switch_tier`` swaps to a per-tier provider instance
cached by ``_provider_for_tier``; conversation history is
provider-neutral, so it flows across the swap untouched.  A tier that
leaves ``provider`` unset uses the session's main provider, which
switches model in place via
``provider.connect(model, skip_model_test=True)`` — no swap path is
taken, so same-provider configs behave exactly as before.

**Schema** — single-level dict mixing tier→model mappings (keys in
:data:`VALID_TIER_NAMES`) and reserved control keys (``initial`` and
``fallback``).  Each tier value can be either the simple shorthand
(model name string) or the rich form (``{"model": ..., "provider": ...}``).

.. code-block:: json

    "model_tiers": {
      "planner": "claude-opus-4-7",
      "dispatcher": {"model": "claude-sonnet-4-6", "provider": "anthropic"},
      "executor": "claude-haiku-4-5",
      "initial": "dispatcher",
      "fallback": "dispatcher"
    }

The reserved keys (``initial`` / ``fallback``) are unambiguous because
they're never valid tier names — the parser splits on
:data:`VALID_TIER_NAMES` membership.

See ``project_backlog_per_turn_model.md`` for the full design.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)


# Valid tier names.  The first three (planner / dispatcher / executor)
# are *cognitive* tiers — order is conceptual (cheapest → most capable)
# but doesn't enforce ordering on the model assignments; operators are
# free to wire them however the provider's pricing makes sense.
#
# ``vision`` is a *modality* role (multimodal-by-composition — see
# docs/design/multimodal-model-support.md): a tier whose model accepts
# image input, switched into via ``enter_tier("vision")`` to view an
# image and back out when done.  It shares the single-``_active_tier``
# machinery with the cognitive tiers (mutually exclusive), so it's listed
# here rather than as a separate axis.  All names are exposed verbatim in
# the ``enter_tier`` tool's schema so the model sees them as the protocol
# vocabulary; whether a given tier is *declared* in a profile is separate
# (an undeclared tier routes to ``tier_fallback`` via ``model_for``).
TIER_PLANNER = "planner"
TIER_DISPATCHER = "dispatcher"
TIER_EXECUTOR = "executor"
TIER_VISION = "vision"
VALID_TIER_NAMES: FrozenSet[str] = frozenset(
    {TIER_PLANNER, TIER_DISPATCHER, TIER_EXECUTOR, TIER_VISION}
)

# Canonical presentation order for the ``enter_tier`` tool schema.  A set
# has no order and ``sorted()`` would put ``dispatcher`` before ``planner``
# — neither is wrong, but the order must be DETERMINISTIC: the tool block
# sits in the prompt-cache prefix, so a tier list that reordered between
# processes would invalidate the cache for no reason.  Names outside this
# tuple sort alphabetically after it.
TIER_ORDER: Tuple[str, ...] = (
    TIER_PLANNER, TIER_DISPATCHER, TIER_EXECUTOR, TIER_VISION,
)

# Framework-supplied prose for each known tier, used by the ``enter_tier``
# tool schema when a profile's tier entry declares no ``description``.
# A profile that DOES declare one replaces the corresponding line — that is
# the whole point of the key: the framework cannot know what a given
# deployment means by "planner", only what the name suggests.
DEFAULT_TIER_DESCRIPTIONS: Dict[str, str] = {
    TIER_PLANNER: (
        "deep thought, multi-step reasoning, complex problem "
        "decomposition.  Most expensive; use when you genuinely need the "
        "strongest model."
    ),
    TIER_DISPATCHER: (
        "coordination, light reasoning, deciding which tools to call."
    ),
    TIER_EXECUTOR: (
        "mechanical tool calls and result interpretation when the plan is "
        "clear.  Cheapest; use when the work doesn't need reasoning."
    ),
    TIER_VISION: (
        "view image content (diagrams, screenshots).  Switch here BEFORE "
        "reading an image with a tool (e.g. viewing a file that is an "
        "image), then switch back when done.  If you try to read an image "
        "while in a non-vision tier, the image is withheld and the tool "
        "result tells you to switch here first."
    ),
}

# Framework defaults when neither profile nor env vars specify them.
DEFAULT_INITIAL_TIER = TIER_DISPATCHER
DEFAULT_TIER_FALLBACK = TIER_DISPATCHER

# Reserved keys inside the unified ``model_tiers`` dict.  These are
# control knobs (which tier to start in / which to use as fallback);
# every other key in the dict must be a member of VALID_TIER_NAMES.
RESERVED_INITIAL_KEY = "initial"
RESERVED_FALLBACK_KEY = "fallback"
RESERVED_KEYS: FrozenSet[str] = frozenset(
    {RESERVED_INITIAL_KEY, RESERVED_FALLBACK_KEY}
)

# Env var keys consulted when no profile-level config is present.
# (Naming kept as JAATO_TIER_* even though the JSON field renamed to
# model_tiers — the env-var prefix is already in a tier-specific
# namespace and the verbosity tradeoff isn't worth it for one-off
# operator experimentation.)
ENV_TIER_MODEL_KEYS: Dict[str, str] = {
    TIER_PLANNER: "JAATO_TIER_PLANNER",
    TIER_DISPATCHER: "JAATO_TIER_DISPATCHER",
    TIER_EXECUTOR: "JAATO_TIER_EXECUTOR",
}
ENV_TIER_INITIAL = "JAATO_TIER_INITIAL"
ENV_TIER_FALLBACK = "JAATO_TIER_FALLBACK"


class ModelTierConfigError(ValueError):
    """Raised when tier config can't be parsed or is internally inconsistent."""


@dataclass(frozen=True)
class TierEntry:
    """One tier's model + optional provider.

    Tiers may name different providers; the session layer handles the
    swap (see the module docstring's *Cross-provider tiers* note).

    Attributes:
        model: Model name (e.g. ``"claude-opus-4-7"``).  Required.
        description: What this tier is FOR, in the second person, as the
            model reads it.  Rendered verbatim as the tier's bullet in the
            ``enter_tier`` tool description, replacing the framework's
            default prose for that tier name
            (:data:`DEFAULT_TIER_DESCRIPTIONS`).  ``None`` keeps the
            default.  This is the only channel by which a profile tells
            the model what its own tier ladder means — the framework knows
            the names, not the deployment's intent behind them.

            It lands in the tool block, which is part of the prompt-cache
            prefix, so it must be stable for the life of a session: it is
            read once when the tool schema is built.  A budget-control
            degrade rung therefore cannot set it (see
            :meth:`shared.budget_control.DegradeRung.from_dict`) — a
            brownout rebinds a tier's model, never its role.
        provider: Provider plugin name (e.g. ``"anthropic"``).  When
            ``None``, the session's main provider is used and entering
            the tier just re-points it via
            ``provider.connect(new_model_name, skip_model_test=True)``.
            When set to something other than the active provider,
            entering the tier swaps to a cached per-tier provider
            instance instead.  Tiers need not agree — leaving this
            ``None`` everywhere keeps the whole session on one provider.
    """
    model: str
    provider: Optional[str] = None
    description: Optional[str] = None


def _normalize_tier_entry(name: str, raw: object) -> TierEntry:
    """Coerce a raw tier value into a :class:`TierEntry`.

    Accepts:
        * ``str`` — shorthand for ``{"model": <str>}``
        * ``dict`` with ``model`` (required) and optional ``provider`` /
          ``description``

    Raises:
        ModelTierConfigError: Invalid shape, empty model, or a
            ``provider`` / ``description`` present but not a non-empty
            string.
    """
    if isinstance(raw, str):
        if not raw.strip():
            raise ModelTierConfigError(f"tier {name!r} has empty model string")
        return TierEntry(model=raw.strip())
    if isinstance(raw, dict):
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ModelTierConfigError(
                f"tier {name!r}: 'model' must be a non-empty string"
            )
        provider = raw.get("provider")
        if provider is not None and (
            not isinstance(provider, str) or not provider.strip()
        ):
            raise ModelTierConfigError(
                f"tier {name!r}: 'provider' must be a non-empty string when set"
            )
        description = raw.get("description")
        if description is not None and (
            not isinstance(description, str) or not description.strip()
        ):
            raise ModelTierConfigError(
                f"tier {name!r}: 'description' must be a non-empty string "
                f"when set"
            )
        return TierEntry(
            model=model.strip(),
            provider=provider.strip() if provider else None,
            description=description.strip() if description else None,
        )
    raise ModelTierConfigError(
        f"tier {name!r}: expected str or dict, got {type(raw).__name__}"
    )


@dataclass(frozen=True)
class ModelTierConfig:
    """Resolved tier config for one session.

    Built either from a ``SubagentProfile`` (with non-empty ``tiers``)
    or from env vars.  Held on the session as ``_tier_config`` and
    consulted by ``LifecycleTools`` (to decide whether to register
    ``enter_tier``), ``JaatoSession`` (to compute the initial model
    name and to switch the provider on tier transitions), and
    ``get_system_instructions`` (to append the tier-mode line naming
    :attr:`initial_tier`).  That line is deliberately *stable* — it
    reports where the session started, never which tier is active now,
    because the system block must stay byte-identical across tier
    switches or every switch invalidates the prompt cache.  See
    ``docs/design/model-tier-prompt-cache.md`` §5.1.

    Attributes:
        tiers: Map of tier name → :class:`TierEntry`.  Must be
            non-empty.  Tier names must be a subset of
            :data:`VALID_TIER_NAMES`.
        initial_tier: Tier name to use at session start.  Must be a
            key in ``tiers``.
        tier_fallback: Tier name to route to when ``enter_tier(name)``
            references a tier that isn't in ``tiers``.  Must be a key
            in ``tiers``.
    """
    tiers: Dict[str, TierEntry] = field(default_factory=dict)
    initial_tier: str = DEFAULT_INITIAL_TIER
    tier_fallback: str = DEFAULT_TIER_FALLBACK

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ModelTierConfigError(
                "ModelTierConfig requires at least one tier mapping"
            )
        unknown = set(self.tiers) - VALID_TIER_NAMES
        if unknown:
            raise ModelTierConfigError(
                f"unknown tier names: {sorted(unknown)} "
                f"(must be subset of {sorted(VALID_TIER_NAMES)})"
            )
        for name, entry in self.tiers.items():
            if not isinstance(entry, TierEntry):
                raise ModelTierConfigError(
                    f"tier {name!r}: expected TierEntry, got {type(entry).__name__}"
                )
        if self.initial_tier not in self.tiers:
            raise ModelTierConfigError(
                f"initial_tier {self.initial_tier!r} not in declared "
                f"tiers {sorted(self.tiers)}"
            )
        if self.tier_fallback not in self.tiers:
            raise ModelTierConfigError(
                f"tier_fallback {self.tier_fallback!r} not in declared "
                f"tiers {sorted(self.tiers)}"
            )
        # No same-provider gate: tiers may name different providers.
        # JaatoSession.switch_tier swaps to a cached per-tier provider instance
        # when the entered tier's provider differs from the active one (history
        # is provider-neutral, so it flows across the swap).  Same-provider
        # configs are unaffected — no swap path is taken.

    def ordered_tier_names(self) -> Tuple[str, ...]:
        """Declared tier names in canonical (cache-stable) order.

        Known names come first in :data:`TIER_ORDER`; anything else sorts
        alphabetically after them.  Deterministic because the result feeds
        the ``enter_tier`` tool schema, which sits in the prompt-cache
        prefix — an order that varied between processes would invalidate
        the cache without changing meaning.
        """
        known = [n for n in TIER_ORDER if n in self.tiers]
        rest = sorted(n for n in self.tiers if n not in TIER_ORDER)
        return tuple(known + rest)

    def describe_tier(self, tier_name: str) -> str:
        """Prose for one tier, as the model should read it.

        Resolution order: the tier entry's own ``description`` (set in the
        profile), then the framework default for that name
        (:data:`DEFAULT_TIER_DESCRIPTIONS`), then a bare fallback naming
        the model — which is all the framework can honestly say about a
        tier nobody described.

        Returns:
            A single-sentence-ish fragment with no leading bullet or tier
            name; the caller formats it into the tool description.
        """
        entry = self.tiers.get(tier_name)
        if entry is not None and entry.description:
            return entry.description
        default = DEFAULT_TIER_DESCRIPTIONS.get(tier_name)
        if default:
            return default
        model = entry.model if entry is not None else "an unspecified model"
        return f"routes this session to {model}."

    def model_for(self, tier_name: str) -> Tuple[str, TierEntry]:
        """Resolve a tier name to ``(actual_tier, entry)``.

        When ``tier_name`` isn't declared in :attr:`tiers`, routes to
        :attr:`tier_fallback` and returns the fallback's
        ``(name, entry)``.  Caller can compare the returned tier name
        against the requested one to detect that fallback fired (and
        surface that to the model in the tool result).

        Raises:
            ModelTierConfigError: If ``tier_name`` is not even a valid
                tier identifier (callers should validate before getting
                here; this is a defence-in-depth guard).
        """
        if tier_name not in VALID_TIER_NAMES:
            raise ModelTierConfigError(
                f"unknown tier {tier_name!r}; valid: {sorted(VALID_TIER_NAMES)}"
            )
        if tier_name in self.tiers:
            return tier_name, self.tiers[tier_name]
        return self.tier_fallback, self.tiers[self.tier_fallback]

    @classmethod
    def from_unified_dict(cls, raw: Dict[str, Any]) -> "ModelTierConfig":
        """Build from the unified JSON-profile dict shape.

        Splits a single dict mixing tier→model mappings (keys in
        :data:`VALID_TIER_NAMES`) and reserved control keys
        (``initial`` / ``fallback``).  This is the public ingress
        point for parsed profile JSON — it does the split, normalises
        each tier value via :func:`_normalize_tier_entry`, and hands
        off to ``__post_init__`` for the rest of validation.

        Args:
            raw: Raw dict as it appears under ``profile.model_tiers``.

        Returns:
            Validated :class:`ModelTierConfig`.

        Raises:
            ModelTierConfigError: If a reserved key isn't a string, if
                a tier name is invalid, or any other validation
                failure.
        """
        tiers: Dict[str, TierEntry] = {}
        initial: Optional[str] = None
        fallback: Optional[str] = None
        for key, value in raw.items():
            if key == RESERVED_INITIAL_KEY:
                if not isinstance(value, str):
                    raise ModelTierConfigError(
                        f"model_tiers.{RESERVED_INITIAL_KEY!r} must be a string"
                    )
                initial = value
            elif key == RESERVED_FALLBACK_KEY:
                if not isinstance(value, str):
                    raise ModelTierConfigError(
                        f"model_tiers.{RESERVED_FALLBACK_KEY!r} must be a string"
                    )
                fallback = value
            else:
                tiers[key] = _normalize_tier_entry(key, value)
        return cls(
            tiers=tiers,
            initial_tier=initial or DEFAULT_INITIAL_TIER,
            tier_fallback=fallback or DEFAULT_TIER_FALLBACK,
        )

    @classmethod
    def from_env(
        cls, env: Optional[Dict[str, str]] = None
    ) -> Optional["ModelTierConfig"]:
        """Build from env vars, or return ``None`` if no tier vars set.

        Env vars only support the simple shorthand (model name only), so
        an env-built config is always single-provider — a cross-provider
        tier set has to come from a profile's ``model_tiers``.  There is
        also no ``JAATO_TIER_VISION``: the env path covers the three
        cognitive tiers only (see :data:`ENV_TIER_MODEL_KEYS`).

        Args:
            env: Optional override for ``os.environ`` (test injection).

        Returns:
            A fully-validated :class:`ModelTierConfig`, or ``None`` if
            no ``JAATO_TIER_*`` model var is present (caller falls back
            to single-model mode).

        Raises:
            ModelTierConfigError: If at least one tier model env var is
                set but the resulting config is invalid.
        """
        source = env if env is not None else os.environ
        unified: Dict[str, Any] = {}
        for tier, key in ENV_TIER_MODEL_KEYS.items():
            value = source.get(key)
            if value and value.strip():
                unified[tier] = value.strip()
        if not unified:
            return None
        initial_env = source.get(ENV_TIER_INITIAL)
        if initial_env:
            unified[RESERVED_INITIAL_KEY] = initial_env
        fallback_env = source.get(ENV_TIER_FALLBACK)
        if fallback_env:
            unified[RESERVED_FALLBACK_KEY] = fallback_env
        return cls.from_unified_dict(unified)

    @classmethod
    def resolve(
        cls,
        profile_model_tiers: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional["ModelTierConfig"]:
        """Resolve tier config in priority order: profile → env → None.

        Use this from session-init code paths.  Pass the
        ``SubagentProfile.model_tiers`` dict (or ``None`` for the
        no-profile / no-tiers case).  Returns ``None`` to indicate
        single-model mode.
        """
        if profile_model_tiers:
            return cls.from_unified_dict(profile_model_tiers)
        return cls.from_env(env=env)


def bound_model_for_profile(profile: object) -> Optional[str]:
    """The model a profile binds for session START, by EITHER route.

    A profile binds a model with a flat ``model``, or with ``model_tiers``
    whose initial tier declares one -- and the tiers route is authoritative
    at runtime: ``JaatoSession`` assigns
    ``tier_config.tiers[initial_tier].model`` over whatever ``model`` held.

    ONE definition, deliberately, because two of them disagreed:

      * ``core.py``'s bootstrap gate asked "is a model bound?" and, before
        PR #574, consulted ``model`` alone -- so a tiers-only profile was
        rejected outright.
      * ``runner_spawn`` builds ``envelope.model_name`` and ALSO consulted
        ``model`` alone.  After #574 opened the gate, that disagreement moved
        the failure one layer down: the gate passed, the runner rejected the
        envelope with "envelope.model_name is empty", and the caller saw a
        dropped IPC connection and "session not bootstrapped on this runner"
        instead of a configuration error.  Worse than before the gate opened.

    Both callers now ask this function, so "bound" cannot mean two things.

    Returns:
        The bound model name, or ``None`` when the profile binds none by
        either route.  A malformed tier entry yields ``None`` rather than
        raising: callers use this as a precondition, and
        ``ModelTierConfig.resolve`` reports shape errors precisely later.
    """
    if profile is None:
        return None

    flat = getattr(profile, "model", None)
    if flat:
        return flat

    tiers = getattr(profile, "model_tiers", None) or {}
    if not tiers:
        return None

    initial = tiers.get(RESERVED_INITIAL_KEY) or DEFAULT_INITIAL_TIER
    if initial not in tiers:
        return None
    try:
        # Delegate the entry grammar -- mapping OR the documented bare-string
        # shorthand -- instead of re-reading it.  Re-reading it is what made
        # the shorthand the one shape that could not boot.
        return _normalize_tier_entry(initial, tiers[initial]).model or None
    except ModelTierConfigError:
        return None
