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

**V1 constraint**: all tiers must use the same provider.  The schema
already supports per-tier ``provider`` overrides (forward-compat for
V2's cross-provider tiers) — but at construction time the config
rejects any mix.  When V2 lifts this, drop the
``_validate_same_provider_v1`` call and add cross-provider provider
swap logic at the session layer.

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

    The provider field is forward-compat for V2 (cross-provider tiers).
    In V1 the same-provider check in
    :meth:`ModelTierConfig._validate_same_provider_v1` rejects configs
    where tiers declare different providers; when the constraint lifts,
    drop that call and let the session layer handle provider swaps.

    Attributes:
        model: Model name (e.g. ``"claude-opus-4-7"``).  Required.
        provider: Provider plugin name (e.g. ``"anthropic"``).  When
            ``None``, the session's main provider is used.  V1: if any
            tier sets this, all tiers that set it must agree (and
            usually you'd leave it ``None`` everywhere — the session's
            provider then handles the model switch via
            ``provider.connect(new_model_name, skip_model_test=True)``).
    """
    model: str
    provider: Optional[str] = None


def _normalize_tier_entry(name: str, raw: object) -> TierEntry:
    """Coerce a raw tier value into a :class:`TierEntry`.

    Accepts:
        * ``str`` — shorthand for ``{"model": <str>}``
        * ``dict`` with ``model`` (required) and optional ``provider``

    Raises:
        ModelTierConfigError: Invalid shape or empty model.
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
        return TierEntry(
            model=model.strip(),
            provider=provider.strip() if provider else None,
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
    ``get_system_instructions`` (to append the per-turn tier-identity
    line).

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
        # V2 (cross-provider tiers): the V1 same-provider gate is lifted.  A
        # tier may declare its own ``provider``; JaatoSession.switch_tier swaps
        # to a cached per-tier provider instance when the active tier's provider
        # differs (history is provider-neutral, so it flows across the swap).
        # Same-provider configs are unaffected (no swap path is taken).

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

        Env vars only support the simple shorthand (model name only) —
        per-tier provider overrides have to come from a profile.  This
        is fine for V1 where same-provider is the only mode; V2 may
        extend with paired ``JAATO_TIER_<NAME>_PROVIDER`` env vars.

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
