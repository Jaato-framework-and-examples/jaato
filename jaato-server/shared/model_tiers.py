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
from dataclasses import dataclass, field, replace as _dc_replace
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

# Non-text input modalities a tier may declare a ROLE for.  Deliberately
# duplicated from ``MODALITY_*`` in
# ``shared/plugins/model_provider/base.py`` rather than imported: this
# module costs ~14ms to import and sits on the profile-load path (it is
# pulled in by ``budget_control`` and by every session bootstrap), while
# the provider base costs ~196ms.  ``test_tier_modalities.py`` pins the two
# sets equal, so the duplication cannot drift.
#
# ``text`` is deliberately ABSENT.  Every text-completion model accepts
# text, so a tier declaring it would assert nothing; the parser rejects it
# with that explanation rather than accepting a no-op.
MODALITY_IMAGE = "image"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"
MODALITY_FILE = "file"  # PDFs / documents (OpenRouter's term)
VALID_TIER_MODALITIES: FrozenSet[str] = frozenset(
    {MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO, MODALITY_FILE}
)

# Directions a modality role can be declared in.
#
# ``bidirectional`` rather than ``both``: "both" says nothing about what it is
# both OF, and does not parallel ``inbound``/``outbound`` grammatically.  Not
# ``duplex`` either — that connotes SIMULTANEITY, and a tier declares
# capability, not concurrency (a half-duplex voice loop is still a tier that
# does audio in and out).  None of these three are YAML 1.1 booleans, unlike
# ``on``/``off``/``yes``/``no``, so a profile can write them unquoted.
DIRECTION_INBOUND = "inbound"
DIRECTION_OUTBOUND = "outbound"
DIRECTION_BIDIRECTIONAL = "bidirectional"
VALID_MODALITY_DIRECTIONS: FrozenSet[str] = frozenset(
    {DIRECTION_INBOUND, DIRECTION_OUTBOUND, DIRECTION_BIDIRECTIONAL}
)

# Modality roles implied by a tier's NAME when it declares none itself.
#
# This is what keeps every profile written before the ``modalities`` key
# working unchanged: a tier called ``vision`` has always meant "the tier to
# enter to look at an image", and the content gate and the startup
# capability check both branched on that literal name.  They now read the
# modality role instead, so the name has to keep implying the role.
#
# It is also the ONLY place a tier name carries built-in meaning.  Anything
# else — including a differently-named image tier — must say so with the
# key.
IMPLICIT_TIER_MODALITIES: Dict[str, Dict[str, FrozenSet[str]]] = {
    TIER_VISION: {DIRECTION_INBOUND: frozenset({MODALITY_IMAGE})},
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
        inbound_modalities: Non-text modalities this tier can ACCEPT as
            input (a subset of :data:`VALID_TIER_MODALITIES`).  Declaring
            ``{"image"}`` means "this is the tier to enter to look at an
            image": the content gate names it when it withholds an image
            from a model that can't see one, and the startup capability
            check verifies the tier's model really does accept that input.

            This DECLARES a role and is VERIFIED — the opposite direction
            from ``plugin_configs.<provider>.modalities``, which ASSERTS
            what a model supports in order to correct catalog detection.
            Declaring a role the model can't fill is a config error, and
            the whole point of the check.

            Empty by default, except for a tier named ``vision``, which
            implies ``{"image"}`` (:data:`IMPLICIT_TIER_MODALITIES`) so
            profiles written before this key behave unchanged.
        outbound_modalities: Non-text modalities this tier can EMIT.
            Parsed and validated in full, but **the framework cannot yet
            deliver model-generated media**: no adapter parses response
            media and the streaming callback is text-only.  So an outbound
            role is honest declaration ahead of delivery —
            ``jaato-scaffold validate`` warns that it is inert, and the
            startup check verifies it only against a provider that
            implements ``supports_output_modality`` (none do today, so it
            is skipped rather than failing falsely).  See
            ``docs/design/binary-media-chunks.md``.

            Two sets rather than one ``{kind: direction}`` map because
            every consumer asks a DIRECTIONAL question ("which tier
            accepts an image?", "which tier can emit audio?"); a map would
            make each of them filter.
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
    inbound_modalities: FrozenSet[str] = frozenset()
    outbound_modalities: FrozenSet[str] = frozenset()

    def modalities_for(self, direction: str) -> FrozenSet[str]:
        """The roles this tier declares in ``direction``.

        Args:
            direction: :data:`DIRECTION_INBOUND` or
                :data:`DIRECTION_OUTBOUND`.  ``bidirectional`` is a
                DECLARATION spelling, not a query one — a bidirectional
                role lands in both sets at parse time, so asking for it
                here would be ambiguous and raises.

        Raises:
            ValueError: ``direction`` is not inbound or outbound.
        """
        if direction == DIRECTION_INBOUND:
            return self.inbound_modalities
        if direction == DIRECTION_OUTBOUND:
            return self.outbound_modalities
        raise ValueError(
            f"direction must be {DIRECTION_INBOUND!r} or "
            f"{DIRECTION_OUTBOUND!r}, got {direction!r}"
        )

    @property
    def declares_any_modality(self) -> bool:
        """Whether this tier claims any modality role, either direction."""
        return bool(self.inbound_modalities or self.outbound_modalities)


def _normalize_modality_token(name: str, raw: object) -> str:
    """Validate one modality token (the KEY side of a role declaration).

    Raises:
        ModelTierConfigError: Not a non-empty string, ``"text"`` (which
            would assert nothing — every model accepts text), or a token
            outside :data:`VALID_TIER_MODALITIES`.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ModelTierConfigError(
            f"tier {name!r}: 'modalities' entries must be non-empty strings"
        )
    kind = raw.strip().lower()
    valid = ", ".join(sorted(VALID_TIER_MODALITIES))
    if kind == "text":
        raise ModelTierConfigError(
            f"tier {name!r}: 'modalities' may not list 'text' — every model "
            f"accepts text, so declaring it asserts nothing.  List only the "
            f"non-text roles this tier fills ({valid})"
        )
    if kind not in VALID_TIER_MODALITIES:
        raise ModelTierConfigError(
            f"tier {name!r}: '{kind}' is not a modality ({valid})"
        )
    return kind


def _normalize_direction(name: str, kind: str, raw: object) -> str:
    """Validate the DIRECTION side of a role declaration.

    Raises:
        ModelTierConfigError: Not a non-empty string, or outside
            :data:`VALID_MODALITY_DIRECTIONS`.  The message calls out
            ``both`` and ``duplex`` by name, because they are the two
            spellings an author is most likely to reach for.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ModelTierConfigError(
            f"tier {name!r}: direction for modality '{kind}' must be a "
            f"non-empty string ({', '.join(sorted(VALID_MODALITY_DIRECTIONS))})"
        )
    direction = raw.strip().lower()
    if direction in VALID_MODALITY_DIRECTIONS:
        return direction
    hint = ""
    if direction in ("both", "duplex", "inout", "in_out", "io"):
        hint = f"  (use '{DIRECTION_BIDIRECTIONAL}')"
    raise ModelTierConfigError(
        f"tier {name!r}: '{direction}' is not a modality direction for "
        f"'{kind}' ({', '.join(sorted(VALID_MODALITY_DIRECTIONS))}){hint}"
    )


def _normalize_tier_modalities(
    name: str, raw: object
) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Coerce a tier entry's ``modalities`` value into (inbound, outbound).

    Two accepted spellings:

    * **list sugar** — ``[image, file]`` means those roles INBOUND.  This
      is the form every profile written before directions existed uses, so
      it must keep meaning exactly what it meant.
    * **direction map** — ``{image: inbound, audio: bidirectional}``.
      ``bidirectional`` lands the role in BOTH returned sets, which is why
      the stored form is two sets and not the map itself.

    Either way the name's implicit role
    (:data:`IMPLICIT_TIER_MODALITIES`) is unioned in, so a tier called
    ``vision`` is an inbound image tier whatever else it declares.

    Args:
        name: Tier name — for error messages and the implicit-role lookup.
        raw: The entry's raw ``modalities`` value, or ``None``.

    Returns:
        ``(inbound, outbound)`` frozensets.

    Raises:
        ModelTierConfigError: Not a list or map, or a malformed token or
            direction within it.
    """
    implicit = IMPLICIT_TIER_MODALITIES.get(name, {})
    inbound = set(implicit.get(DIRECTION_INBOUND, frozenset()))
    outbound = set(implicit.get(DIRECTION_OUTBOUND, frozenset()))

    if raw is None:
        return frozenset(inbound), frozenset(outbound)

    if isinstance(raw, dict):
        for kind_raw, direction_raw in raw.items():
            kind = _normalize_modality_token(name, kind_raw)
            direction = _normalize_direction(name, kind, direction_raw)
            if direction in (DIRECTION_INBOUND, DIRECTION_BIDIRECTIONAL):
                inbound.add(kind)
            if direction in (DIRECTION_OUTBOUND, DIRECTION_BIDIRECTIONAL):
                outbound.add(kind)
        return frozenset(inbound), frozenset(outbound)

    # A bare string must NOT be walked as a sequence of characters.
    if isinstance(raw, str) or not isinstance(raw, (list, tuple, set, frozenset)):
        raise ModelTierConfigError(
            f"tier {name!r}: 'modalities' must be a list of modality names "
            f"({', '.join(sorted(VALID_TIER_MODALITIES))}) or a map of "
            f"name -> direction "
            f"({', '.join(sorted(VALID_MODALITY_DIRECTIONS))}), got "
            f"{type(raw).__name__}"
        )
    for token in raw:
        inbound.add(_normalize_modality_token(name, token))
    return frozenset(inbound), frozenset(outbound)


def _normalize_tier_entry(name: str, raw: object) -> TierEntry:
    """Coerce a raw tier value into a :class:`TierEntry`.

    Accepts:
        * ``str`` — shorthand for ``{"model": <str>}``
        * ``dict`` with ``model`` (required) and optional ``provider`` /
          ``description`` / ``modalities``

    Either form picks up the name's implicit modality role (a tier called
    ``vision`` is the image tier unless it says otherwise) — see
    :func:`_normalize_tier_modalities`.

    Raises:
        ModelTierConfigError: Invalid shape, empty model, a ``provider`` /
            ``description`` present but not a non-empty string, or a
            malformed ``modalities`` list.
    """
    if isinstance(raw, str):
        if not raw.strip():
            raise ModelTierConfigError(f"tier {name!r} has empty model string")
        inbound, outbound = _normalize_tier_modalities(name, None)
        return TierEntry(
            model=raw.strip(),
            inbound_modalities=inbound,
            outbound_modalities=outbound,
        )
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
        inbound, outbound = _normalize_tier_modalities(
            name, raw.get("modalities"))
        return TierEntry(
            model=model.strip(),
            provider=provider.strip() if provider else None,
            description=description.strip() if description else None,
            inbound_modalities=inbound,
            outbound_modalities=outbound,
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
        # Apply name-implied modality roles here, not only in
        # _normalize_tier_entry, so a config built DIRECTLY (tests, premium
        # code, anything not going through from_unified_dict) still gets
        # them.  Without this a hand-built TierEntry("...") under the key
        # "vision" carried no role, and the content gate + startup check —
        # which now read the role rather than the name — silently went
        # quiet.  Mutating in place is safe: the dataclass is frozen but
        # its tiers mapping is not (overlay_tier_table relies on the same).
        for _name, _entry in list(self.tiers.items()):
            _implicit = IMPLICIT_TIER_MODALITIES.get(_name)
            if not _implicit:
                continue
            _in = _implicit.get(DIRECTION_INBOUND, frozenset())
            _out = _implicit.get(DIRECTION_OUTBOUND, frozenset())
            if _in <= _entry.inbound_modalities and _out <= _entry.outbound_modalities:
                continue
            self.tiers[_name] = _dc_replace(
                _entry,
                inbound_modalities=_entry.inbound_modalities | _in,
                outbound_modalities=_entry.outbound_modalities | _out,
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

    def tiers_for_modality(
        self, kind: str, direction: str = DIRECTION_INBOUND
    ) -> Tuple[str, ...]:
        """Declared tiers that play the role for modality ``kind``.

        The replacement for every ``"vision" in tier_config.tiers`` check.
        Returns names in canonical order (:meth:`ordered_tier_names`), so a
        caller that needs exactly one — the content gate naming a tier for
        the agent to enter — can take the first deterministically.

        Args:
            kind: A modality token such as ``"image"``
                (:data:`VALID_TIER_MODALITIES`).
            direction: :data:`DIRECTION_INBOUND` (default — the content
                gate's question, "who can look at this?") or
                :data:`DIRECTION_OUTBOUND`.  Defaulted because inbound is
                the only direction with machinery behind it today, so an
                unqualified call is asking the question that can be
                answered.

        Returns:
            Matching tier names, empty when this session has no tier for
            that modality in that direction.

        Raises:
            ValueError: ``direction`` is not inbound or outbound (a
                ``bidirectional`` tier appears under BOTH, so querying for
                it would be ambiguous).
        """
        return tuple(
            name for name in self.ordered_tier_names()
            if kind in self.tiers[name].modalities_for(direction)
        )

    def describe_tier(self, tier_name: str) -> str:
        """Prose for one tier, as the model should read it.

        Resolution order for the base prose: the tier entry's own
        ``description`` (set in the profile), then the framework default for
        that name (:data:`DEFAULT_TIER_DESCRIPTIONS`), then a bare fallback
        naming the model — which is all the framework can honestly say about
        a tier nobody described.

        A tier that declares modality roles its NAME doesn't already imply
        gets a sentence appended saying so — one clause per direction, since
        "can look at an image" and "can emit audio" are different
        instructions to the model — unless the author wrote their own
        description (in which case they own the whole bullet).  Without it a
        ``planner`` tier declaring ``modalities: [image]`` read as pure
        cognitive prose, so the model had no reason to switch there for an
        image — it would only find out from the content gate after trying
        and failing.  The ``vision`` tier is unaffected: image is its
        implicit role and its default prose already covers it.

        Returns:
            A fragment with no leading bullet or tier name; the caller
            formats it into the tool description.
        """
        entry = self.tiers.get(tier_name)
        if entry is not None and entry.description:
            return entry.description
        base = DEFAULT_TIER_DESCRIPTIONS.get(tier_name)
        if base is None:
            model = entry.model if entry is not None else "an unspecified model"
            base = f"routes this session to {model}."
        if entry is None:
            return base
        implicit = IMPLICIT_TIER_MODALITIES.get(tier_name, {})
        extra_in = entry.inbound_modalities - implicit.get(
            DIRECTION_INBOUND, frozenset())
        extra_out = entry.outbound_modalities - implicit.get(
            DIRECTION_OUTBOUND, frozenset())
        clauses = []
        if extra_in:
            kinds = ", ".join(sorted(extra_in))
            # "accept ... input" rather than "view ...": the clause is
            # generated for audio and file roles too, where "view" is wrong.
            clauses.append(
                f"This tier can accept {kinds} input — switch here BEFORE "
                f"reading such content with a tool, then switch back when "
                f"done."
            )
        if extra_out:
            kinds = ", ".join(sorted(extra_out))
            clauses.append(
                f"This tier can produce {kinds} output — switch here before "
                f"work that must emit it."
            )
        if not clauses:
            return base
        return f"{base}  " + "  ".join(clauses)

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
