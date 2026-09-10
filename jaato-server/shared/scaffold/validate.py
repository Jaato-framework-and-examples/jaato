"""The validator — checks hand-authored assets against the live registry.

This is the SHARED check layer.  Both verbs use it:

- ``validate`` runs it on a hand-authored profile / profile-set.
- ``new`` runs it on the profile it just emitted (emit-then-validate), so
  scaffolded output is valid by construction — there is no separate
  "is the generated profile ok" code path.

Profile **resolution is reused from the framework**: ``discover_profiles()``
flattens the ``inherits`` chain and applies the ``JAATO_PROFILE_SET`` /
``force_profile_set`` overlay exactly as the daemon does, so the validator
checks the same *effective* profile the runtime would.  The validator only
adds the introspect-driven checks on top: unknown provider / plugin / tool /
config-knob / quirk — the silent-ignore failures (a mistyped
``api_params.temprature`` is dropped without a word at runtime) this tool
exists to surface.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.plugins.model_provider.base import KNOB_LAYERS
# The wire-type predicate lives with the contract it enforces, so this
# static check and the two runtime spawn boundaries cannot drift apart
# about what "string-shaped" means.  #883 ratified that contract; the
# loader's module notes carry the decision and the rejected alternatives.
from shared.spawn_schema_loader import unreachable_spawn_types
from jaato_sdk.plugins.model_provider.types import DISCOVERABILITY_EAGER
from . import introspect

# Layer names that nest under plugin_configs.<provider> as sub-dicts.
# ``top_level`` is not a nesting key — its knobs sit directly under the
# provider — so it is excluded from the "is this key a layer?" test.
_NESTING_LAYERS = frozenset(n for n in KNOB_LAYERS if n != "top_level")

#: The framework's deterministic test double.  It is deliberately absent from
#: the ``explain providers`` catalogue (nobody should PICK it for production),
#: but it is installed, it is what the live conformance suite runs on, and a
#: profile naming it is legitimate — so it must not be reported as an unknown
#: provider.  Doing so told every harness author that their working, zero-cost
#: echo profile was invalid.
ECHO_PROVIDER = "echo"


@dataclass
class Diagnostic:
    """One validation finding."""

    severity: str            # "error" | "warn" | "info"
    code: str                # stable machine code, e.g. "unknown_provider"
    message: str
    profile: Optional[str] = None
    where: Optional[str] = None   # dotted field path, e.g. "plugin_configs.nebius.api_params.temprature"
    tier: Optional[str] = None    # source tier of the asset: "workspace" | "user".
                                  # None = not tier-attributable (e.g. unscoped).

    def as_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity, "code": self.code,
            "message": self.message, "profile": self.profile, "where": self.where,
            "tier": self.tier,
        }


# ---------------------------------------------------------------- per-profile

def _check_plugins(names: Any, plugins: Dict[str, Any], add) -> None:
    """Check a profile's ``plugins:`` list: installed, and loadable.

    Two different findings, deliberately not collapsed into one:

    ``unknown_plugin``
        the name resolves to nothing installed.

    ``plugin_missing_tier``
        it IS installed and discoverable — and the runner will still
        drop it, because tier-filtered discovery excludes a plugin whose
        package declares no ``PLUGIN_TIER`` (issue #917).  An *error*
        rather than a warning: the session is already broken (every tool
        the profile asked for is absent) while the profile is valid by
        every other measure, so nothing else in the pipeline says a
        word.  The in-tree build gate is an AST scan of
        ``shared/plugins/`` and cannot see the out-of-tree distribution
        where this actually happens.

    They never double-fire: an uninstalled plugin has no tier to be
    missing.

    Split out of :func:`validate_profile` rather than inlined — that
    function is far over the complexity ceiling and frozen in the audit
    baseline (see ``test_cyclomatic_complexity_audit``).

    Args:
        names: The profile's resolved ``plugins`` list (bare names;
            ``(preload)`` modifiers are parsed out at profile-load time
            into ``preloaded_plugins``).
        plugins: The introspected inventory, keyed by plugin name.
        add: The per-profile diagnostic sink from :func:`validate_profile`.
    """
    for plug in names:
        if plug not in plugins:
            add("error", "unknown_plugin",
                f"plugin '{plug}' is not installed (run "
                "`jaato-scaffold explain plugins`)", where=f"plugins.{plug}")
            continue
        if getattr(plugins[plug], "tier_missing", False):
            add("error", "plugin_missing_tier",
                f"plugin '{plug}' declares no PLUGIN_TIER, so the runner "
                "will not load it — this session would come up without "
                "its tools. Add PLUGIN_TIER = \"runner\" to the plugin "
                "package's __init__.py",
                where=f"plugins.{plug}")


def validate_profile(
    profile: Any,
    *,
    providers: Dict[str, introspect.ProviderInfo],
    plugins: Dict[str, introspect.PluginInfo],
    gc_names: List[str],
) -> List[Diagnostic]:
    """Validate one RESOLVED profile against the introspected framework.

    ``profile`` is a flattened ``SubagentProfile`` (inherits already merged).
    The introspect maps are passed in so a whole workspace is introspected
    once, not per profile.
    """
    name = getattr(profile, "name", "?")
    out: List[Diagnostic] = []

    def add(sev, code, msg, where=None):
        out.append(Diagnostic(sev, code, msg, profile=name, where=where))

    provider_name = getattr(profile, "provider", None)

    # --- what the profile says about ITSELF ------------------------------
    _check_profile_identity(profile, add)

    # --- provider --------------------------------------------------------
    pinfo = _resolve_and_check_provider(profile, provider_name, providers, add)
    # model present? (a resolved, runnable profile should bind one; a pure
    # base/abstract profile legitimately has neither provider nor model)
    model = getattr(profile, "model", None)
    # A provider-set profile binds a model EITHER via a flat ``model`` OR via a
    # ``model_tiers`` map (the active model is then selected per turn from the
    # tiers).  Only warn when NEITHER is present: a tiers-based profile that
    # omits ``model`` is correct, not missing one — and a flat ``model`` set
    # alongside ``model_tiers`` is silently IGNORED at runtime, so the validator
    # must not push authors toward adding a dead one.
    if provider_name and not model and not (
        getattr(profile, "model_tiers", None) or {}
    ):
        add("warn", "missing_model",
            f"provider '{provider_name}' set but no model and no model_tiers — "
            "set-overlay or inherits did not bind a model", where="model")

    # --- plugins ---------------------------------------------------------
    _check_plugins(getattr(profile, "plugins", None) or [], plugins, add)

    # --- model_tiers (V2: cross-provider tiers allowed) ------------------
    _check_model_tiers(getattr(profile, "model_tiers", None) or {}, add,
                       getattr(profile, "provider", None))

    # --- budget_control (incl. its ABSENCE, #947) -----------------------
    _check_budget_control(profile, add)

    # --- per-plugin tool allow-lists (tool_scopes) -----------------------
    for plug, tools in (getattr(profile, "tool_scopes", None) or {}).items():
        pi = plugins.get(plug)
        if pi is None:
            continue  # unknown plugin already flagged
        known = {t.name for t in pi.tools}
        if pi.dynamic or not known:
            continue  # dynamic plugin — tool list not statically knowable
        for t in tools:
            if t not in known:
                add("warn", "unknown_tool",
                    f"tool '{t}' not exposed by plugin '{plug}' "
                    f"(has: {', '.join(sorted(known))})",
                    where=f"tool_scopes.{plug}")

    # --- discovery-gated tools (the deferred-loading nuance) -------------
    # A tool is in the model's INITIAL schema iff it is [core] OR its plugin is
    # (preload)-ed.  [disc] tools of non-preloaded plugins are reachable only
    # after the model calls list_tools/get_tool_schemas — introspection is
    # always core, so they're never LOST, just deferred.  Surface them (info,
    # not a defect) so an author who assumed a tool was immediately available
    # isn't surprised.
    preloaded = getattr(profile, "preloaded_plugins", None) or set()
    gated: List[str] = []
    for plug in getattr(profile, "plugins", None) or []:
        pi = plugins.get(plug)
        if pi is None or pi.dynamic or plug in preloaded:
            continue
        scope = (getattr(profile, "tool_scopes", None) or {}).get(plug)
        for t in pi.tools:
            if t.discoverability == DISCOVERABILITY_EAGER:
                continue
            if scope is not None and t.name not in scope:
                continue  # scoped out entirely — not exposed at all
            gated.append(f"{plug}.{t.name}")
    if gated:
        preview = ", ".join(gated[:8]) + (" …" if len(gated) > 8 else "")
        add("info", "discovery_gated_tools",
            f"{len(gated)} tool(s) are discovery-gated — DEFERRED, not in the "
            f"model's initial schema (the model must call list_tools/"
            f"get_tool_schemas to reach them): {preview}.  Add "
            f"`<plugin>(preload)` to force a plugin's tools eager.",
            where="plugins")

    # --- plugin_configs knobs (the silent-ignore class) ------------------
    plugin_configs = getattr(profile, "plugin_configs", None) or {}
    for cfg_name, cfg in plugin_configs.items():
        cfg_provider = introspect.resolve_provider(cfg_name)
        if cfg_provider is None or cfg_provider.knobs is None:
            # Non-provider plugin config (permission / cli / notebook / …):
            # validate top-level knob NAMES against the plugin's declared
            # get_config_schema (a mistyped knob is silently ignored at
            # runtime otherwise), and each declared knob's VALUE against the
            # ``enum`` / ``type`` that same schema publishes (#925) — a knob
            # violating its own declared enum used to validate clean and then
            # fall back silently.  Nested / free-form sub-structures — e.g.
            # ``permission.policy`` tree, ``permission.evaluators`` map — are
            # still NOT descended; only top-level knobs are.  A knob whose
            # STRUCTURE decides behaviour badly enough to need more than a
            # declared type registers a check in ``_PLUGIN_VALUE_CHECKS``.
            _validate_plugin_knobs(cfg_name, cfg, plugins, add)
            _check_plugin_knob_values(cfg_name, cfg, add)
            continue
        knobs = cfg_provider.knobs
        if not isinstance(cfg, dict):
            continue
        # Credential keys declared via an ``api_key_param`` AuthSource (api_key /
        # api_token / …) ARE valid top-level knobs honored at runtime (mapped to
        # ProviderConfig), even though they live in PROVIDER_AUTH_RESOLUTION
        # rather than the knob layers — so ``knobs.accepts("top_level", …)``
        # alone would miss them and falsely flag a working credential knob
        # (notably for providers with no ``top_level`` KnobLayer, e.g. zhipuai).
        auth_param_keys = {
            a.name for a in (getattr(cfg_provider, "auth", None) or ())
            if a.kind == "api_key_param" and a.name
        }
        for key, val in cfg.items():
            if key in _NESTING_LAYERS and isinstance(val, dict):
                # a layer sub-dict — check each knob inside it
                layer = knobs.get_layer(key)
                if layer is None:
                    add("warn", "unknown_layer",
                        f"provider '{cfg_name}' has no '{key}' config layer",
                        where=f"plugin_configs.{cfg_name}.{key}")
                    continue
                if layer.opaque:
                    continue  # pass-through — any key valid
                for subkey in val:
                    if not knobs.accepts(key, subkey):
                        add("error", "unknown_knob",
                            f"'{subkey}' is not a valid {cfg_name} {key} knob "
                            "(silently ignored at runtime)",
                            where=f"plugin_configs.{cfg_name}.{key}.{subkey}")
            elif key == "quirks" and isinstance(val, dict):
                _check_quirks(val, cfg_provider, cfg_name, add)
            else:
                # a top_level knob (or an api_key_param credential key)
                if key not in auth_param_keys and not knobs.accepts("top_level", key):
                    add("error", "unknown_knob",
                        f"'{key}' is not a valid {cfg_name} top-level knob "
                        "(silently ignored at runtime)",
                        where=f"plugin_configs.{cfg_name}.{key}")

    # --- profile-level quirks -------------------------------------------
    prof_quirks = getattr(profile, "quirks", None)
    if isinstance(prof_quirks, dict) and pinfo is not None:
        _check_quirks(prof_quirks, pinfo, provider_name, add,
                      where_prefix="quirks")

    # --- secret env scrub (#863) -----------------------------------------
    _check_secret_scrub(profile, add)

    # --- gc strategy -----------------------------------------------------
    gc = getattr(profile, "gc", None)
    gc_type = getattr(gc, "type", None) if gc is not None else None
    if gc_type:
        candidates = {gc_type, f"gc_{gc_type}"}
        if not (candidates & set(gc_names)):
            add("warn", "unknown_gc",
                f"gc type '{gc_type}' not among {gc_names}", where="gc.type")

    return out


def _load_spawn_schema(profile, config_root: str):
    """Resolve a profile's ``spawn_payload_schema`` to a dict, or None.

    Accepts both declared forms: an inline dict, or a path resolved against
    ``config_root`` the way ``shared/spawn_schema_loader.py`` resolves it.
    """
    raw = getattr(profile, "spawn_payload_schema", None)
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return None
    for candidate in (Path(config_root) / raw,
                      Path(config_root) / "spawn_schemas" / raw):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


def _check_profile_identity(profile: Any, add) -> None:
    """The two things a profile says about ITSELF, and both went unchecked.

    ``description`` is REQUIRED — it has no default on ``SubagentProfile``, and
    ``explain profile`` prints ``(required)`` beside it.  The YAML loader is
    lenient (a missing key becomes ``""``) and inheritance does NOT rescue it:
    the merge takes ``description=child.description``, so a tier-2 set profile
    that omits it OVERRIDES its base's with the empty string.  Nothing fails —
    it just becomes the empty half of the line the subagent plugin advertises
    to the model, ``- worker:  (tools: cli)``, which is the one piece of prose
    a model chooses a delegate from.  Every profile ``jaato-scaffold new
    profile-set`` emitted was in exactly that state.

    ``system_instructions`` is DEPRECATED in favour of a persona in
    ``.jaato/agents/<name>.md``.  ``explain profile`` said so and ``validate``
    did not, so the field kept working and nothing corrected an author who had
    never found the agents directory.

    Both WARN.  A profile with either is loadable and runnable, and an error
    would fail existing workspaces wholesale — the posture ``unknown_knob``
    and ``budget_control_absent`` already take.
    """
    if not (getattr(profile, "description", "") or "").strip():
        add("warn", "missing_description",
            "no `description` — it is a required profile field, and the "
            "subagent tool advertises it to the model verbatim as the prose a "
            "delegate is chosen from (an empty one renders as `- <name>:  "
            "(tools: ...)`).  NOTE inheritance does not supply it: a child's "
            "description REPLACES its parents', so an omitted key overrides "
            "the base's with the empty string.",
            where="description")

    if getattr(profile, "system_instructions", None):
        add("warn", "deprecated_system_instructions",
            "`system_instructions` is DEPRECATED — define the persona in "
            ".jaato/agents/<name>.md and bind it with `default_agent: <name>` "
            "(or pass agent=\"<name>\" when creating the session).  A persona "
            "is one LAYER of the prompt, survives "
            "`suppress_base_instructions`, and is reusable across profiles; "
            "this key is none of those.  See `jaato-scaffold explain agents`."
            "  (Inherited: the value may come from a parent profile — "
            "`system_instructions` concatenates down the inherits chain.)",
            where="system_instructions")


def _check_spawn_schema_wire_types(profiles, config_root: str, out) -> None:
    """Flag a ``spawn_payload_schema`` that the IPC wire can never satisfy.

    ``spawn_payload_schema`` is documented as the input-boundary mirror of
    ``completion_payload_schema`` and is validated with ``jsonschema`` against
    the ``agent_params`` dict.  But agent_params do not cross the IPC wire as
    JSON: ``create_session`` flattens them into ``key=value`` argv tokens, so
    the daemon validates a dict whose every value is a **string**.

    A property declared ``integer`` / ``number`` / ``boolean`` / ``object`` /
    ``array`` therefore fails validation on EVERY spawn, no matter what the
    caller passes.  The failure is expensive to read: the daemon logs the
    rejection and does not answer, so the caller waits out its own timeout and
    then reports ``SessionNotConfirmed`` — whose message says the session may
    have been created, which for this cause it never is.

    Measured 2026-09-08: a cascade's fix-loop stage declared
    ``iteration: {type: integer}``, passed ``iteration=1`` as an int, and the
    daemon rejected ``'1' is not of type 'integer'`` on every attempt.

    #883 ratified the wire's behaviour as the contract, which makes this a
    check on a stated rule rather than a warning about an accident.  The two
    runtime boundaries now enforce the same rule
    (``spawn_schema_loader.validate_spawn_params``) and append
    ``spawn_type_contract_note`` to the refusal, so an author who never runs
    the validator still gets told the profile is at fault.  This check remains
    the cheap half: it fires without a spawn.

    Args:
        profiles: Mapping of profile name -> resolved profile object.
        config_root: Directory that path-form schemas resolve against.
        out: Diagnostic list to append to.
    """
    for pname, profile in sorted((profiles or {}).items()):
        schema = _load_spawn_schema(profile, config_root)
        if not isinstance(schema, dict):
            continue
        for key, offending in sorted(unreachable_spawn_types(schema).items()):
            out.append(Diagnostic(
                "error", "spawn_schema_type_unreachable",
                f"spawn_payload_schema property '{key}' is typed "
                f"{'/'.join(offending)}, which no spawn can satisfy over IPC: "
                f"agent_params are sent as `key=value` argv tokens, so the "
                f"daemon always validates STRINGS.  The spawn is refused "
                f"server-side and the caller sees a 60s timeout reported as "
                f"SessionNotConfirmed.  Declare it as a string (add a "
                f"`pattern` if you need the shape, e.g. '^[0-9]+$') and parse "
                f"it in the prefetch/persona.",
                profile=pname, where=f"spawn_payload_schema.properties.{key}"))


def _resolve_and_check_provider(
    profile, provider_name, providers, add,
) -> Optional[introspect.ProviderInfo]:
    """Resolve the profile's provider, reporting what the catalogue says.

    Returns the resolved ``ProviderInfo`` — the caller needs it to check
    ``quirks`` against what that provider actually honors — or ``None``
    when the profile names no provider, or names one the catalogue does
    not carry.

    ``echo`` is the one name that resolves to nothing and is still
    correct: it is the framework's own deterministic test double, and it
    is deliberately excluded from the ``explain providers`` catalogue, so
    the plain unknown-provider error would be a false positive on every
    conformance profile in the tree.  It gets its own checks instead.

    Lives outside ``validate_profile`` because that function is over the
    complexity ceiling and frozen at its recorded size; new provider
    checks belong here.
    """
    if not provider_name:
        return None
    pinfo = introspect.resolve_provider(provider_name)
    if pinfo is not None:
        return pinfo
    if provider_name == ECHO_PROVIDER:
        _check_echo_profile(profile, add)
        return None
    add("error", "unknown_provider",
        f"provider '{provider_name}' is not a known model provider "
        f"(have: {', '.join(sorted(providers))})", where="provider")
    return None


def _check_echo_profile(profile, add) -> None:
    """Check a profile bound to the ``echo`` test double.

    Two findings, and the second is the load-bearing one.

    ``echo`` is installed and legitimate — the live conformance suite runs on
    it — but it is excluded from the ``explain providers`` catalogue, so the
    generic branch reported it as an unknown provider.  That is a false
    positive on a working profile, so it is stated as an ``info`` instead.

    The real trap is ``usage``.  Echo reports the spend it is TOLD to and none
    otherwise, and a turn is recorded in ``turn_accounting`` only when the
    provider reported tokens (``jaato_session.py``: ``if turn_data['total'] >
    0``).  The post-turn hook gated on that record
    (``server/runner/rpc.py:_forward_post_turn_hooks``) is the single site that
    fires BOTH ``TurnCompletedEvent`` AND ``flush_session_quiescent()`` ->
    ``SessionTerminatedEvent``.  So an echo profile with no ``usage`` runs its
    turn, does its work, delivers ``AgentCompletedEvent`` — and emits NO
    terminal event at all.  Every driver that waits on the documented terminus
    (``Session.complete`` / ``.ask`` / ``.stream``) then waits until its own
    timeout, with no error anywhere to say why.

    Measured 2026-09-08: an otherwise-correct cascade hung at its first stage
    on exactly this, while the same profile plus a ``usage`` block returned
    immediately.  The framework's own conformance profiles all pass
    ``usage=TURN_USAGE`` for this reason.

    Args:
        profile: The resolved profile object under validation.
        add: The diagnostic sink ``(severity, code, message, where=...)``.
    """
    add("info", "echo_is_a_test_double",
        "provider 'echo' is the framework's deterministic test double — no "
        "credentials, no network, fixed responses.  It is absent from "
        "`explain providers` (nobody should pick it for production) but it IS "
        "installed, and it is what the live conformance suite runs on.",
        where="provider")
    echo_cfg = (getattr(profile, "plugin_configs", None) or {}).get(ECHO_PROVIDER) or {}
    if not echo_cfg.get("usage"):
        add("warn", "echo_reports_no_usage",
            "echo is configured without `usage`, so it reports ZERO tokens "
            "every turn — and a turn is recorded only when the provider "
            "reported tokens.  The post-turn hook gated on that record is the "
            "one site that emits BOTH TurnCompletedEvent and "
            "SessionTerminatedEvent, so this session will do its work, deliver "
            "AgentCompletedEvent, and then emit NO terminal event: a driver "
            "awaiting the terminus (Session.complete/.ask/.stream) hangs to "
            "its timeout with nothing logged.  Declare a spend, e.g. "
            "plugin_configs.echo.usage: {prompt_tokens: 1000, output_tokens: "
            "200}.",
            where=f"plugin_configs.{ECHO_PROVIDER}.usage")


def _check_tier_exit(key, raw, add):
    """Check a tier's ``exit_on`` trigger before a session ever runs.

    An unknown value is an ERROR, not a warning, because the runtime
    refuses it too -- and because the shape of the failure it prevents is
    the worst kind: a misspelled trigger means the tier is never left,
    which surfaces as a session wedged in a specialist tier with nothing
    logged and the model simply stopping.  Catching it here turns a
    silent hang into a line of lint.

    The names considered and rejected during design (``once``,
    ``switch_back``, ``per_request``, ``turn``) are pointed at the real
    one rather than merely refused -- they are the spellings a reader
    reaches for first, and ``turn`` in particular is the plausible-
    sounding wrong answer, since a turn boundary is NOT a terminus (#767).
    """
    if raw is None:
        return
    from shared.model_tiers import EXIT_ON_COMPLETION, VALID_TIER_EXITS
    where = f"model_tiers.{key}.exit_on"
    if not isinstance(raw, str) or not raw.strip():
        add("error", "invalid_tier_exit",
            f"model_tiers.{key} 'exit_on' must be a non-empty string "
            f"({', '.join(sorted(VALID_TIER_EXITS))})", where=where)
        return
    value = raw.strip().lower()
    if value in VALID_TIER_EXITS:
        return
    hint = ""
    if value in ("once", "single", "turn", "per_request", "switch_back"):
        hint = f"  (did you mean '{EXIT_ON_COMPLETION}'?)"
    add("error", "invalid_tier_exit",
        f"model_tiers.{key} 'exit_on' {value!r} is not a known exit trigger "
        f"({', '.join(sorted(VALID_TIER_EXITS))}){hint}", where=where)


def _check_tier_modalities(key, raw, add, provider_name=None):
    """Validate one tier entry's ``modalities`` declaration statically.

    Mirrors ``shared.model_tiers._normalize_tier_modalities`` so an author
    sees the defect from ``jaato-scaffold validate`` rather than at session
    create.  Kept separate from :func:`_check_model_tiers` so neither grows
    past the complexity ceiling.

    Accepts both spellings: the list sugar (``[image]``, meaning inbound)
    and the direction map (``{image: bidirectional}``).

    Emits a **warning**, not an error, for an outbound role: it parses and
    is stored, but no adapter can deliver model-generated media yet, so the
    declaration is inert.  Warning rather than error because a profile
    should be writable ahead of the delivery work landing — see
    ``docs/design/binary-media-chunks.md``.

    Args:
        key: Tier name, for the diagnostic's ``where``.
        raw: The entry's raw ``modalities`` value, or ``None``.
        add: ``validate_profile``'s diagnostic collector.
    """
    if raw is None:
        return
    from shared.model_tiers import (
        DIRECTION_INBOUND, VALID_MODALITY_DIRECTIONS, VALID_TIER_MODALITIES,
    )
    where = f"model_tiers.{key}.modalities"
    valid = ", ".join(sorted(VALID_TIER_MODALITIES))

    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        pairs = [(tok, DIRECTION_INBOUND) for tok in raw]
    else:
        add("error", "invalid_tier_modalities",
            f"model_tiers.{key} modalities must be a LIST of modality names "
            f"({valid}) or a MAP of name -> direction "
            f"({', '.join(sorted(VALID_MODALITY_DIRECTIONS))})", where=where)
        return

    for token, direction in pairs:
        if not isinstance(token, str) or not token.strip():
            add("error", "invalid_tier_modalities",
                f"model_tiers.{key} modalities entries must be non-empty "
                "strings", where=where)
            continue
        kind = token.strip().lower()
        if kind == "text":
            add("error", "invalid_tier_modalities",
                f"model_tiers.{key} may not declare the 'text' modality — "
                "every model accepts text, so it asserts nothing; list only "
                f"the non-text roles this tier fills ({valid})", where=where)
            continue
        if kind not in VALID_TIER_MODALITIES:
            add("error", "invalid_tier_modalities",
                f"model_tiers.{key} modality '{kind}' is not a modality "
                f"({valid})", where=where)
            continue
        _check_modality_direction(key, kind, direction, where, add,
                                  provider_name)


def _delivers_output_media(provider_name) -> bool:
    """Whether this provider's adapter can deliver model-generated media.

    Reads ``ProviderCapabilities.output_media`` -- the adapter's own
    declaration that its streaming loop decodes model media and hands it
    to ``on_chunk`` as a ``MediaDelta``.  An unknown or unnamed provider
    answers False, so the caller warns rather than blessing a tier it
    cannot check.
    """
    if not provider_name:
        return False
    from .introspect import resolve_provider
    info = resolve_provider(str(provider_name))
    if info is None or info.capabilities is None:
        return False
    return bool(getattr(info.capabilities, "output_media", False))


#: INBOUND modality role -> the ``ProviderCapabilities`` field that says
#: whether the adapter puts that content on the wire.  A role with no entry
#: is unchecked rather than assumed inert: ``video`` has no capability
#: column, and warning about it would be inventing a verdict.
_INBOUND_CAPABILITY_FOR = {
    "image": "user_message_images",
    "file": "pdf_input",
    "audio": "audio_input",
}


def _carries_inbound_modality(provider_name, kind) -> bool:
    """Whether this provider's converter marshals ``kind`` onto the wire.

    The inbound mirror of :func:`_delivers_output_media`, and it exists for
    the same reason: a tier role is a declaration, and a declaration the
    adapter cannot honour is inert.  An unknown provider, or a kind with no
    capability column, answers ``True`` — the caller must not warn about a
    role it cannot actually check, because a false INERT is what made the
    outbound warning tell working profiles they were broken.
    """
    field = _INBOUND_CAPABILITY_FOR.get(kind)
    if field is None or not provider_name:
        return True
    from .introspect import resolve_provider
    info = resolve_provider(str(provider_name))
    if info is None or info.capabilities is None:
        return True
    return bool(getattr(info.capabilities, field, False))


def _warn_inert_inbound(key, kind, value, where, add, provider_name):
    """Flag an inbound role whose converter would drop the content.

    #830's shape exactly: OpenRouter's catalog reported ``audio`` input for
    an audio model, a profile could declare ``audio: inbound`` against it,
    the session-time modality check passed — and the converter had no
    branch to put the bytes on the wire, so every clip was withheld at the
    last step with nothing upstream saying why.
    """
    if _carries_inbound_modality(provider_name, kind):
        return
    add("warning", "inbound_modality_not_marshalled",
        f"model_tiers.{key} declares '{kind}' {value}, which parses but is "
        f"INERT: provider '{provider_name or '<unset>'}' does not declare "
        f"`{_INBOUND_CAPABILITY_FOR[kind]}`, so its message converter does "
        f"not put {kind} content on the wire — it is withheld with a note "
        f"instead.  See docs/design/provider-capability-contract.md.",
        where=where)


def _check_modality_direction(key, kind, direction, where, add,
                              provider_name=None):
    """Validate the direction of one modality role, and flag it if inert.

    Both directions are checked, and a bidirectional role can be inert in
    one and live in the other — which is why the outbound warning's "the
    inbound half IS live" clause is itself conditional.

    Split from :func:`_check_tier_modalities` to keep both under the
    complexity ceiling.
    """
    from shared.model_tiers import (
        DIRECTION_BIDIRECTIONAL, DIRECTION_INBOUND, DIRECTION_OUTBOUND,
        VALID_MODALITY_DIRECTIONS,
    )
    if not isinstance(direction, str) or not direction.strip():
        add("error", "invalid_tier_modalities",
            f"model_tiers.{key} direction for '{kind}' must be a string "
            f"({', '.join(sorted(VALID_MODALITY_DIRECTIONS))})", where=where)
        return
    value = direction.strip().lower()
    if value not in VALID_MODALITY_DIRECTIONS:
        hint = (f"  (use '{DIRECTION_BIDIRECTIONAL}')"
                if value in ("both", "duplex", "inout", "in_out", "io") else "")
        add("error", "invalid_tier_modalities",
            f"model_tiers.{key} direction '{value}' for '{kind}' is not a "
            f"direction ({', '.join(sorted(VALID_MODALITY_DIRECTIONS))})"
            f"{hint}", where=where)
        return
    if value in (DIRECTION_INBOUND, DIRECTION_BIDIRECTIONAL):
        _warn_inert_inbound(key, kind, value, where, add, provider_name)
    if value in (DIRECTION_OUTBOUND, DIRECTION_BIDIRECTIONAL) \
            and not _delivers_output_media(provider_name):
        # Warn only when the adapter cannot actually deliver.  This used
        # to fire unconditionally, saying no adapter existed -- true when
        # written, and false the moment one did, at which point it told
        # every author of a WORKING speaking tier that their profile was
        # inert.  The provider's own `output_media` capability is the
        # thing that changes, so it is the thing to ask.
        add("warning", "outbound_modality_not_deliverable",
            f"model_tiers.{key} declares '{kind}' {value}, which parses but "
            f"is INERT: provider '{provider_name or '<unset>'}' does not "
            "declare `output_media`, so its adapter does not decode "
            "model-generated media — nothing can deliver it.  See "
            "docs/design/binary-media-chunks.md for the three touches that "
            "wire a provider."
            + ("  The inbound half of this role IS live."
               if value == DIRECTION_BIDIRECTIONAL
               and _carries_inbound_modality(provider_name, kind) else ""),
            where=where)


def _check_budget_control(profile: Any, add) -> None:
    """Check a profile's ``budget_control`` — starting with its ABSENCE (#947).

    ``budget_control`` is fully implemented and entirely opt-in, and nothing
    told an author their profile had no ceiling.  The failure mode is silent
    by construction: an unbudgeted profile behaves identically to a budgeted
    one right up until something loops, and then it does not stop.  Observed
    as a subagent retrying a tool whose result never reached its history —
    127 identical calls, ~57k tokens a request, killed by hand.

    Two findings here, ordered by how protected the profile LOOKS while it
    is not.  The second is the load-bearing one:

    ``budget_control_absent``
        no block at all, so every dimension is unbounded.

    ``budget_limits_without_abort``
        ``limits`` are declared and nothing enforces them.  A ceiling in
        ``limits`` is observed, never enforced — the ``degrade`` ladder is
        the only consumer of :meth:`BudgetTracker.usage_fraction`, so a
        profile with ceilings and no ``abort`` rung crosses them in
        silence.  Without this check the fix for the first finding is
        actively misleading: an author warned "you have no budget" writes
        ``limits: {usd: 5}``, the warning goes away, and the loop is still
        unbounded.

    **Warnings, not errors**, on the reasoning the issue sets out: an
    unbudgeted profile is a legitimate choice for a short-lived local
    agent, and these would otherwise fail every existing workspace at
    once.  Surfacing the knob must not break the people who need it.

    The ladder's own shape checks (``budget_overlay_*``) live in
    :func:`_check_budget_overlays`, called from here — split so neither
    function approaches the complexity ceiling, and so that
    :func:`validate_profile` (baselined far above it) gets smaller rather
    than larger.

    Args:
        profile: The resolved profile object.
        add: The per-profile diagnostic sink from :func:`validate_profile`.
    """
    from shared.budget_control import DIMENSIONS

    budget = getattr(profile, "budget_control", None)
    if budget is None:
        add("warn", "budget_control_absent",
            "declares no budget_control, so this session is unbounded on "
            f"every dimension ({', '.join(DIMENSIONS)}) — nothing stops a "
            "tool-call loop, and the first sign is the provider bill. "
            "Declare a ceiling AND a rung that enforces it, e.g. "
            "budget_control: {limits: {tool_calls: 200, usd: 5.0}, "
            "degrade: [{at: 100, action: abort}]}."
            + _backgrounded_profile_note(profile),
            where="budget_control")
        return

    # getattr, like every other read in this module: a validator that raises
    # tells the author nothing at all.  The default is the fail-LOUD direction
    # — an object that cannot answer "do you abort?" is treated as one that
    # does not, so an unrecognised shape draws the warning rather than a
    # clean bill.
    if not getattr(budget, "has_abort_rung", False):
        limits = getattr(budget, "limits", None) or {}
        add("warn", "budget_limits_without_abort",
            f"budget_control declares limits ({', '.join(sorted(limits))}) "
            f"but {_ladder_shape(budget)} — the ceilings are observed, never "
            "enforced. The degrade ladder is the only consumer of the usage "
            "fraction, so this run crosses 100% in silence; of the three "
            "actions only 'abort' stops it (finalize and escalate are latched "
            "for a layer above and are advice a looping model can decline). "
            "Add a terminal rung: degrade: [{at: 100, action: abort}]."
            + _backgrounded_profile_note(profile),
            where="budget_control.degrade")

    _check_budget_overlays(profile, budget, add)


def _ladder_shape(budget: Any) -> str:
    """Describe what the ladder does instead of aborting, for the message.

    "no degrade ladder at all" and "a ladder that ends in finalize" are the
    same defect and want the same code, but not the same sentence — telling
    an author who wrote a three-rung ladder that they have none reads as a
    validator bug and gets the finding dismissed.
    """
    rungs = tuple(getattr(budget, "degrade", ()) or ())
    if not rungs:
        return "declares no degrade ladder at all"
    last = rungs[-1]
    # Ladder order, deduped — not sorted: the message reads as a description
    # of the author's own ladder, and re-alphabetising it ("escalate/finalize"
    # for a finalize-then-escalate ladder) reads as a different ladder.
    actions = list(dict.fromkeys(r.action for r in rungs if r.action))
    if not actions:
        return (f"its {len(rungs)}-rung ladder only rebinds tiers (a brownout, "
                f"never a stop)")
    return (f"its ladder ends at {last.at_percent:g}% with "
            f"{'/'.join(actions)}, and no rung aborts")


def _backgrounded_profile_note(profile: Any) -> str:
    """Extra sentence for a profile that is bound to a persona (#947).

    The danger is not uniform.  A profile spawned as a subagent outlives
    the thing that would have noticed: ``spawn_subagent`` backgrounds it,
    and the parent's shutdown deliberately preserves it ("Subagent plugin
    shutdown (running subagents preserved)") — correct behaviour, and
    exactly what leaves an unbudgeted loop with nothing left in the
    session to stop it.  In the incident the parent had been gone for two
    minutes and the child was still spending.

    ``default_agent`` is the honest discriminator available here.  Every
    discovered profile is reachable by name through ``spawn_subagent``, so
    "is it spawnable" cannot separate anything; a profile that names its
    own persona is one built to be spawned by profile name alone, which is
    what #944 added ``default_agent`` for.  The signal is therefore used
    ONE WAY — to strengthen a message that fires regardless — never to
    weaken or suppress one, because its absence proves nothing.

    Returns a leading-space sentence, or ``""``.
    """
    if not getattr(profile, "default_agent", None):
        return ""
    return (" This profile binds a default_agent, so it is built to be spawned "
            "by name — and a subagent is deliberately preserved when its "
            "parent shuts down, so an unbounded loop here outlives the "
            "session that could have noticed it.")


def _check_budget_overlays(profile: Any, budget: Any, add) -> None:
    """Check a degrade ladder's tier overlays against the profile's tiers.

    The block is already parsed + structurally validated at profile-load
    time (``BudgetControlConfig.from_dict``; a malformed one surfaces as
    ``parse_error``, so it never reaches here).  What load time CANNOT know
    is (a) which providers are installed and (b) how the ladder relates to
    the profile's own ``model_tiers`` — both checked here, reusing the exact
    same resolve_provider / tier-name machinery :func:`_check_model_tiers`
    uses (an overlay IS a tier table, so it inherits the same defect
    classes).

    Args:
        profile: The resolved profile object.
        budget: Its non-``None`` :class:`~shared.budget_control.BudgetControlConfig`.
        add: The per-profile diagnostic sink from :func:`validate_profile`.
    """
    # Local import: _check_model_tiers imports these inside its own body,
    # which may not have run (a profile can declare a budget ladder with no
    # tiers — exactly the case flagged below).
    from shared.model_tiers import RESERVED_KEYS
    declared_tiers = {
        k for k in (getattr(profile, "model_tiers", None) or {})
        if k not in RESERVED_KEYS
    }
    for i, rung in enumerate(getattr(budget, "degrade", ()) or ()):
        overlay = getattr(rung, "model_tiers", None) or {}
        if overlay and not declared_tiers:
            # An overlay patches the session's tier table; with no
            # model_tiers there is no table to patch, so the rung would
            # silently do nothing at runtime.
            add("error", "budget_overlay_without_tiers",
                f"budget_control.degrade[{i}] overlays model_tiers "
                f"({', '.join(sorted(overlay))}) but the profile declares no "
                "model_tiers — the overlay would have no table to rebind. "
                "Declare model_tiers, or use an action-only rung "
                "(finalize / abort / escalate).",
                where=f"budget_control.degrade[{i}].model_tiers")
            continue
        for tier_name, entry in overlay.items():
            if tier_name not in declared_tiers:
                add("warn", "budget_overlay_undeclared_tier",
                    f"budget_control.degrade[{i}] rebinds tier "
                    f"'{tier_name}', which the profile's model_tiers does "
                    f"not declare (has: {', '.join(sorted(declared_tiers))})"
                    " — degrading would ADD a tier the agent could not "
                    "reach before.",
                    where=f"budget_control.degrade[{i}].model_tiers.{tier_name}")
            tprov = getattr(entry, "provider", None)
            if tprov and introspect.resolve_provider(tprov) is None:
                add("error", "unknown_provider",
                    f"budget_control.degrade[{i}].model_tiers.{tier_name} "
                    f"provider '{tprov}' is not installed (a degrade overlay "
                    "may cross providers, but must name a real one — see "
                    "`jaato-scaffold explain providers`)",
                    where=f"budget_control.degrade[{i}].model_tiers."
                          f"{tier_name}.provider")


def _check_model_tiers(mt_cfg, add, provider_name=None):
    """Static checks on a profile's ``model_tiers`` table.

    Catches, before a session is ever created, what
    :class:`~shared.model_tiers.ModelTierConfig` would only raise at
    session-create time: tier-name typos, a deployment-named tier with no
    ``description`` to stand in for the prose the framework cannot supply,
    a cross-provider tier naming an uninstalled provider, and a malformed
    ``description``.  See
    ``jaato-scaffold explain tiers``.  (``model_tiers`` survives the
    inherits/set merge — see ``config._merge_profiles``.)

    Split out of :func:`validate_profile` to keep that function under the
    complexity ceiling.

    Args:
        mt_cfg: The profile's raw ``model_tiers`` dict (possibly empty).
        add: ``validate_profile``'s diagnostic collector,
            ``add(severity, code, message, where=...)``.
    """
    if not mt_cfg:
        return
    from shared.model_tiers import (
        CANONICAL_TIER_NAMES, RESERVED_KEYS, is_canonical_tier_name,
        tier_name_error,
    )
    for key, entry in mt_cfg.items():
        if key in RESERVED_KEYS:
            continue
        reason = tier_name_error(key)
        if reason is not None:
            add("error", "unknown_tier",
                f"model_tiers key {reason}",
                where=f"model_tiers.{key}")
            continue
        # A deployment-named tier has no framework prose behind it, so
        # ``description`` is required rather than optional — caught here as
        # well as at session-create time, because this is the surface an
        # author runs BEFORE paying for a session.  The shorthand
        # (``coder: some-model``) can never satisfy it, hence the check
        # sitting above the dict guard.
        #
        # The control-key hint rides along because a MISSPELLED control key
        # (``initail: executor``) now reads as a perfectly legal tier name
        # the framework has never heard of, and lands here rather than in
        # the branch above.  "needs a description" alone would be a true
        # statement about the wrong problem.
        if not is_canonical_tier_name(key):
            described = isinstance(entry, dict) and entry.get("description")
            if not described:
                add("error", "tier_description_required",
                    f"model_tiers.{key} needs a 'description' — the "
                    f"framework only has prose for "
                    f"{', '.join(sorted(CANONICAL_TIER_NAMES))}, so without "
                    f"one the model is told only which model this tier "
                    f"routes to, which is not a reason to enter it.  (If "
                    f"'{key}' was meant to be a control key, those are "
                    f"{', '.join(sorted(RESERVED_KEYS))}.)",
                    where=f"model_tiers.{key}")
        if not isinstance(entry, dict):
            continue
        tprov = entry.get("provider")
        if tprov and introspect.resolve_provider(tprov) is None:
            add("error", "unknown_provider",
                f"model_tiers.{key} provider '{tprov}' is not installed "
                "(V2 cross-provider tiers must name a real provider — see "
                "`jaato-scaffold explain providers`)",
                where=f"model_tiers.{key}.provider")
        # ``description`` reaches the MODEL (it becomes this tier's bullet
        # in the enter_tier tool schema), so a malformed one is worth
        # catching before a session pays to discover it.
        tdesc = entry.get("description")
        if tdesc is not None and (
            not isinstance(tdesc, str) or not tdesc.strip()
        ):
            add("error", "invalid_tier_description",
                f"model_tiers.{key} description must be a non-empty string "
                "— it is rendered verbatim as this tier's bullet in the "
                "enter_tier tool description",
                where=f"model_tiers.{key}.description")
        # ``modalities`` declares which input roles this tier fills.  A typo
        # here is silent at runtime in the worst way: the content gate finds
        # no tier for an image and the agent is told none exists, while the
        # profile plainly declares one.
        _check_tier_exit(key, entry.get("exit_on"), add)
        _check_tier_modalities(key, entry.get("modalities"), add,
                               entry.get("provider") or provider_name)


def _effective_scrub_value(profile, surface):
    """The ``scrub_secret_env`` value surface ``surface`` will resolve.

    Mirrors the runtime precedence exactly — ``plugin_configs.<surface>``
    wins when it names the key at all, else the profile-level key, else
    ``None`` (which the plugin reads as the framework default).  Returns
    ``(value, where)`` so a finding can point at the field that decided.
    """
    cfg = (getattr(profile, "plugin_configs", None) or {}).get(surface)
    if isinstance(cfg, dict) and "scrub_secret_env" in cfg:
        return cfg["scrub_secret_env"], f"plugin_configs.{surface}.scrub_secret_env"
    return getattr(profile, "scrub_secret_env", None), "scrub_secret_env"


def _check_secret_scrub(profile, add):
    """Flag a subprocess surface that runs with the runner's full environment.

    The scrub is ON by default (#863), so a profile that says nothing is
    fine.  What this surfaces is the DELIBERATE leaky posture — a profile
    (or one of its ``plugin_configs`` sections) that resolves to no scrub
    pattern for a plugin that spawns model-driven subprocesses — and the
    two defects around it: a value the grammar rejects (which the plugin
    fails CLOSED on, so the author's intent is silently replaced by the
    default set) and a profile-level key with no surface to apply to.
    """
    from shared.secret_scrub import (
        SCRUB_HINT, SCRUB_SURFACES, is_scrub_disabled, normalize_scrub_patterns,
    )
    enabled = [s for s in SCRUB_SURFACES
               if s in (getattr(profile, "plugins", None) or [])]
    profile_value = getattr(profile, "scrub_secret_env", None)
    if profile_value is not None and not enabled:
        add("info", "scrub_secret_env_inert",
            "scrub_secret_env is set but the profile enables none of the "
            f"plugins it applies to ({', '.join(SCRUB_SURFACES)}) — it "
            "changes nothing here", where="scrub_secret_env")
    for surface in enabled:
        value, where = _effective_scrub_value(profile, surface)
        try:
            patterns = normalize_scrub_patterns(value)
        except ValueError as exc:
            add("error", "invalid_scrub_secret_env",
                f"{exc} — the {surface} plugin fails CLOSED on this (the "
                "framework default set is applied and the value ignored)",
                where=where)
            continue
        if is_scrub_disabled(patterns):
            add("warn", "secret_scrub_disabled",
                f"plugin '{surface}' spawns model-driven subprocesses with the "
                "runner's FULL environment — every provider API key and token "
                "the daemon holds is readable by any command the model runs "
                f"(`env`, `echo $GITHUB_TOKEN`).  {SCRUB_HINT}.",
                where=where)


# A declared type token → the predicate a value must satisfy.  Two
# vocabularies reach here and both are the plugin author's own: JSON Schema
# (``integer`` / ``boolean`` / ``array`` / ``object``) from a raw-dict
# ``get_config_schema``, and Python-ish names (``int`` / ``bool`` / ``dict``)
# from the ``PluginSetting`` object form.  A token in neither is UNKNOWN and
# checks nothing — the table is a source of findings, never of guesses.
#
# ``bool`` is excluded from the numeric predicates deliberately: Python makes
# ``True`` an ``int``, so ``timeout: true`` would otherwise satisfy a knob
# declared ``integer`` — which is exactly the silent-ignore shape this check
# exists to catch.
_KNOB_TYPE_PREDICATES = {
    "string":  lambda v: isinstance(v, str),
    "str":     lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "int":     lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "float":   lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "bool":    lambda v: isinstance(v, bool),
    "array":   lambda v: isinstance(v, (list, tuple)),
    "list":    lambda v: isinstance(v, (list, tuple)),
    "object":  lambda v: isinstance(v, dict),
    "dict":    lambda v: isinstance(v, dict),
    "null":    lambda v: v is None,
}

# ``scheme://…`` — a secret URI (``pass://``, ``vault://``) the daemon
# resolves at spawn, or a plain URL.
_KNOB_URI_RE = re.compile(r"^[a-z][a-z0-9_+.-]*://")


def _knob_value_is_deferred(value) -> bool:
    """True when a knob's value is not this validator's to judge.

    A ``${VAR}`` placeholder and a ``scheme://`` secret URI are both resolved
    LATER — by ``expand_variables`` at session-prep, against an environment
    the validator does not have.  Their literal form is a ``str`` whatever the
    knob declares, so checking it would report ``timeout: ${HTTP_TIMEOUT}``
    as a type error and ``lookup_strategy: ${STRATEGY}`` as an enum
    violation.  Deferring matches what ``subagent.config`` already does at
    every other boundary that meets an unexpanded value.
    """
    return isinstance(value, str) and (
        "${" in value or bool(_KNOB_URI_RE.match(value)))


def _check_knob_value(cfg_name, key, value, setting, add):
    """Check one knob's VALUE against the plugin's own declared schema (#925).

    Two findings, and their severities differ because the declarations differ
    in strength:

    * ``invalid_knob_value`` (**error**) — the value is outside the knob's
      declared ``enum``.  A plugin that spells out ``["memory","file",
      "hybrid"]`` has left nothing to be generous about, and the runtime
      consequence is the silent-fallback shape validate exists to catch:
      ``todo.storage_type: sqlite`` raises inside ``create_storage``, is
      caught, printed to daemon stdout, and replaced with in-memory storage —
      so an operator who asked for persistence gets none and nothing fails.
    * ``knob_type_mismatch`` (**warn**) — the value does not match the
      declared ``type``.  Softer on purpose: YAML scalar typing is easy to
      trip over (a quoted ``"30"``), a plugin may coerce, and a declared type
      can be an incomplete summary of what the knob accepts.

    Both are generic, driven by the declaration a plugin already publishes, so
    an OUT-OF-TREE plugin gets them with nothing to register — unlike
    ``_PLUGIN_VALUE_CHECKS``, which is a hardcoded jaato-server dict keyed by
    plugin name and therefore unreachable from a third-party distribution.
    That dict stays for genuinely structural knobs
    (``template.file_conventions``), which no declared type can describe.
    """
    where = f"plugin_configs.{cfg_name}.{key}"
    # ``None`` is "unset", not "wrongly typed" — many knobs default to it.
    if value is None or _knob_value_is_deferred(value):
        return
    if setting.enum is not None and value not in setting.enum:
        valid = ", ".join(repr(c) for c in setting.enum)
        add("error", "invalid_knob_value",
            f"{cfg_name}.{key} = {value!r} is not one of the values the "
            f"plugin declares ({valid}) — the value is not rejected at "
            f"runtime, it is silently replaced by a fallback", where=where)
        return
    predicates = [_KNOB_TYPE_PREDICATES[tok]
                  for tok in setting.type.split("|")
                  if tok in _KNOB_TYPE_PREDICATES]
    if not predicates:
        return          # undeclared or unrecognised type — nothing asserted
    if not any(pred(value) for pred in predicates):
        add("warn", "knob_type_mismatch",
            f"{cfg_name}.{key} = {value!r} ({type(value).__name__}) does not "
            f"match the declared type '{setting.type}' — assigned without "
            f"coercion at runtime and carried downstream", where=where)


def _validate_plugin_knobs(cfg_name, cfg, plugins, add):
    """Flag top-level knob names — and values — a non-provider plugin rejects.

    Uses the plugin's introspected ``get_config_schema``
    (``config_settings``).  Only validates when the plugin declares a schema —
    a plugin that declares none opts out (we cannot tell a typo from an
    accepted free-form key).

    An unknown NAME emits ``warn`` (not ``error``): a plugin's schema may be
    incomplete, so a hard failure would risk false positives; the signal still
    surfaces likely typos (e.g. ``evaluatorss``) which are silently ignored at
    runtime.  That generosity does not carry over to a declared knob's VALUE —
    see :func:`_check_knob_value`, which is where a declared ``enum`` or
    ``type`` is actually checked.  Nested / free-form sub-structures are still
    not descended: only a top-level knob's own scalar shape is judged.
    """
    if not isinstance(cfg, dict):
        return
    pinfo = plugins.get(cfg_name)
    if pinfo is None or not pinfo.config_settings:
        return
    declared = {s.name: s for s in pinfo.config_settings}
    for key, value in cfg.items():
        setting = declared.get(key)
        if setting is None:
            valid = ", ".join(sorted(declared))
            add("warn", "unknown_knob",
                f"'{key}' is not a declared {cfg_name} config knob "
                f"(silently ignored at runtime; known: {valid})",
                where=f"plugin_configs.{cfg_name}.{key}")
            continue
        _check_knob_value(cfg_name, key, value, setting, add)


def _check_template_routing(cfg, add):
    """Validate ``plugin_configs.template.file_conventions``'s shape (#900).

    The generic knob check (:func:`_validate_plugin_knobs`) verifies knob
    NAMES and deliberately does not descend into a knob's value.  Routing
    earns the exception: the plugin drops a malformed rule and carries on,
    so a bad table is not an error at runtime — it is *no routing*, and
    generated files then land outside the declared source root, where a
    validator gate does not look.  The gate examines zero files and
    reports a clean verdict over nothing.

    Also flags the one case the precedence rule makes surprising: a
    profile that declares ``file_conventions`` WITHOUT an
    ``output_path_routing`` list.  A knowledge base's stack declaration
    carries other keys under that name (``source_dirs``,
    ``source_extension``, ``build_file``), and carrying such a block into
    the profile verbatim suppresses ``template_routing.yaml`` — the key
    being present IS the declaration — while declaring no rules of its
    own.  Routing then silently stops.
    """
    if not isinstance(cfg, dict) or "file_conventions" not in cfg:
        return
    where = "plugin_configs.template.file_conventions"
    conventions = cfg["file_conventions"]
    if not isinstance(conventions, dict):
        add("error", "invalid_template_routing",
            f"file_conventions must be a mapping carrying "
            f"'output_path_routing', got {type(conventions).__name__} — the "
            "plugin reads it as no routing and, because the key is present, "
            "reads no template_routing.yaml either", where=where)
        return

    if "output_path_routing" not in conventions:
        add("warn", "template_routing_empty",
            "file_conventions declares no 'output_path_routing' — the key's "
            "presence alone suppresses template_routing.yaml, so this "
            "profile routes nothing.  Declare the rules here, or drop the "
            "key to keep the file", where=where)
        return

    rules = conventions["output_path_routing"]
    where_rules = f"{where}.output_path_routing"
    if not isinstance(rules, list):
        add("error", "invalid_template_routing",
            f"output_path_routing must be a list of {{glob, prefix}} entries, "
            f"got {type(rules).__name__} (dropped at runtime — no routing)",
            where=where_rules)
        return

    for i, rule in enumerate(rules):
        rule_where = f"{where_rules}[{i}]"
        if not isinstance(rule, dict):
            add("error", "invalid_template_routing",
                f"entry must be a mapping with 'glob' (and optionally "
                f"'prefix'), got {type(rule).__name__} (dropped at runtime)",
                where=rule_where)
            continue
        glob, prefix = rule.get("glob"), rule.get("prefix", "")
        if not isinstance(glob, str) or not glob:
            add("error", "invalid_template_routing",
                f"entry needs a non-empty string 'glob', got {glob!r} "
                "(dropped at runtime)", where=f"{rule_where}.glob")
        if not isinstance(prefix, str):
            add("error", "invalid_template_routing",
                f"'prefix' must be a string ('' means match-and-leave-alone), "
                f"got {type(prefix).__name__} (dropped at runtime)",
                where=f"{rule_where}.prefix")


#: Per-plugin value-shape checks, keyed by ``plugin_configs`` name.  The
#: generic name check (:func:`_validate_plugin_knobs`) deliberately does not
#: descend into a knob's value; an entry here is a knob whose CONTENT decides
#: behaviour badly enough to earn the exception.
_PLUGIN_VALUE_CHECKS = {
    "template": _check_template_routing,
}


def _check_plugin_knob_values(cfg_name, cfg, add):
    """Run the per-plugin value-shape check for ``cfg_name``, if any."""
    check = _PLUGIN_VALUE_CHECKS.get(cfg_name)
    if check is not None:
        check(cfg, add)


def _check_quirks(quirks_dict, pinfo, provider_name, add, where_prefix=None):
    """Flag quirk names the provider does not honor (silently dropped)."""
    for q in quirks_dict:
        if q not in pinfo.quirks:
            valid = ", ".join(sorted(pinfo.quirks)) or "(none — provider honors no quirks)"
            where = f"{where_prefix}.{q}" if where_prefix else \
                f"plugin_configs.{provider_name}.quirks.{q}"
            add("error", "unknown_quirk",
                f"quirk '{q}' is not honored by provider '{provider_name}' "
                f"(silently dropped at runtime; valid: {valid})", where=where)


# --------------------------------------------------------------------- .env

def _parse_env(text: str) -> Dict[str, str]:
    """Parse a ``.env`` into a dict (KEY=VALUE; ``#`` comments / blanks skipped)."""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def validate_env(workspace: str) -> List[Diagnostic]:
    """Validate a workspace ``.env``'s registry cross-references.

    Checks the two vars that name framework entities (so a typo is caught
    before it silently selects the wrong thing at runtime):

    - ``JAATO_PROVIDER`` must be a known provider.
    - ``JAATO_PROFILE_SET`` must name an existing set directory under
      ``.jaato/profiles/`` (the high-value catch — a mistyped set silently
      falls back to the base/wrong profiles).

    The ``.env``'s *absence* is not flagged here — that's the doctor's
    runtime-preflight job (the env_file=None handshake-crash surface).
    """
    ws = Path(workspace).resolve()
    envf = ws / ".env"
    out: List[Diagnostic] = []
    if not envf.exists():
        return out
    env = _parse_env(envf.read_text(encoding="utf-8"))

    prov = env.get("JAATO_PROVIDER")
    if prov and introspect.resolve_provider(prov) is None:
        out.append(Diagnostic(
            "error", "unknown_provider",
            f".env JAATO_PROVIDER='{prov}' is not a known provider "
            f"(have: {', '.join(sorted(introspect.providers()))})",
            profile=".env", where="JAATO_PROVIDER"))

    pset = env.get("JAATO_PROFILE_SET")
    if pset and not (ws / ".jaato" / "profiles" / pset).is_dir():
        out.append(Diagnostic(
            "error", "unknown_profile_set",
            f".env JAATO_PROFILE_SET='{pset}' has no matching set directory "
            f"under .jaato/profiles/ — the run will silently use base/wrong "
            f"profiles (see `jaato-scaffold explain sets`)",
            profile=".env", where="JAATO_PROFILE_SET"))

    # typo HINT (info, not error): a JAATO_* var no installed FRAMEWORK code
    # reads.  Deliberately INFO — it cannot distinguish a typo from a
    # legitimate app-level var read by the workspace's OWN cascade scripts /
    # reactors (e.g. kb's JAATO_INPUTS_DIR), so it must never fail a
    # workspace.  Scoped to our namespace (avoids HTTPS_PROXY etc.); safelists
    # vars read via session-context rather than a literal os.getenv.
    known = set(introspect.env_vars())
    known.update({"JAATO_PROFILE_SET"})
    for key in env:
        if key.startswith("JAATO_") and key not in known:
            out.append(Diagnostic(
                "info", "unread_env_var",
                f".env {key} is not read by installed framework code — a typo, "
                f"or an app-level var your own scripts read "
                f"(see `jaato-scaffold explain env`)",
                profile=".env", where=key))
    return out


# ------------------------------------------------------------- single file

def validate_profile_file(file_path: str) -> List[Diagnostic]:
    """Validate a STANDALONE profile file directly against the live registry.

    ``validate_workspace`` only sees profiles under the canonical
    ``<ws>/.jaato/profiles[/<set>]/`` layout (it goes through
    ``discover_profiles``).  A profile file *outside* that layout — a docs
    example, an ad-hoc ``/tmp/foo.yaml`` — previously resolved to a bogus
    workspace where ``discover_profiles`` found nothing, so the per-profile
    checks never ran and the file was silently reported "valid — no findings"
    (a false pass; the exact ``plugin_configs`` typo the tool exists to catch
    slipped through).

    This loads the file itself, reusing the framework scanner + inheritance
    resolver so the built profile matches what the runtime would construct
    (``plugins`` modifiers, ``plugin_configs``, ``gc``, …), then runs the full
    :func:`validate_profile` checks on it.  Inheritance is resolved within the
    file's own directory (sibling base profiles are picked up); a sibling that
    fails to parse does not affect the target — only the target profile's
    diagnostics are returned.
    """
    from shared.plugins.subagent.config import (
        SubagentProfile,
        _parse_profile_file,
        _scan_profiles_dir,
        resolve_profiles,
    )

    fp = Path(file_path).resolve()
    name, data, err = _parse_profile_file(fp)
    if err:
        return [Diagnostic("error", "parse_error", err, profile=fp.stem)]
    if not data:
        return [Diagnostic(
            "error", "parse_error",
            f"'{fp.name}' is not a profile file (expected a YAML/JSON object)",
            profile=fp.stem)]

    # Build via the framework scanner so the profile object matches runtime.
    scanned: Dict[str, SubagentProfile] = {}
    scan_errors: Dict[str, str] = {}
    _scan_profiles_dir(fp.parent, scanned, scan_errors)
    resolved, resolve_errors = resolve_profiles(scanned)

    out: List[Diagnostic] = []
    if name in resolve_errors:
        out.append(Diagnostic(
            "error", "inherit_error", resolve_errors[name], profile=name))

    target = (resolved.get(name) or scanned.get(name)
              or resolved.get(fp.stem) or scanned.get(fp.stem))
    if target is None:
        se = scan_errors.get(fp.stem) or scan_errors.get(name)
        return [Diagnostic(
            "error", "parse_error",
            se or f"could not load profile from '{fp.name}'",
            profile=name or fp.stem)]

    out.extend(validate_profile(
        target,
        providers=introspect.providers(),
        plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()),
    ))
    return out


# ---------------------------------------------------------------- workspace

def _check_prefetch_directives(
    ws: Path, config_root: str, out: List[Diagnostic],
) -> None:
    """Validate ``{{!py[?]:...}}`` prefetch directives in agent personas + base
    instructions: the referenced script must RESOLVE and define a top-level
    ``def render(context, args)``.  A MANDATORY directive (``{{!py:}}``) whose
    script is missing raises PrefetchError at session-prep; an OPTIONAL one
    (``{{!py?:}}``) silently degrades — so the model never sees the content.

    AST-only — does NOT import/execute the script (validate must be side-effect
    free); it checks the contract structurally, not by running render().
    """
    import ast
    from shared.script_loader import resolve_script_path
    from shared.dynamic_instructions import _PY_PLACEHOLDER

    for sub in ("agents", "instructions"):
        d = ws / ".jaato" / sub
        if not d.is_dir():
            continue
        for md in sorted(d.rglob("*.md")):
            try:
                content = md.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if "{{!py" not in content:
                continue
            where = str(md.relative_to(ws))
            for m in _PY_PLACEHOLDER.finditer(content):
                is_optional = bool(m.group(1))
                script_ref = m.group(2)
                directive = f"{{{{!py{'?' if is_optional else ''}:{script_ref}}}}}"
                path = resolve_script_path(
                    script_ref, workspace_path=str(ws), config_root=config_root)
                if path is None:
                    out.append(Diagnostic(
                        "warn" if is_optional else "error",
                        "prefetch_script_missing",
                        f"{directive} references a prefetch script that does not "
                        f"resolve (searched <config_root>/ then ~/.jaato/)"
                        + (" — optional, so it degrades at runtime"
                           if is_optional else
                           " — MANDATORY: session-prep will raise PrefetchError"),
                        where=where))
                    continue
                try:
                    tree = ast.parse(Path(path).read_text())
                except (OSError, SyntaxError) as exc:
                    out.append(Diagnostic(
                        "error", "prefetch_script_syntax_error",
                        f"{directive}: prefetch script {script_ref} fails to "
                        f"parse: {exc}", where=where))
                    continue
                renders = [n for n in tree.body
                           if isinstance(n, ast.FunctionDef) and n.name == "render"]
                if not renders:
                    out.append(Diagnostic(
                        "error", "prefetch_render_missing",
                        f"{directive}: prefetch script {script_ref} resolves but "
                        f"defines no top-level `def render(context, args)` — it "
                        f"will fail at session-prep", where=where))
                    continue
                a = renders[0].args
                if len(a.args) < 2 and a.vararg is None:
                    out.append(Diagnostic(
                        "warn", "prefetch_render_signature",
                        f"{directive}: prefetch script {script_ref} `render` takes "
                        f"{len(a.args)} positional param(s); the contract is "
                        f"`render(context, args)` (2)", where=where))


#: Every completion-asset path is resolved by joining it onto the config root
#: (``<config_root>/<path>``, i.e. ``<workspace>/.jaato/<path>`` by default),
#: so writing the prefix yourself asks for ``<ws>/.jaato/.jaato/...``.
_REDUNDANT_PATH_PREFIXES = (".jaato/", "./.jaato/")


def _redundant_prefix(ref: str) -> Optional[str]:
    """The ``.jaato/`` prefix a path should not carry, or ``None``.

    ``resolve_completion_schema`` / ``resolve_script_path`` join a relative
    reference onto the CONFIG ROOT — which is ``<workspace>/.jaato`` unless a
    client overrode it — so ``.jaato/completion_schemas/x.json`` resolves to
    ``<ws>/.jaato/.jaato/completion_schemas/x.json``.  That path never exists,
    the resolver returns ``None``, and the consequence is silent: with no
    schema the gate is dropped, ``signal_completion`` is hidden from the model,
    and the session ends without ever completing.

    The prefix is easy to write precisely because every OTHER path a profile
    author touches — the workspace paths in tool calls, the paths in
    ``explain paths`` — is spelled from the workspace root.
    """
    for prefix in _REDUNDANT_PATH_PREFIXES:
        if ref.startswith(prefix):
            return prefix
    return None


def _check_completion_assets(profiles, ws: Path, config_root: str, out) -> None:
    """Every file a profile's completion gate names must RESOLVE.

    Nothing checked these.  A ``completion_payload_schema`` or a
    ``completion_processors[].script`` that does not resolve is a WARNING in
    the runner log and nothing else: the schema-less gate hides
    ``signal_completion`` entirely (``_should_hide_signal_completion``), so the
    agent cannot signal, the framework spends its nudges re-prompting a model
    that is hunting for a tool it will never find, and the driver gets ``None``
    back from a session that looks like it ran.

    Two findings, and the first is the cheap one:

    ``redundant_config_root_prefix`` (**error**) — the path starts ``.jaato/``,
    which the resolver adds itself.  Deterministically unresolvable, and named
    separately because "file not found" sends an author looking on disk for a
    file that is sitting exactly where they put it.

    ``completion_asset_missing`` (**error**) — it resolves nowhere.  Error, not
    warn, for the same reason ``prefetch_script_missing`` is one: this is a
    declared asset the session cannot start correctly without, not a knob
    somebody might be ignoring on purpose.

    Side-effect free, like the rest of ``validate``: paths are LOCATED, never
    loaded — importing a processor would execute it.
    """
    from shared.script_loader import resolve_script_path
    from shared.completion_schema_loader import _resolve_schema_path

    for pname, profile in sorted(profiles.items()):
        # (reference, where, resolver, what an unresolved one COSTS)
        _SCHEMA_COST = ("signal_completion is then HIDDEN from the model "
                        "entirely, so the agent cannot signal and the session "
                        "never completes")
        _SCRIPT_COST = ("the gate then fails to load, and a processor that "
                        "cannot load blocks every completion it was meant to "
                        "check")
        refs = []
        schema = getattr(profile, "completion_payload_schema", None)
        if isinstance(schema, str) and schema:
            refs.append((schema, "completion_payload_schema",
                         _resolve_schema_path, _SCHEMA_COST))
        for i, entry in enumerate(getattr(profile, "completion_processors", None) or ()):
            script = getattr(entry, "script", None)
            if isinstance(script, str) and script:
                refs.append((script, f"completion_processors[{i}].script",
                             resolve_script_path, _SCRIPT_COST))

        for ref, where, resolver, cost in refs:
            prefix = _redundant_prefix(ref)
            if prefix is not None:
                out.append(Diagnostic(
                    "error", "redundant_config_root_prefix",
                    f"{ref!r} starts with {prefix!r} — the resolver joins this "
                    f"path onto the config root (<workspace>/.jaato by "
                    f"default), so it resolves to "
                    f"<config_root>/{prefix}{ref[len(prefix):]}, which does not "
                    f"exist.  Drop the prefix: {ref[len(prefix):]!r}.  Nothing "
                    f"fails loudly: {cost}.",
                    profile=pname, where=where))
                continue
            if Path(ref).is_absolute():
                continue     # an absolute path is the author's own business
            if resolver(ref, str(ws), config_root) is None:
                out.append(Diagnostic(
                    "error", "completion_asset_missing",
                    f"{ref!r} resolves in no tier (tried <config_root>/{ref} "
                    f"then ~/.jaato/{ref}).  Nothing fails loudly: {cost}.",
                    profile=pname, where=where))


def _check_default_agent_exists(profiles, ws: Path, config_root: str, out) -> None:
    """Flag a profile whose ``default_agent`` is not on disk (#944).

    ``default_agent`` binds a profile's persona to the profile, so
    ``spawn_subagent(profile=...)`` alone yields a subagent that has both
    tools and instructions.  A name that resolves to no file fails at the
    spawn — the one moment the caller can do nothing about it, since the
    caller passed no agent at all.  This is the cheap half: it fires
    without a spawn, like :func:`_check_spawn_schema_wire_types`.

    Lookup only (:func:`find_agent_file`), never a render: rendering a
    persona executes its ``{{!py:...}}`` prefetch scripts, and validate is
    side-effect free.

    Args:
        profiles: Mapping of profile name -> resolved profile object.
        ws: Workspace root.
        config_root: Directory that the workspace agent tier resolves against.
        out: Diagnostic list to append to.
    """
    from shared.plugins.subagent.config import find_agent_file

    for pname, profile in sorted((profiles or {}).items()):
        agent_name = getattr(profile, "default_agent", None)
        if not agent_name:
            continue
        if find_agent_file(agent_name, str(ws), config_root) is not None:
            continue
        out.append(Diagnostic(
            "error", "default_agent_missing",
            f"profile declares default_agent '{agent_name}', which resolves "
            f"to no file under <config_root>/agents|prompts/ or "
            f"~/.jaato/agents|prompts/ — every spawn_subagent(profile="
            f"'{pname}') that names no agent will fail",
            profile=pname, where="default_agent"))


def validate_workspace(
    workspace: str,
    *,
    profile_set: Optional[str] = None,
    only: Optional[str] = None,
    config_root: Optional[str] = None,
) -> List[Diagnostic]:
    """Resolve + validate every profile in a workspace (optionally one set).

    Reuses the framework's ``discover_profiles`` for resolution so the
    effective profiles match what the daemon would load.  ``config_root``
    overrides the ``<workspace>/.jaato`` tier the same way a client's
    ``config_root`` does at session creation (the doctor passes its own).
    """
    from shared.plugins.subagent.config import discover_profiles

    ws = Path(workspace).resolve()
    config_root = str(Path(config_root).resolve()) if config_root else str(ws / ".jaato")
    result = discover_profiles(
        profiles_dir=".jaato/profiles",
        base_path=str(ws),
        config_root=config_root,
        force_profile_set=profile_set,
    )

    # Source tier per profile.  ``discover_profiles`` merges the workspace tier
    # AND the inherited user tier (~/.jaato/profiles) into one EFFECTIVE set (as
    # the daemon resolves), so a workspace validation can surface findings from
    # user-tier profiles the author doesn't own.  Tag each finding with its tier
    # so the author can tell "mine" (workspace) from "inherited" (user): a
    # profile is ``workspace`` iff a file with its name lives under
    # ``<ws>/.jaato/profiles`` (the config_root tier); else it came from the user
    # tier.  Workspace-level checks (.env, prefetch) are ``workspace``.
    _ws_profiles = Path(config_root) / "profiles"
    ws_stems = {
        p.stem for p in _ws_profiles.rglob("*")
        if p.is_file() and p.suffix in (".yaml", ".yml", ".json")
    } if _ws_profiles.is_dir() else set()

    def _tier(pname: Optional[str]) -> str:
        return "workspace" if pname in ws_stems else "user"

    out: List[Diagnostic] = []
    for stem, err in (result.errors or {}).items():
        out.append(Diagnostic("error", "parse_error", err, profile=stem,
                              tier=_tier(stem)))

    # workspace-level .env cross-references (provider / profile-set)
    for d in validate_env(str(ws)):
        d.tier = "workspace"
        out.append(d)

    providers = introspect.providers()
    plugins = introspect.plugins()
    gc_names = list(introspect.gc_strategies().keys())

    items = result.profiles.items()
    for pname, profile in sorted(items):
        if only and pname != only:
            continue
        tier = _tier(pname)
        for d in validate_profile(
            profile, providers=providers, plugins=plugins, gc_names=gc_names,
        ):
            d.tier = tier
            out.append(d)

    # Prefetch directives in agent personas + base instructions: a {{!py:...}}
    # pointing at a missing script (or a script without render()) raises
    # PrefetchError at session-prep — surface it here, before runtime.  These are
    # workspace-tier assets.
    _before = len(out)
    _check_prefetch_directives(ws, config_root, out)
    _check_spawn_schema_wire_types(result.profiles, config_root, out)
    _check_completion_assets(result.profiles, ws, config_root, out)
    _check_default_agent_exists(result.profiles, ws, config_root, out)
    for d in out[_before:]:
        d.tier = "workspace"
    return out
