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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.plugins.model_provider.base import KNOB_LAYERS
from jaato_sdk.plugins.model_provider.types import DISCOVERABILITY_EAGER
from . import introspect

# Layer names that nest under plugin_configs.<provider> as sub-dicts.
# ``top_level`` is not a nesting key — its knobs sit directly under the
# provider — so it is excluded from the "is this key a layer?" test.
_NESTING_LAYERS = frozenset(n for n in KNOB_LAYERS if n != "top_level")


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
    pinfo: Optional[introspect.ProviderInfo] = None

    # --- provider --------------------------------------------------------
    if provider_name:
        pinfo = introspect.resolve_provider(provider_name)
        if pinfo is None:
            add("error", "unknown_provider",
                f"provider '{provider_name}' is not a known model provider "
                f"(have: {', '.join(sorted(providers))})", where="provider")
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
    for plug in getattr(profile, "plugins", None) or []:
        if plug not in plugins:
            add("error", "unknown_plugin",
                f"plugin '{plug}' is not installed (run "
                "`jaato-scaffold explain plugins`)", where=f"plugins.{plug}")

    # --- model_tiers (V2: cross-provider tiers allowed) ------------------
    _check_model_tiers(getattr(profile, "model_tiers", None) or {}, add)

    # --- budget_control --------------------------------------------------
    # The block is already parsed + structurally validated at profile-load
    # time (BudgetControlConfig.from_dict; a malformed one surfaces as
    # parse_error, so it never reaches here).  What load time CANNOT know is
    # (a) which providers are installed and (b) how the ladder relates to the
    # profile's own model_tiers — both checked here, reusing the exact same
    # resolve_provider / tier-name machinery the model_tiers branch above uses
    # (an overlay IS a tier table, so it inherits the same defect classes).
    budget = getattr(profile, "budget_control", None)
    if budget is not None:
        # Local import: the model_tiers branch above imports these inside its
        # own `if`, which may not have run (a profile can declare a budget
        # ladder with no tiers — exactly the case flagged below).
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
            # runtime otherwise).  Nested / free-form sub-structures — e.g.
            # ``permission.policy`` tree, ``permission.evaluators`` map — are
            # NOT descended; only the top-level knob names are checked.
            _validate_plugin_knobs(cfg_name, cfg, plugins, add)
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

    # --- gc strategy -----------------------------------------------------
    gc = getattr(profile, "gc", None)
    gc_type = getattr(gc, "type", None) if gc is not None else None
    if gc_type:
        candidates = {gc_type, f"gc_{gc_type}"}
        if not (candidates & set(gc_names)):
            add("warn", "unknown_gc",
                f"gc type '{gc_type}' not among {gc_names}", where="gc.type")

    return out


def _check_model_tiers(mt_cfg, add):
    """Static checks on a profile's ``model_tiers`` table.

    Catches, before a session is ever created, what
    :class:`~shared.model_tiers.ModelTierConfig` would only raise at
    session-create time: tier-name typos, a cross-provider tier naming an
    uninstalled provider, and a malformed ``description``.  See
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
    from shared.model_tiers import VALID_TIER_NAMES, RESERVED_KEYS
    for key, entry in mt_cfg.items():
        if key not in VALID_TIER_NAMES and key not in RESERVED_KEYS:
            add("error", "unknown_tier",
                f"model_tiers key '{key}' is neither a tier name "
                f"({', '.join(sorted(VALID_TIER_NAMES))}) nor a control key "
                f"({', '.join(sorted(RESERVED_KEYS))})",
                where=f"model_tiers.{key}")
            continue
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


def _validate_plugin_knobs(cfg_name, cfg, plugins, add):
    """Flag top-level knob names a non-provider plugin does not declare.

    Uses the plugin's introspected ``get_config_schema`` (``config_keys``).
    Only validates when the plugin declares a schema — a plugin that declares
    none opts out (we cannot tell a typo from an accepted free-form key).
    Emits ``warn`` (not ``error``): a plugin's schema may be incomplete, so a
    hard failure would risk false positives; the signal still surfaces likely
    typos (e.g. ``evaluatorss``) which are silently ignored at runtime.
    """
    if not isinstance(cfg, dict):
        return
    pinfo = plugins.get(cfg_name)
    if pinfo is None or not pinfo.config_keys:
        return
    known = set(pinfo.config_keys)
    for key in cfg:
        if key not in known:
            valid = ", ".join(sorted(known))
            add("warn", "unknown_knob",
                f"'{key}' is not a declared {cfg_name} config knob "
                f"(silently ignored at runtime; known: {valid})",
                where=f"plugin_configs.{cfg_name}.{key}")


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


def validate_workspace(
    workspace: str,
    *,
    profile_set: Optional[str] = None,
    only: Optional[str] = None,
) -> List[Diagnostic]:
    """Resolve + validate every profile in a workspace (optionally one set).

    Reuses the framework's ``discover_profiles`` for resolution so the
    effective profiles match what the daemon would load.
    """
    from shared.plugins.subagent.config import discover_profiles

    ws = Path(workspace).resolve()
    config_root = str(ws / ".jaato")
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
    _ws_profiles = ws / ".jaato" / "profiles"
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
    for d in out[_before:]:
        d.tier = "workspace"
    return out
