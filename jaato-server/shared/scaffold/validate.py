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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity, "code": self.code,
            "message": self.message, "profile": self.profile, "where": self.where,
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
    if provider_name and not model:
        add("warn", "missing_model",
            f"provider '{provider_name}' set but no model — set-overlay or "
            "inherits did not bind a model", where="model")

    # --- plugins ---------------------------------------------------------
    for plug in getattr(profile, "plugins", None) or []:
        if plug not in plugins:
            add("error", "unknown_plugin",
                f"plugin '{plug}' is not installed (run "
                "`jaato-scaffold explain plugins`)", where=f"plugins.{plug}")

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

    # --- plugin_configs knobs (the silent-ignore class) ------------------
    plugin_configs = getattr(profile, "plugin_configs", None) or {}
    for cfg_name, cfg in plugin_configs.items():
        # only provider-named config blocks have a knob contract today
        cfg_provider = introspect.resolve_provider(cfg_name)
        if cfg_provider is None or cfg_provider.knobs is None:
            continue
        knobs = cfg_provider.knobs
        if not isinstance(cfg, dict):
            continue
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
                # a top_level knob
                if not knobs.accepts("top_level", key):
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


# ---------------------------------------------------------------- workspace

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

    out: List[Diagnostic] = []
    for stem, err in (result.errors or {}).items():
        out.append(Diagnostic("error", "parse_error", err, profile=stem))

    # workspace-level .env cross-references (provider / profile-set)
    out.extend(validate_env(str(ws)))

    providers = introspect.providers()
    plugins = introspect.plugins()
    gc_names = list(introspect.gc_strategies().keys())

    items = result.profiles.items()
    for pname, profile in sorted(items):
        if only and pname != only:
            continue
        out.extend(validate_profile(
            profile, providers=providers, plugins=plugins, gc_names=gc_names,
        ))
    return out
