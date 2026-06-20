"""The ``explain`` verb — renders the introspect core by scope.

Progressive interrogation: an author drills from the overview into a plugin's
tools, a provider's knobs/quirks, the GC strategies, or a workspace's profile
sets — BEFORE committing to a ``new`` build.  Every function returns a
``(structured_dict, text)`` pair so the CLI can emit either ``--json`` (for an
agent) or a human table.  No metadata is computed here — it all comes from
:mod:`introspect`, the single source the validator also reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import introspect

Rendered = Tuple[Dict[str, Any], str]


def _yn(flag: bool) -> str:
    return "✓" if flag else "—"


# ---------------------------------------------------------------- overview

def overview() -> Rendered:
    P = introspect.providers()
    PL = introspect.plugins()
    GC = introspect.gc_strategies()
    data = {
        "providers": sorted(P),
        "plugins": len(PL),
        "gc_strategies": sorted(GC),
        "archetypes": ["client", "fire", "cascade", "observer"],
    }
    text = (
        "jaato-scaffold — interrogate the installed framework, then build.\n\n"
        f"  {len(P)} providers   {len(PL)} plugins   "
        f"{len(GC)} gc strategies   4 client archetypes\n\n"
        "drill down:\n"
        "  jaato-scaffold explain plugins\n"
        "  jaato-scaffold explain plugin <name>\n"
        "  jaato-scaffold explain providers\n"
        "  jaato-scaffold explain provider <name>\n"
        "  jaato-scaffold explain gc\n"
        "  jaato-scaffold explain sets [--workspace DIR]\n"
    )
    return data, text


# ----------------------------------------------------------------- plugins

def plugins() -> Rendered:
    PL = introspect.plugins()
    rows = []
    data = {}
    for name in sorted(PL):
        pi = PL[name]
        core = sum(1 for t in pi.tools if t.discoverability == "core")
        disc = len(pi.tools) - core
        data[name] = {
            "kind": pi.kind, "tier": pi.tier,
            "tools": len(pi.tools), "core": core, "dynamic": pi.dynamic,
        }
        tools = "dynamic" if pi.dynamic else f"{len(pi.tools)} ({core} core/{disc} disc)"
        rows.append(f"  {name:22} {pi.kind:10} {str(pi.tier or '-'):8} {tools}")
    text = (f"{'plugin':24}{'kind':12}{'tier':10}tools\n"
            + "  " + "-" * 56 + "\n" + "\n".join(rows))
    return data, text


def plugin(name: str) -> Rendered:
    PL = introspect.plugins()
    pi = PL.get(name)
    if pi is None:
        return ({"error": f"unknown plugin {name!r}"},
                f"unknown plugin {name!r} — see `explain plugins`")
    lines = [f"plugin: {name}",
             f"  kind={pi.kind}  tier={pi.tier or '-'}"
             + ("  (tools dynamic — need a live session)" if pi.dynamic else "")]
    if pi.tools:
        lines.append("  tools:")
        for t in pi.tools:
            badge = "core" if t.discoverability == "core" else "disc"
            lines.append(f"    [{badge}] {t.name:28} {t.description}")
    if pi.config_keys:
        lines.append("  config keys: " + ", ".join(pi.config_keys))
    data = {"kind": pi.kind, "tier": pi.tier, "dynamic": pi.dynamic,
            "tools": [{"name": t.name, "discoverability": t.discoverability}
                      for t in pi.tools],
            "config_keys": pi.config_keys}
    return data, "\n".join(lines)


# ---------------------------------------------------------------- providers

def providers() -> Rendered:
    P = introspect.providers()
    from shared.plugins.model_provider.base import CAPABILITY_FIELDS
    rows = []
    data = {}
    for name in sorted(P):
        info = P[name]
        caps = info.capabilities
        cap_dict = caps.as_dict() if caps else {}
        nknobs = sum(len(l.knobs) for l in info.knobs.layers) if info.knobs else 0
        data[name] = {"capabilities": cap_dict, "knobs": nknobs,
                      "quirks": sorted(info.quirks)}
        flags = " ".join(_yn(cap_dict.get(f, False)) for f in CAPABILITY_FIELDS)
        rows.append(f"  {name:16} {flags}   knobs:{nknobs:<3} quirks:{len(info.quirks)}")
    hdr = "  " + " " * 16 + " ".join(c[:4] for c in CAPABILITY_FIELDS)
    text = ("providers (capability flags: "
            + ", ".join(f"{c[:4]}={c}" for c in CAPABILITY_FIELDS) + ")\n"
            + hdr + "\n" + "\n".join(rows))
    return data, text


def provider(name: str) -> Rendered:
    info = introspect.resolve_provider(name)
    if info is None:
        return ({"error": f"unknown provider {name!r}"},
                f"unknown provider {name!r} — see `explain providers`")
    caps = info.capabilities.as_dict() if info.capabilities else {}
    knobs = info.knobs.as_dict() if info.knobs else {}
    data = {"provider": info.dir_name, "capabilities": caps,
            "quirks": sorted(info.quirks), "knobs": knobs}

    lines = [f"provider: {info.dir_name}"]
    lines.append("  capabilities: "
                 + ", ".join(k for k, v in caps.items() if v) or "  (none)")
    lines.append("  quirks: " + (", ".join(sorted(info.quirks)) or "(none)"))
    lines.append("  knobs (plugin_configs.%s.*):" % info.dir_name)
    if info.knobs:
        for layer in info.knobs.layers:
            tag = " (opaque pass-through)" if layer.opaque else ""
            desc = f"  — {layer.description}" if layer.description else ""
            lines.append(f"    [{layer.layer}]{tag}{desc}")
            for k in layer.knobs:
                dflt = f"  (default {k.default!r})" if k.default is not None else ""
                d = f"  {k.description}" if k.description else ""
                lines.append(f"      {k.name:22} {k.type:6}{d}{dflt}")
    return data, "\n".join(lines)


# ---------------------------------------------------------------------- gc

def gc() -> Rendered:
    GC = introspect.gc_strategies()
    data = GC
    names = sorted(GC)
    fields = GC[names[0]] if names else []
    text = ("gc strategies: " + ", ".join(names)
            + "\n  GCConfig fields: " + ", ".join(fields))
    return data, text


# --------------------------------------------------------------------- env

def env(filter_: str = None) -> Rendered:
    """Env vars the installed daemon + plugins read (optionally filtered).

    ``filter_`` matches a category substring or a var-name substring, so
    ``explain env nebius`` → ``provider:nebius`` vars, ``explain env gc`` →
    GC knobs, ``explain env framework`` → the daemon-general knobs.
    """
    EV = introspect.env_vars()
    groups: Dict[str, list] = {}
    for name in sorted(EV):
        v = EV[name]
        if filter_ and filter_ not in v.category and filter_ not in name:
            continue
        groups.setdefault(v.category, []).append(v)

    data = {
        cat: {v.name: {"default": v.default, "sources": v.sources[:2]}
              for v in vs}
        for cat, vs in groups.items()
    }
    head = f"env vars read by the installed daemon + plugins ({len(EV)} total)"
    if filter_:
        head += f" — filter '{filter_}'"
    lines = [head, "  (set these in the workspace .env; commented = optional)"]
    for cat in sorted(groups):
        lines.append(f"\n  [{cat}]")
        for v in groups[cat]:
            d = f" = {v.default}" if v.default not in (None, "") else ""
            lines.append(f"    {v.name}{d}")
    return data, "\n".join(lines)


# -------------------------------------------------------------------- sets

def sets(workspace: str) -> Rendered:
    """Enumerate profile-sets in a workspace + which provider/model each pins.

    A *set* is a subdirectory under ``.jaato/profiles/`` (sibling of the
    ``_base_*.yaml`` tier-1 profiles).  Selected at runtime by
    ``JAATO_PROFILE_SET``.
    """
    import yaml

    pdir = Path(workspace).resolve() / ".jaato" / "profiles"
    data: Dict[str, Any] = {}
    if not pdir.is_dir():
        return ({}, f"no .jaato/profiles/ under {workspace}")
    for sub in sorted(p for p in pdir.iterdir() if p.is_dir()):
        bindings = set()
        agents = []
        for yf in sorted(sub.glob("*.y*ml")):
            agents.append(yf.stem)
            try:
                doc = yaml.safe_load(yf.read_text()) or {}
            except Exception:
                continue
            prov, model = doc.get("provider"), doc.get("model")
            if prov or model:
                bindings.add((prov, model))
        data[sub.name] = {
            "agents": agents,
            "bindings": [{"provider": p, "model": m} for p, m in sorted(
                b for b in bindings if b[0])],
        }
    if not data:
        return ({}, f"no profile-sets (subdirs) under {pdir}")
    lines = ["profile-sets (select with JAATO_PROFILE_SET=<name>):"]
    for sname, d in data.items():
        binds = ", ".join(f"{b['provider']}/{b['model']}" for b in d["bindings"]) \
            or "(no provider/model bound)"
        lines.append(f"  {sname:24} {len(d['agents'])} agents → {binds}")
    return data, "\n".join(lines)
