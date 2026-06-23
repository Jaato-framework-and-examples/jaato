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
        "  jaato-scaffold explain transports\n"
        "  jaato-scaffold explain clients\n"
        "  jaato-scaffold explain runtime\n"
        "  jaato-scaffold explain tiers\n"
        "  jaato-scaffold explain sets [--workspace DIR]\n"
        "  jaato-scaffold explain profile\n"
        "  jaato-scaffold explain paths\n"
    )
    return data, text


# ------------------------------------------------------------- transports

def transports() -> Rendered:
    """The two client transports + the daemon flags / auth that gate them.

    Source of truth for "how does a client connect" — IPC (Python SDK) vs
    WebSocket (TypeScript SDK).  The Python SDK is IPC-only; WS clients are
    authored with ``jaato-sdk-ts`` / the browser web-client.
    """
    data = {
        "ipc": {
            "sdk": "jaato-sdk (Python) — jaato_sdk.IPCClient",
            "scope": "local (same host / container)",
            "daemon_flags": ["--ipc-socket PATH", "--socket-mode MODE (default 660)"],
            "auth": "none — any principal that can open the socket drives the agent",
        },
        "websocket": {
            "sdk": "jaato-sdk-ts (TypeScript) / browser web-client — NOT the Python SDK",
            "scope": "remote / browser",
            "daemon_flags": ["--web-socket [HOST:]PORT", "--ws-token TOKEN",
                             "--ws-token-file PATH", "--ws-unsafe-no-auth"],
            "token_default": "~/.jaato/ws.token (auto-generated, 0600, on first WS start)",
            "auth": "bearer token (required unless --ws-unsafe-no-auth)",
            "client_auth": ["Authorization: Bearer <token>  (header)",
                            "?token=<token>  (query param — browsers)"],
            "bad_token": "WS close code 1008",
            "preflight": "jaato-doctor --web-socket [host:]port",
        },
    }
    text = (
        "client transports\n"
        "  ----------------------------------------------------------------\n"
        "  IPC (Unix socket)             WebSocket\n"
        "  - Python SDK                  - TypeScript SDK (jaato-sdk-ts)\n"
        "    jaato_sdk.IPCClient           / browser web-client\n"
        "  - local only                  - remote / browser\n"
        "  - unauthenticated             - bearer-token authenticated\n"
        "    (socket-mode 660)\n\n"
        "daemon flags:\n"
        "  IPC:  --ipc-socket PATH   [--socket-mode 660]\n"
        "  WS:   --web-socket [HOST:]PORT\n"
        "        --ws-token TOKEN | --ws-token-file PATH | --ws-unsafe-no-auth\n"
        "        (no token flag → daemon auto-generates ~/.jaato/ws.token, 0600)\n\n"
        "WS auth contract — the TS client presents ONE of:\n"
        "  Authorization: Bearer <token>   (header — SDK / proxies / curl)\n"
        "  ?token=<token>                  (query — browsers can't set headers on\n"
        "                                   new WebSocket())\n"
        "  bad token → WS close 1008.  Daemon stores only the SHA-256 digest and\n"
        "  compares with hmac.compare_digest.\n\n"
        "preflight the WS daemon side (port + token file + auth mode):\n"
        "  jaato-doctor --web-socket [host:]port\n\n"
        "NOTE: the Python SDK is IPC-only.  `jaato-scaffold new client` scaffolds\n"
        "the Python IPC client; a WebSocket client is authored with the TypeScript\n"
        "SDK (jaato-sdk-ts) — start from the web-client.\n"
    )
    return data, text


# --------------------------------------------------------------- clients

def clients() -> Rendered:
    """The two Python IPC client classes — when to use each.

    The Python SDK ships two IPC clients (WebSocket is the TS SDK — see
    ``transports``).  ``new client`` emits ``IPCClient`` by default; pass
    ``--recoverable`` to emit ``IPCRecoveryClient``.
    """
    data = {
        "ipc_client": {
            "class": "jaato_sdk.IPCClient",
            "recovery": "none — a dropped connection ends the client",
            "use_for": ["short-lived / one-shot", "a single send_message",
                        "a scripted run that exits"],
            "scaffold": "jaato-scaffold new client ...",
        },
        "ipc_recovery_client": {
            "class": "jaato_sdk.IPCRecoveryClient",
            "recovery": ("auto-reconnect state machine "
                         "(DISCONNECTED→CONNECTING→CONNECTED→RECONNECTING→CLOSED); "
                         "on_status_change callback; IncompatibleServerError is "
                         "permanent (no retry)"),
            "use_for": ["long-lived / resilient",
                        "TUI / observer / cascade driver",
                        "must survive a daemon restart "
                        "(per-run jaato-server --stop + autostart)"],
            "scaffold": "jaato-scaffold new client --recoverable ...",
        },
    }
    text = (
        "Python IPC clients  (WebSocket is the TS SDK — see `explain transports`)\n"
        "  ----------------------------------------------------------------\n"
        "  IPCClient                       IPCRecoveryClient\n"
        "  - no reconnect                  - auto-reconnect (state machine)\n"
        "  - dropped conn ends the client  - on_status_change callback\n"
        "  - short-lived / one-shot        - long-lived / resilient\n"
        "    (one send, a scripted run)      (TUI, observer, cascade driver;\n"
        "                                     survives a daemon restart)\n\n"
        "the SDK README features IPCRecoveryClient for anything long-lived — it\n"
        "rides through a per-run `jaato-server --stop` + autostart; on a permanent\n"
        "IncompatibleServerError it stops rather than looping.\n\n"
        "scaffold either (the flag works for every archetype):\n"
        "  jaato-scaffold new client ...                # IPCClient\n"
        "  jaato-scaffold new client --recoverable ...  # IPCRecoveryClient\n"
    )
    return data, text


# --------------------------------------------------------------- runtime

def runtime() -> Rendered:
    """How a session runs + how to DEBUG it — entities, the workspace flow, the
    log map, and the one-command session diagnostic.

    Curated (the runtime architecture is not in the plugin registry).  Pairs with
    ``jaato-doctor --session <id>``, which applies this map to a live session.
    """
    data = {
        "entities": {
            "daemon": "long-lived singleton on the IPC socket; daemon-tier plugins, "
                      "provider/OAuth/GC/sessions live here",
            "session": "per-conversation state (history/profile/workspace); "
                       "session.new -> bootstrap -> runner spawn -> plugin init -> turns",
            "runner": "per-session SUBPROCESS running the model loop + runner-tier "
                      "plugins (filesystem_query/file_edit/cli/lsp/mcp/notebook); "
                      "pool-served or cold-spawned; its OWN ContextVars + env",
        },
        "workspace_flow": [
            "client: IPCClient(workspace_path=ws) -> set_workspace + working_dir",
            "daemon: working_dir -> session.workspace_path -> envelope.workspace_path",
            "runner: envelope.workspace_path -> registry.set_workspace_path (config+hook)"
            " + set_workspace_root() ContextVar + os.environ, BEFORE expose_all",
            "plugin: initialize() reads config['workspace_root'] / get_workspace_root()"
            " -> caches self._workspace_root",
            "BREAKS: working_dir not sent OR runner didn't seed ContextVar/env"
            " -> workspace=none -> path tools Permission-denied (#344 class)",
        ],
        "logs": {
            "logs/runner-<sid>.log": "runner bootstrap + runner-tier plugin init "
                                     "(where workspace=none surfaces)",
            "logs/session_<id>_client_*.log": "session-level",
            "sessions/<id>.json": "history + workspace_path",
            "daemon log": "daemon-tier (e.g. /tmp/jaato.log)",
        },
        "debug": "jaato-doctor --session <id|latest> --workspace DIR",
    }
    text = (
        "jaato runtime — entities, workspace flow, logs, how to debug\n"
        "  ----------------------------------------------------------------\n"
        "ENTITIES\n"
        "  daemon   long-lived singleton on the IPC socket; daemon-tier plugins,\n"
        "           provider / OAuth / GC / sessions live here.\n"
        "  session  per-conversation state (history/profile/workspace):\n"
        "           session.new -> bootstrap -> runner spawn -> plugin init -> turns.\n"
        "  runner   per-session SUBPROCESS running the model loop + RUNNER-TIER\n"
        "           plugins (filesystem_query/file_edit/cli/lsp/mcp/notebook);\n"
        "           pool-served (warm) or cold-spawned; its OWN ContextVars + env.\n\n"
        "WORKSPACE FLOW  (the path-tool 'workspace=none' class of bug)\n"
        "  client   IPCClient(workspace_path=ws) -> set_workspace + working_dir\n"
        "  daemon   working_dir -> session.workspace_path -> envelope.workspace_path\n"
        "  runner   envelope.workspace_path -> registry.set_workspace_path (config+hook)\n"
        "           + set_workspace_root() ContextVar + os.environ, BEFORE expose_all\n"
        "  plugin   initialize() reads config['workspace_root'] / get_workspace_root()\n"
        "           -> caches self._workspace_root (path tools enforce this boundary)\n"
        "  BREAKS   working_dir not sent, OR runner didn't seed the ContextVar/env\n"
        "           -> workspace=none -> path tools Permission-denied (the #344 class)\n\n"
        "LOGS  (under <workspace>/.jaato/)\n"
        "  logs/runner-<sid>.log           runner bootstrap + runner-tier plugin init\n"
        "                                  (where 'workspace=none' surfaces)\n"
        "  logs/session_<id>_client_*.log  session-level\n"
        "  sessions/<id>.json              history + workspace_path\n"
        "  <daemon stdout/log>             daemon-tier (e.g. /tmp/jaato.log)\n\n"
        "DEBUG A SESSION (one command — reads the logs above):\n"
        "  jaato-doctor --session <id|latest> --workspace DIR\n"
        "  -> reports whether the runner-tier path plugins resolved the workspace\n"
        "     (PASS=<ws>) or got workspace=none (FAIL + the fix), plus the log map.\n"
    )
    return data, text


# ------------------------------------------------------------------- tiers

def tiers() -> Rendered:
    """Model tiers — multi-model sessions: cognitive roles + modality (vision)
    roles, switched mid-session via ``enter_tier``.  V2 (#354): tiers may span
    PROVIDERS.  Introspects ``shared.model_tiers`` (VALID_TIER_NAMES /
    RESERVED_KEYS) so it tracks the installed framework.
    """
    from shared import model_tiers as mt
    valid = sorted(mt.VALID_TIER_NAMES)
    reserved = sorted(mt.RESERVED_KEYS)
    data = {
        "tier_names": valid,
        "reserved_keys": reserved,
        "shape": "model_tiers: { <tier>: <model-str> | {model, provider}, "
                 "initial: <tier>, fallback: <tier> }",
        "switching": "the MODEL calls enter_tier('<tier>') mid-session; the "
                     "active tier selects the model (and, V2, the provider); "
                     "history is preserved across the switch",
        "vision": "a 'vision' tier maps to an image-capable model; an image to a "
                  "non-vision active provider trips the content gate (a synthetic "
                  "'enter_tier(\"vision\") first' the agent self-corrects on). "
                  "user-message images ride the attachment ferry (#353): SDK "
                  "send_message(attachments=[path | {mime_type,data,display_name}])",
        "cross_provider": "V2 (#354): tiers may declare DIFFERENT providers; "
                          "switch_tier swaps to a cached per-tier provider "
                          "instance (history is provider-neutral; switch-back is "
                          "O(1)). e.g. a cheap zhipuai text executor + a gemini/"
                          "OpenRouter vision tier in ONE profile.",
    }
    text = (
        "jaato model tiers — multi-model sessions (cognitive + modality roles)\n"
        "  ----------------------------------------------------------------\n"
        f"TIER NAMES   {', '.join(valid)}\n"
        f"CONTROL KEYS {', '.join(reserved)}  (reserved: initial tier + fallback)\n\n"
        "SHAPE  (in a profile)\n"
        "  model_tiers:\n"
        "    <tier>:   <model-string>   OR   {model: <m>, provider: <p>}\n"
        "    initial:  <tier>           # the tier a session starts in\n"
        "    fallback: <tier>           # when enter_tier names an undeclared tier\n\n"
        "SWITCHING\n"
        "  the MODEL calls enter_tier('<tier>') mid-session; the active tier picks\n"
        "  the model (and, V2, the provider).  conversation history is preserved.\n\n"
        "VISION  (a modality tier)\n"
        "  map a 'vision' tier to an image-capable model.  an image reaching a\n"
        "  non-vision active provider trips the content gate: a synthetic note\n"
        "  'enter_tier(\"vision\") first' the agent self-corrects on.  user-message\n"
        "  images ride the attachment ferry — SDK send_message(attachments=...).\n\n"
        "CROSS-PROVIDER  (V2)\n"
        "  tiers may declare DIFFERENT providers — switch_tier swaps to a cached\n"
        "  per-tier provider instance (history is provider-neutral; switch-back is\n"
        "  O(1)).  e.g. a cheap zhipuai text executor + a gemini/OpenRouter vision\n"
        "  tier in ONE profile (no need to relocate the whole profile).\n"
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
            + "  " + "-" * 56 + "\n" + "\n".join(rows)
            + "\n\n  core = in the model's initial schema; disc = deferred "
              "(discoverable via\n  list_tools/get_tool_schemas, or force eager "
              "with `<plugin>(preload)` in a profile)")
    return data, text


def plugin(name: str) -> Rendered:
    PL = introspect.plugins()
    pi = PL.get(name)
    if pi is None:
        return ({"error": f"unknown plugin {name!r}"},
                f"unknown plugin {name!r} — see `explain plugins`")
    lines = [f"plugin: {name}"]
    if pi.description:
        lines.append(f"  {pi.description}")
    lines.append(f"  kind={pi.kind}  tier={pi.tier or '-'}"
                 + ("  (tools dynamic — need a live session)" if pi.dynamic else ""))
    if pi.tools:
        lines.append("  tools:")
        for t in pi.tools:
            badge = "core" if t.discoverability == "core" else "disc"
            lines.append(f"    [{badge}] {t.name:28} {t.description}")
        if any(t.discoverability != "core" for t in pi.tools):
            lines.append(
                f"  note: [core] tools are in the model's INITIAL schema; [disc] "
                f"are DEFERRED — the model reaches them by calling list_tools / "
                f"get_tool_schemas (introspection is always core, so they're never "
                f"lost), OR add `{name}(preload)` to a profile's plugins to force "
                f"ALL of this plugin's tools eager.")
    if pi.config_settings:
        lines.append(f"  config (plugin_configs.{name}.*):")
        for s in pi.config_settings:
            dflt = f"  (default {s.default!r})" if s.default is not None else ""
            d = f"  {s.description}" if s.description else ""
            lines.append(f"    {s.name:22} {s.type:8}{d}{dflt}")
    data = {"description": pi.description,
            "kind": pi.kind, "tier": pi.tier, "dynamic": pi.dynamic,
            "tools": [{"name": t.name, "discoverability": t.discoverability,
                       "description": t.description} for t in pi.tools],
            "config": [{"name": s.name, "type": s.type, "default": s.default,
                        "description": s.description} for s in pi.config_settings]}
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


def _resolution_order(info, EV) -> list:
    """Provider-specific resolution chains, from the uniform helper contracts.

    The precedence is the framework's (``resolve_context_window`` /
    ``resolve_modalities`` / the ``_knob`` helper); the concrete env-var name
    is the provider's own (looked up in the introspected env vars).
    """
    d = info.dir_name
    lines = []
    if info.auth:
        steps = [a.kind if not a.name else f"{a.kind}:{a.name}"
                 for a in info.auth]
        lines.append("    credentials    : " + " → ".join(steps)
                     + "   (first source that yields a credential wins)")
    ctx_env = sorted(n for n in EV
                     if EV[n].category == f"provider:{d}"
                     and n.endswith("CONTEXT_LENGTH"))
    tail = f" → env {ctx_env[0]}" if ctx_env else ""
    lines.append(
        f"    context window : catalog/endpoint detect → profile knob "
        f"'context_length'{tail} → else error  (detect WINS over the knob)")
    if info.knobs and any(any(k.name == "modalities" for k in l.knobs)
                          for l in info.knobs.layers):
        lines.append(
            "    modalities     : catalog detect → profile knob 'modalities' "
            "→ static table → text floor")
    if info.knobs and any(l.layer in ("api_params", "framework_overrides")
                          for l in info.knobs.layers):
        lines.append(
            "    layered knobs  : layer dict → flat key (deprecated, warns) "
            "→ default")
    return lines


def provider(name: str) -> Rendered:
    info = introspect.resolve_provider(name)
    if info is None:
        return ({"error": f"unknown provider {name!r}"},
                f"unknown provider {name!r} — see `explain providers`")
    caps = info.capabilities.as_dict() if info.capabilities else {}
    knobs = info.knobs.as_dict() if info.knobs else {}
    res = _resolution_order(info, introspect.env_vars())
    data = {"provider": info.dir_name, "capabilities": caps,
            "quirks": sorted(info.quirks), "knobs": knobs,
            "auth": [{"kind": a.kind, "name": a.name, "note": a.note}
                     for a in info.auth],
            "resolution_order": res}

    lines = [f"provider: {info.dir_name}"]
    lines.append("  capabilities: "
                 + ", ".join(k for k, v in caps.items() if v) or "  (none)")
    lines.append("  quirks: " + (", ".join(sorted(info.quirks)) or "(none)"))
    lines.append("  resolution order:")
    lines.extend(res)
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


# ----------------------------------------------------------------- profile

def profile() -> Rendered:
    """The ``SubagentProfile`` schema — every knob a profile author can set.

    Surfaces the security knobs an author would otherwise have to dig out of
    ``config.py``: ``apparmor`` (opt into per-session confinement) and
    ``apparmor_fragments`` (which client-side ``.rules`` fragments compose the
    AppArmor policy).
    """
    PF = introspect.profile_schema()
    data = [{"name": f.name, "type": f.type, "default": f.default,
             "description": f.description, "allowed": f.allowed} for f in PF]
    lines = ["profile schema  (.jaato/profiles/<set>/<agent>.yaml fields):"]
    for f in PF:
        if f.default == "<required>":
            tail = "  (required)"
        elif f.default in (None, "", [], {}, set()):
            tail = ""
        else:
            tail = f"  (default {f.default!r})"
        lines.append(f"  {f.name:22} {f.type:26}{tail}")
        if f.allowed:
            lines.append(f"      allowed → {f.allowed}")
        if f.description:
            lines.append(f"      {f.description}")
    lines.append(
        "\n  AppArmor — add client-side extra rules via the profile:\n"
        "    apparmor: true              opt the session into kernel-enforced confinement\n"
        "    apparmor_fragments:         compose only these .rules fragments (by basename), from the\n"
        "      - my_extra_rules          search path ~/.jaato/apparmor-fragments/ and\n"
        "                                <workspace>/.jaato/apparmor-fragments/ (+ the .cache/ layer).\n"
        "    drop <name>.rules in that dir, then list <name>.  null = ALL fragments; [] = none.")
    return data, "\n".join(lines)


def paths() -> Rendered:
    """The path & isolation model — daemon-global ``~/.jaato`` vs per-session
    workspace + ``config_root``.

    Authored to head off the common "override ``$HOME`` to isolate a test run"
    mistake: jaato keeps ``~/.jaato`` deliberately daemon-global (creds + the
    auto-installed reactors live there), and isolates PER SESSION at the
    workspace / ``config_root`` layer — not at ``$HOME``.
    """
    data = {
        "daemon_global": {
            "root": "~/.jaato/  (HOME-based, resolved via Path.home())",
            "holds": ["reactors/<name>.json", "scripts/<name>.py",
                      "<provider>_auth.json", "ws.token"],
            "scope": "shared across every session on the daemon",
            "note": "do NOT override $HOME to isolate a run",
        },
        "per_session": {
            "root": "<workspace>/  +  config_root (default <workspace>/.jaato)",
            "holds": [".jaato/profiles/<set>/<agent>.yaml",
                      ".jaato/agents|instructions/", ".jaato/logs/",
                      ".jaato/sessions/"],
            "workspace_root_env": "JAATO_WORKSPACE_ROOT",
            "scope": "the isolation boundary — one per session",
        },
    }
    lines = [
        "paths & isolation model:",
        "",
        "  ~/.jaato/   — DAEMON-GLOBAL (HOME-based, resolved via Path.home()):",
        "    reactors/<name>.json      installed reactor rule fragments (premium)",
        "    scripts/<name>.py         installed reactor scripts",
        "    <provider>_auth.json      provider credentials",
        "    ws.token                  WS bearer token (auto-generated)",
        "    -> SHARED across every session on the daemon.  Do NOT override $HOME",
        "       to 'isolate' a run: creds + the auto-installed reactors live here",
        "       BY DESIGN, and a $HOME override hides them from the daemon.",
        "",
        "  <workspace>/   — PER-SESSION (this is the isolation boundary):",
        "    .jaato/profiles/<set>/<agent>.yaml   profiles (resolved under config_root)",
        "    .jaato/agents | instructions/        persona + base instructions",
        "    .jaato/logs/                         per-session logs",
        "    .jaato/sessions/                     persisted session records",
        "    -> Isolate a run with a FRESH workspace dir; its .jaato/ is the",
        "       config_root.  Workspace-scoped tools (file_edit, cli cwd,",
        "       filesystem_query) resolve against JAATO_WORKSPACE_ROOT / the",
        "       per-session workspace, NOT $HOME.",
        "",
        "  config_root   = <workspace>/.jaato by default — the resolution root for",
        "    profiles / instructions / agents.  Override per-profile (config_root:)",
        "    or per-client (working_dir / env_file).",
        "",
        "  TL;DR  ~/.jaato = daemon-global (creds + reactors, shared).  Per-session",
        "  isolation = a fresh workspace + config_root, NEVER a $HOME override.",
    ]
    return data, "\n".join(lines)
