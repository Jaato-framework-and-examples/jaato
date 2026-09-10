"""The ``explain`` verb — renders the introspect core by scope.

Progressive interrogation: an author drills from the overview into a plugin's
tools, a provider's knobs/quirks, the GC strategies, or a workspace's profile
sets — BEFORE committing to a ``new`` build.  Every function returns a
``(structured_dict, text)`` pair so the CLI can emit either ``--json`` (for an
agent) or a human table.  No metadata is computed here — it all comes from
:mod:`introspect`, the single source the validator also reads.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import DISCOVERABILITY_EAGER
from . import archetypes as _archetypes
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
        "archetypes": list(_archetypes.accepted()),
    }
    # Counted, never spelled: a literal here is how the banner came to advertise
    # "4 client archetypes" while `new` accepted six (jaato #716).
    n_arch = len(_archetypes.ARCHETYPES)
    text = (
        "jaato-scaffold — interrogate the installed framework, then build.\n\n"
        f"  {len(P)} providers   {len(PL)} plugins   "
        f"{len(GC)} gc strategies   {n_arch} archetypes\n\n"
        "drill down:\n"
        "  jaato-scaffold explain plugins\n"
        "  jaato-scaffold explain plugin <name>\n"
        "  jaato-scaffold explain commands\n"
        "  jaato-scaffold explain providers\n"
        "  jaato-scaffold explain provider <name>\n"
        "  jaato-scaffold explain gc\n"
        "  jaato-scaffold explain transports\n"
        "  jaato-scaffold explain clients\n"
        "  jaato-scaffold explain runtime\n"
        "  jaato-scaffold explain tiers\n"
        "  jaato-scaffold explain sets [--workspace DIR]\n"
        "  jaato-scaffold explain profile           # a session's CAPABILITIES\n"
        "  jaato-scaffold explain agents            # ... and its PERSONA\n"
        "  jaato-scaffold explain services          # named HTTP APIs (.jaato/services/)\n"
        "  jaato-scaffold explain paths\n"
        "  jaato-scaffold explain prefetch\n"
        "  jaato-scaffold explain completion       # the OUTPUT-side hook\n"
        "  jaato-scaffold explain archetypes        # what `new` WRITES\n"
        "  jaato-scaffold explain archetype <name>\n"
        "\n"
        "  jaato-scaffold explain integrations      # tools jaato can wire into\n"
        "\n"
        "append `dependencies` (or `deps`) to ANY of the above — it is a facet of\n"
        "every scope, not a scope of its own.  Derived by parsing what is installed,\n"
        "never from a table that could be wrong:\n"
        "  jaato-scaffold explain dependencies            # distributions, skew, extras\n"
        "  jaato-scaffold explain provider <name> deps    # what its code imports\n"
        "  jaato-scaffold explain plugin <name> deps\n"
        "a named unit is read as its own source PLUS the shared machinery it\n"
        "imports, and a MISSING package is printed with the extra declaring it.\n"
        "\n"
        "integrations ship WITH this build, so an installed copy cannot describe a\n"
        "different framework than the one running:\n"
        "  jaato-scaffold integration               # list them + their state\n"
        "  jaato-scaffold integration claude-code   # -> ~/.claude/skills/ (all repos)\n"
        "  jaato-doctor                             # says when one has gone stale\n"
    )
    return data, text


# ------------------------------------------------------------- transports

def transports() -> Rendered:
    """The three client transports + the daemon flags / auth that gate them.

    Source of truth for "how does a client run an agent": in-process (embedded,
    no daemon), IPC (local daemon), and WebSocket (remote daemon). The Python
    SDK ships a client for ALL THREE — and the same convenience facade
    (``jaato.session(mode=...)`` -> ``Session.ask`` / ``.complete`` /
    ``.stream``) rides on each, so ``mode`` is the only thing that changes.
    """
    data = {
        "in_process": {
            "sdk": "jaato (Python) — jaato.session(mode='in_process') / jaato.InProcessClient",
            "scope": "embedded — no daemon, no socket; the agent runs in your process",
            "daemon_flags": [],
            "auth": "n/a — in-process, no wire",
        },
        "ipc": {
            "sdk": "jaato-sdk (Python) — jaato_sdk.IPCClient / jaato.session(mode='ipc')",
            "scope": "local (same host / container) daemon",
            "daemon_flags": ["--ipc-socket PATH", "--socket-mode MODE (default 660)"],
            "auth": "none — any principal that can open the socket drives the agent",
        },
        "websocket": {
            "sdk": "jaato-sdk (Python) — jaato_sdk.WSClient / jaato.session(mode='ws'); "
                   "also jaato-sdk-ts (TypeScript) / browser client jaato-web",
            "scope": "remote / browser daemon",
            "daemon_flags": ["--web-socket [HOST:]PORT", "--ws-token TOKEN",
                             "--ws-token-file PATH", "--ws-unsafe-no-auth"],
            "token_default": "~/.jaato/ws.token (auto-generated, 0600, on first WS start)",
            "auth": "bearer token (required unless --ws-unsafe-no-auth)",
            "client_auth": ["Authorization: Bearer <token>  (header)",
                            "?token=<token>  (query param — browsers)"],
            "bad_token": "WS close code 1008",
            "preflight": "jaato-doctor --web-socket [host:]port",
            "python_extra": "pip install 'jaato-sdk[ws]'  (the websockets dependency)",
        },
    }
    text = (
        "client transports — one facade (jaato.session(mode=...)), three modes\n"
        "  ----------------------------------------------------------------\n"
        "  in_process            IPC (Unix socket)        WebSocket\n"
        "  - embedded, no daemon - local daemon           - remote daemon / browser\n"
        "  - InProcessClient     - IPCClient              - WSClient (Python)\n"
        "                                                   + TS SDK / jaato-web\n"
        "  - n/a (no wire)       - unauthenticated        - bearer-token authenticated\n"
        "                          (socket-mode 660)\n\n"
        "  Python ships a client for ALL THREE; the SAME Session.ask/.complete/\n"
        "  .stream facade rides on each.  mode is the only variable.\n\n"
        "daemon flags (ipc / ws only — in_process needs no daemon):\n"
        "  IPC:  --ipc-socket PATH   [--socket-mode 660]\n"
        "  WS:   --web-socket [HOST:]PORT\n"
        "        --ws-token TOKEN | --ws-token-file PATH | --ws-unsafe-no-auth\n"
        "        (no token flag → daemon auto-generates ~/.jaato/ws.token, 0600)\n\n"
        "WS auth contract — a client presents ONE of:\n"
        "  Authorization: Bearer <token>   (header — SDK / proxies / curl)\n"
        "  ?token=<token>                  (query — browsers can't set headers on\n"
        "                                   new WebSocket())\n"
        "  bad token → WS close 1008.  Daemon stores only the SHA-256 digest and\n"
        "  compares with hmac.compare_digest.\n\n"
        "preflight the WS daemon side (port + token file + auth mode):\n"
        "  jaato-doctor --web-socket [host:]port\n\n"
        "scaffold any transport:\n"
        "  jaato-scaffold new client --transport in_process ...   # embedded\n"
        "  jaato-scaffold new client --transport ipc ...          # local daemon (default)\n"
        "  jaato-scaffold new client --transport ws ...           # remote daemon\n"
        "The Python WS client needs the extra:  pip install 'jaato-sdk[ws]'\n"
    )
    return data, text


# --------------------------------------------------------------- clients

def clients() -> Rendered:
    """The Python SDK client classes — one per transport (+ recovery) — and when
    to use each.

    All expose the same facade-client contract, so the convenience
    ``Session`` (``ask`` / ``complete`` / ``stream``) and ``jaato.session(
    mode=...)`` ride on any of them.  ``new client --transport <mode>`` emits the
    matching one; ``--recoverable`` upgrades a daemon transport to its
    auto-reconnect client (``ipc`` → ``IPCRecoveryClient``, ``ws`` →
    ``WSRecoveryClient``).
    """
    data = {
        "in_process_client": {
            "class": "jaato.InProcessClient  (jaato.session(mode='in_process'))",
            "transport": "embedded — no daemon, no socket",
            "use_for": ["lowest latency / simplest deploy", "no daemon to run",
                        "develop embedded, deploy behind a daemon later"],
            "scaffold": "jaato-scaffold new client --transport in_process ...",
        },
        "ipc_client": {
            "class": "jaato_sdk.IPCClient  (jaato.session(mode='ipc'))",
            "transport": "local daemon (Unix socket)",
            "recovery": "none — a dropped connection ends the client",
            "use_for": ["short-lived / one-shot", "a single send_message",
                        "a scripted run that exits"],
            "scaffold": "jaato-scaffold new client --transport ipc ...",
        },
        "ws_client": {
            "class": "jaato_sdk.WSClient  (jaato.session(mode='ws'))",
            "transport": "remote daemon (ws:// / wss://)",
            "recovery": "none — use --recoverable (WSRecoveryClient) to survive drops",
            "auth": "bearer token (url= + token=)",
            "tls": "wss:// self-signed/dev cert -> --ca <bundle> (scoped ca=, "
                   "loaded into a per-connection SSLContext, never os.environ)",
            "extra": "pip install 'jaato-sdk[ws]'",
            "use_for": ["a daemon on another host", "browser-reachable endpoint"],
            "scaffold": "jaato-scaffold new client --transport ws ...",
        },
        "ipc_recovery_client": {
            "class": "jaato_sdk.IPCRecoveryClient",
            "transport": "local daemon (Unix socket), auto-reconnecting",
            "recovery": ("auto-reconnect state machine "
                         "(DISCONNECTED→CONNECTING→CONNECTED→RECONNECTING→CLOSED); "
                         "on_status_change callback; IncompatibleServerError is "
                         "permanent (no retry)"),
            "use_for": ["long-lived / resilient",
                        "TUI / observer / cascade driver",
                        "must survive a daemon restart "
                        "(per-run jaato-server --stop + autostart)"],
            "scaffold": "jaato-scaffold new client --transport ipc --recoverable ...",
        },
        "ws_recovery_client": {
            "class": "jaato_sdk.WSRecoveryClient  "
                     "(jaato.session(mode='ws', recovery=True))",
            "transport": "remote daemon (ws:// / wss://), auto-reconnecting",
            "auth": "bearer token (url= + token=)",
            "tls": "wss:// self-signed/dev cert -> --ca <bundle> (scoped ca=, "
                   "never os.environ — so it can't leak into a restarted daemon)",
            "recovery": ("same auto-reconnect state machine as IPCRecoveryClient, "
                         "over the WebSocket; reattaches via the transport-agnostic "
                         "server replay; on_status_change callback"),
            "use_for": ["long-lived / resilient over a REMOTE daemon",
                        "must survive a daemon restart or a dropped WebSocket"],
            "scaffold": "jaato-scaffold new client --transport ws --recoverable ...",
        },
    }
    text = (
        "Python SDK clients — one facade, one client per transport\n"
        "  ----------------------------------------------------------------\n"
        "  InProcessClient     IPCClient          WSClient         IPCRecoveryClient\n"
        "  - embedded          - local daemon     - remote daemon  - local daemon\n"
        "    (no daemon)         (Unix socket)      (ws:// / wss://)  + auto-reconnect\n"
        "  - lowest latency    - no reconnect     - bearer token   - on_status_change\n"
        "  - simplest deploy   - one-shot/script  - jaato-sdk[ws]    survives restarts\n\n"
        "  All expose the same Session.ask/.complete/.stream facade; pick the\n"
        "  transport with jaato.session(mode='in_process'|'ipc'|'ws') and add\n"
        "  recovery=True on a daemon mode for auto-reconnect.  Use --recoverable\n"
        "  for anything long-lived (IPCRecoveryClient on ipc, WSRecoveryClient on\n"
        "  ws) — it rides through a per-run `jaato-server --stop` + autostart; on a\n"
        "  permanent IncompatibleServerError it stops rather than looping.\n\n"
        "scaffold any of them (the flags work for every archetype):\n"
        "  jaato-scaffold new client --transport in_process ...      # InProcessClient\n"
        "  jaato-scaffold new client --transport ipc ...             # IPCClient (default)\n"
        "  jaato-scaffold new client --transport ipc --recoverable . # IPCRecoveryClient\n"
        "  jaato-scaffold new client --transport ws ...              # WSClient\n"
        "  jaato-scaffold new client --transport ws --recoverable .  # WSRecoveryClient\n"
    )
    return data, text


# ------------------------------------------------------------ archetypes

def archetypes() -> Rendered:
    """What ``new`` PRODUCES, one line per archetype.

    ``explain``'s other scopes document the framework's inputs; this one
    documents the generator's output contract, so a reader can decide whether
    to run ``new`` without first reverse-engineering the templates.  Sourced
    from :mod:`archetypes`, whose registry is guarded to cover every archetype
    ``new`` accepts.
    """
    # The WHOLE registry, ordered with the default first — never
    # profile-set-plus-the-client-templates, which silently omits any
    # archetype that is neither (the processor generator is the first).
    # A hand-kept subset standing in for the registry is how `explain` came
    # to advertise four archetypes out of six (jaato #716).
    docs = [_archetypes.ARCHETYPES[_archetypes.PROFILE_SET]] + [
        _archetypes.ARCHETYPES[n] for n in sorted(_archetypes.ARCHETYPES)
        if n != _archetypes.PROFILE_SET]
    data = {
        d.name: {
            "kind": d.kind,
            "summary": d.summary,
            "aliases": list(d.aliases),
            "requires": list(d.requires),
            "writes": [e.render_path(archetype=d.name, set="<set>",
                                     agent="<agent>", name="<name>")
                       for e in d.writes],
        }
        for d in docs
    }
    import textwrap
    lines = ["`jaato-scaffold new <archetype>` — what each one WRITES into your",
             "workspace.  Every archetype re-checks its own output — see each",
             "one's `check` under `explain archetype <name>`.",
             ""]
    for d in docs:
        tags = []
        if d.name == _archetypes.PROFILE_SET:
            tags.append("default")
        tags += [f"also `{a}`" for a in d.aliases]
        tag = f"   [{', '.join(tags)}]" if tags else ""
        lines.append(f"  {d.name}{tag}")
        lines += textwrap.wrap(d.summary, width=76,
                               initial_indent="      ", subsequent_indent="      ")
        paths = ", ".join(e.render_path(archetype=d.name, set="<set>",
                                        agent="<agent>", name="<name>")
                          for e in d.writes)
        lines += textwrap.wrap("writes: " + paths, width=76,
                               initial_indent="      ", subsequent_indent="              ")
        lines.append(f"      needs:  {' '.join(d.requires)}")
        lines.append("")
    lines += [
        "drill down — the file tree, what is in each file, and which parts are",
        "placeholders you must edit:",
        "  jaato-scaffold explain archetype <name>",
        "",
        "or ask what YOUR flags would write, without writing it:",
        "  jaato-scaffold new <name> --workspace DIR ... --dry-run",
    ]
    return data, "\n".join(lines) + "\n"


def archetype(name: str) -> Rendered:
    """One archetype's full output contract — files, contents, flags, checks.

    Answers the question ``new --help`` does not: what lands in the workspace,
    what is inside it, which parts are placeholders versus generated-and-correct,
    and how each flag changes that.
    """
    doc = _archetypes.resolve(name)
    if doc is None:
        known = ", ".join(_archetypes.accepted())
        return ({"error": f"unknown archetype {name!r}", "known": list(_archetypes.accepted())},
                f"unknown archetype {name!r} — one of: {known}")

    subs = dict(archetype=doc.name,
                **{"set": "<set>", "agent": "<agent>", "name": "<name>"})
    data = {
        "name": doc.name,
        "kind": doc.kind,
        "summary": doc.summary,
        "aliases": list(doc.aliases),
        "requires": list(doc.requires),
        "writes": [
            {
                "path": e.render_path(**subs),
                "what": e.what,
                "status": e.status,
                "detail": list(e.detail),
                "when": e.when,
            }
            for e in doc.writes
        ],
        "flags": [{"flag": f, "effect": eff} for f, eff in doc.flags],
        "edit_before_running": list(doc.edit_before_running),
        "generated_correct": list(doc.generated_correct),
        "check": doc.check,
        "next_steps": [_fill_subs(n, subs) for n in doc.next_steps],
    }

    import textwrap
    alias = (f"  (also `{'`, `'.join(doc.aliases)}`)" if doc.aliases else "")
    out = [f"jaato-scaffold new {doc.name}{alias}"]
    out += textwrap.wrap(doc.summary, width=78,
                         initial_indent="  ", subsequent_indent="  ")
    out += ["", f"required: {' '.join(doc.requires)}", "",
            "writes (relative to --workspace):"]
    out += _render_writes(doc, subs)
    out.append("")
    out.append("status: generated = correct as emitted · fill-in = a blank you "
               "must complete")
    out.append("        edit = a worked example to replace · merged = an "
               "existing file is appended to")
    if doc.edit_before_running:
        out += ["", "you must edit before it runs:"]
        for d in doc.edit_before_running:
            out += _wrap_bullet(d, indent=2)
    if doc.generated_correct:
        out += ["", "generated and correct — the recipe this archetype exists to "
                    "carry:"]
        for d in doc.generated_correct:
            out += _wrap_bullet(d, indent=2)
    if doc.flags:
        out += ["", "flags that change the output:"]
        for f, eff in doc.flags:
            out.append(f"  {f}")
            out += _wrap_bullet(eff, indent=6, glyph=" ")
    out += ["", "self-check `new` runs on its own output:"]
    out += _wrap_bullet(doc.check, indent=2, glyph=" ")
    out += ["", "next:"]
    out += [f"  {_fill_subs(n, subs)}" for n in doc.next_steps]
    out += ["", "to see the exact tree for YOUR flags without writing it:",
            f"  jaato-scaffold new {doc.name} --workspace DIR ... --dry-run"]
    return data, "\n".join(out) + "\n"


def _render_writes(doc, subs: Dict[str, str]) -> List[str]:
    """The ``writes:`` block — one stanza per emitted file.

    Each stanza is the path, its ownership status, the condition under which it
    is written, its one-line purpose, and the bullets describing its contents.
    """
    import textwrap
    out: List[str] = []
    for e in doc.writes:
        when = f"   — {e.when}" if e.when else ""
        out.append(f"  {e.render_path(**subs)}   [{e.status}]{when}")
        out += textwrap.wrap(e.what, width=78,
                             initial_indent="      ", subsequent_indent="      ")
        for d in e.detail:
            out += _wrap_bullet(d, indent=6)
    return out


def _fill_subs(text: str, subs: Dict[str, str]) -> str:
    """Fill ``{archetype}`` / ``{set}`` / ``{agent}`` in a rendered line."""
    for k, v in subs.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def _wrap_bullet(text: str, indent: int, glyph: str = "-") -> List[str]:
    """Wrap one bullet to ~78 columns, hanging-indented under its glyph."""
    import textwrap
    pad = " " * indent
    return textwrap.wrap(text, width=78, initial_indent=f"{pad}{glyph} ",
                         subsequent_indent=f"{pad}  ") or [f"{pad}{glyph}"]


# --------------------------------------------------------------- runtime

# ``runtime_limits`` fields, by the layer that ENFORCES them.  The split is
# the class docstring's own; naming it here rather than re-deriving keeps
# one description of each field and one place to add the next one.
_RUNTIME_LIMIT_FIELDS = (
    ("memory_max_mb", "kernel", "cgroup v2 memory.max — OOM-kills the slice"),
    ("pids_max", "kernel", "cgroup v2 pids.max — bounds fork/thread count"),
    ("cpu_weight", "kernel", "cgroup v2 cpu.weight — fair-share, 1..10000"),
    ("tool_timeout_seconds", "cli/shell",
     "subprocess.run(timeout=) per tool call"),
    ("max_output_bytes", "cli/shell", "truncates captured stdout/stderr"),
    ("max_parallel_tools", "session",
     "width of the tool thread pool (and of the token-count fan-out)"),
)


def _runtime_limits_report() -> Dict[str, Any]:
    """Machine-readable ``runtime_limits`` summary for ``explain runtime``.

    Reads :mod:`shared.runtime_limits` — the dataclass's own fields, the
    framework's concurrency default, and the isolated-subagent defaults —
    so an added field or a changed default shows up here without an edit.

    Returns:
        ``{"fields": [{name, layer, effective, note}, ...],
           "isolated_subagent_defaults": {...}}``.  ``effective`` is what
        applies when NO profile declares the field.
    """
    from shared import runtime_limits as rl

    unset = "host default (no limit)"
    effective = {
        "max_parallel_tools": (
            f"{rl.DEFAULT_MAX_PARALLEL_TOOLS} (framework default)"
        ),
    }
    iso = rl.ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS
    return {
        "fields": [
            {
                "name": name,
                "layer": layer,
                "effective": effective.get(name, unset),
                "note": note,
            }
            for name, layer, note in _RUNTIME_LIMIT_FIELDS
        ],
        "isolated_subagent_defaults": {
            name: getattr(iso, name)
            for name, _layer, _note in _RUNTIME_LIMIT_FIELDS
        },
        "inheritance": {
            "ceilings": "child REPLACES the block (parents must agree)",
            "max_parallel_tools": "MIN across every layer that declares it",
        },
    }


def _runtime_limits_lines(report: Dict[str, Any]) -> List[str]:
    """Render :func:`_runtime_limits_report` as the ``explain runtime`` block."""
    lines = [
        "RUNTIME LIMITS  (profile `runtime_limits:` — what a session may CONSUME)",
        "  field                 enforced by  effective when unset",
    ]
    for row in report["fields"]:
        lines.append(
            f"  {row['name']:<21} {row['layer']:<12} {row['effective']}"
        )
        lines.append(f"  {'':<21} {'':<12} {row['note']}")
    iso = report["isolated_subagent_defaults"]
    declared = ", ".join(
        f"{k}={v}" for k, v in iso.items() if v is not None
    )
    lines.append(
        f"  agent_params.isolated=true fills any field the profile omits: {declared}"
    )
    lines.append(
        "  inheritance: the ceilings are child-REPLACES (parents must agree);"
    )
    lines.append(
        "               max_parallel_tools is MIN across every layer that sets it,"
    )
    lines.append(
        "               so a child may only ever narrow the pool it was spawned under."
    )
    return lines


def runtime() -> Rendered:
    """How a session runs + how to DEBUG it — entities, the workspace flow, the
    log map, and the one-command session diagnostic.

    Also reports the ``runtime_limits`` block — what a session may CONSUME —
    introspected from :mod:`shared.runtime_limits` (fields, enforcement layer,
    framework defaults) so it tracks the installed framework rather than a
    prose copy that drifts.

    Curated (the runtime architecture is not in the plugin registry).  Pairs with
    ``jaato-doctor --session <id>``, which applies this map to a live session.
    """
    limits = _runtime_limits_report()
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
        "runtime_limits": limits,
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
        "     (PASS=<ws>) or got workspace=none (FAIL + the fix), plus the log map.\n\n"
        + "\n".join(_runtime_limits_lines(limits)) + "\n"
    )
    return data, text


# ------------------------------------------------------------------- tiers

def tiers() -> Rendered:
    """Model tiers — multi-model sessions: cognitive roles + modality (vision)
    roles, switched mid-session via ``enter_tier``.  V2 (#354): tiers may span
    PROVIDERS.  #831: a deployment may NAME its own tiers.  Introspects
    ``shared.model_tiers`` (CANONICAL_TIER_NAMES / TIER_NAME_PATTERN /
    MAX_DECLARED_TIERS / RESERVED_KEYS) so it tracks the installed framework.
    """
    from shared import model_tiers as mt
    valid = sorted(mt.CANONICAL_TIER_NAMES)
    reserved = sorted(mt.RESERVED_KEYS)
    modalities = sorted(mt.VALID_TIER_MODALITIES)
    directions = sorted(mt.VALID_MODALITY_DIRECTIONS)
    data = {
        "canonical_tier_names": valid,
        # Kept under the old key too: this dict is a machine-readable
        # surface other tools read, and dropping a key is a breaking change
        # where adding one is not.
        "tier_names": valid,
        "free_tier_names": (
            "a profile may declare tiers under names of its OWN — 'coder', "
            "'reviewer', 'researcher' — matching "
            + mt.TIER_NAME_PATTERN + " (lowercase, digits, underscores, "
            "starting with a letter, 2-32 chars).  a free name MUST carry a "
            "'description': the four canonical names come with framework "
            "prose, a name the framework has never heard of does not, and "
            "'routes this session to <model>' is not a reason for the model "
            "to enter a tier.  free names sort after the canonical ones in "
            "the enter_tier schema, so adding one never reorders the "
            "prompt-cache prefix ahead of them."
        ),
        "max_tiers": mt.MAX_DECLARED_TIERS,
        "reserved_keys": reserved,
        "modalities": modalities,
        "modality_directions": directions,
        "shape": "model_tiers: { <tier>: <model-str> | "
                 "{model, provider, description, modalities}, "
                 "initial: <tier>, fallback: <tier> }",
        "switching": "the MODEL calls enter_tier('<tier>') mid-session; the "
                     "active tier selects the model (and, V2, the provider); "
                     "history is preserved across the switch",
        "vision": "'vision' is the ONE tier name with a built-in meaning: it "
                  "implies modalities {image: inbound}, so profiles written "
                  "before the modalities key are unchanged.  Nothing else is "
                  "name-based — the gate and the startup check resolve a tier "
                  "BY ROLE, so an image tier may be called anything as long as "
                  "it declares the role.  content the ACTIVE model cannot "
                  "accept trips the content gate: a synthetic "
                  "'enter_tier(\"<tier>\") first' naming whichever tier "
                  "declares the role, which the agent self-corrects on.  "
                  "user-message images ride the attachment ferry (#353): SDK "
                  "send_message(attachments=[path | {mime_type,data,display_name}])",
        "description": "each tier entry may carry a 'description' — prose the "
                       "MODEL reads as that tier's bullet in the enter_tier "
                       "tool schema (default: the framework's own wording for "
                       "the name).  the enter_tier enum lists only the tiers "
                       "the profile DECLARES, so an undeclared tier is never "
                       "advertised.  read once when the tool schema is built "
                       "(it sits in the prompt-cache prefix), so a budget "
                       "degrade rung cannot set it.",
        "modalities_key": "a tier declares which non-text modalities it "
                          "fills (" + ", ".join(modalities) + ") and in which "
                          "direction (" + ", ".join(directions) + ").  list "
                          "form [image] is sugar for inbound.  the content "
                          "gate and the startup capability check resolve the "
                          "tier BY ROLE, not by the name 'vision' — which "
                          "still implies image inbound so pre-existing "
                          "profiles are unchanged.  this DECLARES a role and "
                          "is VERIFIED (a tier whose model can't accept the "
                          "modality fails loud at connect); the opposite of "
                          "plugin_configs.<provider>.modalities, which "
                          "ASSERTS capability to correct catalog detection.",
        "outbound_is_inert": "an OUTBOUND role parses and is stored, but it "
                             "is INERT unless the named provider declares "
                             "`output_media` — the adapters that decode "
                             "model-generated media off the stream "
                             "(openrouter + the _openai_compat five "
                             "today).  validate warns only when it does "
                             "not, so a working speaking tier is not told "
                             "it does nothing.  see "
                             "docs/design/binary-media-chunks.md.",
        "inbound_is_inert_when_unmarshalled": "an INBOUND role is INERT in "
                             "the same way when the provider does not "
                             "declare the capability that carries it "
                             "(image -> user_message_images, file -> "
                             "pdf_input, audio -> audio_input): the model "
                             "may well accept the content, but the "
                             "converter withholds it with a note.  that "
                             "asymmetry — a catalog saying `audio` and a "
                             "converter with no branch for it — was #830.  "
                             "a kind with no capability column (video) is "
                             "left unchecked rather than declared inert.",
        "cross_provider": "V2 (#354): tiers may declare DIFFERENT providers; "
                          "switch_tier swaps to a cached per-tier provider "
                          "instance (history is provider-neutral; switch-back is "
                          "O(1)). e.g. a cheap zhipuai text executor + a gemini/"
                          "OpenRouter vision tier in ONE profile.",
    }
    text = (
        "jaato model tiers — multi-model sessions (cognitive + modality roles)\n"
        "  ----------------------------------------------------------------\n"
        f"CANONICAL     {', '.join(valid)}\n"
        f"              (the names the FRAMEWORK has prose for; 'vision' also\n"
        f"               implies modalities {{image: inbound}})\n"
        f"YOUR OWN      any name matching {mt.TIER_NAME_PATTERN} -- 'coder',\n"
        f"              'reviewer', 'researcher'.  a free name REQUIRES a\n"
        f"              'description' (the framework has none for it) and has\n"
        f"              no JAATO_TIER_* env spelling.  at most {mt.MAX_DECLARED_TIERS}\n"
        f"              tiers per session -- each one is a bullet + an enum entry\n"
        f"              in the prompt-cache prefix, paid on EVERY request.\n"
        f"CONTROL KEYS {', '.join(reserved)}  (reserved: initial tier + fallback)\n\n"
        "SHAPE  (in a profile)\n"
        "  model_tiers:\n"
        "    <tier>:   <model-string>   OR\n"
        "              {model: <m>, provider: <p>, description: <prose>,\n"
        "               modalities: {<kind>: <direction>}}\n"
        "    initial:  <tier>           # the tier a session starts in\n"
        "    fallback: <tier>           # when enter_tier names an undeclared tier\n\n"
        "EXIT  (what ends a tier's turn at the wheel)\n"
        "  exit_on: switch      (default) stays until something switches it\n"
        "  exit_on: completion  entered, does ONE completion, left again --\n"
        "                       the framework returns to the calling tier and\n"
        "                       reports what the delegate produced, so the\n"
        "                       model in the specialist tier does NOTHING to\n"
        "                       hand back.  use it when that model is the one\n"
        "                       least able to: a speaking tier measured over\n"
        "                       four runs never returned on its own.\n\n"
        "SWITCHING\n"
        "  the MODEL calls enter_tier('<tier>') mid-session; the active tier picks\n"
        "  the model (and, V2, the provider).  conversation history is preserved.\n\n"
        "DESCRIPTION  (what the model reads)\n"
        "  the enter_tier tool advertises ONLY the tiers this profile declares,\n"
        "  each with a bullet.  the bullet is the tier's 'description' when set,\n"
        "  else the framework's own wording for that name — so a ladder whose\n"
        "  'executor' means something specific to your deployment can say so,\n"
        "  and a ladder that would rather call it 'coder' can do THAT instead\n"
        "  (a deployment-named tier must carry the description).\n"
        "  it is read once, when the tool schema is built: the tool block sits in\n"
        "  the prompt-cache prefix, so a budget degrade rung may NOT set one.\n\n"
        "MODALITY ROLES  (which tier can SEE what)\n"
        f"  a tier may declare  modalities: {{<kind>: <direction>}}  where kind is\n"
        f"  one of {', '.join(modalities)} and direction is one of\n"
        f"  {', '.join(directions)}.  the list form  modalities: [image]  is sugar\n"
        "  for inbound.  content of a modality the ACTIVE model can't accept is\n"
        "  withheld by the content gate, which names a tier that declares the role:\n"
        "  a synthetic 'enter_tier(\"<tier>\") first' the agent self-corrects on.\n"
        "  the tier is found BY ROLE, so an image tier need not be called 'vision'\n"
        "  — though that name still implies ['image'], so older profiles are\n"
        "  unchanged.  a tier whose model cannot accept a role it declares fails\n"
        "  LOUD at connect, not at the first image.\n"
        "  NOTE this DECLARES a role and is VERIFIED — the opposite direction from\n"
        "  plugin_configs.<provider>.modalities, which ASSERTS what a model\n"
        "  supports to correct catalog detection.\n"
        "  OUTBOUND roles are DELIVERED by any provider declaring `output_media`\n"
        "  (openrouter + the _openai_compat five today); `validate` warns only\n"
        "  when the named provider does not.  see docs/design/binary-media-chunks.md.\n"        "  INBOUND roles are MARSHALLED by any provider declaring the matching\n"
        "  capability (image -> user_message_images, file -> pdf_input, audio ->\n"
        "  audio_input); `validate` warns when the named provider does not, since\n"
        "  the model may accept content its own converter then withholds (#830).\n"
        "  user-message images ride the attachment ferry — SDK\n"
        "  send_message(attachments=...).\n\n"
        "CROSS-PROVIDER  (V2)\n"
        "  tiers may declare DIFFERENT providers — switch_tier swaps to a cached\n"
        "  per-tier provider instance (history is provider-neutral; switch-back is\n"
        "  O(1)).  e.g. a cheap zhipuai text executor + a gemini/OpenRouter vision\n"
        "  tier in ONE profile (no need to relocate the whole profile).\n"
    )
    return data, text


# ----------------------------------------------------------------- plugins

def _tier_missing_note(PL: Dict[str, Any]) -> str:
    """The footer explaining ``[no PLUGIN_TIER]``, or ``""`` when clean.

    Printed only when at least one listed plugin lacks the annotation,
    so a healthy workspace's table is unchanged.  The text names the fix
    rather than the rule, because the author reading it has just been
    told their plugin will not load and needs the next action, not the
    history of §3.3.5.

    Split out of :func:`plugins` deliberately: that function is near the
    complexity ceiling and radon counts the generator below as a
    decision point (see ``test_cyclomatic_complexity_audit``).
    """
    if not any(getattr(pi, "tier_missing", False) for pi in PL.values()):
        return ""
    return (
        "\n  `[no PLUGIN_TIER - will not load in the runner]` marks a "
        "plugin discovered\n  here but EXCLUDED by the runner's tier "
        "filter: sessions naming it come up\n  without its tools. Add "
        "`PLUGIN_TIER = \"runner\"` to the plugin package's\n  "
        "__init__.py (\"daemon\" for daemon-side only, "
        "\"daemon_callable\" for both)."
    )


def plugins() -> Rendered:
    PL = introspect.plugins()
    rows = []
    data = {}
    for name in sorted(PL):
        pi = PL[name]
        core = sum(1 for t in pi.tools if t.discoverability == DISCOVERABILITY_EAGER)
        disc = len(pi.tools) - core
        data[name] = {
            "kind": pi.kind, "tier": pi.tier,
            "tools": len(pi.tools), "core": core, "dynamic": pi.dynamic,
            # Provenance (issue #684) — which distribution supplied this
            # plugin, and whether that is the framework itself.
            "source": pi.source, "builtin": pi.builtin,
            # Tier reachability (issue #917) — this walk discovers with
            # NO tier filter, so an unannotated plugin appears here and
            # is nonetheless dropped by the runner.  A consumer reading
            # ``--json`` needs the same fact the table renders.
            "tier_missing": pi.tier_missing,
        }
        tools = "dynamic" if pi.dynamic else f"{len(pi.tools)} ({core} core/{disc} disc)"
        # Built-ins render bare; anything else is named, so a plugin
        # supplied by an installed distribution stands out in the table.
        src = "" if pi.builtin else f"   <- {pi.source}"
        # An unannotated plugin is listed but WILL NOT LOAD in the
        # runner, so the row has to say so — silently listing it is the
        # defect (#917): this is the surface an author consults to
        # confirm the plugin is wired, and it was answering "yes" for a
        # plugin the session would come up without.
        warn = "  [no PLUGIN_TIER - will not load in the runner]" if pi.tier_missing else ""
        rows.append(
            f"  {name:22} {pi.kind:10} {str(pi.tier or '-'):8} {tools}{src}{warn}"
        )
    text = (f"{'plugin':24}{'kind':12}{'tier':10}tools\n"
            + "  " + "-" * 56 + "\n" + "\n".join(rows)
            + "\n\n  core = in the model's initial schema; disc = deferred "
              "(discoverable via\n  list_tools/get_tool_schemas, or force eager "
              "with `<plugin>(preload)` in a profile)"
            + "\n  `<- dist (module)` marks a plugin supplied by an "
              "installed distribution\n  rather than the built-in "
              "package — see JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST"
            + _tier_missing_note(PL))
    return data, text


def _signature(parameters: "Optional[Dict[str, Any]]") -> str:
    """Render a tool's arguments as ``(a, b=..., ...)``.

    Required arguments bare, optional ones suffixed ``=...`` -- enough to
    check a call site against, without printing a JSON-Schema block into a
    terminal.  The full schema is in ``--json`` for anything that needs to
    compare types.

    ``None`` (schema omitted one) renders as ``(?)``; ``{}`` -- a tool that
    genuinely takes no arguments -- renders as ``()``.  Those are different
    facts and the rendering keeps them apart.
    """
    if parameters is None:
        return "(?)"
    props = parameters.get("properties")
    if not isinstance(props, dict):
        return "()"
    required = set(parameters.get("required") or ())
    return "(" + ", ".join(
        p if p in required else f"{p}=..." for p in props) + ")"


def _spec_summary(spec: "Dict[str, Any]") -> str:
    """One column of type facts for a parameter: its type and its closed set.

    ``enum`` is rendered because it is the difference between a parameter a
    caller can guess and one they cannot: ``{"type": "string"}`` invites any
    string, ``one of: header, query`` is the whole contract.  Union types
    (``["string", "null"]``) render joined rather than as a Python repr.
    """
    if not isinstance(spec, dict):
        return ""
    t = spec.get("type")
    if isinstance(t, list):
        t = "|".join(str(x) for x in t)
    parts = [str(t)] if t else []
    enum = spec.get("enum")
    if isinstance(enum, (list, tuple)) and enum:
        parts.append("one of: " + ", ".join(str(e) for e in enum))
    return "  ".join(parts)


def _nested_params(parameters: "Optional[Dict[str, Any]]") -> List[str]:
    """Expand the parameters a flat signature cannot describe.

    :func:`_signature` renders ``configure_service_auth(service, auth)`` — true,
    and useless, because ``auth`` is an object whose shape IS the tool.  Its
    five accepted forms (``apiKey`` in a header or a query, ``bearer``,
    ``basic``, ``oauth2_client``) and the ``*_env`` field each one wants were
    declared in the schema and rendered nowhere, so the only way to call the
    tool correctly was to read the plugin source — for a page whose entire job
    is to make that unnecessary.  (``--json`` carried the schema all along;
    what was missing was a HUMAN rendering of it.)

    Expanded here: any parameter carrying an ``enum``, and any ``object``
    parameter's own properties.  Deliberately ONE level deep — deeper nesting
    is a schema dump, which is what ``--json`` is for.

    Args:
        parameters: A tool's JSON-Schema ``parameters`` block, or ``None``.

    Returns:
        Indented lines, or ``[]`` when nothing needs expanding (the common
        case: a tool whose arguments are all plain scalars).
    """
    props = (parameters or {}).get("properties")
    if not isinstance(props, dict):
        return []
    required = set((parameters or {}).get("required") or ())
    out: List[str] = []
    for pname, spec in props.items():
        if not isinstance(spec, dict):
            continue
        if isinstance(spec.get("properties"), dict):
            out.extend(_object_param_block(pname, spec, pname in required))
        elif spec.get("enum"):
            out.append(f"           {pname:20} {_spec_summary(spec)}")
    return out


def _object_param_block(pname: str, spec: "Dict[str, Any]",
                        is_required: bool) -> List[str]:
    """One ``object`` parameter's own properties, as an indented block.

    Split from :func:`_nested_params` to keep both under the complexity
    ceiling: radon counts the ``or``-defaults and the width arithmetic here as
    decision points, and the caller is a loop over every parameter.

    ``*`` marks a property required WITHIN the object — a different question
    from whether the object itself is required, which the header states.
    """
    req = "required" if is_required else "optional"
    sub_req = set(spec.get("required") or ())
    out = [f"           {pname} (object, {req}):"]
    for k, ks in spec["properties"].items():
        mark = "*" if k in sub_req else " "
        desc = (ks.get("description") or "") if isinstance(ks, dict) else ""
        tail = f"   {desc}" if desc else ""
        pad = max(0, 18 - len(k))
        out.append(f"             {k}{mark:<2}{'':<{pad}}{_spec_summary(ks)}{tail}")
    if sub_req:
        out.append("             (* = required within this object)")
    return out


def _commands_json(commands: List[Any]) -> List[Dict[str, Any]]:
    """The ``--json`` view of a plugin's user commands.

    A separate function for the same reason as :func:`_command_block`: the
    comprehension this replaces sat inside ``plugin``, which is frozen in the
    complexity baseline, and radon counts a comprehension as a decision point.
    """
    return [{"name": c.name, "description": c.description,
             "share_with_model": c.share_with_model,
             "subcommands": c.subcommands} for c in commands]


def _command_block(commands: List[Any]) -> List[str]:
    """The per-plugin ``commands:`` block for :func:`plugin`.

    Split out rather than inlined because ``plugin`` is over the complexity
    ceiling and frozen in the audit baseline, so added logic belongs in a
    helper (see ``test_cyclomatic_complexity_audit``).

    The heading says *typed directly in the TUI, not via the model* because
    every other block on that page describes MODEL-facing surface (tools,
    plugin_configs); without the qualifier a reader reasonably assumes these
    are tools too.
    """
    if not commands:
        return []
    out = ["  commands (typed directly in the TUI, not via the model):"]
    for c in commands:
        subs = f"  subcommands: {', '.join(c.subcommands)}" if c.subcommands else ""
        out.append(f"    {c.name}{subs}")
        if c.description:
            out.append(f"           {c.description}")
    return out


def _config_block(settings) -> List[str]:
    """Render one ``plugin_configs.<plugin>.*`` knob per line.

    The declared ``enum`` is rendered because it is CHECKED: ``validate``
    reports a value outside it as ``invalid_knob_value`` (#925), and this
    listing is where an author reads the permitted set.  It sits beside the
    type rather than after the description because it IS type information,
    and a knob's description can run to several lines.  ``type`` may be a
    union (``string|array``), which is why the column is wider than the
    single JSON-Schema token it used to hold.
    """
    out: List[str] = []
    for s in settings:
        dflt = f"  (default {s.default!r})" if s.default is not None else ""
        desc = f"  {s.description}" if s.description else ""
        enum = ("  one of: " + ", ".join(repr(c) for c in s.enum)
                if s.enum else "")
        out.append(f"    {s.name:22} {s.type:12}{enum}{desc}{dflt}")
    return out


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
            badge = "core" if t.discoverability == DISCOVERABILITY_EAGER else "disc"
            lines.append(f"    [{badge}] {t.name}{_signature(t.parameters)}")
            if t.description:
                lines.append(f"           {t.description}")
            lines.extend(_nested_params(t.parameters))
        if any(t.discoverability != DISCOVERABILITY_EAGER for t in pi.tools):
            lines.append(
                f"  note: [core] tools are in the model's INITIAL schema; [disc] "
                f"are DEFERRED — the model reaches them by calling list_tools / "
                f"get_tool_schemas (introspection is always core, so they're never "
                f"lost), OR add `{name}(preload)` to a profile's plugins to force "
                f"ALL of this plugin's tools eager.")
    lines.extend(_command_block(pi.commands))
    if pi.config_settings:
        lines.append(f"  config (plugin_configs.{name}.*):")
        lines.extend(_config_block(pi.config_settings))
    data = {"description": pi.description,
            "kind": pi.kind, "tier": pi.tier, "dynamic": pi.dynamic,
            "commands": _commands_json(pi.commands),
            # ``parameters`` is what makes a tool SIGNATURE machine-checkable
            # from the CLI.  Without it a consumer validating a published spec
            # against the framework can compare names and prose but not the
            # arguments -- which is exactly how four drifts survived in a
            # public spec (a parameter that was never implemented, a renamed
            # one, two stale return shapes).
            "tools": [{"name": t.name, "discoverability": t.discoverability,
                       "description": t.description,
                       "parameters": t.parameters} for t in pi.tools],
            "config": [{"name": s.name, "type": s.type, "default": s.default,
                        "description": s.description, "enum": s.enum}
                       for s in pi.config_settings]}
    return data, "\n".join(lines)


# --------------------------------------------------------------- commands

def commands() -> Rendered:
    """Every user-facing TUI command, flat, grouped by owning plugin.

    The lookup ``explain plugin <name>`` cannot answer: "I know I want to
    change permissions / switch a model / inspect a plan — which command is
    it?"  The reader does not yet know which plugin owns the verb, so the
    catalog must be flat and grouped by owner.  These are the commands typed
    directly into the TUI (``permissions allow *``, ``memory …``), NOT the
    model's function-calling tools.
    """
    PL = introspect.plugins()
    data: Dict[str, Any] = {}
    lines = ["user commands — typed directly in the TUI, grouped by owning plugin",
             "  (`explain plugin <name>` for the full per-plugin page)"]
    total = 0
    for name in sorted(PL):
        pi = PL[name]
        if not pi.commands:
            continue
        data[name] = [{"name": c.name, "description": c.description,
                       "share_with_model": c.share_with_model,
                       "subcommands": c.subcommands} for c in pi.commands]
        total += len(pi.commands)
        lines.append(f"\n  [{name}]")
        for c in pi.commands:
            subs = (f"  ({', '.join(c.subcommands)})" if c.subcommands else "")
            lines.append(f"    {c.name}{subs}")
            if c.description:
                lines.append(f"      {c.description}")
    if not total:
        lines.append("\n  (no plugin exposes user commands)")
    else:
        lines.insert(2, f"  {total} command(s) across "
                        f"{len(data)} plugin(s)")
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


#: The caveat every provider page owes an ``api_params`` author.
#:
#: ``validate`` checks an ``api_params`` key against the PROVIDER's declared
#: allow-list, which is the only thing it can check: the layer is a property of
#: the wire, and which values a given MODEL on that wire accepts is not
#: declared anywhere in the tree and moves whenever a vendor ships.  So
#: ``temperature: 0.0`` validates clean against every OpenAI-shaped provider
#: and is a ``400`` on the reasoning models that accept only their default.
#:
#: The framework will not grow a per-model incompatibility table to close
#: that: a stale row would reject a parameter the vendor accepts, or pass one
#: it rejects, and either is worse than the honest statement that this check
#: stops at the provider boundary.  What IS always true is the escape hatch —
#: an omitted parameter is never the cause of a 400.
_API_PARAMS_CAVEAT = (
    "  api_params — scope of the check:\n"
    "    `validate` checks these against THIS PROVIDER's allow-list, not against\n"
    "    the model you named: a key can be a valid provider knob and still be a\n"
    "    400 on one model (reasoning models commonly accept only the DEFAULT\n"
    "    temperature / top_p, and reject an explicit 0.0).  Which values a model\n"
    "    takes is the vendor's documentation, not a fact this framework holds.\n"
    "    OMITTING a parameter is always safe — it is never the cause of a 400."
)


def _provider_notes(info) -> List[str]:
    """The provider's own declared caveats, if it has any.

    Rendered FIRST — above capabilities and knobs — because these are the
    facts that make the rest of the page readable: on Azure, every knob below
    is described correctly and still misleads anyone who read ``model:`` as a
    catalog model id.
    """
    if not getattr(info, "notes", ()):
        return []
    out = ["  read this first:"]
    for note in info.notes:
        out.extend(_wrap_bullet(note, indent=4, glyph="!"))
    return out


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
            "notes": list(getattr(info, "notes", ())),
            "resolution_order": res}

    lines = [f"provider: {info.dir_name}"]
    lines.extend(_provider_notes(info))
    lines.append("  capabilities: "
                 + ", ".join(k for k, v in caps.items() if v) or "  (none)")
    lines.append("  quirks: " + (", ".join(sorted(info.quirks)) or "(none)"))
    lines.append("  resolution order:")
    lines.extend(res)
    lines.append("  knobs (plugin_configs.%s.*):" % info.dir_name)
    lines.extend(_knob_layers(info.knobs))
    return data, "\n".join(lines)


def _knob_layers(knobs) -> List[str]:
    """Every ``plugin_configs.<provider>.*`` knob, grouped by layer.

    Split out of :func:`provider`, which is frozen in the complexity baseline —
    so the ``api_params`` caveat below had to land in a helper rather than grow
    it (see ``test_cyclomatic_complexity_audit``).
    """
    if not knobs:
        return []
    out: List[str] = []
    for layer in knobs.layers:
        tag = " (opaque pass-through)" if layer.opaque else ""
        desc = f"  — {layer.description}" if layer.description else ""
        out.append(f"    [{layer.layer}]{tag}{desc}")
        for k in layer.knobs:
            dflt = f"  (default {k.default!r})" if k.default is not None else ""
            d = f"  {k.description}" if k.description else ""
            out.append(f"      {k.name:22} {k.type:6}{d}{dflt}")
    if any(l.layer == "api_params" for l in knobs.layers):
        out.append(_API_PARAMS_CAVEAT)
    return out


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
#
# A session's env vars have TWO routes, and only the lower-precedence one was
# ever named here (jaato #752): `explain env` said "set these in the workspace
# .env" while the profile `env:` block outranks it -- and is the ONLY route
# available to a caller that does not own the .env.  The author who needed it
# found it by reading `server/core.py`, which is the failure this whole verb
# exists to prevent.
#
# The note deliberately spends its longest paragraph on PATH RESOLUTION, which
# is not where you would expect the difficulty to be.  Three plausible answers
# are all wrong, and each was believed by someone working on this: the runner
# is NOT chdir'd into the workspace at spawn (measured -- it is forked into
# the daemon's cwd, and `core.py:336` still promises otherwise);
# `${workspaceRoot}` does NOT help (it expands daemon-side, at a point where
# the session's workspace is not the one in scope); and a relative value is
# NOT rewritten on the way in.  What actually makes a relative trace path land
# per-session is the READER -- `jaato_sdk.trace._resolve_trace_file` joins it
# onto JAATO_WORKSPACE_ROOT, which the runner seeds per session.
#
# So the note names the reader instead of stating a rule about paths.  That is
# also the only form that survives the process cwd moving underneath it, which
# it does: `subagent/plugin.py` chdirs the whole runner into the workspace when
# it spawns a subagent.  A cwd that changes mid-session is not something to
# write a path against.

#: Worked example wherever the profile ``env:`` block is documented -- HERE and
#: in the commented block ``new`` emits (``build._set_profile_yaml``).  It is a
#: path knob on purpose: the resolution fact below is the half that bites, and
#: this is the variable people reach for when a session misbehaves.
ENV_EXAMPLE_VAR = "JAATO_PROVIDER_TRACE"

#: The example's VALUE, shared for the same reason as its name.  Deliberately
#: RELATIVE: ``jaato_sdk.trace._resolve_trace_file`` joins a relative trace
#: path onto ``JAATO_WORKSPACE_ROOT``, which the runner seeds per session, so
#: this form gives every session its own trace in its own workspace.  The
#: absolute form is fixed at the PROFILE and every session sharing that
#: profile appends to one interleaved file -- the failure mode this example
#: exists to steer people away from, and the one an earlier draft of this note
#: recommended (jaato #752 review).
ENV_EXAMPLE_VALUE = "provider_trace.log"

#: The three load-bearing, non-obvious properties of the profile ``env:``
#: block, rendered by BOTH halves of its documentation from this ONE
#: definition.
#:
#: Sharing the strings is the anti-drift mechanism.  #716's
#: ``test_a_real_run_writes_only_documented_files`` asserts which FILES ``new``
#: writes, never their content, so a fact stated in the generated comment and
#: not in ``explain env`` (or reworded in one of them) would drift with nothing
#: failing -- "documentation about a generator rots", one level down.  Kept
#: short enough to render as a comment line inside a generated YAML file.
PROFILE_ENV_FACTS = (
    "outranks the workspace .env, per key",
    "takes ${VAR} expansion + secret URIs (pass://, vault://, ...)",
    "is applied verbatim — a relative path is resolved by its READER",
)


def _profile_env_note() -> List[str]:
    """The two-routes note that heads ``explain env``.

    Its own function because it renders :data:`PROFILE_ENV_FACTS` in a
    comprehension, and folding that back into :func:`env` puts that function
    over the repository's cyclomatic-complexity ceiling.

    Returns:
        Rendered lines, blank-line separated from the variable catalogue that
        follows them.
    """
    lines = [
        "",
        "  TWO ROUTES SET THESE — the profile block wins:",
        "    <workspace>/.env                     VAR=value"
        "           per-WORKSPACE  (lower)",
        "    .jaato/profiles/<set>/<agent>.yaml   env: {VAR: value}"
        "   per-SESSION    (higher)",
        "  The profile `env:` block:",
    ]
    lines += [f"    - {fact}" for fact in PROFILE_ENV_FACTS]
    lines += [
        "      Nothing rewrites the value on the way in, so WHERE a relative "
        "path",
        "      lands is the reader's business, and readers differ:",
        "        · the trace vars (JAATO_PROVIDER_TRACE / JAATO_TRACE_LOG) "
        "join theirs",
        "          onto JAATO_WORKSPACE_ROOT, which the runner seeds per "
        "session — so a",
        "          RELATIVE value gives each session its own file in its own "
        "workspace,",
        "          and an ABSOLUTE one is fixed at the profile and shared by "
        "every",
        "          session using it (jaato_sdk/trace.py _resolve_trace_file).",
        "        · a reader that just open()s the value gets the runner "
        "process's cwd,",
        "          which is NOT the workspace: the runner is forked into the "
        "daemon's",
        "          cwd (core.py:1029 — the chdir promised at core.py:336 does "
        "not",
        "          happen), and the subagent path can chdir it later, so it is "
        "not a",
        "          thing to write a path against.  `${workspaceRoot}` / "
        "`${cwd}` are no",
        "          help either: they expand daemon-side, before the session "
        "exists.",
        "    - merges per KEY across `inherits:` (a child wins only the keys "
        "it sets)",
        "    - is the ONLY route when the .env is not yours to write — "
        "jaato-eval writes",
        "      each arm's .env itself (JAATO_PROFILE_SET and nothing else), so "
        "a task",
        "      author cannot put anything there.",
        "",
        "        env:",
        f"          {ENV_EXAMPLE_VAR}: {ENV_EXAMPLE_VALUE}"
        "    # one file per session",
        "",
        "  A var with a TYPED key (the `→ typed:` line on its row below) has a",
        "  better route than `env:`: the typed one is validated, and `env:` is",
        "  not.  The two trace vars are the worked example in both directions —",
        "  `env: {JAATO_PROVIDER_TRACE: 1}` is a valid str and wrote every",
        "  session's trace to a file named `1` (#775); the block refuses it:",
        "",
        "        trace:",
        f"          provider_log: {ENV_EXAMPLE_VALUE}"
        "   # same resolution, checked",
    ]
    return lines


#: Compact scope glyphs for the per-var rows.  The full word is in the
#: ``--json`` payload and in the legend; a 186-row listing cannot spend
#: nine columns per row on it.
_SCOPE_GLYPH = {
    "session": "S",
    "host": "H",
    "ambient": "a",
    "internal": "i",
    "unclassified": "?",
}


def _scope_summary(EV) -> List[str]:
    """The scope/typed-key headline for ``explain env`` (issue #775).

    The catalog's whole point is the count in the third line: how many
    session-scoped knobs a profile author still cannot set with a typed,
    validated key.  Printing it above the listing is what makes the
    number get smaller.
    """
    from shared.env_scope import AWAITING_TYPED_KEY, SESSION

    session = [v for v in EV.values() if v.scope == SESSION]
    typed = [v for v in session if v.typed_key]
    unclassified = [v for v in EV.values() if v.scope == "unclassified"]
    out = [
        "",
        "  scope (shared/env_scope.py — S session, H host, a ambient, "
        "i internal):",
        f"    {len(session):>3} session   a knob two sessions on one host may "
        f"differ on — {len(typed)} have a typed key,",
        f"        {len(session) - len(typed):>3} do not "
        f"(tiers: {_tier_counts(AWAITING_TYPED_KEY)}). `explain env untyped`",
        f"    {sum(1 for v in EV.values() if v.scope == 'host'):>3} host      "
        f"process/host-scoped — a per-session value would be a lie",
        f"    {sum(1 for v in EV.values() if v.scope == 'ambient'):>3} ambient   "
        f"the host environment being READ, not a knob at all",
        f"    {sum(1 for v in EV.values() if v.scope == 'internal'):>3} internal  "
        f"one framework process handing a value to another",
    ]
    if unclassified:
        out.append(f"    {len(unclassified):>3} UNCLASSIFIED — "
                   + ", ".join(sorted(v.name for v in unclassified)))
    return out


def _tier_counts(awaiting: Dict[str, str]) -> str:
    """``A=14 B=11 E=18`` for the summary line."""
    counts: Dict[str, int] = {}
    for entry in awaiting.values():
        counts[entry.tier] = counts.get(entry.tier, 0) + 1
    return " ".join(f"{t}={counts[t]}" for t in sorted(counts))


def _env_selected(v, filter_: str, want_untyped: bool) -> bool:
    """Does this var pass the ``explain env`` filter?

    Split out of :func:`env` for the complexity ceiling, and it reads
    better here anyway: the filter has three independent modes (the
    untyped view, a scope name, a category/name substring) and inlining
    them made the listing loop look like the interesting part.
    """
    from shared.env_scope import SESSION

    if want_untyped:
        return v.scope == SESSION and not v.typed_key
    if not filter_:
        return True
    return (filter_ in v.category or filter_ in v.name
            or filter_ == v.scope)


def _env_rows(vs: list, width: int) -> List[str]:
    """The per-var lines for one category, with the scope annotation.

    A var either has a typed key (named, so an author can reach for it)
    or is session-scoped without one (the reason given, so the gap is
    legible rather than merely absent).  Everything else -- host,
    ambient, internal -- says what it is with the glyph alone.
    """
    from shared.env_scope import AWAITING_TYPED_KEY, SESSION

    out: List[str] = []
    pad = " " * (width + 2)
    for v in vs:
        d = f" = {v.default}" if v.default not in (None, "") else ""
        desc = f"   — {v.description}" if v.description else ""
        glyph = _SCOPE_GLYPH.get(v.scope, "?")
        out.append(f"    {glyph} {v.name:<{width}}{d}{desc}")
        if v.typed_key:
            out.append(f"    {pad}  → typed: {v.typed_key}")
            continue
        if v.scope != SESSION:
            continue
        # A debt entry shows WHERE the key should go, not just that one is
        # missing: the proposal is the part a reader can act on or argue with.
        owed = AWAITING_TYPED_KEY.get(v.name)
        if owed is not None:
            out.append(f"    {pad}  → proposed: {owed.proposed_key}  "
                       f"[tier {owed.tier}]")
            if owed.note:
                out.append(f"    {pad}    {owed.note}")
        elif v.scope_note:
            out.append(f"    {pad}  → no typed key: {v.scope_note}")
    return out


def env(filter_: str = None) -> Rendered:
    """Env vars the installed daemon + plugins read (optionally filtered).

    ``filter_`` matches a category substring, a var-name substring, or a
    scope — so ``explain env nebius`` → ``provider:nebius`` vars,
    ``explain env gc`` → GC knobs, ``explain env host`` → the vars that are
    deliberately NOT per-session, and ``explain env untyped`` → the
    session-scoped knobs that still have no typed profile key (issue #775).

    Each row carries its scope glyph and, where one exists, the typed key
    that already covers it — so the answer to "should this be a profile
    field?" is in the listing rather than in a reader's head.
    """
    EV = introspect.env_vars()
    want_untyped = filter_ in ("untyped", "awaiting")
    groups: Dict[str, list] = {}
    for name in sorted(EV):
        v = EV[name]
        if _env_selected(v, filter_, want_untyped):
            groups.setdefault(v.category, []).append(v)

    data = {
        cat: {v.name: {"default": v.default, "description": v.description,
                       "scope": v.scope, "typed_key": v.typed_key,
                       "scope_note": v.scope_note,
                       "sources": v.sources[:2]}
              for v in vs}
        for cat, vs in groups.items()
    }
    documented = sum(1 for v in EV.values() if v.description)
    head = (f"env vars read by the installed daemon + plugins "
            f"({len(EV)} total, {documented} documented)")
    if filter_:
        head += f" — filter '{filter_}'"
    lines = [head,
             "  (commented = optional; descriptions come from `# env: ...` "
             "comments at the read site)"]
    lines += _scope_summary(EV)
    lines += _profile_env_note()
    for cat in sorted(groups):
        lines.append(f"\n  [{cat}]")
        lines += _env_rows(groups[cat],
                           max((len(v.name) for v in groups[cat]), default=0))
    return data, "\n".join(lines)


# ------------------------------------------------------------------ events

# Compact direction glyphs for the overview table (full words in detail view).
_DIR_GLYPH = {
    "Server → Client": "S→C",
    "Client → Server": "C→S",
    "Server ↔ Client": "S↔C",
}


def events(filter_: str = None) -> Rendered:
    """The client/server event protocol, grouped by domain.

    ``filter_`` matches a domain substring, an event-member substring, or a
    wire-value substring — ``explain events permission`` → the permission-flow
    events, ``explain events tool`` → the tool-exec/status events.  Each row is
    ``MEMBER  wire.value  [direction]``; drill into one with ``explain event
    <NAME>`` (member OR wire value) for its fields + docstring.
    """
    EV = introspect.events()
    groups: Dict[str, list] = {}
    for name in sorted(EV):
        e = EV[name]
        if filter_ and filter_.lower() not in (
                e.domain.lower() + " " + e.name.lower() + " " + e.wire.lower()):
            continue
        groups.setdefault(e.domain or "(ungrouped)", []).append(e)

    data = {
        dom: {e.name: {"wire": e.wire, "direction": e.direction,
                       "event_class": e.event_class,
                       "fields": [f.name for f in e.fields]}
              for e in es}
        for dom, es in groups.items()
    }
    shown = sum(len(v) for v in groups.values())
    head = (f"event protocol — {len(EV)} typed events across "
            f"{len({e.domain for e in EV.values()})} domains"
            + (f"; {shown} match '{filter_}'" if filter_ else ""))
    lines = [head,
             "  (S→C server→client, C→S client→server, S↔C bidirectional; "
             "wire value is the on-the-wire `type`. `explain event <NAME>` for detail)",
             "  NOTE a cascade/observer subscription filters on the CLASS name "
             "(SessionTerminatedEvent),",
             "       not on the member or the wire value — "
             "`cascade_events(event_types=[...])` and the daemon both",
             "       compare type(event).__name__, so a wire value there "
             "matches nothing at all (#821)."]
    for dom in sorted(groups):
        lines.append(f"\n  [{dom}]")
        w = max((len(e.name) for e in groups[dom]), default=0)
        ww = max((len(e.wire) for e in groups[dom]), default=0)
        for e in groups[dom]:
            g = _DIR_GLYPH.get(e.direction, e.direction or "?")
            cls = "" if e.event_class else "   (no class — command/marker)"
            lines.append(f"    {e.name:<{w}}  {e.wire:<{ww}}  [{g}]{cls}")
    return data, "\n".join(lines)


def event(name: str) -> Rendered:
    """One event's direction, domain, fields, and docstring.

    ``name`` matches the ``EventType`` member (``AGENT_OUTPUT``) or the wire
    value (``agent.output``), case-insensitively.
    """
    EV = introspect.events()
    key = name.strip()
    e = EV.get(key) or EV.get(key.upper())
    if e is None:
        low = key.lower()
        e = next((x for x in EV.values()
                  if x.name.lower() == low or x.wire.lower() == low), None)
    if e is None:
        # Separator-insensitive substring match so ``sessionwoke`` still finds
        # ``SESSION_WOKEN`` / ``session.woken``.
        def _norm(s: str) -> str:
            return s.lower().replace("_", "").replace(".", "").replace("-", "")
        nkey = _norm(key)
        near = sorted(x.name for x in EV.values()
                      if nkey in _norm(x.name) or nkey in _norm(x.wire))
        hint = f" — did you mean: {', '.join(near[:8])}" if near else ""
        return ({"error": f"unknown event {name!r}"},
                f"unknown event {name!r}{hint}")

    data = {
        "name": e.name, "wire": e.wire, "direction": e.direction,
        "domain": e.domain, "event_class": e.event_class,
        "note": e.note, "doc": e.doc,
        "fields": [{"name": f.name, "type": f.type} for f in e.fields],
    }
    lines = [f"{e.name}   ({e.wire})",
             f"  direction : {e.direction or '(unspecified)'}",
             f"  domain    : {e.domain or '(ungrouped)'}"]
    if e.event_class:
        lines.append(f"  class     : {e.event_class}")
        # The one place the distinction bites.  A reader who has just been
        # shown a member name and a wire value reasonably reaches for either
        # when writing a filter; only this third string works, and the wrong
        # one fails silently for the life of the subscription (#821).
        lines.append(f"  subscribe : event_types=[\"{e.event_class}\"]   "
                     "(cascade_events / observer filters match the CLASS name)")
    else:
        lines.append("  class     : (none — a wire marker / command, no payload class)")
    if e.note:
        lines.append(f"  note      : {e.note}")
    if e.doc:
        lines.append(f"  purpose   : {e.doc}")
    if e.fields:
        lines.append("  fields    :")
        w = max(len(f.name) for f in e.fields)
        for f in e.fields:
            lines.append(f"    {f.name:<{w}} : {f.type}")
    elif e.event_class:
        lines.append("  fields    : (none declared beyond the base Event)")
    return data, "\n".join(lines)



# ------------------------------------------------------------------ agents

def agents(workspace: str = ".") -> Rendered:
    """Where an agent's PERSONA lives, and why it is not a profile field.

    The gap this closes: nothing under ``explain`` named ``.jaato/agents/`` at
    all, while ``explain profile`` listed ``system_instructions`` — marked
    DEPRECATED, and still the only instruction-shaped key on the page.  An
    author who never found the agents directory reached for the deprecated key,
    which works, so nothing corrected them.

    The search order is read from :func:`agent_search_dirs`, the function the
    runtime itself walks, rather than restated here — a documented order that
    disagrees with the loaded one is worse than none.
    """
    from shared.plugins.subagent.config import (
        AGENT_FILE_FORMS, agent_search_dirs)

    ws = Path(workspace).expanduser().resolve()
    dirs = agent_search_dirs(str(ws))
    found: List[Dict[str, Any]] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.md")):
            found.append({"name": f.stem, "path": str(f)})

    data = {
        "search_order": [str(d) for d in dirs],
        "file_forms": list(AGENT_FILE_FORMS),
        "frontmatter": ["description", "default_profile", "params"],
        "discovered": found,
        "profile_field": "default_agent",
        "deprecated": "system_instructions",
    }

    lines = [
        "agents — the PERSONA layer (.jaato/agents/<name>.md)",
        "",
        "  A profile says what a session CAN DO (plugins, model, limits).",
        "  An agent says WHO IT IS (instructions, voice, task framing).  They are",
        "  orthogonal and compose: the same persona runs on a cheap profile and an",
        "  expensive one; the same profile serves several personas.",
        "",
        "  WHERE — searched in this order, first hit wins:",
    ]
    for d in dirs:
        mark = "  (exists)" if d.is_dir() else ""
        lines.append(f"    {d}{mark}")
    lines += [
        "    In each, the name may take any of these forms:",
    ]
    lines += [f"      {form}" for form in AGENT_FILE_FORMS]
    lines += [
        "    `config_root` REPLACES the workspace tier — it IS that tier, moved.",
        "",
        "  FORMAT — optional YAML frontmatter, then markdown:",
        "    ---",
        "    description: what this agent is for   # shown when listing agents",
        "    default_profile: researcher           # profile to pair it with",
        "    params:",
        "      topic: {default: 'anything'}        # default for {{topic}}",
        "    ---",
        "    You are a research agent working on {{topic}}.",
        "    Budget: {{max_sources:5}} sources.    # inline default, no frontmatter",
        "",
        "    {{param}} substitution: a value from `agent_params`, else the inline",
        "    default, else the frontmatter default.  An unresolved placeholder is",
        "    left LITERAL on purpose (a visible debugging signal) and reported in",
        "    `missing_params` — it is not silently blanked.",
        "    {{!py:scripts/<f>.py}} runs a prefetch script at session-prep —",
        "    see `jaato-scaffold explain prefetch`.",
        "",
        "  HOW A SESSION GETS ONE — three routes, in precedence order:",
        "    1. explicit          client.create_session(agent='researcher',",
        "                                               agent_params={'topic': 'x'})",
        "                         spawn_subagent(profile='...', agent='researcher')",
        "    2. the profile's own default_agent: researcher   (#944)",
        "                         — a profile supplies plugins, an agent supplies",
        "                         instructions; this binds the two so spawning by",
        "                         profile name alone yields a subagent that has",
        "                         both.  An explicit agent= still wins.",
        "    3. neither           the session runs with no persona layer at all.",
        "",
        "  WHY NOT IN THE PROFILE YAML:",
        "    `system_instructions:` is DEPRECATED and describes the wrong thing —",
        "    a persona is not 'the session's system instructions'.  The rendered",
        "    prompt LAYERS: .jaato/instructions/ base + THIS persona + plugin",
        "    instructions + framework constants + the untrusted-content boundary.",
        "    `suppress_base_instructions` can drop every layer EXCEPT the persona",
        "    and its plugins — which is what makes the persona the durable half.",
        "    Markdown also travels: it is reviewable, diffable, and reusable across",
        "    profiles in a way a YAML block scalar is not.",
        "",
        "  NEVER PASS A CREDENTIAL AS AN agent_param.  Params are substituted INTO",
        "  the persona, so they already reach the model in its system prompt — and",
        "  the rendered persona is PERSISTED with the session (#787).  Secrets go in",
        "  the profile's `env:` as a pass:// / vault:// URI, resolved daemon-side.",
        "",
        "  See also: `explain prefetch` (scripted session-start context),",
        "  `explain profile` (the capability half), `explain paths` (config_root).",
    ]
    if found:
        lines += ["", f"  IN THIS WORKSPACE ({ws}) — {len(found)} agent(s):"]
        lines += [f"    {a['name']:24} {a['path']}" for a in found]
    else:
        lines += ["", f"  IN THIS WORKSPACE ({ws}): no agent files found.",
                  "    `jaato-scaffold explain agents --workspace <dir>` to look elsewhere."]
    return data, "\n".join(lines)


# ---------------------------------------------------------------- services

def services(workspace: str = ".") -> Rendered:
    """The ``service_connector`` managed-service flow, and where it stores things.

    The gap this closes: ``.jaato/services/`` appeared in no ``explain`` topic,
    so a session that needed an HTTP API reached for a raw URL — passing the
    base URL, the auth header and the pagination by hand on every call, and
    re-deriving all three next session.  The managed path (discover once, name
    it, call it by alias) was already there and simply invisible.
    """
    from shared.plugins.service_connector.schema_store import (
        DEFAULT_SERVICES_DIR, DISCOVERED_DIR, SERVICE_CONFIG_FILE)

    ws = Path(workspace).expanduser().resolve()
    tiers = [("workspace", ws / DEFAULT_SERVICES_DIR),
             ("user", Path.home() / DEFAULT_SERVICES_DIR)]
    known: List[Dict[str, Any]] = []
    for label, root in tiers:
        if not root.is_dir():
            continue
        disc = root / DISCOVERED_DIR
        if disc.is_dir():
            for f in sorted(disc.glob("*.y*ml")):
                known.append({"tier": label, "service": f.stem,
                              "kind": "discovered", "path": str(f)})
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            if d.name == DISCOVERED_DIR:
                continue
            known.append({"tier": label, "service": d.name,
                          "kind": "defined", "path": str(d)})

    data = {"tiers": [{"tier": t, "path": str(p), "exists": p.is_dir()}
                      for t, p in tiers],
            "writable_tier": "workspace",
            "discovered_dir": DISCOVERED_DIR,
            "service_config_file": SERVICE_CONFIG_FILE,
            "known": known}

    lines = [
        "services — named HTTP APIs the model calls by alias "
        "(.jaato/services/, service_connector plugin)",
        "",
        "  THE FLOW — three calls, then the API has a name:",
        "    1. discover_service(source='https://host/openapi.json', alias='gitlab')",
        "         parses the OpenAPI/Swagger spec, caches it, learns every",
        "         endpoint's parameters and response schema.",
        "    2. configure_service_auth(service='gitlab', auth={...})",
        "         binds a scheme + the ENV VARS its credentials come from, once.",
        "         (`explain plugin service_connector` prints the auth shapes.)",
        "    3. call_service(service='gitlab', method='GET', path='/projects')",
        "         base URL, auth and request validation all come from the alias.",
        "    list_endpoints / get_endpoint_schema browse it in between;",
        "    preview_request dry-runs a call and prints the equivalent curl.",
        "",
        "  WHERE IT IS STORED — two tiers, workspace shadows home:",
    ]
    for label, root in tiers:
        mark = "  (exists)" if root.is_dir() else ""
        note = "  writable" if label == "workspace" else "  read-only here"
        lines.append(f"    [{label:9}] {root}{note}{mark}")
    lines += [
        "    Per tier:",
        f"      {DISCOVERED_DIR + '/<service>.yaml':30} auto-cached OpenAPI specs",
        f"      {'<service>/' + SERVICE_CONFIG_FILE:30} hand-written service config",
        f"      {'<service>/<endpoint>.yaml':30} hand-written endpoint schemas",
        "    The user tier is populated out of band (copy a service you want in",
        "    every workspace); writes always land in the workspace tier.",
        "",
        "  MANAGED ALIAS vs RAW URL — call_service takes either.  Prefer the alias",
        "  when any of these hold:",
        "    - the API needs auth        the scheme is configured ONCE, and the",
        "                                credential stays an env-var NAME on disk",
        "    - it is paginated or wide   the cached schema is what makes",
        "                                list_endpoints / get_endpoint_schema able",
        "                                to answer without another round trip",
        "    - requests get validated    a body is checked against the spec before",
        "                                it is sent, so a typo is a local error",
        "    - more than one call        the alias survives the session; a raw URL",
        "                                is re-derived by hand every time",
        "  A raw `url=` is right for a one-shot unauthenticated fetch, and for an",
        "  API that publishes no spec (save_schema / import_bruno_collection give",
        "  those one by hand).",
        "",
        "  Under confinement the user tier is granted READ only",
        "  (~/.jaato/services/**), so a service the workspace must WRITE belongs",
        "  in the workspace tier.  See `explain paths`.",
    ]
    if known:
        lines += ["", f"  KNOWN HERE — {len(known)}:"]
        lines += [f"    [{k['tier']:9}] {k['service']:24} {k['kind']}"
                  for k in known]
    else:
        lines += ["", "  KNOWN HERE: none — nothing has been discovered yet.",
                  f"    (looked in {tiers[0][1]} and {tiers[1][1]})"]
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
    lines.append(
        "\n  inheritance (`inherits: [_base_<stage>]`) — how a child profile merges with its\n"
        "  parent(s), resolved at discover_profiles() (config.py:_merge_profiles):\n"
        "    plugins, preloaded_plugins   UNION / additive — child ADDS to the parents'; it\n"
        "                                 CANNOT scope DOWN here (the list only grows).\n"
        "    completion_processors        CONCATENATED parent → child; all of them fire. `[]`\n"
        "                                 in the child ADDS NOTHING — it does NOT clear the\n"
        "                                 parents'. Scope DOWN by naming inherited entries in\n"
        "                                 `suppress_inherited_processors` (matches an entry's\n"
        "                                 `name`, else its `script`); an entry matching nothing\n"
        "                                 is a load ERROR, and it is not inherited further.\n"
        "    tool_scopes, env,            per-KEY dict-merge — child wins on keys it sets;\n"
        "    plugin_configs, quirks       the parent's other keys survive.\n"
        "    model, provider, gc, cache,  child REPLACES — the child's value wins outright\n"
        "    model_tiers, runtime_limits, (this is how a child scopes DOWN, unlike plugins).\n"
        "    scrub_secret_env, default_agent,\n"
        "    apparmor_fragments,          For the two payload schemas an empty dict `{}` IS a\n"
        "    completion_payload_schema,   value and overrides; `null`/absent reads as unset and\n"
        "    spawn_payload_schema         inherits.\n"
        "    max_turns,                   MOST RESTRICTIVE wins — a child may only TIGHTEN a\n"
        "    budget_control.limits,       ceiling, never raise the one it was spawned under.\n"
        "    runtime_limits.              (budget_control.degrade is child-REPLACES, and so are\n"
        "      max_parallel_tools         runtime_limits' other, kernel-enforced ceilings.)\n"
        "    suppress_base_instructions,  UNION / OR — STICKY: a piece any layer drops stays\n"
        "    apparmor                     dropped, and a confined parent can't be un-confined.\n"
        "\n  empty vs listed `plugins` (a REQUIRED key — authors must pick):\n"
        "    plugins: []   → tools=[] → NONE of the registry tool plugins; only the framework\n"
        "                   set (permission, reliability, lifecycle/signal_completion) is wired.\n"
        "                   (Pre-2026-06-07 a falsy bug made [] silently load ALL ~30 tools.)\n"
        "    plugins: [x]  → exactly those, UNIONed with any inherited.\n"
        "    To scope DOWN per stage, use tool_scopes (per-plugin allow-list) or the permission\n"
        "    plugin's whitelist — NOT the plugins list, which only ADDS to the inherited set.")
    lines.append(
        "\n  declining ONE inherited completion_processor (the only removal opt-out there is):\n"
        "    inherits: [_base_worker]\n"
        "    suppress_inherited_processors:\n"
        "      - acceptance              a parent entry's `name:`, or its `script:` path\n"
        "    Everything else scopes down by REPLACING a value, never by removing an entry.\n"
        "    Don't stop inheriting just to drop a processor: you lose budget_control,\n"
        "    max_turns, runtime_limits, env and plugin_configs with it, silently.")
    return data, "\n".join(lines)


def profile_cost(name: str, workspace: str) -> Rendered:
    """What a session built from *name* INHERITS, and what it costs.

    A profile file says what it ADDS.  It never says what it INHERITS -- and
    the inherited instruction layers are prepended to every session in the
    workspace.  A 48KB ``.jaato/instructions/`` folder is ~12k tokens on
    EVERY turn of EVERY session, and nothing in the authoring surface says
    so: it shows up later as a cascade budget refusal, three sessions
    downstream of the decision that caused it.

    Measured from the SAME search order ``JaatoRuntime._load_base_system_instructions``
    uses -- premium tier, then the first of (workspace|config_root) /
    (user home) that yields content -- so this reports what would actually
    load, not a plausible reconstruction of it.

    Token figures are ESTIMATES (bytes / 4), labelled as such.  A real count
    needs the model's tokenizer; the point here is the order of magnitude
    that makes a knob worth reaching for.
    """
    from shared.instruction_suppression import (
        PIECE_DISK, normalize_suppression,
    )
    ws = Path(workspace).resolve()

    prof, where = _find_profile_file(ws, name)
    if prof is None:
        return ({"profile": name, "found": False},
                f"no profile {name!r} under {ws}/.jaato/profiles/")

    suppressed = normalize_suppression(prof.get("suppress_base_instructions"))
    disk_suppressed = PIECE_DISK in suppressed

    layers = []
    for label, d in _instruction_search_order(ws):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.md"))
        if not files:
            continue
        size = sum(f.stat().st_size for f in files)
        # Content digest, so two tiers holding the SAME text are reported as
        # a duplicate rather than as two costs that happen to match.  A copy
        # of the premium instructions left in ~/.jaato/instructions is loaded
        # by BOTH tiers -- the runtime layers them, it does not dedupe -- so
        # the identical bytes reach the model twice and the only visible
        # symptom is a prompt that is twice as large as the file.
        digest = hashlib.md5(
            b"".join(f.read_bytes() for f in files)).hexdigest()
        layers.append({
            "layer": label, "dir": str(d), "files": len(files),
            "bytes": size, "approx_tokens": size // 4, "digest": digest,
        })
        # The runtime stops at the FIRST of workspace/user that yields
        # content; mirroring that here is the difference between reporting
        # what loads and reporting what exists.
        if label in ("workspace", "user"):
            break

    persona = ws / ".jaato" / "agents" / f"{name}.md"
    persona_bytes = persona.stat().st_size if persona.is_file() else 0

    seen_digests = {}
    for l in layers:
        seen_digests.setdefault(l["digest"], []).append(l["layer"])
    duplicates = [ls for ls in seen_digests.values() if len(ls) > 1]

    inherited = 0 if disk_suppressed else sum(l["bytes"] for l in layers)
    total = inherited + persona_bytes
    data = {
        "profile": name, "found": True, "profile_file": str(where),
        "suppress_base_instructions": sorted(suppressed),
        "disk_layer_suppressed": disk_suppressed,
        "layers": layers,
        "persona_bytes": persona_bytes,
        "inherited_bytes": inherited,
        "total_bytes": total,
        "approx_total_tokens": total // 4,
        "duplicate_layers": duplicates,
        "note": "token figures are estimates (bytes/4), not a tokenizer count",
    }

    lines = [f"instruction cost for profile {name!r}  ({where})", ""]
    if not layers:
        lines.append("  no instruction layers found on the search path")
    for l in layers:
        mark = "  (SUPPRESSED)" if disk_suppressed else ""
        lines.append(
            f"  {l['layer']:10} {l['files']:>3} file(s)  {l['bytes']:>8,} B  "
            f"~{l['approx_tokens']:>6,} tok{mark}")
        lines.append(f"             {l['dir']}")
    if persona_bytes:
        lines.append(
            f"  {'persona':10} {1:>3} file(s)  {persona_bytes:>8,} B  "
            f"~{persona_bytes // 4:>6,} tok")
    lines += [
        "",
        f"  inherited on EVERY turn : {inherited:,} B  (~{inherited // 4:,} tok)",
        f"  total with persona      : {total:,} B  (~{total // 4:,} tok)",
        "",
        "  token figures are ESTIMATES (bytes/4), not a tokenizer count.",
    ]
    for dup in duplicates:
        dup_bytes = next(l["bytes"] for l in layers if l["layer"] == dup[0])
        lines += [
            "",
            f"  DUPLICATE: {' and '.join(dup)} hold IDENTICAL content.",
            f"  The runtime LAYERS these tiers, it does not dedupe — so those",
            f"  {dup_bytes:,} B (~{dup_bytes // 4:,} tok) reach the model TWICE",
            "  every turn.  Removing the copy halves this cost without",
            "  changing any profile.",
        ]
    if not disk_suppressed and inherited:
        lines += [
            "",
            "  This profile INHERITS the disk instruction layer.  If its persona",
            "  is self-contained, opt out with:",
            "      suppress_base_instructions: {disk: true}",
            "  (the security boundary is KEPT unless you name it explicitly)",
        ]
    return (data, "\n".join(lines))


def _find_profile_file(ws: Path, name: str):
    """The profile's parsed dict + its path, or ``(None, None)``.

    Searches the set subdirectories and the tier-1 root, matching the layout
    ``discover_profiles`` reads.
    """
    import yaml
    pdir = ws / ".jaato" / "profiles"
    if not pdir.is_dir():
        return None, None
    for cand in sorted(pdir.rglob("*.y*ml")) + sorted(pdir.rglob("*.json")):
        if cand.stem != name:
            continue
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                parsed = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed, cand
    return None, None


def _instruction_search_order(ws: Path):
    """The tiers ``JaatoRuntime`` consults, in its order.

    Kept as one list so a change to the runtime's order is a change to ONE
    place here -- reporting a different order than the runtime loads would
    be worse than not reporting at all.
    """
    order = []
    try:
        from shared.jaato_runtime import _get_premium_content_path
        premium = _get_premium_content_path("instructions")
        if premium:
            order.append(("premium", Path(premium)))
    except Exception:
        pass
    order.append(("workspace", ws / ".jaato" / "instructions"))
    order.append(("user", Path.home() / ".jaato" / "instructions"))
    return order


def paths() -> Rendered:
    """The path & isolation model — daemon-global ``~/.jaato`` vs per-session
    workspace + ``config_root``.

    Authored to head off the common "override ``$HOME`` to isolate a test run"
    mistake: jaato keeps ``~/.jaato`` deliberately daemon-global (creds + the
    auto-installed reactors live there), and isolates PER SESSION at the
    workspace / ``config_root`` layer — not at ``$HOME``.

    Also states the ownership rule the contents list only implies (#896):
    ``config_root`` is FRAMEWORK-OWNED, and a driver's own runtime state
    (checkpoints, resume journals, run scratch) belongs at a sibling
    ``<workspace>/.<yourapp>/`` instead.  Read as a list of contents rather
    than a statement of ownership, ``.jaato/`` looks like "the jaato-related
    directory for this workspace", so an SDK author with jaato-shaped state
    files it alongside and nothing fails — until the deny surface moves.  It
    does move: ``server/apparmor.py`` v11 introduced a broad ``.jaato/** w``
    deny, v13 replaced it with narrow per-subpath denies after the broad one
    blocked the daemon's own writes, and v29 added ``.jaato/templates/`` —
    at which point the template plugin's own writer had to relocate to the
    sibling ``.jaato/template_extracts/`` to survive it.  That is the
    framework fixing, for itself, exactly the break a tenant parked under an
    arbitrary ``.jaato/<name>/`` would take on a release it did not author.
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
        "config_root_ownership": {
            "owner": "framework",
            "rule": "config_root (<workspace>/.jaato) is FRAMEWORK-OWNED: "
                    "everything under it is either config jaato reads or "
                    "runtime state jaato writes",
            "tenant_state_goes": "<workspace>/.<yourapp>/  (a SIBLING of "
                                 ".jaato, still inside the workspace so it "
                                 "stays on the isolation boundary)",
            "why": "the confinement deny surface under .jaato is real, "
                   "actively maintained and moves between releases "
                   "(server/apparmor.py v11 -> v13 -> v29); a tenant "
                   "subpath is not denied today, and no allow rule names "
                   "it either",
            "precedent": ".jaato/templates/ became write-denied in template "
                         "v29 (#893) and the template plugin's own writer "
                         "had to move to the sibling .jaato/template_extracts/",
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
        "    profiles / instructions / agents.  It is a CLIENT knob, not a profile",
        "    key: pass `config_root=` to IPCClient / WSClient / InProcessClient",
        "    (all three default it to <workspace>/.jaato).  The default is applied",
        "    client-side, and that matters because the value has two consumers:",
        "      - the config SEARCH PATH falls back to <workspace>/.jaato daemon-side",
        "        whether or not the value is set, so profiles and agents resolve",
        "        either way;",
        "      - a plugin that WRITES under the root reads the VALUE — file_edit",
        "        puts backups in <config_root>/sessions/<id>/backups/ and REFUSES",
        "        to initialize without one, so a session that reached the daemon",
        "        with config_root unset came up with no writeNewFile at all.",
        "    A plugin the profile asked for that fails to initialize is now named",
        "    at WARNING (registry.get_failed_plugins() is the programmatic read).",
        "    -> FRAMEWORK-OWNED.  Everything under it is either config jaato READS",
        "       (profiles, agents, instructions, schemas, scripts, templates) or",
        "       runtime state jaato WRITES (logs/, sessions/, cache/, memory/,",
        "       todos/, ...).  Do NOT park your own application's state here.",
        "",
        "  YOUR OWN state (checkpoints, resume journals, run scratch) goes in a",
        "  SIBLING directory — <workspace>/.<yourapp>/ — not under .jaato/:",
        "    <workspace>/.jaato/       framework config + framework runtime state",
        "    <workspace>/.<yourapp>/   YOUR driver's runtime state",
        "    -> Still inside the workspace, so a fresh workspace still isolates a",
        "       run and the two ownerships stay legible side by side.",
        "    -> WHY not .jaato/<yourapp>/: the confinement deny surface under",
        "       .jaato is real, actively maintained, and MOVES between releases",
        "       (server/apparmor.py: v11 added a broad .jaato/** write deny, v13",
        "       replaced it with narrow per-subpath denies, v29 added",
        "       .jaato/templates/).  A tenant-invented subpath is not denied",
        "       today — and no allow rule names it either, so a later release",
        "       that denies your name EACCESes every confined writer (reactor,",
        "       prefetch, completion processor, tool) with nothing in your",
        "       mental model to debug it.  Precedent: when v29 denied",
        "       .jaato/templates/, the template plugin's OWN writer had to",
        "       relocate to the sibling .jaato/template_extracts/ to survive it.",
        "    -> The workspace outside .jaato/ is rwkl under confinement and under",
        "       no deny, so .<yourapp>/ is writable from confined contexts too.",
        "",
        "  TL;DR  ~/.jaato = daemon-global (creds + reactors, shared).  Per-session",
        "  isolation = a fresh workspace + config_root, NEVER a $HOME override.",
        "  .jaato/ is the FRAMEWORK's; your state goes in <workspace>/.<yourapp>/.",
    ]
    return data, "\n".join(lines)


def completion() -> Rendered:
    """The completion-processor capability — the OUTPUT-side script hook.

    The symmetric sibling of :func:`prefetch`, and the closest analogue to
    read first: prefetch seeds the prompt BEFORE turn 1, a completion
    processor gates ``signal_completion`` AFTER the work.
    ``docs/design/payload-schema-conventions.md`` already frames the two
    boundaries as a pair (``spawn_payload_schema`` in,
    ``completion_payload_schema`` out); scaffold documented only the input
    half, so an author wanting a self-correcting agent had to rediscover
    the shape, and the failure modes are not the ones you would guess
    (jaato #768, #769).

    Every field name, vocabulary and channel below is READ from the
    framework (``introspect.processor_schema``), never spelled here.
    """
    S = introspect.processor_schema()
    vocab = S["vocabularies"]
    data = {
        "lives_in": ".jaato/profiles/<set>/<agent>.yaml  (completion_processors:)",
        "script_at": "<config_root>/scripts/processors/<name>.py  OR  "
                     "~/.jaato/scripts/processors/<name>.py",
        "entries": [{"name": f.name, "type": f.type, "default": f.default}
                    for f in S["fields"]],
        "vocabularies": vocab,
        "validate_channels": S["channels"],
        "render_entry": "def render(payload, context) -> str | bytes",
        "validate_entry": "def validate(payload, context) -> ProcessorResult",
        "retry_budget_is_max_turns": (
            "a blocked signal_completion is retried inside the session's "
            "own max_turns; there is no second attempts knob.  What "
            "max_refusals bounds is how many times THIS GATE may block, "
            "which is a different thing and did not exist before #768."
        ),
        "does_not_terminate_on_its_own": (
            "without max_refusals the processor refuses, the agent "
            "re-claims completion, forever — seven refusals in 156 "
            "seconds, same two errors, no work in between, budget gone "
            "and no verdict (#768 rule 2)."
        ),
        "generator": "jaato-scaffold new processor --name <n> --workspace DIR",
    }
    fields = "\n".join(
        f"    {f.name:16} {f.type:22}"
        + ("(required)" if f.default == "<required>"
           else f"default {f.default!r}")
        for f in S["fields"]
    )
    lines = [
        "completion processors — the OUTPUT-side script hook "
        "(shared/completion_processors.py):",
        "",
        "  WHAT: profile-declared Python that runs when the agent calls",
        "  signal_completion, AFTER its payload passes "
        "completion_payload_schema.",
        "  A `validate` that returns errors BLOCKS the completion and hands the",
        "  agent a `validation_failed` result carrying every string, so it fixes",
        "  and calls signal_completion again.  It is the framework's",
        "  fix-until-it-passes loop.  The INPUT-side sibling is `explain",
        "  prefetch` — read it first; this is the same shape at the other",
        "  boundary.",
        "",
        "  AUTHOR IT in two places:",
        "    .jaato/profiles/<set>/<agent>.yaml   the wiring (fields below)",
        "    <config_root>/scripts/processors/<f>.py   the module (or",
        "                                     ~/.jaato/scripts/processors/<f>.py,",
        "                                     daemon-global; same loader as",
        "                                     prefetch scripts and reactors)",
        "",
        "  WIRING (a `completion_processors:` list entry):",
        fields,
        "    " + "  ".join(f"{k} → {'|'.join(v)}" for k, v in vocab.items()),
        "",
        "  MODULE contract — one or both top-level callables:",
        "    def render(payload, context) -> str | bytes",
        "        content; written to the entry's `output:` template when it",
        "        declares one, else logged for audit only.",
        "    def validate(payload, context) -> ProcessorResult",
        "        from jaato_sdk.cascade_authoring import ProcessorResult",
        "        A dict of four channels (a bare list[str] is still accepted",
        "        and read as all-errors):",
        "",
        "      channel      blocks completion?        spends a refusal?",
        "      errors       yes, per on_error         yes — one per CALL, not",
        "                                             per message",
        "      faults       once per session          never",
        "      warnings     never                     never",
        "      incomplete   never (gates is_complete  never",
        "                   on a phase:completeness",
        "                   processor)",
        "",
        "    context = RenderContext: agent_params, workspace_path,",
        "      config_root, env, session_id, registry, runtime, logger, and",
        "      tool_calls — the paired ledger of every function_call and its",
        "      response, for cross-checking a payload's claims against what",
        "      the session ACTUALLY did.",
        "",
        "  TWO WAYS TO CHECK A CLAIM, AND THE SECOND IS STRONGER.",
        "    The ledger (`context.tool_calls`) proves the agent CALLED a tool.",
        "    The filesystem proves the tool LEFT SOMETHING BEHIND — and that is",
        "    the claim a downstream stage actually depends on.  A `validate` may",
        "    read the workspace, so an agent that reports `output_file:` can be",
        "    held to it:",
        "",
        "      def validate(payload, context):",
        "          out = Path(context.workspace_path) / payload['output_file']",
        "          if not out.is_file():",
        "              return {'errors': [f\"{payload['output_file']} does not \"",
        "                                 f'exist — write it, then signal again']}",
        "          if out.stat().st_size == 0:",
        "              return {'errors': [f\"{payload['output_file']} is empty\"]}",
        "          return {}",
        "",
        "    That turns the payload into a VERIFIABLE contract between driver and",
        "    agent rather than a report taken on trust, and it costs one stat().",
        "    Two rules make it behave:",
        "      - a path from the payload is MODEL-supplied.  Resolve it under",
        "        workspace_path and reject one that escapes (`..`, an absolute",
        "        path) — a processor runs with the session's own file access.",
        "      - report the miss as `errors` (retryable — the agent can still",
        "        write the file within max_turns), NOT as `faults`, which is for",
        "        an environment the agent cannot fix.",
        "",
        "  THE RETRY BUDGET IS max_turns.  A blocked completion is retried",
        "  inside the session's own max_turns; there is no second attempts",
        "  knob, and adding one to a driver is the wrong fix (that PR was",
        "  closed on finding this mechanism).",
        "",
        "  THE TERMINUS A DRIVER WAITS ON IS NOT AGENT_COMPLETED.  An accepted",
        "  signal_completion emits AgentCompletedEvent with the payload, but",
        "  Session.complete() settles on the SESSION: first-of",
        "  {SessionTerminatedEvent, TurnCompletedEvent + the settling status}.",
        "  Both of those are emitted from ONE post-turn site, and that site is",
        "  gated on a turn having been RECORDED — which happens only when the",
        "  provider reported tokens (`if turn_data['total'] > 0`).  So a turn",
        "  whose provider reports NO usage does its work, delivers its payload,",
        "  and emits no terminal event at all: the driver waits out its own",
        "  timeout with nothing logged on either side.",
        "",
        "  THE TRAP IS `echo`.  The test double reports the spend it is TOLD",
        "  to and none otherwise, so a zero-cost harness built on it hangs at",
        "  its first stage unless the profile declares one:",
        "",
        "    plugin_configs:",
        "      echo:",
        "        usage: {prompt_tokens: 1000, output_tokens: 200}",
        "",
        "  `jaato-scaffold validate` flags the omission (echo_reports_no_usage);",
        "  the framework's own conformance profiles all pass a usage block.",
        "",
        "  THE LOOP DOES NOT TERMINATE ON ITS OWN.  Without `max_refusals`",
        "  the processor refuses, the agent re-claims completion, and round",
        "  it goes: an observed run spent SEVEN refusals in 156 seconds on",
        "  the same two errors, with no work in between, and ended with its",
        "  budget gone and no verdict.  Nothing upstream catches this —",
        "  the completion-nudge budget bounds the opposite direction (an",
        "  agent that stops WITHOUT signalling; `max_completion_nudges:`,",
        "  default 2).  Declare a ceiling:",
        "",
        "    completion_processors:",
        "      - script: scripts/processors/acceptance.py",
        "        name: acceptance",
        "        max_refusals: 3",
        "        on_exhausted: allow      # allow | fail",
        "",
        "  on_exhausted: allow  lets the unfinished completion stand (errors",
        "    downgraded to warnings) — right when something grades the run",
        "    afterwards, because a FAIL verdict carries information and a",
        "    BLOCKED arm carries none.",
        "  on_exhausted: fail   keeps blocking — right when an unfinished",
        "    completion is worse than none (it writes to a shared store, say).",
        "  Both are real choices.  The counter lives on the framework's",
        "  per-session LoadedProcessor, so a module-level global in your",
        "  script is no longer the place for it.",
        "",
        "  THE OTHER DIRECTION IS `max_completion_nudges` (#919).  Where",
        "  `max_refusals` bounds how many times a processor may BLOCK a",
        "  completion, this bounds how many times the framework re-prompts",
        "  an agent that ended its loop without calling signal_completion",
        "  at all, before giving up with NudgeExhausted:",
        "",
        "    max_completion_nudges: 4      # profile top level; default 2",
        "",
        "  Two is right for a strong tool-caller and is unchanged.  Raise",
        "  it for a model that reliably does the work and unreliably",
        "  reports it done — an audio tier that hands off, writes, narrates",
        "  the write, and burns one of its two nudges on a redundant",
        "  enter_tier has exactly one real attempt left.  Per TURN — every",
        "  turn of a conversation gets the same allowance, rather than",
        "  inheriting what an earlier turn spent (#934).  Positive integer",
        "  (0 is refused — the give-up test is `fired >= max`, so a budget",
        "  of 0 would report NudgeExhausted on sessions that completed",
        "  cleanly).  Inherits like max_turns: child overrides, else the",
        "  minimum across parents.",
        "",
        "  SEPARATE A WRONG ANSWER FROM AN ENVIRONMENT FAULT.  A missing",
        "  acceptance script, an absent agent_param, a checks timeout: no fix",
        "  the agent makes can clear those, so a retryable message about one",
        "  burns the whole budget without ever producing a verdict.  Return",
        "  them in `faults[]` — budget-exempt, and blocking only the single",
        "  round-trip the agent needs to record the fault in its payload.",
        "",
        "  NEVER RETURN [] ON A BROKEN GATE.  If your checking script exits",
        "  non-zero with no output, the gate did not RUN — returning [] there",
        "  waves the completion through on a check that never happened.  An",
        "  error path returning the same value as success is the defect class",
        "  this hook attracts most.  The framework holds the same line: a",
        "  raise, a malformed return, a missing module and a failed write all",
        "  BLOCK, spend no refusal, and are never waved through by",
        "  exhaustion.",
        "",
        "  WRITE THE STRINGS AS INSTRUCTIONS FOR THE RETRY, not as a report:",
        "  they are read by a model about to try again.  Name the failure and",
        "  what to do about it; the framework appends the attempts remaining.",
        "",
        "  GENERATE ONE with those parts already right:",
        "    jaato-scaffold new processor --name <n> --workspace DIR",
        "  and see `explain archetype processor` for what it writes.",
    ]
    return data, "\n".join(lines)


def prefetch() -> Rendered:
    """The prefetch-script capability — a DETERMINISTIC per-agent way to inject
    computed/fetched content into the system prompt BEFORE the model's first turn.

    Surfaces ``{{!py:...}}`` dynamic-instruction expansion
    (``shared/dynamic_instructions.py``), which ``scaffold explain`` otherwise
    never mentions — so an author/agent self-configuring via explain can
    actually discover it.
    """
    data = {
        "directive_mandatory": "{{!py:scripts/<name>.py [args]}}",
        "directive_optional": "{{!py?:scripts/<name>.py [args]}}",
        "lives_in": ".jaato/agents/<name>.md  (the persona)",
        "script_at": "<config_root>/scripts/<name>.py  OR  ~/.jaato/scripts/<name>.py",
        "entry": "def render(context, args) -> str",
        "context_attrs": ["agent_params", "registry", "runtime", "workspace_path",
                          "config_root", "env", "session_id", "logger", "tool_calls"],
        "example": "shared/plugins/subagent/README.md (prefetch_kyc_aml.py)",
        "agent_params_are_not_secret": (
            "agent_params are substituted into the persona, so anything put "
            "there reaches the model in its system prompt AND is persisted "
            "with the session (the rendered persona is stored so a revive "
            "restores it instead of re-running this script -- issue #787).  "
            "Pass credentials via profile env: with a pass:// / vault:// "
            "URI, which stays unresolved on disk and is resolved "
            "daemon-side at spawn."
        ),
        "runs_once": (
            "once per session, at session-prep.  A revived session restores "
            "the rendered prompt rather than re-running this script; "
            "JAATO_REVIVE_PERSONA=disk opts back into re-running it, so a "
            "side-effecting prefetch should be idempotent."
        ),
    }
    lines = [
        "prefetch scripts — deterministic per-agent session-start behaviour",
        "(dynamic-instruction `{{!py:...}}` expansion; shared/dynamic_instructions.py):",
        "",
        "  WHAT: a persona placeholder that runs a Python script at session-prep",
        "  (configure-time) and injects the returned string into the system prompt",
        "  BEFORE the model's first turn — a deterministic, no-model-round-trip way",
        "  to seed per-agent session-start context (fetched data, computed config,",
        "  mandatory pre-reads).",
        "",
        "  AUTHOR IT in two files:",
        "    .jaato/agents/<name>.md          persona — write the placeholder",
        "    <config_root>/scripts/<f>.py     the script (or ~/.jaato/scripts/<f>.py,",
        "                                     daemon-global; same loader as reactors)",
        "",
        "  DIRECTIVE forms (in the persona .md):",
        "    {{!py:scripts/<f>.py a b}}    MANDATORY — a render() failure raises",
        "        PrefetchError + aborts session-prep (the model must NOT start",
        "        without this content).",
        "    {{!py?:scripts/<f>.py ...}}   OPTIONAL — best-effort: a failure DROPS",
        "        the placeholder instead of aborting.  (A script may also return a",
        "        string starting with '[prefetch error: ...]' for a soft failure.)",
        "",
        "  SCRIPT contract:",
        "    def render(context, args) -> str",
        "      args    = whitespace-split tokens after the script name.",
        "      context = RenderContext: agent_params (the agent's params dict),",
        "        registry (registry.get_plugin('<name>') to reach a plugin),",
        "        runtime, workspace_path, config_root, env (os.environ snapshot),",
        "        session_id, logger, tool_calls (completion-time only; [] for",
        "        input-side prefetch).",
        "",
        "",
        "  NEVER PASS A CREDENTIAL AS AN agent_param.  They are substituted",
        "  into the persona, so they already reach the model in its system",
        "  prompt — and the rendered persona is PERSISTED with the session",
        "  (a revive restores it rather than re-running this script; #787).",
        "  Secrets belong in the profile's `env:` as a pass:// / vault:// URI,",
        "  which stays unresolved on disk and is resolved daemon-side.",
        "",
        "  RUNS ONCE, at session-prep.  A revived session restores the",
        "  rendered prompt instead of re-running the script; the operator can",
        "  opt back into re-rendering with JAATO_REVIVE_PERSONA=disk, which",
        "  re-runs render() against the session's ORIGINAL agent_params.  So",
        "  a prefetch with side effects (fetching, writing a file, taking a",
        "  lock) should be idempotent.",
        "",
        "  WORKED EXAMPLE: shared/plugins/subagent/README.md -> prefetch_kyc_aml.py",
        "  (a full persona placeholder + render() pulling plugin data into the prompt).",
    ]
    return data, "\n".join(lines)


# ---------------------------------------------------------- integrations

def integrations():
    """Tools this build can wire itself into, and where each one stands.

    Delegates to the same renderer the `integration` verb uses: see-then-apply
    is the framework's own shape (`explain archetypes` / `new <archetype>`),
    and one source means the listing and the verb cannot drift apart.
    """
    from . import integrations as _int
    return _int.listing()
