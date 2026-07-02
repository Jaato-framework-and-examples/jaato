"""``jaato-scaffold`` — interrogate the installed framework, validate
hand-authored assets, and scaffold new ones.  Three built-in verbs, one
introspection core (see :mod:`introspect`):

    jaato-scaffold explain [scope] [name] [--workspace DIR] [--json]
    jaato-scaffold validate <workspace-or-profile> [--set S] [--profile P] [--json]
    jaato-scaffold new ...        (see `new --help`)

``explain`` renders the introspect core by scope; ``validate`` checks an
asset against it; ``new`` emits an asset and runs it straight back through
``validate``.  Runnable as ``python -m shared.scaffold`` or via the
``jaato-scaffold`` console script.  It introspects whatever framework build is
installed in the current Python env — run it in the SAME env as the daemon you
target.

**Extension verbs.**  External packages can contribute additional verbs by
registering a :class:`api.ScaffoldVerb` under the ``jaato.scaffold_verbs``
entry-point group; the CLI discovers and mounts them at startup, and they reuse
the framework internals via :mod:`shared.scaffold.api` (introspection, the
validator, and the emit-then-validate plumbing).  A verb whose package is not
installed simply does not appear — the same convention as ``jaato.premium`` /
``jaato.premium_reactors`` elsewhere in the framework.  The premium ``compile``
verb (the Daruma invariant compiler) mounts this way, with no compiler code in
this repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

from . import explain as _explain
from . import validate as _validate


# --------------------------------------------------------------- explain

def _cmd_explain(args) -> int:
    scope = args.scope
    name = args.name
    ws = args.workspace or "."
    if scope is None:
        data, text = _explain.overview()
    elif scope == "plugins":
        data, text = _explain.plugins()
    elif scope == "plugin":
        if not name:
            print("usage: explain plugin <name>", file=sys.stderr); return 2
        data, text = _explain.plugin(name)
    elif scope == "providers":
        data, text = _explain.providers()
    elif scope == "provider":
        if not name:
            print("usage: explain provider <name>", file=sys.stderr); return 2
        data, text = _explain.provider(name)
    elif scope == "gc":
        data, text = _explain.gc()
    elif scope == "env":
        data, text = _explain.env(name)
    elif scope == "transports":
        data, text = _explain.transports()
    elif scope == "clients":
        data, text = _explain.clients()
    elif scope == "runtime":
        data, text = _explain.runtime()
    elif scope == "tiers":
        data, text = _explain.tiers()
    elif scope == "sets":
        data, text = _explain.sets(ws)
    elif scope == "profile":
        data, text = _explain.profile()
    elif scope == "paths":
        data, text = _explain.paths()
    elif scope == "prefetch":
        data, text = _explain.prefetch()
    else:
        print(f"unknown explain scope {scope!r} — one of: plugins, plugin, "
              "providers, provider, gc, env, transports, clients, runtime, "
              "tiers, sets, profile, paths", file=sys.stderr)
        return 2
    print(json.dumps(data, indent=2, default=str) if args.json else text)
    return 0


# -------------------------------------------------------------- validate

def _resolve_target(target: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Map a workspace dir OR a profile file to (workspace, set, profile_name).

    A profile file at ``<ws>/.jaato/profiles/<set>/<name>.yaml`` yields the
    set + profile name; a tier-1 file at ``.../profiles/<name>.yaml`` yields
    no set; a directory is taken as the workspace itself.
    """
    p = Path(target).resolve()
    if p.is_dir():
        return str(p), None, None
    name = p.stem
    parent = p.parent
    if parent.name == "profiles":
        return str(parent.parent.parent), None, name
    return str(parent.parent.parent.parent), parent.name, name


def _cmd_validate(args) -> int:
    workspace, derived_set, derived_name = _resolve_target(args.target)
    profile_set = args.set or derived_set
    only = args.profile or derived_name
    diags = _validate.validate_workspace(
        workspace, profile_set=profile_set, only=only)

    if args.json:
        print(json.dumps([d.as_dict() for d in diags], indent=2))
    else:
        if not diags:
            scope = f"profile '{only}'" if only else "all profiles"
            sset = f" (set {profile_set})" if profile_set else ""
            print(f"✓ {scope}{sset} valid — no findings")
        for d in diags:
            loc = f" @ {d.where}" if d.where else ""
            who = f"{d.profile}: " if d.profile else ""
            print(f"[{d.severity}] {who}{d.code}: {d.message}{loc}")
    return 1 if any(d.severity == "error" for d in diags) else 0


# ------------------------------------------------------------------- new

def _cmd_new(args) -> int:
    from . import build
    return build.run(args)


# ----------------------------------------------------- external verbs (plugins)

def _discover_external_verbs() -> list:
    """Load verbs contributed by external packages via entry points.

    Scans the ``jaato.scaffold_verbs`` group (see :mod:`api`).  Each entry point
    loads to a :class:`api.ScaffoldVerb` — an instance, or a zero-arg
    class/factory producing one.  A verb whose package is not installed simply is
    not discovered; a verb that fails to load is skipped with a warning rather
    than breaking the whole CLI.  This is how the premium ``compile`` verb (the
    Daruma invariant compiler) mounts without any compiler code living here.
    """
    import logging
    from importlib.metadata import entry_points

    log = logging.getLogger(__name__)
    from .api import VERB_ENTRY_POINT_GROUP

    try:  # entry_points(group=) is 3.10+; guard for older interpreters.
        eps = entry_points(group=VERB_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - py<3.10
        eps = entry_points().get(VERB_ENTRY_POINT_GROUP, [])

    verbs = []
    for ep in eps:
        try:
            obj = ep.load()
            verb = obj() if isinstance(obj, type) else obj
            if not getattr(verb, "name", None) or not callable(getattr(verb, "run", None)):
                log.warning("scaffold verb %r does not satisfy ScaffoldVerb; skipped", ep.name)
                continue
            verbs.append(verb)
        except Exception:
            log.warning("failed to load scaffold verb %r", ep.name, exc_info=True)
    return verbs


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="jaato-scaffold",
        description="Interrogate / validate / scaffold jaato profiles + SDK "
                    "clients against the installed framework.")
    sub = ap.add_subparsers(dest="cmd")

    pe = sub.add_parser("explain", help="interrogate the installed framework")
    pe.add_argument("scope", nargs="?",
                    help="plugins | plugin | providers | provider | gc | env "
                         "| transports | clients | runtime | tiers | sets | profile")
    pe.add_argument("name", nargs="?", help="name for plugin/provider scope")
    pe.add_argument("--workspace", help="workspace dir (for `sets`)")
    pe.add_argument("--json", action="store_true")
    pe.set_defaults(func=_cmd_explain)

    pv = sub.add_parser("validate", help="validate a profile / workspace")
    pv.add_argument("target", help="a workspace dir or a profile .yaml file")
    pv.add_argument("--set", help="JAATO_PROFILE_SET name to overlay")
    pv.add_argument("--profile", help="validate only this profile name")
    pv.add_argument("--json", action="store_true")
    pv.set_defaults(func=_cmd_validate)

    pn = sub.add_parser("new", help="scaffold a profile-set / SDK client")
    pn.add_argument("archetype", nargs="?",
                    help="client | fire | cascade | observer | host-tools | profile-set")
    pn.add_argument("--workspace", required=True, help="target workspace dir")
    pn.add_argument("--provider", help="provider name")
    pn.add_argument("--model", help="model name")
    pn.add_argument("--set", help="profile-set name (provider_model)")
    pn.add_argument("--agents", help="comma-separated agent names for a set")
    pn.add_argument("--force", action="store_true", help="overwrite existing")
    pn.add_argument("--recoverable", action="store_true",
                    help="emit the auto-reconnect client (IPCRecoveryClient for "
                         "--transport ipc, WSRecoveryClient for ws) — survives "
                         "daemon restarts — instead of the plain client")
    pn.add_argument("--transport", choices=["ipc", "ws", "in_process"], default="ipc",
                    help="client transport: 'ipc' (local daemon over a Unix socket, "
                         "default), 'ws' (remote daemon over ws:// / wss:// — "
                         "requires --url), or 'in_process' (embedded — runs the "
                         "runtime + session in THIS process, no daemon/socket; "
                         "incompatible with --recoverable).")
    pn.add_argument("--url", help="WebSocket URL for --transport ws (ws:// or wss://)")
    pn.add_argument("--token", help="bearer token for --transport ws (optional)")
    pn.add_argument("--ca", help="CA-bundle path for --transport ws wss:// with a "
                                 "self-signed / dev cert (scoped ca=, never os.environ)")
    pn.add_argument("--json", action="store_true")
    pn.set_defaults(func=_cmd_new)

    # External verbs (e.g. the premium `compile` verb) — discovered via the
    # `jaato.scaffold_verbs` entry-point group.  Built-in names win on collision.
    _builtin = {"explain", "validate", "new"}
    for verb in _discover_external_verbs():
        if verb.name in _builtin:
            continue
        pv_ext = sub.add_parser(verb.name, help=getattr(verb, "help", None))
        verb.configure(pv_ext)
        pv_ext.set_defaults(func=verb.run)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
