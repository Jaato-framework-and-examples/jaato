"""``jaato-scaffold`` — interrogate the installed framework, validate
hand-authored assets, and scaffold new ones.  Three verbs, one
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
    elif scope == "sets":
        data, text = _explain.sets(ws)
    elif scope == "profile":
        data, text = _explain.profile()
    else:
        print(f"unknown explain scope {scope!r} — one of: plugins, plugin, "
              "providers, provider, gc, env, transports, clients, sets, profile",
              file=sys.stderr)
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
                         "| transports | clients | sets | profile")
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
                    help="client | fire | cascade | observer | profile-set")
    pn.add_argument("--workspace", required=True, help="target workspace dir")
    pn.add_argument("--provider", help="provider name")
    pn.add_argument("--model", help="model name")
    pn.add_argument("--set", help="profile-set name (provider_model)")
    pn.add_argument("--agents", help="comma-separated agent names for a set")
    pn.add_argument("--force", action="store_true", help="overwrite existing")
    pn.add_argument("--recoverable", action="store_true",
                    help="emit IPCRecoveryClient (auto-reconnect, survives daemon "
                         "restarts) instead of the plain IPCClient")
    pn.add_argument("--json", action="store_true")
    pn.set_defaults(func=_cmd_new)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
