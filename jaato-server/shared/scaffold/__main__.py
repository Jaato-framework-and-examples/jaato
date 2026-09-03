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
installed simply does not appear — the same convention as ``jaato.premium``
and ``jaato.extensions`` elsewhere in the framework.  (There is no
``jaato.premium_reactors`` group: reactors mount as the ``reactors``
entry in ``jaato.extensions``, and their RULES load from directories
— ``~/.jaato/reactors/`` and ``<workspace>/.jaato/reactors/`` — not from
entry points.)  The premium ``compile``
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
#
# Scopes are TABLES, not an if/elif chain.  The chain is how `explain` came to
# advertise "4 client archetypes" with no scope behind it (jaato #716): adding
# a scope meant finding the right rung of a 20-branch ladder AND its entry in
# two help strings, so the cheap path was to add nothing.

#: Scopes rendered with no argument.
_SIMPLE_SCOPES = {
    "plugins": _explain.plugins,
    "providers": _explain.providers,
    "gc": _explain.gc,
    "transports": _explain.transports,
    "clients": _explain.clients,
    "runtime": _explain.runtime,
    "tiers": _explain.tiers,
    "paths": _explain.paths,
    "prefetch": _explain.prefetch,
    "completion": _explain.completion,
    "commands": _explain.commands,
    "archetypes": _explain.archetypes,
}

#: Scopes REQUIRING a name, with the usage line printed when it is missing.
#: A renderer here signals "no such name" by returning a data dict carrying an
#: ``error`` key; the CLI turns that into a stderr message and exit 2, so a
#: caller that typo'd a name never mistakes the miss for documentation.
_NAMED_SCOPES = {
    "plugin": (_explain.plugin, "explain plugin <name>"),
    "provider": (_explain.provider, "explain provider <name>"),
    "event": (_explain.event, "explain event <NAME|wire.value>"),
    "archetype": (_explain.archetype, "explain archetype <name>"),
}

#: Scopes taking an OPTIONAL filter as the name argument.
_FILTER_SCOPES = {"env": _explain.env, "events": _explain.events}

_SCOPES_HELP = ("plugins | plugin | commands | providers | provider | gc | env | events | "
                "event | transports | clients | runtime | tiers | sets | "
                "profile [<name>] | paths | prefetch | completion | archetypes | "
                "archetype")


def _cmd_explain(args) -> int:
    scope = args.scope
    name = args.name
    ws = args.workspace or "."
    if scope is None:
        data, text = _explain.overview()
    elif scope in _SIMPLE_SCOPES:
        data, text = _SIMPLE_SCOPES[scope]()
    elif scope in _FILTER_SCOPES:
        data, text = _FILTER_SCOPES[scope](name)
    elif scope in _NAMED_SCOPES:
        render, usage = _NAMED_SCOPES[scope]
        if not name:
            print(f"usage: {usage}", file=sys.stderr)
            return 2
        data, text = render(name)
        if isinstance(data, dict) and "error" in data:
            print(text, file=sys.stderr)
            return 2
    elif scope == "sets":
        data, text = _explain.sets(ws)
    elif scope == "profile":
        # ``profile`` alone is the SCHEMA; ``profile <name>`` is what that
        # named profile INHERITS and what it costs per turn.  A profile file
        # states what it adds and never what it inherits, so the instruction
        # tax is invisible at authoring time and shows up later as a budget
        # refusal.
        data, text = (_explain.profile_cost(name, ws) if name
                      else _explain.profile())
    else:
        print(f"unknown explain scope {scope!r} — one of: {_SCOPES_HELP}",
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


def _is_canonical_profile_layout(p: Path) -> bool:
    """True if ``p`` lives under a real ``<ws>/.jaato/profiles[/<set>]/`` tree.

    Only such files can be resolved via ``validate_workspace`` (inherits + set
    overlay).  A file outside this layout (a docs example, an ad-hoc path) must
    be validated directly, or it silently resolves to a bogus workspace where
    ``discover_profiles`` finds nothing and reports a false "valid".
    """
    par = p.parent
    if par.name == "profiles" and par.parent.name == ".jaato":
        return True  # <ws>/.jaato/profiles/<name>.yaml
    if par.parent.name == "profiles" and par.parent.parent.name == ".jaato":
        return True  # <ws>/.jaato/profiles/<set>/<name>.yaml
    return False


def _cmd_validate(args) -> int:
    target = Path(args.target)
    profile_set = args.set
    only = args.profile
    if target.is_file() and not _is_canonical_profile_layout(target.resolve()):
        # Standalone profile file — validate it directly (see
        # ``validate_profile_file``); the workspace path would find nothing and
        # falsely report "valid".
        diags = _validate.validate_profile_file(str(target))
        scope = f"profile file '{target.name}'"
    else:
        workspace, derived_set, derived_name = _resolve_target(args.target)
        profile_set = args.set or derived_set
        only = args.profile or derived_name
        diags = _validate.validate_workspace(
            workspace, profile_set=profile_set, only=only)
        scope = f"profile '{only}'" if only else "all profiles"

    if args.json:
        print(json.dumps([d.as_dict() for d in diags], indent=2))
    else:
        if not diags:
            sset = f" (set {profile_set})" if profile_set else ""
            print(f"✓ {scope}{sset} valid — no findings")
        for d in diags:
            loc = f" @ {d.where}" if d.where else ""
            who = f"{d.profile}: " if d.profile else ""
            tier = f"[{d.tier}] " if d.tier else ""
            print(f"[{d.severity}] {tier}{who}{d.code}: {d.message}{loc}")
    return 1 if any(d.severity == "error" for d in diags) else 0


# ------------------------------------------------------------------- new

def _new_epilog() -> str:
    """The ``new --help`` epilog: what each archetype WRITES.

    ``new --help`` used to list every flag and not one line describing the
    output, so the only way to learn what the generator produced was to run it
    against a throwaway directory and diff, or to read the templates (jaato
    #716).  Sourced from the same registry ``explain archetypes`` renders.
    """
    from . import archetypes as _archetypes
    # Every documented archetype, never a hand-kept subset: the epilog used
    # to enumerate profile-set + the client templates, so an archetype that
    # was neither (the processor generator) would have been absent from
    # `new --help` while `new` accepted it — the same shape of drift that
    # made the banner advertise four archetypes out of six (jaato #716).
    docs = [_archetypes.ARCHETYPES[n] for n in sorted(_archetypes.ARCHETYPES)]
    width = max(len(d.name) for d in docs)
    lines = ["what each archetype writes into --workspace:"]
    for d in docs:
        paths = ", ".join(e.render_path(archetype=d.name, set="<set>",
                                        agent="<agent>", name="<name>")
                          for e in d.writes)
        lines.append(f"  {d.name.ljust(width)}  {paths}")
    lines += [
        "",
        "what is IN those files, and which parts you must edit:",
        "  jaato-scaffold explain archetypes",
        "  jaato-scaffold explain archetype <name>",
        "",
        "the exact tree for YOUR flags, written nowhere:",
        "  jaato-scaffold new <name> --workspace DIR ... --dry-run",
    ]
    return "\n".join(lines)


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
    pe.add_argument("scope", nargs="?", help=_SCOPES_HELP)
    pe.add_argument("name", nargs="?",
                    help="name for plugin/provider/event/archetype scope, or a "
                         "filter for env/events")
    pe.add_argument("--workspace", help="workspace dir (for `sets`)")
    pe.add_argument("--json", action="store_true")
    pe.set_defaults(func=_cmd_explain)

    pv = sub.add_parser("validate", help="validate a profile / workspace")
    pv.add_argument("target", help="a workspace dir or a profile .yaml file")
    pv.add_argument("--set", help="JAATO_PROFILE_SET name to overlay")
    pv.add_argument("--profile", help="validate only this profile name")
    pv.add_argument("--json", action="store_true")
    pv.set_defaults(func=_cmd_validate)

    from . import archetypes as _archetypes
    pn = sub.add_parser(
        "new", help="scaffold a profile-set / SDK client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Scaffold an asset, then re-check it (a profile-set is run "
                    "back through the validator; a client is compile-checked).",
        epilog=_new_epilog())
    pn.add_argument("archetype", nargs="?",
                    help="one of: " + " | ".join(_archetypes.accepted())
                         + "  (default: profile-set)")
    pn.add_argument("--workspace", required=True, help="target workspace dir")
    pn.add_argument("--provider", help="provider name")
    pn.add_argument("--model", help="model name")
    pn.add_argument("--set", help="profile-set name (provider_model)")
    pn.add_argument("--agents", help="comma-separated agent names for a set")
    pn.add_argument("--name", help="processor name for `new processor` — the "
                                   "module stem under "
                                   ".jaato/scripts/processors/ and the "
                                   "`name:` of its profile entry")
    pn.add_argument("--force", action="store_true", help="overwrite existing")
    pn.add_argument("--secrets", metavar="MODE",
                    help="how profiles reference the provider credential: "
                         "'env' (default — ${<PROVIDER>_API_KEY} interpolation, "
                         "runs on a public checkout), 'none' (omit api_key; the "
                         "provider reads its own env var), or a resolver scheme "
                         "like 'pass' / 'pass://' (secret URI — needs an "
                         "out-of-tree resolver plugin, e.g. jaato-premium). The "
                         "choice is recorded in .jaato/scaffold.json so later "
                         "`new` calls stay consistent.")
    pn.add_argument("--secret-path", metavar="TEMPLATE", dest="secret_path",
                    help="path template for --secrets <scheme> URIs "
                         "(default 'jaato/{provider}/api-key'; '{provider}' is "
                         "substituted).")
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
    pn.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="print the file tree this invocation WOULD write "
                         "— annotated with what each file is for — and write "
                         "nothing.  Existence checks still read the real "
                         "workspace, so it distinguishes a created file from "
                         "an appended-to one exactly as the real run would.")
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
