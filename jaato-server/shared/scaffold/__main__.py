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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from . import explain as _explain
from . import validate as _validate


# --------------------------------------------------------------- explain
#
# Scopes are ONE TABLE, not an if/elif chain and not four tables.  The chain is
# how `explain` came to advertise "4 client archetypes" with no scope behind it
# (jaato #716): adding a scope meant finding the right rung of a 20-branch
# ladder AND its entry in two help strings, so the cheap path was to add
# nothing.  Splitting the ladder into four tables plus a stray ``elif scope ==
# "profile"`` fixed the ladder and left the help hand-typed, which is how
# `integrations` came to be dispatched and advertised nowhere (#994): #906
# registered it in one table and nobody edited the string.
#
# So the table below is the ONLY place a topic is declared.  The help line, the
# unknown-scope error, the argparse `--help` and the dispatch all read it, and
# `shared/tests/test_explain_scopes_are_derived_994.py` fails if a sixth
# dispatch shape appears without being wired in.


@dataclass(frozen=True)
class ExplainScope:
    """One `explain` topic: what renders it, how it is called, what it takes.

    Attributes:
        render: the renderer for the argument-less form of the scope.
        kind: which calling convention it takes — a key of :data:`_SCOPE_KINDS`,
            which is what turns this declaration into a dispatch.  A kind no
            handler implements is a build failure, not a runtime surprise.
        arg: the argument hint rendered beside the scope in the help line and
            in the ``usage:`` message a ``named`` scope prints when called
            without one (``"<name>"``, ``"[<filter>]"``, ...).  Empty for a
            scope that takes no argument.  Presentation lives HERE rather than
            in a parallel string, so ``profile [<name>]`` cannot survive the
            scope being renamed.
        render_named: the second renderer of an ``optional_named`` scope — what
            runs when a name IS supplied.  ``None`` for every other kind.
    """

    render: Callable[..., Any]
    kind: str = "simple"
    arg: str = ""
    render_named: Optional[Callable[..., Any]] = None


#: Every `explain` topic, in the order the help line lists them.
#:
#: Kinds:
#:   ``simple``          rendered with no argument.
#:   ``filter``          takes an OPTIONAL filter as the name argument.
#:   ``named``           REQUIRES a name.  A renderer signals "no such name" by
#:                       returning a data dict carrying an ``error`` key; the
#:                       CLI turns that into a stderr message and exit 2, so a
#:                       caller that typo'd a name never mistakes the miss for
#:                       documentation.
#:   ``workspace``       rendered AGAINST A WORKSPACE — it reports on files on
#:                       disk, so the ``--workspace`` value is the argument.
#:   ``optional_named``  both: bare, it is the SCHEMA; with a name, what that
#:                       named unit inherits and costs, resolved against the
#:                       workspace.
_SCOPES = {
    "plugins": ExplainScope(_explain.plugins),
    "plugin": ExplainScope(_explain.plugin, "named", "<name>"),
    "commands": ExplainScope(_explain.commands),
    "providers": ExplainScope(_explain.providers),
    "provider": ExplainScope(_explain.provider, "named", "<name>"),
    "gc": ExplainScope(_explain.gc),
    "env": ExplainScope(_explain.env, "filter", "[<filter>]"),
    "events": ExplainScope(_explain.events, "filter", "[<filter>]"),
    # The hint says "or" rather than "|": the help line separates topics with
    # "|", so a hint carrying one is unreadable there and unparseable by
    # anything reading the line back.
    "event": ExplainScope(_explain.event, "named", "<NAME or wire.value>"),
    "transports": ExplainScope(_explain.transports),
    "clients": ExplainScope(_explain.clients),
    "runtime": ExplainScope(_explain.runtime),
    "tiers": ExplainScope(_explain.tiers),
    "integrations": ExplainScope(_explain.integrations),
    "sets": ExplainScope(_explain.sets, "workspace"),
    "agents": ExplainScope(_explain.agents, "workspace"),
    "services": ExplainScope(_explain.services, "workspace"),
    # ``profile`` alone is the SCHEMA; ``profile <name>`` is what that named
    # profile INHERITS and what it costs per turn.  A profile file states what
    # it adds and never what it inherits, so the instruction tax is invisible
    # at authoring time and shows up later as a budget refusal.
    "profile": ExplainScope(_explain.profile, "optional_named", "[<name>]",
                            render_named=_explain.profile_cost),
    "paths": ExplainScope(_explain.paths),
    "prefetch": ExplainScope(_explain.prefetch),
    "completion": ExplainScope(_explain.completion),
    "archetypes": ExplainScope(_explain.archetypes),
    "archetype": ExplainScope(_explain.archetype, "named", "<name>"),
}


class _ScopeUsageError(Exception):
    """A scope was named correctly and invoked wrongly (missing / unknown name).

    Carries the message to print on stderr and the exit code, so every kind
    handler reports the same way and ``_cmd_explain`` keeps ONE error exit.
    """

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.message = message
        self.code = code


def _scope_usage(scope: str, spec: ExplainScope) -> str:
    """The ``usage:`` line for *scope* — derived, never a second copy."""
    return f"explain {scope} {spec.arg}".rstrip()


def _call_simple(spec, scope, name, ws):
    return spec.render()


def _call_filter(spec, scope, name, ws):
    return spec.render(name)


def _call_workspace(spec, scope, name, ws):
    return spec.render(ws)


def _call_named(spec, scope, name, ws):
    if not name:
        raise _ScopeUsageError(f"usage: {_scope_usage(scope, spec)}")
    data, text = spec.render(name)
    if isinstance(data, dict) and "error" in data:
        raise _ScopeUsageError(text)
    return data, text


def _call_optional_named(spec, scope, name, ws):
    return spec.render_named(name, ws) if name else spec.render()


#: How each :attr:`ExplainScope.kind` is invoked.  Dispatch is a lookup here,
#: so a scope declaring a kind with no handler fails loudly at the one place
#: that would otherwise grow a sixth ``elif``.
_SCOPE_KINDS = {
    "simple": _call_simple,
    "filter": _call_filter,
    "named": _call_named,
    "workspace": _call_workspace,
    "optional_named": _call_optional_named,
}


def _scopes_help() -> str:
    """The ``one of:`` line — every registered topic with its argument hint.

    Derived from :data:`_SCOPES`, so a topic added there appears in the
    unknown-scope error and in ``explain --help`` without anyone editing prose.
    """
    return " | ".join(
        f"{scope} {spec.arg}".rstrip() for scope, spec in _SCOPES.items())


def _workspace_arg_help() -> str:
    """``--workspace``'s help — the topics that actually read it.

    Derived for the same reason the scope list is: this said "(for `sets`)"
    while three more workspace-reading topics had been added beside it.
    """
    readers = [n for n, s in _SCOPES.items()
               if s.kind in ("workspace", "optional_named")]
    return "workspace dir (for " + ", ".join(f"`{n}`" for n in readers) + ")"


_SCOPES_HELP = _scopes_help()

# Derived views of the one table, kept because callers and tests reach for
# them by name.  Each is a projection, never a second declaration: a topic
# added to _SCOPES appears here, and nothing can appear here without being
# dispatched.
_SIMPLE_SCOPES = {n: s.render for n, s in _SCOPES.items() if s.kind == "simple"}
_FILTER_SCOPES = {n: s.render for n, s in _SCOPES.items() if s.kind == "filter"}
_WORKSPACE_SCOPES = {n: s.render for n, s in _SCOPES.items()
                     if s.kind == "workspace"}
_NAMED_SCOPES = {n: (s.render, _scope_usage(n, s)) for n, s in _SCOPES.items()
                 if s.kind == "named"}


_DEPS_WORDS = ("dependencies", "deps")


def _take_deps_word(scope, name, extra):
    """Pull the optional `dependencies` word out of the query, wherever it sits.

    Dependencies are a FACET of every scope rather than a scope of their own —
    a provider imports packages, a plugin shells out, the framework is two
    distributions that drift — so the word is appended to whatever you were
    already asking:

        explain dependencies
        explain provider openrouter dependencies
        explain plugin cli deps

    Accepted in any position after the verb, because a reader who types it
    first is asking the same question as one who types it last.
    """
    words = [w for w in (scope, name, extra) if w]
    kept = [w for w in words if w not in _DEPS_WORDS]
    asked = len(kept) != len(words)
    kept += [None, None]
    return kept[0], kept[1], asked


def _cmd_explain(args) -> int:
    """Render one `explain` topic.

    Every topic is looked up in :data:`_SCOPES` and invoked through the handler
    its ``kind`` names, so the set of topics this dispatches is by construction
    the set the help line advertises (#994).
    """
    scope, name, deps = _take_deps_word(
        args.scope, args.name, getattr(args, "extra", None))
    ws = args.workspace or "."
    if deps:
        from . import dependencies as _deps
        data, text = _deps.render(scope, name)
        print(json.dumps(data, indent=2) if args.json else text)
        return 0
    if scope is None:
        data, text = _explain.overview()
    elif scope in _SCOPES:
        spec = _SCOPES[scope]
        try:
            data, text = _SCOPE_KINDS[spec.kind](spec, scope, name, ws)
        except _ScopeUsageError as exc:
            print(exc.message, file=sys.stderr)
            return exc.code
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

def _cmd_integration(args) -> int:
    from . import integrations as _install
    names = _install.available()
    if not names:
        print("this build ships no integrations", file=sys.stderr)
        return 1
    if not args.name:
        # The bare verb LISTS rather than guessing which one you meant — with
        # more than one shipped, picking for you would be a coin toss.
        data, text = _install.listing()
        print(json.dumps(data, indent=2) if args.json else text)
        return 0
    name = args.name
    # --user and --workspace are mutually exclusive, so "not --workspace" IS
    # user scope; --user is accepted so the default can be stated out loud.
    dest = _install.target_dir(name, user=not args.workspace,
                               workspace=args.workspace)
    changed, lines = _install.install(name, dest, force=args.force, dry_run=args.dry_run)
    if args.json:
        state, detail = _install.compare(name, dest)
        print(json.dumps({"asset": name, "dest": str(dest), "changed": changed,
                          "state": state, "detail": detail,
                          "version": _install.framework_version()}, indent=2))
        return 0
    for line in lines:
        print(line)
    # A refusal is not a crash: the operator asked a reasonable thing and the
    # answer is "there is already one there".  Non-zero so a script notices.
    return 0 if (changed or args.dry_run) else 1


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
    pe.add_argument("extra", nargs="?",
                    help="the optional word `dependencies` (or `deps`) — a facet "
                         "of any scope: what it needs, what is installed, and "
                         "whether this environment agrees with itself")
    pe.add_argument("--workspace", help=_workspace_arg_help())
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
    pn.add_argument("--no-gate", action="store_true", dest="no_gate",
                    help="for `new sweep`: do NOT emit the completion gate "
                         "(acceptance.sh + the processor + the profile wiring "
                         "it needs). The gate is emitted by default because a "
                         "sweep's arms are graded — 'did this arm meet the "
                         "criteria' is the measurement, not a nicety. Pass "
                         "this for a sweep that grades nothing.")
    pn.add_argument("--gate-name", metavar="NAME", dest="gate_name",
                    help="stem shared by the gate's four files (default "
                         "'acceptance'): the processor module, the completion "
                         "schema, the profile, and the profile entry's `name:`.")
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

    pi = sub.add_parser(
        "integration", help="wire jaato into a tool you work in (bare: list them)",
        description="An integration is jaato's side of a contract with another "
                    "tool — today `claude-code`, which installs the jaato-sdk "
                    "skill where Claude Code looks for skills.  Each copy is "
                    "stamped with the build it came from, so `jaato-doctor` can "
                    "say when one has gone stale.  With no name, lists what this "
                    "build ships and where each one stands.")
    pi.add_argument("name", nargs="?", default=None,
                    help="integration name (omit to list)")
    scope = pi.add_mutually_exclusive_group()
    scope.add_argument("--user", action="store_true",
                       help="apply under $HOME — every repo on this machine "
                            "(the default; accepted explicitly so a script can "
                            "say what it means)")
    scope.add_argument("--workspace", default=None,
                       help="apply under DIR instead of $HOME — this project only")
    pi.add_argument("--force", action="store_true",
                    help="overwrite an existing copy")
    pi.add_argument("--dry-run", action="store_true",
                    help="print what would be written, write nothing")
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=_cmd_integration)


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
