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

**Extension topics.**  The same idea for the one verb an agent actually reads.
A package registering an :class:`api.ExplainTopic` under
``jaato.scaffold_topics`` either adds a topic of its own (``explain reactors``)
or appends an attributed SECTION to a built-in one (``extends = "paths"``), and
either way it reaches the dispatch, the overview banner, ``--help`` and the
unknown-scope error through the same merged table — so a contributed topic
cannot be advertised without being served, or served without being advertised
(#994's rule, one layer out).  Built-in names win on collision, and a topic
that fails to load is skipped with a warning.

It exists because a package that contributes a daemon EXTENSION has something
to say and nothing to run, so the verb seam could only ever have offered it a
subcommand nobody would think to type.  The measured cost of not having it:
premium's reactor engine reads four rule-file tiers, and the only surface that
named any of them named the one under the DAEMON's home — so a session asked
to add a reactor could not find out where the file goes, what the schema is,
or what the script must define, and read the engine's source instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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
        blurb: the trailing ``# ...`` note the overview banner prints beside
            this topic, for the topics whose name does not say what they are.
            Empty for the ones that do.  It lives HERE for the same reason
            ``arg`` does: the banner used to be hand-typed prose and had
            drifted to advertise 21 of 23 topics (#1006), so every string the
            banner prints is now a field of the entry it describes.
    """

    render: Callable[..., Any]
    kind: str = "simple"
    arg: str = ""
    render_named: Optional[Callable[..., Any]] = None
    blurb: str = ""


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
    "env": ExplainScope(_explain.env, "filter", "[<filter>]",
                        blurb="vars the daemon + plugins READ"),
    "events": ExplainScope(_explain.events, "filter", "[<filter>]",
                           blurb="the client/server protocol"),
    # The hint says "or" rather than "|": the help line separates topics with
    # "|", so a hint carrying one is unreadable there and unparseable by
    # anything reading the line back.
    "event": ExplainScope(_explain.event, "named", "<NAME or wire.value>",
                          blurb="one event's fields + docstring"),
    "transports": ExplainScope(_explain.transports),
    "clients": ExplainScope(_explain.clients),
    "runtime": ExplainScope(_explain.runtime),
    "tiers": ExplainScope(_explain.tiers),
    "integrations": ExplainScope(_explain.integrations,
                                 blurb="tools jaato can wire into"),
    # The only topic that reaches the network: our own indexes, asked what
    # they carry.  Its own topic rather than a section of `dependencies`,
    # which is an OFFLINE read of the installed tree and must stay one.
    "releases": ExplainScope(_explain.releases,
                             blurb="newer builds on PyPI / TestPyPI"),
    "sets": ExplainScope(_explain.sets, "workspace"),
    "agents": ExplainScope(_explain.agents, "workspace",
                           blurb="the PERSONA layer (.jaato/agents/)"),
    "services": ExplainScope(_explain.services, "workspace",
                             blurb="named HTTP APIs (.jaato/services/)"),
    # ``profile`` alone is the SCHEMA; ``profile <name>`` is what that named
    # profile INHERITS and what it costs per turn.  A profile file states what
    # it adds and never what it inherits, so the instruction tax is invisible
    # at authoring time and shows up later as a budget refusal.
    "profile": ExplainScope(_explain.profile, "optional_named", "[<name>]",
                            render_named=_explain.profile_cost,
                            blurb="a session's CAPABILITIES"),
    # ``oversight`` alone is the framework's Article 14 measures -- the two
    # stop verbs, the decision gate, the built-in constraints; with a name
    # it is what THAT profile has armed, resolved against the workspace.
    "oversight": ExplainScope(_explain.oversight, "optional_named", "[<profile>]",
                              render_named=_explain.oversight_profile,
                              blurb="the HUMAN-OVERSIGHT measures (EU AI Act Art. 14)"),
    # ``audit`` alone is the RECORD-KEEPING CONTRACT -- which events are
    # recorded, what each carries and which store it lands in; with a name it
    # is the concrete paths that profile writes to and what its
    # ``record_keeping:`` block says.  Art. 13(3)(f) asks the instructions for
    # use to describe exactly this, and five stores recorded without any of
    # them saying what was guaranteed.
    "audit": ExplainScope(_explain.audit, "optional_named", "[<profile>]",
                          render_named=_explain.audit_profile,
                          blurb="the AUDIT RECORD (EU AI Act Arts. 12, 19)"),
    "paths": ExplainScope(_explain.paths),
    "prefetch": ExplainScope(_explain.prefetch),
    "completion": ExplainScope(_explain.completion,
                               blurb="the OUTPUT-side hook"),
    "archetypes": ExplainScope(_explain.archetypes,
                               blurb="what `new` WRITES"),
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

    Derived from :data:`_SCOPES` PLUS the contributed topics, so a topic
    appears in the unknown-scope error and in ``explain --help`` without anyone
    editing prose — and so does one an installed package contributed.  A
    contributed topic that dispatches and is advertised nowhere is the gap this
    seam exists to close, reproduced one layer in.
    """
    return " | ".join(
        f"{scope} {spec.arg}".rstrip() for scope, spec in _SCOPES.items())


def _all_scopes_help() -> str:
    """:func:`_scopes_help` PLUS the contributed topics — what a READER is told.

    The built-in line stays a constant derived from :data:`_SCOPES` (#994's
    guard pins that, and discovery must not run at import time), so this is the
    live sibling rather than a replacement: contributed topics are appended, in
    the same ``scope <hint>`` spelling, marked with the distribution that
    supplied them.  A contributed topic that dispatches and is advertised
    nowhere is #994 reproduced one layer out, which is the whole complaint this
    seam answers.
    """
    parts = [_SCOPES_HELP]
    for name, topic in external_own_topics().items():
        hint = f"{name} {getattr(topic, 'arg', '')}".rstrip()
        dist = _topic_dist(topic)
        parts.append(f"{hint} ({dist})" if dist else hint)
    return " | ".join(parts)


def _workspace_readers() -> list:
    """The topics whose renderer is handed the ``--workspace`` value.

    One predicate, two consumers: ``--workspace``'s own help text and the
    overview banner, which appends ``[--workspace DIR]`` to exactly these
    topics.  Written once because the hand-typed banner put that hint on
    ``sets`` alone while ``agents`` and ``services`` read the workspace just
    as much (#1006).
    """
    readers = [n for n, s in _SCOPES.items()
               if s.kind in ("workspace", "optional_named")]
    # A contributed topic DECLARES whether it reads the workspace: every topic
    # is handed the value, so nothing but the topic knows whether it uses it.
    readers += [n for n, t in external_own_topics().items()
                if getattr(t, "reads_workspace", False)]
    return readers


def _workspace_arg_help() -> str:
    """``--workspace``'s help — the topics that actually read it.

    Derived for the same reason the scope list is: this said "(for `sets`)"
    while three more workspace-reading topics had been added beside it.
    """
    readers = _workspace_readers()
    return "workspace dir (for " + ", ".join(f"`{n}`" for n in readers) + ")"


def scope_catalog() -> list:
    """Every `explain` topic as data — what the overview banner renders from.

    The banner is the THIRD surface that used to spell the topic list by hand
    (#1006), after the ``one of:`` error and argparse's ``--help`` (#994).  It
    advertised 21 topics while :data:`_SCOPES` carried 23, with ``env``,
    ``event`` and ``events`` dispatching and named nowhere.  Exporting the
    table as data — rather than letting :mod:`explain` import the CLI's
    private dict — keeps the banner derived without making every field of
    ``ExplainScope`` part of that module's contract.

    Returns:
        One dict per topic, in table order: ``scope``, ``arg`` (the argument
        hint), ``kind``, ``reads_workspace`` (whether ``--workspace`` reaches
        its renderer) and ``blurb``.
    """
    readers = set(_workspace_readers())
    rows = [{"scope": name,
             "arg": spec.arg,
             "kind": spec.kind,
             "reads_workspace": name in readers,
             "blurb": spec.blurb,
             "contributed_by": ""}
            for name, spec in _SCOPES.items()]
    # Contributed topics ride the SAME catalog rather than a parallel one, so
    # the banner cannot advertise a set the dispatcher does not serve.  They
    # carry ``contributed_by`` so a reader — and `--json` — can tell the
    # framework's own answer from an installed package's, which is what decides
    # whose source to go and read.
    rows += [{"scope": name,
              "arg": getattr(t, "arg", ""),
              "kind": "external",
              "reads_workspace": name in readers,
              "blurb": getattr(t, "help", ""),
              "contributed_by": _topic_dist(t)}
             for name, t in external_own_topics().items()]
    return rows



# Derived views of the one table, kept because callers and tests reach for
# them by name.  Each is a projection, never a second declaration: a topic
# added to _SCOPES appears here, and nothing can appear here without being
# dispatched.
_SCOPES_HELP = _scopes_help()

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


def _scope_renderer(scope: str):
    """Resolve a topic name to the thing that renders it, or ``None``.

    The ONE lookup ``_cmd_explain`` performs.  Built-ins are consulted first
    and win on a name collision; contributed topics follow.  Both tiers hand
    back the SAME shape — a callable taking ``(scope, name, ws)`` — so the
    dispatch does not branch on which tier answered, and a topic cannot be
    reachable through a path the help line was not derived from.

    Why a resolver rather than two membership tests in ``_cmd_explain``: #994
    was a topic that dispatched through its own rung and so could never reach
    the derived help.  A second rung for contributed topics would be the same
    defect wearing the fix as a disguise, so there is one rung and the tiers
    are inside it.
    """
    spec = _SCOPES.get(scope)
    if spec is not None:
        return partial(_SCOPE_KINDS[spec.kind], spec)
    topic = external_own_topics().get(scope)
    if topic is not None:
        return partial(_render_external_topic, topic)
    return None


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
    render = None if scope is None else _scope_renderer(scope)
    if scope is None:
        data, text = _explain.overview()
    elif render is not None:
        try:
            data, text = render(scope, name, ws)
        except _ScopeUsageError as exc:
            print(exc.message, file=sys.stderr)
            return exc.code
        # Contributed SECTIONS append to whatever rendered — a built-in or a
        # contributed topic alike — so two packages can answer about one
        # subject without either having to know the other exists.
        data, text = _append_topic_extensions(scope, name, ws, data, text)
    else:
        print(f"unknown explain scope {scope!r} — one of: {_all_scopes_help()}",
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
    try:
        dest = _install.target_dir(name, user=not args.workspace,
                                   workspace=args.workspace)
    except _install.IntegrationManifestError as exc:
        # A packaging error in what we shipped, not a mistake the operator
        # made.  Say so instead of installing to a guessed path.
        print(f"{exc}", file=sys.stderr)
        return 1
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


# ------------------------------------------------- external topics (plugins)

#: Loaded external topics, or ``None`` before the first discovery.
#:
#: Cached because discovery IMPORTS the contributing modules, and the merged
#: table is read several times in one run (the banner, the ``--help`` line, the
#: unknown-scope error, the dispatch).  :func:`reset_external_topics` clears it;
#: nothing but a test has a reason to.
_EXTERNAL_TOPICS: "Optional[list]" = None


def reset_external_topics() -> None:
    """Drop the discovery cache so the next read re-scans the entry points."""
    global _EXTERNAL_TOPICS
    _EXTERNAL_TOPICS = None


def _discover_external_topics() -> list:
    """Load ``explain`` topics contributed by external packages.

    Scans the ``jaato.scaffold_topics`` group (see :mod:`api`), the topic-shaped
    sibling of :func:`_discover_external_verbs` and deliberately its twin in
    every failure behaviour: an entry point loads to an :class:`api.ExplainTopic`
    (an instance, or a zero-arg class/factory producing one), a package that is
    not installed contributes nothing, and a topic that fails to load is skipped
    with a warning rather than taking the CLI down.  A diagnostic that cannot
    survive one broken contributor is not a diagnostic.

    Only ``name`` and ``render`` are required of a contributor; ``help`` /
    ``arg`` / ``extends`` / ``reads_workspace`` are read with defaults, so the
    smallest useful topic is two attributes and one method.
    """
    global _EXTERNAL_TOPICS
    if _EXTERNAL_TOPICS is not None:
        return _EXTERNAL_TOPICS

    import logging
    from importlib.metadata import entry_points

    log = logging.getLogger(__name__)
    from .api import TOPIC_ENTRY_POINT_GROUP

    try:  # entry_points(group=) is 3.10+; guard for older interpreters.
        eps = entry_points(group=TOPIC_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - py<3.10
        eps = entry_points().get(TOPIC_ENTRY_POINT_GROUP, [])

    topics = []
    for ep in eps:
        try:
            obj = ep.load()
            topic = obj() if isinstance(obj, type) else obj
            if not getattr(topic, "name", None) or not callable(
                    getattr(topic, "render", None)):
                log.warning(
                    "scaffold topic %r does not satisfy ExplainTopic; skipped",
                    ep.name)
                continue
            topic = _stamp_provenance(topic, ep)
            topics.append(topic)
        except Exception:
            log.warning("failed to load scaffold topic %r", ep.name,
                        exc_info=True)
    _EXTERNAL_TOPICS = topics
    return topics


def _stamp_provenance(topic, ep):
    """Record which DISTRIBUTION contributed *topic*, best-effort.

    The banner marks a contributed topic with its distribution the way
    ``explain plugins`` marks a contributed plugin (``<- jaato-premium``): a
    reader who cannot tell the framework's own answer from an installed
    package's answer cannot tell which one to go and read the source of, and
    cannot tell what uninstalling premium would take away.

    Best-effort by construction — ``ep.dist`` is absent on a hand-built entry
    point (every test double, and some older metadata shapes).  An unknown
    contributor renders as no marker at all, never as a guess.
    """
    dist = getattr(getattr(ep, "dist", None), "name", "") or ""
    try:
        setattr(topic, "_jaato_dist", dist)
    except Exception:  # pragma: no cover - a frozen/slotted contributor
        pass
    return topic


def _topic_dist(topic) -> str:
    """The distribution that contributed *topic*, or ``""`` if unknown."""
    return getattr(topic, "_jaato_dist", "") or ""


def external_own_topics() -> "Dict[str, Any]":
    """Contributed topics that are topics of their OWN, keyed by name.

    A name a built-in already holds is refused with a warning — the verb seam's
    collision rule, for the verb seam's reason: an installed package must not be
    able to replace the framework's answer about the framework's own subject.
    Two contributors claiming one name resolve first-wins, and the loser is
    NAMED rather than silently dropped, the rule ``PluginRegistry`` already
    applies to a plugin collision.
    """
    import logging
    log = logging.getLogger(__name__)
    out: "Dict[str, Any]" = {}
    for topic in _discover_external_topics():
        if getattr(topic, "extends", ""):
            continue
        name = topic.name
        if name in _SCOPES:
            log.warning(
                "scaffold topic %r from %s collides with a built-in topic; "
                "the built-in wins", name, _topic_dist(topic) or "an extension")
            continue
        if name in out:
            log.warning(
                "scaffold topic %r contributed twice (%s, %s); first wins",
                name, _topic_dist(out[name]) or "?", _topic_dist(topic) or "?")
            continue
        out[name] = topic
    return out


def topic_extensions(topic_name: str) -> list:
    """Contributed SECTIONS appended to *topic_name*, in contributor order.

    An extension naming a topic that does not exist is not an error here: it
    simply never renders.  Reporting it would mean deciding, at discovery time,
    that a topic a later release adds is a mistake today.
    """
    return [t for t in _discover_external_topics()
            if getattr(t, "extends", "") == topic_name]


def _render_external_topic(topic, scope, name, ws):
    """Invoke a contributed own-topic through the one external signature."""
    from .api import TopicRequest
    data, text = topic.render(TopicRequest(topic=scope, name=name, workspace=ws))
    if isinstance(data, dict) and "error" in data:
        raise _ScopeUsageError(text)
    return data, text


def _append_topic_extensions(scope, name, ws, data, text):
    """Append every contributed section for *scope* to a rendered topic.

    Two rules, each attached to a way a contributed section could mislead:

    * the built-in's own data is never touched — a section lands at
      ``data["extensions"][<name>]``, so a contributor cannot redefine what a
      documented key means for a reader who is branching on it;
    * a section that RAISES is reported in place and the built-in's answer still
      prints.  The alternative is that installing a package can delete the
      framework's own documentation, which is a worse failure than a missing
      section and a much harder one to attribute.
    """
    import logging
    log = logging.getLogger(__name__)
    from .api import TopicRequest

    extensions = topic_extensions(scope)
    if not extensions:
        return data, text

    parts = [text]
    for topic in extensions:
        dist = _topic_dist(topic)
        label = f"{topic.name} ({dist})" if dist else topic.name
        try:
            ext_data, ext_text = topic.render(
                TopicRequest(topic=scope, name=name, workspace=ws))
        except Exception as exc:
            log.warning("scaffold topic extension %r failed on %r",
                        topic.name, scope, exc_info=True)
            parts.append(f"\n  -- {label} -- section failed to render: {exc}")
            continue
        if isinstance(data, dict):
            data.setdefault("extensions", {})[topic.name] = ext_data
        parts.append(f"\n{'-' * 70}\ncontributed by {label}:\n\n{ext_text}")
    return data, "\n".join(parts)


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
    pe.add_argument("scope", nargs="?", help=_all_scopes_help())
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
    pn.add_argument("--profile", metavar="NAME",
                    help="bind the generated client to an EXISTING profile "
                         "instead of an inline {model, provider} spec. A "
                         "profile carries plugins, persona, GC, ceilings and "
                         "the completion schema, which a spec cannot; "
                         "mutually exclusive with --provider/--model, and "
                         "refused if NAME does not resolve in --workspace.")
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
    pn.add_argument("--component", action="store_true",
                    help="for `new dossier`: emit the Article 25(4) "
                         "information pack for jaato AS A COMPONENT of "
                         "somebody else's high-risk system, instead of the "
                         "Annex IV dossier for one of yours. Needs no "
                         "profile — it describes the framework.")
    pn.add_argument("--eval-results", metavar="FILE", dest="eval_results",
                    help="for `new dossier --profile`: fill the accuracy "
                         "section (Annex IV §4, Article 15(3)) from a "
                         "jaato-eval results file. The harness's own caveats "
                         "are carried verbatim beside its numbers, and a file "
                         "whose declared format this reader does not know is "
                         "refused by name rather than half-rendered.")
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
