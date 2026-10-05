"""The introspection half of ``jaato-scaffold``: ``explain``, ``validate``,
``dependencies`` and ``releases``, contributed by jaato-server (#1267).

    jaato-scaffold explain [scope] [name] [--workspace DIR] [--json]
    jaato-scaffold validate <workspace-or-profile> [--set S] [--profile P] [--json]
    jaato-scaffold dependencies [scope] [name] [--json]
    jaato-scaffold releases [--json]

``jaato-scaffold`` itself (the console script, the ``new`` and
``integration`` verbs and verb discovery) ships in jaato-sdk, as
:mod:`jaato_sdk.scaffold.cli`, because it is the application developers who
install only the SDK that need to author clients and apply integrations.
These four verbs read the INSTALLED framework — the plugin registry, the
provider contracts, the profile dataclass — so they stay here, and reach the
SDK shell as :class:`ScaffoldVerb` s registered under the
``jaato.scaffold_verbs`` entry-point group in jaato-server's own
``pyproject.toml``.  The shell reserves their four names for jaato-server
(another package claiming one is refused), and answers them with a refusal
naming the fix when jaato-server is not installed.

``python -m jaato_server.shared.scaffold`` still works: its ``__main__`` runs
the SDK shell with these verbs handed to it directly, so it does not depend on
this distribution's entry points being installed (a source checkout on
``PYTHONPATH``, a test that stubs ``importlib.metadata.entry_points``).  Every
name this module defines is reachable as ``jaato_server.shared.scaffold.__main__``
too, which is an alias of this module.

``explain`` renders the introspect core by scope; ``validate`` checks an
asset against it; ``dependencies`` is the ``explain ... dependencies`` facet
as a verb of its own, and ``releases`` the ``explain releases`` topic.  They
introspect whatever framework build is installed in the current Python env —
run them in the SAME env as the daemon you target, or ask the daemon
(``explain --connect``).

**Extension topics.**  A package registering an :class:`api.ExplainTopic`
under ``jaato.scaffold_topics`` either adds a topic of its own (``explain
reactors``) or appends an attributed SECTION to a built-in one (``extends =
"paths"``), and either way it reaches the dispatch, the overview banner,
``--help`` and the unknown-scope error through the same merged table — so a
contributed topic cannot be advertised without being served, or served
without being advertised (#994's rule, one layer out).  Built-in names win on
collision, and a topic that fails to load is skipped with a warning.

It exists because a package that contributes a daemon EXTENSION has something
to say and nothing to run, so the verb seam could only ever have offered it a
subcommand nobody would think to type.  The measured cost of not having it:
premium's reactor engine reads four rule-file tiers, and the only surface that
named any of them named the one under the DAEMON's home — so a session asked
to add a reactor could not find out where the file goes, what the schema is,
or what the script must define, and read the engine's source instead.
(There is no ``jaato.premium_reactors`` group: reactors mount as the
``reactors`` entry in ``jaato.extensions``, and their RULES load from
directories — ``~/.jaato/reactors/`` and ``<workspace>/.jaato/reactors/`` —
not from entry points.)
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from jaato_sdk.scaffold import findings as _findings
from jaato_sdk.scaffold import remote as _remote

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
        live_only: why this topic is NOT in the explain snapshot jaato-sdk
            ships (:mod:`explain_snapshot`), or empty when it is.  Set on a
            topic whose answer describes the machine or the moment rather
            than the installed code (the network, ``$HOME``, this venv's
            paths): a snapshot of it would be a snapshot of whoever generated
            it.  A ``workspace`` topic, a filtered render and the named form
            of an ``optional_named`` one are never snapshotted and need no
            reason here: their argument is the caller's.
        names: for a ``named`` topic, every name it renders, as
            ``{canonical: [accepted spellings]}``.  The snapshot renders each
            canonical name once and maps every spelling to it, so an SDK-only
            ``explain provider zhipuai-openai`` finds what ``zhipuai_openai``
            would.
            A ``named`` topic without one cannot be snapshotted, and the
            generator refuses rather than silently shipping without it.
    """

    render: Callable[..., Any]
    kind: str = "simple"
    arg: str = ""
    render_named: Callable[..., Any] | None = None
    blurb: str = ""
    live_only: str = ""
    names: Callable[[], Dict[str, Any]] | None = None


def _plugin_names() -> Dict[str, Any]:
    """Every name ``explain plugin`` renders: the registry's, plus ``lifecycle``."""
    from . import introspect
    names = {n: [n] for n in introspect.plugins()}
    names[_explain.LIFECYCLE_TOPIC] = [_explain.LIFECYCLE_TOPIC]
    return names


def _provider_names() -> Dict[str, Any]:
    """Every provider, with the spellings ``resolve_provider`` accepts."""
    from . import introspect
    return {n: sorted(i.normalized_names() | {n})
            for n, i in introspect.providers().items()}


def _event_names() -> Dict[str, Any]:
    """Every event; ``explain event`` matches member or wire, any case."""
    from . import introspect
    out = {}
    for key, e in introspect.events().items():
        spellings = {key, e.name, e.wire}
        out[key] = sorted(spellings | {s.lower() for s in spellings}
                          | {s.upper() for s in spellings})
    return out


def _archetype_names() -> Dict[str, Any]:
    """Every archetype with its aliases, as ``archetypes.resolve`` follows them."""
    from . import archetypes
    return {n: [n, *d.aliases] for n, d in archetypes.ARCHETYPES.items()}


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
    "plugin": ExplainScope(_explain.plugin, "named", "<name>",
                          names=_plugin_names),
    "commands": ExplainScope(_explain.commands),
    "providers": ExplainScope(_explain.providers),
    "provider": ExplainScope(_explain.provider, "named", "<name>",
                            names=_provider_names),
    "gc": ExplainScope(_explain.gc),
    "env": ExplainScope(_explain.env, "filter", "[<filter>]",
                        blurb="vars the daemon + plugins READ"),
    "events": ExplainScope(_explain.events, "filter", "[<filter>]",
                           blurb="the client/server protocol"),
    # The hint says "or" rather than "|": the help line separates topics with
    # "|", so a hint carrying one is unreadable there and unparseable by
    # anything reading the line back.
    "event": ExplainScope(_explain.event, "named", "<NAME or wire.value>",
                          blurb="one event's fields + docstring",
                          names=_event_names),
    "transports": ExplainScope(_explain.transports),
    "clients": ExplainScope(_explain.clients),
    "runtime": ExplainScope(_explain.runtime),
    "tiers": ExplainScope(_explain.tiers),
    "integrations": ExplainScope(_explain.integrations,
                                 blurb="tools jaato can wire into",
                                 live_only="reports whether each integration is "
                                           "applied under THIS machine's $HOME"),
    # The only topic that reaches the network: our own indexes, asked what
    # they carry.  Its own topic rather than a section of `dependencies`,
    # which is an OFFLINE read of the installed tree and must stay one.
    "releases": ExplainScope(_explain.releases,
                             blurb="newer builds on PyPI / TestPyPI",
                             live_only="asks PyPI / TestPyPI at the time of "
                                       "asking"),
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
    "gh": ExplainScope(_explain.gh,
                       blurb="driving `gh` / `git` with a per-user token"),
    "runner-user": ExplainScope(_explain.runner_user,
                                blurb="which OS account a root daemon's runners run as",
                                live_only="names the import paths of the "
                                          "install it runs in"),
    "pool": ExplainScope(_explain.pool,
                         blurb="the pre-warm runner pool, resized live"),
    "prefetch": ExplainScope(_explain.prefetch),
    "completion": ExplainScope(_explain.completion,
                               blurb="the OUTPUT-side hook"),
    "archetypes": ExplainScope(_explain.archetypes,
                               blurb="what `new` WRITES"),
    "archetype": ExplainScope(_explain.archetype, "named", "<name>",
                              names=_archetype_names),
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


def render_topic(
    scope: Optional[str], name: Optional[str] = None, workspace: str = ".",
) -> "tuple[bool, Dict[str, Any], str, str]":
    """Render one topic — the ONE dispatch, with no printing and no exit code.

    Two callers, deliberately: ``_cmd_explain`` prints it, and the daemon's
    ``scaffold.explain`` handler puts it on the wire for a CLI whose own
    virtualenv does not have the extension that contributes the topic.  A
    second dispatch for the remote case would be free to disagree with this
    one about what a topic answers, which is the failure the seam exists to
    remove rather than one to reproduce over a socket.

    Args:
        scope: The topic, or ``None`` for the overview.
        name: The topic's argument, when it takes one.
        workspace: What a workspace-reading topic reads.

    Returns:
        ``(ok, data, text, error)``.  ``ok`` is ``False`` for an unknown
        topic, a usage error (a topic that requires a name, given none) and
        a renderer that raised; ``error`` then carries the message and
        ``data`` / ``text`` are empty.  A renderer that RAISES is reported
        rather than propagated: one broken contributed topic must not take
        down the verb, or the daemon serving it.
    """
    if scope is None:
        data, text = _explain.overview()
        return True, data, text, ""
    render = _scope_renderer(scope)
    if render is None:
        return False, {}, "", (
            f"unknown explain scope {scope!r} — one of: {_all_scopes_help()}")
    try:
        data, text = render(scope, name, workspace)
        # Contributed SECTIONS append to whatever rendered — a built-in or a
        # contributed topic alike — so two packages can answer about one
        # subject without either having to know the other exists.
        data, text = _append_topic_extensions(scope, name, workspace, data, text)
    except _ScopeUsageError as exc:
        return False, {}, "", exc.message
    except Exception as exc:                      # pragma: no cover - defensive
        return False, {}, "", f"explain {scope}: renderer failed: {exc}"
    return True, data, text, ""


def _cmd_explain(args) -> int:
    """Render one `explain` topic.

    Every topic is looked up through :func:`_scope_renderer` and invoked
    through the handler its ``kind`` names, so the set of topics this
    dispatches is by construction the set the help line advertises (#994).

    A topic this CLI's own virtualenv cannot answer is not the end of the
    question: see :func:`_consult_daemon`.
    """
    scope, name, deps = _take_deps_word(
        args.scope, args.name, getattr(args, "extra", None))
    ws = args.workspace or "."
    if deps:
        from . import dependencies as _deps
        data, text = _deps.render(scope, name)
        print(json.dumps(data, indent=2) if args.json else text)
        return 0

    asked = getattr(args, "connect", None)
    if asked:
        rc, _note = _remote.render_from_daemon(asked, scope, name, args,
                                                required=True)
        return rc

    ok, data, text, error = render_topic(scope, name, ws)
    if not ok:
        # Only a topic this venv does not HAVE is worth a socket: a usage
        # error is about the caller's own command line, and asking a daemon
        # would answer a question nobody posed.
        note = ""
        if _scope_renderer(scope) is None:
            rc, note = _remote.render_from_daemon(
                None, scope, name, args, required=False)
            if rc is not None:
                return rc
        print(error, file=sys.stderr)
        if note:
            print(note, file=sys.stderr)
        return 2
    print(json.dumps(data, indent=2, default=str) if args.json else text)
    return 0


# -------------------------------------------------------------- validate

# Target resolution and the finding line live in the SDK (#1267, tier 3), so
# this local ``validate`` and the SDK shell's daemon route read a target and
# print a finding the same way.
_resolve_target = _findings.resolve_target
_is_canonical_profile_layout = _findings.is_canonical_profile_layout


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
        scope = _findings.scope_label(only)

    if args.json:
        print(json.dumps([d.as_dict() for d in diags], indent=2))
    else:
        if not diags:
            print(_findings.clean_line(scope, profile_set))
        for d in diags:
            print(_format_diagnostic(d))
    return 1 if any(d.severity == "error" for d in diags) else 0


def _format_diagnostic(d) -> str:
    """One finding as the text line ``validate`` prints.

    The rendering is :func:`jaato_sdk.scaffold.findings.format_finding`, the
    one the SDK shell's daemon route prints with too.
    """
    return _findings.format_finding(d.as_dict())


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


# ------------------------------------------------------------------ verbs
#
# What jaato-server contributes to the SDK shell.  Each verb is its argparse
# declaration plus the ``_cmd_*`` function above, unchanged, so a `jaato-scaffold
# explain` / `validate` run is the same code whichever way the shell found it.


class ExplainVerb:
    """``jaato-scaffold explain``: interrogate the installed framework."""

    name = "explain"
    help = "interrogate the installed framework"

    def configure(self, pe: argparse.ArgumentParser) -> None:
        pe.add_argument("scope", nargs="?", help=_all_scopes_help())
        pe.add_argument("name", nargs="?",
                        help="name for plugin/provider/event/archetype scope, or a "
                             "filter for env/events")
        pe.add_argument("extra", nargs="?",
                        help="the optional word `dependencies` (or `deps`) — a facet "
                             "of any scope: what it needs, what is installed, and "
                             "whether this environment agrees with itself")
        pe.add_argument("--workspace", help=_workspace_arg_help())
        pe.add_argument("--connect", nargs="?", const=True, metavar="SOCKET",
                        help="ask a running daemon to render the topic instead of "
                             "this virtualenv — for a CLI installed beside an "
                             "application, whose daemon holds extensions this "
                             "install does not (default socket when no path given)")
        pe.add_argument("--json", action="store_true")

    def run(self, args: argparse.Namespace) -> int:
        return _cmd_explain(args)


class ValidateVerb:
    """``jaato-scaffold validate``: check a profile / workspace."""

    name = "validate"
    help = "validate a profile / workspace"

    def configure(self, pv: argparse.ArgumentParser) -> None:
        pv.add_argument("target", help="a workspace dir or a profile .yaml file")
        pv.add_argument("--set", help="JAATO_PROFILE_SET name to overlay")
        pv.add_argument("--profile", help="validate only this profile name")
        pv.add_argument("--connect", nargs="?", const=True, metavar="SOCKET",
                        help="validate with a running daemon's install instead "
                             "of this virtualenv's (default socket when no "
                             "path is given): its plugins, its contributed "
                             "validators, and its user-tier profiles")
        pv.add_argument("--json", action="store_true")

    def run(self, args: argparse.Namespace) -> int:
        if getattr(args, "connect", None):
            return _remote.validate_from_daemon(
                args.connect, args.target, args.set, args.profile,
                json_out=args.json, required=True)
        return _cmd_validate(args)


class DependenciesVerb:
    """``jaato-scaffold dependencies [scope] [name]``.

    The same answer as ``explain [scope] [name] dependencies``, through the
    same renderer: a verb of its own so the introspection half of the CLI has
    one name per question, and so an SDK-only install can refuse it by name.
    """

    name = "dependencies"
    help = ("what a scope needs, what is installed, and whether this "
            "environment agrees with itself")

    def configure(self, pd: argparse.ArgumentParser) -> None:
        pd.add_argument("scope", nargs="?",
                        help="plugin | provider | ... (omit for the framework)")
        pd.add_argument("name", nargs="?", help="the plugin/provider name")
        pd.add_argument("--json", action="store_true")

    def run(self, args: argparse.Namespace) -> int:
        from . import dependencies as _deps
        data, text = _deps.render(args.scope, args.name)
        print(json.dumps(data, indent=2) if args.json else text)
        return 0


class ReleasesVerb:
    """``jaato-scaffold releases``: the ``explain releases`` topic as a verb."""

    name = "releases"
    help = "newer builds of the installed jaato packages on PyPI / TestPyPI"

    def configure(self, pr: argparse.ArgumentParser) -> None:
        pr.add_argument("--json", action="store_true")

    def run(self, args: argparse.Namespace) -> int:
        ok, data, text, error = render_topic("releases")
        if not ok:
            print(error, file=sys.stderr)
            return 2
        print(json.dumps(data, indent=2, default=str) if args.json else text)
        return 0


def server_verbs() -> list:
    """The four verbs jaato-server contributes, in the order the shell mounts them."""
    return [ExplainVerb(), ValidateVerb(), DependenciesVerb(), ReleasesVerb()]


def main(argv=None) -> int:
    """Run the SDK shell with this distribution's verbs handed to it directly.

    ``python -m jaato_server.shared.scaffold`` and in-process callers land
    here.  Handing the verbs over rather than relying on entry-point
    discovery keeps them present where this distribution's metadata is not
    (a checkout on ``PYTHONPATH``) or is stubbed (tests), and is equivalent
    otherwise: the shell takes the first verb per name.
    """
    from jaato_sdk.scaffold.cli import main as _shell_main
    return _shell_main(argv, verbs=server_verbs())
