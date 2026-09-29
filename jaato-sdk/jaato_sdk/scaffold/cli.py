"""``jaato-scaffold`` — the CLI shell, shipped with the SDK (#1267).

    jaato-scaffold new ...                  (see `new --help`)
    jaato-scaffold integration [name] ...   (bare: list them)
    jaato-scaffold explain | validate | dependencies | releases ...
                                            (contributed by jaato-server)

**Why the shell lives here.**  ``jaato-server`` (the daemon) is deployed once;
``jaato-sdk`` is embedded in many applications, and it is the developers of
those applications, who install only the SDK, that need ``new`` and
``integration`` on their PATH.  So jaato-sdk owns the ``jaato-scaffold``
console script, this shell, and the authoring verbs.  It imports nothing from
jaato-server; the one place it reaches it is entry-point discovery.

**The one-owner rule.**  The console script is declared in jaato-sdk's
``pyproject.toml`` and nowhere else.  jaato-server contributes its four
introspection verbs (:data:`SERVER_VERBS`) through the ``jaato.scaffold_verbs``
entry-point group, the same seam any external verb uses, so an environment with
both installed has ONE ``jaato-scaffold`` exposing every verb, and an
environment with only the SDK has ``new`` / ``integration`` plus a refusal for
each of the four naming the fix — never an ``ImportError``.

**Who may answer which name.**

* :data:`BUILTIN_VERBS` are this shell's own; a contributed verb claiming one
  is ignored.
* :data:`SERVER_VERBS` are reserved for jaato-server: a contributed verb
  claiming one is accepted only when its code lives under ``jaato_server``,
  and refused with a warning otherwise, so an installed package cannot
  replace the framework's validator (the rule #684 applies to plugins).  The
  shell deliberately does not DEFINE them — it only refuses them when nobody
  contributed them.
* any other name is an ordinary extension verb (the premium ``compile``);
  first discovered wins.

**The refusal.**  ``explain`` and ``validate`` ask a running daemon before
refusing, because a daemon's install is exactly the one that can answer:
``explain`` through the protocol 1.18 ``scaffold.explain`` fallback
(:func:`remote.render_from_daemon`), ``validate`` through the 1.34
``scaffold.validate`` verb (:func:`remote.validate_from_daemon`), which runs
the daemon's full validator on the workspace the connection declares.
``dependencies`` and ``releases`` describe THIS environment, which only this
environment can, so they refuse.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence, Tuple

#: The entry-point group the shell scans for contributed verbs.  Defined here,
#: in the distribution that owns the shell; jaato-server's ``scaffold.api``
#: re-exports it for external verbs that already import it from there.
VERB_ENTRY_POINT_GROUP = "jaato.scaffold_verbs"

#: The verbs this shell defines itself.
BUILTIN_VERBS = ("new", "integration")

#: The verbs jaato-server contributes, and which only jaato-server may.
SERVER_VERBS = ("explain", "validate", "dependencies", "releases")

#: The package a SERVER_VERBS contribution must come from.
SERVER_PACKAGE = "jaato_server"

#: The order the top-level ``--help`` lists them in: the two that predate the
#: split first, as they always were, then the authoring verbs, then the rest.
_VERB_ORDER = ("explain", "validate", "new", "integration",
               "dependencies", "releases")


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


# ----------------------------------------------------------- integration

def _run_refresh(_install, name, dest, *, json_out: bool, dry_run: bool) -> int:
    """Apply ``integration <name> --refresh`` and report the transition.

    A refresh reports what state the copy was in, what it is in now, whether
    anything was written, and — when it declined — why, because "did nothing"
    is a correct outcome here and a caller keeping a copy current needs to tell
    it apart from a failure.  Always exits 0: a skipped refresh of an edited
    copy is correct behaviour, not a failure (#1261).  Split out of
    `_cmd_integration` so that function stays under the complexity ceiling.
    """
    # One computation, shared with the ``scaffold.integration`` daemon verb
    # (#1263): the CLI and the daemon must report the same transition, so both
    # read it from ``integrations.refresh`` rather than each assembling it.
    result = _install.refresh(name, dest, dry_run=dry_run)
    if json_out:
        # ``skipped_reason`` is "" when it applied; the CLI's --json has always
        # printed ``null`` there, so keep that byte-for-byte.
        print(json.dumps({"asset": name, "dest": str(dest),
                          "state_before": result["state_before"],
                          "state_after": result["state_after"],
                          "changed": result["changed"],
                          "skipped_reason": result["skipped_reason"] or None,
                          "version": _install.framework_version()}, indent=2))
    else:
        for line in result["lines"]:
            print(line)
    return 0


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
    if args.refresh:
        return _run_refresh(_install, name, dest, json_out=args.json,
                            dry_run=args.dry_run)

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


# ------------------------------------------------------------ verb discovery

def _discover_external_verbs() -> List[Tuple[object, str]]:
    """Load verbs contributed through the ``jaato.scaffold_verbs`` group.

    Each entry point loads to a ``ScaffoldVerb`` — an instance, or a zero-arg
    class/factory producing one.  A verb whose package is not installed is not
    discovered; a verb that fails to load, or does not satisfy the protocol,
    is skipped with a warning rather than breaking the whole CLI.  This is how
    jaato-server's introspection verbs and the premium ``compile`` verb mount.

    Returns:
        ``(verb, source)`` pairs, ``source`` being the entry point's
        ``module:attr`` value (``""`` when it has none), which is what
        :func:`_may_answer` reads to decide who may claim a reserved name.
    """
    import logging
    from importlib.metadata import entry_points

    log = logging.getLogger(__name__)
    eps = entry_points(group=VERB_ENTRY_POINT_GROUP)

    verbs: List[Tuple[object, str]] = []
    for ep in eps:
        try:
            obj = ep.load()
            verb = obj() if isinstance(obj, type) else obj
            if not getattr(verb, "name", None) or not callable(getattr(verb, "run", None)):
                log.warning("scaffold verb %r does not satisfy ScaffoldVerb; skipped", ep.name)
                continue
            verbs.append((verb, str(getattr(ep, "value", "") or "")))
        except Exception:
            log.warning("failed to load scaffold verb %r", ep.name, exc_info=True)
    return verbs


def _from_server(verb, source: str) -> bool:
    """Whether *verb* is jaato-server's own code, by module, never by name."""
    module = source.split(":", 1)[0] if source else type(verb).__module__
    return module == SERVER_PACKAGE or module.startswith(SERVER_PACKAGE + ".")


def _may_answer(verb, source: str) -> bool:
    """Whether a DISCOVERED verb may take its name; see the module docstring."""
    import logging

    name = verb.name
    if name in BUILTIN_VERBS:
        return False
    if name in SERVER_VERBS and not _from_server(verb, source):
        logging.getLogger(__name__).warning(
            "scaffold verb %r from %s claims a name reserved for jaato-server; "
            "refused", name, source or type(verb).__module__)
        return False
    return True


def _collect_verbs(given: Optional[Sequence[object]]) -> Dict[str, object]:
    """Every contributed verb by name: *given* first, then discovery.

    *given* is what a caller hands in directly (jaato-server's
    ``python -m jaato_server.shared.scaffold``), trusted as-is.  First wins
    per name, so a verb found both ways is mounted once.
    """
    out: Dict[str, object] = {}
    for verb in given or ():
        out.setdefault(verb.name, verb)
    for verb, source in _discover_external_verbs():
        if verb.name not in out and _may_answer(verb, source):
            out[verb.name] = verb
    return out


# ---------------------------------------------------------------- refusal

#: What each SERVER_VERBS name reads, for its refusal.
_SERVER_VERB_NEEDS = {
    "explain": "the installed framework's topics (plugins, providers, the "
               "profile schema, events)",
    "validate": "the installed plugin registry, provider contracts and "
                "profile schema",
    "dependencies": "the installed jaato-server tree and its dependencies",
    "releases": "the installed jaato distributions, through jaato-server's "
                "renderer",
}

#: Where each refusal can send the reader besides installing jaato-server.
_SERVER_VERB_ELSEWHERE = {
    "explain": "or ask the daemon: jaato-scaffold explain <topic> --connect "
               "[SOCKET]",
    "validate": "or start a daemon: a daemon on the default socket is asked "
                "automatically, or name one with --connect SOCKET",
    "dependencies": "or run it where the daemon's jaato-server is installed",
    "releases": "or run `jaato-doctor`, whose `package releases` check needs "
                "only the SDK",
}


def server_verb_refusal(name: str) -> str:
    """The message a SERVER_VERBS name answers with when jaato-server is absent."""
    return (f"jaato-scaffold {name}: this verb reads "
            f"{_SERVER_VERB_NEEDS[name]}, and jaato-server is not installed in "
            f"this environment ({sys.executable}).  Install jaato-server here "
            f"(pip install jaato-server), "
            f"{_SERVER_VERB_ELSEWHERE[name]}.")


def _refuse_server_verb(args) -> int:
    """Answer an uncontributed SERVER_VERBS name; ``explain`` asks a daemon first."""
    name = args._refused_verb
    note = ""
    if name == "explain":
        rc, note = _explain_from_daemon(args)
        if rc is not None:
            return rc
    if name == "validate":
        rc = _validate_from_daemon(args)
        if rc is not None:
            return rc
    print(server_verb_refusal(name), file=sys.stderr)
    if note:
        print(note, file=sys.stderr)
    return 2


def _explain_from_daemon(args) -> "Tuple[Optional[int], str]":
    """The ``scaffold.explain`` fallback for a venv with no jaato-server.

    The same function jaato-server's own ``explain`` uses for a topic its venv
    lacks, with every topic "lacked" here.  ``dependencies`` is never asked
    of a daemon: it describes THIS environment, which only this environment
    can.

    Returns:
        ``(rc, note)`` as :func:`remote.render_from_daemon` returns them: an
        exit code, or ``None`` when the refusal should print, followed by the
        note (if any) that says a daemon was asked and lacked the topic too.
    """
    from . import remote as _remote

    words = [w for w in (args.scope, args.name, args.extra) if w]
    if any(w in ("dependencies", "deps") for w in words):
        return None, ""
    words += [None, None]
    return _remote.render_from_daemon(
        args.connect, words[0], words[1], args, required=bool(args.connect))


def _validate_from_daemon(args) -> Optional[int]:
    """The ``scaffold.validate`` route for a venv with no jaato-server (1.34).

    The daemon's own validator checks the workspace this connection declares;
    the findings print as a local run's would, followed by the line naming
    whose install produced them.  With no daemon to ask (and no
    ``--connect``) it returns ``None`` and the refusal prints: a validator
    that did not run never reports a pass.
    """
    from . import remote as _remote

    return _remote.validate_from_daemon(
        args.connect, args.target, args.set, args.profile,
        json_out=args.json, required=bool(args.connect))


def _add_server_refusal(sub, name: str) -> None:
    """Mount *name* as a refusal: nothing contributed it here."""
    p = sub.add_parser(name, help=f"(needs jaato-server) {name}",
                       description=server_verb_refusal(name))
    if name == "explain":
        # Enough of the real verb's surface to reach a daemon with it.
        p.add_argument("scope", nargs="?")
        p.add_argument("name", nargs="?")
        p.add_argument("extra", nargs="?")
        p.add_argument("--workspace")
        p.add_argument("--connect", nargs="?", const=True, metavar="SOCKET")
        p.add_argument("--json", action="store_true")
    elif name == "validate":
        # The real verb's surface, so the same command line reaches a daemon.
        p.add_argument("target", nargs="?", default=".",
                       help="a workspace dir, or a profile file inside one")
        p.add_argument("--set", help="JAATO_PROFILE_SET name to overlay")
        p.add_argument("--profile", help="validate only this profile name")
        p.add_argument("--connect", nargs="?", const=True, metavar="SOCKET",
                       help="validate with this daemon (default socket when "
                            "no path is given)")
        p.add_argument("--json", action="store_true")
    else:
        p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=_refuse_server_verb, _refused_verb=name)


# ------------------------------------------------------------------ main

def _add_new(sub) -> None:
    """The ``new`` subparser."""
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
                    help="bind the generated client to a profile instead "
                         "of an inline {model, provider} spec. A profile "
                         "carries plugins, persona, GC, ceilings and the "
                         "completion schema, which a spec cannot; mutually "
                         "exclusive with --provider/--model. NAME is written "
                         "as given and need not exist yet; with jaato-server "
                         "installed, a NAME --workspace does not resolve "
                         "gets a note (the client fails at create_session "
                         "until it exists).")
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
                         "default), 'ws' (remote daemon over WebSocket — "
                         "requires --url), or 'in_process' (embedded — runs the "
                         "runtime + session in THIS process, no daemon/socket; "
                         "incompatible with --recoverable).")
    pn.add_argument("--url", help="WebSocket URL for --transport ws")
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


def _add_integration(sub) -> None:
    """The ``integration`` subparser."""
    pi = sub.add_parser(
        "integration", help="wire jaato into a tool you work in (bare: list them)",
        description="An integration is jaato's side of a contract with another "
                    "tool. The `claude-code` and `pi` integrations install the "
                    "shared jaato-sdk skill where each harness looks for skills. "
                    "Each copy is stamped with the build it came from, so "
                    "`jaato-doctor` can say when one has gone stale. With no "
                    "name, lists what this build ships and where each stands.")
    pi.add_argument("name", nargs="?", default=None,
                    help="integration name (omit to list)")
    scope = pi.add_mutually_exclusive_group()
    scope.add_argument("--user", action="store_true",
                       help="apply under $HOME — every repo on this machine "
                            "(the default; accepted explicitly so a script can "
                            "say what it means)")
    scope.add_argument("--workspace", default=None,
                       help="apply under DIR instead of $HOME — this project only")
    # --force and --refresh are opposite intents about local edits: --force
    # overwrites every state, --refresh writes only the states that lose
    # nothing local.  Asking for both is a contradiction, so argparse refuses
    # it (exit 2) rather than the code having to pick a winner (#1261).
    write_mode = pi.add_mutually_exclusive_group()
    write_mode.add_argument("--force", action="store_true",
                            help="overwrite an existing copy, local edits included")
    write_mode.add_argument("--refresh", action="store_true",
                            help="re-apply only when nothing local is lost "
                                 "(absent / stale / outdated); leave edited, "
                                 "diverged and unstamped copies untouched, exit 0")
    pi.add_argument("--dry-run", action="store_true",
                    help="print what would be written, write nothing")
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=_cmd_integration)


def _mount(sub, verb) -> None:
    """Mount one contributed verb: its own subparser, its own ``run``."""
    p = sub.add_parser(verb.name, help=getattr(verb, "help", None))
    verb.configure(p)
    p.set_defaults(func=verb.run)


def main(argv=None, *, verbs: Optional[Sequence[object]] = None) -> int:
    """Run ``jaato-scaffold``.

    Args:
        argv: The arguments, ``sys.argv[1:]`` when ``None``.
        verbs: Verbs to mount before discovery, trusted as-is; jaato-server
            passes its introspection verbs here from
            ``python -m jaato_server.shared.scaffold``.
    """
    ap = argparse.ArgumentParser(
        prog="jaato-scaffold",
        description="Interrogate / validate / scaffold jaato profiles + SDK "
                    "clients against the installed framework.")
    sub = ap.add_subparsers(dest="cmd")

    contributed = _collect_verbs(verbs)
    for name in _VERB_ORDER:
        if name == "new":
            _add_new(sub)
        elif name == "integration":
            _add_integration(sub)
        elif name in contributed:
            _mount(sub, contributed.pop(name))
        else:
            _add_server_refusal(sub, name)
    for verb in contributed.values():
        _mount(sub, verb)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
