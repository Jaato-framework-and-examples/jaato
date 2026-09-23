"""The `explain` topic list is derived from the dispatch, not typed beside it (#994).

``jaato-scaffold explain <nonsense>`` prints the topics that exist, and the
same string is the ``scope`` argument's ``--help``.  It was hand-typed, and it
omitted ``integrations`` -- a topic that works: #906 registered it in
``_SIMPLE_SCOPES`` and did not edit the string.  An introspection tool
misreporting what it offers is the one failure it must not have.

THE TRAP THIS GUARD EXISTS FOR.  The obvious derivation reads the scope tables
-- and there were FOUR of them plus a standalone ``elif scope == "profile"``
branch, so a derivation from the first three silently drops ``agents``,
``services``, ``sets`` and ``profile``.  The issue's own author made that
mistake while writing it.  The fix put every topic in ONE table
(``_SCOPES``) whose entries carry their calling convention (``kind``) and
argument hint (``arg``), with ``_SCOPE_KINDS`` mapping kind -> handler; the
help line, the usage lines, the ``--workspace`` help and the dispatch all read
that table.

So this guard checks three separable things, because a sixth dispatch path can
be added without touching the help, and a help string can be re-hardcoded
without touching the dispatch:

1. every dispatched scope is advertised, and nothing advertised is undispatched
   -- the round trip the issue asks for;
2. ``_cmd_explain`` dispatches ONLY through ``_SCOPES`` -- an AST check, so a
   new ``elif scope == "..."`` rung fails the build rather than quietly
   rejoining the ladder the tables replaced;
3. every ``kind`` a scope declares has a handler and every handler is used --
   so a topic with a genuinely new calling convention must wire it in, which is
   the sixth-path case stated in the vocabulary the table uses.
"""

import ast
from pathlib import Path

from jaato_server.shared.scaffold import __main__ as scaffold_main
from jaato_server.shared.scaffold.__main__ import (
    _FILTER_SCOPES,
    _NAMED_SCOPES,
    _SCOPE_KINDS,
    _SCOPES,
    _SCOPES_HELP,
    _SIMPLE_SCOPES,
    _WORKSPACE_SCOPES,
)
from jaato_server.shared.tests.reversion import Reversion

_MAIN = "jaato-server/jaato_server/shared/scaffold/__main__.py"
_MAIN_PY = Path(__file__).resolve().parents[1] / "scaffold" / "__main__.py"

_HAND_TYPED_HELP = (
    '_SCOPES_HELP = ("plugins | plugin | commands | providers | provider | gc '
    '| env | events | "\n'
    '                "event | transports | clients | runtime | tiers | sets | '
    'agents | "\n'
    '                "services | profile [<name>] | paths | prefetch | '
    'completion | "\n'
    '                "archetypes | archetype")'
)

REVERSIONS = [
    Reversion(
        target=_MAIN,
        find="_SCOPES_HELP = _scopes_help()",
        replace=_HAND_TYPED_HELP,
        test="test_every_dispatched_scope_is_advertised",
        because="the topic list is hand-typed again, so a registered topic "
                "(integrations) is dispatched and advertised nowhere",
    ),
    Reversion(
        target=_MAIN,
        find="""    render = _scope_renderer(scope)""",
        replace="""    render = (_SCOPE_KINDS[_SCOPES[scope].kind]
              if scope == "profile" else None)""",
        test="test_dispatch_reads_only_the_scope_table",
        because="the dispatch grew a scope-literal rung again, so the resolver "
                "is no longer the one place a topic is looked up",
    ),
]


def _help_entries():
    """The scope NAMES the help line advertises, hints stripped.

    ``profile [<name>]`` advertises the topic ``profile``; the hint is
    presentation and is checked separately.
    """
    return [part.strip().split()[0]
            for part in _SCOPES_HELP.split("|") if part.strip()]


def _function_ast(name):
    """The AST of one top-level function of ``shared/scaffold/__main__.py``.

    A MISSING function is an assertion failure rather than a skip: this guard
    reads a dispatch, and a dispatch that moved without the guard moving with
    it is exactly the state where the guard reports nothing and means nothing.
    """
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"{name} is gone from shared/scaffold/__main__.py -- this guard reads "
        f"its dispatch and can no longer say anything about it."
    )


def _cmd_explain_ast():
    return _function_ast("_cmd_explain")


def test_every_dispatched_scope_is_advertised():
    """The round trip: dispatched <-> advertised, in both directions."""
    advertised = _help_entries()

    missing = sorted(set(_SCOPES) - set(advertised))
    assert not missing, (
        f"dispatched but not advertised: {missing}. `explain {missing[0]}` "
        f"works and `explain <nonsense>` does not list it -- #994 exactly."
    )

    phantom = sorted(set(advertised) - set(_SCOPES))
    assert not phantom, (
        f"advertised but not dispatched: {phantom}. The help promises a topic "
        f"`explain` answers with 'unknown explain scope'."
    )

    assert len(advertised) == len(set(advertised)), (
        f"the help line lists a topic twice: {advertised}"
    )


def test_the_help_line_carries_each_scopes_argument_hint():
    """Hints live on the entry, not in a parallel string.

    ``profile [<name>]`` was the one hint the hand-typed string carried, and it
    was the reason the string existed at all.  Deriving the list without the
    hints would have lost it.
    """
    for scope, spec in _SCOPES.items():
        expected = f"{scope} {spec.arg}".rstrip()
        assert expected in _SCOPES_HELP, (
            f"the help line does not carry {scope}'s hint {spec.arg!r}"
        )
    assert "profile [<name>]" in _SCOPES_HELP


def test_no_argument_hint_contains_the_help_separator():
    """A hint carrying `|` corrupts the list it is rendered into.

    Found by this guard rather than by a reader: ``event``'s hint was
    ``<NAME|wire.value>``, which made the help line advertise a topic called
    ``wire.value>``.  The separator is structure; a hint must not counterfeit
    it.
    """
    offenders = {scope: spec.arg for scope, spec in _SCOPES.items()
                 if "|" in spec.arg}
    assert not offenders, (
        f"argument hints containing the help separator '|': {offenders}. "
        f"Spell the alternation some other way ('<A or B>')."
    )


def test_dispatch_reads_only_the_scope_table():
    """No rung of the old ladder may grow back.

    The dispatch resolves a topic through ``_scope_renderer`` and nothing
    else: not a string literal, not a membership test against a second table.
    A topic reachable through any other rung would be dispatched and
    unadvertised, which is the defect, in the exact shape it took.

    **Both halves of the dispatch are scanned.**  It used to live wholly in
    ``_cmd_explain``; ``render_topic`` was extracted so the daemon's
    ``scaffold.explain`` verb renders through the same function the CLI does,
    and the resolution went with it.  A guard that kept reading only the
    printing half would have passed over a literal rung in the half that now
    does the lookup -- which is how this reversion went stale rather than
    failing: the anchor moved, and only the meta-guard noticed.

    The seam that lets an installed package contribute a topic is why this
    reads a RESOLVER rather than ``scope in _SCOPES`` as it did when it was
    written: there are now legitimately two tiers, and the point of the guard
    is that the dispatch cannot see them separately.  Everything the resolver
    serves is in :func:`_all_scopes_help`, which
    ``test_every_dispatched_scope_is_advertised`` checks the round trip of.
    """
    for fname in ("render_topic", "_cmd_explain"):
        fn = _function_ast(fname)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            if not (isinstance(node.left, ast.Name) and node.left.id == "scope"):
                continue
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    # `scope is None` is the overview branch and is an Is, not
                    # an Eq; a literal comparison here is a dispatch rung.
                    assert not isinstance(comparator, ast.Constant), (
                        f"{fname} compares scope against the literal "
                        f"{comparator.value!r} -- that topic is dispatched "
                        f"outside the resolver and so cannot reach the "
                        f"derived help."
                    )
                assert not isinstance(op, (ast.In, ast.NotIn)), (
                    f"{fname} dispatches on a table of its own "
                    f"({ast.dump(comparator)}); _scope_renderer must stay the "
                    f"one place a topic is looked up."
                )

    called = {n.func.id for n in ast.walk(_function_ast("render_topic"))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_scope_renderer" in called, (
        "render_topic no longer calls _scope_renderer -- it has grown its own "
        "way to find a topic, which is how a topic becomes dispatchable "
        "without being advertised."
    )


def test_every_declared_kind_has_a_handler_and_every_handler_is_used():
    """A new calling convention must be wired in, not appended as a branch."""
    declared = {spec.kind for spec in _SCOPES.values()}
    implemented = set(_SCOPE_KINDS)

    unhandled = sorted(declared - implemented)
    assert not unhandled, (
        f"scope kinds with no handler in _SCOPE_KINDS: {unhandled}. The "
        f"dispatch would raise KeyError for those topics."
    )

    unused = sorted(implemented - declared)
    assert not unused, (
        f"_SCOPE_KINDS handlers no scope declares: {unused}. Either a topic "
        f"was dropped or the handler is dead."
    )


def test_the_legacy_tables_are_projections_of_the_one_table():
    """The four old tables are derived views, not rival declarations.

    They are still imported by name elsewhere in the suite, so they stay -- but
    a name may not appear in one without being in ``_SCOPES``, or #994 is back
    with a different container.
    """
    views = {
        "simple": set(_SIMPLE_SCOPES),
        "filter": set(_FILTER_SCOPES),
        "named": set(_NAMED_SCOPES),
        "workspace": set(_WORKSPACE_SCOPES),
    }
    for kind, names in views.items():
        assert names == {n for n, s in _SCOPES.items() if s.kind == kind}, (
            f"the {kind} view disagrees with _SCOPES"
        )
        assert names <= set(_SCOPES)


def test_integrations_is_the_topic_that_was_missing():
    """The reported symptom, named.

    `explain integrations` worked and appeared in neither the error nor
    ``--help``.
    """
    assert "integrations" in _SCOPES
    assert "integrations" in _SCOPES_HELP


def test_unknown_scope_prints_the_derived_list_and_exits_2(capsys):
    """The error path is a consumer of the derived string, not a copy of it."""
    rc = scaffold_main.main(["explain", "definitely-not-a-scope"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown explain scope" in err
    assert _SCOPES_HELP in err


def test_a_named_scopes_usage_line_is_derived_from_its_hint(capsys):
    """`explain plugin` with no name prints usage built from the same entry."""
    rc = scaffold_main.main(["explain", "plugin"])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() == "usage: explain plugin <name>"
