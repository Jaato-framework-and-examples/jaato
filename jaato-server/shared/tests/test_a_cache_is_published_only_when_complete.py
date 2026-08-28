"""A lazily-discovered cache must never be observable half-built.

The defect this guards (#652): ``_discover_secret_resolvers`` assigned its
module global ``_resolvers = {}`` at the TOP of the function and filled the
dict afterwards, while the fast path used ``_resolvers is not None`` to mean
"already discovered".  Everything in between is slow -- ``entry_points()``
scans installed distributions and ``ep.load()`` imports jaato-premium -- so a
caller arriving in that window took the fast path and got an EMPTY registry.
It reported ``(available: none)`` and passed a literal ``pass://`` URI to a
provider as its api_key.

There is already a behavioural guard for this
(``test_second_caller_never_sees_an_empty_registry`` in the subagent plugin
tests) which races two threads at a barrier.  This one checks the same
property STRUCTURALLY, for two reasons:

* it is deterministic -- no threads, no barrier, no scheduler luck;
* it is an INDEPENDENT witness.  A thread test can go quietly stale if a
  refactor moves the window out from under its barrier: it keeps passing
  while guarding nothing.

Shape note (see the AST-guard vacuity modes): every assertion here is a COUNT
or an IDENTITY claim, never "the first X after Y" -- ``ast.walk`` yields
breadth-first, so ordering logic built on it is vacuous.  The mode these CAN
suffer is the empty match: a rename or a move makes the anchor match nothing
and the property assertion then passes trivially.  So each anchor is asserted
FIRST, with a message saying the guard is stale rather than satisfied.
"""

import ast
import pathlib


# Resolved from __file__, not the CWD.  A relative path would make a wrong
# working directory fire the "guard is stale" anchor below, sending the reader
# to hunt a rename that never happened.
CONFIG = (pathlib.Path(__file__).resolve().parents[1]
          / "plugins" / "subagent" / "config.py")
GLOBAL = "_resolvers"
LOCK = "_resolvers_lock"
DISCOVERY = "_discover_secret_resolvers"


def _tree():
    return ast.parse(CONFIG.read_text(encoding="utf-8"))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _stores_to_global(node):
    """Every ``_resolvers = ...`` statement anywhere under *node*.

    Only Store context -- ``return _resolvers`` and ``if _resolvers is not
    None`` are Loads and are not writes.
    """
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Name) and tgt.id == GLOBAL:
                    found.append(n)
    return found


def _mentions_lock(with_node):
    return any(
        isinstance(n, ast.Name) and n.id == LOCK
        for item in with_node.items
        for n in ast.walk(item.context_expr)
    )


def test_the_module_still_has_the_pieces_this_guard_inspects():
    """Anchor. Without this, every assertion below passes on an empty match."""
    assert CONFIG.exists(), f"{CONFIG} not found — guard is stale, not passing"
    tree = _tree()

    assert _find_function(tree, DISCOVERY) is not None, (
        f"no function named {DISCOVERY!r} — this guard can no longer find "
        "what it inspects and must be re-aimed, NOT deleted"
    )
    assert any(
        isinstance(n, ast.Name) and n.id == LOCK for n in ast.walk(tree)
    ), (
        f"no reference to {LOCK!r} — the lock this guard assumes is gone; "
        "re-aim the guard or explain how the invariant is held without it"
    )
    stores = _stores_to_global(tree)
    assert len(stores) >= 2, (
        f"expected at least 2 assignments to {GLOBAL!r} module-wide (the "
        f"publish in {DISCOVERY} and the reset), found {len(stores)} — the "
        "guard is looking at the wrong name"
    )


def test_the_cache_is_published_in_one_assignment_of_a_finished_value():
    """The publish is a single store of an already-complete local.

    ``_resolvers = {}`` followed by ``_resolvers[k] = v`` is the bug: the
    global is readable between those two statements.  ``_resolvers =
    discovered`` is not, because ``discovered`` is finished before the name
    is ever bound.
    """
    fn = _find_function(_tree(), DISCOVERY)
    assert fn is not None, "discovery function not found — guard is stale"

    stores = _stores_to_global(fn)
    assert len(stores) == 1, (
        f"{DISCOVERY} assigns {GLOBAL!r} {len(stores)} times at line(s) "
        f"{[n.lineno for n in stores]}; it must publish exactly once, with a "
        "value that is already complete. Assigning it twice (or assigning "
        "then mutating) makes the half-built registry readable by any "
        "concurrent caller taking the is-not-None fast path."
    )

    published = stores[0].value
    assert isinstance(published, ast.Name), (
        f"line {stores[0].lineno}: {GLOBAL} is published as "
        f"{type(published).__name__}, i.e. a value built in place. It must be "
        "assigned a local that is already finished — otherwise the global is "
        "observable while it is still being filled, which is exactly #652."
    )


def test_every_write_to_the_cache_happens_under_the_lock():
    """One rule for every write, so a reader sees None or a finished registry.

    Includes ``reset_secret_resolvers``: if the reset writes outside the lock
    there is a third state again, just a shorter-lived one.
    """
    tree = _tree()
    all_stores = _stores_to_global(tree)

    guarded = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and _mentions_lock(node):
            guarded.extend(_stores_to_global(node))
    guarded_ids = {id(n) for n in guarded}

    assert guarded, (
        f"no assignment to {GLOBAL!r} occurs inside a `with {LOCK}` block — "
        "either the locking was removed or this guard no longer recognises "
        "it; re-aim rather than delete"
    )

    unguarded = [n.lineno for n in all_stores if id(n) not in guarded_ids]
    assert not unguarded, (
        f"{GLOBAL} is assigned outside `with {LOCK}` at line(s) {unguarded}. "
        "Every write must happen under the lock, so a reader taking the "
        "is-not-None fast path sees either None or a finished registry and "
        "never a third state."
    )
