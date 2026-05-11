"""AST partition gate for SessionManager-level session-bootstrap sites
(Phase 3 §3.12.0).

Asserts that every code path constructing a :class:`JaatoServer` +
attendant :class:`Session` record either (a) goes through the
unified :meth:`SessionManager._bootstrap_session` helper, or (b)
appears in an explicit allow-list of construction-permitted modules
+ functions.

Why an AST partition test:

The §3.12.0 spec carries the same shape as PLUGIN_TIER's build-fail
gate (§3.3.5 / ``test_plugin_tier_partition.py``): the discipline
this commit establishes is only enforceable structurally.  A
contributor adding a new daemon extension or transport that
constructs sessions inline would silently bypass the
``_bootstrap_session`` helper without this test catching it.

The allow-list deliberately enumerates the three §3.12 follow-on
paths that haven't migrated yet (disk-restore, ephemeral, WS
standalone), plus any test fixtures that legitimately construct
``JaatoServer`` directly (e.g., for testing the server itself
without going through the SessionManager bootstrap).  As §3.12
follow-ons land, entries here are removed; the partition gates
the migration arc.

Implementation: walk every Python file under ``server/`` and
``shared/`` for ``Call`` nodes whose ``func`` resolves to
``JaatoServer`` (in any aliasing form).  Each call site is
attributed to its enclosing function via a lightweight scope
walker.  Any (module, function) pair NOT in
:data:`PERMITTED_CONSTRUCTION_SITES` fails the test with a clear
explanation of how to remediate.

The intentional acceptance criteria per the §3.12.0 spec:

> Implementation: AST scan of ``server/**.py`` + ``shared/**.py``
> for any non-bootstrap-helper code path that constructs
> ``JaatoServer`` / ``JaatoRuntime`` / ``JaatoSession`` directly.
> An explicit allow-list of construction-permitted modules
> (``server/session_manager.py`` hosting ``_bootstrap_session``,
> plus test fixture modules under ``*/tests/``) defines the
> legitimate construction sites; anything else fails the test —
> catches the v1→v2 cycle's "no commit actually edited X" class
> of bug structurally.

This commit ships the JaatoServer scan (the §3.12.0 helper's
direct concern).  JaatoRuntime + JaatoSession remain
pre-existing-construction-sites that pass through unchecked
until their own migration arcs land — adding their scans now
would just allow-list every existing site, defeating the gate.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Set, Tuple

import pytest


# Project root; tests run from anywhere so anchor on this file's path.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCAN_ROOTS = (
    _REPO_ROOT / "jaato-server" / "server",
    _REPO_ROOT / "jaato-server" / "shared",
)

# (module-relative-path, function-name) tuples permitted to
# construct JaatoServer directly.  Each entry must carry a comment
# explaining why direct construction is justified (per the §3.12.0
# spec's contributor guidance).
PERMITTED_CONSTRUCTION_SITES: Set[Tuple[str, str]] = {
    # The unified bootstrap sub-helper — the single source of truth
    # that all SessionManager-level paths funnel through.  Phase 3
    # §3.12 disk-restore migration extracted this from
    # ``_bootstrap_session`` so the disk-restore path can share the
    # construction logic without inheriting the create-session
    # Session-record shape.  This entry stays permanently.
    ("server/session_manager.py", "_construct_and_initialize_server"),
    # WS standalone bootstrap — §3.12 follow-on.  The
    # ``_bootstrap_and_initialize`` helper here mirrors the
    # SessionManager pattern but is a separate code path that
    # doesn't yet route through ``_construct_and_initialize_server``.
    ("server/websocket.py", "_bootstrap_and_initialize"),
    # Test fixture in server/tests/ that constructs a server
    # directly to test outbound-event transformers.  Although
    # under server/ rather than under */tests/, it's a test file
    # (filename prefix ``test_``).  Allow-listed explicitly since
    # the directory-name filter doesn't match this case.
    ("server/test_outbound_event_transformers.py", "_server_with_capture"),
}


def _enclosing_function(
    tree: ast.AST,
    target: ast.Call,
) -> str:
    """Return the name of the function/method that contains *target*,
    or "<module>" if the call is at module top level.

    Walks the tree once collecting (FunctionDef|AsyncFunctionDef) →
    enclosed-call mappings; cheap enough for our scan size."""
    # Bottom-up search: walk all FunctionDef nodes and check whether
    # *target* is in their subtree.
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if inner is target:
                    candidates.append(node)
                    break
    if not candidates:
        return "<module>"
    # Most specific (deepest) wins — pick the function whose body
    # column-range is smallest (i.e., the innermost wrapping fn).
    candidates.sort(
        key=lambda fn: (fn.end_lineno or fn.lineno) - fn.lineno,
    )
    return candidates[0].name


def _is_jaato_server_call(node: ast.Call) -> bool:
    """Return True if *node* is a call to ``JaatoServer(...)`` (in
    any aliasing form: bare name, attribute access, etc.)."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "JaatoServer":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "JaatoServer":
        return True
    return False


def _iter_python_files(root: pathlib.Path):
    """Yield every .py file under *root*, skipping __pycache__ and
    test directories.

    Tests legitimately construct JaatoServer (fixture setup, unit
    tests of the server itself, etc.); we don't enforce the
    partition there per the spec's allow-list of "test fixture
    modules under */tests/"."""
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if "__pycache__" in parts:
            continue
        if "tests" in parts:
            continue
        yield path


def _find_jaato_server_constructions():
    """Return list of (module-rel-path, fn-name) for every
    ``JaatoServer(...)`` call site outside test directories."""
    sites = []
    for root in _SCAN_ROOTS:
        if not root.exists():
            continue
        for py_path in _iter_python_files(root):
            try:
                source = py_path.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                tree = ast.parse(source, filename=str(py_path))
            except SyntaxError:
                continue
            rel_path = py_path.relative_to(_REPO_ROOT / "jaato-server")
            rel_str = rel_path.as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_jaato_server_call(node):
                    fn_name = _enclosing_function(tree, node)
                    sites.append((rel_str, fn_name))
    return sites


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_jaato_server_construction_sites_are_partitioned() -> None:
    """Every non-test code path that constructs ``JaatoServer`` must
    be listed in :data:`PERMITTED_CONSTRUCTION_SITES`.

    A new daemon extension or transport that constructs sessions
    inline must EITHER:
      (a) route the new path through
          ``SessionManager._bootstrap_session(envelope)`` (preferred —
          the helper is the single source of truth), OR
      (b) extend ``PERMITTED_CONSTRUCTION_SITES`` with a comment
          explaining why direct construction is justified.
    """
    found = set(_find_jaato_server_constructions())
    unauthorized = found - PERMITTED_CONSTRUCTION_SITES
    assert not unauthorized, (
        f"Unauthorized JaatoServer construction sites detected:\n"
        f"  {sorted(unauthorized)!r}\n\n"
        f"Either route them through "
        f"``SessionManager._bootstrap_session(envelope)`` (Phase 3 "
        f"§3.12.0) or add the (module, function) tuple to "
        f"PERMITTED_CONSTRUCTION_SITES with a justification "
        f"comment in test_bootstrap_partition.py."
    )


def test_permitted_sites_are_actually_present() -> None:
    """Reverse direction: each entry in
    :data:`PERMITTED_CONSTRUCTION_SITES` must correspond to a real
    construction site in the source.  Stale entries indicate a
    site was migrated but the allow-list wasn't pruned — defeats
    the gate's value as the migration arc proceeds.

    A failure here means: prune the allow-list entry; the
    migration to ``_bootstrap_session`` is complete for that
    path."""
    found = set(_find_jaato_server_constructions())
    stale = PERMITTED_CONSTRUCTION_SITES - found
    assert not stale, (
        f"Stale entries in PERMITTED_CONSTRUCTION_SITES — these "
        f"sites no longer construct JaatoServer (migration "
        f"complete?):\n  {sorted(stale)!r}\n\n"
        f"Remove them from the allow-list to keep the partition "
        f"gate honest."
    )
