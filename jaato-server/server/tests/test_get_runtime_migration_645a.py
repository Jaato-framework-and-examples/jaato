"""Regression-pin tests for §7c step 6.6.4.5a — migrate
``_jaato.get_runtime()`` reads to ``_runtime`` direct field
access.

Per the §7c step 6.6.4.5 implementation-review audit (G3 split,
Refinement 1): 5a is the lowest-risk mechanical sub-commit.
Migrates 4 daemon-side call sites in `session_manager.py` (3) +
`websocket.py` (1) from the indirection
``server._jaato.get_runtime()`` → direct ``server._runtime``.

The 5th `get_runtime()` site (core.py:1565) is the **populator**
of `self._runtime` — it stays until 5d's construction refactor
replaces the JaatoClient with direct JaatoRuntime construction.

Behavior preservation: ``_runtime`` is non-None iff ``_jaato`` was
successfully connected (populated at line 1565 inside
``initialize()``).  The truthiness checks pivot from
``if server._jaato:`` → implicit ``runtime is not None`` checks
that already existed downstream.

Tests pin via AST that the deleted call patterns stay deleted
(call-site grep, not text-match — docstrings legitimately
mention the migration source).
"""

from __future__ import annotations

import ast
import inspect

import server.core as core_module
import server.session_manager as sm_module
import server.websocket as ws_module


def _module_calls_jaato_method(module, method_name: str) -> list:
    """Walk the module AST and return ``Call`` nodes shaped like
    ``<anything>._jaato.<method_name>(...)`` — matches both
    ``self._jaato.X`` and ``server._jaato.X`` and
    ``session.server._jaato.X``."""
    src = inspect.getsource(module)
    tree = ast.parse(src)
    found = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if not isinstance(f, ast.Attribute) or f.attr != method_name:
            continue
        inner = f.value
        if isinstance(inner, ast.Attribute) and inner.attr == "_jaato":
            found.append(n)
    return found


# ----------------------------------------------------------------------
# Deletion pins: session_manager.py + websocket.py have no
# remaining `_jaato.get_runtime()` call sites.
# ----------------------------------------------------------------------


def test_session_manager_no_jaato_get_runtime_calls_remain() -> None:
    """All 3 ``server._jaato.get_runtime()`` call sites in
    session_manager.py migrated to ``server._runtime`` in §7c
    step 6.6.4.5a."""
    calls = _module_calls_jaato_method(sm_module, "get_runtime")
    assert calls == [], (
        f"§7c step 6.6.4.5a regression: ``_jaato.get_runtime()`` call "
        f"re-introduced in server/session_manager.py ({len(calls)} sites). "
        f"Use ``server._runtime`` directly."
    )


def test_websocket_no_jaato_get_runtime_calls_remain() -> None:
    """The 1 ``session.server._jaato.get_runtime()`` call site in
    websocket.py migrated to ``session.server._runtime`` in
    §7c step 6.6.4.5a."""
    calls = _module_calls_jaato_method(ws_module, "get_runtime")
    assert calls == [], (
        f"§7c step 6.6.4.5a regression: ``_jaato.get_runtime()`` call "
        f"re-introduced in server/websocket.py ({len(calls)} sites). "
        f"Use ``session.server._runtime`` directly."
    )


# ----------------------------------------------------------------------
# Preservation pin: the populator at core.py:1565 stays until 5d.
# ----------------------------------------------------------------------


def test_core_populator_get_runtime_call_still_present_until_5d() -> None:
    """Pin the construction-site populator
    ``self._runtime = self._jaato.get_runtime()`` is NOT yet
    deleted — it's the only place that populates
    ``self._runtime``.  Stays until §7c step 6.6.4.5d's
    construction refactor replaces the JaatoClient pathway with
    direct JaatoRuntime construction.  If this test starts failing,
    5d has landed and this pin can be deleted."""
    calls = _module_calls_jaato_method(core_module, "get_runtime")
    assert len(calls) >= 1, (
        f"Expected ≥1 ``_jaato.get_runtime()`` call site in core.py "
        f"(the construction-time populator at line 1565); found "
        f"{len(calls)}.  If 5d has landed and the populator was "
        f"refactored, delete this pin."
    )


# ----------------------------------------------------------------------
# Pattern preservation: callers should read _runtime directly.
# ----------------------------------------------------------------------


def test_session_manager_reads_runtime_direct() -> None:
    """Pin the migrated sites in session_manager.py read
    ``server._runtime`` directly (no helper method call)."""
    src = inspect.getsource(sm_module)
    # The migrated comment marker should appear at all 3 sites.
    marker_count = src.count(
        "Phase 3 §7c step 6.6.4.5a: read ``server._runtime`` directly"
    )
    assert marker_count == 3, (
        f"Expected 3 §7c step 6.6.4.5a migration markers in "
        f"session_manager.py; found {marker_count}.  Comments document "
        f"each migrated site for future maintainers."
    )


def test_websocket_reads_runtime_direct() -> None:
    """Pin the migrated site in websocket.py reads
    ``session.server._runtime`` directly."""
    src = inspect.getsource(ws_module)
    assert (
        "Phase 3 §7c step 6.6.4.5a: read ``session.server._runtime`` directly"
        in src
    ), (
        "Expected §7c step 6.6.4.5a migration marker in "
        "server/websocket.py; not found.  Comment documents the "
        "migrated site for future maintainers."
    )
