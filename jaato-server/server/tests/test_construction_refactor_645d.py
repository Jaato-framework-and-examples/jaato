"""Regression-pin tests for §7c step 6.6.4.5d — daemon-direct
JaatoRuntime construction.

Per the §7c step 6.6.4.5d disposition audit (commit 2a69f82e,
Path A): daemon now constructs ``JaatoRuntime`` directly inside
``_run_connect_provider`` instead of aliasing from
``self._jaato.get_runtime()``.

JaatoClient construction stays in this commit (transitional) —
5 unguarded daemon-side calls (``configure_plugins_only``,
``configure_tools`` ×2, ``set_agent_identity``, ``set_ui_hooks``)
still operate on it.  5e drops the JaatoClient + the dependent
calls atomically.

Audit findings (1-5) validated:

1. JaatoRuntime.__init__ signature — 4 primitive args, zero
   client references; daemon-direct construction safe.
2. JaatoClient.connect() side-effects — 3 of 5 are client-state
   stores the daemon doesn't need.
3. Auth flow — already on JaatoRuntime (verify_auth) at line
   1829; daemon already calls runtime.verify_auth() directly.
4. SDK consumers — only 1 production caller of JaatoClient (the
   transitional construction here).
5. Construction site — preserves ThreadPoolExecutor concurrency
   with plugin loading.

Tests pin (per the user's expected ~8-12 tests):

- Daemon-direct JaatoRuntime construction in _run_connect_provider
- Daemon-direct runtime.connect() invocation
- Daemon-direct runtime.configure_plugins() after threadpool join
- ThreadPoolExecutor scope preserved (concurrency invariant)
- _jaato.get_runtime() alias deleted (regression pin)
- Confine-context-factory propagation still wires (operates on
  daemon-direct runtime now)
- JaatoClient construction stays transitionally (for the 5
  dependent unguarded calls 5e drops)
- JaatoRuntime imported at module level
"""

from __future__ import annotations

import ast
import inspect

import server.core as core_module
from server.core import JaatoServer


def _function_calls_in(func, attr_name: str) -> list:
    """Walk a function's AST and return Call nodes shaped like
    ``<anything>.<attr_name>(...)``."""
    src = inspect.getsource(func)
    import textwrap
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    return [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr_name
        )
    ]


def _function_constructs(func, class_name: str) -> list:
    """Walk a function's AST and return Call nodes that look
    like a constructor call ``<class_name>(...)`` (Name lookup,
    not attribute lookup)."""
    src = inspect.getsource(func)
    import textwrap
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    return [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == class_name
        )
    ]


# ----------------------------------------------------------------------
# Construction-site pins (the architectural pivot)
# ----------------------------------------------------------------------


def test_initialize_constructs_jaato_runtime_directly() -> None:
    """Pin daemon constructs JaatoRuntime directly (Path A) inside
    initialize()'s _run_connect_provider thread function.  The
    _run_connect_provider function is a nested closure inside
    initialize, so we walk initialize()'s AST."""
    calls = _function_constructs(JaatoServer.initialize, "JaatoRuntime")
    assert len(calls) >= 1, (
        f"§7c step 6.6.4.5d regression: ``JaatoRuntime(...)`` "
        f"daemon-direct construction not found in initialize() "
        f"({len(calls)} call sites).  Path A requires daemon to "
        f"construct JaatoRuntime independently of JaatoClient."
    )


def test_initialize_calls_runtime_connect_directly() -> None:
    """Pin daemon calls ``runtime.connect(project, location)``
    directly post-construction (the second daemon-relevant op
    from the audit's finding #2)."""
    calls = _function_calls_in(JaatoServer.initialize, "connect")
    # Filter to runtime.connect (vs jaato.connect).
    runtime_connects = [
        c for c in calls
        if isinstance(c.func.value, ast.Attribute)
        and c.func.value.attr == "_runtime"
    ]
    assert len(runtime_connects) >= 1, (
        f"§7c step 6.6.4.5d regression: ``self._runtime.connect(...)`` "
        f"call missing from initialize() ({len(runtime_connects)} sites). "
        f"Daemon-direct construction needs ``runtime.connect()`` to "
        f"populate the provider config."
    )


def test_initialize_calls_runtime_configure_plugins() -> None:
    """Pin daemon calls ``self._runtime.configure_plugins(...)``
    after the threadpool join — needed because the daemon-direct
    runtime is independent of JaatoClient's internal runtime, and
    daemon-side reads on ``self._runtime.registry`` (e.g.
    session_manager.py:3365) require the runtime to know about
    the plugin registry."""
    calls = _function_calls_in(JaatoServer.initialize, "configure_plugins")
    # Filter to self._runtime.configure_plugins specifically.
    runtime_configures = [
        c for c in calls
        if isinstance(c.func.value, ast.Attribute)
        and c.func.value.attr == "_runtime"
    ]
    assert len(runtime_configures) >= 1, (
        f"§7c step 6.6.4.5d regression: "
        f"``self._runtime.configure_plugins(...)`` call missing from "
        f"initialize() ({len(runtime_configures)} sites).  Without "
        f"this, daemon-side ``self._runtime.registry`` reads return "
        f"None."
    )


# ----------------------------------------------------------------------
# Deletion pins (the populator + the alias indirection)
# ----------------------------------------------------------------------


def test_initialize_no_jaato_get_runtime_alias() -> None:
    """Pin the pre-§7c-step-6.6.4.5d alias
    ``self._runtime = self._jaato.get_runtime()`` (was line 1572)
    is deleted.  Daemon-direct construction replaces the
    indirection."""
    calls = _function_calls_in(JaatoServer.initialize, "get_runtime")
    # Filter to _jaato.get_runtime() specifically (not other
    # get_runtime calls that might exist for testing).
    jaato_get_runtimes = [
        c for c in calls
        if isinstance(c.func.value, ast.Attribute)
        and c.func.value.attr == "_jaato"
    ]
    assert jaato_get_runtimes == [], (
        f"§7c step 6.6.4.5d regression: ``self._jaato.get_runtime()`` "
        f"alias re-introduced in initialize() "
        f"({len(jaato_get_runtimes)} sites).  Daemon should construct "
        f"``JaatoRuntime`` directly, not alias from JaatoClient."
    )


# ----------------------------------------------------------------------
# Concurrency-preservation pin (per the user's flagged observation)
# ----------------------------------------------------------------------


def test_initialize_preserves_threadpool_executor_construction() -> None:
    """Pin the construction site stays inside the
    ThreadPoolExecutor's executor-thread scope (concurrent with
    plugin loading, ~100-200ms saved during bootstrap).  The
    user flagged this as load-bearing in the 5d green-light:
    'concurrency-preservation pin asserting _runtime is populated
    within the executor's task scope.'

    Detection strategy: the JaatoRuntime construction must appear
    INSIDE a function definition (the _run_connect_provider
    nested function), not at the top level of initialize().
    The nested function is what gets pool.submit'd."""
    src = inspect.getsource(JaatoServer.initialize)
    import textwrap
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    # Find the JaatoRuntime construction node.
    runtime_constructs = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "JaatoRuntime"
        )
    ]
    assert len(runtime_constructs) >= 1, (
        "JaatoRuntime construction missing — see "
        "test_initialize_constructs_jaato_runtime_directly"
    )

    # Walk the AST tracking enclosing FunctionDef ancestors for
    # each Call node.  The construction must be inside a nested
    # FunctionDef (depth >= 2 — initialize itself is depth 1).
    def _find_function_depth(target: ast.Call) -> int:
        """Return the nesting depth of FunctionDef ancestors
        enclosing ``target``."""
        for outer in ast.walk(tree):
            if not isinstance(outer, ast.FunctionDef):
                continue
            for child in ast.walk(outer):
                if child is target:
                    # Now find the deepest FunctionDef enclosing target.
                    depth = 1
                    for inner in ast.walk(outer):
                        if (
                            isinstance(inner, ast.FunctionDef)
                            and inner is not outer
                        ):
                            for sub in ast.walk(inner):
                                if sub is target:
                                    depth = 2
                                    break
                    return depth
        return 0

    depths = [_find_function_depth(c) for c in runtime_constructs]
    assert any(d >= 2 for d in depths), (
        f"§7c step 6.6.4.5d regression: JaatoRuntime construction "
        f"escaped the ThreadPoolExecutor's nested-function scope. "
        f"Construction depths: {depths}.  Must be inside the "
        f"_run_connect_provider nested function (depth 2), not at "
        f"initialize()'s top level (depth 1).  The concurrency "
        f"with plugin loading saves ~100-200ms during bootstrap."
    )


# ----------------------------------------------------------------------
# Transitional-state pin (JaatoClient still constructed in 5d)
# ----------------------------------------------------------------------


def test_initialize_jaato_client_construction_dropped_in_5e() -> None:
    """Pin JaatoClient construction was dropped by §7c step 6.6.4.5e's
    seat-flip moment.  Daemon constructs JaatoRuntime directly; no
    JaatoClient is constructed daemon-side anymore."""
    calls = _function_constructs(JaatoServer.initialize, "JaatoClient")
    assert calls == [], (
        f"§7c step 6.6.4.5e regression: ``JaatoClient(...)`` "
        f"construction re-introduced in initialize() "
        f"({len(calls)} sites).  Daemon should construct "
        f"``JaatoRuntime`` directly; JaatoClient stays only as the "
        f"external-SDK facade."
    )


# ----------------------------------------------------------------------
# Confine-context-factory propagation pin
# ----------------------------------------------------------------------


def test_initialize_confine_context_factory_still_propagates() -> None:
    """Pin the AppArmor confine-context-factory propagation
    (server 0.6.50+) still wires to the daemon-direct runtime.
    Pre-§7c-step-6.6.4.5d this operated on the alias from
    JaatoClient; post-5d it operates on the daemon-direct
    instance."""
    src = inspect.getsource(JaatoServer.initialize)
    assert "set_confine_context_factory" in src, (
        "§7c step 6.6.4.5d regression: confine-context-factory "
        "propagation removed from initialize().  Sessions created "
        "on the daemon-direct runtime need this for AppArmor "
        "wrapping of dynamic-instructions expansion."
    )


# ----------------------------------------------------------------------
# Module-level import pin
# ----------------------------------------------------------------------


def test_jaato_runtime_imported_at_module_level() -> None:
    """Pin JaatoRuntime is importable from server.core's namespace
    (added in 5d alongside the existing JaatoClient import).
    Required for the daemon-direct construction site."""
    assert hasattr(core_module, "JaatoRuntime"), (
        "§7c step 6.6.4.5d regression: ``JaatoRuntime`` not "
        "importable at server.core module level.  Daemon-direct "
        "construction requires the import in the ``from shared "
        "import ...`` block."
    )


# ----------------------------------------------------------------------
# Self._runtime field declaration pin
# ----------------------------------------------------------------------


def test_runtime_field_still_declared() -> None:
    """Pin the ``self._runtime: Optional[JaatoRuntime] = None`` field
    declaration in ``__init__`` survives 5d.  This is the daemon-side
    canonical handle (5e drops the parallel ``self._jaato`` field)."""
    src = inspect.getsource(JaatoServer.__init__)
    assert "self._runtime" in src, (
        "§7c step 6.6.4.5d regression: ``self._runtime`` field "
        "declaration removed from JaatoServer.__init__."
    )
