"""Regression-pin tests for §7c step 6.6.4.5e — the seat-flip
moment.

Atomic ``_jaato`` field removal + all dependent deletions.  Per
the audit-cross-reference table in the commit message, every
deletion in this commit is independently justified by a prior
audit (5a-5d):

| Deletion | Justified by |
|---|---|
| ``self._jaato = JaatoClient(...)`` + ``.connect(...)`` | 5d construction refactor |
| ``_jaato.configure_plugins_only`` / ``_jaato.configure_tools`` ×2 | 6.6.4.4 audit (WIRING fold) |
| ``_jaato.set_agent_identity`` / ``_jaato.set_ui_hooks`` | 5c.0 missing-method audit |
| ``self._jaato`` field declaration | zero consumers remain |
| 4 truthiness checks (``if not self._jaato``) | always-true post-seat-flip |
| ``_jaato.get_session()`` reference-authorizer leg | runner-RPC forwarder handles it |
| ``JaatoClient`` import | unused after construction drop |

After 5e:
- ``self._jaato`` field gone from JaatoServer
- ``JaatoClient`` no longer imported by server.core
- All call sites that previously read/wrote ``_jaato`` migrated to
  ``self._runtime`` (5a, 5d) or ``self._runner_rpc`` (5b, 5c.1-5c.5)
- §7c step 6.6.4 closes; seat-flip complete.
"""

from __future__ import annotations

import ast
import inspect

import server.core as core_module
import server.session_manager as sm_module
from server.core import JaatoServer


def _grep_attr_reads(module, attr_name: str) -> list:
    """Walk a module's AST and return Attribute nodes that read
    ``<anything>.<attr_name>`` (NOT comment text or docstrings)."""
    src = inspect.getsource(module)
    tree = ast.parse(src)
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == attr_name
    ]


# ----------------------------------------------------------------------
# Field removal pin
# ----------------------------------------------------------------------


def test_jaato_field_no_longer_referenced_in_core() -> None:
    """Pin: ``self._jaato`` / ``server._jaato`` is no longer
    accessed as an attribute anywhere in server.core (AST-level
    grep, ignores comment/docstring text mentions).

    This is the closure pin for the entire §7c step 6.6.4 series.
    """
    refs = _grep_attr_reads(core_module, "_jaato")
    assert refs == [], (
        f"§7c step 6.6.4.5e regression: ``_jaato`` attribute access "
        f"re-introduced in server/core.py ({len(refs)} sites).  Use "
        f"``self._runtime`` (for runtime-tier reads) or "
        f"``self._runner_rpc`` (for session-tier reads) instead."
    )


def test_jaato_field_no_longer_referenced_in_session_manager() -> None:
    """Pin: ``server._jaato`` is no longer accessed in
    session_manager.py.  Was used as a truthiness guard at line
    2704; migrated to ``server._runtime`` in 5e."""
    refs = _grep_attr_reads(sm_module, "_jaato")
    assert refs == [], (
        f"§7c step 6.6.4.5e regression: ``_jaato`` attribute access "
        f"re-introduced in server/session_manager.py ({len(refs)} "
        f"sites).  Use ``server._runtime`` instead."
    )


# ----------------------------------------------------------------------
# Import decoupling pin
# ----------------------------------------------------------------------


def test_jaato_client_not_imported_in_core() -> None:
    """Pin: ``JaatoClient`` is no longer imported into server.core
    after 5e.  The transitional dual construction is gone — daemon
    constructs ``JaatoRuntime`` directly.  ``JaatoClient`` stays
    only as the external-SDK facade."""
    assert not hasattr(core_module, "JaatoClient"), (
        "§7c step 6.6.4.5e regression: ``JaatoClient`` re-imported "
        "into server.core.  Daemon should not depend on JaatoClient "
        "post-seat-flip — direct ``JaatoRuntime(...)`` construction is "
        "the daemon's source of runtime state."
    )


# ----------------------------------------------------------------------
# Construction-site pin (the seat-flip moment)
# ----------------------------------------------------------------------


def test_initialize_constructs_only_jaato_runtime_not_jaato_client() -> None:
    """Pin: ``initialize()`` constructs ``JaatoRuntime`` (the
    daemon-direct seat) but NOT ``JaatoClient`` (which was the
    transitional construction in 5d).  Closes the seat-flip."""
    import textwrap
    src = textwrap.dedent(inspect.getsource(JaatoServer.initialize))
    tree = ast.parse(src)
    runtime_constructs = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "JaatoRuntime"
        )
    ]
    client_constructs = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "JaatoClient"
        )
    ]
    assert len(runtime_constructs) >= 1, (
        "JaatoRuntime construction missing from initialize()."
    )
    assert client_constructs == [], (
        f"§7c step 6.6.4.5e regression: ``JaatoClient(...)`` "
        f"construction re-introduced in initialize() "
        f"({len(client_constructs)} sites)."
    )


# ----------------------------------------------------------------------
# Deleted-call-site pins (each justified by a prior audit)
# ----------------------------------------------------------------------


def test_no_jaato_configure_plugins_only_calls() -> None:
    """Justified by 6.6.4.4 audit (WIRING bucket fold)."""
    src = inspect.getsource(core_module)
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "configure_plugins_only"
            and isinstance(n.func.value, ast.Attribute)
            and n.func.value.attr == "_jaato"
        )
    ]
    assert calls == [], (
        f"§7c step 6.6.4.5e regression: "
        f"``_jaato.configure_plugins_only(...)`` re-introduced "
        f"({len(calls)} sites)."
    )


def test_no_jaato_configure_tools_calls() -> None:
    """Justified by 6.6.4.4 audit (WIRING bucket fold)."""
    src = inspect.getsource(core_module)
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "configure_tools"
            and isinstance(n.func.value, ast.Attribute)
            and n.func.value.attr == "_jaato"
        )
    ]
    assert calls == [], (
        f"§7c step 6.6.4.5e regression: ``_jaato.configure_tools(...)`` "
        f"re-introduced ({len(calls)} sites)."
    )


def test_no_jaato_set_agent_identity_calls() -> None:
    """Justified by 5c.0 missing-method audit — daemon-side
    JaatoClient state mutation with no runner-side propagation."""
    src = inspect.getsource(core_module)
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_agent_identity"
            and isinstance(n.func.value, ast.Attribute)
            and n.func.value.attr == "_jaato"
        )
    ]
    assert calls == [], (
        f"§7c step 6.6.4.5e regression: ``_jaato.set_agent_identity(...)`` "
        f"re-introduced ({len(calls)} sites)."
    )


def test_no_jaato_set_ui_hooks_calls() -> None:
    """Justified by 5c.0 missing-method audit + Finding 3 (runner-side
    ``_ui_hooks`` is already None post-6.6.4.3b; deletion has no
    behavioral effect)."""
    src = inspect.getsource(core_module)
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_ui_hooks"
            and isinstance(n.func.value, ast.Attribute)
            and n.func.value.attr == "_jaato"
        )
    ]
    assert calls == [], (
        f"§7c step 6.6.4.5e regression: ``_jaato.set_ui_hooks(...)`` "
        f"re-introduced ({len(calls)} sites)."
    )


# ----------------------------------------------------------------------
# Runtime-as-canonical-handle pin (the post-5e shape)
# ----------------------------------------------------------------------


def test_runtime_is_canonical_daemon_handle() -> None:
    """Pin: ``self._runtime`` is the daemon's canonical runtime
    handle post-5e.  Field declaration intact + construction in
    initialize() + post-join configure_plugins call."""
    init_src = inspect.getsource(JaatoServer.__init__)
    assert "self._runtime" in init_src, (
        "self._runtime field declaration missing"
    )
    initialize_src = inspect.getsource(JaatoServer.initialize)
    assert "self._runtime = JaatoRuntime(" in initialize_src, (
        "Daemon-direct JaatoRuntime construction missing from "
        "initialize()."
    )
    assert "self._runtime.configure_plugins(" in initialize_src, (
        "Post-threadpool-join runtime.configure_plugins call missing."
    )
