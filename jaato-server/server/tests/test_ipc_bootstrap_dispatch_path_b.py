"""Regression-pin tests for the post-Step-7 Path B fix —
synchronous IPC session bootstrap dispatch.

**Root cause** (integration-test diagnosis):

The IPC pre-init path (``SessionManager._provision_ipc_apparmor_and_spawn_runner``
→ ``_spawn_session_runner_unconditional``) spawned the runner but
**never dispatched** the ``session.bootstrap`` RPC.  The WS path
called both ``spawn_session_runner`` AND
``dispatch_bootstrap_envelope`` (websocket.py:663+:690) — the
asymmetry left IPC sessions with ``RunnerRPC._session_host = None``
for the session lifetime.

Every daemon-side ``self._runner_rpc.session_X_threadsafe()`` call
during ``server.initialize()`` then raced against an
unbootstrapped runner-side session.  The handler correctly
returned ``stage="no_session"`` per the §3.3c surface defensive
contract, but the wrapper raised ``RunnerCallError`` on
``ok=False`` envelopes — crashing daemon-side init for any
profile that triggered early session.* RPCs.

The narrow Path A fix (commit 20b16326) wrapped 2 specific
callsites but the next race-victim hit immediately — whack-a-mole
shape.

**Path B fix** (architectural — what THIS commit lands):

Mirror the WS path's bootstrap dispatch inside
``_spawn_session_runner_unconditional`` so ``session.bootstrap``
completes synchronously BEFORE ``_provision_ipc_apparmor_and_spawn_runner``
returns.  ``server.initialize()`` then runs with a bootstrapped
runner-side host; all downstream ``session.*`` RPCs see a valid
session.

Tests pin:

- ``_spawn_session_runner_unconditional`` calls
  ``dispatch_bootstrap_envelope`` after ``spawn_session_runner``
  (AST-level + structural).
- Bootstrap dispatch is wrapped in try/except so a hiccup
  doesn't roll back the spawn-success return.
- WS path's bootstrap dispatch sequence is unchanged
  (regression pin against accidental WS path edits).
"""

from __future__ import annotations

import ast
import inspect

import server.session_manager as sm_module
import server.websocket as ws_module
from server.session_manager import SessionManager


# ----------------------------------------------------------------------
# AST helpers
# ----------------------------------------------------------------------


def _function_calls_in(func, name: str) -> list:
    """Return Call nodes inside *func* whose ``func`` is a bare
    ``Name(name)`` (function call, not attribute access).  Used to
    detect ``spawn_session_runner(...)`` /
    ``dispatch_bootstrap_envelope(...)`` invocations."""
    import textwrap
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    return [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == name
        )
    ]


# ----------------------------------------------------------------------
# IPC path: synchronous bootstrap dispatch (Path B fix)
# ----------------------------------------------------------------------


def test_ipc_spawn_helper_dispatches_bootstrap_envelope() -> None:
    """Pin: ``SessionManager._spawn_session_runner_unconditional``
    calls ``dispatch_bootstrap_envelope`` after ``spawn_session_runner``.

    Pre-Path-B: only ``spawn_session_runner`` was called — the
    runner-side session host was never bootstrapped on IPC
    sessions, causing every daemon-side ``session.*`` RPC during
    ``server.initialize()`` to race against an unbootstrapped
    host."""
    spawn_calls = _function_calls_in(
        SessionManager._spawn_session_runner_unconditional,
        "spawn_session_runner",
    )
    dispatch_calls = _function_calls_in(
        SessionManager._spawn_session_runner_unconditional,
        "dispatch_bootstrap_envelope",
    )
    assert len(spawn_calls) >= 1, (
        "spawn_session_runner call missing from "
        "_spawn_session_runner_unconditional — has the IPC path "
        "structure changed?"
    )
    assert len(dispatch_calls) >= 1, (
        "Path B regression: "
        "``dispatch_bootstrap_envelope`` call missing from "
        "``_spawn_session_runner_unconditional``.  This is the "
        "structural fix for the IPC bootstrap race — without it, "
        "the runner-side session host stays None and every "
        "daemon-side session.* RPC races."
    )


def test_ipc_bootstrap_dispatch_after_spawn() -> None:
    """Pin the dispatch happens AFTER the spawn (line order matters
    — dispatch must have a spawned runner with an attached
    ``runner_rpc``)."""
    spawn_calls = _function_calls_in(
        SessionManager._spawn_session_runner_unconditional,
        "spawn_session_runner",
    )
    dispatch_calls = _function_calls_in(
        SessionManager._spawn_session_runner_unconditional,
        "dispatch_bootstrap_envelope",
    )
    assert min(c.lineno for c in spawn_calls) < min(
        c.lineno for c in dispatch_calls
    ), (
        "Path B regression: ``dispatch_bootstrap_envelope`` must "
        "be called AFTER ``spawn_session_runner`` — the dispatch "
        "needs the spawned runner's RPC handle attached to the "
        "server via ``set_runner_rpc`` (which spawn_session_runner "
        "calls)."
    )


def test_ipc_bootstrap_dispatch_is_failure_tolerant() -> None:
    """Pin the dispatch call is wrapped in try/except so a
    bootstrap-RPC hiccup doesn't roll back the spawn-success
    return.  Matches ``dispatch_bootstrap_envelope``'s own
    failure-tolerant contract (its internal logic logs WARNING
    + returns on transport errors; the outer wrap here handles
    setup-time AttributeError for test fakes)."""
    import textwrap
    src = textwrap.dedent(
        inspect.getsource(SessionManager._spawn_session_runner_unconditional),
    )
    tree = ast.parse(src)

    # Find the dispatch_bootstrap_envelope Call node.
    dispatch_call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dispatch_bootstrap_envelope"
        ):
            dispatch_call = node
            break
    assert dispatch_call is not None

    # Verify it's inside a Try block.
    inside_try = False
    for try_node in ast.walk(tree):
        if not isinstance(try_node, ast.Try):
            continue
        for child in ast.walk(try_node):
            if child is dispatch_call:
                inside_try = True
                break
        if inside_try:
            break

    assert inside_try, (
        "Path B regression: ``dispatch_bootstrap_envelope`` "
        "call must be wrapped in try/except inside "
        "``_spawn_session_runner_unconditional`` so a bootstrap-"
        "RPC hiccup doesn't roll back the spawn-success return.  "
        "Particularly important for test stubs where "
        "``server.runner_rpc`` may be absent."
    )


# ----------------------------------------------------------------------
# WS path: unchanged regression pin
# ----------------------------------------------------------------------


def test_ws_bootstrap_dispatch_unchanged() -> None:
    """Pin the WS path's bootstrap-dispatch sequence is
    unchanged.  The WS pre-init hook has shipped this pattern
    since §7c step 2; Path B mirrors it for IPC but must NOT
    accidentally edit the WS path."""
    src = inspect.getsource(ws_module)
    # WS path uses dispatch_bootstrap_envelope inside its
    # apparmor pre-init hook.
    assert "dispatch_bootstrap_envelope" in src, (
        "WS path's bootstrap dispatch missing — Path B's IPC fix "
        "must not affect the WS path."
    )


# ----------------------------------------------------------------------
# Architectural invariant pin
# ----------------------------------------------------------------------


def test_ipc_and_ws_paths_both_dispatch_bootstrap_envelope() -> None:
    """Architectural pin: BOTH the IPC and WS paths dispatch
    ``session.bootstrap`` synchronously before
    ``server.initialize()`` runs.

    Pre-Path-B only the WS path did this; the IPC path was
    structurally asymmetric.  This pin catches accidental
    re-introduction of the asymmetry."""
    ipc_src = inspect.getsource(
        SessionManager._spawn_session_runner_unconditional,
    )
    ws_src = inspect.getsource(ws_module)
    assert "dispatch_bootstrap_envelope" in ipc_src, (
        "IPC path missing bootstrap dispatch — Path B regression"
    )
    assert "dispatch_bootstrap_envelope" in ws_src, (
        "WS path missing bootstrap dispatch — should be unchanged "
        "from pre-Path-B"
    )
