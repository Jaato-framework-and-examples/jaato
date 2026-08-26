"""Regression-pin tests for §7c step 6.6.4.5b — migrate
existing-RPC daemon-side reads to runner-RPC threadsafe wrappers.

Per the §7c step 6.6.4.5 implementation-review audit (G3 split,
revised in mid-flight per the user's pre-greenlit in-flight-split
policy): 5b's scope narrowed to **15 truly-mechanical migrations**
through existing RPC handlers.

The original audit's "8 daemon-side runtime reads" category was
mis-labeled as runtime-tier; cross-grep verified those methods
live on `JaatoSession` (`auth_info`, `get_user_commands`,
`execute_user_command`, `get_model_completions`).  Migration via
new RPCs deferred to 5c.

The `get_tool_schemas` cache add (Refinement 2) was also deferred
to 5c after verifying that `JaatoRuntime.get_tool_schemas()` returns
the registry's full set, NOT the session-resolved subset that
`JaatoSession.get_tool_schemas()` returns — semantically different
and would cause behavior regressions at the
`signal_completion_in_surface` filtering site.

Sites covered (15 mechanical migrations):

| Method | Sites | RPC used |
|---|---|---|
| `get_context_usage` | 4 (core.py:2052, 3649, 4486; session_manager.py:2626) | `session_get_context_usage_threadsafe` |
| `get_context_limit` | 6 (core.py:2053, 2612, 2766, 3513, 3650, 4487) | `session_get_context_limit_threadsafe` |
| `get_turn_accounting` | 1 (core.py:3518) | `session_get_turn_accounting_threadsafe` |
| `set_initial_history` (replaces `reset_session(history)`) | 1 (session_manager.py:2575) | `session_set_initial_history_threadsafe` |
| `get_history` | 1 (session_manager.py:3026) | `session_get_history_threadsafe` |
| `snapshot_instruction_budget` (replaces `get_session().instruction_budget.snapshot()`) | 2 (core.py:2002, 4452) | `session_snapshot_instruction_budget_threadsafe` |
| **Total** | **15 sites** | |
"""

from __future__ import annotations

import ast
import inspect

import server.core as core_module
import server.session_manager as sm_module


def _module_calls_jaato_method(module, method_name: str) -> list:
    """Walk the module AST and return ``Call`` nodes shaped like
    ``<anything>._jaato.<method_name>(...)``."""
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


def _module_calls_runner_rpc_method(module, method_name: str) -> list:
    """Walk the module AST and return ``Call`` nodes shaped like
    ``<anything>._runner_rpc.<method_name>(...)``."""
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
        if isinstance(inner, ast.Attribute) and inner.attr == "_runner_rpc":
            found.append(n)
    return found


# ----------------------------------------------------------------------
# Deletion pins — no remaining `_jaato.X` calls for migrated methods
# ----------------------------------------------------------------------


def test_no_jaato_get_context_usage_calls_remain() -> None:
    """All 3 `_jaato.get_context_usage()` call sites in core.py
    + 1 in session_manager.py migrated to RPC in §7c step 6.6.4.5b."""
    core_calls = _module_calls_jaato_method(core_module, "get_context_usage")
    sm_calls = _module_calls_jaato_method(sm_module, "get_context_usage")
    assert core_calls == [], (
        f"§7c step 6.6.4.5b regression: ``_jaato.get_context_usage()`` "
        f"call re-introduced in server/core.py ({len(core_calls)} sites). "
        f"Use ``server._runner_rpc.session_get_context_usage_threadsafe()``."
    )
    assert sm_calls == [], (
        f"§7c step 6.6.4.5b regression: ``_jaato.get_context_usage()`` "
        f"call re-introduced in server/session_manager.py ({len(sm_calls)} "
        f"sites)."
    )


def test_no_jaato_get_context_limit_calls_remain() -> None:
    """All 6 `_jaato.get_context_limit()` call sites in core.py
    migrated to RPC in §7c step 6.6.4.5b."""
    core_calls = _module_calls_jaato_method(core_module, "get_context_limit")
    sm_calls = _module_calls_jaato_method(sm_module, "get_context_limit")
    assert core_calls == [], (
        f"§7c step 6.6.4.5b regression: ``_jaato.get_context_limit()`` "
        f"call re-introduced in server/core.py ({len(core_calls)} sites). "
        f"Use ``server._runner_rpc.session_get_context_limit_threadsafe()``."
    )
    assert sm_calls == [], (
        f"§7c step 6.6.4.5b regression: ``_jaato.get_context_limit()`` "
        f"call re-introduced in server/session_manager.py."
    )


def test_no_jaato_get_turn_accounting_calls_remain() -> None:
    """The 1 `_jaato.get_turn_accounting()` call site (core.py:3518)
    migrated to RPC in §7c step 6.6.4.5b."""
    core_calls = _module_calls_jaato_method(core_module, "get_turn_accounting")
    assert core_calls == [], (
        f"§7c step 6.6.4.5b regression: ``_jaato.get_turn_accounting()`` "
        f"call re-introduced in server/core.py ({len(core_calls)} sites)."
    )


def test_no_jaato_reset_session_calls_remain() -> None:
    """The 1 `_jaato.reset_session(history)` call site
    (session_manager.py:2575) migrated to
    `session_set_initial_history_threadsafe(history)` in §7c step
    6.6.4.5b — semantically equivalent at this site (session was
    just bootstrapped with empty history)."""
    sm_calls = _module_calls_jaato_method(sm_module, "reset_session")
    assert sm_calls == [], (
        f"§7c step 6.6.4.5b regression: ``_jaato.reset_session()`` "
        f"call re-introduced in server/session_manager.py "
        f"({len(sm_calls)} sites). Use "
        f"``session_set_initial_history_threadsafe`` instead — semantically "
        f"equivalent at the restore-from-disk site."
    )


def test_no_jaato_get_history_calls_remain() -> None:
    """The 1 `_jaato.get_history()` call site (session_manager.py:3026)
    migrated to RPC in §7c step 6.6.4.5b."""
    sm_calls = _module_calls_jaato_method(sm_module, "get_history")
    assert sm_calls == [], (
        f"§7c step 6.6.4.5b regression: ``_jaato.get_history()`` "
        f"call re-introduced in server/session_manager.py "
        f"({len(sm_calls)} sites)."
    )


def test_no_jaato_get_session_calls_for_budget_snapshot() -> None:
    """The 2 `_jaato.get_session()` reads at
    initialize+auth-completion sites that read
    `session.instruction_budget.snapshot()` migrated to the
    `session.snapshot_instruction_budget` RPC in §7c step 6.6.4.5b.

    Note: the `core.py:767` site
    (``set_reference_authorizer`` daemon-side leg) is a deferred
    split-out per the §7c step 6.6.4 audit — kept until the
    references-plugin runner-side migration sub-track lands.  This
    pin asserts ≤1 remaining call site (the deferred references
    one), not zero."""
    calls = _module_calls_jaato_method(core_module, "get_session")
    # Only the references-plugin deferred site at core.py:767 should remain.
    assert len(calls) <= 1, (
        f"§7c step 6.6.4.5b regression: more than 1 `_jaato.get_session()` "
        f"call site remains in server/core.py ({len(calls)} sites). "
        f"Only the references-plugin deferred site should remain; the "
        f"initialize+auth-completion budget-snapshot reads should use "
        f"``session_snapshot_instruction_budget_threadsafe`` instead."
    )


# ----------------------------------------------------------------------
# Migration pattern presence pins
# ----------------------------------------------------------------------


def test_runner_rpc_get_context_usage_calls_present() -> None:
    """Pin the migrated sites use the RPC threadsafe wrapper."""
    core_calls = _module_calls_runner_rpc_method(
        core_module, "session_get_context_usage_threadsafe",
    )
    sm_calls = _module_calls_runner_rpc_method(
        sm_module, "session_get_context_usage_threadsafe",
    )
    # 3 in core.py (2052, 3649, 4486) + 1 in session_manager.py (2626).
    assert len(core_calls) >= 3, (
        f"Expected ≥3 ``session_get_context_usage_threadsafe`` calls in "
        f"server/core.py; found {len(core_calls)}.  Migration sites: "
        f"initialize() context-update, model-thread post-loop, "
        f"auth-completion context-update."
    )
    assert len(sm_calls) >= 1, (
        f"Expected ≥1 ``session_get_context_usage_threadsafe`` call in "
        f"server/session_manager.py; found {len(sm_calls)}."
    )


def test_runner_rpc_get_context_limit_calls_present() -> None:
    core_calls = _module_calls_runner_rpc_method(
        core_module, "session_get_context_limit_threadsafe",
    )
    # Path E removed the in-band demuxer-branch call (race fix).  The two
    # aspect-callback fallbacks (on_agent_context_updated + on_turn_progress)
    # that this pin used to count were removed too: those callbacks run on
    # the RPC read loop's thread, where a ``*_threadsafe`` round-trip
    # SELF-DEADLOCKS -- the loop blocks on a coroutine only it could run,
    # broken only by the 10s timeout, whose exception path kept the cache
    # unhealed so the miss repeated on every streaming notification.
    # Diagnosed by #631's watchdog (seven stalls, seven identical stacks,
    # deepest frame this very method).  They now emit with the limit unknown
    # and schedule a NON-BLOCKING off-band fill instead.
    # Remaining sites: initialize() x1, model-thread post-loop x1,
    # auth-completion x1.
    assert len(core_calls) >= 3, (
        f"Expected >=3 ``session_get_context_limit_threadsafe`` calls in "
        f"server/core.py; found {len(core_calls)}.  If this dropped below 3, "
        f"a legitimate off-thread site was removed; if new sites appeared, "
        f"make sure none of them is reachable from the notification path."
    )
    assert len(core_calls) <= 4, (
        f"{len(core_calls)} call sites -- a NEW site was added.  Verify it "
        f"cannot run on the RPC read loop's thread before raising this "
        f"bound: that exact mistake was a 10s loop stall per streaming "
        f"notification, forever."
    )


def test_runner_rpc_set_initial_history_call_present() -> None:
    """The session_manager.py:2575 site migrated to use
    ``session_set_initial_history_threadsafe`` (semantically
    equivalent to the pre-§7c ``reset_session(history)`` at this
    code path)."""
    sm_calls = _module_calls_runner_rpc_method(
        sm_module, "session_set_initial_history_threadsafe",
    )
    assert len(sm_calls) >= 1, (
        f"Expected ≥1 ``session_set_initial_history_threadsafe`` call "
        f"in server/session_manager.py; found {len(sm_calls)}."
    )


def test_runner_rpc_snapshot_instruction_budget_calls_present() -> None:
    """The 2 budget-snapshot sites in core.py (initialize +
    auth-completion mirror) migrated to use the
    ``session_snapshot_instruction_budget_threadsafe`` RPC."""
    core_calls = _module_calls_runner_rpc_method(
        core_module, "session_snapshot_instruction_budget_threadsafe",
    )
    assert len(core_calls) >= 2, (
        f"Expected ≥2 ``session_snapshot_instruction_budget_threadsafe`` "
        f"calls in server/core.py; found {len(core_calls)}."
    )


def test_runner_rpc_get_history_call_present() -> None:
    """session_manager.py:3026 migrated to use the
    ``session_get_history_threadsafe`` RPC (captures in-progress
    turns; was previously
    ``session.server._jaato.get_history()``)."""
    sm_calls = _module_calls_runner_rpc_method(
        sm_module, "session_get_history_threadsafe",
    )
    assert len(sm_calls) >= 1, (
        f"Expected ≥1 ``session_get_history_threadsafe`` call in "
        f"server/session_manager.py; found {len(sm_calls)}."
    )


# ----------------------------------------------------------------------
# Deferred-site presence pins (the user-command/auth_info reads
# stay until 5c, and the construction-site populator stays until 5d)
# ----------------------------------------------------------------------


def test_all_5c_session_tier_reads_migrated() -> None:
    """Pin all 5c sub-commits closed the original audit's set of
    session-tier `_jaato.X` reads:

    - 5c.1 ✅ `_jaato.auth_info` (commit bff782a4)
    - 5c.2 ✅ `_jaato.get_user_commands()` (commit 592cbf94)
    - 5c.3 ✅ `_jaato.execute_user_command()` (commit 175e9220)
    - 5c.4 ✅ `_jaato.get_model_completions()` (commit 68572a92)
    - 5c.5 ✅ `_jaato.get_tool_schemas()` (this commit)

    The remaining `_jaato.X` calls in core.py are construction-site
    + truthiness-check sites handled by 5d/5e.
    """
    for method_name in (
        "get_user_commands",
        "execute_user_command",
        "get_model_completions",
        "get_tool_schemas",
    ):
        calls = _module_calls_jaato_method(core_module, method_name)
        assert calls == [], (
            f"§7c step 6.6.4.5c regression: ``_jaato.{method_name}()`` "
            f"call re-introduced in server/core.py ({len(calls)} sites). "
            f"All 5c sub-commits should have migrated these to "
            f"``self._runner_rpc.session_*_threadsafe()`` RPCs."
        )
