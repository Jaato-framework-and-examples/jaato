"""``SessionManager._lock`` must never be held across a call into the loop.

THE DEADLOCK (found 2026-08-28 by #655's all-thread dump, on its first run):

    asyncio_0    save_session:9315      with self._lock:            HOLDS
                 _save_session          session_get_history_threadsafe()
                 runner_rpc_client:194  future.result(timeout=...)  WAITS ON LOOP
    MainThread   _emit_to_session:4351  with self._lock:            WAITS FOR LOCK

A worker holds the manager lock and blocks on a coroutine only the loop can
run; the loop is blocked acquiring that same lock.  Circular wait, broken only
when the RPC timeout fires ~35s later.  Behind them a convoy: a run_message
thread and twelve workspace-monitor threads on the same mutex.

WHY THE EXISTING CONTRACT DID NOT PREVENT IT.  ``*_threadsafe`` is documented
as "call from worker threads", and ``asyncio_0`` IS a worker.  The contract is
satisfied and the daemon still deadlocks, because it never said *and not while
holding a lock the loop takes*.  That missing clause is what this test is.

THE RULE.  ``self._lock`` guards the session DICTS -- membership, iteration,
and the small in-memory mutations.  It is not a save lock and not a shutdown
lock: per-session save serialisation already lives in ``session.save_lock``,
whose own comment says the guard is there "so a tenth caller inherits it
instead of having to know it exists".  Resolve or snapshot under the lock,
then do the loop-bound work outside it.

WHY THIS FOLLOWS THE CALL GRAPH.  The original defect was NOT lexical: the
lock is taken in ``save_session`` and the RPC happens one call deeper, in
``_save_session``.  A scan for ``with self._lock:`` blocks containing a
``*_threadsafe`` call finds a different site and misses this one entirely --
verified when writing this.  So the guard resolves one level of call graph,
which is the depth the real defect used.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Dict, List, Set

MANAGER = (pathlib.Path(__file__).resolve().parents[1]
           / "session_manager.py")

#: A call is loop-bound when completing it requires the daemon loop to run.
#: ``*_threadsafe`` schedules a coroutine and blocks; ``JaatoServer.shutdown``
#: issues ``session_end_threadsafe`` / ``session_shutdown_threadsafe`` and
#: closes the RPC via ``run_coroutine_threadsafe``.
LOOP_BOUND_SUFFIX = "_threadsafe"
LOOP_BOUND_NAMES = {"shutdown"}


def _tree() -> ast.Module:
    return ast.parse(MANAGER.read_text(encoding="utf-8"))


def _is_loop_bound(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and (call.func.attr.endswith(LOOP_BOUND_SUFFIX)
             or call.func.attr in LOOP_BOUND_NAMES)
    )


def _functions(tree: ast.Module) -> List[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _methods_that_reach_the_loop(tree: ast.Module) -> Dict[str, str]:
    """``{method name: the loop-bound call it makes}`` — one level deep.

    Direct callers first, then anything calling one of those.  That is the
    depth the real defect used and no more: a fixed point over the whole file
    would flag ``handle_request`` and every dispatcher above it, which is true
    but useless as a guard.
    """
    direct: Dict[str, str] = {}
    for fn in _functions(tree):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and _is_loop_bound(n):
                direct.setdefault(fn.name, ast.unparse(n.func) + "()")

    indirect: Dict[str, str] = {}
    for fn in _functions(tree):
        if fn.name in direct:
            continue
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in direct):
                indirect.setdefault(
                    fn.name, f"{n.func.attr}() -> {direct[n.func.attr]}")
    return {**direct, **indirect}


def _lock_blocks(fn: ast.FunctionDef) -> List[ast.With]:
    return [
        w for w in ast.walk(fn)
        if isinstance(w, ast.With)
        and any(isinstance(n, ast.Attribute) and n.attr == "_lock"
                for item in w.items for n in ast.walk(item.context_expr))
    ]


def test_the_pieces_this_guard_inspects_still_exist():
    """Anchor. Without it every assertion below passes on an empty match."""
    assert MANAGER.exists(), f"{MANAGER} not found — guard is stale, not passing"
    tree = _tree()

    with_lock = [fn.name for fn in _functions(tree) if _lock_blocks(fn)]
    assert with_lock, (
        "no function takes `self._lock` — either the manager lock was renamed "
        "or this guard no longer recognises it. Re-aim it; do NOT delete it."
    )

    reachers = _methods_that_reach_the_loop(tree)
    assert reachers, (
        "no method makes a loop-bound call, which cannot be true while the "
        "runner RPC exists. The guard is looking for the wrong thing."
    )


def test_no_loop_bound_call_happens_under_the_manager_lock():
    """The clause the ``*_threadsafe`` contract was missing.

    Fails against the pre-fix tree at six sites: ``save_session``,
    ``update_pending_tool_calls``, ``save_all``, ``shutdown`` (twice) and
    ``delete_session``.
    """
    tree = _tree()
    reachers = _methods_that_reach_the_loop(tree)

    offences: List[str] = []
    for fn in _functions(tree):
        for block in _lock_blocks(fn):
            for n in ast.walk(block):
                if not isinstance(n, ast.Call):
                    continue
                if _is_loop_bound(n):
                    offences.append(
                        f"{fn.name}:{n.lineno} calls {ast.unparse(n.func)}() "
                        f"directly, inside `with self._lock` opened at "
                        f"line {block.lineno}"
                    )
                elif (isinstance(n.func, ast.Attribute)
                      and n.func.attr in reachers):
                    offences.append(
                        f"{fn.name}:{n.lineno} calls {n.func.attr}() inside "
                        f"`with self._lock` opened at line {block.lineno}, "
                        f"and that reaches {reachers[n.func.attr]}"
                    )

    assert not offences, (
        "the manager lock is held across a call that needs the daemon loop:\n  "
        + "\n  ".join(sorted(offences))
        + "\n\nA worker holding this lock and waiting for the loop, while the "
          "loop waits for this lock, is a circular wait that only the RPC "
          "timeout breaks.\n\n"
          "`self._lock` guards the session DICTS. Resolve or snapshot under "
          "it, then do the loop-bound work outside it. Per-session save "
          "serialisation is already provided by `session.save_lock`."
    )
