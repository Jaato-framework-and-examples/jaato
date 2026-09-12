"""The orphan bound may only judge a session it has SEEN attached (#812).

An earlier draft of the bound judged a session orphaned on one condition —
``attached_clients`` is empty — and documented an invariant to justify it:
*every path that drives a loaded session attaches a client id*.

**That invariant is false in this tree**, which is why this guard does not
police it.  ``SessionManager._load_session_impl`` takes a ``client_id`` and
uses it for config, env and progress events; it never attaches it.  So a
session revived through ``wake_session`` → ``resume_session`` has an empty
``attached_clients`` by construction, and ``wake_session`` branches on exactly
that ("revived cold, no client — DEFERRED").  An AST guard asserting the
invariant would have failed on ``main``; a guard exempting the revive path
would assert almost nothing.

So the dependency was removed instead of policed: the sweep requires an
OBSERVED attachment before a session becomes eligible, which is a fact it
measures rather than an invariant maintained at call sites it cannot see.
What this module guards is that the removal stays removed.

The failure mode being guarded is silent and destructive in the same
direction as the bug #812 fixed: a drive path that does not attach would not
raise — its session would simply be cancelled 900 s later, which reads to a
reporter as exactly the runaway-session symptom the issue was filed about.

Precedents for the shape: ``test_budget_mid_turn_955.py`` (every path that
records a tool call observes it) and ``test_registry_iteration_snapshots.py``
(every read path iterates a snapshot).
"""

from __future__ import annotations

import ast
import pathlib
import threading
from datetime import datetime
from unittest.mock import MagicMock

from server.session_manager import Session, SessionManager
from shared.runtime_limits import RuntimeLimits

_MANAGER_PY = pathlib.Path(__file__).resolve().parents[2] / (
    "server/session_manager.py")


# ----------------------------------------------------------------- AST guard

def _functions_writing(attr: str) -> set:
    """Names of methods that write to ``self.<attr>``.

    A "write" is a ``setdefault`` / ``add`` / ``pop`` / subscript-assignment
    against the attribute — i.e. anything that decides membership.
    """
    tree = ast.parse(_MANAGER_PY.read_text(encoding="utf-8"))
    writers = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            # self.<attr>.setdefault(...) / .add(...) / .pop(...)
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Attribute)
                    and inner.func.value.attr == attr
                    and isinstance(inner.func.value.value, ast.Name)
                    and inner.func.value.value.id == "self"
                    and inner.func.attr in {"setdefault", "add", "pop",
                                            "discard", "update", "clear"}):
                writers.add(node.name)
            # self.<attr>[...] = ...
            if isinstance(inner, ast.Assign):
                for tgt in inner.targets:
                    if (isinstance(tgt, ast.Subscript)
                            and isinstance(tgt.value, ast.Attribute)
                            and tgt.value.attr == attr):
                        writers.add(node.name)
    return writers


def test_the_orphan_clock_is_written_only_by_the_observer():
    """``_orphan_since`` decides what the bound may stop, so exactly one
    place may decide membership in it.

    A second writer is how "empty attached_clients ⇒ orphan" comes back: it
    would set the clock without consulting ``_ever_attached``, and the
    eligibility gate would be bypassed rather than removed — silently.
    ``sweep_session_lifetimes`` is allowed because it only ever *forgets* an
    entry after acting on it.
    """
    writers = _functions_writing("_orphan_since")
    assert writers, (
        "no method writes _orphan_since — did the sweep's state change name? "
        "This guard asserts nothing if it cannot find the attribute."
    )
    allowed = {"_observe_session_lifetimes", "sweep_session_lifetimes"}
    assert writers <= allowed, (
        f"{sorted(writers - allowed)} write(s) SessionManager._orphan_since. "
        "Only _observe_session_lifetimes may decide that a session is "
        "orphaned, because it is the one place that first checks "
        "_ever_attached. A second writer reintroduces 'no attached clients "
        "⇒ orphan', which cancels cold wake revives (#812)."
    )


def _starts_the_clock(node: ast.AST) -> bool:
    """True if ``node``'s subtree starts the orphan clock."""
    return any(
        isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "setdefault"
        and isinstance(c.func.value, ast.Attribute)
        and c.func.value.attr == "_orphan_since"
        for c in ast.walk(node)
    )


def _tests_eligibility(node: ast.If) -> bool:
    """True if ``node``'s CONDITION consults ``_ever_attached``."""
    return any(
        isinstance(a, ast.Attribute) and a.attr == "_ever_attached"
        for a in ast.walk(node.test)
    )


def test_the_clock_starts_only_inside_the_eligibility_test():
    """The gate must GUARD the clock, not merely be mentioned nearby.

    An earlier version of this guard asserted only that
    ``_observe_session_lifetimes`` references ``_ever_attached`` somewhere.
    That has no teeth: the method also *populates* the set in its ``else``
    branch, so deleting the ``if session_id in self._ever_attached:`` check
    left the reference intact and the guard green — verified by running the
    predicate against a modified copy of the source.  This asserts the
    containment instead, which is the property that actually holds the
    behaviour.
    """
    tree = ast.parse(_MANAGER_PY.read_text(encoding="utf-8"))
    observer = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "_observe_session_lifetimes"),
        None,
    )
    assert observer is not None, "_observe_session_lifetimes is gone"

    starters = [n for n in ast.walk(observer) if isinstance(n, ast.If)
                and _starts_the_clock(n)]
    assert starters, (
        "_observe_session_lifetimes no longer starts the orphan clock inside "
        "any `if` — did the sweep change shape? This guard asserts nothing "
        "if it cannot find the statement."
    )
    assert any(_tests_eligibility(n) for n in starters), (
        "the orphan clock is started without an `if` that consults "
        "_ever_attached — the bound would again apply to a session that "
        "never had a client, i.e. it would cancel a cold wake revive mid-turn "
        "(#812). _load_session_impl does NOT attach its client_id, so 'no "
        "attached clients' does not mean 'its client went away'."
    )


# ------------------------------------------------------------ the behaviour

def _sm() -> SessionManager:
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._orphan_since = {}
    sm._ever_attached = set()
    sm._lifetime_watchdog = None
    sm._lifetime_watchdog_stop = threading.Event()
    sm._lifetime_sweep_interval = 15.0
    sm._session_index = MagicMock()
    sm._emit_to_session = MagicMock()
    return sm


def _session(sid: str, clients=(), limits=None) -> Session:
    server = MagicMock()
    server._model_running = True
    server._main_agent_id = "main"
    server.stop.return_value = True
    server._profile = MagicMock()
    server._profile.runtime_limits = limits
    s = Session(session_id=sid, name=sid, server=server,
                created_at=datetime.now().isoformat(), loaded_at=0.0)
    s.attached_clients = set(clients)
    return s


def test_a_cold_revive_that_never_had_a_client_is_not_stopped():
    """``wake_session`` → ``resume_session`` → ``_load_session`` leaves
    ``attached_clients`` EMPTY; the turn it drives may be long."""
    sm = _sm()
    sm._sessions["woken"] = _session(
        "woken", clients=(), limits=RuntimeLimits(max_orphan_seconds=10))
    sm.sweep_session_lifetimes(now=0.0)
    assert sm.sweep_session_lifetimes(now=10_000.0) == []
    sm._sessions["woken"].server.stop.assert_not_called()


def test_a_session_whose_client_went_away_is_stopped():
    """The #812 case itself: a real client, attached, then gone."""
    sm = _sm()
    session = _session(
        "arm", clients=("client_ipc_14",),
        limits=RuntimeLimits(max_orphan_seconds=10))
    sm._sessions["arm"] = session

    sm.sweep_session_lifetimes(now=0.0)        # observed WITH its client
    session.attached_clients.clear()           # the client process dies
    sm.sweep_session_lifetimes(now=1.0)        # orphan clock starts

    verdicts = sm.sweep_session_lifetimes(now=100.0)
    assert [v.session_id for v in verdicts] == ["arm"]
    session.server.stop.assert_called_once()


def test_an_explicit_total_ceiling_still_binds_a_never_attached_session():
    """The eligibility gate narrows the ORPHAN bound only.
    ``max_session_seconds`` is an operator's own ceiling, not an inference
    about who is watching, so it applies regardless."""
    sm = _sm()
    sm._sessions["woken"] = _session(
        "woken", clients=(), limits=RuntimeLimits(max_session_seconds=10))
    sm.sweep_session_lifetimes(now=0.0)
    verdicts = sm.sweep_session_lifetimes(now=100.0)
    assert [v.reason for v in verdicts] == ["max_session_seconds"]


def test_the_listing_says_whether_the_bound_applies():
    """An operator must be able to tell "no client, and nothing will stop it"
    from "no client, and it is on the clock"."""
    sm = _sm()
    sm._sessions["woken"] = _session("woken", clients=())
    sm._sessions["arm"] = _session("arm", clients=("c1",))
    sm.sweep_session_lifetimes(now=0.0)
    sm._sessions["arm"].attached_clients.clear()
    sm.sweep_session_lifetimes(now=1.0)

    rows = {r["session_id"]: r for r in sm.list_orphan_sessions()}
    assert rows["woken"]["orphan_bound_applies"] is False
    assert rows["arm"]["orphan_bound_applies"] is True


def test_eligibility_does_not_grow_without_bound():
    """``_ever_attached`` is pruned to live sessions on every sweep."""
    sm = _sm()
    sm._sessions["a"] = _session("a", clients=("c1",))
    sm.sweep_session_lifetimes(now=0.0)
    assert sm._ever_attached == {"a"}
    del sm._sessions["a"]
    sm.sweep_session_lifetimes(now=1.0)
    assert sm._ever_attached == set()
