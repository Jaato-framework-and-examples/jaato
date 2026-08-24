"""A caller can hand a new session its ceiling AND its spent meter.

A budget reaches a runner ONLY via ``profile.budget_control`` (#583).  So a
caller whose ceiling is not expressible in a profile had no route: notably a
FORK, which continues work from an earlier point and must inherit the source's
ceiling, while ``inline_profile_data`` is a whole-profile replacement rather
than an overlay.

Without the meter, a fork starts at ZERO against a full ceiling -- so N
branches from an exhausted source each run the budget again and branching
becomes a way out of the ceiling.  Reproduced against the real
``build_session_envelope`` before this existed.

Mirrors the reload split deliberately:
    CEILING  pre-spawn  (attached to the profile the envelope reads)
    USAGE    post-spawn (the same restore RPC a reload uses)
"""
from types import SimpleNamespace

import pytest

from server.session_manager import SessionManager
from shared.budget_control import BudgetControlConfig

CFG = {"limits": {"turns": 2.0}, "degrade": [{"at": 100, "action": "abort"}]}


def _attach(supplied, profile_budget):
    profile = SimpleNamespace(budget_control=profile_budget, name="fork")
    applied = SessionManager._attach_budget_ceiling(
        SimpleNamespace(), supplied, profile, "fork-1")
    return applied, profile


def test_a_caller_supplied_ceiling_reaches_the_profile():
    applied, profile = _attach(CFG, None)
    assert applied is True
    assert profile.budget_control.limits == {"turns": 2.0}, (
        "without this the fork spawns with no BudgetTracker and no ceiling "
        "can fire, however carefully the caller computed one")


def test_an_authored_profile_budget_still_wins():
    declared = BudgetControlConfig.from_dict({"limits": {"turns": 99.0}})
    applied, profile = _attach(CFG, declared)
    assert applied is False
    assert profile.budget_control is declared, (
        "this param fills a gap; it must not override authored policy")


def test_no_ceiling_supplied_is_a_no_op():
    applied, profile = _attach(None, None)
    assert applied is False and profile.budget_control is None


class _RPC:
    def __init__(self): self.calls = []

    def session_restore_budget_usage_threadsafe(self, usage, **kw):
        self.calls.append((usage, kw))
        return True


def test_supplied_usage_is_charged_onto_the_new_session():
    rpc = _RPC()
    applied = SessionManager._restore_budget_usage(
        SimpleNamespace(), SimpleNamespace(_runner_rpc=rpc),
        {"turns": 2.0, "usd": 0.5}, None, "fork-1")
    assert applied is True
    assert rpc.calls[0][0]["turns"] == 2.0, (
        "a fork starting at zero gets a FRESH full ceiling — N branches from "
        "an exhausted source would each run the budget again")


def test_both_entry_points_accept_the_params():
    """The params must exist where a fork actually calls in."""
    import inspect
    for fn in (SessionManager.create_headless_session,
               SessionManager._create_session_impl):
        params = inspect.signature(fn).parameters
        assert "budget_control" in params, f"{fn.__name__} lacks budget_control"
        assert "budget_usage" in params, f"{fn.__name__} lacks budget_usage"


def test_headless_forwards_both_to_the_impl():
    """Accepting the kwarg is not enough — it must be forwarded.

    A signature check passes on an entry point that takes the param and drops
    it, which changes nothing while looking correct.
    """
    seen = {}

    def _fake_create_session(**kw):
        seen.update(kw)
        return "sid"

    mgr = SimpleNamespace(
        create_session=_fake_create_session,
        _HEADLESS_CLIENT_ID="_headless",
    )
    SessionManager.create_headless_session(
        mgr, profile_name="p", budget_control=CFG,
        budget_usage={"turns": 2.0})
    assert seen.get("budget_control") == CFG
    assert seen.get("budget_usage") == {"turns": 2.0}


# --------------------------------------------------- the wiring, and its ORDER

def _impl_calls():
    """Line numbers of the calls that matter inside ``_create_session_impl``."""
    import ast, pathlib
    import server.session_manager as sm
    tree = ast.parse(pathlib.Path(sm.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_create_session_impl")
    found = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = (getattr(node.func, "attr", None)
                    or getattr(node.func, "id", None))
            if name in ("_attach_budget_ceiling", "_restore_budget_usage",
                        "BootstrapEnvelope", "_bootstrap_session"):
                found.setdefault(name, node.lineno)
    return found


def test_create_actually_applies_the_ceiling_and_the_usage():
    """Structural, because ``_create_session_impl`` needs a live daemon.

    The direct-call tests above pass even when nothing invokes these helpers:
    deleting either call site failed ZERO tests until this existed — the same
    hole that let a daemon dispatch branch be removed silently in #587.
    """
    calls = _impl_calls()
    assert "_attach_budget_ceiling" in calls, (
        "_create_session_impl never applies the supplied ceiling — the param "
        "is accepted and dropped, so a fork spawns unbudgeted")
    assert "_restore_budget_usage" in calls, (
        "_create_session_impl never charges the supplied usage — a fork "
        "starts at zero against a full ceiling")


def test_the_ceiling_lands_before_the_envelope_is_built():
    """The no-window property, encoded.

    ``profile.budget_control`` -> the envelope's wire field is the ONLY route a
    budget takes to the runner.  A ceiling attached after the envelope is
    built never reaches the session at all; one applied after the spawn would
    leave an interval in which the session exists unbudgeted.  Ordering is the
    guarantee, so the test is about ordering.
    """
    calls = _impl_calls()
    assert calls["_attach_budget_ceiling"] < calls["BootstrapEnvelope"], (
        "the ceiling is attached AFTER the envelope is built — the profile "
        "the runner receives was already snapshotted without it")


def test_the_usage_lands_after_the_runner_exists():
    """Usage travels by RPC, so it needs a runner to talk to."""
    calls = _impl_calls()
    assert calls["_restore_budget_usage"] > calls["_bootstrap_session"], (
        "usage is charged before the runner is spawned — there is no RPC "
        "endpoint yet, so the restore silently no-ops")
