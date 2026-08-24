"""The budget CEILING must survive a suspend/resume, not just the spend.

Root cause (live evidence 2026-08-23): a budget reaches the runner ONLY as
``profile.budget_control``.  A budget declared outside the profile --
``cascade_budget_set`` on a driver, where limits are a per-run operator choice
-- had no vehicle across a reload, so the revived session came back with NO
BudgetTracker and a ``turns: 2`` ceiling let a goal run three turns and exit 0.

Second defect, worse: with no tracker, ``get_budget_usage`` took its unbudgeted
``{"tokens": N}`` fallback, and the next save wrote that OVER the real
five-dimension snapshot.  The ceiling stopped being restorable at all.  A
session file holding ``{"tokens": 247004.0}`` where five dimensions had been in
flight is that fallback's fingerprint.
"""
from types import SimpleNamespace

import pytest

from server.session_manager import SessionManager
from shared.jaato_session import JaatoSession
from shared.budget_control import BudgetControlConfig
from shared.plugins.session.base import SessionState
from datetime import datetime, timezone
from shared.plugins.session import serializer as S


def _state(**kw):
    now = datetime.now(timezone.utc)
    return SessionState(session_id="s1", history=[], created_at=now,
                        updated_at=now, **kw)


CFG = {"limits": {"turns": 2.0, "usd": 1.0},
       "degrade": [{"at": 100, "action": "abort"}]}


# ---------------------------------------------------------------- B: the read

def _usage(tracker, **kw):
    """Call the REAL method against a minimal self (it touches two fields)."""
    return JaatoSession.get_budget_usage(
        SimpleNamespace(_budget_tracker=tracker,
                        _turn_accounting=[{"spend_total": 247004}]), **kw)


def test_persistence_read_never_returns_the_fallback_shape():
    # No tracker: tracker_only must yield NOTHING rather than a shape that
    # would overwrite a real snapshot with one synthetic key.
    assert _usage(None, tracker_only=True) == {}


def test_fallback_still_available_for_pool_reconciliation():
    # An unbudgeted child still spends real tokens against a shared pot.
    assert _usage(None) == {"tokens": 247004.0}


def test_tracker_usage_is_returned_verbatim_either_way():
    tracker = SimpleNamespace(
        usage=SimpleNamespace(as_dict=lambda: {"turns": 2.0, "usd": 0.14}))
    assert _usage(tracker, tracker_only=True) == {"turns": 2.0, "usd": 0.14}
    assert _usage(tracker) == {"turns": 2.0, "usd": 0.14}


# ------------------------------------------------- 1: the ceiling round-trips

def test_ceiling_round_trips_through_disk():
    state = _state(budget_control=CFG)
    back = S.deserialize_session_state(S.serialize_session_state(state))
    assert back.budget_control == CFG, (
        "the ceiling did not survive the serializer -- it writes a FIXED key "
        "list, so a field absent there never reaches disk")


def test_usage_round_trips_with_the_same_keys():
    # The peer's framing: present-and-non-empty is not the invariant.  A dict
    # that survives with one of five entries passes that and still cannot
    # satisfy a turns ceiling.
    usage = {"turns": 2.0, "usd": 0.14, "tokens": 66182.0,
             "seconds": 11.26, "tool_calls": 3.0}
    back = S.deserialize_session_state(
        S.serialize_session_state(_state(budget_usage=usage)))
    assert back.budget_usage == usage
    assert set(back.budget_usage) == set(usage), "keys were dropped in transit"


# ------------------------------------------------- 1: the ceiling re-attaches

def _reattach(persisted, profile_budget):
    profile = SimpleNamespace(budget_control=profile_budget, name="goal-actor")
    applied = SessionManager._attach_budget_ceiling(
        SimpleNamespace(), persisted, profile, "s1")
    return applied, profile


def test_reattaches_when_the_profile_declares_none():
    applied, profile = _reattach(CFG, None)
    assert applied is True
    assert profile.budget_control is not None
    assert profile.budget_control.limits == {"turns": 2.0, "usd": 1.0}, (
        "without this the revived session has no BudgetTracker and no "
        "cross-turn ceiling can fire")


def test_authored_profile_policy_is_not_shadowed_by_a_stale_snapshot():
    declared = BudgetControlConfig.from_dict({"limits": {"turns": 99.0}})
    applied, profile = _reattach(CFG, declared)
    assert applied is False
    assert profile.budget_control is declared


def test_no_snapshot_is_a_no_op():
    applied, profile = _reattach(None, None)
    assert applied is False
    assert profile.budget_control is None


def test_malformed_snapshot_warns_rather_than_raising(caplog):
    with caplog.at_level("WARNING"):
        applied, profile = _reattach({"limits": {"nonsense": "x"}}, None)
    assert applied is False
    assert any("UNBUDGETED" in r.getMessage() for r in caplog.records), \
        "a ceiling that fails to rebuild must be loud, not silent"
