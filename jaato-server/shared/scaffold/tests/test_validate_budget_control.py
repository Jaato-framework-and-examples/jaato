"""``jaato-scaffold validate`` says when a profile has no ceiling (#947).

``budget_control`` is fully implemented and entirely opt-in, and nothing told
an author their profile was unbounded.  Measured 2026-09-10: a subagent whose
``file_edit`` plugin had failed to initialise retried ``writeNewFile`` 127
times over four and a half minutes at ~57k tokens a request, and had to be
stopped with ``kill -TERM``.  Its parent had ended two minutes earlier —
"running subagents preserved" is correct for a backgrounded child, and is
exactly what leaves an unbudgeted loop with nothing left in the session to
notice it.

Two findings, and the second is the load-bearing one:

1. ``budget_control_absent`` — no block, so every dimension is unbounded.

2. ``budget_limits_without_abort`` — ``limits`` and nothing that enforces
   them.  ``limits`` are OBSERVED, never enforced: the ``degrade`` ladder is
   the only consumer of the usage fraction, so ceilings with no ``abort``
   rung are crossed in silence.  Without this check the first finding is
   actively misleading — an author told "you have no budget" writes
   ``limits: {usd: 5}``, the warning clears, and the loop is still unbounded.

Both are warnings: an unbudgeted profile is a legitimate choice for a
short-lived local agent, and an error would fail every existing workspace at
once.
"""
from types import SimpleNamespace

import pytest

from shared.budget_control import BudgetControlConfig
from shared.scaffold.validate import _check_budget_control


def _check(budget=None, *, default_agent=None, model_tiers=None):
    prof = SimpleNamespace(budget_control=budget, default_agent=default_agent,
                           model_tiers=model_tiers or {})
    out: list = []

    def add(severity, code, message, where=None):
        out.append(SimpleNamespace(severity=severity, code=code,
                                   message=message, where=where))

    _check_budget_control(prof, add)
    return out


def _codes(diags):
    return [d.code for d in diags]


def _bc(limits=None, degrade=()):
    return BudgetControlConfig.from_dict(
        {"limits": limits or {}, "degrade": list(degrade)})


# ------------------------------------------------------- absent (#947 core)

def test_a_profile_with_no_budget_control_is_flagged():
    found = _check(None)
    assert _codes(found) == ["budget_control_absent"]
    assert found[0].severity == "warn"
    assert found[0].where == "budget_control"


def test_the_absent_message_names_every_unbounded_dimension():
    """"Missing budget_control" teaches nothing to an author who has never
    seen the knob; naming the five dimensions is what teaches it."""
    msg = _check(None)[0].message
    for dim in ("usd", "tokens", "seconds", "tool_calls", "turns"):
        assert dim in msg
    # And it must show the shape that actually stops a run, not just limits.
    assert "action: abort" in msg


def test_the_absent_message_is_stronger_for_a_persona_bound_profile():
    """A profile naming its own persona is built to be spawned by name
    (#944), and a spawned subagent outlives the session that would have
    noticed the loop."""
    plain = _check(None)[0].message
    bound = _check(None, default_agent="documentalista")[0].message
    assert "default_agent" not in plain
    assert "outlives the session" in bound
    assert bound.startswith(plain.split(" This profile binds")[0])


def test_default_agent_only_strengthens_and_never_suppresses():
    """Every discovered profile is spawnable by name, so the absence of
    ``default_agent`` proves nothing — it may not gate the finding."""
    assert _codes(_check(None, default_agent=None)) == ["budget_control_absent"]


# ------------------------------------ limits that nothing enforces (#947 §2)

def test_limits_with_no_ladder_at_all_are_flagged():
    found = _check(_bc({"usd": 5.0, "tool_calls": 200}))
    assert _codes(found) == ["budget_limits_without_abort"]
    assert found[0].severity == "warn"
    assert found[0].where == "budget_control.degrade"
    assert "no degrade ladder at all" in found[0].message


def test_a_ladder_that_only_finalizes_is_flagged():
    """``finalize`` injects "wrap up with what you have", which a looping
    model can ignore — the documentalista ignored 35 consecutive failures
    without ever emitting text."""
    found = _check(_bc({"tool_calls": 200},
                       [{"at": 80, "action": "finalize"},
                        {"at": 100, "action": "escalate"}]))
    assert _codes(found) == ["budget_limits_without_abort"]
    assert "finalize/escalate" in found[0].message


def test_the_ladder_is_described_in_its_own_order():
    """Re-alphabetising a finalize-then-escalate ladder into
    "escalate/finalize" describes a ladder the author did not write, and
    reads as a validator bug."""
    msg = _check(_bc({"turns": 10},
                     [{"at": 50, "action": "finalize"},
                      {"at": 90, "action": "escalate"}]))[0].message
    assert "finalize/escalate" in msg
    assert "ends at 90%" in msg


def test_a_brownout_only_ladder_is_flagged_as_never_stopping():
    """Rungs that only rebind tiers dim the lights; none turns them off."""
    found = _check(
        _bc({"usd": 5.0}, [{"at": 90, "model_tiers": {"planner": "cheap"}}]),
        model_tiers={"planner": "expensive"})
    assert "budget_limits_without_abort" in _codes(found)
    assert "only rebinds tiers" in found[0].message


def test_an_abort_rung_is_quiet():
    assert _check(_bc({"tool_calls": 200},
                      [{"at": 80, "action": "finalize"},
                       {"at": 100, "action": "abort"}])) == []


def test_an_abort_below_100_percent_is_quiet():
    """A ceiling that stops early is still a ceiling."""
    assert _check(_bc({"usd": 5.0}, [{"at": 75, "action": "abort"}])) == []


def test_the_two_findings_never_both_fire():
    """They are the same defect at two stages; reporting both would read as
    a contradiction ("you have no budget" / "your budget is weak")."""
    for budget in (None, _bc({"usd": 1.0}),
                   _bc({"usd": 1.0}, [{"at": 100, "action": "abort"}])):
        assert len(_check(budget)) <= 1


# ------------------------------------------- the overlay checks still fire

def test_overlay_checks_survive_the_extraction():
    """``budget_overlay_without_tiers`` moved into a helper called from
    :func:`_check_budget_control`; it must still fire."""
    found = _check(
        _bc({"usd": 5.0}, [{"at": 90, "model_tiers": {"planner": "cheap"}},
                           {"at": 100, "action": "abort"}]))
    assert _codes(found) == ["budget_overlay_without_tiers"]
    assert found[0].severity == "error"


# ------------------------------------------------- the property underneath

@pytest.mark.parametrize("degrade,expected", [
    ((), False),
    (({"at": 100, "action": "finalize"},), False),
    (({"at": 100, "action": "escalate"},), False),
    (({"at": 100, "action": "abort"},), True),
    (({"at": 50, "action": "finalize"}, {"at": 100, "action": "abort"}), True),
])
def test_has_abort_rung(degrade, expected):
    """Only ``abort`` reaches ``request_stop``; the other two are latched for
    a layer above and are advice, not a ceiling."""
    assert _bc({"usd": 1.0}, degrade).has_abort_rung is expected
