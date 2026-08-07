"""SessionManager cascade-budget registry + accumulation (slice 2, §8/b).

The cap is declared on the cascade OWNER, never on a leaf profile (§3.1):
a profile is a reusable template, a cascade cap is a runtime aggregate
over one live cid.
"""

from types import SimpleNamespace

import pytest

from shared.budget_control import BudgetControlConfig
from server.session_manager import SessionManager


class _SM:
    """Minimal harness: the registry methods are self-contained, so bind
    them to a namespace rather than constructing a whole SessionManager."""

    def __init__(self):
        import threading
        self._cascade_budgets = {}
        self._cascade_budgets_lock = threading.Lock()

    def __getattr__(self, name):
        fn = getattr(SessionManager, name)
        return lambda *a, **k: fn(self, *a, **k)


def _cfg(**limits):
    return BudgetControlConfig.from_dict({"limits": limits})


def test_declare_and_read_back():
    sm = _SM()
    assert sm.get_cascade_budget("cid1") is None
    sm.set_cascade_budget("cid1", _cfg(tokens=12000))
    pool = sm.get_cascade_budget("cid1")
    assert pool is not None and pool.remaining()["tokens"] == 12000.0


def test_absent_cid_and_none_are_uncapped():
    sm = _SM()
    sm.set_cascade_budget("cid1", _cfg(tokens=100))
    assert sm.get_cascade_budget("other") is None
    assert sm.get_cascade_budget(None) is None


def test_clear_drops_the_pool():
    sm = _SM()
    sm.set_cascade_budget("cid1", _cfg(tokens=100))
    sm.clear_cascade_budget("cid1")
    assert sm.get_cascade_budget("cid1") is None


def _turn_event(spend, total, duration=1.0, calls=1):
    from jaato_sdk.events import TurnCompletedEvent, UsageBreakdown
    return TurnCompletedEvent(
        agent_id="main", turn_number=0,
        usage=UsageBreakdown(prompt_tokens=0, output_tokens=0,
                             total_tokens=total, spend_total_tokens=spend),
        duration_seconds=duration,
        function_calls=[{"name": "t"}] * calls,
    )


def test_accumulation_uses_SPEND_not_total():
    """total_tokens is end-of-turn context size and undercounts by ~41% on
    tool-calling turns. The pool and the per-session tracker must count the
    same thing or min(profile, remaining) composes different scales."""
    sm = _SM()
    sm.set_cascade_budget("cid1", _cfg(tokens=10000))
    session = SimpleNamespace(cascade_driver_id="cid1")
    sm._accumulate_cascade_budget(session, _turn_event(spend=4654, total=2504))
    assert sm.get_cascade_budget("cid1").remaining()["tokens"] == 10000 - 4654


def test_accumulation_ignores_sessions_outside_the_cascade():
    sm = _SM()
    sm.set_cascade_budget("cid1", _cfg(tokens=10000))
    sm._accumulate_cascade_budget(
        SimpleNamespace(cascade_driver_id=None), _turn_event(4654, 2504))
    assert sm.get_cascade_budget("cid1").remaining()["tokens"] == 10000.0


def test_accumulation_is_a_noop_for_non_turn_events():
    from jaato_sdk.events import AgentOutputEvent
    sm = _SM()
    sm.set_cascade_budget("cid1", _cfg(tokens=10000))
    sm._accumulate_cascade_budget(
        SimpleNamespace(cascade_driver_id="cid1"),
        AgentOutputEvent(agent_id="main", source="model", text="x", mode="write"))
    assert sm.get_cascade_budget("cid1").remaining()["tokens"] == 10000.0


def test_accumulation_never_raises_on_the_emit_path():
    """It runs inside _emit_to_session; a budget failure must not break
    event delivery."""
    from jaato_sdk.events import TurnCompletedEvent, UsageBreakdown

    class _Exploding(UsageBreakdown):
        @property
        def spend_total_tokens(self):        # type: ignore[override]
            raise RuntimeError("boom")

    sm = _SM()
    sm.set_cascade_budget("cid1", _cfg(tokens=100))
    evt = TurnCompletedEvent(agent_id="main", turn_number=0)
    object.__setattr__(evt, "usage", _Exploding())
    # must not propagate — this runs inside _emit_to_session
    sm._accumulate_cascade_budget(SimpleNamespace(cascade_driver_id="cid1"), evt)
    assert sm.get_cascade_budget("cid1") is not None


def test_depletion_across_stages_is_cumulative():
    """The linear-chain premise: stage 3 sees what stages 1-2 left."""
    sm = _SM()
    sm.set_cascade_budget("cid1", _cfg(tokens=12000))
    session = SimpleNamespace(cascade_driver_id="cid1")
    sm._accumulate_cascade_budget(session, _turn_event(spend=9300, total=3000))
    pool = sm.get_cascade_budget("cid1")
    assert pool.remaining()["tokens"] == 2700.0
    cfg, eff = pool.child_config(_cfg(tokens=9000))
    assert cfg.limits["tokens"] == 2700.0     # stage 2 clamped
    assert eff.clamped == ("tokens",)
