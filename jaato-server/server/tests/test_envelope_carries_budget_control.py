"""The envelope PRODUCER must populate budget_control — regression for the
"field exists, nothing sets it" class.

Why this test exists: budget_control shipped with the dataclass field,
``to_dict`` and ``from_dict`` all landing, and a round-trip test that
passed — but NOTHING populated the field at spawn.  ``envelope.budget_control``
was therefore always None on the runner path (the LIVE production path), the
runner re-parsed None, no BudgetTracker was ever built, and no rung could
ever fire.  A profile could declare ``limits: {turns: 4}`` and blow through
200% of it silently.

Found by the PoC exerciser, not by the suite: the round-trip test asserted
serialisation, which is not the same claim as "the producer sets it".  These
tests assert the producer.
"""

from types import SimpleNamespace

import pytest

from shared.budget_control import BudgetControlConfig


def _budget():
    return BudgetControlConfig.from_dict({
        "limits": {"turns": 4},
        "degrade": [{"at": 50, "model_tiers": {"planner": "cheap"}},
                    {"at": 100, "action": "abort"}],
    })


def _profile(budget=None, **kw):
    base = dict(
        name="p", description="d", provider="openrouter", model="m",
        plugins=[], preloaded_plugins=set(), plugin_configs={},
        tool_scopes={}, model_tiers={}, gc=None, runtime_limits=None,
        env={}, completion_payload_schema=None, spawn_payload_schema=None,
        completion_processors=[], budget_control=budget, quirks={},
        apparmor=False, apparmor_fragments=None, max_turns=10,
        system_instructions=None, agent_params={},
        suppress_base_instructions=False, config_root=None, inherits=None,
        icon=None, description_for_model=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_session_envelope_populates_budget_control():
    """The runner path. THE regression: this is the producer that was missing."""
    from server.runner_spawn import build_session_envelope
    server = SimpleNamespace(
        _profile=_profile(budget=_budget()), config_root=None,
        _main_agent_id="main", _cascade_driver_id=None,
    )
    env = build_session_envelope(
        server=server, session_id="s1", workspace_path="/tmp/ws",
        profile_name="p",
    )
    assert env.budget_control is not None, (
        "envelope.budget_control is None — the profile declared a budget but "
        "the producer dropped it; the session will run UNBUDGETED"
    )
    assert env.budget_control["limits"] == {"turns": 4}
    assert len(env.budget_control["degrade"]) == 2


def test_build_session_envelope_unbudgeted_profile_stays_none():
    from server.runner_spawn import build_session_envelope
    server = SimpleNamespace(
        _profile=_profile(budget=None), config_root=None,
        _main_agent_id="main", _cascade_driver_id=None,
    )
    env = build_session_envelope(
        server=server, session_id="s2", workspace_path="/tmp/ws",
        profile_name="p")
    assert env.budget_control is None


def test_envelope_survives_the_wire_from_the_producer():
    """Producer -> to_dict -> from_dict -> the runner's re-parse.

    The end-to-end claim: what the producer sets is what the runner can
    actually build a tracker from.
    """
    from server.runner_spawn import build_session_envelope
    from shared.session_envelope import SessionInitEnvelope
    server = SimpleNamespace(
        _profile=_profile(budget=_budget()), config_root=None,
        _main_agent_id="main", _cascade_driver_id=None,
    )
    env = build_session_envelope(
        server=server, session_id="s3", workspace_path="/tmp/ws",
        profile_name="p")
    revived = SessionInitEnvelope.from_dict(env.to_dict())
    rebuilt = BudgetControlConfig.from_dict(revived.budget_control)
    assert rebuilt is not None
    assert rebuilt.limits == {"turns": 4}
    assert rebuilt.degrade[1].action == "abort"


def test_profile_summary_keeps_budget_control():
    """pydantic SILENTLY DROPS unknown kwargs — without a declared field the
    producer's value vanishes with no error at all."""
    from jaato_sdk.events import ProfileSummary
    s = ProfileSummary(name="x", description="d",
                       budget_control={"limits": {"usd": 1}})
    assert s.budget_control == {"limits": {"usd": 1}}
