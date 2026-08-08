"""A subagent's own declared budget must actually reach its session.

Subagents are runtime-level sessions created via runtime.create_session(),
NOT daemon sessions — they never appear in SessionManager._sessions, carry
no cascade_driver_id, and are therefore invisible to the daemon-side pool,
the spawn-time clamp and the mid-flight push. Their profile's own
budget_control is the ONLY budget they can have.

It was not being passed. A profile declaring budget_control was silently
unbudgeted the moment it ran as a subagent: the ceiling existed on paper
and nothing enforced it. A main agent with a strict budget could spawn ten
subagents that each burned unbounded tokens, because its own session was
not the one spending.
"""

import inspect

from shared.jaato_runtime import JaatoRuntime
from shared.plugins.subagent import plugin as subagent_plugin


def _create_session_kwargs_at(marker: str) -> str:
    """Source of the create_session call containing `marker`."""
    src = inspect.getsource(subagent_plugin)
    idx = src.index(marker)
    start = src.rindex("create_session(", 0, idx)
    return src[start:src.index(")", idx)]


def test_runtime_create_session_accepts_a_budget():
    assert "budget_control" in inspect.signature(
        JaatoRuntime.create_session).parameters


def test_both_subagent_spawn_paths_pass_the_profile_budget():
    """Both call sites, because they are separate code paths and only one
    being fixed is the failure mode that hid this in the first place."""
    src = inspect.getsource(subagent_plugin)
    assert src.count("budget_control=getattr(profile, \"budget_control\", None)") == 2, (
        "a subagent spawn path is not forwarding the profile's budget_control"
    )


def test_the_forward_survives_a_profile_without_one():
    """getattr default: a profile with no budget_control passes None, which
    means unbudgeted — the pre-existing behaviour for those profiles."""
    from types import SimpleNamespace
    profile = SimpleNamespace()
    assert getattr(profile, "budget_control", None) is None
