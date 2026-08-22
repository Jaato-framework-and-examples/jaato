"""A tiers-only profile must satisfy the bootstrap gate.

The profile loader warns that ``model`` "will be ignored" when ``model_tiers``
is present -- which is TRUE at the session layer: jaato_session assigns
``tier_config.tiers[initial_tier].model`` over whatever ``model`` held.

But the bootstrap gate consulted ``model`` alone, so acting on that warning
produced a profile the session refused to start with "Missing required
environment variables: JAATO_PROVIDER and MODEL_NAME" -- a value it was about
to discard. Authors had to keep a key the runtime throws away, and the working
profile in a sibling repo does exactly that.

Reported by an SDK-only cascade whose goal actor is tiers-driven.
"""
import pytest

from server.core import _profile_binds_a_model
from shared.plugins.subagent.config import SubagentProfile


def _profile(**kw):
    return SubagentProfile(name="t", description="d", plugins=[], **kw)


def test_tiers_only_profile_binds_a_model():
    """THE regression: no flat `model`, tier table supplies it."""
    assert _profile_binds_a_model(_profile(model_tiers={
        "executor": {"provider": "openrouter", "model": "vendor/model"},
        "initial": "executor",
    })) is True


def test_tiers_without_explicit_initial_uses_the_default_tier():
    """`initial` is optional; the default tier still binds a model."""
    from shared.model_tiers import DEFAULT_INITIAL_TIER
    assert _profile_binds_a_model(_profile(model_tiers={
        DEFAULT_INITIAL_TIER: {"provider": "openrouter", "model": "vendor/model"},
    })) is True


def test_flat_model_still_binds():
    """The pre-existing shape must keep working -- this is a WIDENING."""
    assert _profile_binds_a_model(_profile(model="vendor/model")) is True


def test_both_keys_still_bind():
    """The shape authors were forced into stays valid."""
    assert _profile_binds_a_model(_profile(
        model="vendor/model",
        model_tiers={"executor": {"model": "other/model"}, "initial": "executor"},
    )) is True


@pytest.mark.parametrize("profile", [
    None,
    _profile(),
    _profile(model_tiers={}),
    # a tier table whose initial tier declares no model binds nothing
    _profile(model_tiers={"executor": {"provider": "openrouter"},
                          "initial": "executor"}),
    # ...and neither does one whose initial names a tier that isn't there
    _profile(model_tiers={"executor": {"model": "vendor/model"},
                          "initial": "planner"}),
])
def test_nothing_bound_is_still_rejected(profile):
    """The gate must not become a rubber stamp.

    A malformed tier table returning False here would be WRONG for a
    different reason -- ModelTierConfig.resolve reports those precisely, and
    masking it behind "no model bound" would hurt. These cases bind no model
    by any route, which is the honest rejection.
    """
    assert _profile_binds_a_model(profile) is False
