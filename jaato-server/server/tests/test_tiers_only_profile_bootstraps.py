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

class TestGateAcceptsEveryDocumentedTierShape:
    """The gate must accept whatever the RUNTIME accepts.

    PR #574 added the gate but read the tier grammar itself, handling only the
    mapping form.  The documented shorthand -- a bare model string, which
    `jaato-scaffold explain tiers` lists FIRST --

        <tier>:   <model-string>   OR   {model: <m>, provider: <p>}

    was therefore rejected, so shorthand + no top-level `model` became the one
    shape that could not boot: exactly the defect #574 existed to close, one
    form down.  It survived review because every in-tree profile uses the
    mapping form, so no existing exemplar exercised it.

    These tests compare the gate against ``_normalize_tier_entry`` -- the
    single definition of the grammar -- rather than against a list of forms
    someone remembered.
    """

    @pytest.mark.parametrize("entry", [
        "vendor/model",                              # documented shorthand
        "  vendor/model  ",                          # ...tolerant of padding
        {"model": "vendor/model"},                   # mapping
        {"model": "vendor/model", "provider": "openrouter"},   # V2 cross-provider
    ])
    def test_every_shape_the_normalizer_accepts_also_binds(self, entry):
        from shared.model_tiers import _normalize_tier_entry
        # precondition: the runtime really does accept this shape
        assert _normalize_tier_entry("planner", entry).model

        assert _profile_binds_a_model(
            _profile(model_tiers={"planner": entry, "initial": "planner"})
        ) is True, (
            "the gate rejects a tier shape the runtime accepts -- it is "
            "re-reading the grammar instead of delegating to it"
        )

    @pytest.mark.parametrize("entry", ["", "   ", {}, {"provider": "openrouter"}])
    def test_shapes_the_normalizer_rejects_do_not_bind(self, entry):
        from shared.model_tiers import ModelTierConfigError, _normalize_tier_entry
        with pytest.raises(ModelTierConfigError):
            _normalize_tier_entry("planner", entry)

        assert _profile_binds_a_model(
            _profile(model_tiers={"planner": entry, "initial": "planner"})
        ) is False


class TestGateAndEnvelopeAgree:
    """The gate and the spawn envelope must mean the same thing by "bound".

    PR #574 opened the bootstrap gate for tiers-only profiles but left
    ``runner_spawn`` sourcing ``envelope.model_name`` from ``profile.model``
    alone.  The failure moved one layer down and got WORSE: the gate passed,
    the runner rejected the envelope with "envelope.model_name is empty", and
    the caller saw a dropped IPC connection plus "session not bootstrapped on
    this runner" -- where before #574 they had got an immediate, accurate
    ConfigurationError naming the fix.

    Both now call ``bound_model_for_profile``.  These tests fail if they ever
    diverge again, which is the defect -- not either value on its own.
    """

    @staticmethod
    def _envelope_model(profile):
        from unittest.mock import MagicMock
        from server.runner_spawn import build_session_envelope
        server = MagicMock()
        server._profile = profile
        server._session_env = {}          # no MODEL_NAME: profile must suffice
        server._cascade_budget_pool = None
        server._suppress_base_instructions = frozenset()
        return build_session_envelope(
            server=server, session_id="s1",
            workspace_path="/tmp/ws", profile_name="t",
        ).model_name

    @pytest.mark.parametrize("tiers,expected", [
        ({"planner": {"model": "vendor/mapping"}, "initial": "planner"},
         "vendor/mapping"),
        ({"planner": "vendor/shorthand", "initial": "planner"},
         "vendor/shorthand"),
    ])
    def test_gate_open_implies_envelope_carries_that_model(self, tiers, expected):
        profile = _profile(provider="openrouter", model_tiers=tiers)

        assert _profile_binds_a_model(profile) is True
        assert self._envelope_model(profile) == expected, (
            "the gate admitted this profile but the envelope carries a "
            "different model -- the runner will refuse it and the caller sees "
            "a dropped connection instead of a config error"
        )

    def test_flat_model_still_reaches_the_envelope(self):
        profile = _profile(provider="openrouter", model="vendor/flat")
        assert _profile_binds_a_model(profile) is True
        assert self._envelope_model(profile) == "vendor/flat"

    def test_gate_closed_and_envelope_empty_agree(self):
        """When nothing is bound, BOTH must say so -- no silent guess."""
        profile = _profile(provider="openrouter")
        assert _profile_binds_a_model(profile) is False
        assert self._envelope_model(profile) == ""
