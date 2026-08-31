"""Every producer of a session's model must use the same binder.

A tiers-only profile (``model_tiers`` and no flat ``model``) was rejected at
FOUR independent layers, each found only after the previous was fixed:

  1. the bootstrap gate            -- ConfigurationError            (#574)
  2. runner_spawn's envelope       -- "envelope.model_name is empty" (#575)
  3. core.py's model assignment    -- SessionInfoEvent(model_name=None)
                                      -> pydantic ValidationError
  4. session_manager's SECOND envelope builder, for isolated subagents

Three of the four presented to the client identically -- a dropped IPC
connection and "spawn refused" -- with the real cause only in the daemon log.

Fixing them one at a time is what produced four rounds. These tests pin the
CONTRACT AT EVERY PRODUCER: given a profile that binds a model by any
documented route, each layer must yield that model. A fifth producer added
later without the binder fails here rather than in someone's live run.
"""
from unittest.mock import MagicMock

import pytest

from shared.model_tiers import bound_model_for_profile
from shared.plugins.subagent.config import SubagentProfile

MODEL = "anthropic/claude-sonnet-4.5"

# Every documented way to bind a model, including the string shorthand that
# `jaato-scaffold explain tiers` lists first.
BINDING_PROFILES = {
    "flat model": dict(model=MODEL),
    "tiers, mapping form": dict(model_tiers={
        "planner": {"model": MODEL, "provider": "openrouter"}, "initial": "planner"}),
    "tiers, string shorthand": dict(model_tiers={
        "planner": MODEL, "initial": "planner"}),
}


def _profile(**kw):
    return SubagentProfile(name="goal-actor", description="d", plugins=[],
                           provider="openrouter", **kw)


@pytest.fixture(params=list(BINDING_PROFILES), ids=list(BINDING_PROFILES))
def profile(request):
    return _profile(**BINDING_PROFILES[request.param])


def test_binder_itself_resolves(profile):
    assert bound_model_for_profile(profile) == MODEL


def test_producer_bootstrap_gate_admits(profile):
    from server.core import _profile_binds_a_model
    assert _profile_binds_a_model(profile) is True


def test_producer_daemon_envelope_carries_it(profile):
    from server.runner_spawn import build_session_envelope
    server = MagicMock()
    server._profile = profile
    server._session_env = {}          # no MODEL_NAME: the profile must suffice
    server._cascade_budget_pool = None
    server._suppress_base_instructions = frozenset()
    # A bare MagicMock answers ``config_root`` with an auto-created child
    # whose ``__fspath__`` is a fake RELATIVE path, which the envelope now
    # refuses (#742).  Real servers carry a str or None.
    server.config_root = None
    envelope = build_session_envelope(
        server=server, session_id="s1", workspace_path="/tmp/ws",
        profile_name="goal-actor")
    assert envelope.model_name == MODEL


def test_producer_isolated_subagent_envelope_carries_it(profile):
    """Isolated subagents build their envelope on a DIFFERENT path.

    Asserted by CALLING the builder, not by grepping its source. An earlier
    version of this test looked for the binder's name in the method body and
    was defeated by a partial revert that left the import behind -- checking
    that a symbol is mentioned is not checking that it is used.
    """
    from server.session_manager import SessionManager

    envelope = SessionManager._build_isolated_envelope(
        MagicMock(),                       # self: nothing else is touched
        profile=profile,
        isolated_session_id="iso-1",
        workspace_path="/tmp/ws",
        sub_apparmor_profile="jaato-ws-iso-1",
        agent_params=None,
    )
    assert envelope.model_name == MODEL, (
        "the isolated-subagent envelope carries no model for a tiers-only "
        "profile -- the runner rejects it with 'envelope.model_name is empty'"
    )


def test_producer_server_model_name_assignment_uses_the_binder():
    """core.py must not read profile.model directly for the session model.

    That assignment left JaatoServer.model_name as None, which surfaced as a
    pydantic ValidationError building SessionInfoEvent -- and to the client as
    a dropped connection.
    """
    import inspect
    from server.core import JaatoServer

    src = inspect.getsource(JaatoServer.initialize)
    assert "bound_model_for_profile" in src, (
        "initialize() no longer resolves the profile's model through the "
        "shared binder"
    )
    assert "model_name = self._profile.model" not in src, (
        "initialize() reads profile.model directly again -- tiers-only "
        "profiles will store None as the session model"
    )
