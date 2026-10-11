"""A profile binds decision models by name, outside the tier ladder.

Step 3 of docs/design/decision-models.md (§7).  A decision model (TypeSafe
Jev) answers typed questions through ``provider.decide()`` and cannot chat,
so it must never become the session's active model.  The profile binds it
in a block of its own::

    decision_models:
      triage: {provider: openrouter, model: typesafe/jev-1.13}

and a consumer asks the session for an entry by name.  These tests follow
the block from the file to the provider a consumer gets:

- the loader parses it and refuses a malformed entry by name;
- inheritance replaces by entry name, and parents that disagree conflict;
- the session snapshot, the envelope producer and the wire keep it;
- the runner bootstrap hands it to ``create_session``;
- ``JaatoSession.decision_provider`` builds each entry's provider once,
  separate from the chat provider, and refuses an unknown name or a
  provider with no ``decide()``;
- a tier bound to a decisions-only model is refused at startup;
- ``jaato-scaffold validate`` reports a provider that cannot decide and a
  tier binding a declared decision model.

No test touches the network.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_server.shared.decision_models import (
    DecisionModelBinding,
    DecisionModelUnavailableError,
    DecisionModelsConfigError,
    UnknownDecisionModelError,
    merge_decision_models,
    parse_decision_models,
)
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.model_tiers import (
    RESERVED_FALLBACK_KEY,
    RESERVED_INITIAL_KEY,
    ModelTierConfig,
    ModelTierConfigError,
)
from jaato_server.shared.plugins.subagent.config import (
    SubagentProfile,
    _merge_profiles,
    build_inline_profile,
    profile_from_snapshot,
    profile_to_snapshot,
)
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_DM = "jaato-server/jaato_server/shared/decision_models.py"
_CONFIG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_ENVELOPE = "jaato-server/jaato_server/shared/session_envelope.py"
_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"
_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_DM,
        find="    if unknown:\n        raise DecisionModelsConfigError(",
        replace="    if False:\n        raise DecisionModelsConfigError(",
        test="test_a_malformed_entry_is_refused_by_name",
        because="an entry with a misspelt key loads and the key is ignored",
    ),
    Reversion(
        target=_DM,
        find="    merged.update(child or {})\n",
        replace="",
        test="test_the_child_replaces_an_entry_by_name",
        because="a child cannot rebind the entry its parent declared",
    ),
    Reversion(
        target=_CONFIG,
        find="        decision_models=parse_decision_models_block(data),\n"
             "        budget_control=blocks[\"budget_control\"],",
        replace="        budget_control=blocks[\"budget_control\"],",
        test="test_the_block_survives_the_session_snapshot",
        because="a revived session loses its decision models",
    ),
    Reversion(
        target=_SPAWN,
        find="        decision_models=decision_models_dict,\n",
        replace="",
        test="test_the_envelope_producer_carries_the_block",
        because="a runner-served session never sees the profile's bindings",
    ),
    Reversion(
        target=_SESSION,
        find="            self._decision_providers[name] = prov\n",
        replace="",
        test="test_an_entry_is_built_once_and_apart_from_the_chat_provider",
        because="every decide() call builds and connects a new provider",
    ),
    Reversion(
        target=_SESSION,
        find="            self._refuse_decisions_only_tiers()\n",
        replace="",
        test="test_a_tier_bound_to_a_decision_model_is_refused_at_startup",
        because="enter_tier into a decision model strands the agent",
    ),
    Reversion(
        target=_VALIDATE,
        find="    _check_decision_models(profile, add)\n",
        replace="",
        test="test_validate_reports_a_provider_that_cannot_decide",
        because="a binding no consumer can use validates clean",
    ),
]

JEV = {"provider": "openrouter", "model": "typesafe/jev-1.13"}


# ============================================================ the block


def test_a_well_formed_block_parses():
    bindings = parse_decision_models(
        {"triage": JEV, "local": {"model": "convaiinnovations/laya"}})
    assert bindings["triage"] == DecisionModelBinding(
        name="triage", model="typesafe/jev-1.13", provider="openrouter")
    assert bindings["local"].provider is None
    assert parse_decision_models(None) == {}


@pytest.mark.parametrize("raw, names", [
    ({"Triage": JEV}, "Triage"),
    ({"triage": {"provider": "openrouter"}}, "model"),
    ({"triage": {**JEV, "modle": "x"}}, "modle"),
    ({"triage": "typesafe/jev-1.13"}, "triage"),
    ({"triage": {**JEV, "model": ""}}, "triage.model"),
    (["triage"], "object"),
    ({f"m{i:02d}": JEV for i in range(17)}, "at most 16"),
])
def test_a_malformed_entry_is_refused_by_name(raw, names):
    with pytest.raises(DecisionModelsConfigError) as exc:
        parse_decision_models(raw)
    assert names in str(exc.value)


def test_a_malformed_block_refuses_the_profile():
    with pytest.raises(ValueError, match="decision_models"):
        build_inline_profile(
            {"plugins": [], "decision_models": {"triage": {"provider": "x"}}})


# ============================================================ inheritance


def _profile(name, decision_models=None, **kw):
    return SubagentProfile(
        name=name, description="d", plugins=[],
        decision_models=dict(decision_models or {}), **kw)


def test_the_child_replaces_an_entry_by_name():
    parent = {"triage": JEV, "local": {"model": "laya"}}
    child = {"triage": {"model": "typesafe/jev-2"}}
    merged = merge_decision_models(parent, child)
    assert merged["triage"] == {"model": "typesafe/jev-2"}, (
        "the child's entry must replace the parent's whole, provider too")
    assert merged["local"] == {"model": "laya"}


def test_inherits_resolves_the_block():
    base = _profile("base", {"triage": JEV})
    child = _profile("child", {"local": {"model": "laya"}}, inherits=["base"])
    errors = {}
    merged = _merge_profiles("child", [base], child, errors)
    assert not errors
    assert set(merged.decision_models) == {"triage", "local"}


def test_two_parents_binding_one_name_differently_conflict():
    a = _profile("a", {"triage": JEV})
    b = _profile("b", {"triage": {"model": "other"}})
    child = _profile("child", inherits=["a", "b"])
    errors = {}
    assert _merge_profiles("child", [a, b], child, errors) is None
    assert "decision_models.triage" in errors["child"]


# ============================================================ carried


def test_the_block_survives_the_session_snapshot():
    profile = build_inline_profile(
        {"plugins": [], "model": "m", "decision_models": {"triage": JEV}})
    revived = profile_from_snapshot(profile_to_snapshot(profile))
    assert revived.decision_models == {"triage": JEV}


def _envelope_profile(decision_models):
    return SimpleNamespace(
        name="p", description="d", provider="openrouter", model="m",
        plugins=[], preloaded_plugins=set(), plugin_configs={},
        tool_scopes={}, model_tiers={}, gc=None, runtime_limits=None,
        env={}, completion_payload_schema=None, spawn_payload_schema=None,
        completion_processors=[], budget_control=None, quirks={},
        apparmor=False, apparmor_fragments=None, system_instructions=None,
        agent_params={}, suppress_base_instructions=False, config_root=None,
        inherits=None, decision_models=decision_models,
    )


def test_the_envelope_producer_carries_the_block():
    from jaato_server.server.runner_spawn import build_session_envelope
    server = SimpleNamespace(
        _profile=_envelope_profile({"triage": JEV}), config_root=None,
        _main_agent_id="main", _cascade_driver_id=None,
    )
    env = build_session_envelope(
        server=server, session_id="s1", workspace_path="/tmp/ws",
        profile_name="p")
    assert env.decision_models == {"triage": JEV}
    wire = SessionInitEnvelope.from_dict(env.to_dict())
    assert parse_decision_models(wire.decision_models)["triage"].model == (
        "typesafe/jev-1.13")


def test_a_profile_without_the_block_sends_none():
    from jaato_server.server.runner_spawn import build_session_envelope
    server = SimpleNamespace(
        _profile=_envelope_profile({}), config_root=None,
        _main_agent_id="main", _cascade_driver_id=None,
    )
    env = build_session_envelope(
        server=server, session_id="s2", workspace_path="/tmp/ws",
        profile_name="p")
    assert env.decision_models is None


def test_the_runner_bootstrap_hands_the_block_to_create_session():
    """The call site, by AST: #1133's lesson is that the parse can exist
    and nothing pass it on."""
    src = Path(__file__).resolve().parents[2] / "server" / "runner" / "session.py"
    tree = ast.parse(src.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_session"
        and any(k.arg == "tier_config" for k in node.keywords)
    ]
    assert calls, "the runner's create_session call was not found"
    assert all(
        any(k.arg == "decision_models" for k in c.keywords) for c in calls
    ), "the runner bootstrap does not pass decision_models to create_session"


# ============================================================ the session


def _chat_provider(decisions_only_models=()):
    p = MagicMock()
    p.name = "openrouter"
    p.model_name = None

    def connect(model, skip_model_test=True):
        p.model_name = model
    p.connect.side_effect = connect
    p.get_context_limit.return_value = 100_000
    p.is_decisions_only.side_effect = (
        lambda model=None: model in decisions_only_models)
    return p


def _decision_provider():
    p = MagicMock(spec=["connect", "decide", "name", "set_agent_context"])
    p.name = "openrouter"
    return p


def _session(*, tier_config=None, decision_models=None, chat=None,
             build=None):
    chat = chat or _chat_provider()
    built = []

    def create_provider(model=None, provider_name=None, **_kw):
        built.append((provider_name, model))
        if build is not None and model in build:
            return build[model]
        chat.connect(model)
        return chat

    runtime = MagicMock()
    runtime.provider_name = "openrouter"
    runtime.create_provider.side_effect = create_provider
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.ledger = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    session = JaatoSession(runtime, "chat-model", provider_name="openrouter")
    session.configure(tier_config=tier_config, decision_models=decision_models)
    return session, chat, built


def test_an_entry_is_built_once_and_apart_from_the_chat_provider():
    jev = _decision_provider()
    session, chat, built = _session(
        decision_models={"triage": JEV}, build={"typesafe/jev-1.13": jev})
    first = session.decision_provider("triage")
    second = session.decision_provider("triage")
    assert first is jev and second is jev
    assert built.count(("openrouter", "typesafe/jev-1.13")) == 1, built
    assert chat.model_name != "typesafe/jev-1.13", (
        "the chat provider was re-pointed at the decision model")
    assert session.decision_model_names() == ["triage"]


def test_an_entry_with_no_provider_uses_the_sessions():
    jev = _decision_provider()
    session, _, built = _session(
        decision_models={"triage": {"model": "typesafe/jev-1.13"}},
        build={"typesafe/jev-1.13": jev})
    session.decision_provider("triage")
    assert ("openrouter", "typesafe/jev-1.13") in built


def test_an_unknown_name_says_what_is_bound():
    session, _, _ = _session(decision_models={"triage": JEV})
    with pytest.raises(UnknownDecisionModelError) as exc:
        session.decision_provider("trige")
    assert exc.value.available == ["triage"]
    assert "triage" in str(exc.value)


def test_a_provider_without_decide_is_refused():
    no_decide = MagicMock(spec=["connect", "name"])
    session, _, _ = _session(
        decision_models={"triage": JEV}, build={"typesafe/jev-1.13": no_decide})
    with pytest.raises(DecisionModelUnavailableError, match="decide"):
        session.decision_provider("triage")


def _tiers(executor_model, planner_model):
    return ModelTierConfig.from_unified_dict({
        "executor": {"model": executor_model},
        "planner": {"model": planner_model},
        RESERVED_INITIAL_KEY: "executor",
        RESERVED_FALLBACK_KEY: "executor",
    })


def test_a_tier_bound_to_a_decision_model_is_refused_at_startup():
    chat = _chat_provider(decisions_only_models={"typesafe/jev-1.13"})
    session, _, _ = _session(
        tier_config=_tiers("chat-model", "typesafe/jev-1.13"), chat=chat)
    with pytest.raises(ModelTierConfigError, match="decision_models"):
        session._ensure_provider()


def test_chat_tiers_start():
    chat = _chat_provider(decisions_only_models={"typesafe/jev-1.13"})
    session, _, _ = _session(
        tier_config=_tiers("chat-model", "other-chat"), chat=chat)
    session._ensure_provider()


def test_a_failed_catalog_lookup_does_not_stop_the_session():
    chat = _chat_provider()
    chat.is_decisions_only.side_effect = RuntimeError("catalog down")
    session, _, _ = _session(
        tier_config=_tiers("chat-model", "other-chat"), chat=chat)
    session._ensure_provider()


# ============================================================ validate


@pytest.fixture(scope="module")
def introspected():
    from jaato_server.shared.scaffold import introspect
    return introspect.providers(), introspect.plugins()


def _validate(introspected, **kw):
    from jaato_server.shared.scaffold import validate
    providers, plugins = introspected
    profile = SubagentProfile(name="t", description="d", plugins=[], **kw)
    return [(d.severity, d.code) for d in validate.validate_profile(
        profile, providers=providers, plugins=plugins, gc_names=[])]


def test_validate_reports_a_provider_that_cannot_decide(introspected):
    codes = _validate(
        introspected, provider="anthropic", model="claude",
        decision_models={"triage": {"model": "typesafe/jev-1.13"}})
    assert ("error", "decision_model_provider_unsupported") in codes


def test_validate_accepts_a_provider_that_can(introspected):
    codes = _validate(
        introspected, provider="openrouter", model="m",
        decision_models={"triage": JEV})
    assert not [c for c in codes if c[1].startswith("decision_model")]


def test_validate_reports_an_unknown_provider(introspected):
    codes = _validate(
        introspected, provider="openrouter", model="m",
        decision_models={"triage": {"provider": "nope", "model": "x"}})
    assert ("error", "decision_model_provider_unknown") in codes


def test_validate_reports_a_tier_bound_to_a_declared_decision_model(
        introspected):
    codes = _validate(
        introspected, provider="openrouter",
        model_tiers={"executor": "chat", "planner": "typesafe/jev-1.13",
                     RESERVED_INITIAL_KEY: "executor"},
        decision_models={"triage": JEV})
    assert ("error", "decision_model_as_tier") in codes
