"""The ``decide`` tool asks a session's bound decision model typed questions.

Step 4 of docs/design/decision-models.md (§8.1).  The tool names a
``decision_models:`` entry of the CALLING session, the session builds that
entry's provider (step 3), and the answers come back typed.  These tests
drive the real plugin, a real ``JaatoSession`` and the real
``OpenRouterProvider`` against the local decision stand-in, so the request
the stand-in records is the one the tool's arguments produce.

No test touches the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.decision.plugin import (
    TOOL_NAME,
    DecisionPlugin,
    build_questions,
)
from jaato_server.shared.session_context import (
    isolated_current_session,
    set_current_session,
)
from jaato_server.shared.tests.decision_standin import (
    JEV,
    DecisionStandIn,
    openrouter_against,
)
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/decision/plugin.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="        return names is None or bool(names)\n",
        replace="        return True\n",
        test="test_a_session_binding_nothing_does_not_see_the_tool",
        because="a session with no decision model is offered a tool that "
                "can only fail",
    ),
    Reversion(
        target=_PLUGIN,
        find='        params["properties"]["model"]["enum"] = sorted(names)\n',
        replace="",
        test="test_model_is_limited_to_the_sessions_entries",
        because="the model is free to invent an entry name",
    ),
    Reversion(
        target=_PLUGIN,
        find="            out[name] = item.get(\"description\") "
             "if isinstance(item, dict) else None\n",
        replace="            out[name] = None\n",
        test="test_the_request_is_built_from_the_tools_arguments",
        because="an option's description never reaches the decision model",
    ),
    Reversion(
        target=_PLUGIN,
        find='            return False, {"error": str(exc), "model": name}\n',
        replace='            return {"error": str(exc), "model": name}\n',
        test="test_an_unknown_entry_is_a_failed_call",
        because="a refused call reads as a success to the reliability "
                "plugin and the trace (#1053)",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/validate.py",
        find="        _check_decide_has_a_model(profile, add)\n",
        replace="",
        test="test_validate_warns_when_decide_has_no_model",
        because="a profile enabling a tool that can never appear validates "
                "clean",
    ),
    Reversion(
        target=_PLUGIN,
        find="        record(\"decision\", entry)\n",
        replace="        return\n        record(\"decision\", entry)\n",
        test="test_the_spend_is_recorded_in_the_ledger",
        because="the decision call's spend reaches no record",
    ),
]

STATE = "Customer: my order #4411 never arrived and nobody answers."
QUESTIONS = [
    {"id": "urgent", "type": "noul", "instructions": "Is this urgent?",
     "true_means": "needs a reply today"},
    {"id": "team", "type": "choice", "instructions": "Which team?",
     "options": [{"option": "shipping", "description": "deliveries"},
                 {"option": "billing"}]},
    {"id": "anger", "type": "score", "instructions": "How frustrated?",
     "levels": ["calm", "annoyed", "furious"]},
]


@pytest.fixture
def standin():
    with DecisionStandIn() as server:
        yield server


@pytest.fixture(autouse=True)
def _restore_session():
    with isolated_current_session():
        yield


def _session(decision_models, provider=None):
    ledger = MagicMock()
    runtime = MagicMock()
    runtime.provider_name = "openrouter"
    runtime.ledger = ledger
    runtime.create_provider.side_effect = (
        lambda model=None, provider_name=None, **_kw: provider)
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    session = JaatoSession(runtime, "chat-model", provider_name="openrouter")
    session.configure(decision_models=decision_models)
    set_current_session(session)
    return session, ledger


def _decide(**args):
    return DecisionPlugin().get_executors()[TOOL_NAME](
        {"model": "triage", "state": STATE, "questions": QUESTIONS, **args})


# ============================================================ the call


def test_the_request_is_built_from_the_tools_arguments(standin):
    _session({"triage": {"provider": "openrouter", "model": JEV}},
             provider=openrouter_against(standin))
    result = _decide()
    (path, _headers, body), = standin.posts()
    assert path == "/api/alpha/decisions"
    assert body["model"] == JEV
    assert body["state"] == STATE
    assert body["questions"] == {
        "urgent": {"type": "noul", "instructions": "Is this urgent?",
                   "criteria": {"true": "needs a reply today", "false": None}},
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"shipping": "deliveries", "billing": None}},
        "anger": {"type": "score", "instructions": "How frustrated?",
                  "criteria": ["calm", "annoyed", "furious"]},
    }
    assert isinstance(result, dict), result
    assert result["model"] == "triage"
    assert set(result["answers"]) == {"urgent", "team", "anger"}
    assert result["answers"]["team"]["value"] == "shipping"


def test_an_unknown_entry_is_a_failed_call(standin):
    _session({"triage": {"provider": "openrouter", "model": JEV}},
             provider=openrouter_against(standin))
    ok, payload = _decide(model="trige")
    assert ok is False
    assert "triage" in payload["error"]
    assert standin.posts() == []


def test_a_malformed_question_sends_nothing(standin):
    _session({"triage": {"provider": "openrouter", "model": JEV}},
             provider=openrouter_against(standin))
    ok, payload = _decide(questions=[
        {"id": "team", "type": "choice", "instructions": "Which team?"}])
    assert ok is False and "options" in payload["error"]
    ok, payload = _decide(questions=[
        {"id": "s", "type": "score", "instructions": "?", "levels": ["one"]}])
    assert ok is False
    assert standin.posts() == []


def test_the_spend_is_recorded_in_the_ledger(standin):
    _, ledger = _session({"triage": {"provider": "openrouter", "model": JEV}},
                         provider=openrouter_against(standin))
    _decide()
    kinds = [c.args[0] for c in ledger._record.call_args_list]
    assert "decision" in kinds
    record = ledger._record.call_args_list[kinds.index("decision")].args[1]
    assert record["entry"] == "triage" and record["questions"] == 3


def test_outside_a_session_the_call_is_refused():
    ok, payload = _decide()
    assert ok is False and "session" in payload["error"]


@pytest.mark.parametrize("raw, words", [
    ([], "non-empty"),
    ([{"id": "a", "type": "noul", "instructions": "?"}] * 2, "twice"),
    ([{"type": "noul", "instructions": "?"}], "id"),
    ([{"id": "a", "type": "rank", "instructions": "?"}], "noul, choice"),
    ([{"id": f"q{i}", "type": "noul", "instructions": "?"}
      for i in range(33)], "at most"),
])
def test_questions_are_checked_before_anything_is_built(raw, words):
    with pytest.raises(ValueError, match=words):
        build_questions(raw)


# ============================================================ the schema


def test_a_session_binding_nothing_does_not_see_the_tool():
    _session({})
    assert DecisionPlugin().is_tool_visible(TOOL_NAME) is False


def test_a_session_binding_an_entry_sees_it():
    _session({"triage": {"model": JEV}})
    assert DecisionPlugin().is_tool_visible(TOOL_NAME) is True


def test_model_is_limited_to_the_sessions_entries():
    _session({"triage": {"model": JEV}, "local": {"model": "laya"}})
    plugin = DecisionPlugin()
    schema, = plugin.get_tool_schemas()
    narrowed = plugin.narrow_tool_schema(schema)
    assert narrowed.parameters["properties"]["model"]["enum"] == [
        "local", "triage"]
    assert "enum" not in schema.parameters["properties"]["model"], (
        "the shared schema was mutated; another session would see these names")


def test_the_tool_is_in_the_decision_category():
    from jaato_sdk.plugins.model_provider.types import TOOL_CATEGORIES
    schema, = DecisionPlugin().get_tool_schemas()
    assert schema.category == "decision"
    assert schema.category in TOOL_CATEGORIES
    registry = MagicMock()
    DecisionPlugin().set_plugin_registry(registry)
    name, description = registry.register_category.call_args.args
    assert name == "decision" and description


def test_the_tool_is_not_auto_approved():
    assert TOOL_NAME not in DecisionPlugin().get_auto_approved_tools()


# ============================================================ validate


def test_validate_warns_when_decide_has_no_model():
    from jaato_server.shared.plugins.subagent.config import SubagentProfile
    from jaato_server.shared.scaffold import introspect, validate
    def codes(**kw):
        profile = SubagentProfile(name="t", description="d", **kw)
        return [d.code for d in validate.validate_profile(
            profile, providers=introspect.providers(),
            plugins=introspect.plugins(), gc_names=[])]
    assert "decision_tool_without_models" in codes(
        plugins=["decision"], provider="openrouter", model="m")
    assert "decision_tool_without_models" not in codes(
        plugins=["decision"], provider="openrouter", model="m",
        decision_models={"triage": {"model": JEV}})
