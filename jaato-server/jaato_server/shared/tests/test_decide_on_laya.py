"""``decide()`` on ``laya``: step 6 of docs/design/decision-models.md.

Laya is served two ways, and both are driven here:

* ``mode: http`` (the default) against the local decision stand-in, which
  answers ``POST /v1/systemone`` as ``laya-serve`` does (TypeSafe Jev's
  shape): the URL, the bearer token, retries and refusals, the window;
* ``mode: local`` against a stand-in ``laya`` module, so no torch and no
  weights: the call shape ``laya.load(...).predict(state, questions)`` and
  the process retirement an in-process load must leave behind, reported
  by the runner's ``session.end`` sweep (#1565).

Plus the binding: a ``decision_models:`` entry naming ``provider: laya``
is served by a real ``JaatoSession.decision_provider``, and ``complete()``
refuses a chat turn.

No test touches the network.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict

import pytest

from jaato_sdk.plugins.model_provider.decisions import (
    DecisionModelOnlyError,
    DecisionQuestion,
    DecisionQuestionError,
)
from jaato_server.server.runner.rpc import slot_retire_reasons
from jaato_server.shared import process_retirement
from jaato_server.shared.plugins.model_provider import load_provider
from jaato_server.shared.plugins.model_provider.base import ProviderConfig
from jaato_server.shared.plugins.model_provider.laya import LayaProvider
from jaato_server.shared.plugins.model_provider.laya.errors import (
    DecisionRequestRejectedError,
    LayaNotInstalledError,
)
from jaato_server.shared.plugins.model_provider.laya.provider import (
    LOCAL_RETIRE_REASON,
)
from jaato_server.shared.tests.decision_standin import (
    DecisionStandIn,
    default_reply,
    laya_against,
)
from jaato_server.shared.tests.reversion import Reversion

_LAYA = "jaato-server/jaato_server/shared/plugins/model_provider/laya/provider.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"
_EXPLAIN = "jaato-server/jaato_server/shared/scaffold/explain.py"

REVERSIONS = [
    Reversion(
        target=_EXPLAIN,
        find='    if getattr(info, "context_resolution", None):\n',
        replace="    if False:\n",
        test="test_explain_states_lays_own_window_chain",
        because="explain would tell an author an unknown Laya window is an "
                "error, and that detection outranks the knob",
    ),
    Reversion(
        target=_LAYA,
        find="        raise DecisionModelOnlyError(self._model_name or DEFAULT_MODEL, self.name)\n",
        replace="        return None\n",
        test="test_complete_refuses_a_chat_turn",
        because="a chat turn would be handed to a model that writes no text",
    ),
    Reversion(
        target=_LAYA,
        find='        headers = ({"Authorization": f"Bearer {self._api_key}"}\n',
        replace='        headers = ({}\n',
        test="test_the_request_reaches_laya_serve_with_the_token",
        because="a laya-serve started with LAYA_API_KEY refuses every call",
    ),
    Reversion(
        target=_LAYA,
        find="        if isinstance(exc, InfrastructureError):\n",
        replace="        if False:\n",
        test="test_a_server_failure_is_retried",
        because="a restarting laya-serve would fail the decision outright",
    ),
    Reversion(
        target=_LAYA,
        find="        if estimate > limit:\n",
        replace="        if False:\n",
        test="test_a_state_over_the_window_is_refused_before_sending",
        because="a state over Laya's 512-token window would be sent and "
                "silently truncated by the encoder",
    ),
    Reversion(
        target=_LAYA,
        find="        refuse_truncated(raw, model)\n",
        replace="",
        test="test_a_truncated_answer_is_refused",
        because="an answer about the first 512 tokens of a state would be "
                "taken as an answer about the state",
    ),
    Reversion(
        target=_LAYA,
        find="            note_process_retirement(LOCAL_RETIRE_REASON)\n",
        replace="",
        test="test_a_local_load_retires_the_slot",
        because="a pool slot holding torch and the weights would be reused",
    ),
    Reversion(
        target=_RPC,
        find="    reasons.extend(r for r in process_retirement_reasons() if r not in reasons)\n",
        replace="",
        test="test_a_local_load_retires_the_slot",
        because="the runner would not report a provider's retirement, so "
                "the daemon pools the slot",
    ),
]

QUESTIONS = {
    "urgent": DecisionQuestion("noul", "Is this urgent?"),
    "team": DecisionQuestion("choice", "Which team?",
                             {"billing": "payments", "tech": None}),
    "mood": DecisionQuestion("score", "How upset?", ["Calm", "Upset", "Angry"]),
}


@pytest.fixture
def standin():
    with DecisionStandIn() as server:
        yield server


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setenv("AI_RETRY_BASE_DELAY", "0.01")
    monkeypatch.setenv("AI_RETRY_MAX_DELAY", "0.02")
    monkeypatch.setenv("AI_RETRY_LOG_SILENT", "1")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("JAATO_LAYA_HOST", "JAATO_LAYA_API_KEY", "LAYA_API_KEY",
                 "JAATO_LAYA_MODE", "JAATO_LAYA_CONTEXT_LENGTH"):
        monkeypatch.delenv(name, raising=False)
    # A fresh process-retirement record: a note lasts for the process.
    monkeypatch.setattr(process_retirement, "_reasons", [])


# ------------------------------------------------------------------ http

def test_the_request_reaches_laya_serve_with_the_token(standin):
    result = laya_against(standin, api_key="laya-secret").decide(
        "Payouts failing for 3 days", QUESTIONS)
    (path, headers, body), = standin.posts()
    assert path == "/v1/systemone"
    assert headers.get("Authorization") == "Bearer laya-secret"
    assert body["state"] == "Payouts failing for 3 days"
    assert set(body["questions"]) == set(QUESTIONS)
    assert result.answers["urgent"].value == pytest.approx(0.95)
    assert result.answers["team"].value == "billing"


def test_no_token_sends_no_authorization(standin):
    laya_against(standin).decide("x", {"urgent": QUESTIONS["urgent"]})
    (_path, headers, _body), = standin.posts()
    assert "Authorization" not in headers


def test_the_token_falls_back_to_laya_serves_own_variable(standin, monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "from-server-env")
    laya_against(standin).decide("x", {"urgent": QUESTIONS["urgent"]})
    (_path, headers, _body), = standin.posts()
    assert headers.get("Authorization") == "Bearer from-server-env"


def test_a_server_failure_is_retried(standin, fast_retries):
    # 502 (a proxy in front of a restarting server) and 529: neither is a
    # status the shared message-pattern fallback would retry on its own.
    standin.queue += [(502, {"error": "bad gateway"}, None),
                      (529, {"error": "overloaded"}, None)]
    result = laya_against(standin).decide("x", {"urgent": QUESTIONS["urgent"]})
    assert len(standin.posts()) == 3
    assert "urgent" in result.answers


def test_a_422_is_not_retried(standin, fast_retries):
    standin.queue.append((422, {"detail": "criteria must be a list"}, None))
    with pytest.raises(DecisionRequestRejectedError) as info:
        laya_against(standin).decide("x", {"mood": QUESTIONS["mood"]})
    assert len(standin.posts()) == 1
    assert "criteria must be a list" in str(info.value)


def test_a_state_over_the_window_is_refused_before_sending(standin):
    provider = laya_against(standin)          # convaiinnovations/laya: 512
    assert provider.get_context_limit() == 512
    with pytest.raises(DecisionQuestionError, match="512-token window"):
        provider.decide("word " * 1000, {"urgent": QUESTIONS["urgent"]})
    assert standin.posts() == []


def test_the_window_follows_the_checkpoint_and_the_knob(standin):
    assert laya_against(standin, subfolder="multilingual").get_context_limit() == 1024
    assert laya_against(standin, context_length=4096).get_context_limit() == 4096
    unknown = laya_against(standin, model="someone/fine-tune")
    assert unknown.get_context_limit() == 0   # unknown: not checked
    unknown.decide("word " * 1000, {"urgent": QUESTIONS["urgent"]})
    assert len(standin.posts()) == 1


def _truncated_reply(questions, **usage):
    reply = default_reply({"questions": questions})
    reply["usage"] = {"input_tokens": 512, "output_tokens": 0,
                      "state_tokens": 900, "state_tokens_dropped": 0,
                      "truncated": False, "truncated_questions": [], **usage}
    return reply


def test_a_truncated_answer_is_refused(standin):
    asked = {"urgent": QUESTIONS["urgent"]}
    wire = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}
    standin.queue.append((200, _truncated_reply(
        wire, truncated=True, state_tokens_dropped=388), None))
    with pytest.raises(DecisionQuestionError, match="388 of 900 state tokens dropped"):
        laya_against(standin, model="someone/fine-tune").decide("x", asked)


def test_a_question_without_room_is_refused_by_name(standin):
    asked = {"urgent": QUESTIONS["urgent"]}
    wire = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}
    standin.queue.append((200, _truncated_reply(
        wire, truncated_questions=["urgent"]), None))
    with pytest.raises(DecisionQuestionError) as info:
        laya_against(standin, model="someone/fine-tune").decide("x", asked)
    assert info.value.question_id == "urgent"


def test_a_whole_state_and_a_strict_body_pass(standin):
    asked = {"urgent": QUESTIONS["urgent"]}
    wire = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}
    standin.queue.append((200, _truncated_reply(wire), None))   # truncated: False
    provider = laya_against(standin)
    assert provider.decide("x", asked).usage.prompt_tokens == 512
    # LAYA_JEV_STRICT reduces usage to the two counts: no evidence either way.
    assert provider.decide("x", asked).answers


def test_complete_refuses_a_chat_turn(standin):
    provider = laya_against(standin)
    with pytest.raises(DecisionModelOnlyError, match="decide"):
        provider.complete([])
    assert provider.is_decisions_only()
    assert provider.output_modalities() == {"decisions"}


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        LayaProvider().initialize(ProviderConfig(extra={"mode": "grpc"}))


def test_discovery_finds_it_and_it_needs_no_credential():
    provider = load_provider("laya", ProviderConfig(extra={}))
    assert provider.name == "laya"
    assert LayaProvider().verify_auth() is True


# ----------------------------------------------------------------- local

class _FakeAgent:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((state, questions))
        # The package's own shape (laya 0.4.6, Agent.system_one).
        reply = default_reply({"questions": questions})
        reply["model"] = "laya-rl-agent"
        reply["usage"] = {"input_tokens": 140, "output_tokens": 0,
                          "state_tokens": 12, "state_tokens_dropped": 0,
                          "truncated": False, "truncated_questions": []}
        return reply


@pytest.fixture
def fake_laya(monkeypatch):
    module = types.ModuleType("laya")
    module.loads = []
    agent = _FakeAgent()

    def load(model, **kwargs):
        module.loads.append((model, kwargs))
        return agent

    module.load = load
    module.agent = agent
    monkeypatch.setitem(sys.modules, "laya", module)
    return module


def _local(**extra):
    provider = LayaProvider()
    provider.initialize(ProviderConfig(extra={"mode": "local", **extra}))
    provider.connect("convaiinnovations/laya")
    return provider


def test_local_mode_calls_the_package_once(fake_laya):
    provider = _local(subfolder="typed-decisions")
    provider.decide("ticket", QUESTIONS)
    result = provider.decide("ticket two", QUESTIONS)
    assert fake_laya.loads == [("convaiinnovations/laya",
                                {"subfolder": "typed-decisions"})]
    state, questions = fake_laya.agent.calls[-1]
    assert state == "ticket two" and questions["team"]["type"] == "choice"
    assert result.answers["mood"].value == pytest.approx(1.0)
    assert result.model == "laya-rl-agent" and result.usage.prompt_tokens == 140


def test_a_local_load_retires_the_slot(fake_laya):
    class _NoPlugins:
        def list_available(self):
            return []

    assert slot_retire_reasons(_NoPlugins()) == []
    _local().decide("ticket", {"urgent": QUESTIONS["urgent"]})
    assert slot_retire_reasons(_NoPlugins()) == [LOCAL_RETIRE_REASON]


def test_http_mode_retires_nothing(standin):
    class _NoPlugins:
        def list_available(self):
            return []

    laya_against(standin).decide("x", {"urgent": QUESTIONS["urgent"]})
    assert slot_retire_reasons(_NoPlugins()) == []


def test_local_mode_without_the_package_says_how_to_fix_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)   # import raises
    with pytest.raises(LayaNotInstalledError, match="laya-serve"):
        _local().decide("x", {"urgent": QUESTIONS["urgent"]})


# --------------------------------------------------------------- binding

def test_a_decision_models_entry_is_served_by_laya(standin):
    from unittest.mock import MagicMock

    from jaato_server.shared.jaato_session import JaatoSession

    runtime = MagicMock()
    runtime.provider_name = "anthropic"
    runtime.create_provider.side_effect = (
        lambda model=None, provider_name=None, **_kw:
        laya_against(standin, model=model) if provider_name == "laya" else None)
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    session = JaatoSession(runtime, "claude-sonnet-4-6", provider_name="anthropic")
    session.configure(decision_models={
        "local": {"provider": "laya", "model": "convaiinnovations/laya"}})
    provider = session.decision_provider("local")
    assert isinstance(provider, LayaProvider)
    assert provider.decide("x", {"urgent": QUESTIONS["urgent"]}).answers
    assert runtime.create_provider.call_args.kwargs["provider_name"] == "laya"


# ---------------------------------------------------------------- explain

def test_explain_states_lays_own_window_chain():
    from jaato_server.shared.scaffold.introspection_verbs import render_topic

    ok, _data, text, error = render_topic("provider", "laya")
    assert ok, error
    window = [ln for ln in text.splitlines() if "context window" in ln]
    assert window and "not checked" in window[0], text
    assert "detect WINS" not in window[0]
    assert "decision_models" in text   # the note an author reads first
