"""``decide()`` on ``openrouter``: step 2 of docs/design/decision-models.md.

Two halves:

* the wire contract in ``jaato_sdk.plugins.model_provider.decisions``,
  checked against the request and response the vendor documents (TypeSafe
  API reference, Jev 1.13), so a parser that disagrees with them fails here
  rather than on the first live call;
* ``OpenRouterProvider.decide()`` driven against a local stand-in for
  ``POST /api/alpha/decisions`` (``decision_standin.py``): the URL, the
  headers, retries, refusals, cancellation, and ``complete()`` refusing a
  decisions-only model.

No test touches the network.  ``examples/provider_smoke_decisions.py
--live-openrouter`` is the check against the real endpoint.
"""

from __future__ import annotations

import copy
import threading
import time

import pytest

from jaato_sdk.plugins.model_provider.decisions import (
    DecisionModelOnlyError,
    DecisionQuestion,
    DecisionQuestionError,
    DecisionResponseError,
    build_decision_request,
    parse_decision_response,
)
from jaato_sdk.plugins.model_provider.types import CancelledException, CancelToken
from jaato_server.shared.plugins.model_provider.openrouter.decisions import (
    decisions_url,
)
from jaato_server.shared.plugins.model_provider.openrouter.errors import (
    AuthenticationError,
    DecisionRequestRejectedError,
)
from jaato_server.shared.tests.decision_standin import (
    DecisionStandIn,
    openrouter_against,
)
from jaato_server.shared.tests.reversion import Reversion

_SDK = "jaato-sdk/jaato_sdk/plugins/model_provider/decisions.py"
_OR = "jaato-server/jaato_server/shared/plugins/model_provider/openrouter/"

REVERSIONS = [
    Reversion(
        target=_SDK,
        find="    if missing:\n        raise DecisionResponseError(",
        replace="    if False:\n        raise DecisionResponseError(",
        test="test_an_unanswered_question_is_an_error_naming_it",
        because="a missing answer would surface as a KeyError, or as a default",
    ),
    Reversion(
        target=_OR + "provider.py",
        find="            raise DecisionModelOnlyError(self._model_name, self.name)\n",
        replace="            pass\n",
        test="test_complete_refuses_a_decisions_only_model",
        because="a chat turn would be sent to a model that cannot answer one",
    ),
    Reversion(
        target=_OR + "decisions.py",
        find="            if cancel_token is not None and cancel_token.is_cancelled:\n",
        replace="            if False:\n",
        test="test_a_cancel_stops_the_wait",
        because="a cancelled decision would wait out the whole request",
    ),
    Reversion(
        target=_OR + "decisions.py",
        find="    if status >= 500:\n        raise InfrastructureError(",
        replace="    if status >= 400:\n        raise InfrastructureError(",
        test="test_a_422_is_not_retried",
        because="a malformed request would be retried as if transient",
    ),
    Reversion(
        target=_OR + "decisions.py",
        find='        root = root[: -len("/v1")]\n',
        replace="        pass\n",
        test="test_the_request_reaches_the_decision_endpoint",
        because="the endpoint sits beside /v1, not inside it",
    ),
]


# --------------------------------------------- the vendor's documented example

VENDOR_REQUEST = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                      "criteria": {"true": "Explicitly time-sensitive",
                                   "false": "No urgency expressed"}},
        "department": {"type": "choice",
                       "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds",
                                    "technical": "Bugs, outages, integrations",
                                    "sales": None}},
        "frustration": {"type": "score",
                        "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}

VENDOR_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.95},
        "department": {"type": "choice", "choice": "billing",
                       "probabilities": {"billing": 0.88, "technical": 0.12,
                                         "sales": 0.0},
                       "confidence": 0.81},
        "frustration": {"type": "score", "score": 1.05,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                        "confidence": 0.92},
    },
    "usage": {"input_tokens": 296, "output_tokens": 20},
}


def _vendor_questions():
    return {
        qid: DecisionQuestion(q["type"], q["instructions"], q.get("criteria"))
        for qid, q in VENDOR_REQUEST["questions"].items()
    }


def test_the_request_body_is_the_vendors():
    body = build_decision_request(VENDOR_REQUEST["state"], _vendor_questions(),
                                  "jev-latest")
    assert body == VENDOR_REQUEST


def test_the_vendors_response_parses():
    result = parse_decision_response(VENDOR_RESPONSE, _vendor_questions(), "x")
    assert result.model == "jev-1.13.0"
    urgent, dept, frust = (result.answers[k] for k in
                           ("is_urgent", "department", "frustration"))
    assert urgent.value == 0.95 and urgent.confidence is None
    assert dept.value == "billing" and dept.confidence == 0.81
    assert dept.probabilities == {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    assert frust.value == 1.05 and frust.legend["2"] == "Very angry"
    assert result.usage.reported and result.usage.prompt_tokens == 296
    assert result.usage.output_tokens == 20


def test_an_unanswered_question_is_an_error_naming_it():
    body = copy.deepcopy(VENDOR_RESPONSE)
    del body["answers"]["department"]
    with pytest.raises(DecisionResponseError) as err:
        parse_decision_response(body, _vendor_questions())
    assert err.value.question_id == "department"


@pytest.mark.parametrize("qid,answer", [
    ("department", {"type": "choice", "choice": "legal"}),
    ("is_urgent", {"type": "noul", "noul": 1.5}),
    ("frustration", {"type": "score", "score": 7}),
    ("is_urgent", {"type": "choice", "choice": "billing"}),
])
def test_an_answer_outside_what_was_asked_is_refused(qid, answer):
    body = copy.deepcopy(VENDOR_RESPONSE)
    body["answers"][qid] = answer
    with pytest.raises(DecisionResponseError) as err:
        parse_decision_response(body, _vendor_questions())
    assert err.value.question_id == qid


def test_no_usage_is_unreported_not_zero():
    body = copy.deepcopy(VENDOR_RESPONSE)
    del body["usage"]
    assert parse_decision_response(body, _vendor_questions()).usage.reported is False


@pytest.mark.parametrize("question", [
    DecisionQuestion("maybe", "x"),
    DecisionQuestion("noul", ""),
    DecisionQuestion("noul", "x", {"yes": "y"}),
    DecisionQuestion("choice", "x", {}),
    DecisionQuestion("choice", "x", {f"o{i}": None for i in range(256)}),
    DecisionQuestion("score", "x", ["only one"]),
    DecisionQuestion("score", "x", [str(i) for i in range(11)]),
    DecisionQuestion("score", "x", {"0": "a", "1": "b"}),
])
def test_a_malformed_question_is_refused_locally(question):
    with pytest.raises(DecisionQuestionError) as err:
        build_decision_request("state", {"q": question}, "m")
    assert err.value.question_id == "q"


# ----------------------------------------------------- decide() on openrouter


@pytest.fixture
def standin():
    with DecisionStandIn() as server:
        yield server


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setenv("AI_RETRY_BASE_DELAY", "0.01")
    monkeypatch.setenv("AI_RETRY_MAX_DELAY", "0.02")
    monkeypatch.setenv("AI_RETRY_LOG_SILENT", "1")


def test_decisions_url_sits_beside_v1():
    assert decisions_url("https://openrouter.ai/api/v1") == (
        "https://openrouter.ai/api/alpha/decisions")
    assert decisions_url("https://openrouter.ai/api/v1/") == (
        "https://openrouter.ai/api/alpha/decisions")
    assert decisions_url("http://proxy/or") == "http://proxy/or/alpha/decisions"
    assert decisions_url("https://x/api/v1", "https://y/d") == "https://y/d"


def test_the_request_reaches_the_decision_endpoint(standin):
    provider = openrouter_against(standin)
    result = provider.decide(VENDOR_REQUEST["state"], _vendor_questions())
    (path, headers, body), = standin.posts()
    assert path == "/api/alpha/decisions"
    assert headers["Authorization"] == "Bearer sk-or-test"
    assert headers["X-OpenRouter-Title"]
    assert body["model"] == "typesafe/jev-1.13"
    assert body["questions"] == VENDOR_REQUEST["questions"]
    assert set(result.answers) == {"is_urgent", "department", "frustration"}


def test_complete_refuses_a_decisions_only_model(standin):
    provider = openrouter_against(standin)
    with pytest.raises(DecisionModelOnlyError):
        provider.complete([])
    assert standin.posts() == []


def test_a_chat_model_is_not_refused(standin):
    provider = openrouter_against(standin, model="openai/gpt-4o")
    assert provider._decisions_only is False


def test_a_transient_status_is_retried(standin, fast_retries):
    standin.queue += [(529, {"error": "overloaded"}, None),
                      (429, {"error": "slow down"}, {"Retry-After": "0"})]
    result = openrouter_against(standin).decide("s", _vendor_questions())
    assert len(standin.posts()) == 3
    assert result.answers["is_urgent"].value == 0.95


def test_a_422_is_not_retried(standin, fast_retries):
    standin.queue.append((422, {"error": "questions.x.criteria invalid"}, None))
    with pytest.raises(DecisionRequestRejectedError) as err:
        openrouter_against(standin).decide("s", _vendor_questions())
    assert err.value.status_code == 422
    assert "criteria invalid" in str(err.value)
    assert len(standin.posts()) == 1


def test_a_401_is_not_retried(standin, fast_retries):
    standin.queue.append((401, {"error": "bad key"}, None))
    with pytest.raises(AuthenticationError):
        openrouter_against(standin).decide("s", _vendor_questions())
    assert len(standin.posts()) == 1


def test_a_malformed_question_sends_nothing(standin):
    with pytest.raises(DecisionQuestionError):
        openrouter_against(standin).decide(
            "s", {"q": DecisionQuestion("choice", "x", {})})
    assert standin.posts() == []


def test_a_state_over_the_window_sends_nothing(standin):
    with pytest.raises(DecisionQuestionError, match="window"):
        openrouter_against(standin).decide("x" * 200_000, _vendor_questions())
    assert standin.posts() == []


def test_a_cancel_stops_the_wait(standin):
    standin.hang = threading.Event()
    provider = openrouter_against(standin)
    token = CancelToken()
    cancelled_at = []

    def cancel_once_in_flight():
        if standin.arrived.wait(10):
            cancelled_at.append(time.monotonic())
            token.cancel()

    threading.Thread(target=cancel_once_in_flight, daemon=True).start()
    with pytest.raises(CancelledException):
        provider.decide("s", _vendor_questions(), cancel_token=token)
    # The request was on the wire when the cancel came, and the call
    # returned long before the stand-in would have answered.
    assert cancelled_at and len(standin.posts()) == 1
    assert time.monotonic() - cancelled_at[0] < 2.0
