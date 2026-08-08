"""Tests for the runner→daemon clarification relay channel + the
plugin's relay selection (the fix for runner-tier clarification cancel)."""

from shared.plugins.clarification.channels import RunnerRPCClarificationChannel
from shared.plugins.clarification.models import (
    Choice,
    ClarificationRequest,
    Question,
    QuestionType,
)
from shared.plugins.clarification.plugin import ClarificationPlugin


def _req():
    return ClarificationRequest(
        context="why",
        questions=[
            Question(
                text="Pick one",
                question_type=QuestionType.SINGLE_CHOICE,
                choices=[Choice(text="A"), Choice(text="B")],
                default_choice=1,
            )
        ],
    )


def test_relay_round_trip_builds_payload_and_parses_answers():
    captured = {}

    def fake_rpc(payload):
        captured["payload"] = payload
        return {"cancelled": False, "answers": ["2"]}

    ch = RunnerRPCClarificationChannel(fake_rpc, agent_id="main")
    resp = ch.request_clarification(_req())

    assert resp.cancelled is False
    assert resp.answers[0].selected_choices == [2]

    p = captured["payload"]
    assert p["agent_id"] == "main"
    assert p["tool_name"] == "request_clarification"
    assert p["context"] == "why"
    assert "request_id" in p and p["request_id"]
    q = p["questions"][0]
    assert q["text"] == "Pick one"
    assert q["question_type"] == "single_choice"
    assert q["choices"][0]["text"] == "A"
    # default_choice=1 → the first choice is flagged default + echoed.
    assert q["choices"][0].get("default") is True
    assert q["default_choice"] == 1


def test_relay_rpc_error_fails_closed_to_cancel():
    def boom(payload):
        raise RuntimeError("RunnerRPCClient is closed")

    ch = RunnerRPCClarificationChannel(boom, agent_id="main")
    resp = ch.request_clarification(_req())
    assert resp.cancelled is True


def test_relay_cancelled_result_propagates():
    ch = RunnerRPCClarificationChannel(
        lambda p: {"cancelled": True, "answers": []}, agent_id="main"
    )
    resp = ch.request_clarification(_req())
    assert resp.cancelled is True


def test_relay_missing_answers_parses_to_fallback_not_crash():
    # Fewer answers than questions → the channel must not index-error.
    ch = RunnerRPCClarificationChannel(
        lambda p: {"cancelled": False, "answers": []}, agent_id="main"
    )
    resp = ch.request_clarification(_req())
    assert resp.cancelled is False
    assert len(resp.answers) == 1  # parsed from "" → default/fallback


class _FakeRPCClient:
    def request_clarification(self, payload):  # pragma: no cover - selection test
        return {"cancelled": False, "answers": []}


class _FakeRegistry:
    def __init__(self, with_rpc):
        self.runner_rpc_client = _FakeRPCClient() if with_rpc else None

    def register_category(self, *args, **kwargs):
        pass


def test_plugin_selects_relay_when_runner_rpc_client_present():
    p = ClarificationPlugin()
    # No registry → no relay.
    assert p._get_runner_rpc_channel() is None
    # Registry with a runner_rpc_client → relay channel.
    p.set_plugin_registry(_FakeRegistry(with_rpc=True))
    ch = p._get_runner_rpc_channel()
    assert isinstance(ch, RunnerRPCClarificationChannel)


def test_plugin_no_relay_when_runner_rpc_client_absent():
    p = ClarificationPlugin()
    p.set_plugin_registry(_FakeRegistry(with_rpc=False))
    assert p._get_runner_rpc_channel() is None
