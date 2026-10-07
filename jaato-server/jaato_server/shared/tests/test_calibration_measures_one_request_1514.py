"""A calibration factor is measured on one request, not across two (#1514).

The #1440 calibration compares the provider's prompt size with the
budget's estimate and, beyond a margin, scales the estimate by the ratio.
On the tool-results path the budget is refreshed BEFORE the results are
appended, so the estimate it held when the provider answered described
the request without them.  A session (MiniMax-M3, 868,928-token input
limit) received one ~228k-token tool result: the provider reported
384,966 for the request carrying it against an estimate of 161,748, the
factor came out 2.38, and the next estimate (389,423, which already
counted the result) was scaled to 106.6% -- GC fired at ~45%.

These tests drive the real ``JaatoSession._do_send_tool_results`` with a
stub provider that reports a prompt size consistent with what it was
actually handed (``k`` times its chars/4 count), and pin:

A. a large tool result between two sends is not estimation error: the
   factor stays ~k, utilization after it is ~e2*k/limit, GC does not fire;
B. a genuinely short estimate (k > 1 on unchanged content) is still
   learned;
C. a response with no usage changes nothing, and its request's estimate
   is not paired with a later figure;
D. one estimate measures one response.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolResult,
    TurnResult,
)
from jaato_server.shared.instruction_budget import InstructionBudget
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.gc import GCConfig
from jaato_server.shared.plugins.gc.utils import message_wire_texts
from jaato_server.shared.plugins.gc_budget.plugin import create_plugin as create_gc_budget
from jaato_server.shared.tests.reversion import Reversion

_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"
_BUDGET = "jaato-server/jaato_server/shared/instruction_budget.py"

REVERSIONS = [
    Reversion(
        target=_BUDGET,
        find="        estimate = self.sent_estimate_tokens\n",
        replace="        estimate = self.total_tokens()\n",
        because="the budget's total when the answer arrives is not the request it answers",
        test="test_a_large_tool_result_is_not_estimation_error",
    ),
    Reversion(
        target=_SESSION,
        find="        self._note_send_estimate(wire)\n",
        replace="",
        because="without an estimate of the request sent, nothing is measured",
        test="test_a_short_estimate_is_still_learned",
    ),
    Reversion(
        target=_SESSION,
        find=(
            "            # with a later response's figure (#1514).\n"
            "            budget.discard_sent_estimate()\n"
        ),
        replace="            # with a later response's figure (#1514).\n",
        because="an answered request's estimate must not meet a later figure",
        test="test_a_response_without_usage_changes_nothing",
    ),
    Reversion(
        target=_BUDGET,
        find=(
            "        estimate = self.sent_estimate_tokens\n"
            "        self.sent_estimate_tokens = None\n"
        ),
        replace="        estimate = self.sent_estimate_tokens\n",
        because="one estimate measures one response",
        test="test_one_estimate_measures_one_response",
    ),
]

_LIMIT = 100_000


class _StubProvider:
    """Reports a prompt of ``k`` times the chars/4 size of what it was handed."""

    replay_reasoning = False
    name = "stub"

    def __init__(self, k: float, *, report: bool = True):
        self.k = k
        self.report = report
        self.sent: list = []

    def complete(self, messages, **_kw):
        size = sum(max(1, len(t) // 4)
                   for m in messages for t in message_wire_texts(m))
        self.sent.append(size)
        usage = TokenUsage(prompt_tokens=int(size * self.k), output_tokens=5)
        if not self.report:
            usage = TokenUsage()
            usage.reported = False
        return TurnResult.from_provider_response(ProviderResponse(
            parts=[Part(text="ok")], usage=usage,
            finish_reason=FinishReason.STOP))

    def get_max_output_tokens(self):
        return None

    def supports_streaming(self):
        return False


def _session(k: float, *, report: bool = True) -> JaatoSession:
    s = JaatoSession(MagicMock(), "m")
    s._provider = _StubProvider(k, report=report)
    s._count_tokens = lambda text: max(1, len(text) // 4)
    s._instruction_budget = InstructionBudget.create_default(
        session_id="s", context_limit=_LIMIT)
    s._ui_hooks = None
    s._history.replace([Message.from_text(Role.USER, "start " * 1200)])
    return s


def _call(s: JaatoSession, call_id: str) -> None:
    s._history.append(Message(role=Role.MODEL, parts=[Part(
        function_call=FunctionCall(id=call_id, name="listReferences", args={}))]))


def _send(s: JaatoSession, call_id: str, result: str) -> None:
    """One intra-turn round, in the order the chat loop runs it."""
    _call(s, call_id)
    # Steps 2.5 / 2.7: the budget is refreshed BEFORE the results are
    # appended -- the ordering that made the old comparison stale.
    s._update_conversation_budget()
    s._do_send_tool_results(
        [ToolResult(call_id=call_id, name="listReferences", result=result)],
        False, None, None, {})


def _usage_after_refresh(s: JaatoSession) -> dict:
    s._update_conversation_budget()  # the next before_send
    return s.get_context_usage()


def test_a_large_tool_result_is_not_estimation_error():
    s = _session(1.0)
    _send(s, "c1", "small")
    assert s._instruction_budget.calibration_factor == 1.0

    _send(s, "c2", "r" * 180_000)  # ~45k tokens joins between the sends
    budget = s._instruction_budget
    assert budget.calibration_factor == 1.0, budget.calibration_factor

    usage = _usage_after_refresh(s)
    e2 = budget.total_tokens()
    assert e2 > 40_000
    assert abs(usage["percent_used"] - 100.0 * e2 / _LIMIT) < 0.5
    assert usage["percent_used"] < 60.0

    plugin = create_gc_budget()
    plugin.initialize({"preserve_recent_turns": 1})
    should, _ = plugin.should_collect(
        usage, GCConfig(threshold_percent=80.0, target_percent=60.0))
    assert not should, usage


def test_a_short_estimate_is_still_learned():
    s = _session(1.5)
    _send(s, "c1", "small")
    budget = s._instruction_budget
    assert 1.45 <= budget.calibration_factor <= 1.55, budget.calibration_factor

    # The factor survives a large result and stays ~k, not k * e2 / e1.
    _send(s, "c2", "r" * 120_000)
    assert 1.45 <= budget.calibration_factor <= 1.55, budget.calibration_factor
    usage = _usage_after_refresh(s)
    expected = 100.0 * budget.total_tokens() * budget.calibration_factor / _LIMIT
    assert abs(usage["percent_used"] - expected) < 0.5
    assert usage["tokens_source"] == "calibrated"


def test_a_response_without_usage_changes_nothing():
    s = _session(1.5)
    _send(s, "c1", "small")
    budget = s._instruction_budget
    learned = budget.calibration_factor
    assert learned > 1.0

    s._provider.report = False
    _send(s, "c2", "r" * 120_000)
    assert budget.calibration_factor == learned
    assert budget.sent_estimate_tokens is None

    # A figure that arrives with no request sized for it measures nothing.
    s._calibrate_budget_against_provider(TokenUsage(prompt_tokens=90_000))
    assert budget.calibration_factor == learned


def test_one_estimate_measures_one_response():
    s = _session(1.0)
    _send(s, "c1", "small")
    budget = s._instruction_budget
    assert budget.calibration_factor == 1.0
    # A second figure (a duplicate usage, a stray late report) has no
    # request of its own and must not re-measure the first one's estimate.
    s._calibrate_budget_against_provider(TokenUsage(prompt_tokens=90_000))
    assert budget.calibration_factor == 1.0
    assert budget.provider_prompt_tokens == 90_000
