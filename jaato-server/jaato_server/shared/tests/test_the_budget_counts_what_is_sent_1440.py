"""The instruction budget counts what is sent (#1440).

A web-coder session on a 1M model reported 513k tracked by the
``InstructionBudget`` while the provider reported an 861.8k prompt.
``gc_budget`` decides on the budget's figure, so it freed nothing every
turn while the context climbed to the window.  The budget counted text,
tool results and media, and missed:

* tool-call ARGUMENTS -- a notebook cell's code, a whole file handed to
  ``writeNewFile``;
* replayed REASONING -- ``Part.thought`` a ``replay_reasoning`` provider
  sends back as ``reasoning_content`` on every request.

The properties pinned here:

A. the budget's conversation figure is within a tolerance of the size of
   the request the real OpenAI-shaped converter builds from the history;
B. ``gc_budget`` collects when that real size is over its threshold;
C. when the provider reports a prompt larger than the budget beyond the
   margin, the GC threshold is judged on the larger figure, a GC pass frees
   in estimate units (not ``factor`` times too much), and the drift is
   logged once per session;
D. a message whose parts change is recounted, never kept at its old figure;
E. the per-part rule is shared: ``estimate_message_tokens`` and the budget
   count the same strings.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    TokenUsage,
    ToolResult,
)
from jaato_server.shared.instruction_budget import InstructionBudget, InstructionSource
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.gc import GCConfig, GCTriggerReason
from jaato_server.shared.plugins.gc.utils import estimate_message_tokens, message_wire_texts
from jaato_server.shared.plugins.gc_budget.plugin import create_plugin as create_gc_budget
from jaato_server.shared.plugins.model_provider._openai_compat.converters import (
    message_to_openai,
)
from jaato_server.shared.tests.reversion import Reversion

_UTILS = "jaato-server/jaato_server/shared/plugins/gc/utils.py"
_BUDGET = "jaato-server/jaato_server/shared/instruction_budget.py"

REVERSIONS = [
    Reversion(
        target=_UTILS,
        find=(
            "    if part.function_call:\n"
            "        texts.append(function_call_wire_text(part.function_call))\n"
        ),
        replace="",
        because="tool-call arguments are what the model sent; the wire carries them",
        test="test_the_budget_matches_the_request_the_converter_builds",
    ),
    Reversion(
        target=_UTILS,
        find="    if include_thought and part.thought:\n        texts.append(part.thought)\n",
        replace="",
        because="a replaying wire sends reasoning back on every request",
        test="test_replayed_reasoning_is_counted_on_a_replaying_wire",
    ),
    Reversion(
        target=_UTILS,
        find="    if include_thought and part.thought:\n        texts.append(part.thought)\n",
        replace="",
        because="gc_budget must see the real size to collect",
        test="test_gc_budget_collects_when_the_real_size_is_over_threshold",
    ),
    Reversion(
        target=_BUDGET,
        find="            ratio if ratio > 1.0 + self.CALIBRATION_MARGIN else 1.0)",
        replace="            1.0)",
        because="a denominator known to be short must not decide on its own",
        test="test_a_larger_provider_prompt_decides_the_threshold",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/gc_budget/plugin.py",
        find="        current_tokens = budget.effective_total_tokens()\n",
        replace="        current_tokens = budget.total_tokens()\n",
        because="collect must judge the same total should_collect did",
        test="test_a_larger_provider_prompt_decides_the_threshold",
    ),
]

_LIMIT = 100_000


def _session(*, replay: bool = True, limit: int = _LIMIT) -> JaatoSession:
    """A session with a real budget, a replaying stub provider and chars/4."""
    s = JaatoSession(MagicMock(), "m")
    s._provider = SimpleNamespace(replay_reasoning=True if replay else False)
    s._count_tokens = lambda text: max(1, len(text) // 4)
    s._instruction_budget = InstructionBudget.create_default(
        session_id="s", context_limit=limit)
    s._ui_hooks = None
    return s


def _cell(i: int) -> str:
    """A notebook-cell-sized argument: ~6 KB of code."""
    return "\n".join(
        f"df_{i}_{j} = frame.groupby('k{j}').agg({{'v': 'sum'}})  # step {j}"
        for j in range(90)
    )


def _history(turns: int) -> list:
    """USER -> MODEL(thought + big tool call) -> TOOL(small result), repeated."""
    history = []
    for i in range(turns):
        history.append(Message(role=Role.USER, parts=[Part(text=f"do step {i}")]))
        history.append(Message(role=Role.MODEL, parts=[
            Part(thought="Reasoning about the frame. " * 120),
            Part(function_call=FunctionCall(
                id=f"c{i}", name="notebook_execute", args={"code": _cell(i)})),
        ]))
        history.append(Message(role=Role.TOOL, parts=[
            Part(function_response=ToolResult(
                call_id=f"c{i}", name="notebook_execute", result="ok")),
        ]))
    return history


def _wire_chars(history: list) -> int:
    """Characters of content the real OpenAI-shaped converter puts on the wire."""
    total = 0
    for msg in history:
        for d in message_to_openai(msg, lambda t: {"reasoning_content": t}):
            total += len(d.get("content") or "")
            total += len(d.get("reasoning_content") or "")
            for tc in d.get("tool_calls") or []:
                total += len(tc["function"]["name"]) + len(tc["function"]["arguments"])
    return total


def _conversation_tokens(s: JaatoSession) -> int:
    return s._instruction_budget.get_entry(InstructionSource.CONVERSATION).total_tokens()


def test_the_budget_matches_the_request_the_converter_builds():
    s = _session()
    history = _history(6)
    s._history.replace(history)
    s._update_conversation_budget()
    wire_tokens = _wire_chars(history) // 4
    budget = _conversation_tokens(s)
    assert wire_tokens > 10_000
    assert 0.85 * wire_tokens <= budget <= 1.15 * wire_tokens, (budget, wire_tokens)


def test_replayed_reasoning_is_counted_on_a_replaying_wire():
    history = [Message(role=Role.MODEL, parts=[
        Part(thought="x" * 40_000), Part(text="done"),
    ])]
    replaying = _session(replay=True)
    replaying._history.replace(history)
    replaying._update_conversation_budget()
    assert _conversation_tokens(replaying) >= 10_000

    # A wire that does not replay drops thought parts; the budget must not
    # charge for bytes that are never sent.
    silent = _session(replay=False)
    silent._history.replace(history)
    silent._update_conversation_budget()
    assert _conversation_tokens(silent) < 100


def test_gc_budget_collects_when_the_real_size_is_over_threshold():
    limit = 30_000
    s = _session(limit=limit)
    history = _history(14)
    s._history.replace(history)
    s._update_conversation_budget()
    # The real request is over the threshold of this window.
    assert _wire_chars(history) // 4 > 0.8 * limit

    plugin = create_gc_budget()
    plugin.initialize({"preserve_recent_turns": 2})
    config = GCConfig(threshold_percent=80.0, target_percent=60.0)
    usage = s.get_context_usage()
    should, reason = plugin.should_collect(usage, config)
    assert should, usage
    new_history, result = plugin.collect(
        history, usage, config, reason, budget=s._instruction_budget)
    assert result.items_collected > 0
    assert len(new_history) < len(history)


def test_a_larger_provider_prompt_decides_the_threshold(caplog):
    s = _session()
    history = [Message(role=Role.USER, parts=[Part(text=f"m{i} " + "w" * 4000)])
               for i in range(45)]
    s._history.replace(history)
    s._update_conversation_budget()
    estimate = s._instruction_budget.total_tokens()
    assert estimate < 0.5 * _LIMIT

    # The provider says the request was nearly twice as large, most of it
    # served from cache.
    reported = int(estimate * 1.9)
    usage = TokenUsage(prompt_tokens=1000, cache_read_tokens=reported - 1000,
                       output_tokens=50)
    with caplog.at_level(logging.WARNING):
        # Each response measures the request built for it (#1514).
        s._history_for_provider()
        s._calibrate_budget_against_provider(usage)
        s._history_for_provider()
        s._calibrate_budget_against_provider(usage)
    drift = [r for r in caplog.records if "#1440" in r.getMessage()]
    assert len(drift) == 1
    assert str(estimate) in drift[0].getMessage()
    assert str(reported) in drift[0].getMessage()

    context = s.get_context_usage()
    assert context["tokens_source"] == "calibrated"
    assert context["provider_prompt_tokens"] == reported
    assert context["total_tokens"] >= reported - 1
    assert context["percent_used"] >= 80.0

    plugin = create_gc_budget()
    plugin.initialize({"preserve_recent_turns": 1})
    config = GCConfig(threshold_percent=80.0, target_percent=60.0)
    should, reason = plugin.should_collect(context, config)
    assert should
    _, result = plugin.collect(history, context, config, reason,
                               budget=s._instruction_budget)
    freed = result.details["tokens_freed"]
    # Enough to reach the target on the real scale, and not ``factor``
    # times more: freeing (reported - target) estimate tokens would take
    # almost everything.
    target = int(_LIMIT * 0.6)
    needed_estimate = (reported - target) / 1.9
    assert freed >= needed_estimate
    assert freed <= needed_estimate + 1500


def test_an_unreported_usage_is_not_a_measurement():
    s = _session()
    s._history.replace([Message(role=Role.USER, parts=[Part(text="w" * 4000)])])
    s._update_conversation_budget()
    usage = TokenUsage(prompt_tokens=90_000)
    usage.reported = False
    s._calibrate_budget_against_provider(usage)
    assert s.get_context_usage()["tokens_source"] == "estimate"
    assert s._instruction_budget.provider_prompt_tokens is None


def test_a_message_that_changes_is_recounted():
    s = _session()
    msg = Message(role=Role.MODEL, parts=[Part(text="short")])
    s._history.replace([msg])
    s._update_conversation_budget()
    before = _conversation_tokens(s)
    msg.parts.append(Part(function_call=FunctionCall(
        id="c", name="writeNewFile", args={"content": "y" * 8000})))
    s._update_conversation_budget()
    assert _conversation_tokens(s) >= before + 1900


def test_the_budget_and_the_gc_estimate_count_the_same_strings():
    msg = _history(1)[1]
    texts = message_wire_texts(msg)
    assert any("notebook_execute" in t and "groupby" in t for t in texts)
    assert any(t.startswith("Reasoning") for t in texts)
    assert estimate_message_tokens(msg) == sum(len(t) for t in texts) // 4
    fc = msg.parts[1].function_call
    assert json.dumps(fc.args) in "".join(texts)
