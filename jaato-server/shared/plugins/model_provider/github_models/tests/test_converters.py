"""Converter pin tests for the github_models provider.

Focus: a TOOL message carrying N parallel ``function_response`` parts must
produce N SDK ``ToolMessage``s — one per ``tool_call_id`` — not just the
first.  Regression guard for the parallel-tool-result truncation bug
(2026-06-12: build_descriptor fired 7 parallel ``call_service`` lookups; only
result #1 reached the model because ``message_to_sdk`` returned
``function_responses[0]``).
"""

from __future__ import annotations

import pytest

from jaato_sdk.plugins.model_provider.types import Message, Part, Role, ToolResult

# The github_models converter builds azure-ai-inference SDK message objects;
# skip cleanly if that SDK isn't importable in this environment.
azure = pytest.importorskip("azure.ai.inference")  # noqa: F841
from shared.plugins.model_provider.github_models.converters import (  # noqa: E402
    message_to_sdk,
)


def test_parallel_tool_results_all_reach_wire():
    trs = [
        ToolResult(call_id=f"call_{i}", name="call_service",
                   result={"docs": [{"a": f"artifact-{i}"}]})
        for i in range(7)
    ]
    msg = Message(role=Role.TOOL,
                  parts=[Part(function_response=tr) for tr in trs])
    out = message_to_sdk(msg)
    assert len(out) == 7, "all 7 parallel results must reach the wire"
    assert [m.tool_call_id for m in out] == [f"call_{i}" for i in range(7)]


def test_single_tool_result_one_message():
    tr = ToolResult(call_id="call_1", name="read_file", result={"x": 1})
    msg = Message(role=Role.TOOL, parts=[Part(function_response=tr)])
    out = message_to_sdk(msg)
    assert len(out) == 1
    assert out[0].tool_call_id == "call_1"


def test_non_tool_message_is_single_element_list():
    out = message_to_sdk(Message.from_text(Role.USER, "hello"))
    assert len(out) == 1
