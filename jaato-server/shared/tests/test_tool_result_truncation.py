"""Unit tests for shared.tool_result_truncation.

Locks the behavior of the two pure tool-result shrinking transforms
extracted from JaatoSession: proactive cap (before results enter history)
and reactive truncate-to-fit (context-limit recovery).
"""

from __future__ import annotations

from jaato_sdk.plugins.model_provider.types import ToolResult

from shared.tool_result_truncation import (
    TRUNCATION_NOTICE,
    TRUNCATION_PRESERVE_LINES,
    cap_tool_results,
    truncate_results_to_fit,
)


def _noop_trace(_msg: str) -> None:
    pass


def _tr(name, result):
    return ToolResult(call_id=f"c-{name}", name=name, result=result, is_error=False)


def _has_notice(result_str: str) -> bool:
    # The notice template starts with this stable prefix.
    return "[NOTICE: This tool result was automatically truncated" in result_str


class TestCapToolResults:
    def test_small_results_pass_through_unchanged(self):
        results = [_tr("a", "x" * 100), _tr("b", "y" * 100)]
        out = cap_tool_results(
            results, context_limit=100_000, current_total_tokens=0,
            on_trace=_noop_trace,
        )
        # Identity preserved when everything fits.
        assert out is results

    def test_large_result_is_truncated_with_notice(self):
        big = "line\n" * 50_000  # ~250k chars -> ~62k tokens
        results = [_tr("big", big)]
        out = cap_tool_results(
            results, context_limit=8_000, current_total_tokens=0,
            on_trace=_noop_trace,
        )
        assert len(out) == 1
        assert len(str(out[0].result)) < len(big)
        assert _has_notice(str(out[0].result))
        # Identity/metadata preserved.
        assert out[0].call_id == "big" or out[0].name == "big"
        assert out[0].attachments is None

    def test_zero_context_limit_via_no_cap_space(self):
        # current_total already past the 80% target -> cap_tokens floors at 0,
        # so any non-trivial result gets truncated to the preserve floor.
        big = "z" * 20_000
        out = cap_tool_results(
            [_tr("big", big)], context_limit=4_000,
            current_total_tokens=10_000, on_trace=_noop_trace,
        )
        assert _has_notice(str(out[0].result))


class TestTruncateResultsToFit:
    def test_line_based_truncation_keeps_first_lines(self):
        big = "\n".join(f"line-{i}" for i in range(5_000))
        results = [_tr("big", big)]
        out = truncate_results_to_fit(
            results, current_tokens=100_000, limit_tokens=8_000,
            on_trace=_noop_trace,
        )
        body = str(out[0].result)
        assert _has_notice(body)
        # First lines preserved; content beyond the preserve window dropped.
        assert body.startswith("line-0")
        assert f"line-{TRUNCATION_PRESERVE_LINES - 1}" in body
        assert f"line-{TRUNCATION_PRESERVE_LINES + 1}" not in body

    def test_small_results_untouched(self):
        results = [_tr("small", "tiny output")]
        out = truncate_results_to_fit(
            results, current_tokens=100_000, limit_tokens=8_000,
            on_trace=_noop_trace,
        )
        # Below the 200-token skip threshold -> unchanged content.
        assert str(out[0].result) == "tiny output"

    def test_returns_new_list_not_mutating_input(self):
        big = "\n".join(f"row-{i}" for i in range(5_000))
        original = _tr("big", big)
        results = [original]
        out = truncate_results_to_fit(
            results, current_tokens=100_000, limit_tokens=8_000,
            on_trace=_noop_trace,
        )
        # Original object unchanged; a new ToolResult took its place.
        assert original.result == big
        assert out[0] is not original
