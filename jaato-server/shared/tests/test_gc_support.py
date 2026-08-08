"""Unit tests for shared.gc_support.

Locks the behavior of the GC helper functions extracted from
JaatoSession: pre-GC span attributes, post-GC span population, and the
removal-list -> instruction-budget synchronization (child removal, bulk
source clear, summary-entry creation, cache-plugin notification).
"""

from __future__ import annotations

from shared.instruction_budget import (
    DEFAULT_SYSTEM_POLICIES,
    GCPolicy,
    InstructionBudget,
    InstructionSource,
    SystemChildType,
)
from shared.plugins.gc import GCRemovalItem, GCResult, GCTriggerReason
from shared.gc_support import (
    apply_gc_removal_list,
    build_gc_span_attributes,
    populate_gc_span_result,
)


def _noop_trace(_msg: str) -> None:
    pass


def _budget(context_limit=1000):
    return InstructionBudget.create_default(
        session_id="s", agent_id="a", agent_type="main", context_limit=context_limit,
    )


def _result(**kw):
    base = dict(
        success=True,
        items_collected=1,
        tokens_before=100,
        tokens_after=40,
        plugin_name="gc_truncate",
        trigger_reason=GCTriggerReason.THRESHOLD,
    )
    base.update(kw)
    return GCResult(**base)


class _FakeSpan:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, k, v):
        self.attrs[k] = v


class TestBuildGcSpanAttributes:
    def test_basic_context_usage(self):
        usage = {"percent_used": 82.5, "total_tokens": 900, "context_limit": 1000}
        attrs = build_gc_span_attributes(usage, budget=None, cache_plugin=None)
        assert attrs["gc.percent_used"] == 82.5
        assert attrs["gc.tokens_total"] == 900
        assert attrs["gc.context_limit"] == 1000
        assert "gc.tokens_before" not in attrs  # no budget

    def test_includes_budget_total(self):
        budget = _budget()
        budget.add_child(
            InstructionSource.SYSTEM, SystemChildType.BASE.value, 120,
            DEFAULT_SYSTEM_POLICIES[SystemChildType.BASE], label="Base",
        )
        budget.update_tokens(InstructionSource.SYSTEM, 120)
        attrs = build_gc_span_attributes({}, budget=budget, cache_plugin=None)
        assert attrs["gc.tokens_before"] == budget.total_tokens()

    def test_includes_cache_anchor(self):
        class _Cache:
            def get_cache_anchor_message_id(self):
                return "msg-42"
        attrs = build_gc_span_attributes({}, budget=None, cache_plugin=_Cache())
        assert attrs["gc.cache_anchor_message_id"] == "msg-42"


class TestPopulateGcSpanResult:
    def test_none_span_is_noop(self):
        # Should not raise
        populate_gc_span_result(None, _result(), on_trace=_noop_trace)

    def test_sets_core_and_detail_attrs(self):
        span = _FakeSpan()
        result = _result(details={"ephemeral_removed": 3, "target_tokens": 500})
        populate_gc_span_result(span, result, on_trace=_noop_trace)
        assert span.attrs["gc.success"] is True
        assert span.attrs["gc.items_collected"] == 1
        assert span.attrs["gc.tokens_freed"] == 60  # 100 - 40
        assert span.attrs["gc.tokens_after"] == 40
        assert span.attrs["gc.ephemeral_removed"] == 3
        assert span.attrs["gc.target_tokens"] == 500


class TestApplyGcRemovalList:
    def test_none_budget_is_noop(self):
        apply_gc_removal_list(
            _result(removal_list=[GCRemovalItem(source=InstructionSource.CONVERSATION)]),
            budget=None, cache_plugin=None, on_trace=_noop_trace,
        )

    def test_removes_specific_child(self):
        budget = _budget()
        budget.add_child(
            InstructionSource.CONVERSATION, "turn_1", 50,
            GCPolicy.EPHEMERAL, label="Turn 1",
        )
        budget.update_tokens(InstructionSource.CONVERSATION, 50)
        result = _result(removal_list=[
            GCRemovalItem(source=InstructionSource.CONVERSATION, child_key="turn_1"),
        ])
        apply_gc_removal_list(
            result, budget=budget, cache_plugin=None, on_trace=_noop_trace,
        )
        conv = budget.get_entry(InstructionSource.CONVERSATION)
        assert "turn_1" not in conv.children

    def test_bulk_clears_source(self):
        budget = _budget()
        budget.add_child(
            InstructionSource.ENRICHMENT, "e1", 30, GCPolicy.EPHEMERAL, label="E1",
        )
        budget.update_tokens(InstructionSource.ENRICHMENT, 30)
        result = _result(removal_list=[
            GCRemovalItem(source=InstructionSource.ENRICHMENT, child_key=None),
        ])
        apply_gc_removal_list(
            result, budget=budget, cache_plugin=None, on_trace=_noop_trace,
        )
        enrichment = budget.get_entry(InstructionSource.ENRICHMENT)
        assert enrichment.tokens == 0
        assert len(enrichment.children) == 0

    def test_creates_summary_entry(self):
        budget = _budget()
        result = _result(
            removal_list=[
                GCRemovalItem(source=InstructionSource.CONVERSATION, child_key="turn_1"),
            ],
            details={"summary_tokens": 200},
        )
        apply_gc_removal_list(
            result, budget=budget, cache_plugin=None, on_trace=_noop_trace,
        )
        conv = budget.get_entry(InstructionSource.CONVERSATION)
        summary_keys = [k for k in conv.children if k.startswith("gc_summary_")]
        assert summary_keys == ["gc_summary_1"]
        assert conv.children["gc_summary_1"].tokens == 200

    def test_notifies_cache_plugin_with_span(self):
        budget = _budget()
        calls = {}

        class _Cache:
            def on_gc_result(self, result, gc_span=None):
                calls["result"] = result
                calls["gc_span"] = gc_span

        span = _FakeSpan()
        result = _result(removal_list=[
            GCRemovalItem(source=InstructionSource.CONVERSATION, child_key="x"),
        ])
        apply_gc_removal_list(
            result, budget=budget, cache_plugin=_Cache(),
            on_trace=_noop_trace, gc_span=span,
        )
        assert calls["result"] is result
        assert calls["gc_span"] is span

    def test_cache_plugin_legacy_signature_fallback(self):
        budget = _budget()
        calls = {}

        class _LegacyCache:
            def on_gc_result(self, result):  # no gc_span kwarg
                calls["result"] = result

        result = _result(removal_list=[
            GCRemovalItem(source=InstructionSource.CONVERSATION, child_key="x"),
        ])
        apply_gc_removal_list(
            result, budget=budget, cache_plugin=_LegacyCache(),
            on_trace=_noop_trace, gc_span=_FakeSpan(),
        )
        assert calls["result"] is result
