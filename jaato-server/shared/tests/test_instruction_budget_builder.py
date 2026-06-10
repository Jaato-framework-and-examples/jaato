"""Unit tests for shared.instruction_budget_builder.

Locks the behavior of the pure-ish budget-construction helpers extracted
from JaatoSession: token counting (cache/provider/estimate fallback),
the override-aware instruction-text collection, and applying resolved
counts onto an InstructionBudget.
"""

from __future__ import annotations

from shared.instruction_budget import (
    InstructionBudget,
    InstructionSource,
    SystemChildType,
    estimate_tokens,
)
from shared.instruction_token_cache import InstructionTokenCache
from shared.instruction_budget_builder import (
    TokenCountRequest,
    apply_instruction_counts,
    collect_instruction_texts,
    count_tokens,
)


def _noop_trace(_msg: str) -> None:
    pass


class _Provider:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def count_tokens(self, text):
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class TestCountTokens:
    def test_empty_text_is_zero(self):
        cache = InstructionTokenCache()
        assert count_tokens(
            "", cache=cache, provider=None, provider_name="p", on_trace=_noop_trace
        ) == 0

    def test_cache_hit_skips_provider(self):
        cache = InstructionTokenCache()
        cache.put("p", "hello", 123)
        provider = _Provider(999)
        result = count_tokens(
            "hello", cache=cache, provider=provider,
            provider_name="p", on_trace=_noop_trace,
        )
        assert result == 123
        assert provider.calls == 0  # cache short-circuited the provider

    def test_provider_result_is_cached(self):
        cache = InstructionTokenCache()
        provider = _Provider(42)
        result = count_tokens(
            "fresh", cache=cache, provider=provider,
            provider_name="p", on_trace=_noop_trace,
        )
        assert result == 42
        # Second call serves from cache without hitting the provider again.
        again = count_tokens(
            "fresh", cache=cache, provider=provider,
            provider_name="p", on_trace=_noop_trace,
        )
        assert again == 42
        assert provider.calls == 1

    def test_provider_exception_falls_back_to_estimate(self):
        cache = InstructionTokenCache()
        provider = _Provider(RuntimeError("boom"))
        text = "some text " * 10
        result = count_tokens(
            text, cache=cache, provider=provider,
            provider_name="p", on_trace=_noop_trace,
        )
        assert result == estimate_tokens(text)

    def test_no_provider_uses_estimate(self):
        cache = InstructionTokenCache()
        text = "abcd" * 25
        result = count_tokens(
            text, cache=cache, provider=None,
            provider_name="p", on_trace=_noop_trace,
        )
        assert result == estimate_tokens(text)


class TestCollectInstructionTexts:
    def test_override_emits_single_entry(self):
        reqs, deferred = collect_instruction_texts(
            None, "ignored session instructions",
            system_instruction_override="WIRE SYSTEM MESSAGE",
        )
        assert deferred == set()
        assert len(reqs) == 1
        assert reqs[0].child_key == SystemChildType.OVERRIDE.value
        assert reqs[0].text == "WIRE SYSTEM MESSAGE"
        assert reqs[0].source == InstructionSource.SYSTEM

    def test_empty_override_emits_nothing(self):
        reqs, deferred = collect_instruction_texts(
            None, "ignored", system_instruction_override="",
        )
        assert reqs == []
        assert deferred == set()


class TestApplyInstructionCounts:
    def _budget(self):
        return InstructionBudget.create_default(
            session_id="s", agent_id="a", agent_type="main", context_limit=1000,
        )

    def test_totals_and_children_built(self):
        budget = self._budget()
        from shared.instruction_budget import DEFAULT_SYSTEM_POLICIES
        reqs = [
            TokenCountRequest(
                text="x", source=InstructionSource.SYSTEM,
                child_key=SystemChildType.BASE.value,
                gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.BASE],
                label="Base", token_count=100,
            ),
            TokenCountRequest(
                text="y", source=InstructionSource.SYSTEM,
                child_key=SystemChildType.FRAMEWORK.value,
                gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.FRAMEWORK],
                label="Framework", token_count=50,
            ),
        ]
        apply_instruction_counts(budget, reqs, 1000, on_trace=_noop_trace)

        system_entry = budget.get_entry(InstructionSource.SYSTEM)
        assert system_entry.tokens == 150
        assert SystemChildType.BASE.value in system_entry.children
        assert SystemChildType.FRAMEWORK.value in system_entry.children
        # ENRICHMENT / CONVERSATION reset to 0
        assert budget.get_entry(InstructionSource.CONVERSATION).tokens == 0

    def test_phase2_updates_existing_child_in_place(self):
        budget = self._budget()
        from shared.instruction_budget import DEFAULT_SYSTEM_POLICIES
        base = TokenCountRequest(
            text="x", source=InstructionSource.SYSTEM,
            child_key=SystemChildType.BASE.value,
            gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.BASE],
            label="Base", token_count=100, is_estimate=True,
        )
        apply_instruction_counts(budget, [base], 1000, on_trace=_noop_trace)
        # Phase 2: refined count for the same child_key.
        base.token_count = 250
        base.is_estimate = False
        apply_instruction_counts(budget, [base], 1000, on_trace=_noop_trace)

        system_entry = budget.get_entry(InstructionSource.SYSTEM)
        assert system_entry.children[SystemChildType.BASE.value].tokens == 250
        assert system_entry.tokens == 250
