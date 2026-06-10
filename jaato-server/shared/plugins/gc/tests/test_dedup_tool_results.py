"""Tests for identical tool-result dedup (gc.utils.dedup_identical_tool_results).

Targets the qwen3 ``listAvailableTemplates`` re-call pathology: a model
re-invokes the same tool and gets the byte-identical payload back N times,
bloating the wire with redundant copies that GC eviction can't reclaim
(they sit in the recency-preserved window).  Dedup SHRINKS each earlier
duplicate's body to a marker without removing the message — preserving
recency, structure, and tool_call/tool_result pairing; only exact
duplicates are touched.
"""

from jaato_sdk.plugins.model_provider.types import (
    Message,
    Part,
    Role,
    ToolResult,
)
from shared.plugins.gc.utils import dedup_identical_tool_results


_BIG = {"templates": ["t%d" % i for i in range(200)]}  # ~big, byte-identical
_BIG2 = {"templates": ["other%d" % i for i in range(200)]}


def _result_msg(call_id, name, result):
    return Message(
        role=Role.TOOL,
        parts=[Part.from_function_response(
            ToolResult(call_id=call_id, name=name, result=result)
        )],
    )


def _text(role, txt):
    return Message(role=role, parts=[Part.from_text(txt)])


class TestDedupIdenticalToolResults:

    def test_identical_results_elide_all_but_most_recent(self):
        history = [
            _result_msg("c1", "listAvailableTemplates", _BIG),
            _result_msg("c2", "listAvailableTemplates", _BIG),
            _result_msg("c3", "listAvailableTemplates", _BIG),
        ]
        new_history, freed, _ = dedup_identical_tool_results(history)

        # Same count of messages (shrink, not remove).
        assert len(new_history) == 3
        # The first two are elided; the most recent keeps the full payload.
        assert new_history[0].parts[0].function_response.result.get("_gc_deduped") is True
        assert new_history[1].parts[0].function_response.result.get("_gc_deduped") is True
        assert new_history[2].parts[0].function_response.result == _BIG
        assert freed > 0

    def test_non_identical_results_untouched(self):
        history = [
            _result_msg("c1", "listAvailableTemplates", _BIG),
            _result_msg("c2", "listAvailableTemplates", _BIG2),
        ]
        new_history, freed, _ = dedup_identical_tool_results(history)
        assert new_history is history  # nothing changed → same object
        assert freed == 0

    def test_different_tools_same_payload_not_merged(self):
        # Same payload but DIFFERENT tool name → not duplicates.
        history = [
            _result_msg("c1", "toolA", _BIG),
            _result_msg("c2", "toolB", _BIG),
        ]
        new_history, freed, _ = dedup_identical_tool_results(history)
        assert freed == 0

    def test_small_payloads_below_threshold_skipped(self):
        small = {"ok": True}
        history = [
            _result_msg("c1", "ping", small),
            _result_msg("c2", "ping", small),
        ]
        new_history, freed, _ = dedup_identical_tool_results(history, min_payload_chars=200)
        assert freed == 0
        assert new_history is history

    def test_preserves_structure_and_pairing_fields(self):
        history = [
            _result_msg("c1", "listAvailableTemplates", _BIG),
            _result_msg("c2", "listAvailableTemplates", _BIG),
        ]
        new_history, _, _ = dedup_identical_tool_results(history)
        # message_id preserved; call_id/name preserved; one part each.
        for old, new in zip(history, new_history):
            assert new.message_id == old.message_id
            assert len(new.parts) == 1
            assert new.parts[0].function_response.call_id == old.parts[0].function_response.call_id
            assert new.parts[0].function_response.name == old.parts[0].function_response.name

    def test_unrelated_messages_shared_by_reference(self):
        keep = _text(Role.USER, "hello")
        history = [
            keep,
            _result_msg("c1", "listAvailableTemplates", _BIG),
            _result_msg("c2", "listAvailableTemplates", _BIG),
        ]
        new_history, _, _ = dedup_identical_tool_results(history)
        # The untouched text message is the SAME object (only changed msgs copied).
        assert new_history[0] is keep
        # The elided one is a copy (not the original).
        assert new_history[1] is not history[1]

    def test_key_order_insensitive(self):
        # Equal dicts with different key insertion order must dedup.
        a = {"x": 1, "y": 2, "pad": "z" * 300}
        b = {"y": 2, "x": 1, "pad": "z" * 300}
        history = [
            _result_msg("c1", "t", a),
            _result_msg("c2", "t", b),
        ]
        _, freed, _ = dedup_identical_tool_results(history)
        assert freed > 0

    def test_no_function_response_parts_noop(self):
        history = [_text(Role.USER, "hi"), _text(Role.MODEL, "yo")]
        new_history, freed, _ = dedup_identical_tool_results(history)
        assert new_history is history
        assert freed == 0
