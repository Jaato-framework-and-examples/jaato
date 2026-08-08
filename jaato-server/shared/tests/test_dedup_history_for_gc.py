"""Session-level wiring of identical tool-result dedup (GC Phase 0).

Verifies _dedup_history_for_gc():
- shrinks the duplicate tool-results in history,
- invalidates the per-message token cache ONLY for the shrunk messages
  (so the budget re-sync recomputes their smaller size; the survivor's
  cache entry is left intact),
- triggers a CONVERSATION budget re-sync,
- is a no-op when disabled or when nothing duplicates.
"""

from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import (
    Message,
    Part,
    Role,
    ToolResult,
)
from shared.tests.test_two_phase_budget import _make_session


_BIG = {"templates": ["t%d" % i for i in range(200)]}


def _result_msg(mid, call_id, name, result):
    return Message(
        role=Role.TOOL,
        message_id=mid,
        parts=[Part.from_function_response(
            ToolResult(call_id=call_id, name=name, result=result)
        )],
    )


def _prep(session, gc_config, history):
    session._instruction_budget = MagicMock()  # truthy gate only
    session._gc_config = gc_config
    session._msg_token_cache = {m.message_id: 9999 for m in history}
    session._trace = lambda *a, **k: None
    session._emit_instruction_budget_update = lambda: None
    session._update_conversation_budget = MagicMock()
    session._history.replace(history)


class _Cfg:
    """Minimal GC config stand-in."""
    def __init__(self, dedup=True):
        if dedup is not None:
            self.dedup_identical_tool_results = dedup


class TestDedupHistoryForGC:

    def test_shrinks_and_invalidates_only_elided_cache(self):
        s = _make_session()
        history = [
            _result_msg("m1", "c1", "listAvailableTemplates", _BIG),
            _result_msg("m2", "c2", "listAvailableTemplates", _BIG),
            _result_msg("m3", "c3", "listAvailableTemplates", _BIG),
        ]
        _prep(s, _Cfg(dedup=True), history)

        freed = s._dedup_history_for_gc()

        assert freed > 0
        # Earlier duplicates' cache entries invalidated; survivor kept.
        assert "m1" not in s._msg_token_cache
        assert "m2" not in s._msg_token_cache
        assert "m3" in s._msg_token_cache
        # History shrunk in place (survivor keeps full payload).
        hist = s.get_history()
        assert hist[0].parts[0].function_response.result.get("_gc_deduped") is True
        assert hist[2].parts[0].function_response.result == _BIG
        # Budget re-sync triggered.
        s._update_conversation_budget.assert_called_once()

    def test_noop_when_disabled(self):
        s = _make_session()
        history = [
            _result_msg("m1", "c1", "listAvailableTemplates", _BIG),
            _result_msg("m2", "c2", "listAvailableTemplates", _BIG),
        ]
        _prep(s, _Cfg(dedup=False), history)

        assert s._dedup_history_for_gc() == 0
        # Nothing invalidated, no re-sync.
        assert "m1" in s._msg_token_cache
        s._update_conversation_budget.assert_not_called()

    def test_noop_when_no_duplicates(self):
        s = _make_session()
        history = [
            _result_msg("m1", "c1", "listAvailableTemplates", _BIG),
            _result_msg("m2", "c2", "listAvailableTemplates",
                        {"templates": ["other%d" % i for i in range(200)]}),
        ]
        _prep(s, _Cfg(dedup=True), history)

        assert s._dedup_history_for_gc() == 0
        s._update_conversation_budget.assert_not_called()

    def test_noop_without_budget(self):
        s = _make_session()
        history = [_result_msg("m1", "c1", "t", _BIG)]
        _prep(s, _Cfg(dedup=True), history)
        s._instruction_budget = None

        assert s._dedup_history_for_gc() == 0
