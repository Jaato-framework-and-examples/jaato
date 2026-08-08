"""Regression tests for abnormal-finish surfacing.

Before this change an abnormal terminal finish (``MAX_TOKENS`` / ``SAFETY`` /
``ERROR``) was *log-only*: the turn ended looking like a clean completion with
no client-facing signal, so every client had to infer truncation from empty
output (telegram) or showed nothing at all (the a11y console).

The fix threads the terminal ``FinishReason`` two ways:

- **thin A (visible banner):** ``_classify_finish_reason`` emits a
  ``source="system"`` output line for an abnormal stop, mirroring the
  cancellation path — a human-readable "model stopped early" notice.
- **B (structured):** the reason is recorded on ``turn_data['finish_reason']``
  (for EVERY response, including ``STOP``) so it rides
  ``on_agent_turn_completed`` out to ``TurnCompletedEvent.finish_reason`` —
  a machine-readable field a client can branch on WITHOUT string-matching.

These tests pin both behaviors at the classifier — the single source of truth
for the non-parts loop — plus the event/payload contract.
"""

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    Part,
    ProviderResponse,
    TurnOutcome,
)

from shared.jaato_session import JaatoSession


def _response(finish_reason: FinishReason, text: str = "") -> ProviderResponse:
    parts = [Part(text=text)] if text else []
    return ProviderResponse(parts=parts, finish_reason=finish_reason)


def _classify(response, turn_data=None, on_output=None):
    """Call the self-contained classifier on a bare (un-__init__'d) session.

    ``_classify_finish_reason`` touches no instance state beyond the
    ``_abnormal_finish_message`` staticmethod + the module logger, so a
    ``__new__``-constructed instance is a faithful, fast harness.
    """
    session = JaatoSession.__new__(JaatoSession)
    return JaatoSession._classify_finish_reason(
        session, response, turn_data, on_output
    )


# ---------------------------------------------------------------------------
# B — structured finish_reason recorded on turn_data
# ---------------------------------------------------------------------------


class TestFinishReasonRecording:
    def test_max_tokens_records_value_and_returns_abnormal(self):
        turn_data = {}
        result = _classify(_response(FinishReason.MAX_TOKENS, "half a plan"), turn_data)
        assert turn_data["finish_reason"] == "max_tokens"
        assert result is not None
        assert result.outcome == TurnOutcome.MAX_TOKENS

    def test_safety_records_value(self):
        turn_data = {}
        _classify(_response(FinishReason.SAFETY), turn_data)
        assert turn_data["finish_reason"] == "safety"

    def test_error_records_value(self):
        turn_data = {}
        _classify(_response(FinishReason.ERROR), turn_data)
        assert turn_data["finish_reason"] == "error"

    def test_normal_stop_records_stop_and_continues(self):
        # A normal STOP is NOT abnormal: classifier returns None (loop
        # continues / ends normally) but STILL records the true reason so the
        # terminal classification leaves 'stop' on the accounting dict.
        turn_data = {}
        result = _classify(_response(FinishReason.STOP, "done"), turn_data)
        assert result is None
        assert turn_data["finish_reason"] == "stop"

    def test_tool_use_records_value_and_continues(self):
        turn_data = {}
        result = _classify(_response(FinishReason.TOOL_USE), turn_data)
        assert result is None
        assert turn_data["finish_reason"] == "tool_use"

    def test_recording_is_opt_in(self):
        # No turn_data passed -> no crash, classification still works.
        assert _classify(_response(FinishReason.STOP)) is None


# ---------------------------------------------------------------------------
# thin A — visible source="system" banner on abnormal finishes
# ---------------------------------------------------------------------------


class TestAbnormalBanner:
    def _capture(self, finish_reason):
        outputs = []
        _classify(
            _response(finish_reason, "partial"),
            {},
            lambda source, text, mode: outputs.append((source, text, mode)),
        )
        return outputs

    def test_max_tokens_emits_system_banner(self):
        outputs = self._capture(FinishReason.MAX_TOKENS)
        assert len(outputs) == 1
        source, text, mode = outputs[0]
        assert source == "system"
        assert mode == "write"
        assert "max_tokens" in text

    def test_safety_emits_system_banner(self):
        outputs = self._capture(FinishReason.SAFETY)
        assert outputs[0][0] == "system"
        assert "safety" in outputs[0][1]

    def test_normal_stop_emits_no_banner(self):
        outputs = self._capture(FinishReason.STOP)
        assert outputs == []

    def test_tool_use_emits_no_banner(self):
        outputs = self._capture(FinishReason.TOOL_USE)
        assert outputs == []

    def test_banner_is_opt_in(self):
        # No on_output passed -> abnormal still classified, just not surfaced.
        result = _classify(_response(FinishReason.MAX_TOKENS), {})
        assert result is not None


class TestBannerMessages:
    def test_each_abnormal_reason_has_a_distinct_human_message(self):
        msgs = {
            fr: JaatoSession._abnormal_finish_message(fr)
            for fr in (FinishReason.MAX_TOKENS, FinishReason.SAFETY, FinishReason.ERROR)
        }
        assert len(set(msgs.values())) == 3
        for text in msgs.values():
            assert text.startswith("Model stopped early:")

    def test_unknown_reason_falls_back_to_generic_message(self):
        # Never crashes for a reason outside the abnormal trio.
        text = JaatoSession._abnormal_finish_message(FinishReason.UNKNOWN)
        assert "unknown" in text


# ---------------------------------------------------------------------------
# Event contract
# ---------------------------------------------------------------------------


class TestEventContract:
    def test_turn_completed_event_defaults_to_stop(self):
        from jaato_sdk.events import TurnCompletedEvent

        assert TurnCompletedEvent(agent_id="main").finish_reason == "stop"

    def test_turn_completed_event_carries_abnormal_reason(self):
        from jaato_sdk.events import TurnCompletedEvent

        evt = TurnCompletedEvent(agent_id="main", finish_reason="max_tokens")
        assert evt.finish_reason == "max_tokens"
