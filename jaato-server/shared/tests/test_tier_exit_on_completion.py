"""A tier that is entered, does one completion, and is left again.

`enter_tier` is a MODE SWITCH, so returning requires a deliberate act by
the model in the tier -- routinely the model LEAST able to perform one.
Measured against openai/gpt-audio-mini through OpenRouter, a speaking
tier never handed back on its own across four runs: it said its sentence
and stopped, and the completion nudge -- a safety net for an agent that
forgot to finish -- was the only thing that ever unblocked the return.
In one of those runs the delegate closed the session itself, from the
wrong tier, against its persona.

`exit_on: completion` removes the model from that loop entirely.
"""
import pytest

from shared.model_tiers import (
    EXIT_ON_COMPLETION, EXIT_ON_SWITCH, ModelTierConfig, ModelTierConfigError,
)


def _cfg(**tiers):
    raw = {"initial": "planner", "fallback": "planner",
           "planner": {"model": "text-model"}}
    raw.update(tiers)
    return ModelTierConfig.from_unified_dict(raw)


class TestExitOnDeclaration:
    def test_absent_means_switch(self):
        """Every profile written before this key keeps its behaviour."""
        assert _cfg().tiers["planner"].exit_on == EXIT_ON_SWITCH

    def test_string_sugar_means_switch(self):
        cfg = ModelTierConfig.from_unified_dict(
            {"planner": "m", "initial": "planner", "fallback": "planner"})
        assert cfg.tiers["planner"].exit_on == EXIT_ON_SWITCH

    def test_completion_is_parsed(self):
        cfg = _cfg(executor={"model": "audio", "exit_on": "completion"})
        assert cfg.tiers["executor"].exit_on == EXIT_ON_COMPLETION

    @pytest.mark.parametrize("bad", ["once", "switch_back", "per_request",
                                     "turn", "", "nonsense", 3])
    def test_an_unknown_trigger_is_refused(self, bad):
        """Refused, not defaulted.

        A misspelled `exit_on` silently meaning "stays forever" would
        surface as a session wedged in a specialist tier -- nothing
        errors, the model simply stops -- which is the hardest kind of
        defect to attribute and precisely what this key exists to remove.
        """
        with pytest.raises(ModelTierConfigError):
            _cfg(executor={"model": "audio", "exit_on": bad})

    def test_the_names_we_rejected_are_pointed_somewhere(self):
        """`once`/`switch_back`/`per_request` were all considered."""
        with pytest.raises(ModelTierConfigError) as exc:
            _cfg(executor={"model": "audio", "exit_on": "once"})
        assert EXIT_ON_COMPLETION in str(exc.value)


class TestTheDelegatedTierHandsBackWithoutTheModel:
    """The session half: arm on entry, consume when the work settles."""

    def _session(self, active="executor", pending="planner"):
        from shared.jaato_session import JaatoSession
        s = JaatoSession.__new__(JaatoSession)
        s._active_tier = active
        s._pending_tier_return = pending
        s._trace = lambda *a, **k: None
        s.switch_tier = lambda name: setattr(s, "_active_tier", name)
        from shared.message_queue import MessageQueue
        s._message_queue = MessageQueue()
        return s

    def _response(self, text="", calls=False, media=0):
        class _P:
            def __init__(self, t): self.text, self.function_call = t, None
        class _R:
            parts = []
            media_chunks = 0
            def has_function_calls(self): return calls
        r = _R()
        r.parts = [_P(text)] if text else []
        r.media_chunks = media
        return r

    def test_a_settled_response_returns_to_the_caller(self):
        s = self._session()
        s._exit_completion_tier_if_settled(self._response("said it"))
        assert s._active_tier == "planner"
        assert s._pending_tier_return is None

    def test_a_tier_still_calling_tools_is_not_evicted(self):
        """Not "one provider call": a delegate that legitimately calls a
        tool must keep the wheel until it stops asking for things."""
        s = self._session()
        s._exit_completion_tier_if_settled(self._response("thinking", calls=True))
        assert s._active_tier == "executor"
        assert s._pending_tier_return == "planner"

    def test_nothing_pending_is_a_no_op(self):
        s = self._session(pending=None)
        s._exit_completion_tier_if_settled(self._response("x"))
        assert s._active_tier == "executor"

    def test_the_outcome_is_reported_back(self):
        """Returning the BINDING is not returning CONTROL.

        The delegate's completion settling ends the turn, so the caller
        gets the wheel with no turn to steer -- measured: the manual
        `enter_tier` back disappeared and the NUDGE was still the only
        thing that woke the caller.  Queuing the outcome resumes it
        through the ordinary mid-turn path.
        """
        s = self._session()
        s._exit_completion_tier_if_settled(self._response("The sky is blue."))
        queued = s._message_queue.pop_first_parent_message()
        assert queued is not None, "the caller was given nothing to act on"
        text = queued[0] if isinstance(queued, tuple) else str(queued)
        assert "The sky is blue." in text
        assert "executor" in text

    def test_a_silent_delegate_is_reported_as_silent(self):
        """Model media never enters history, so without this the caller
        could only learn what was said if a transcript happened to
        arrive.  A stated hole beats a silent one."""
        s = self._session()
        s._exit_completion_tier_if_settled(self._response("", media=12))
        queued = s._message_queue.pop_first_parent_message()
        text = queued[0] if isinstance(queued, tuple) else str(queued)
        assert "no text" in text
        assert "12" in text
