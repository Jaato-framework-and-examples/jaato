"""A tier declared ``exit_on: completion`` leaves, however its turn ended.

THE CONTRACT (`switch_tier`'s own comment): such a tier is entered, does
one completion, and is left again, "with the model doing nothing to
return".  That matters because the model in a specialist tier is
routinely the one LEAST able to hand back -- a speaking tier measured
over four runs never returned on its own.

THE DEFECT (#1025).  The exit was evaluated at exactly ONE point --
``_exit_completion_tier_if_settled``, called on the tool-continuation
slot of ``_run_chat_loop`` -- and that point sits AFTER two earlier
returns from the same function::

    response, abnormal = self._finish_or_continue(...)   # #749
    if abnormal is not None:
        return None, abnormal, False                    # <-- never reached
    response = self._nudge_for_tool_use(...)
    self._exit_completion_tier_if_settled(response)      # the only check

Arming is deterministic (`switch_tier`, every entry from a different
tier).  The pop was not, and a tier that failed to pop LINGERED into
later turns -- which for a speech tier compounds: an output-only tier
cannot ingest the next turn's audio, so #847's modality gate withholds
it and the user gets text where they asked for voice.  Measured live:
the pop fired on 2 of 3 consecutive voice turns.

WHICH BYPASSES ARE REAL.  The issue names three candidates and could not
say which fired.  Driven against the real methods (see the probe in each
test below), the answer is:

  * **the abnormal finish -- YES, and it is the one.**  ``SAFETY``,
    ``MAX_TOKENS`` and ``ERROR`` each return before the check, leaving
    the tier armed and resident.  A spoken answer is expensive in output
    tokens, so ``MAX_TOKENS`` is the routine shape here.
  * **the `tool_use` nudge -- NO.**  The check sits AFTER
    ``_nudge_for_tool_use``, so the nudge cannot skip it; both
    ``TOOL_USE``-without-calls and empty-``UNKNOWN`` pop correctly.  The
    issue's reading is wrong on this one.
  * **the `has_function_calls()` early-return -- NOT ON ITS OWN.**  It is
    a deliberate hold (a delegate that legitimately calls a tool must
    keep the wheel), and the loop re-evaluates on the next continuation.
    It leaks only in combination with a terminal path that never
    re-evaluates.

Two bypasses the issue does NOT name are also real, and one of them is
the voice path itself:

  * **``signal_completion`` terminating the turn**, which returns ~70
    lines above the check.
  * **``_run_chat_loop_with_parts``** -- the attachment-carrying loop, the
    one a spoken question is routed to -- which evaluated the exit
    NOWHERE.

THE FIX, and what "terminal" means in it.  The mid-turn check stays,
because it is the only point at which there is still a turn to resume the
caller into.  The GUARANTEE is a second evaluation in the ``finally`` of
both loops: the framework has stopped making provider calls on the
delegate's behalf, so nothing is outstanding and a pop cannot strand
work.  That is NOT "a turn boundary is a terminus" (#767) -- the tier is
evicted because the turn it was delegated FOR is over, not because a
boundary arrived.  ``_take_pending_tier_return`` is the sole clearing
site, so whichever point gets there first spends the arming and a double
pop is unrepresentable.
"""

import ast
import base64
import pathlib
import logging
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Part,
    ProviderResponse,
    TokenUsage,
    ToolSchema,
    TurnResult,
)

from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.model_tiers import ModelTierConfig
from jaato_server.shared.tests.reversion import Reversion


SESSION_PATH = pathlib.Path(__file__).resolve().parents[1] / "jaato_session.py"


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/jaato_session.py",
        find="""            self._finalize_completion_tier_exit(
                turn_data.get('finish_reason'), reason="turn end")""",
        replace="""            pass  # no turn-end completion-tier exit""",
        test="test_an_abnormal_finish_still_returns_the_delegated_tier",
        because="a delegated tier whose turn ended abnormally staying "
                "resident into every later turn -- #1025 exactly, and for "
                "a speech tier that also deafens the session (#847)",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/jaato_session.py",
        find="""            self._finalize_completion_tier_exit(
                turn_data.get('finish_reason'), reason="parts turn end")""",
        replace="""            pass  # no turn-end completion-tier exit""",
        test="test_the_attachment_loop_returns_the_delegated_tier_too",
        because="a tier delegated from a spoken question never popping at "
                "all, because the attachment loop evaluates the exit "
                "nowhere",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/jaato_session.py",
        find="""        target = getattr(self, "_pending_tier_return", None)
        if target is not None:
            self._pending_tier_return = None
        return target""",
        replace="""        return getattr(self, "_pending_tier_return", None)""",
        test="test_the_arming_is_spent_so_the_exit_cannot_fire_twice",
        because="the arming outliving the pop it caused, so a second "
                "evaluation point returns PAST the tier that armed it",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/jaato_session.py",
        find="""        if response.has_function_calls():
            logger.info(
                "Completion-tier exit HELD: %s has not settled (its "
                "response carries function calls); still armed to return "
                "to %s", getattr(self, "_active_tier", None), target,
            )
            return                      # still working; it has not settled""",
        replace="""        if False:
            return""",
        test="test_a_delegate_still_calling_tools_keeps_the_wheel",
        because="a delegate being evicted mid-tool-loop, handing control "
                "back with its own work outstanding",
    ),
]


SPEAK = ToolSchema(
    name="speak",
    description="Say something out loud.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}},
                "required": ["text"]},
)


TIERS = {
    "initial": "executor",
    "fallback": "executor",
    "executor": {"model": "text-model"},
    "voz": {"model": "audio-model", "exit_on": "completion",
            "description": "Speaks the reply aloud."},
}


# ==================== harness =========================================


def _live_session(provider):
    """A session driven through the REAL chat loops.

    Same shape as ``test_a_truncated_turn_is_continued_not_lost``'s
    fixture: everything below the session is a mock, the session itself
    and both of its loops are the shipped code.  A unit test of
    ``_exit_completion_tier_if_settled`` alone cannot see this defect --
    that method was always correct; what was wrong was where it was
    called from -- so the assertions here have to come out the far side
    of ``send_message``.
    """
    runtime = MagicMock()
    runtime.create_provider.return_value = provider
    runtime.get_tool_schemas.return_value = [SPEAK]
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.ledger = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = [SPEAK]
    runtime.registry.enrich_prompt.side_effect = (
        lambda prompt, **_k: MagicMock(prompt=prompt, metadata={})
    )
    session = JaatoSession(runtime, "text-model")
    session.configure()
    session._tier_config = ModelTierConfig.from_unified_dict(dict(TIERS))
    session._active_tier = "executor"
    return session


def _continuation_slot(session, response):
    """Drive the shipped tool-continuation slot with *response*.

    ``_execute_tools_and_continue`` is the method the issue's root-cause
    reading is about, and its steps 4-7 -- cancellation,
    ``_finish_or_continue``, ``_nudge_for_tool_use``, the exit check --
    are the shipped code with nothing stubbed.  Only the two ends are
    replaced: the tool batch (empty) and the provider round trip that
    produces the continuation, which is what lets a test name the finish
    reason the upstream reported.
    """
    session._execute_function_call_group = lambda *a, **k: []
    session._send_tool_results_and_continue = lambda *a, **k: response
    session._check_and_handle_mid_turn_prompt = lambda *a, **k: None
    return session._execute_tools_and_continue(
        [], False, None, None, {}, False, None, context="probe")


def _delegated(session):
    """Enter the completion tier through the real arming path.

    ``switch_tier`` is what stamps ``_pending_tier_return``, so driving
    it here means the test arms the way ``enter_tier`` does rather than
    writing the attribute itself.
    """
    session.switch_tier("voz")
    assert session._active_tier == "voz"
    assert session._pending_tier_return == "executor", (
        "the arming is deterministic; if this fails the test is not "
        "measuring the pop"
    )
    return session


def _response(text="", finish=FinishReason.STOP, calls=(), media=0):
    parts = [Part(text=text)] if text else []
    parts += [Part(function_call=fc) for fc in calls]
    return ProviderResponse(
        parts=parts,
        finish_reason=finish,
        media_chunks=media,
        usage=TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15),
    )


def _scripted_provider(responses, hard_stop=12):
    calls = []

    def complete(messages, **_kwargs):
        calls.append(len(calls))
        if len(calls) > hard_stop:
            raise AssertionError(
                f"{len(calls)} requests in one turn: the loop is not "
                f"terminating"
            )
        return TurnResult.from_provider_response(
            responses[min(len(calls) - 1, len(responses) - 1)]
        )

    provider = MagicMock()
    provider.name = "fake"
    provider.supports_streaming.return_value = False
    provider.get_context_limit.return_value = 0
    provider.complete.side_effect = complete
    provider._calls = calls
    return provider


# ==================== the abnormal finish (bypass 1) ==================


@pytest.mark.parametrize("finish", [
    FinishReason.SAFETY, FinishReason.MAX_TOKENS,
])
def test_an_abnormal_finish_still_returns_the_delegated_tier(finish):
    """The bypass that actually fires.

    ``_finish_or_continue`` hands back a ``TurnResult`` for each of
    these, from a ``return`` above the exit check -- so before #1025 the
    tier was still ``voz`` when ``send_message`` returned, and stayed
    there for every later turn.  ``MAX_TOKENS`` is the routine one for a
    speech tier: a spoken answer is expensive in output tokens.
    """
    session = _delegated(_live_session(_scripted_provider([
        _response("half a sentence", finish=finish),
    ])))

    session.send_message("say something")

    assert session._active_tier == "executor", (
        f"a delegation that finished {finish} is still resident in voz"
    )
    assert session._pending_tier_return is None


def test_a_provider_error_that_ends_the_turn_still_returns_the_tier():
    """The exit a per-branch fix would never have been written for.

    A provider error does not RETURN from the loop, it propagates out of
    ``send_message`` -- so there is no exit point anybody would think to
    instrument, and the turn is over all the same.  The ``finally``
    covers it for free, which is the argument for putting the guarantee
    there rather than beside each of the loop's eleven returns.
    """
    session = _delegated(_live_session(_scripted_provider([
        _response("", finish=FinishReason.ERROR),
    ])))

    with pytest.raises(RuntimeError):
        session.send_message("say something")

    assert session._active_tier == "executor"
    assert session._pending_tier_return is None


def test_an_output_cap_on_the_continuation_still_returns_the_tier():
    """The same bypass one slot later: the tool-continuation path.

    The delegate calls a tool, its continuation hits the cap, and
    ``_execute_tools_and_continue`` returns at step 5 -- the exact
    ``return None, abnormal, False`` the issue names, with the exit
    check three statements below it.
    """
    session = _delegated(_live_session(_scripted_provider([
        _response(calls=[FunctionCall(id="c1", name="speak",
                                      args={"text": "hi"})],
                  finish=FinishReason.TOOL_USE),
        _response("spoke, then was cut off", finish=FinishReason.MAX_TOKENS),
    ])))

    session.send_message("say something")

    assert session._active_tier == "executor"
    assert session._pending_tier_return is None


def test_the_continuation_slot_alone_does_not_pop_an_abnormal_finish():
    """Why the turn-end backstop is load-bearing and not belt-and-braces.

    At the slot the issue points at, an abnormal finish still returns
    before the exit check -- deliberately, because #749 owns that exit
    and re-ordering it to suit tiers would put a tier concern in front of
    a truncation-recovery one.  The arming therefore SURVIVES this
    method, and the turn-end evaluation is what spends it.  Recorded so
    the two halves are not mistaken for duplicates of each other.
    """
    session = _delegated(_live_session(MagicMock()))

    _, turn_result, _ = _continuation_slot(
        session, _response("cut off", finish=FinishReason.MAX_TOKENS))

    assert turn_result is not None, "the turn did not end here"
    assert session._pending_tier_return == "executor"


# ==================== the nudge (bypass 2 -- not a bypass) ============


@pytest.mark.parametrize("finish,text", [
    (FinishReason.TOOL_USE, ""),      # TOOL_USE without function calls
    (FinishReason.UNKNOWN, ""),       # empty UNKNOWN
    (FinishReason.STOP, "said it"),   # the clean settle, for contrast
])
def test_the_nudge_path_was_never_a_bypass(finish, text):
    """Recorded because the issue names it as one, and it is not.

    Driven at the tool-continuation slot the issue's reading is about,
    with everything between it and the exit check shipped.  The check
    sits AFTER ``_nudge_for_tool_use``, so a re-fetch cannot skip it --
    it can only change WHICH response is judged.  All three of these
    popped correctly BEFORE the fix and still do, which is why this file
    says the issue is wrong about that candidate; the row exists so a
    future reader does not go hunting for a defect here.
    """
    session = _delegated(_live_session(MagicMock()))

    _continuation_slot(session, _response(text, finish=finish))

    assert session._active_tier == "executor"
    assert session._pending_tier_return is None


# ==================== the hold (bypass 3 -- deliberate) ===============


def test_a_delegate_still_calling_tools_keeps_the_wheel():
    """Not "one provider call".

    Popping here would hand control back with the delegate's own work
    outstanding, which is the design trap the fix must not fall into.
    The arming survives the response carrying calls; the loop
    re-evaluates on the next continuation.
    """
    session = _delegated(_live_session(MagicMock()))

    session._exit_completion_tier_if_settled(
        _response(calls=[FunctionCall(id="c1", name="speak", args={})],
                  finish=FinishReason.TOOL_USE))

    assert session._active_tier == "voz"
    assert session._pending_tier_return == "executor"


# ==================== the attachment loop =============================


def test_the_attachment_loop_returns_the_delegated_tier_too():
    """The voice path, which evaluated the exit nowhere at all.

    A spoken question routes to ``_run_chat_loop_with_parts``, which has
    its own tool loop and never called
    ``_exit_completion_tier_if_settled``.  So a speech tier delegated
    from an audio turn could not pop on ANY finish reason, clean settle
    included -- a strictly larger hole than the one the issue reports,
    in the path the reported symptom came from.
    """
    session = _delegated(_live_session(_scripted_provider([
        _response("spoke", finish=FinishReason.STOP, media=3),
    ])))

    session.send_message_with_parts(
        [Part(text=""), Part(inline_data={
            "mime_type": "audio/wav",
            "data": base64.b64encode(b"RIFF....").decode(),
        })],
        lambda *a, **k: None,
    )

    assert session._active_tier == "executor"
    assert session._pending_tier_return is None


# ==================== the arming is spent exactly once ================


def test_the_arming_is_spent_so_the_exit_cannot_fire_twice():
    """A double pop would return PAST the tier that armed the exit.

    ``_take_pending_tier_return`` is the sole clearing site, so the
    mid-turn pop spends the arming and the turn-end backstop that runs
    moments later is a no-op.  Asserted on the tier rather than on a call
    count: the failure mode is a session landing somewhere nobody asked
    for.
    """
    session = _delegated(_live_session(MagicMock()))
    session._active_tier = "voz"

    session._exit_completion_tier_if_settled(_response("said it"))
    assert session._active_tier == "executor"

    # what the turn-end backstop does moments later
    session._finalize_completion_tier_exit(reason="turn end")

    assert session._active_tier == "executor"
    assert session._pending_tier_return is None


def test_handing_back_to_a_completion_tier_does_not_re_arm():
    """A hand-back is not a delegation.

    ``switch_tier`` arms on ANY entry into a completion tier from a
    different one, so a profile where the caller ALSO declares
    ``exit_on: completion`` would have the return itself re-arm --
    pointing back at the tier just left.  That is a ping-pong, and on the
    terminal path it is an arming standing into the next turn, which is
    the bug this whole change is about, one layer along.
    """
    tiers = dict(TIERS)
    tiers["executor"] = {"model": "text-model", "exit_on": "completion"}
    session = _live_session(MagicMock())
    session._tier_config = ModelTierConfig.from_unified_dict(tiers)
    session._active_tier = "executor"
    _delegated(session)

    session._finalize_completion_tier_exit(reason="turn end")

    assert session._active_tier == "executor"
    assert session._pending_tier_return is None, (
        "the hand-back re-armed, so the next turn starts owing a return"
    )


def test_a_turn_that_never_delegated_pops_nothing():
    """The overwhelmingly common session: no tier, nothing pending.

    The backstop runs in every turn's ``finally``, so "does nothing when
    nothing is armed" is a property of every session in the tree, not an
    edge case.
    """
    session = _live_session(_scripted_provider([_response("hello")]))
    before = session._active_tier

    session.send_message("hello")

    assert session._active_tier == before
    assert session._pending_tier_return is None


# ==================== the terminal pop resumes nobody =================


def test_the_turn_end_pop_queues_no_report():
    """Returning the BINDING at turn end is not resuming the caller.

    The mid-turn pop queues ``_report_delegated_tier`` so the caller is
    resumed through the ordinary mid-turn path.  At turn end there is no
    turn to steer, and a queued "You are back in control; continue."
    would be drained by whatever caller-originated turn arrives next --
    the stray-text symptom the issue's third turn reports.
    """
    session = _delegated(_live_session(MagicMock()))

    session._finalize_completion_tier_exit(
        finish_reason="max_tokens", reason="turn end")

    assert session._active_tier == "executor"
    assert session._message_queue.pop_first_parent_message() is None


def test_the_mid_turn_pop_still_reports():
    """...and the resumable half is unchanged.

    Stated beside the row above so the two are read together: the fix
    adds a terminal evaluation, it does not move or weaken the mid-turn
    one.
    """
    session = _delegated(_live_session(MagicMock()))

    session._exit_completion_tier_if_settled(_response("The sky is blue."))

    queued = session._message_queue.pop_first_parent_message()
    assert queued is not None
    text = queued[0] if isinstance(queued, tuple) else str(queued)
    assert "The sky is blue." in text and "voz" in text


def test_a_failed_return_does_not_fail_the_turn():
    """Best-effort, like the entry.

    A turn that has already produced its answer must not be destroyed by
    a tier switch that could not reconnect -- and the arming is spent
    first, so a failing switch cannot leave the exit armed to retry
    forever.
    """
    session = _delegated(_live_session(MagicMock()))
    session.switch_tier = MagicMock(side_effect=RuntimeError("no endpoint"))

    session._finalize_completion_tier_exit(reason="turn end")

    assert session._pending_tier_return is None


# ==================== the INFO lines the issue asked for ==============


def test_the_finish_reason_is_logged_while_a_tier_exit_is_armed(caplog):
    """"finish_reason is not logged at INFO on either side" -- issue #1025.

    Which of the bypasses a given turn took could not be established
    from a production log, which is why the issue could describe the
    symptom exactly and not name the path.  The line names the reason,
    the outcome and whether the response carried calls.
    """
    session = _delegated(_live_session(MagicMock()))

    with caplog.at_level(logging.INFO, logger="jaato_server.shared.jaato_session"):
        session._log_finish_for_pending_tier(
            _response("x", finish=FinishReason.MAX_TOKENS),
            "abnormal", "after tool results")

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "max_tokens" in line
    assert "abnormal" in line
    assert "executor" in line, "the line must name where it would return to"


def test_a_session_with_no_tier_exit_armed_logs_nothing(caplog):
    """The line is gated on the arming, not on the finish reason.

    Otherwise it is a per-response INFO in every session in the fleet,
    which is how a diagnostic gets turned off and stops being available
    for the next incident.
    """
    session = _live_session(MagicMock())

    with caplog.at_level(logging.INFO, logger="jaato_server.shared.jaato_session"):
        session._log_finish_for_pending_tier(
            _response("x", finish=FinishReason.MAX_TOKENS), "abnormal", "-")

    assert not [r for r in caplog.records if "Completion-tier" in r.getMessage()]


def test_the_turn_end_pop_is_announced(caplog):
    """A tier switch that happens silently is the other half of #1025.

    #675 records that nothing emits on a model/tier switch at all; this
    does not fix that, it makes THIS switch -- the one the framework
    performs on the model's behalf -- visible in the log an operator
    already has.
    """
    session = _delegated(_live_session(MagicMock()))

    with caplog.at_level(logging.INFO, logger="jaato_server.shared.jaato_session"):
        session._finalize_completion_tier_exit(
            finish_reason="safety", reason="turn end")

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Completion-tier exit at turn end" in text
    assert "safety" in text
    assert "voz -> executor" in text


# ==================== structural: no loop may omit the backstop =======


def _function(name):
    tree = ast.parse(SESSION_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {SESSION_PATH}")


def _finally_calls(fn):
    names = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Try):
            for stmt in node.finalbody:
                for inner in ast.walk(stmt):
                    if (isinstance(inner, ast.Call)
                            and isinstance(inner.func, ast.Attribute)):
                        names.append(inner.func.attr)
    return names


@pytest.mark.parametrize("loop", ["_run_chat_loop", "_run_chat_loop_with_parts"])
def test_every_chat_loop_evaluates_the_exit_at_turn_end(loop):
    """The guarantee is structural, not per-exit.

    ``_run_chat_loop`` alone has eleven ways out.  Instrumenting each is
    the shape that leaves one path armed and one silently not -- which is
    the defect, and is also why the parts loop had no check at all.  One
    evaluation in the ``finally`` covers every exit a loop grows later,
    including ones whose author never heard of tiers.
    """
    assert "_finalize_completion_tier_exit" in _finally_calls(_function(loop)), (
        f"{loop} can end a turn with a completion tier still armed"
    )


def test_the_pending_tier_return_has_one_clearing_site():
    """One writer each way is what bounds the state.

    ``switch_tier`` arms it; ``_take_pending_tier_return`` spends it.  A
    second clearing site is how a pop and a backstop start disagreeing
    about whether the exit already happened.
    """
    tree = ast.parse(SESSION_PATH.read_text())
    clearers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for tgt in targets:
                if (isinstance(tgt, ast.Attribute)
                        and tgt.attr == "_pending_tier_return"):
                    clearers.add(fn.name)

    assert clearers == {"__init__", "switch_tier", "_take_pending_tier_return"}, (
        f"unexpected writers of _pending_tier_return: {sorted(clearers)}"
    )
