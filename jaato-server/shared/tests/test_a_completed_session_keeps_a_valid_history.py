"""A session that signalled completion must still be drivable (#913).

THE INVARIANT is the one #751 states in full on
``JaatoSession._reconcile_unanswered_calls``: every ``tool_use`` block in
history must have a matching ``tool_result``.  OpenAI/Azure-shaped
upstreams enforce it on the *next* request and reject the whole
conversation when it does not hold::

    An assistant message with 'tool_calls' must be followed by tool
    messages responding to those tool_call_ids

THE HOLE THIS FILLS.  ``signal_completion`` terminates the turn: the
continuation -- ``_send_tool_results_and_continue`` -- is skipped, which
is right, because there is no model call left worth making (a strong
model answers the spur with prose, a weak one calls signal_completion
again forever).  But that continuation was also the ONLY writer of the
batch's results into history.  So a completed session ended on a
``function_call`` nothing answered, and the next request 400'd.

"Session is over" was the load-bearing assumption, and the framework
itself does not hold it: ``session.wake``, ``revive_policy`` and the
documented cold-revive path exist precisely to reopen a finished
session.  None of them worked on one that completed with a typed
payload, and none of them reported anything at wake time -- the failure
surfaced one model call later.  Which cost a whole legitimate pattern:
complete every turn to enforce a contract (a completion processor is the
only way to *make* a model call a tool), then keep talking.  Enforcement
and multi-turn were mutually exclusive.

These tests read HISTORY, not the return value.  The result object
always reached the caller; being absent from the persisted conversation
is the entire defect.
"""
from unittest.mock import MagicMock

from shared.jaato_session import JaatoSession
from shared.session_history import SessionHistory
from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
)


COMPLETION_RESULT = {
    "status": "completed",
    "agent_id": "prueba",
    "summary": "said uno",
}


def _session(*messages: Message) -> JaatoSession:
    """A session shell carrying just a history.

    ``__new__`` rather than ``__init__``: the recorder touches the
    history, the budget (absent = no capping) and the provider (absent =
    no modality gating), and nothing else.
    """
    session = JaatoSession.__new__(JaatoSession)
    session._history = SessionHistory()
    for m in messages:
        session._history.append(m)
    session._instruction_budget = None
    session._provider = None
    session._trace = lambda msg: None
    return session


def _model_turn(*calls: FunctionCall, text: str = "") -> Message:
    parts = ([Part(text=text)] if text else []) + [
        Part(function_call=c) for c in calls
    ]
    return Message(role=Role.MODEL, parts=parts)


def _dangling_call_ids(session: JaatoSession) -> set:
    """Call ids in history that no ``function_response`` answers.

    This is the property the upstream actually checks, so it is what the
    tests assert -- not "a TOOL message exists", which a wrong-id result
    would also satisfy while still 400ing.
    """
    called, answered = set(), set()
    for message in session._history.messages:
        for part in message.parts:
            if part.function_call is not None:
                called.add(part.function_call.id)
            if part.function_response is not None:
                answered.add(part.function_response.call_id)
    return called - answered


# ==================== The recorder ====================


def test_the_completion_result_reaches_history():
    """THE regression: a completed turn left its own call unanswered."""
    session = _session(_model_turn(
        FunctionCall(id="call_1", name="signal_completion", args={}),
    ))

    session._record_terminal_tool_results([
        ToolResult(call_id="call_1", name="signal_completion",
                   result=COMPLETION_RESULT),
    ])

    last = session._history.messages[-1]
    assert last.role == Role.TOOL
    assert [p.function_response.call_id for p in last.parts] == ["call_1"]
    assert last.parts[0].function_response.result == COMPLETION_RESULT
    assert not _dangling_call_ids(session)


def test_a_call_the_terminal_batch_never_dispatched_is_answered_too():
    """Parallel calls are dropped whole, not just the ones after it.

    ``signal_completion`` ends the batch, so a model that emitted
    ``store_memory`` alongside it leaves that ``tool_use`` unanswered as
    well -- the same 400, reached from the same turn.
    """
    session = _session(_model_turn(
        FunctionCall(id="call_1", name="signal_completion", args={}),
        FunctionCall(id="call_2", name="store_memory", args={}),
    ))

    session._record_terminal_tool_results([
        ToolResult(call_id="call_1", name="signal_completion",
                   result=COMPLETION_RESULT),
    ])

    assert not _dangling_call_ids(session)
    last = session._history.messages[-1]
    answers = {p.function_response.call_id: p.function_response
               for p in last.parts}
    assert set(answers) == {"call_1", "call_2"}
    dropped = answers["call_2"]
    assert dropped.is_error
    assert dropped.result["unexecuted"] is True
    # The remedy has to differ from an abandoned call's: re-sending is
    # right after a truncation and wrong after a completion.
    assert "store_memory" in dropped.result["error"]
    assert "Do not re-send it unless the conversation continues" in (
        dropped.result["error"]
    )


def test_calls_already_answered_are_not_answered_twice():
    """Narrowness: only the batch's own undispatched calls are synthesised.

    Without this, a batch whose results all came back would still grow a
    duplicate answer per call -- two results for one id is as invalid as
    none.
    """
    session = _session(_model_turn(
        FunctionCall(id="call_1", name="signal_completion", args={}),
        FunctionCall(id="call_2", name="store_memory", args={}),
    ))

    session._record_terminal_tool_results([
        ToolResult(call_id="call_1", name="signal_completion",
                   result=COMPLETION_RESULT),
        ToolResult(call_id="call_2", name="store_memory", result={"ok": True}),
    ])

    last = session._history.messages[-1]
    assert [p.function_response.call_id for p in last.parts] == [
        "call_1", "call_2",
    ]
    assert last.parts[1].function_response.result == {"ok": True}


def test_earlier_turns_are_left_alone():
    """Only a TRAILING model message can hold undispatched calls.

    Mirrors ``_reconcile_unanswered_calls``.  Reaching further back would
    answer calls a mid-turn interrupt already handled, or a rewind
    already dropped.
    """
    session = _session(
        Message(role=Role.USER, parts=[Part(text="Di 'uno'.")]),
        _model_turn(FunctionCall(id="call_0", name="readFile", args={})),
        Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
            call_id="call_0", name="readFile", result={"content": "uno"}))]),
        _model_turn(FunctionCall(id="call_1", name="signal_completion",
                                 args={})),
    )
    before = len(session._history.messages)

    session._record_terminal_tool_results([
        ToolResult(call_id="call_1", name="signal_completion",
                   result=COMPLETION_RESULT),
    ])

    assert len(session._history.messages) == before + 1
    assert not _dangling_call_ids(session)


def test_a_batch_whose_turn_is_not_trailing_still_records_its_results():
    """The results are the point; the orphan sweep is the extra.

    An interleaved batch executes against a snapshot of an older
    response, so the trailing message may be a later model turn.  The
    sweep is then a no-op -- but the batch's own results must still land.
    """
    session = _session(
        _model_turn(FunctionCall(id="call_1", name="signal_completion",
                                 args={})),
        Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
            call_id="call_1", name="signal_completion",
            result=COMPLETION_RESULT))]),
        Message(role=Role.USER, parts=[Part(text="and again")]),
    )

    session._record_terminal_tool_results([
        ToolResult(call_id="call_9", name="store_memory", result={"ok": True}),
    ])

    last = session._history.messages[-1]
    assert last.role == Role.TOOL
    assert [p.function_response.call_id for p in last.parts] == ["call_9"]


# ==================== The turn path reaches it ====================


def test_the_terminating_turn_records_before_it_returns():
    """The recorder is wired to the early return, not merely present.

    The optimisation (skip the spur + round-trip) and the bookkeeping
    were coupled in one ``return``; this pins that only the optimisation
    survives.
    """
    session = _session(_model_turn(
        FunctionCall(id="call_1", name="signal_completion", args={}),
    ))
    session._signal_completion_called = True
    session._cancel_token = None
    session._is_cancelled = lambda: False
    session._executor = None
    session._gc_plugin = None
    session._gc_config = None
    session._execute_function_call_group = MagicMock(return_value=[
        ToolResult(call_id="call_1", name="signal_completion",
                   result=COMPLETION_RESULT),
    ])
    session._send_tool_results_and_continue = MagicMock(
        side_effect=AssertionError(
            "the terminal path must not make another model call"
        )
    )

    response, turn_result, interrupted = JaatoSession._execute_tools_and_continue(
        session,
        fc_group=[FunctionCall(id="call_1", name="signal_completion", args={})],
        use_streaming=False,
        on_output=None,
        wrapped_usage_callback=None,
        turn_data={},
        cancellation_notified=False,
        accumulated_text=["done"],
    )

    assert response is None
    assert turn_result is not None and turn_result.text == "done"
    assert interrupted is False
    session._send_tool_results_and_continue.assert_not_called()
    assert not _dangling_call_ids(session), (
        "the turn returned leaving a function_call nothing answers -- the "
        "next request on this session, including a session.wake revive, "
        "would be rejected by the provider"
    )
