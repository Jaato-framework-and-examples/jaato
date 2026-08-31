"""A tool call the turn never ran must still get a tool result.

THE GAP (#751).  Every ``tool_use`` block in history needs a matching
``tool_result``; OpenAI/Azure-shaped upstreams enforce it on the *next*
request and reject the whole conversation when it does not hold::

    {"message": "No tool output found for function call call_mAyQ...",
     "type": "invalid_request_error", "param": "input"}

A turn severed at the output cap can leave exactly that.  Observed on an
eval arm (gpt-5-mini via OpenRouter, Azure upstream, 2026-08-31) as four
lines and 2.2 seconds: ``COMPLETE_STREAM_END finish_reason=MAX_TOKENS``,
one more ``SESSION_SEND_MESSAGE``, ``BadRequestError: 400``.  The arm was
neither out of budget nor out of wall-clock; the session simply could not
make another request.  It did not degrade, it stopped.

WHY IT WAS NEWLY REACHABLE, AND WHY NOTHING ELSE COVERS IT.  Before #745
a truncated turn reported ``TOOL_USE``, so the loop continued and the
call -- fabricated, empty-args (#750) -- was *executed*.  That produced a
tool output, and history stayed valid by accident.  #745 is right and
stands; removing the execution removed the accident.

The two mechanisms that look adjacent both miss this shape, and the
distinction is the whole point:

  * ``rewind.detect_truncated_tool_call`` fires on a *damaged* call
    (empty or missing-required arguments) and drops it from history.
    A call cut off **after** it was fully serialized is not damaged --
    ``parse_tool_call_arguments('{"path": "a.py"}')`` returns
    ``({'path': 'a.py'}, None)`` -- so the detector correctly declines
    and history keeps the call.
  * ``unreadable_arguments_error`` (#750) keys on
    ``FunctionCall.unreadable_args``, which a complete call does not
    carry, and is reached from the tool-execution path the abnormal
    finish returns before ever entering.

So the two truncation shapes diverge: cut *mid*-arguments is answered by
#750's refusal; cut *after* a complete call is answered by nothing.  A
model emitting several calls in one turn, or one call followed by
narration, lands in the second far more readily than the first.

THE CONTRACT.  On an abnormal finish, any call left unanswered in
history gets a synthesised tool result saying it was not executed and
why.  Two properties, in that order of importance: history is valid so
the session survives, and the tool-result slot -- where the model
already looks for the outcome of the call it just made -- is the natural
vehicle for telling it that it was cut off (#749's nudge, in-band).

WHY THIS GUARD SHAPE.  A test that asserts only on the finish reason
(as #745's do, correctly for their scope) passes today while the session
is dead: the truncation is reported perfectly and the *next* request is
the one that fails.  So the assertion has to span two requests, and the
first one's call has to be **complete and valid** -- a test built on
malformed arguments now passes through #750's path and would prove
nothing about this one.  That trap is not hypothetical; it is why #750
was first reported as closing this issue outright.
"""

import ast
import pathlib
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolResult,
    ToolSchema,
    TurnResult,
    parse_tool_call_arguments,
    unexecuted_call_error,
)

from shared.jaato_session import JaatoSession
from shared.rewind import detect_truncated_tool_call
from shared.session_history import SessionHistory
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


SESSION_PATH = pathlib.Path(__file__).resolve().parents[1] / "jaato_session.py"


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""        abnormal = self._classify_finish_reason(response, turn_data, on_output)
        if abnormal is not None:
            self._reconcile_unanswered_calls(response.finish_reason)
        return abnormal""",
        replace="""        abnormal = self._classify_finish_reason(response, turn_data, on_output)
        return abnormal""",
        test="test_the_session_can_still_make_a_request_after_a_severed_turn",
        because="a turn severed after a complete tool call leaving that "
                "call unanswered in history, so the next request is "
                "rejected 400 and the session is dead",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""                self._reconcile_unanswered_calls(response.finish_reason)
                response_text = response.get_text()""",
        replace="""                response_text = response.get_text()""",
        test="test_the_parts_loop_reconciles_too",
        because="the multi-part chat loop abandoning its own severed "
                "call, which the non-parts loop's fix does not reach",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""        last = messages[-1]
        if last.role != Role.MODEL:
            return 0""",
        replace="""        last = messages[-1]
        if last.role != Role.MODEL:
            return 0
        return 0""",
        test="test_the_reconciler_answers_every_call_in_the_severed_turn",
        because="the reconciler quietly answering nothing while every "
                "caller believes history was made valid",
    ),
]


WRITE_FILE = ToolSchema(
    name="write_file",
    description="Write a file.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)


def _complete_call(call_id: str = "call_mAyQ") -> FunctionCall:
    """A call that arrived WHOLE, then the turn was cut off after it.

    Built through the real parser so the premise is checked rather than
    asserted: complete arguments carry no ``unreadable_args`` marker,
    which is precisely why #750's refusal path never sees this call.
    """
    args, unreadable = parse_tool_call_arguments(
        '{"path": "notes.md", "content": "hello"}'
    )
    assert unreadable is None, "the premise of this whole module"
    return FunctionCall(id=call_id, name="write_file", args=args)


# ==================== The neighbours genuinely do not cover it ==========


def test_the_rewind_detector_declines_a_complete_call():
    """The discriminator, stated as a test.

    Without this, someone reading the fix could reasonably conclude the
    rewind path already handles ``MAX_TOKENS`` + a tool call and that
    the reconciler is redundant.  It is not: rewind fires on *damaged*
    arguments, and these are intact.
    """
    response = ProviderResponse(
        parts=[Part(text="writing it now"),
               Part(function_call=_complete_call())],
        finish_reason=FinishReason.MAX_TOKENS,
    )
    assert detect_truncated_tool_call(response, [WRITE_FILE]) is None


def test_the_rewind_detector_still_fires_on_a_damaged_one():
    """...and the reason it declines is the arguments, not the reason.

    Pins the contrast: same finish reason, same tool, same turn shape --
    only the completeness of the arguments differs.
    """
    response = ProviderResponse(
        parts=[Part(text="writing it now"),
               Part(function_call=FunctionCall(
                   id="call_bad", name="write_file", args={}))],
        finish_reason=FinishReason.MAX_TOKENS,
    )
    assert detect_truncated_tool_call(response, [WRITE_FILE]) is not None


# ==================== The reconciler ====================


def _session_with_history(*messages: Message) -> JaatoSession:
    """A session shell carrying just a history.

    ``__new__`` rather than ``__init__`` for the same reason the other
    session guards do: a real session needs a provider, a runtime and a
    registry, and the reconciler touches none of them.
    """
    session = JaatoSession.__new__(JaatoSession)
    session._history = SessionHistory()
    for m in messages:
        session._history.append(m)
    session._provider = None
    session._trace = lambda msg: None
    return session


def _model_turn(*calls: FunctionCall, text: str = "on it") -> Message:
    return Message(
        role=Role.MODEL,
        parts=[Part(text=text)] + [Part(function_call=c) for c in calls],
    )


def test_the_reconciler_answers_every_call_in_the_severed_turn():
    """One result per call, paired by id.

    Answering only the first would satisfy a naive "history is no longer
    empty of results" check while leaving the same 400 for call two.
    """
    calls = [_complete_call("call_a"), _complete_call("call_b")]
    session = _session_with_history(_model_turn(*calls))

    answered = session._reconcile_unanswered_calls(FinishReason.MAX_TOKENS)

    assert answered == 2
    last = session._history.messages[-1]
    assert last.role == Role.TOOL
    assert [p.function_response.call_id for p in last.parts] == [
        "call_a", "call_b",
    ]
    assert all(p.function_response.is_error for p in last.parts)


def test_the_reconciler_is_a_no_op_with_nothing_pending():
    """Narrowness.

    Without this the contract could be met by appending a tool message
    unconditionally, which would corrupt every clean turn in the fleet.
    """
    session = _session_with_history(
        Message(role=Role.MODEL, parts=[Part(text="all done")]),
    )
    before = len(session._history.messages)

    assert session._reconcile_unanswered_calls(FinishReason.MAX_TOKENS) == 0
    assert len(session._history.messages) == before


def test_the_reconciler_is_a_no_op_on_an_already_answered_turn():
    """Idempotent, because history is the source of truth, not the response.

    Once results are appended the trailing message is no longer a model
    turn, so a second call finds nothing -- which is what makes it safe
    to reconcile from more than one exit path.
    """
    session = _session_with_history(_model_turn(_complete_call()))
    assert session._reconcile_unanswered_calls(FinishReason.MAX_TOKENS) == 1
    before = len(session._history.messages)

    assert session._reconcile_unanswered_calls(FinishReason.MAX_TOKENS) == 0
    assert len(session._history.messages) == before


def test_the_reconciler_survives_an_empty_history():
    session = _session_with_history()
    assert session._reconcile_unanswered_calls(FinishReason.ERROR) == 0


# ==================== What the model is told ====================


def test_the_result_names_truncation_and_not_a_parse_failure():
    """The message has to be actionable, and #750's is the wrong one.

    Telling a model its arguments "could not be parsed" when they parsed
    perfectly sends it to re-serialize a call that was already correct,
    instead of to shorten its output -- which is the only thing that
    gets it past a cap.
    """
    payload = unexecuted_call_error(_complete_call(), FinishReason.MAX_TOKENS)

    assert "write_file" in payload["error"]
    assert "NOT executed" in payload["error"]
    assert "output-token limit" in payload["error"]
    assert "could not be parsed" not in payload["error"]
    assert payload["unexecuted"] is True
    assert payload["finish_reason"] == "max_tokens"


@pytest.mark.parametrize("reason,needle", [
    (FinishReason.SAFETY, "safety filter"),
    (FinishReason.ERROR, "reported an error"),
    (None, "the turn ended"),
])
def test_the_result_names_whichever_cause_ended_the_turn(reason, needle):
    """``MAX_TOKENS`` is the observed case, not the only one.

    ``_classify_finish_reason`` routes ``SAFETY`` and ``ERROR`` down the
    same abnormal exit, so both can strand a call too.
    """
    payload = unexecuted_call_error(_complete_call(), reason)
    assert needle in payload["error"]


# ==================== End to end: the next request is accepted =========


class _PairingViolation(AssertionError):
    """The 400, reproduced locally.

    Named for what the upstream reports rather than for the HTTP status,
    because the status is the symptom.
    """


def _dangling_call_ids(messages):
    """Call ids in *messages* with no matching ``function_response``."""
    answered = {
        p.function_response.call_id
        for m in messages for p in m.parts
        if p.function_response is not None
    }
    return [
        p.function_call.id
        for m in messages for p in m.parts
        if p.function_call is not None and p.function_call.id not in answered
    ]


def _severed_then_clean_provider(seen):
    """A provider that answers once with a severed turn, then normally.

    Every call records the dangling ids in the history it was handed, so
    the assertion is on what the *upstream would have seen* rather than
    on an internal.  Azure rejects the request outright on a dangling
    id; recording instead of raising keeps the test measuring history
    validity rather than the framework's retry behaviour.
    """
    responses = [
        ProviderResponse(
            parts=[Part(text="I'll write that file now"),
                   Part(function_call=_complete_call())],
            finish_reason=FinishReason.MAX_TOKENS,
            usage=TokenUsage(prompt_tokens=10, output_tokens=5,
                             total_tokens=15),
        ),
        ProviderResponse(
            parts=[Part(text="Understood — I'll write it in pieces.")],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=20, output_tokens=5,
                             total_tokens=25),
        ),
    ]

    def complete(messages, **_kwargs):
        seen.append(_dangling_call_ids(messages))
        return TurnResult.from_provider_response(
            responses[min(len(seen) - 1, len(responses) - 1)]
        )

    provider = MagicMock()
    provider.name = "fake"
    provider.supports_streaming.return_value = True
    provider.get_context_limit.return_value = 0
    provider.complete.side_effect = complete
    return provider


def _live_session(provider):
    runtime = MagicMock()
    runtime.create_provider.return_value = provider
    runtime.get_tool_schemas.return_value = [WRITE_FILE]
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.ledger = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = [WRITE_FILE]
    runtime.registry.enrich_prompt.side_effect = (
        lambda prompt, **_k: MagicMock(prompt=prompt, metadata={})
    )
    session = JaatoSession(runtime, "gpt-5-mini")
    session.configure()
    return session


def test_the_session_can_still_make_a_request_after_a_severed_turn():
    """The issue, end to end.

    Turn one is cut off at the cap holding a complete ``write_file``
    call.  Turn two is the request that used to 400 -- and the assertion
    is on the history that request carries, because that is what the
    upstream validates.
    """
    seen = []
    session = _live_session(_severed_then_clean_provider(seen))

    session.send_message("Write me a long file")
    session.send_message("Try again, smaller")

    assert len(seen) == 2, (
        f"expected one provider request per message, got {len(seen)}"
    )
    assert seen[0] == [], "the first request cannot be at fault"
    assert seen[1] == [], (
        f"the request after the severed turn carried unanswered tool "
        f"call(s) {seen[1]}. Azure rejects it with 'No tool output found "
        f"for function call {seen[1][0] if seen[1] else ''}' and the "
        f"session cannot continue (#751)."
    )


def test_the_severed_turn_still_reports_its_own_truncation():
    """Reconciling must not paper over the finish reason.

    The whole value of #745 is that a truncated turn says so; a fix that
    made it look like a clean tool turn again would trade one defect for
    the one before it.
    """
    seen = []
    session = _live_session(_severed_then_clean_provider(seen))

    session.send_message("Write me a long file")

    accounting = session.get_turn_accounting()
    assert accounting, "the turn recorded no accounting at all"
    assert accounting[-1]["finish_reason"] == "max_tokens"


def test_the_parts_loop_reconciles_too():
    """The multi-part loop is a second, independent exit.

    ``_run_chat_loop_with_parts`` has its own inline abnormal check and
    never touches ``_finish_abnormally``, so a fix applied only to the
    main loop holds exactly until an attachment is in the message.
    """
    seen = []
    provider = _severed_then_clean_provider(seen)
    provider.supports_streaming.return_value = False
    session = _live_session(provider)

    session.send_message_with_parts(
        [Part(text="Describe this and write it up")], lambda *a, **k: None,
    )
    session.send_message_with_parts(
        [Part(text="Try again, smaller")], lambda *a, **k: None,
    )

    assert len(seen) == 2
    assert seen[1] == [], (
        f"the parts loop left {seen[1]} unanswered after a severed turn"
    )


# ==================== The classifier stays the single chokepoint =======


def _attribute_calls(tree, name):
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_no_chat_loop_exit_bypasses_the_reconciler():
    """``_classify_finish_reason`` is reached through one wrapper only.

    The classifier is deliberately pure -- it is driven on its own by
    ``test_abnormal_finish_surfacing`` -- so the reconciliation lives in
    ``_finish_abnormally`` beside it.  That is safe exactly as long as
    nothing else calls the classifier directly: a new exit that did
    would reintroduce the dangling call with no test noticing.
    """
    tree = ast.parse(SESSION_PATH.read_text(encoding="utf-8"))
    callers = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _attribute_calls(node, "_classify_finish_reason"):
            callers.append(node.name)

    assert callers == ["_finish_abnormally"], (
        f"_classify_finish_reason is called from {callers}. Every "
        f"abnormal exit from the chat loop must go through "
        f"_finish_abnormally, which reconciles the calls the severed "
        f"turn abandoned (#751); calling the bare classifier skips that "
        f"and leaves history invalid for the next request."
    )
