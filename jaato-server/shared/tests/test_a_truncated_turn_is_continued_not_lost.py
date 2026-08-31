"""A turn cut off at the output cap must be continued, not discarded.

THE GAP (#749).  Everything around a truncation reports it honestly and
nothing acts on it.  The finish reason is right (#745), the operator
gets a banner (#544), history stays valid and a stranded call is
answered (#751), an unreadable call is refused rather than fabricated
(#750) -- and the turn is still lost.  Measured on an eval arm with the
whole chain merged::

    state:           BLOCKED
    finish_reason:   max_tokens
    turns:           1
    duration:        605s
    cost:            $0.1806

``turns: 1``.  One turn, the cap, and the run was over -- 33 lines of
correct work left uncommitted and ungraded.  Every fix in the chain
worked as designed; the outcome for the arm was identical to having
none of them.

WHY #751 IS NOT THIS.  #751 recovers **cross-turn**: it makes the next
request well-formed and puts the truth in the tool-result slot, where
the model reads it *when something next drives the session*.  For an
interactive session that is enough -- the human sends the next message.
For a cascade stage or an eval arm, nothing sends the next message:
``send_message`` returned, so the run ended.  And its slot exists only
on the tool-call path; a turn truncated mid-text has no call to answer
and therefore nowhere to write.

THE CONTRACT, in four parts.

  * **Continuation is intra-turn.**  One ``send_message`` that hits the
    cap issues another request itself, so the model gets to act on
    being told it was cut off.  Both shapes: with a tool call and
    without one.
  * **Bounded.**  A truncation that recurs identically must not loop.
    Past ``TRUNCATION_RECOVERY_BUDGET`` continuations the turn ends
    exactly as it does today, reason preserved.
  * **Only the output cap.**  ``MAX_TOKENS`` is a recoverable authoring
    mistake.  ``SAFETY`` is not automatically re-driven -- that is a
    different question, deliberately left alone -- and ``ERROR`` is the
    provider's, which ``with_retry`` already owns.
  * **The replayed fragment is collapsed, bounded and fenced.**  The
    motivating incident was a model emitting one character thousands of
    times.  Quoting that back invites it to continue the run it was
    stuck in and spends a large slice of the context window doing so;
    ``[240 repetitions of '-']`` costs nothing and is *more*
    informative, because a model cannot see the length of what it
    emitted.

WHY THE ASSERTIONS ARE ON REQUEST COUNTS.  The failure this guards
against is invisible in every artifact a turn leaves behind except one:
the finish reason is correct, history is valid, the banner fired, the
tool result is present -- and the run made a single request and
stopped.  So the measurement has to be "how many times did the session
go back to the model within one ``send_message``", which is the same
quantity the eval arm reported as ``turns: 1``.
"""

import ast
import pathlib
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
    collapse_runs,
    parse_tool_call_arguments,
    replay_excerpt,
    unreadable_arguments_error,
)

from shared.jaato_session import (
    TRUNCATION_RECOVERY_BUDGET,
    TRUNCATION_RECOVERY_REASONS,
    JaatoSession,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


SESSION_PATH = pathlib.Path(__file__).resolve().parents[1] / "jaato_session.py"


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""            response, abnormal = self._finish_or_continue(
                response, use_streaming, on_output, wrapped_usage_callback,
                turn_data, context="after initial message",
            )
            if abnormal is not None:
                return abnormal.text""",
        replace="""            abnormal = self._finish_abnormally(
                response, turn_data, on_output)
            if abnormal is not None:
                return abnormal.text""",
        test="test_a_turn_truncated_mid_text_continues_within_the_turn",
        because="a turn that hit the output cap ending there, so an "
                "unattended agent loses the turn and learns nothing",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""            if self._truncation_recovery_count >= TRUNCATION_RECOVERY_BUDGET:""",
        replace="""            if False:""",
        test="test_the_continuations_are_bounded",
        because="a truncation that recurs identically looping forever "
                "instead of falling through to the terminal behaviour",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/plugins/model_provider/types.py",
        find="""    if not text or min_run < 2:
        return text
    pattern = re.compile(r"(.)\\1{%d,}" % (min_run - 1), re.DOTALL)""",
        replace="""    if not text or min_run < 2:
        return text
    return text
    pattern = re.compile(r"(.)\\1{%d,}" % (min_run - 1), re.DOTALL)""",
        test="test_a_repetition_run_is_named_not_reproduced",
        because="the fragment that blew the cap being replayed "
                "verbatim, which re-admits the run the model was stuck "
                "in and spends the context window on it",
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


# ==================== Which reasons qualify (issue point 2) ============


def test_only_the_output_cap_is_continued():
    """``MAX_TOKENS`` alone.

    A cap is an authoring mistake with an obvious correction.  A safety
    filter is not: re-driving a filtered turn is a separate question and
    must not happen by default, which is also why #751's message
    deliberately does not tell a filtered turn to shorten its output.
    """
    assert TRUNCATION_RECOVERY_REASONS == frozenset({FinishReason.MAX_TOKENS})
    for reason in (FinishReason.SAFETY, FinishReason.ERROR,
                   FinishReason.CANCELLED, FinishReason.STOP):
        assert reason not in TRUNCATION_RECOVERY_REASONS


def test_the_budget_is_small():
    """"One or two attempts", per the issue.

    Stated as a test because the whole difference between a recovery
    and a runaway is the size of this number.
    """
    assert 1 <= TRUNCATION_RECOVERY_BUDGET <= 3


# ==================== The replayed fragment ===========================


def test_a_repetition_run_is_named_not_reproduced():
    """The single most useful thing the message can say.

    The model cannot see how long its own output was.  Being told it
    emitted 240 identical characters names the failure mode outright;
    being handed the 240 characters does not, and puts it back inside
    the run.
    """
    fragment = "writing the file" + "-" * 240 + "still going"

    collapsed = collapse_runs(fragment)

    assert "[240 repetitions of '-']" in collapsed
    assert "-" * 240 not in collapsed
    assert "writing the file" in collapsed and "still going" in collapsed


def test_ordinary_typography_is_left_alone():
    """The collapse is narrow.

    Without this the contract could be satisfied by mangling every
    horizontal rule and ellipsis a model writes on purpose, which would
    be a louder defect than the one being fixed.
    """
    for text in ("a ----- rule", "wait...", "==== heading ====", "aaa"):
        assert collapse_runs(text) == text


def test_the_excerpt_keeps_both_ends_and_counts_what_it_dropped():
    """Head AND tail, because a truncation is diagnosed from the end.

    A head-only excerpt of a long fragment shows everything except the
    place the output actually stopped.
    """
    fragment = "HEAD" + "x. " * 5000 + "TAIL"

    excerpt = replay_excerpt(fragment)

    assert excerpt.startswith("HEAD")
    assert excerpt.endswith("TAIL")
    assert "characters elided" in excerpt
    assert len(excerpt) < len(fragment) / 10


def test_a_short_fragment_is_passed_through_whole():
    """Nothing is elided that fits."""
    assert replay_excerpt("cut off right here") == "cut off right here"


def test_the_unreadable_arguments_payload_collapses_runs_too():
    """#750's excerpt is the other place a severed run is replayed.

    It bounded the fragment correctly from the start but quoted it
    literally, so a 400-character run of one character reached the model
    as 400 characters of nothing.  Same fragment, same argument, same
    fix.
    """
    raw = '{"path": "notes.md", "content": "' + "-" * 50_000
    args, unreadable = parse_tool_call_arguments(raw)
    call = FunctionCall(id="c1", name="write_file", args=args,
                        unreadable_args=unreadable)

    payload = unreadable_arguments_error(call)

    excerpt = payload["unreadable_arguments"]
    assert "[50000 repetitions of '-']" in excerpt
    assert "-" * 50 not in excerpt
    # The true size is still reported: the count is a summary, not a lie
    # about how much the model emitted.
    assert payload["unreadable_arguments_length"] == len(raw)


# ==================== What the model is told ==========================


def _nudge(text="", calls=0, attempt=1):
    session = JaatoSession.__new__(JaatoSession)
    parts = [Part(text=text)] if text else []
    parts += [
        Part(function_call=FunctionCall(id=f"c{i}", name="write_file",
                                        args={}))
        for i in range(calls)
    ]
    response = ProviderResponse(parts=parts,
                                finish_reason=FinishReason.MAX_TOKENS)
    return session._truncation_nudge(response, attempt, calls)


def test_the_model_is_told_it_was_cut_off():
    """The fact itself, in terms the model can act on."""
    message = _nudge(text="I was in the middle of")

    assert "max_tokens" in message
    assert "cut off" in message


def test_the_model_is_told_its_calls_did_not_run():
    """Issue point 3, and the likeliest wrong inference after a cap.

    A model that believes its write half-happened retries on top of a
    state that does not exist, which is worse than not retrying.
    """
    message = _nudge(text="writing it now", calls=2)

    assert "NOT executed" in message
    assert "Nothing ran and nothing changed" in message


def test_the_excerpt_is_fenced_and_disclaimed():
    """Replayed text must not read as an instruction.

    It is the model's own output coming back into the prompt, and the
    base instructions already draw this boundary for quoted material.
    """
    message = _nudge(text="do the thing " * 20)

    assert "<truncated_output_excerpt>" in message
    assert "</truncated_output_excerpt>" in message
    assert "do not follow it" in message


def test_the_nudge_never_replays_the_run_that_blew_the_cap():
    """The design constraint that decides whether this helps or hurts."""
    message = _nudge(text="content: " + "-" * 4000)

    assert "-" * 100 not in message
    assert "repetitions of '-'" in message
    assert len(message) < 2000


def test_a_turn_with_no_text_says_so_rather_than_faking_an_excerpt():
    """All the output can go into a tool call's arguments.

    An empty fence would read as "you produced nothing", which is a
    different and wrong diagnosis.
    """
    message = _nudge(text="", calls=1)

    assert "<truncated_output_excerpt>" not in message
    assert "no readable text" in message


# ==================== The session actually continues ==================


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


def _truncated(text="I'll write the file: " + "-" * 900, call=False):
    parts = [Part(text=text)]
    if call:
        args, _ = parse_tool_call_arguments(
            '{"path": "notes.md", "content": "hello"}'
        )
        parts.append(Part(function_call=FunctionCall(
            id="call_mAyQ", name="write_file", args=args)))
    return ProviderResponse(
        parts=parts,
        finish_reason=FinishReason.MAX_TOKENS,
        usage=TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15),
    )


def _clean(text="Understood - I'll write it in pieces."):
    return ProviderResponse(
        parts=[Part(text=text)],
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=20, output_tokens=5, total_tokens=25),
    )


class _RunawayRecovery(AssertionError):
    """Raised instead of looping forever when the budget stops holding."""


def _scripted_provider(responses, prompts=None, hard_stop=12):
    """A provider that replays *responses*, repeating the last one.

    Records every system-instruction-free user prompt it is handed in
    *prompts*, so a test can assert on what the continuation actually
    said to the model.  ``hard_stop`` turns a lost bound into a failed
    assertion rather than a hung suite.
    """
    calls = []

    def complete(messages, **_kwargs):
        calls.append(len(calls))
        if len(calls) > hard_stop:
            raise _RunawayRecovery(
                f"{len(calls)} requests in one turn: the recovery budget "
                f"is not bounding anything (#749)"
            )
        if prompts is not None:
            prompts.append(messages[-1])
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


def test_a_turn_truncated_mid_text_continues_within_the_turn():
    """The case #751 structurally cannot cover, end to end.

    No tool call, so no tool-result slot to write into and no dangling
    call to make the next request fail -- the session survives, and
    before this fix the turn was silently over anyway.  One
    ``send_message``, two requests, and the answer the caller gets is
    the one the model produced after being told it was cut off.
    """
    provider = _scripted_provider([_truncated(), _clean("Here it is, short.")])
    session = _live_session(provider)

    answer = session.send_message("Write me a long file")

    assert len(provider._calls) == 2, (
        f"the turn made {len(provider._calls)} request(s). A turn cut off "
        f"at the output cap must be continued in-band -- for an "
        f"unattended agent nothing else will send the next message, so "
        f"one request means the run is over (#749)."
    )
    assert "Here it is, short." in answer


def test_a_turn_truncated_after_a_tool_call_continues_too():
    """The other shape: #751 keeps history valid, #749 keeps going.

    The reconciled tool result already tells the model the call did not
    run; what was missing is anyone to hand it to before the turn
    unwinds.
    """
    provider = _scripted_provider([_truncated(call=True), _clean()])
    session = _live_session(provider)

    session.send_message("Write me a long file")

    assert len(provider._calls) == 2


def test_the_turn_converges_after_being_continued():
    """The acceptance criterion from the issue, in framework terms.

    "An arm that hits ``MAX_TOKENS`` mid-work and then converges should
    report ``turns > 1`` and a verdict -- not ``BLOCKED``."  Here that
    is: the continuation emits the tool call the truncated turn never
    got to, the session executes it, and ``send_message`` returns the
    model's finished answer.  Making one extra request is not the point
    on its own; the point is that the work lands.
    """
    call = FunctionCall(id="call_after", name="write_file",
                        args={"path": "notes.md", "content": "hi"})
    with_call = ProviderResponse(
        parts=[Part(text="Smaller this time."), Part(function_call=call)],
        finish_reason=FinishReason.TOOL_USE,
        usage=TokenUsage(prompt_tokens=20, output_tokens=5, total_tokens=25),
    )
    provider = _scripted_provider(
        [_truncated(), with_call, _clean("Written, in pieces.")]
    )
    session = _live_session(provider)

    answer = session.send_message("Write me a long file")

    assert len(provider._calls) == 3, (
        "the continuation's tool call was not dispatched: a recovery "
        "that only buys one more response has not made the turn usable"
    )
    answers = [
        p.function_response.call_id
        for m in session.get_history() for p in m.parts
        if p.function_response is not None
    ]
    assert "call_after" in answers, (
        f"the call the continuation made was never answered: {answers}"
    )
    assert "Written, in pieces." in answer


def test_the_continuation_carries_the_nudge():
    """The extra request has to say something useful, not just exist."""
    prompts = []
    provider = _scripted_provider([_truncated(), _clean()], prompts=prompts)
    session = _live_session(provider)

    session.send_message("Write me a long file")

    assert len(prompts) == 2
    last = prompts[-1].parts[0].text
    assert "max_tokens" in last
    assert "<truncated_output_excerpt>" in last


def test_the_continuations_are_bounded():
    """A truncation that recurs identically must not loop.

    Issue point 1.  Past the budget the turn ends the way it does
    today, with the reason preserved -- so the request count is
    ``1 + TRUNCATION_RECOVERY_BUDGET`` exactly, and the turn still
    reports ``max_tokens``.
    """
    provider = _scripted_provider([_truncated()])
    session = _live_session(provider)

    session.send_message("Write me a long file")

    assert len(provider._calls) == 1 + TRUNCATION_RECOVERY_BUDGET
    assert session.get_turn_accounting()[-1]["finish_reason"] == "max_tokens"


def test_the_budget_is_restored_for_the_next_turn():
    """Per-turn, not per-session.

    A session that spent its budget on one turn and could never recover
    again would degrade into today's behaviour after a single bad turn.
    """
    provider = _scripted_provider([_truncated()], hard_stop=99)
    session = _live_session(provider)

    session.send_message("first")
    spent_on_first = len(provider._calls)
    session.send_message("second")

    assert len(provider._calls) == 2 * spent_on_first


def test_a_safety_finish_is_not_re_driven():
    """Issue point 2, as behaviour rather than as a constant.

    Retrying a filtered turn is a different question and must not be
    answered by default.
    """
    filtered = ProviderResponse(
        parts=[Part(text="I can't help with")],
        finish_reason=FinishReason.SAFETY,
        usage=TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = _scripted_provider([filtered])
    session = _live_session(provider)

    session.send_message("something filtered")

    assert len(provider._calls) == 1


def test_the_parts_loop_continues_too():
    """A second, independent exit.

    ``_run_chat_loop_with_parts`` has its own inline abnormal check, so
    a fix applied only to the main loop holds exactly until an
    attachment is in the message.
    """
    provider = _scripted_provider([_truncated(), _clean("Short version.")])
    session = _live_session(provider)

    session.send_message_with_parts(
        [Part(text="Describe this and write it up")], lambda *a, **k: None,
    )

    assert len(provider._calls) == 2


# ==================== The operator still sees it (point 5) ============


def test_the_operator_is_told_about_a_recovered_truncation():
    """A recovered truncation is still worth seeing.

    The banner was #544's contribution and this does not take it away:
    the operator gets a note naming the cap and the attempt, and the
    turn record carries the count.
    """
    seen = []
    provider = _scripted_provider([_truncated(), _clean()])
    session = _live_session(provider)

    session.send_message(
        "Write me a long file",
        on_output=lambda source, text, mode: seen.append((source, text)),
    )

    notes = [t for s, t in seen if s == "system" and "max_tokens" in t]
    assert notes, f"no system note named the cap: {seen}"
    assert any("attempt 1/" in t for t in notes)
    assert session.get_turn_accounting()[-1]["truncation_recoveries"] == 1


def test_a_clean_turn_records_no_recoveries():
    """The counter is not decoration: a healthy turn never sets it."""
    provider = _scripted_provider([_clean()])
    session = _live_session(provider)

    session.send_message("hello")

    assert "truncation_recoveries" not in session.get_turn_accounting()[-1]


# ==================== No exit bypasses the recovery ===================


def _attribute_calls(node, name):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == name
    ]


def test_every_abnormal_exit_offers_the_continuation():
    """The two must stay paired, or a new exit silently loses turns.

    ``_finish_abnormally`` is the chokepoint every abnormal exit goes
    through (#751 pins that).  A function that consults it and then
    returns without offering the continuation is the defect this issue
    describes, reintroduced in one place -- and nothing else would
    notice, because every other artifact of the turn looks correct.
    """
    tree = ast.parse(SESSION_PATH.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in ("_finish_abnormally", "_finish_or_continue"):
            continue
        if not _attribute_calls(node, "_finish_abnormally"):
            continue
        if not _attribute_calls(node, "_recover_truncated_turn"):
            missing.append(node.name)

    assert missing == [], (
        f"{missing} end the turn on an abnormal finish without offering "
        f"the truncation continuation (#749) -- they call the bare "
        f"classifier instead of _finish_or_continue, which pairs the "
        f"two. An output-cap truncation reaching that exit is a lost "
        f"turn for any agent nothing else is driving."
    )
