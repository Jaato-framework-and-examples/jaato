"""Batched clarification handling in the TUI (#704).

The defect these cover: a runner-tier session relays a whole
``request_clarification`` in one ``ClarificationBatchEvent`` and blocks
until the client answers it.  The TUI had no branch for that event, so it
never prompted, never answered, and the turn hung with no way out.

The cases below pin the four things a client owes such a batch — prompt
once, walk every question, send exactly one reply, and offer a way out —
plus the thing it must NOT do: prompt twice when the same event arrives
as a preview of the per-question flow.
"""

import asyncio

from jaato_sdk.events import (
    ClarificationBatchEvent,
    ClarificationInputModeEvent,
)

import clarification_batch as cb


# ---------------------------------------------------------------------------
# Fakes.  Each records what the module did to it; none simulate rendering.
# ---------------------------------------------------------------------------

class FakeBuffer:
    """Stands in for an ``OutputBuffer``, recording tool-tree calls."""

    def __init__(self):
        self.appended = []          # (source, text, mode)
        self.awaiting = []          # (tool_name, index, total)
        self.resolved = []          # (tool_name, qa_pairs)

    def append(self, source, text, mode):
        self.appended.append((source, text, mode))

    def set_tool_awaiting_clarification(self, tool_name, index, total):
        self.awaiting.append((tool_name, index, total))

    def set_tool_clarification_resolved(self, tool_name, qa_pairs):
        self.resolved.append((tool_name, qa_pairs))


class FakeRegistry:
    """Single-buffer agent registry."""

    def __init__(self, buffer):
        self.buffer = buffer

    def get_buffer(self, agent_id):
        return None

    def get_selected_buffer(self):
        return self.buffer


class FakeDisplay:
    def __init__(self):
        self.waiting = []
        self.refreshes = 0

    def set_waiting_for_channel_input(self, waiting, response_options=None):
        self.waiting.append(waiting)

    def refresh(self):
        self.refreshes += 1


class FakeClient:
    """Records the replies the module sends back to the daemon."""

    def __init__(self):
        self.batches = []           # (request_id, answers, cancelled)
        self.singles = []           # (request_id, response)

    async def respond_to_clarification_batch(self, request_id, answers,
                                             cancelled=False):
        self.batches.append((request_id, list(answers), cancelled))

    async def respond_to_clarification(self, request_id, response):
        self.singles.append((request_id, response))


def _env():
    buffer = FakeBuffer()
    return buffer, FakeRegistry(buffer), FakeDisplay(), FakeClient()


def _batch_event(questions, *, batch_only=True, context="why we ask"):
    return ClarificationBatchEvent(
        agent_id="",
        request_id="req-1",
        tool_name="request_clarification",
        context=context,
        questions=questions,
        batch_only=batch_only,
    )


SINGLE_CHOICE = {
    "index": 1,
    "text": "Which database?",
    "question_type": "single_choice",
    "required": True,
    "choices": [{"text": "postgres"}, {"text": "sqlite", "default": True}],
    "default_choice": 2,
}

FREE_TEXT = {
    "index": 2,
    "text": "Anything else?",
    "question_type": "free_text",
    "required": False,
}


def _trace(_msg):
    pass


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_context_is_shown_once_above_the_first_question():
    first = cb.format_question_lines(SINGLE_CHOICE, 1, 2, "why we ask")
    second = cb.format_question_lines(FREE_TEXT, 2, 2, "why we ask")
    assert first[0] == "Context: why we ask"
    assert not any(line.startswith("Context:") for line in second)


def test_choices_are_numbered_and_the_default_is_marked():
    lines = cb.format_question_lines(SINGLE_CHOICE, 1, 1)
    assert "    1. postgres" in lines
    assert "    2. sqlite (default)" in lines
    assert lines[-1] == "  Enter choice [1-2], or 'cancel':"


def test_every_question_advertises_the_way_out():
    """Cancelling is the only exit from a question the user cannot answer."""
    for question in (SINGLE_CHOICE, FREE_TEXT):
        lines = cb.format_question_lines(question, 1, 1)
        assert "cancel" in lines[-1]


# ---------------------------------------------------------------------------
# Answer summaries (tool-tree display)
# ---------------------------------------------------------------------------

def test_a_choice_is_summarised_by_its_text_not_its_number():
    assert cb.answer_summary(SINGLE_CHOICE, "1") == "postgres"


def test_an_empty_choice_answer_falls_back_to_the_default():
    assert cb.answer_summary(SINGLE_CHOICE, "") == "sqlite"


def test_an_unanswered_optional_question_reads_as_skipped():
    assert cb.answer_summary(FREE_TEXT, "") == "(skipped)"


def test_free_text_is_summarised_verbatim():
    assert cb.answer_summary(FREE_TEXT, "  ship it ") == "ship it"


# ---------------------------------------------------------------------------
# The batch flow
# ---------------------------------------------------------------------------

def test_a_preview_batch_does_not_prompt():
    """batch_only=False means the per-question flow follows; prompting on
    both would ask the user everything twice."""
    _buffer, registry, display, client = _env()
    event = _batch_event([SINGLE_CHOICE], batch_only=False)

    pending = asyncio.run(cb.enter_clarification_input_mode(
        event, None, client, registry, display, _trace))

    assert pending is None
    assert display.waiting == []
    assert client.batches == []


def test_a_batch_only_request_prompts_for_the_first_question():
    buffer, registry, display, client = _env()
    event = _batch_event([SINGLE_CHOICE, FREE_TEXT])

    pending = asyncio.run(cb.enter_clarification_input_mode(
        event, None, client, registry, display, _trace))

    assert pending["request_id"] == "req-1"
    assert pending["batch"] is True
    assert display.waiting == [True]
    assert buffer.awaiting == [("request_clarification", 1, 2)]
    source, text, mode = buffer.appended[0]
    assert (source, mode) == ("clarification", "write")
    assert "Which database?" in text


def test_answers_are_held_until_the_last_question_then_sent_as_one_batch():
    buffer, registry, display, client = _env()
    event = _batch_event([SINGLE_CHOICE, FREE_TEXT])

    async def walk():
        pending = await cb.enter_clarification_input_mode(
            event, None, client, registry, display, _trace)
        pending = await cb.submit_clarification_answer(
            client, pending, "1", registry, display)
        assert client.batches == [], "no reply until every question is answered"
        assert buffer.awaiting[-1] == ("request_clarification", 2, 2)
        return await cb.submit_clarification_answer(
            client, pending, "nope", registry, display)

    pending = asyncio.run(walk())

    assert pending is None, "the batch is finished, nothing is pending"
    assert client.batches == [("req-1", ["1", "nope"], False)]
    assert buffer.resolved == [
        ("request_clarification", [("Which database?", "postgres"),
                                   ("Anything else?", "nope")]),
    ]
    assert display.waiting[-1] is False


def test_cancelling_abandons_the_batch_instead_of_answering_it():
    """The escape hatch: without it an unanswerable question is terminal for
    the session, because cancelling a turn does not interrupt a blocked tool."""
    buffer, registry, display, client = _env()
    event = _batch_event([SINGLE_CHOICE, FREE_TEXT])

    async def walk():
        pending = await cb.enter_clarification_input_mode(
            event, None, client, registry, display, _trace)
        return await cb.submit_clarification_answer(
            client, pending, "cancel", registry, display)

    pending = asyncio.run(walk())

    assert pending is None
    assert client.batches == [("req-1", [], True)]
    assert buffer.resolved == [("request_clarification", None)]
    assert display.waiting[-1] is False


def test_a_batch_with_no_questions_is_answered_rather_than_left_blocking():
    _buffer, registry, display, client = _env()
    event = _batch_event([])

    pending = asyncio.run(cb.enter_clarification_input_mode(
        event, None, client, registry, display, _trace))

    assert pending is None
    assert client.batches == [("req-1", [], False)]


# ---------------------------------------------------------------------------
# The per-question flow is untouched
# ---------------------------------------------------------------------------

def test_a_per_question_request_replies_one_answer_at_a_time():
    buffer, registry, display, client = _env()
    event = ClarificationInputModeEvent(
        agent_id="", request_id="req-2", tool_name="request_clarification",
        question_index=1, total_questions=2,
    )

    async def walk():
        pending = await cb.enter_clarification_input_mode(
            event, None, client, registry, display, _trace)
        return pending, await cb.submit_clarification_answer(
            client, pending, "1", registry, display)

    pending, after = asyncio.run(walk())

    assert pending["request_id"] == "req-2"
    assert "batch" not in pending
    assert buffer.awaiting == [("request_clarification", 1, 2)]
    assert client.singles == [("req-2", "1")]
    assert client.batches == []
    # The daemon drives the next question and clears the state via
    # ClarificationResolvedEvent, so the client keeps waiting on it.
    assert after is pending
