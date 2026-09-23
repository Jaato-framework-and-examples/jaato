"""A double-JSON-encoded tool-call argument reaches the tool with real
newlines, not the two-character literal ``\\n`` (issue #1242).

The reported symptom: a ``request_clarification`` prompt rendered its
multi-paragraph ``context`` / ``question_text`` with visible ``\\n``
sequences instead of line breaks.  ``white-space: pre-wrap`` in the web
client renders a real ``U+000A`` as a break and does nothing for the
two-char sequence backslash-n, so the string arriving at the browser
already held the literal escape.

The whole delivery chain preserves a real newline end to end -- the
provider decodes the wire ``arguments`` JSON exactly once, the runner ->
daemon RPC round-trips it, ``ClarificationBatchEvent`` carries it as a
``str`` field, and ``serialize_event`` -> ``JSON.parse`` restores it.  So
a literal ``\\n`` at the browser means a literal ``\\n`` was already in
``FunctionCall.args``, i.e. the wire ``arguments`` was **JSON-encoded a
second time**: the single decode then yields a ``str`` (the arguments
object as text) rather than a mapping, whose ``\\n`` escapes only resolve
on a second decode.

``parse_tool_call_arguments`` now recovers that object one level down --
but ONLY when the inner value is itself an object, so a genuinely
malformed slot stays unreadable (#750) and a genuine literal backslash-n
inside a *normally*-encoded object is never touched (its first decode is
already a ``dict``).

These tests drive the **real** MiniMax streaming loop, so they fail if
the recovery is removed from ``parse_tool_call_arguments``.
"""

import json
from unittest.mock import MagicMock

from jaato_sdk.events import (
    ClarificationBatchEvent,
    deserialize_event,
    serialize_event,
)
from jaato_sdk.plugins.model_provider.types import parse_tool_call_arguments
from jaato_server.shared.plugins.clarification.channels import question_payload
from jaato_server.shared.plugins.clarification.plugin import ClarificationPlugin
from jaato_server.shared.plugins.model_provider.minimax.provider import MiniMaxProvider


# The multi-line clarification the model meant to send.
_CONTEXT = "Para 1.\n\nPara 2.\n- bullet a\n- bullet b"
_QUESTION = "Line 1.\n\nOpen questions:\n1. first\n2. second"


def _double_encoded_arguments() -> str:
    """The wire ``arguments`` when the payload is JSON-encoded twice.

    ``json.dumps`` once is the correct wire form; a second ``json.dumps``
    turns that text into a JSON *string* -- the shape a gateway or a
    provider that stringifies an already-serialised argument produces.
    """
    obj = {
        "context": _CONTEXT,
        "questions": [{"text": _QUESTION, "question_type": "free_text"}],
    }
    return json.dumps(json.dumps(obj))


# ---------------------------------------------------------------------------
# parse_tool_call_arguments — the seam the recovery lives in
# ---------------------------------------------------------------------------

def test_double_encoded_arguments_recover_with_real_newlines():
    args, unreadable = parse_tool_call_arguments(_double_encoded_arguments())
    assert unreadable is None, "a recoverable double-encoded payload is not unreadable"
    assert args["context"] == _CONTEXT
    assert "\n" in args["context"] and "\\n" not in args["context"]
    assert args["questions"][0]["text"] == _QUESTION


def test_a_normally_encoded_object_is_untouched_including_literal_backslash_n():
    """The genuine-literal guard: a backslash-n a user really typed (a
    question quoting code) survives, because a single-encoded object is
    returned by the first decode and never reaches the second."""
    raw = json.dumps({"context": r"regex: \n matches a newline"})
    args, unreadable = parse_tool_call_arguments(raw)
    assert unreadable is None
    # The literal two-char sequence is preserved verbatim, not "fixed".
    assert args["context"] == r"regex: \n matches a newline"
    assert "\\n" in args["context"]
    assert "\n" not in args["context"]


def test_a_bare_json_string_stays_unreadable():
    """Recovery fires only when the inner value is an OBJECT -- a bare
    string is a malformed slot, not a double-encoded mapping (#750)."""
    args, unreadable = parse_tool_call_arguments('"just a string"')
    assert args == {}
    assert unreadable == '"just a string"'


# ---------------------------------------------------------------------------
# End to end: the real MiniMax loop -> the clarification batch event
# ---------------------------------------------------------------------------

def _tc(index, *, call_id, name, arguments):
    tc = MagicMock()
    tc.index = index
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _chunk(tool_calls=None, finish_reason=None):
    chunk = MagicMock()
    chunk.usage = None
    choice = MagicMock()
    choice.finish_reason = finish_reason
    delta = MagicMock()
    delta.content = None
    delta.tool_calls = tool_calls
    delta.reasoning_content = None
    delta.reasoning_details = None
    delta.audio = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    return stream


def _minimax_tool_call(arguments: str):
    """Drive the real MiniMax streaming loop and return the sole call."""
    provider = MiniMaxProvider()
    provider._client = MagicMock()
    provider._model_name = "MiniMax-M3"
    provider._enable_thinking = False
    provider._trace = lambda _msg: None
    chunks = [
        _chunk(tool_calls=[_tc(0, call_id="c1",
                               name="request_clarification",
                               arguments=arguments)]),
        _chunk(finish_reason="tool_calls"),
    ]
    provider._client.chat.completions.create = lambda **kw: _stream(chunks)
    result = provider.complete([], on_chunk=lambda _c: None)
    response = getattr(result, "response", result)
    calls = [p.function_call for p in response.parts if p.function_call]
    assert len(calls) == 1
    return calls[0]


def test_minimax_loop_recovers_double_encoded_clarification_args():
    fc = _minimax_tool_call(_double_encoded_arguments())
    # The tool is runnable (recovered), not refused as unreadable.
    assert fc.unreadable_args is None
    assert fc.name == "request_clarification"
    assert fc.args["context"] == _CONTEXT
    assert "\n" in fc.args["context"] and "\\n" not in fc.args["context"]


def test_recovered_args_carry_real_newlines_through_the_batch_event():
    """The end-to-end #1242 assertion: from the wire double-encoding to a
    serialized ClarificationBatchEvent the client parses, newlines are
    real, not literal ``\\n``."""
    fc = _minimax_tool_call(_double_encoded_arguments())
    request = ClarificationPlugin()._parse_request(fc.args)

    event = ClarificationBatchEvent(
        agent_id="",
        request_id="r1",
        tool_name="request_clarification",
        context=request.context,
        questions=[question_payload(i, q)
                   for i, q in enumerate(request.questions, 1)],
        batch_only=True,
    )
    # serialize_event -> the browser's JSON.parse (deserialize_event here).
    back = deserialize_event(serialize_event(event))
    assert back.context == _CONTEXT
    assert "\n" in back.context and "\\n" not in back.context
    assert back.questions[0]["text"] == _QUESTION
    assert "\n" in back.questions[0]["text"]
