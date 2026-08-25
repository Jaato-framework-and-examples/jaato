"""Frames must survive the INNER wrap, where the exception is caught.

#613 taught the OUTER wrap to carry ``ErrorPayload.traceback`` — and the
probe's crash still arrived frameless, because the frames were discarded one
layer further in.

``_handle_session_send_message`` catches the model-loop exception and RETURNS
a failure dict rather than raising:

    return False, {"error": f"...model loop raised {type(exc).__name__}: {exc}",
                   "stage": "send"}

So the dispatcher builds ``ErrorPayload`` from that DICT via
``_extract_error_message`` — there is no exception left to read a traceback
off, and ``ErrorPayload.traceback`` was therefore always ``None`` on this
path.  #613 carried frames that were never created.

Both of us were right and both readings were incomplete: the fix worked on
envelopes that HAD frames, and this path never made any.

The message is what hid it: ``model loop raised AttributeError: <text>`` has
the type and the text, so it reads like a complete report of an exception.
What it cannot say is that it is a SUMMARY — the discarded frames are
invisible in its own output.
"""

import json

import pytest

from server.runner.envelope import ErrorPayload, ResponseEnvelope
from server.runner.rpc import _extract_error_message, _extract_error_traceback
from server.runner_rpc_client import _call_error


FRAMES = (
    'Traceback (most recent call last):\n'
    '  File "jaato_session.py", line 4888, in _emit_text_parts\n'
    '    if part.text:\n'
    "AttributeError: 'NoneType' object has no attribute 'text'"
)


def _failure(with_frames=True):
    d = {"error": "session.send_message: model loop raised AttributeError: "
                  "'NoneType' object has no attribute 'text'",
         "stage": "send"}
    if with_frames:
        d["traceback"] = FRAMES
    return d


def test_the_model_loop_catch_captures_frames():
    """The half #613 missed: this path RETURNS, so nothing else can."""
    import inspect
    from server.runner.rpc import RunnerRPC
    src = inspect.getsource(RunnerRPC._handle_session_send_message)
    assert "traceback.format_exc()" in src, (
        "the model-loop catch stringifies the exception without keeping "
        "frames — nothing downstream can recover them"
    )
    assert "sanitize_traceback" in src, "frames must be sanitized (§3.1)"


def test_a_failure_dict_can_carry_its_own_frames():
    assert _extract_error_traceback(_failure()) == FRAMES


def test_absent_frames_stay_absent():
    """No placeholder — it would read like evidence to a debugger."""
    assert _extract_error_traceback(_failure(with_frames=False)) is None
    assert _extract_error_traceback("not a dict") is None


def test_the_dispatcher_carries_them_into_the_payload():
    result = _failure()
    err = ErrorPayload(type="ToolError",
                       message=_extract_error_message(result),
                       traceback=_extract_error_traceback(result))
    assert err.traceback == FRAMES
    assert "model loop raised" in err.message


def test_the_frames_reach_the_daemon_across_the_wire():
    """End to end, through a real JSON round-trip."""
    result = _failure()
    env = ResponseEnvelope(
        id=1, ok=False, result=result,
        error=ErrorPayload(type="ToolError",
                           message=_extract_error_message(result),
                           traceback=_extract_error_traceback(result)),
    )
    back = ResponseEnvelope.from_dict(json.loads(json.dumps(env.to_dict())))
    err = _call_error("session.send_message", back)

    assert err.traceback_text is not None, "frames lost crossing the wire"
    assert "line 4888" in err.traceback_text, (
        "the point of frames is naming the LINE; the summary already had "
        "the type and the message"
    )


def test_the_summary_line_is_unchanged():
    """Adding frames must not alter the message anyone already parses."""
    result = _failure()
    assert _extract_error_message(result).startswith(
        "session.send_message: model loop raised AttributeError")
