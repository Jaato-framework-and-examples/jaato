"""Runner-side frames must reach a consumer, not die at the RPC boundary.

The runner sanitizes every executor exception and ships the frames in
``ErrorPayload.traceback`` (``runner/rpc.py``); ``envelope.py`` puts them on
the wire.  **Nothing daemon-side read them.**  All three
``RunnerCallError`` raise-sites rebuilt the message from ``error.type`` and
``error.message`` and dropped ``error.traceback`` on the floor.

So a crash inside a model loop reached every witness — the ErrorEvent, its
``details``, the daemon log, the runner log — as ONE SANITIZED LINE:

    session.send_message failed: ToolError: session.send_message: model loop
    raised AttributeError: 'NoneType' object has no attribute 'text'

Exception type and message intact, frames gone.  A line that reads like a
finished error is worse for a consumer than an obviously truncated one: the
cascade-coordination probe spent twenty minutes assuming it had the wrong log
rather than that the frames had been dropped.

Eleventh instance in this arc of *the channel exists and one end doesn't use
it* — and this one was the channel for diagnosing the other ten.
"""

import logging

import pytest

from server.runner.envelope import ErrorPayload, ResponseEnvelope
from server.runner_rpc_client import RunnerCallError, _call_error


FRAMES = (
    'Traceback (most recent call last):\n'
    '  File "jaato_session.py", line 4888, in _emit_text_parts\n'
    '    if part.text:\n'
    "AttributeError: 'NoneType' object has no attribute 'text'"
)


def _env(traceback_text=FRAMES):
    return ResponseEnvelope(
        id=1, ok=False, result=None,
        error=ErrorPayload(
            type="ToolError",
            message="model loop raised AttributeError: ...",
            traceback=traceback_text,
        ),
    )


def test_the_frames_survive_the_boundary():
    """The whole finding: they were populated, shipped, and dropped."""
    err = _call_error("session.send_message", _env())
    assert err.traceback_text == FRAMES


def test_the_message_is_unchanged():
    """Carrying frames must not alter what already worked."""
    err = _call_error("session.send_message", _env())
    assert "session.send_message failed: ToolError:" in str(err)


def test_an_envelope_without_frames_is_not_invented():
    """Absent frames stay absent — no placeholder that reads like evidence."""
    assert _call_error("x", _env(traceback_text=None)).traceback_text is None


def test_a_missing_error_payload_does_not_crash():
    env = ResponseEnvelope(id=1, ok=False, result=None, error=None)
    err = _call_error("session.bootstrap", env)
    assert err.traceback_text is None
    assert "UnknownError" in str(err)


def test_every_raise_site_uses_the_helper():
    """Three sites dropped the frames identically; a fourth would too.

    Asserted structurally rather than by fixing the three: the defect was
    that the same two lines were written three times, so the guard has to be
    about the SHAPE, not the instances.
    """
    import pathlib
    src = pathlib.Path(
        "jaato-server/server/runner_rpc_client.py").read_text(encoding="utf-8")
    assert "response.error.message if response.error" not in src, (
        "a raise-site is rebuilding the message by hand and dropping "
        "error.traceback — use _call_error()"
    )


def test_the_terminal_error_path_surfaces_frames_to_both_witnesses():
    """Log AND event: one serves whoever is on the machine, one whoever isn't.

    Emitting to only one leaves a remote consumer — which is what a cascade
    driver is — with the same single line it had before.
    """
    import inspect
    from server import core
    src = inspect.getsource(core)
    idx = src.index("MODEL_THREAD_TERMINAL_ERROR error_type=%s")
    window = src[idx - 900:idx + 900]
    assert "traceback_text" in window, "frames not read at the terminal site"
    assert "runner_traceback" in window, "frames not put on the ErrorEvent"


def test_error_event_can_carry_the_frames():
    """``details`` is the machine-readable slot; it must accept this."""
    from jaato_sdk.events import ErrorEvent
    ev = ErrorEvent(error="boom", error_type="RunnerCallError",
                    details={"runner_traceback": FRAMES})
    assert ev.details["runner_traceback"] == FRAMES
    assert "AttributeError" in ev.to_dict()["details"]["runner_traceback"]
