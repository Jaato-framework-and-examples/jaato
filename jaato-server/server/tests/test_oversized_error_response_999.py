"""An oversized ERROR response is answered, not dropped (#999).

WHAT WAS WRONG.  #920 made the runner refuse to *write* a frame over
``MAX_MESSAGE_SIZE`` — the peer cannot skip one (the length prefix is read,
the body is not, so the stream desynchronises and the whole transport
closes) — and turned that drop into a small typed error for the call it
belonged to.  The substitution was guarded by::

    if not ok or self._closed:
        # An error frame that did not fit will not fit a second
        # time, and a closed channel has nowhere to put either.
        return

**The premise names the wrong frame.**  The substitute is a different,
bounded frame with no ``result`` and no traceback, so "will not fit a second
time" is not true of it.  What did not fit is the ORIGINAL, whose ``result``
dict carries megabytes of tool output beside ``ErrorPayload.traceback`` —
and ``_handle_request`` passes exactly that on its domain-failure branch.

So an error response over the cap was dropped in full, **nothing was
written**, and ``_closed`` stayed ``False``: the channel open, the runner
idle with an empty active set, the daemon waiting.  #856's ack watchdog
bounds that at 120 s with a ``RunnerResultLost`` verdict — the correct
classification, since the work DID run — but it cannot recover the reason
the runner already had in hand and threw away.

WHY IT WAS LUCK THAT #920 WORKED AT ALL.  The success path fails loudly on
SIZE before it can fail quietly on content.  Its substitute is small for the
same reason this one is; nothing enforced that, which is why
``OVERSIZE_MESSAGE_CHARS`` bounds the borrowed text here rather than the
call site asserting it.
"""

from __future__ import annotations

import socket

import pytest

from shared.framing import MAX_MESSAGE_SIZE, read_frame_sync
from server.runner.envelope import ErrorPayload
from server.runner.json_codec import loads as _json_loads
from server.runner.rpc import OVERSIZE_MESSAGE_CHARS, RunnerRPC

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_RPC = "jaato-server/server/runner/rpc.py"


REVERSIONS = [
    Reversion(
        target=_RPC,
        find=(
            "        if self._closed:\n"
            "            # A closed channel has nowhere to put either.  This is the\n"
            "            # only half of the old guard that was ever true.\n"
            "            return"
        ),
        replace=(
            "        if not ok or self._closed:\n"
            "            # A closed channel has nowhere to put either.  This is the\n"
            "            # only half of the old guard that was ever true.\n"
            "            return"
        ),
        test="test_an_oversized_error_response_still_answers_the_call",
        because="the pre-#999 guard is back, so an error response over the "
                "frame cap is dropped in full with the channel left open — "
                "the caller is answered only by #856's 120-second watchdog",
    ),
    Reversion(
        target=_RPC,
        find=(
            "            detail += (\n"
            '                f". The call had already FAILED with '
            '{error.type}: {message}"\n'
            "            )"
        ),
        replace="            detail += \".\"",
        test="test_the_substitute_names_the_failure_the_runner_already_knew",
        because="the substitute stops carrying the failure the runner already "
                "held, so the caller is told only that something did not fit",
    ),
    Reversion(
        target=_RPC,
        find=(
            "            if len(message) > OVERSIZE_MESSAGE_CHARS:\n"
            '                message = message[:OVERSIZE_MESSAGE_CHARS] + '
            '" […truncated]"'
        ),
        replace="            message = message",
        test="test_the_borrowed_message_is_bounded",
        because="the borrowed message is unbounded again, so the substitute "
                "can be as oversized as the frame it is substituting for — "
                "the unchecked 'it will fit' claim the fix exists to retire",
    ),
]



def _rpc_pair():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.settimeout(0.3)
    return RunnerRPC(a, lambda name, args: (True, {})), a, b


def _read(sock):
    """The next frame, decoded.  ``read_frame_sync`` hands back the raw JSON."""
    raw = read_frame_sync(sock)
    return None if raw is None else _json_loads(raw)


def _oversized() -> str:
    return "x" * (MAX_MESSAGE_SIZE + 1024)


# ------------------------------------------------------------------ the pin

def test_an_oversized_error_response_still_answers_the_call():
    """THE PIN.  On the pre-#999 tree this frame was never written."""
    rpc, a, b = _rpc_pair()
    try:
        rpc._emit_response(
            request_id=42, ok=False, result={"stdout": _oversized()},
            error=ErrorPayload(type="ToolError", message="the tool failed",
                               traceback="Traceback...\n" + _oversized()),
        )
        frame = _read(b)
        assert frame is not None, (
            "nothing was written, so the caller is answered only by #856's "
            "120-second watchdog — the state #999 is about"
        )
        assert frame["id"] == 42
        assert frame["ok"] is False
        assert frame["error"]["type"] == "FrameTooLargeError"
    finally:
        a.close()
        b.close()


def test_the_substitute_names_the_failure_the_runner_already_knew():
    """The runner holds the reason; telling the caller only "too large" loses it."""
    rpc, a, b = _rpc_pair()
    try:
        rpc._emit_response(
            request_id=43, ok=False, result={"stdout": _oversized()},
            error=ErrorPayload(type="ToolError",
                               message="permission denied on /etc/shadow"),
        )
        message = _read(b)["error"]["message"]
        assert "ToolError" in message
        assert "permission denied on /etc/shadow" in message
        assert str(MAX_MESSAGE_SIZE) in message
    finally:
        a.close()
        b.close()


def test_the_borrowed_message_is_bounded():
    """"Small by construction" is enforced here, not argued at the call site."""
    rpc, a, b = _rpc_pair()
    try:
        rpc._emit_response(
            request_id=44, ok=False, result=None,
            error=ErrorPayload(type="ToolError", message=_oversized()),
        )
        frame = _read(b)
        message = frame["error"]["message"]
        assert "truncated" in message
        assert len(message) < OVERSIZE_MESSAGE_CHARS + 500
    finally:
        a.close()
        b.close()


def test_the_traceback_is_not_carried():
    """It is usually what made the frame oversized in the first place."""
    rpc, a, b = _rpc_pair()
    try:
        rpc._emit_response(
            request_id=45, ok=False, result=None,
            error=ErrorPayload(type="ToolError", message="short",
                               traceback="SECRET-MARKER\n" + _oversized()),
        )
        frame = _read(b)
        assert "SECRET-MARKER" not in frame["error"]["message"]
        assert frame["error"].get("traceback") is None
    finally:
        a.close()
        b.close()


def test_a_failure_with_no_error_payload_still_says_it_failed():
    """``ok=False`` with no payload is rare and must not read as a success."""
    rpc, a, b = _rpc_pair()
    try:
        rpc._emit_response(
            request_id=46, ok=False, result={"stdout": _oversized()},
            error=None,
        )
        frame = _read(b)
        assert frame["ok"] is False
        assert "FAILED" in frame["error"]["message"]
    finally:
        a.close()
        b.close()


# ------------------------------------------------------- what did not change

def test_the_success_path_is_unchanged():
    """#920's case, still answered and still not claiming the call failed twice."""
    rpc, a, b = _rpc_pair()
    try:
        rpc._emit_response(request_id=41, ok=True,
                           result={"stdout": _oversized()})
        frame = _read(b)
        assert frame["ok"] is False          # a dropped result is not a success
        assert frame["error"]["type"] == "FrameTooLargeError"
        assert "FAILED" not in frame["error"]["message"]
        assert rpc._closed is False
    finally:
        a.close()
        b.close()


def test_a_closed_channel_is_still_the_one_true_half_of_the_old_guard():
    """It genuinely has nowhere to put either frame."""
    rpc, a, b = _rpc_pair()
    try:
        rpc._closed = True
        rpc._emit_response(
            request_id=47, ok=False, result={"stdout": _oversized()},
            error=ErrorPayload(type="ToolError", message="x"),
        )
        with pytest.raises(socket.timeout):
            read_frame_sync(b)
    finally:
        a.close()
        b.close()


def test_the_channel_survives_and_serves_the_next_call():
    """The whole reason the oversized frame is dropped rather than written."""
    rpc, a, b = _rpc_pair()
    try:
        rpc._emit_response(
            request_id=48, ok=False, result={"stdout": _oversized()},
            error=ErrorPayload(type="ToolError", message="boom"),
        )
        assert _read(b)["id"] == 48
        rpc._emit_response(request_id=49, ok=True, result={"fine": True})
        nxt = _read(b)
        assert nxt["id"] == 49 and nxt["ok"] is True
        assert rpc._closed is False
    finally:
        a.close()
        b.close()


def test_an_oversized_stream_chunk_is_still_dropped_silently():
    """A stated decision, not an inherited one (#999 asked for it in writing).

    A chunk is display output, not an answer: the call still ends in a
    response, so dropping one costs the viewer text and the caller nothing.
    """
    rpc, a, b = _rpc_pair()
    try:
        rpc._emit_stream(50, "agent", _oversized(), None)
        with pytest.raises(socket.timeout):
            read_frame_sync(b)
        assert rpc._closed is False
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------- last resort

def test_if_even_the_substitute_did_not_fit_the_borrowed_text_is_dropped():
    """The one thing a bounded substitute can still give up is the quotation.

    Unreachable with the bound above and a 10 MB cap; pinned because the
    defect being fixed *was* an unchecked claim that a frame would fit.
    """
    rpc, a, b = _rpc_pair()
    written = []

    def _fake_write(payload):
        written.append(payload)
        return len(written) >= 3          # original + informative both refused

    rpc._write = _fake_write
    try:
        rpc._emit_response(
            request_id=51, ok=False, result=None,
            error=ErrorPayload(type="ToolError", message="quote me"),
        )
        assert len(written) == 3
        assert "quote me" in written[1]["error"]["message"]
        assert "quote me" not in written[2]["error"]["message"]
        assert written[2]["error"]["type"] == "FrameTooLargeError"
        assert written[2]["ok"] is False
    finally:
        a.close()
        b.close()
