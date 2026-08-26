"""``create_session`` answered every failure with the same ``None``.

Five things produce it -- the command never left the client, the daemon
refused, the wait timed out, the connection dropped, or the recovery client
had no inner client -- and they were MEASURED returning ``None`` from the same
call, four of them indistinguishable even by elapsed time:

    1 send failed              -> None  in 1.00s
    2 timeout, silent          -> None  in 1.00s
    3 timeout, daemon alive    -> None  in 1.00s
    4 disconnect mid-wait      -> None  in 0.20s
    5 daemon refused           -> None  in 1.00s

The information was never missing -- these are five distinct branches inside
the client.  It was DISCARDED at the point of failure.

The axis that matters is not the mechanism but ``may_exist``: whether a
session may ALREADY be running on the daemon.  ``session.new`` has no
idempotency key (``request_id`` is echoed for correlation, never used to
dedupe), so a retry after an unconfirmed create makes a SECOND session holding
its own runner subprocess and pool slot.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from jaato_sdk import (
    IPCClient,
    SessionCreateFailed,
    SessionNotConfirmed,
    SessionNotSent,
    SessionRefused,
)
from jaato_sdk.events import ErrorEvent, SessionInfoEvent


def _client(*, write=None):
    """A client wired to nothing, with the outgoing request_id captured.

    ``sent`` collects what ``create_session`` puts on the wire so a test can
    echo the SAME ``request_id`` back.  That matters: the daemon echoes it and
    the client CORRELATES on it, so an answer without it is filed as
    incidental and the wait runs to its timeout.  Feeding an uncorrelated
    event would make every test here look like the timeout case regardless of
    what it was actually testing.
    """
    c = IPCClient.__new__(IPCClient)
    c._buffered_events, c._event_subscribers = [], []
    c._session_id, c._protocol_version, c._server_info = None, "1.3", {}
    c._server_protocol_version = "1.3"
    c.sent = []

    async def _capture(data):
        c.sent.append(json.loads(data.decode() if isinstance(data, bytes) else data))
        return None

    c._write_message = write or _capture

    async def _disc():
        return None

    c.disconnect = _disc
    return c


def _request_id(client):
    """The ``request_id`` create_session just wrote, for echoing back."""
    for msg in client.sent:
        rid = (msg.get("payload") or {}).get("request_id")
        if rid:
            return rid
    return None


def _feed(client, make_event, *, after=0.02):
    """Deliver ``make_event(request_id)`` once create_session has subscribed.

    Takes a factory rather than an event because the request_id does not exist
    until create_session generates it.
    """
    async def _go():
        await asyncio.sleep(after)
        event = (make_event(_request_id(client))
                 if callable(make_event) else make_event)
        for q in list(client._event_subscribers):
            q.put_nowait(event)

    asyncio.create_task(_go())


# ---------------------------------------------------------------- not sent

@pytest.mark.asyncio
async def test_a_failed_send_does_not_wait_for_an_answer():
    """The bug that cost a full minute per occurrence.

    ``_send_event`` catches the ConnectionError, logs it, disconnects -- and
    used to return ``None``, which is what "sent fine" also returned.  So
    ``create_session`` waited out its whole timeout for a reply to a command
    it had ALREADY been told never left the process, then reported a timeout,
    blaming the daemon for a local socket fault.
    """
    async def _boom(_data):
        raise ConnectionError("socket is gone")

    c = _client(write=_boom)

    t0 = time.monotonic()
    with pytest.raises(SessionNotSent) as excinfo:
        await c.create_session(name="n", timeout=5.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, (
        f"waited {elapsed:.2f}s for a reply to a command that was never "
        "sent (the timeout was 5.0s)"
    )
    assert excinfo.value.may_exist is False
    assert excinfo.value.cause == "not_sent"


# ----------------------------------------------------------------- refused

@pytest.mark.asyncio
async def test_a_refusal_carries_the_daemons_own_reason():
    """Not a summary of it, and not a guess at it.

    The facade used to render every failure as "check provider auth" -- one of
    five causes, wrong for the other four.  A hardcoded likely-cause sends the
    reader somewhere specific and wrong, which is worse than saying nothing.
    """
    c = _client()
    _feed(c, lambda rid: ErrorEvent(error="Profile 'nope' not found",
                                    error_type="ProfileNotFoundError",
                                    recoverable=True,
                                    request_id=rid))

    with pytest.raises(SessionRefused) as excinfo:
        await c.create_session(name="n", timeout=5.0)

    assert "Profile 'nope' not found" in str(excinfo.value)
    assert excinfo.value.error_type == "ProfileNotFoundError"
    assert excinfo.value.may_exist is False


# ----------------------------------------------------------- not confirmed

@pytest.mark.asyncio
async def test_a_timeout_admits_a_session_may_exist():
    c = _client()

    with pytest.raises(SessionNotConfirmed) as excinfo:
        await c.create_session(name="n", timeout=0.15)

    assert excinfo.value.may_exist is True
    assert excinfo.value.cause == "timeout"
    # The warning a retrying caller needs, in the message it will actually
    # read -- a duplicate here costs a runner subprocess and a pool slot.
    assert "SECOND" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_disconnect_admits_a_session_may_exist():
    """Same axis as the timeout: the command was already on the wire."""
    c = _client()
    _feed(c, None)          # the drain loop's disconnect sentinel

    with pytest.raises(SessionNotConfirmed) as excinfo:
        await c.create_session(name="n", timeout=5.0)

    assert excinfo.value.may_exist is True
    assert excinfo.value.cause == "disconnect"


# ------------------------------------------------------- the axis, directly

@pytest.mark.asyncio
async def test_the_two_definite_failures_are_retry_safe():
    """``may_exist`` is the whole point, so it is asserted as a set.

    A caller that branches on nothing else stays correct: False means nothing
    was created and a retry cannot duplicate; True means look before you
    create.
    """
    assert SessionNotSent("x").may_exist is False
    assert SessionRefused("x").may_exist is False
    assert SessionNotConfirmed("x").may_exist is True


def test_every_failure_is_catchable_as_one_thing():
    for exc in (SessionNotSent("x"), SessionRefused("x"),
                SessionNotConfirmed("x")):
        assert isinstance(exc, SessionCreateFailed)


def test_the_base_is_a_runtime_error_on_purpose():
    """The compatibility guarantee, pinned.

    A survey of 18 repositories found that every existing handler around
    ``create_session`` catches ``RuntimeError`` -- because the line under the
    call read ``if not sid: raise RuntimeError(...)``.  Consumers had already
    written this change by hand, locally.  Inheriting from ``RuntimeError``
    means those handlers keep catching instead of being bypassed, which is the
    difference between a migration and an outage.

    Changing this base class silently converts working handlers in other
    repositories into uncaught exceptions.  If it must change, that is a
    coordinated release, not an edit.
    """
    assert issubclass(SessionCreateFailed, RuntimeError)


def test_the_exceptions_are_importable_from_the_package_root():
    """An exception nobody can conveniently import is one nobody catches.

    The SDK's pre-existing ``ReconnectingError`` / ``ConnectionClosedError``
    were reachable only via ``jaato_sdk.client.recovery``, and its
    ``ConnectionError`` SHADOWS the builtin without subclassing it -- so
    consumers writing the obvious ``except ConnectionError`` caught the
    builtin and missed the SDK's entirely.  Two out-of-tree consumers were
    found doing exactly that.
    """
    import jaato_sdk

    for name in ("SessionCreateFailed", "SessionNotSent", "SessionRefused",
                 "SessionNotConfirmed", "ReconnectingError",
                 "ConnectionClosedError"):
        assert hasattr(jaato_sdk, name), f"{name} is not importable from jaato_sdk"
        assert name in jaato_sdk.__all__, f"{name} is missing from __all__"


# ------------------------------------------------------------ the happy path

@pytest.mark.asyncio
async def test_success_still_returns_the_id():
    """The control.  A contract change that breaks the working path is not a
    contract change, it is a regression."""
    c = _client()
    _feed(c, lambda rid: SessionInfoEvent(session_id="sess-ok",
                                          request_id=rid))

    assert await c.create_session(name="n", timeout=5.0) == "sess-ok"
