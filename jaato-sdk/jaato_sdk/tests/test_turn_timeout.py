"""``timeout=`` on the facade's turn methods, and what it means when it fires.

WHY THE FACADE NEEDED A CLOCK AT ALL.

A fan-out driver runs N independent jobs and must be able to give up on one of
them.  The obvious place to put that bound is a cascade task pool's ``seconds``
ceiling, and it does not work: a pool is an aggregate over COMPLETED work,
reconciled when a session ENDS, so it never charges for a job that has not
finished — and the runaway job is precisely the one that has not finished.
Observed as ``cascade_remaining`` unchanged across two spawns while a job ran
past sixteen minutes.  The pool is coherent; it is simply not a timeout.

So the scaffolded sweep driver wrapped the facade in ``asyncio.wait_for``,
which is the plumbing the facade exists to own (jaato #826).  These tests pin
the parameter that let it stop.

WHAT A TIMEOUT DOES NOT DO: it does not stop the session.  It stops waiting.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jaato_sdk import Session, TurnTimeout
from jaato_sdk.events import EventType

from .test_convenience import FakeClient, _out, _TURN


async def test_ask_without_a_timeout_still_waits_unbounded():
    """The default is unchanged: a turn's natural length is the model's to
    decide, and guessing a ceiling for every caller would cut long agentic
    work short."""
    s = Session(FakeClient(fire=[_out("model", "hi"), _TURN]), "sid-1")
    assert await s.ask("q") == "hi"


async def test_ask_raises_turn_timeout_when_no_terminal_arrives():
    """A client that never fires a terminal — ``ask`` must not hang."""
    s = Session(FakeClient(fire=[]), "sid-1")
    with pytest.raises(TurnTimeout) as exc:
        await s.ask("q", timeout=0.05)
    assert exc.value.timeout == 0.05


async def test_complete_raises_turn_timeout_when_no_terminal_arrives():
    s = Session(FakeClient(fire=[]), "sid-1")
    with pytest.raises(TurnTimeout):
        await s.complete("q", timeout=0.05)


async def test_stream_raises_turn_timeout_when_no_terminal_arrives():
    s = Session(FakeClient(fire=[_out("model", "partial")]), "sid-1")
    seen = []
    with pytest.raises(TurnTimeout):
        async for chunk in s.stream("q", timeout=0.05):
            seen.append(chunk)
    assert seen == ["partial"], "chunks already yielded stay yielded"


async def test_a_turn_that_finishes_in_time_is_not_timed_out():
    s = Session(FakeClient(fire=[_out("model", "hi"), _TURN]), "sid-1")
    assert await s.ask("q", timeout=30.0) == "hi"


async def test_complete_returns_the_payload_within_its_timeout():
    fire = [
        (EventType.AGENT_STATUS_CHANGED,
         SimpleNamespace(agent_id="main", status="active")),
        (EventType.AGENT_COMPLETED,
         SimpleNamespace(agent_id="main", payload={"spoken": "a rainbow"})),
        (EventType.SESSION_TERMINATED, SimpleNamespace(reason="natural")),
    ]
    s = Session(FakeClient(fire=fire), "sid-1")
    assert await s.complete("q", timeout=30.0) == {"spoken": "a rainbow"}


async def test_the_settle_grace_is_not_the_callers_timeout():
    """Two clocks, and they mean different things.

    ``SETTLE_GRACE`` bounds ONE proposal's confirmation and SETTLES on it —
    a daemon that goes quiet between a turn and its status event costs
    accuracy, never a wait with no end (#767).  The caller's ``timeout``
    bounds the whole call and RAISES.  Conflating them would turn a healthy
    single-turn session into a timeout the moment the status event was late.
    """
    from jaato_sdk.client import convenience

    fire = [
        (EventType.AGENT_STATUS_CHANGED,
         SimpleNamespace(agent_id="main", status="active")),
        (EventType.TURN_COMPLETED, SimpleNamespace(agent_id="main")),
        # ...and NO confirming status event: the proposal must settle on
        # SETTLE_GRACE rather than raise TurnTimeout.
    ]
    original = convenience.SETTLE_GRACE
    convenience.SETTLE_GRACE = 0.01
    try:
        s = Session(FakeClient(fire=fire), "sid-1")
        assert await s.complete("q", timeout=30.0) is None
    finally:
        convenience.SETTLE_GRACE = original


def test_turn_timeout_is_catchable_as_asyncio_timeout_error():
    """Code already written around ``asyncio.wait_for`` keeps working.

    A driver that adopts the parameter should not have to rewrite the handler
    it already had, and ``asyncio.TimeoutError`` is the builtin
    ``TimeoutError`` from 3.11 onward.
    """
    assert issubclass(TurnTimeout, TimeoutError)
    assert issubclass(TurnTimeout, asyncio.TimeoutError)
