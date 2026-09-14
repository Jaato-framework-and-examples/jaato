"""A turn that ended the SESSION is not handed back as a turn (jaato #1007).

``ask`` settled on first-of ``{TURN_COMPLETED, SESSION_TERMINATED}``, and the
daemon emits those two from one thread 2-3 ms apart, turn first.  So a session
whose budget ceiling ended it answered ``''`` -- indistinguishable from a model
that said nothing -- and the caller found out on its NEXT send, which on a
cascade-stamped session never returned at all (that terminal also unloads the
session, the daemon replies ``ErrorEvent("Session not found")``, and nothing in
the facade listened).  Measured on a real run: 12+ minutes blocked.

Every test here is DETERMINISTIC: ``ScriptedClient`` delivers one event per
event-loop tick, so "did the call return before event 3 arrived?" is answered
by a counter rather than by a clock.  The two ``asyncio.wait_for`` bounds are
harness guards that turn a hang into a failure -- never the assertion itself.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jaato_sdk import AgentError, Session, SessionEnded
from jaato_sdk.client import convenience
from jaato_sdk.events import EventType
from jaato_sdk.tests.test_convenience import (FakeClient, ScriptedClient,
                                              _out, _status, _turn)


_DETAILS = {"dimension": "tokens", "used": 1200, "limit": 10}


def _terminal(reason, **kw):
    return (EventType.SESSION_TERMINATED,
            SimpleNamespace(reason=reason, error_type=None,
                            error_summary=None, **kw))


def _error(error, *, error_type="SessionError", session_id=None):
    return (EventType.ERROR,
            SimpleNamespace(error=error, error_type=error_type,
                            session_id=session_id))


#: The daemon's wire shape for the #1007 repro: the turn ends, and the ceiling
#: ends the SESSION two events later.  The turn event comes FIRST, which is the
#: whole defect -- a rule that settles on it never sees the terminal.
_BUDGET_END = [
    _status("main", "active"),
    _out("model", ""),
    _turn("main"),
    _terminal("budget_exhausted", details=_DETAILS),
]


# ------------------------------------------------------- ask()

async def test_ask_raises_session_ended_when_the_turn_ended_the_session():
    c = ScriptedClient(fire=_BUDGET_END)
    s = Session(c, "sid-1")
    with pytest.raises(SessionEnded) as ei:
        await asyncio.wait_for(s.ask("turn 1"), timeout=10)
    assert ei.value.reason == "budget_exhausted"
    assert ei.value.details == _DETAILS
    assert ei.value.session_id == "sid-1"


async def test_ask_does_not_settle_before_the_terminal_that_follows_the_turn():
    """The ordering assertion, with no clock in it.

    ``delivered`` counts events the script has handed over.  Pre-#1007 the
    call returned on event 3 of 4; the terminal was still unsent.
    """
    c = ScriptedClient(fire=_BUDGET_END)
    with pytest.raises(SessionEnded):
        await asyncio.wait_for(Session(c, "s").ask("turn 1"), timeout=10)
    assert c.delivered == len(_BUDGET_END), (
        "returned before the terminal that says the session is over")


async def test_ask_records_the_terminus_it_raised_on():
    c = ScriptedClient(fire=_BUDGET_END)
    s = Session(c, "s")
    with pytest.raises(SessionEnded):
        await asyncio.wait_for(s.ask("turn 1"), timeout=10)
    assert s.terminus.session_ended is True
    assert s.terminus.reason == "budget_exhausted"
    assert s.terminus.details == _DETAILS


async def test_ask_still_returns_the_turn_when_the_agent_settles():
    """The common path is unchanged: a turn the session survived comes back."""
    c = ScriptedClient(fire=[_status("main", "active"), _out("model", "hi"),
                            _turn("main"), _status("main", "done")])
    s = Session(c, "s")
    assert await asyncio.wait_for(s.ask("q"), timeout=10) == "hi"
    assert s.terminus.session_ended is False
    assert s.terminus.reason == "natural"


async def test_ask_hands_back_a_turn_the_daemon_immediately_follows():
    """``span="turn"``: an ``active`` status confirms the turn is over.

    ``complete`` reads the same event as "another turn is starting, keep
    waiting".  That is the ONE difference between the two spans, and a turn
    verb must not inherit the session verb's answer -- an interactive session
    has to hand back each turn as it lands.
    """
    c = ScriptedClient(fire=[_status("main", "active"), _out("model", "hi"),
                            _turn("main"), _status("main", "active")])
    assert await asyncio.wait_for(Session(c, "s").ask("q"), timeout=10) == "hi"


async def test_ask_ignores_a_subagent_turn():
    c = ScriptedClient(fire=[
        _status("main", "active"),
        _turn("sub-1"), _status("sub-1", "done"),      # not ours
        _out("model", "mine"), _turn("main"), _status("main", "done"),
    ])
    assert await asyncio.wait_for(Session(c, "s").ask("q"),
                                  timeout=10) == "mine"


async def test_ask_settles_on_the_turn_when_no_confirmation_arrives(monkeypatch):
    """The no-hang escape, kept: an unconfirmed proposal settles on the turn.

    This is what the grace is FOR, and its expiry costs information rather
    than correctness -- the call returns the turn's text, exactly as the
    pre-#1007 rule did.
    """
    monkeypatch.setattr(convenience, "TERMINUS_GRACE", 0.05)
    c = ScriptedClient(fire=[_status("main", "active"), _out("model", "hi"),
                            _turn("main")])
    assert await asyncio.wait_for(Session(c, "s").ask("q"), timeout=10) == "hi"


async def test_ask_does_not_spend_the_grace_when_the_confirmation_arrives(
        monkeypatch):
    """The grace must be a backstop, not the common path.

    With it set to 30 s, a call that WAITED it out cannot finish inside the
    harness bound -- so this fails loudly if the confirmation stops being
    consulted and every turn starts costing a grace.
    """
    monkeypatch.setattr(convenience, "TERMINUS_GRACE", 30.0)
    c = ScriptedClient(fire=[_status("main", "active"), _out("model", "hi"),
                            _turn("main"), _status("main", "done")])
    assert await asyncio.wait_for(Session(c, "s").ask("q"), timeout=5) == "hi"


# ------------------------------------- a session that is not there

async def test_ask_fails_fast_when_the_daemon_says_the_session_is_gone():
    """#1007's hang: the daemon answered at once and nobody was listening."""
    c = ScriptedClient(fire=[_error("Session not found: sid-1",
                                    session_id="sid-1")])
    with pytest.raises(SessionEnded) as ei:
        await asyncio.wait_for(Session(c, "sid-1").ask("turn 2"), timeout=10)
    assert ei.value.reason == "not_found"
    assert "Session not found" in str(ei.value)


async def test_an_unattributed_session_error_is_accepted():
    """A daemon predating the stamp sends it with no ``session_id``.

    Degrading to "wait forever" there is the defect, so an unattributed
    ``SessionError`` is taken as ours.
    """
    c = ScriptedClient(fire=[_error("Session not found: sid-1")])
    with pytest.raises(SessionEnded):
        await asyncio.wait_for(Session(c, "sid-1").ask("q"), timeout=10)


async def test_a_session_error_naming_another_session_is_ignored():
    c = ScriptedClient(fire=[
        _error("Session not found: other", session_id="other"),
        _status("main", "active"), _out("model", "hi"),
        _turn("main"), _status("main", "done"),
    ])
    assert await asyncio.wait_for(Session(c, "sid-1").ask("q"),
                                  timeout=10) == "hi"


async def test_an_ordinary_error_does_not_end_the_turn():
    """Only ``error_type="SessionError"`` means the session is not there.

    A mid-turn error belongs to the turn; settling on it would turn every
    recoverable failure into a dead session.
    """
    c = ScriptedClient(fire=[
        _status("main", "active"),
        _error("tool blew up", error_type="ToolError"),
        _out("model", "hi"), _turn("main"), _status("main", "done"),
    ])
    assert await asyncio.wait_for(Session(c, "s").ask("q"),
                                  timeout=10) == "hi"


# ------------------------------------------------------- stream()

async def test_stream_yields_its_chunks_then_raises_session_ended():
    c = ScriptedClient(fire=[
        _status("main", "active"), _out("model", "par"), _out("model", "tial"),
        _turn("main"), _terminal("budget_exhausted", details=_DETAILS),
    ])
    s = Session(c, "sid-1")
    chunks = []
    with pytest.raises(SessionEnded) as ei:
        async for chunk in s.stream("q"):
            chunks.append(chunk)
    assert chunks == ["par", "tial"], "the turn's output must survive"
    assert ei.value.reason == "budget_exhausted"
    assert s.terminus.details == _DETAILS


async def test_stream_fails_fast_when_the_session_is_gone():
    c = ScriptedClient(fire=[_error("Session not found: sid-1",
                                    session_id="sid-1")])

    async def _drain():
        async for _ in Session(c, "sid-1").stream("q"):
            pass

    with pytest.raises(SessionEnded):
        await asyncio.wait_for(_drain(), timeout=10)


# ------------------------------------------------------- complete()

async def test_complete_records_the_terminus_instead_of_raising():
    """A settling session is what ``complete`` was asked to wait for.

    So it does not raise -- but "no payload" was the caller's whole account
    of a budget stop, and the reason was lost.  It is on ``terminus`` now.
    """
    c = ScriptedClient(fire=[
        _status("main", "active"), _turn("main"),
        _terminal("budget_exhausted", details=_DETAILS),
    ])
    s = Session(c, "s")
    assert await asyncio.wait_for(s.complete("go"), timeout=10) is None
    assert s.terminus.reason == "budget_exhausted"
    assert s.terminus.details == _DETAILS
    assert s.terminus.session_ended is True


async def test_complete_fails_fast_when_the_session_is_gone():
    c = ScriptedClient(fire=[_error("Session not found: sid-1",
                                    session_id="sid-1")])
    with pytest.raises(SessionEnded) as ei:
        await asyncio.wait_for(Session(c, "sid-1").complete("go"), timeout=10)
    assert ei.value.reason == "not_found"


# ------------------------------- an error terminal is an error FIRST

@pytest.mark.parametrize("verb", ["ask", "complete"])
async def test_an_error_terminal_still_raises_agent_error(verb):
    """``AgentError`` outranks ``SessionEnded``: the richer type wins.

    An error terminal ends the session too, so both types apply; the one
    that names the failure is the one a caller can act on, and it is what
    every verb has always raised.
    """
    c = FakeClient(fire=[(EventType.SESSION_TERMINATED,
                          SimpleNamespace(reason="error", details=None,
                                          error_type="APIError",
                                          error_summary="boom"))])
    s = Session(c, "s")
    with pytest.raises(AgentError) as ei:
        await asyncio.wait_for(getattr(s, verb)("q"), timeout=10)
    assert ei.value.error_type == "APIError"


async def test_an_error_terminal_arriving_after_a_status_settle_still_raises():
    """The same-batch race, in the direction that loses information.

    A status event settles the call and the terminal follows it in the same
    dispatch batch -- both handlers run before the wait even resumes.  The
    status event carries no reason, so ``_note_terminal``'s ``setdefault``
    holds ``"natural"``; the terminal's own account lands elsewhere.  Reading
    only the first would hand back a bare ``SessionEnded`` and drop the
    ``error_type`` a caller branches on.
    """
    c = FakeClient(fire=[
        _status("main", "active"), _turn("main"), _status("main", "done"),
        (EventType.SESSION_TERMINATED,
         SimpleNamespace(reason="error", details=None,
                         error_type="APIError", error_summary="boom")),
    ])
    with pytest.raises(AgentError) as ei:
        await asyncio.wait_for(Session(c, "s").ask("q"), timeout=10)
    assert ei.value.error_type == "APIError"


async def test_a_timeout_does_not_leave_the_previous_terminus_standing():
    """``terminus`` describes the call that last RETURNED, or nothing.

    It is written on the way out, so a call that raises before settling never
    writes one -- and a driver reading it after a ``TurnTimeout`` would get
    the turn before last, which reads exactly like an answer.
    """
    s = Session(ScriptedClient(fire=[_status("main", "active"),
                                     _out("model", "hi"), _turn("main"),
                                     _status("main", "done")]), "s")
    assert await asyncio.wait_for(s.ask("q"), timeout=10) == "hi"
    assert s.terminus is not None

    s._client = ScriptedClient(fire=[_status("main", "active")])  # never ends
    with pytest.raises(convenience.TurnTimeout):
        await s.ask("q", timeout=0.05)
    assert s.terminus is None


@pytest.mark.parametrize("reason", sorted(convenience.CLEAN_TERMINAL_REASONS))
async def test_a_clean_end_returns_the_turn_and_still_says_the_session_is_over(
        reason):
    """A turn verb raises when a turn was CUT SHORT, not whenever a session ends.

    ``natural`` is the case with a consumer in this tree: a scaffolded
    ``client`` drives a completion-gated profile with ``ask``, its one
    successful turn ends the session, and it must exit 0 -- see
    ``shared/scaffold/tests/test_client_template_completion_wait.py``.
    ``client_request`` / ``stopped`` are the caller's own doing.  The ending
    still reaches the caller, on ``terminus``.
    """
    c = ScriptedClient(fire=[_status("main", "active"), _out("model", "bye"),
                            _turn("main"), _terminal(reason)])
    s = Session(c, "s")
    assert await asyncio.wait_for(s.ask("q"), timeout=10) == "bye"
    assert s.terminus.session_ended is True
    assert s.terminus.reason == reason


async def test_an_unknown_terminal_reason_raises_rather_than_passing_quietly():
    """The allow-list is the point: #1007 IS a new reason going unconsidered.

    ``budget_exhausted`` was the fifth reason, arrived after the code that
    decides what a terminal means, and came back as an ordinary empty reply.
    A deny-list would let the sixth do the same.
    """
    c = ScriptedClient(fire=[_status("main", "active"), _turn("main"),
                            _terminal("some_future_reason")])
    with pytest.raises(SessionEnded) as ei:
        await asyncio.wait_for(Session(c, "s").ask("q"), timeout=10)
    assert ei.value.reason == "some_future_reason"
