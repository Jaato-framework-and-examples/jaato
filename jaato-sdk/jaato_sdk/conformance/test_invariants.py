"""The invariants, asserted against a running daemon.

Each maps to a defect a consumer hit against a live daemon while ~3600 tests
passed here.  They are written as statements of what the SDK PROMISES, so a
failure reads as "the daemon does not do what the docs say" rather than "a
test broke".
"""

from __future__ import annotations

import asyncio

import pytest

from jaato_sdk import IPCClient
from jaato_sdk.events import (
    ClientType, EventType, HistoryEvent, TurnCompletedEvent,
)

pytestmark = pytest.mark.conformance


async def _client(daemon):
    c = IPCClient(socket_path=daemon.socket_path,
                  client_type=ClientType.API,
                  workspace_path=str(daemon.workspace),
                  auto_start=False)
    assert await c.connect(timeout=60), "could not connect to the test daemon"
    return c


async def _drive(daemon, prompt="say ok", profile="conformance",
                 **create_kwargs):
    """Create a session, send one prompt, collect what comes back."""
    c = await _client(daemon)
    seen: list = []
    try:
        sid = await c.create_session(profile=profile, **create_kwargs)

        async def collect():
            async for ev in c.events():
                seen.append(ev)
                if isinstance(ev, TurnCompletedEvent):
                    return

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.2)
        await c.send_message(prompt)
        try:
            await asyncio.wait_for(task, timeout=60)
        except asyncio.TimeoutError:
            task.cancel()
        return c, sid, seen
    except Exception:
        await c.disconnect()
        raise


def _kinds(events):
    return {type(e).__name__ for e in events}


# ------------------------------------------------------------ the baseline

def test_a_session_that_runs_a_turn_reports_it(daemon):
    """The floor. If this fails nothing else here means anything."""
    async def go():
        c, sid, seen = await _drive(daemon)
        try:
            assert sid, "create_session returned no id"
            assert any(isinstance(e, TurnCompletedEvent) for e in seen), (
                f"no TurnCompletedEvent; got {_kinds(seen)}"
            )
        finally:
            await c.disconnect()

    asyncio.run(go())


def test_history_is_answered_not_met_with_silence(daemon):
    """A request must produce data OR an error, never nothing.

    ``_handle_history_request`` returns without emitting when its guard fails,
    so the consumer cannot distinguish "no history" from "not your session"
    and waits out its own timeout.
    """
    async def go():
        # Terminus profile for the same reason as the cid invariant: the
        # history handler's guard fails for a client the cascade policy
        # detached at SessionTerminated, and a prose-ending session never
        # gets there.
        c, sid, seen = await _drive(daemon, profile="conformance-terminus",
                                    cascade_driver_id="conf-cid-hist")
        try:
            got = asyncio.Event()
            result: list = []

            def on_any(ev):
                if isinstance(ev, HistoryEvent) or type(ev).__name__ == "ErrorEvent":
                    result.append(ev)
                    got.set()

            c.subscribe_all(on_any)
            await c.request_history()
            await asyncio.wait_for(got.wait(), timeout=20)
            assert result, "history request answered with silence"
        finally:
            await c.disconnect()

    asyncio.run(go())


# ------------------------------------------------------------- the cascade

def test_a_cascade_id_does_not_silence_the_turn_stream(daemon):
    """Passing ``cascade_driver_id`` must not cost the creating connection its
    own events.

    Measured by a consumer: an arm came back turns=0, tokens=0 with its file
    written and its completion payload delivered -- a silent zero that reads
    as "the model did nothing".
    """
    async def go():
        # THE TERMINUS PROFILE, NOT THE PROSE ONE.  With a prose ending this
        # invariant passes and proves nothing: the first version of this
        # suite went green here while a five-scenario matrix showed the case
        # broken, because the session settled on TURN_COMPLETED and never
        # reached the path under test.  The condition is cid PLUS a
        # signal_completion terminus.
        c, sid, seen = await _drive(daemon, profile="conformance-terminus",
                                    cascade_driver_id="conf-cid-1")
        try:
            assert any(isinstance(e, TurnCompletedEvent) for e in seen), (
                f"a cid'd session reached its declared terminus and never "
                f"reported the turn; got {_kinds(seen)}"
            )
        finally:
            await c.disconnect()

    asyncio.run(go())


# -------------------------------------------------------------- the budget

def test_an_exhausted_budget_refuses_rather_than_handing_back_a_handle(daemon):
    """Never return a handle to a thing that cannot work.

    The refusal exists -- the daemon logs it and emits an ErrorEvent -- but it
    carries no ``request_id``, so the SDK's correlation filter files it as
    incidental and ``SessionRefused`` never fires.  The caller waits 30s and
    gets a runner-not-ready timeout naming the wrong cause.
    """
    from jaato_sdk import SessionCreateFailed

    async def go():
        c = await _client(daemon)
        try:
            # THE POOL MUST ACTUALLY BE SPENT.  A fresh pool of 1 token is
            # not exhausted -- it has 1 token left -- so allowing that
            # session is CORRECT and an earlier version of this test
            # asserted a premise the framework rightly refused.  Each turn
            # charges TURN_USAGE (1200 tokens), so a 1200-token pool is
            # exactly spent by one turn.
            cid = "conf-exhausted"
            await c.cascade_budget_set(cid, limits={"tokens": 1200})
            sid = await c.create_session(profile="conformance",
                                         cascade_driver_id=cid)
            assert sid, "the first arm should be allowed — the pool is full"
            done = asyncio.Event()
            c.subscribe_all(
                lambda ev: done.set()
                if isinstance(ev, TurnCompletedEvent) else None)
            await c.send_message("go")
            await asyncio.wait_for(done.wait(), timeout=60)

            with pytest.raises(SessionCreateFailed):
                await c.create_session(profile="conformance",
                                       cascade_driver_id=cid)
        finally:
            await c.disconnect()

    asyncio.run(go())


def test_a_rejected_budget_declaration_is_reported_not_swallowed(daemon):
    """``limits={"tokens": 0}`` is INVALID, and the caller must be told.

    Reported as "a zero limit refuses nothing".  It is not a defect: the
    framework rejects the declaration outright (``limits.tokens: must be
    > 0``), so no pool is created and there is correctly nothing to refuse.

    What the reporter actually hit is that the rejection was invisible to
    them -- they saw a later create succeed and concluded the ceiling was
    ignored.  So the invariant worth holding is not "zero means zero", it is
    "a refused declaration says so", which is the same visibility class as
    every other defect in this suite.
    """
    async def go():
        c = await _client(daemon)
        try:
            got = asyncio.Event()
            errors: list = []

            def on_any(ev):
                if type(ev).__name__ == "ErrorEvent":
                    errors.append(ev)
                    got.set()

            c.subscribe_all(on_any)
            await c.cascade_budget_set("conf-zero", limits={"tokens": 0})
            await asyncio.wait_for(got.wait(), timeout=20)
            assert errors, (
                "an invalid budget declaration was swallowed; the caller "
                "cannot tell a rejected ceiling from an accepted one"
            )
        finally:
            await c.disconnect()

    asyncio.run(go())


def test_a_reported_cost_reaches_the_turn_event(daemon):
    """echo reports ``cost_usd`` per turn; the event must carry it.

    A consumer measured ``cost_usd=None`` on the event beside a budget tracker
    holding a real figure for the SAME turn -- the token counts agreeing to
    the unit, so it is one measurement whose cost survives on one path only.
    """
    async def go():
        c, sid, seen = await _drive(daemon)
        try:
            turns = [e for e in seen if isinstance(e, TurnCompletedEvent)]
            assert turns, "no turn to inspect"
            usage = getattr(turns[-1], "usage", None)
            assert usage is not None, "TurnCompletedEvent carried no usage"
            cost = getattr(usage, "cost_usd", None)
            assert cost is not None, (
                "the provider reported a cost and the event carries None"
            )
        finally:
            await c.disconnect()

    asyncio.run(go())
