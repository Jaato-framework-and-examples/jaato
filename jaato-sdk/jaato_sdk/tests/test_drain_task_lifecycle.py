"""Tests for the drain-task / subscriber-queue model in IPCClient (SDK 0.13.0+).

Pre-0.13.0 the IPCClient used a single-reader pattern gated by
``_events_active``; consumers had to dance around the flag and a
deferred-aclose race made back-to-back ``events()`` →
``create_session`` patterns return ``None`` even when the daemon
created the session correctly.  0.13.0 replaces that with a background
drain task that fans every received event out to every active
subscriber queue.

These tests exercise the new model directly — bypassing the socket by
injecting events into the drain loop's fan-out path.
"""

import asyncio
import pytest

from jaato_sdk import IPCClient
from jaato_sdk.events import (
    AgentCompletedEvent,
    ClientType,
    ErrorEvent,
    SessionInfoEvent,
    SystemMessageEvent,
)


def _make_client():
    """A client that doesn't actually connect — we drive the subscriber
    queues directly.  ``_connected = True`` so the subscriber loops
    don't immediately exit."""
    client = IPCClient(
        socket_path="/tmp/_unused_drain_test.sock",
        client_type=ClientType.API,
        auto_start=False,
    )
    client._connected = True
    return client


def _broadcast(client: IPCClient, event) -> None:
    """Inject an event into every active subscriber queue and the
    buffer when no subscriber exists — same fan-out the drain loop
    performs."""
    if client._event_subscribers:
        for q in list(client._event_subscribers):
            q.put_nowait(event)
    else:
        client._buffered_events.append(event)


# =====================================================================
# Buffer + replay
# =====================================================================


@pytest.mark.asyncio
async def test_buffered_events_replayed_on_subscribe():
    client = _make_client()
    e1 = SystemMessageEvent(message="first", style="info")
    e2 = SystemMessageEvent(message="second", style="info")
    _broadcast(client, e1)
    _broadcast(client, e2)
    # Both went into the buffer because no subscriber existed.
    assert len(client._buffered_events) == 2

    # New events() iterator drains the buffer first.
    out = []
    gen = client.events()
    out.append(await asyncio.wait_for(gen.__anext__(), 1.0))
    out.append(await asyncio.wait_for(gen.__anext__(), 1.0))
    await gen.aclose()

    assert [e.message for e in out] == ["first", "second"]
    # Buffer drained on subscribe.
    assert client._buffered_events == []


@pytest.mark.asyncio
async def test_subscribe_then_broadcast_yields_via_queue():
    client = _make_client()
    gen = client.events()

    async def consume_one():
        return await asyncio.wait_for(gen.__anext__(), 1.0)

    consume_task = asyncio.create_task(consume_one())
    # Yield once so the iterator subscribes its queue.
    await asyncio.sleep(0)

    e = SystemMessageEvent(message="live", style="info")
    _broadcast(client, e)

    received = await consume_task
    assert received.message == "live"
    await gen.aclose()


# =====================================================================
# Concurrent subscribers (the fix's core property)
# =====================================================================


@pytest.mark.asyncio
async def test_two_concurrent_iterators_both_receive_all_events():
    client = _make_client()
    gen_a = client.events()
    gen_b = client.events()

    # Start consumption so both subscribers are on the list.
    async def take_three(gen):
        out = []
        for _ in range(3):
            out.append(await asyncio.wait_for(gen.__anext__(), 1.0))
        return out

    task_a = asyncio.create_task(take_three(gen_a))
    task_b = asyncio.create_task(take_three(gen_b))
    await asyncio.sleep(0)

    for i in range(3):
        _broadcast(client, SystemMessageEvent(message=f"e{i}", style="info"))

    received_a = await task_a
    received_b = await task_b
    assert [e.message for e in received_a] == ["e0", "e1", "e2"]
    assert [e.message for e in received_b] == ["e0", "e1", "e2"]
    await gen_a.aclose()
    await gen_b.aclose()


@pytest.mark.asyncio
async def test_await_session_info_filters_for_session_info_event():
    client = _make_client()
    task = asyncio.create_task(client._await_session_info())
    await asyncio.sleep(0)

    # Inject a non-target event first; await_session_info should ignore
    # it (filter-and-continue) and pick up the SessionInfoEvent next.
    _broadcast(client, SystemMessageEvent(message="warming up", style="info"))
    _broadcast(client, SessionInfoEvent(session_id="s_42"))

    sid = await asyncio.wait_for(task, 1.0)
    assert sid == "s_42"


# =====================================================================
# The pre-0.13.0 race: back-to-back events() iterator → create_session
# =====================================================================


@pytest.mark.asyncio
async def test_back_to_back_events_then_create_session_no_race():
    """Reproduces the deferred-aclose race that pre-0.13.0 made
    create_session return None when called immediately after exiting
    an events() iterator.  In 0.13.0+ each subscriber owns its queue;
    there's no shared ``_events_active`` gate to race against."""
    client = _make_client()

    # Phase 1: events() iterator that exits early (mimicking
    # walker._wait_for_completion seeing AgentCompletedEvent).
    gen = client.events()

    async def consume_until_completion():
        async for ev in gen:
            if isinstance(ev, AgentCompletedEvent):
                return True
        return False

    consumer_task = asyncio.create_task(consume_until_completion())
    await asyncio.sleep(0)
    _broadcast(client, AgentCompletedEvent(agent_id="manual_writer"))
    assert await asyncio.wait_for(consumer_task, 1.0) is True

    # Phase 2: immediately call create_session.  In pre-0.13.0 SDK,
    # ``_events_active`` was still True (deferred aclose) and
    # create_session would return None.  In 0.13.0+ the subscriber
    # model removes that gate — a fresh _await_session_info subscribes
    # its own queue and resolves on the injected SessionInfoEvent.
    captured = []
    async def fake_send_event(ev):
        captured.append(ev)
    client._send_event = fake_send_event

    async def driver():
        await asyncio.sleep(0)  # let create_session subscribe its queue
        _broadcast(client, SessionInfoEvent(session_id="s_99"))

    drive_task = asyncio.create_task(driver())
    sid = await asyncio.wait_for(
        client.create_session(name="next-session"), 1.0,
    )
    await drive_task

    assert sid == "s_99", (
        "create_session should resolve to the injected SessionInfoEvent "
        "even when called immediately after an events() iterator exited"
    )
    assert len(captured) == 1, (
        "create_session must still send the session.new CommandRequest"
    )
