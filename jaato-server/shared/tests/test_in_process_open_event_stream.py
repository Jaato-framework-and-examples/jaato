"""InProcessClient event-iterator parity: subscribe_all + open_event_stream.

The facade rides on any transport (IPC / WS / recovery / in-process); the new
public ``open_event_stream()`` primitive must exist across ALL of them so the
surface is uniform.  IPC/WS get it from ``IPCClient`` / ``IPCRecoveryClient``;
these tests pin the in-process implementation (emitter all-events fan-out +
eager subscribe + None-sentinel on disconnect).
"""

import asyncio
import pytest

from jaato.in_process import InProcessClient, InProcessEventEmitter
from jaato_sdk.events import AgentOutputEvent, SystemMessageEvent, TurnCompletedEvent


def _make_client() -> InProcessClient:
    c = object.__new__(InProcessClient)
    c._emitter = InProcessEventEmitter()
    c._event_queues = []
    c._event_queue_unsubs = {}
    return c


# ---- emitter subscribe_all ----------------------------------------------------

def test_subscribe_all_fires_on_every_type():
    em = InProcessEventEmitter()
    seen = []
    unsub = em.subscribe_all(seen.append)
    em.emit(SystemMessageEvent(message="a", style="info"))
    em.emit(TurnCompletedEvent(agent_id="main"))
    assert [type(e).__name__ for e in seen] == ["SystemMessageEvent", "TurnCompletedEvent"]
    unsub()
    em.emit(SystemMessageEvent(message="b", style="info"))
    assert len(seen) == 2                       # unsubscribed → no more


def test_subscribe_all_coexists_with_typed_subscribe():
    em = InProcessEventEmitter()
    allev, typed = [], []
    em.subscribe_all(allev.append)
    em.subscribe(TurnCompletedEvent(agent_id="").type, typed.append)
    em.emit(TurnCompletedEvent(agent_id="main"))
    em.emit(SystemMessageEvent(message="x", style="info"))
    assert len(allev) == 2 and len(typed) == 1  # typed only sees its type


# ---- open_event_stream (parity with IPCClient) --------------------------------

@pytest.mark.asyncio
async def test_open_event_stream_subscribes_eagerly():
    c = _make_client()
    assert len(c._event_queues) == 0
    stream = c.open_event_stream()
    assert len(c._event_queues) == 1            # registered at call time
    await stream.aclose()
    assert len(c._event_queues) == 0


@pytest.mark.asyncio
async def test_open_event_stream_receives_all_events():
    c = _make_client()
    stream = c.open_event_stream()
    c._emitter.emit(AgentOutputEvent(source="model", text="hi"))
    c._emitter.emit(TurnCompletedEvent(agent_id="main"))
    e1 = await asyncio.wait_for(stream.__anext__(), 1.0)
    e2 = await asyncio.wait_for(stream.__anext__(), 1.0)
    assert e1.text == "hi" and type(e2).__name__ == "TurnCompletedEvent"
    await stream.aclose()


@pytest.mark.asyncio
async def test_disconnect_pushes_none_sentinel():
    c = _make_client()
    c._embedded = None                          # disconnect() guards on this
    stream = c.open_event_stream()
    await c.disconnect()                        # signals the sentinel
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), 1.0)
    assert len(c._event_queues) == 0           # auto-unsubscribed


@pytest.mark.asyncio
async def test_events_iterator_parity():
    c = _make_client()
    out = []

    async def consume():
        async for ev in c.events():
            out.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)                       # let events() subscribe
    c._emitter.emit(SystemMessageEvent(message="one", style="info"))
    await asyncio.sleep(0)
    c._embedded = None
    await c.disconnect()                         # None sentinel ends the loop
    await asyncio.wait_for(task, 1.0)
    assert [e.message for e in out] == ["one"]
