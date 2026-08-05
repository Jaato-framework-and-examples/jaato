"""``open_event_stream()`` — synchronous-subscribe event iterator (SDK).

Unlike ``events()`` (an async generator that subscribes lazily on first
``__anext__``), ``open_event_stream()`` registers the subscriber queue EAGERLY,
at call time, so a caller can guarantee the subscription is live before an action
that triggers server-side output (e.g. ``attach`` a session the daemon drives a
woken turn on).  Closes the gap that otherwise forced clients to poke the private
``_subscribe_events`` / ``_event_subscribers`` internals.
"""

import asyncio
import pytest

from jaato_sdk import IPCClient
from jaato_sdk.events import ClientType, SystemMessageEvent


def _make_client():
    client = IPCClient(
        socket_path="/tmp/_unused_open_stream_test.sock",
        client_type=ClientType.API,
        auto_start=False,
    )
    client._connected = True
    return client


def _broadcast(client, event) -> None:
    """Same fan-out the drain loop performs (queue if subscribers, else buffer)."""
    if client._event_subscribers:
        for q in list(client._event_subscribers):
            q.put_nowait(event)
    else:
        client._buffered_events.append(event)


@pytest.mark.asyncio
async def test_open_event_stream_subscribes_eagerly_at_call_time():
    # THE contract: the queue is registered the moment open_event_stream()
    # returns — NO __anext__ needed.  (events() would still be at 0 here.)
    client = _make_client()
    assert len(client._event_subscribers) == 0
    stream = client.open_event_stream()
    assert len(client._event_subscribers) == 1   # registered synchronously
    await stream.aclose()
    assert len(client._event_subscribers) == 0


@pytest.mark.asyncio
async def test_events_is_lazy_by_contrast():
    # Documents the gap open_event_stream() fills: events() registers NOTHING
    # until first driven.
    client = _make_client()
    gen = client.events()
    assert len(client._event_subscribers) == 0   # deferred to first __anext__
    await gen.aclose()


@pytest.mark.asyncio
async def test_subscribe_before_emit_is_not_missed():
    # The ordering guarantee: subscribe (eager) THEN emit → delivered.
    client = _make_client()
    stream = client.open_event_stream()
    _broadcast(client, SystemMessageEvent(message="after-subscribe", style="info"))
    ev = await asyncio.wait_for(stream.__anext__(), 1.0)
    assert ev.message == "after-subscribe"
    await stream.aclose()


@pytest.mark.asyncio
async def test_buffered_events_replayed_into_eager_stream():
    # IPCClient drains _buffered_events into a new subscriber queue, so an
    # eager stream sees events that flowed before it subscribed.
    client = _make_client()
    _broadcast(client, SystemMessageEvent(message="pre", style="info"))
    assert client._buffered_events                # went to buffer (no subscriber)
    stream = client.open_event_stream()
    ev = await asyncio.wait_for(stream.__anext__(), 1.0)
    assert ev.message == "pre"
    assert client._buffered_events == []          # drained on subscribe
    await stream.aclose()


@pytest.mark.asyncio
async def test_none_sentinel_ends_and_unsubscribes():
    client = _make_client()
    stream = client.open_event_stream()
    _broadcast(client, None)                      # drain-loop disconnect sentinel
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), 1.0)
    assert len(client._event_subscribers) == 0    # auto-unsubscribed on sentinel


@pytest.mark.asyncio
async def test_async_with_unsubscribes_on_exit():
    client = _make_client()
    async with client.open_event_stream() as stream:
        _broadcast(client, SystemMessageEvent(message="x", style="info"))
        ev = await asyncio.wait_for(stream.__anext__(), 1.0)
        assert ev.message == "x"
        assert len(client._event_subscribers) == 1
    assert len(client._event_subscribers) == 0    # unsubscribed on block exit


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    client = _make_client()
    stream = client.open_event_stream()
    await stream.aclose()
    await stream.aclose()                         # no error, still unsubscribed
    assert len(client._event_subscribers) == 0
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()                  # closed → StopAsyncIteration


@pytest.mark.asyncio
async def test_multiple_eager_streams_each_get_every_event():
    client = _make_client()
    s1 = client.open_event_stream()
    s2 = client.open_event_stream()
    assert len(client._event_subscribers) == 2
    _broadcast(client, SystemMessageEvent(message="both", style="info"))
    e1 = await asyncio.wait_for(s1.__anext__(), 1.0)
    e2 = await asyncio.wait_for(s2.__anext__(), 1.0)
    assert e1.message == e2.message == "both"
    await s1.aclose()
    await s2.aclose()
