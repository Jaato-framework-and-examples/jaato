"""Functional tests for the typed event subscription API.

Exercises ``IPCClient.subscribe / subscribe_once / subscribe_all /
subscribe_many`` semantics without any actual IPC connection — we
inject events via the internal ``_dispatch`` method that the real
``events()`` loop uses.

Each test focuses on one observable contract; see
``docs/sdk-event-catalog.md`` for the full surface.
"""

import asyncio

import pytest

from jaato_sdk import IPCClient, EventType
from jaato_sdk.events import (
    AgentCompletedEvent,
    AgentOutputEvent,
    PermissionRequestedEvent,
    SystemMessageEvent,
)


def _output(text: str = "x") -> AgentOutputEvent:
    return AgentOutputEvent(source="model", text=text, mode="append")


def _completed() -> AgentCompletedEvent:
    return AgentCompletedEvent()


def _perm() -> PermissionRequestedEvent:
    return PermissionRequestedEvent(
        request_id="r1",
        tool_name="cli",
        arguments={},
    )


def _client() -> IPCClient:
    """Fresh client without connecting (we drive ``_dispatch`` directly)."""
    return IPCClient(socket_path="/tmp/_unused_subscribe_test.sock", auto_start=False)


@pytest.mark.asyncio
async def test_subscribe_filters_by_type():
    """A typed subscription only fires for matching event types."""
    c = _client()
    seen = []
    c.subscribe(EventType.AGENT_OUTPUT, lambda e: seen.append(e.text))

    c._dispatch(_output("a"))
    c._dispatch(_completed())     # different type
    c._dispatch(_output("b"))

    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_subscribe_all_receives_everything():
    """``subscribe_all`` is a catchall firehose."""
    c = _client()
    seen = []
    c.subscribe_all(lambda e: seen.append(e.type.value))

    c._dispatch(_output())
    c._dispatch(_completed())
    c._dispatch(_perm())

    assert seen == [
        "agent.output",
        "agent.completed",
        "permission.requested",
    ]


@pytest.mark.asyncio
async def test_subscribe_once_fires_exactly_once():
    """``subscribe_once`` removes the handler after the first delivery."""
    c = _client()
    seen = []
    c.subscribe_once(EventType.AGENT_OUTPUT, lambda e: seen.append(e.text))

    c._dispatch(_output("first"))
    c._dispatch(_output("second"))
    c._dispatch(_output("third"))

    assert seen == ["first"]


@pytest.mark.asyncio
async def test_subscribe_many_atomic_unsub():
    """``subscribe_many`` returns a single unsub that removes all handlers."""
    c = _client()
    seen = []

    unsub_all = c.subscribe_many({
        EventType.AGENT_OUTPUT: lambda e: seen.append(("out", e.text)),
        EventType.AGENT_COMPLETED: lambda e: seen.append(("done",)),
        EventType.PERMISSION_REQUESTED: lambda e: seen.append(("perm", e.tool_name)),
    })

    c._dispatch(_output("hi"))
    c._dispatch(_completed())
    c._dispatch(_perm())

    unsub_all()

    c._dispatch(_output("after"))
    c._dispatch(_completed())
    c._dispatch(_perm())

    assert seen == [("out", "hi"), ("done",), ("perm", "cli")]


@pytest.mark.asyncio
async def test_async_handler_does_not_block_stream():
    """Async handlers are scheduled fire-and-forget; later events still dispatch."""
    c = _client()
    order = []

    async def slow_handler(event):
        await asyncio.sleep(0.05)
        order.append(("async-done", event.text))

    c.subscribe(EventType.AGENT_OUTPUT, slow_handler)
    c.subscribe(EventType.AGENT_OUTPUT, lambda e: order.append(("sync", e.text)))

    c._dispatch(_output("first"))
    c._dispatch(_output("second"))

    # Sync handlers fire immediately; async ones complete after sleep.
    await asyncio.sleep(0.1)

    syncs = [o for o in order if o[0] == "sync"]
    asyncs = [o for o in order if o[0] == "async-done"]
    assert syncs == [("sync", "first"), ("sync", "second")]
    assert sorted(asyncs) == [("async-done", "first"), ("async-done", "second")]


@pytest.mark.asyncio
async def test_async_handlers_run_concurrently():
    """Multiple async handlers for the same event do not serialize."""
    c = _client()
    timings = []

    async def slow(event):
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0.05)
        end = asyncio.get_event_loop().time()
        timings.append((start, end))

    c.subscribe(EventType.AGENT_OUTPUT, slow)
    c.subscribe(EventType.AGENT_OUTPUT, slow)
    c.subscribe(EventType.AGENT_OUTPUT, slow)

    c._dispatch(_output())
    await asyncio.sleep(0.12)

    assert len(timings) == 3
    # All three should have started before any finished — overlap proves concurrency.
    starts = sorted(t[0] for t in timings)
    ends = sorted(t[1] for t in timings)
    assert starts[-1] < ends[0], "Handlers serialized instead of running concurrently"


@pytest.mark.asyncio
async def test_sync_exception_isolated():
    """A sync handler that raises does not break the stream or other handlers."""
    c = _client()
    seen = []

    def boom(event):
        raise RuntimeError("intentional")

    c.subscribe(EventType.AGENT_OUTPUT, boom)
    c.subscribe(EventType.AGENT_OUTPUT, lambda e: seen.append(("good", e.text)))

    c._dispatch(_output("a"))
    c._dispatch(_output("b"))

    assert seen == [("good", "a"), ("good", "b")]


@pytest.mark.asyncio
async def test_async_rejection_isolated():
    """An async handler that rejects is logged but does not affect others."""
    c = _client()
    seen = []

    async def boom(event):
        await asyncio.sleep(0.01)
        raise RuntimeError("intentional async failure")

    c.subscribe(EventType.AGENT_OUTPUT, boom)
    c.subscribe(EventType.AGENT_OUTPUT, lambda e: seen.append(e.text))

    c._dispatch(_output("a"))
    c._dispatch(_output("b"))

    await asyncio.sleep(0.05)

    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_unsub_during_dispatch_takes_effect_next_event():
    """Unsubscribing during dispatch only affects the *next* event."""
    c = _client()
    seen = []

    state = {"unsub": None}

    def handler_a(event):
        seen.append(("a", event.text))
        # Unsubscribe handler_b mid-dispatch
        if state["unsub"] is not None:
            state["unsub"]()
            state["unsub"] = None

    def handler_b(event):
        seen.append(("b", event.text))

    c.subscribe(EventType.AGENT_OUTPUT, handler_a)
    state["unsub"] = c.subscribe(EventType.AGENT_OUTPUT, handler_b)

    c._dispatch(_output("first"))
    # handler_b should still fire on this event because the snapshot
    # was taken before handler_a removed it.
    assert ("b", "first") in seen

    c._dispatch(_output("second"))
    # On the next event, handler_b is gone.
    assert ("b", "second") not in seen
    assert ("a", "second") in seen


@pytest.mark.asyncio
async def test_handler_registered_before_connect_sees_first_dispatch():
    """Subscribing before any event arrives still captures the first one.

    (Real-world meaning: subscribing before ``connect()`` doesn't drop
    the inaugural ``ConnectedEvent``.)
    """
    c = _client()
    seen = []
    c.subscribe(EventType.AGENT_OUTPUT, lambda e: seen.append(e.text))

    # No events have been dispatched yet.  First dispatch should hit.
    c._dispatch(_output("inaugural"))

    assert seen == ["inaugural"]


@pytest.mark.asyncio
async def test_no_replay_on_late_subscribe():
    """A handler registered after events have flowed sees no past events."""
    c = _client()

    c._dispatch(_output("missed"))

    seen = []
    c.subscribe(EventType.AGENT_OUTPUT, lambda e: seen.append(e.text))

    c._dispatch(_output("seen"))

    assert seen == ["seen"]


@pytest.mark.asyncio
async def test_unsubscribe_is_idempotent():
    """Calling the returned unsubscribe twice is a safe no-op."""
    c = _client()
    seen = []
    unsub = c.subscribe(EventType.AGENT_OUTPUT, lambda e: seen.append(e.text))

    c._dispatch(_output("a"))
    unsub()
    unsub()  # second call must not raise
    c._dispatch(_output("b"))

    assert seen == ["a"]
