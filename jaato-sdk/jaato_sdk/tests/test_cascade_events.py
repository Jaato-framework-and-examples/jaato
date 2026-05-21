"""SDK-side tests for IPCClient.cascade_events (Phase 2 cascade-as-client).

Pins the async-iterator contract: register on first iteration,
yield events from subscriber queue, unregister on close
(natural / break / exception).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from jaato_sdk import IPCClient
from jaato_sdk.events import ClientType, CommandRequest, SessionTerminatedEvent


def _make_client_capture():
    """Return (client, captured_events, event_queue) tuple.

    Patches _send_event to capture outgoing CommandRequests + lets
    tests push events into the subscriber queue to simulate
    server-side dispatch.
    """
    client = IPCClient(
        socket_path="/tmp/_unused.sock",
        client_type=ClientType.API,
        auto_start=False,
    )
    captured: list = []

    async def fake_send_event(event):
        captured.append(event)

    # The test will create the subscriber queue then push events.
    client._send_event = fake_send_event
    return client, captured


# ======================================================================
# Subscribe / yield / unregister
# ======================================================================


@pytest.mark.asyncio
async def test_cascade_events_sends_register_then_yields():
    """First action of the iterator: cascade.register CommandRequest.
    Then yields events from the subscriber queue.  Cleanup verified
    via explicit aclose() (per the docstring's cleanup contract:
    async-for-break does not synchronously trigger finally; the
    server-side disconnect cleanup is the production backstop)."""
    client, captured = _make_client_capture()

    yielded: list = []
    # Hold the generator explicitly so we can aclose() it.
    gen = client.cascade_events(
        "cascade-abc",
        event_types=["SessionTerminatedEvent"],
        role="owner",
    )

    async def driver():
        async for event in gen:
            yielded.append(event)
            break  # exit after first event

    task = asyncio.create_task(driver())

    # Wait for register to arrive, then push an event.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if captured:
            break
    assert len(captured) >= 1
    first = captured[0]
    assert isinstance(first, CommandRequest)
    assert first.command == "cascade.register"
    assert first.args == ["cascade-abc", "owner", "SessionTerminatedEvent"]

    # Push an event to the subscriber queue.
    queue = next(iter(client._event_subscribers))
    fake_event = SessionTerminatedEvent(
        session_id="s1", agent_id="main", reason="error",
    )
    await queue.put(fake_event)
    await task

    # Event was yielded.
    assert yielded == [fake_event]
    # Explicit aclose() triggers the finally block deterministically.
    await gen.aclose()
    cmds = [e.command for e in captured if isinstance(e, CommandRequest)]
    assert "cascade.unregister" in cmds


@pytest.mark.asyncio
async def test_cascade_events_no_event_types_defaults_to_all():
    """event_types=None → args carries [cid, role] only (no filter)."""
    client, captured = _make_client_capture()

    async def driver():
        async for _ in client.cascade_events("cascade-X"):
            break

    task = asyncio.create_task(driver())
    for _ in range(50):
        await asyncio.sleep(0.01)
        if captured:
            break

    first = captured[0]
    assert first.command == "cascade.register"
    assert first.args == ["cascade-X", "observer"]  # role default

    # Unblock the iterator so test completes.
    queue = next(iter(client._event_subscribers))
    await queue.put(None)  # drain-loop-ended sentinel
    await task


@pytest.mark.asyncio
async def test_cascade_events_unregister_on_exception():
    """If the consumer raises mid-iteration AFTER an aclose() / GC,
    the finally block runs and sends unregister.  Async-generator
    cleanup contract: finally runs at aclose / GC, not at break /
    raise — the server-side disconnect cleanup is the prod backstop.
    """
    client, captured = _make_client_capture()
    gen = client.cascade_events("cascade-Y")

    async def driver():
        async for _ in gen:
            raise RuntimeError("consumer panic")

    task = asyncio.create_task(driver())

    for _ in range(50):
        await asyncio.sleep(0.01)
        if captured:
            break

    # Push one event so the iterator advances + the consumer raises.
    queue = next(iter(client._event_subscribers))
    await queue.put(SessionTerminatedEvent(
        session_id="s", agent_id="main", reason="error",
    ))
    with pytest.raises(RuntimeError, match="consumer panic"):
        await task

    # Explicit aclose() runs the finally block deterministically.
    await gen.aclose()
    cmds = [e.command for e in captured if isinstance(e, CommandRequest)]
    assert "cascade.unregister" in cmds


@pytest.mark.asyncio
async def test_cascade_events_unregister_on_drain_loop_end():
    """If the connection's drain loop ends (None sentinel), the
    iterator exits cleanly + sends unregister."""
    client, captured = _make_client_capture()

    yielded: list = []

    async def driver():
        async for event in client.cascade_events("cascade-Z"):
            yielded.append(event)

    task = asyncio.create_task(driver())

    for _ in range(50):
        await asyncio.sleep(0.01)
        if captured:
            break

    # Push None sentinel — simulates drain loop ending.
    queue = next(iter(client._event_subscribers))
    await queue.put(None)
    await task

    assert yielded == []  # no events yielded
    cmds = [e.command for e in captured if isinstance(e, CommandRequest)]
    assert "cascade.unregister" in cmds


@pytest.mark.asyncio
async def test_cascade_events_multiple_event_types():
    """Multiple event_types are passed to the server as positional args."""
    client, captured = _make_client_capture()

    async def driver():
        async for _ in client.cascade_events(
            "cascade-M",
            event_types=["SessionTerminatedEvent", "AgentCompletedEvent"],
            role="owner",
        ):
            break

    task = asyncio.create_task(driver())

    for _ in range(50):
        await asyncio.sleep(0.01)
        if captured:
            break

    first = captured[0]
    assert first.args == [
        "cascade-M", "owner",
        "SessionTerminatedEvent", "AgentCompletedEvent",
    ]

    queue = next(iter(client._event_subscribers))
    await queue.put(None)
    await task
