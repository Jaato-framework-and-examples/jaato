"""Regression tests for the IPCRecoveryClient background event pump
(facade-over-recovery hang fix) + the workspace_path str coerce."""

import asyncio

from jaato_sdk.client.recovery import IPCRecoveryClient, ConnectionState
from jaato_sdk.client.config import _find_config_files
from jaato_sdk.events import AgentOutputEvent, EventType, ClientType


class _FakeInner:
    """Minimal stand-in for the inner IPCClient: yields events then idles."""

    def __init__(self, events):
        self._events = events
        self.client_id = "test-client"

    async def events(self):
        for e in self._events:
            yield e
        await asyncio.sleep(3600)  # live stream that doesn't end

    async def disconnect(self):
        pass


def _evt():
    return AgentOutputEvent(agent_id="main", source="model", text="hello", mode="final")


def _wire(c):
    c._client = _FakeInner([_evt()])
    c._state = ConnectionState.CONNECTED
    c._events_running = True


async def _stop(c, pump):
    c._events_running = False
    pump.cancel()
    try:
        await pump
    except BaseException:
        pass


def test_pump_dispatches_to_subscribers_without_events_iteration():
    """The bug: recovery subscribe() handlers only fired while events() was
    iterated. Fix: the background pump dispatches regardless."""
    async def run():
        c = IPCRecoveryClient(socket_path="/tmp/x.sock", client_type=ClientType.API)
        _wire(c)
        got = []
        c.subscribe(EventType.AGENT_OUTPUT, lambda e: got.append(e))
        pump = asyncio.create_task(c._event_pump())
        for _ in range(100):       # NO one iterates events()
            if got:
                break
            await asyncio.sleep(0.01)
        await _stop(c, pump)
        assert got, "subscribe handler did not fire from the background pump"
        assert got[0].text == "hello"
    asyncio.run(run())


def test_pump_fans_out_to_events_iterator():
    """events() still yields (queue subscriber to the pump's fan-out)."""
    async def run():
        c = IPCRecoveryClient(socket_path="/tmp/x.sock", client_type=ClientType.API)
        _wire(c)
        pump = asyncio.create_task(c._event_pump())
        received = []

        async def consume():
            async for e in c.events():
                received.append(e)
                break

        try:
            await asyncio.wait_for(consume(), timeout=2.0)
        finally:
            await _stop(c, pump)
        assert received and received[0].text == "hello"
    asyncio.run(run())


def test_find_config_files_accepts_str_workspace_path():
    """Bug B: a str workspace_path must not TypeError on the Path op."""
    result = _find_config_files(workspace_path="/tmp/some-workspace")
    assert isinstance(result, list)
