"""IPCRecoveryClient.create_session aligns with IPCClient — a plain->recovery swap is drop-in.

The recovery client mirrored every IPCClient.create_session kwarg EXCEPT `timeout`,
so passing timeout= (which IPCClient accepts) raised "unexpected keyword argument".
Found by the jaato-tui-driven-tests harness migration on a LIVE walk (static
import/compile missed it). Now timeout is accepted + forwarded; a signature-parity
test guards against future divergence.
"""
import asyncio
import inspect
from unittest.mock import AsyncMock

from jaato_sdk.client.recovery import IPCRecoveryClient
from jaato_sdk.client.ipc import IPCClient


def test_create_session_forwards_timeout():
    rc = IPCRecoveryClient.__new__(IPCRecoveryClient)
    rc._check_can_send = lambda: None
    rc._client = AsyncMock()
    rc._client.create_session = AsyncMock(return_value="sid")
    rc._session_id = None
    out = asyncio.run(rc.create_session(name="n", timeout=30.0))
    assert out == "sid"
    _, kwargs = rc._client.create_session.call_args
    assert kwargs.get("timeout") == 30.0


def test_recovery_create_session_accepts_every_ipcclient_kwarg():
    ipc = set(inspect.signature(IPCClient.create_session).parameters)
    rec = set(inspect.signature(IPCRecoveryClient.create_session).parameters)
    assert ipc <= rec, f"recovery missing kwargs: {ipc - rec}"


def test_recovery_register_client_tools_remembers_and_delegates():
    """(4) register_client_tools is proxied AND remembered (so it can be
    re-registered on the fresh inner client after a reconnect)."""
    rc = IPCRecoveryClient.__new__(IPCRecoveryClient)
    rc._check_can_send = lambda: None
    rc._client = AsyncMock()
    rc._registered_client_tools = []
    tools = [{"name": "t", "description": "d", "parameters": {}}]
    asyncio.run(rc.register_client_tools(tools))
    assert rc._registered_client_tools == tools          # remembered for reconnect
    rc._client.register_client_tools.assert_awaited_once_with(tools)


def test_recovery_list_sessions_delegates():
    rc = IPCRecoveryClient.__new__(IPCRecoveryClient)
    rc._check_can_send = lambda: None
    rc._client = AsyncMock()
    asyncio.run(rc.list_sessions())
    rc._client.list_sessions.assert_awaited_once_with()


def test_recovery_batch_clarification_delegates():
    """(2) respond_to_clarification_batch proxied to the inner client."""
    rc = IPCRecoveryClient.__new__(IPCRecoveryClient)
    rc._check_can_send = lambda: None
    rc._client = AsyncMock()
    asyncio.run(rc.respond_to_clarification_batch("rid", ["a", "b"]))
    rc._client.respond_to_clarification_batch.assert_awaited_once_with("rid", ["a", "b"])
