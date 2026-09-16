"""``IPCClient.toggle_workspace_ignore`` -- sends the verb, and refuses a daemon that would ignore it."""
import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import ClientType, CommandRequest


def _client(protocol):
    c = IPCClient(socket_path="/tmp/_unused_ignore.sock", client_type=ClientType.API,
                  auto_start=False)
    c._connected = True
    c._server_protocol_version = protocol
    sent = []

    async def fake_send(event):
        sent.append(event)
    c._send_event = fake_send
    return c, sent


@pytest.mark.asyncio
async def test_sends_the_verb_with_the_entry_as_its_one_arg():
    c, sent = _client("1.12")
    await c.toggle_workspace_ignore(".jaato/logs/")
    assert isinstance(sent[0], CommandRequest)
    assert sent[0].command == "workspace.ignore" and sent[0].args == [".jaato/logs/"]


@pytest.mark.asyncio
async def test_refused_below_the_floor():
    c, sent = _client("1.11")
    with pytest.raises(ValueError, match="workspace.ignore"):
        await c.toggle_workspace_ignore("x")
    assert sent == []


@pytest.mark.asyncio
async def test_refused_before_the_handshake():
    c, sent = _client(None)
    with pytest.raises(ValueError):
        await c.toggle_workspace_ignore("x")
