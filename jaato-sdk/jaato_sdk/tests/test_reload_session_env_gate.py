"""``IPCClient.reload_session_env`` -- sends the verb, and refuses a daemon that would ignore it."""
import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import ClientType, CommandRequest


def _client(protocol):
    c = IPCClient(socket_path="/tmp/_unused_reload.sock", client_type=ClientType.API,
                  auto_start=False)
    c._connected = True
    c._server_protocol_version = protocol
    sent = []

    async def fake_send(event):
        sent.append(event)
    c._send_event = fake_send
    return c, sent


@pytest.mark.asyncio
async def test_sends_the_verb_for_the_attached_session():
    c, sent = _client("1.11")
    await c.reload_session_env()
    assert isinstance(sent[0], CommandRequest)
    assert sent[0].command == "session.reload_env" and sent[0].args == []


@pytest.mark.asyncio
async def test_an_explicit_id_is_the_first_arg():
    c, sent = _client("1.12")
    await c.reload_session_env("20260916_102126")
    assert sent[0].args == ["20260916_102126"]


@pytest.mark.asyncio
async def test_refused_below_the_floor():
    c, sent = _client("1.10")
    with pytest.raises(ValueError, match="session.reload_env"):
        await c.reload_session_env()
    assert sent == []


@pytest.mark.asyncio
async def test_refused_before_the_handshake():
    c, sent = _client(None)
    with pytest.raises(ValueError):
        await c.reload_session_env()
