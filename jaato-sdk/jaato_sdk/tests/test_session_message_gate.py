"""``IPCClient.send_session_message`` -- sends the verb, and refuses a daemon that would ignore it (1.23)."""
import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import ClientType, CommandRequest


def _client(protocol):
    c = IPCClient(socket_path="/tmp/_unused_message.sock", client_type=ClientType.API,
                  auto_start=False)
    c._connected = True
    c._server_protocol_version = protocol
    sent = []

    async def fake_send(event):
        sent.append(event)
    c._send_event = fake_send
    return c, sent


@pytest.mark.asyncio
async def test_sends_the_verb_with_a_payload_carrying_the_correlation_id():
    c, sent = _client("1.23")
    await c.send_session_message("s-b", "hello", event_id="e1", request_id="r1")
    assert isinstance(sent[0], CommandRequest)
    assert sent[0].command == "session.message" and sent[0].args == []
    assert sent[0].payload == {"target": "s-b", "text": "hello",
                               "event_id": "e1", "request_id": "r1"}


@pytest.mark.asyncio
async def test_attachments_are_content_and_travel_normalised():
    c, sent = _client("1.23")
    att = {"mime_type": "audio/wav", "data": "AAAA", "display_name": "n.wav"}
    await c.send_session_message("s-b", attachments=[att])
    assert sent[0].payload["text"] == ""
    assert sent[0].payload["attachments"][0]["mime_type"] == "audio/wav"


@pytest.mark.asyncio
async def test_no_content_is_refused_before_sending():
    c, sent = _client("1.23")
    with pytest.raises(ValueError, match="text or attachments"):
        await c.send_session_message("s-b")
    assert sent == []


@pytest.mark.asyncio
async def test_refused_below_the_floor():
    c, sent = _client("1.22")
    with pytest.raises(ValueError, match="session.message"):
        await c.send_session_message("s-b", "hi")
    assert sent == []


@pytest.mark.asyncio
async def test_refused_before_the_handshake():
    c, sent = _client(None)
    with pytest.raises(ValueError):
        await c.send_session_message("s-b", "hi")
    assert sent == []
