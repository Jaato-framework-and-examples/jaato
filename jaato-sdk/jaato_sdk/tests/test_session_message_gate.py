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
    with pytest.raises(ValueError, match="requires text, attachments"):
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


@pytest.mark.asyncio
async def test_file_refs_and_text_attachments_ride_the_payload_at_1_24():
    c, sent = _client("1.24")
    await c.send_session_message(
        "s-b", "", file_refs=["reports/q3.md", {"path": "a", "workspace": "/w"}],
        text_attachments=[{"name": "fix.patch", "text": "--- a"}])
    p = sent[0].payload
    assert p["file_refs"] == ["reports/q3.md", {"path": "a", "workspace": "/w"}]
    assert p["text_attachments"] == [{"name": "fix.patch", "text": "--- a"}]
    assert p["text"] == "" and "attachments" not in p


@pytest.mark.asyncio
async def test_files_are_refused_below_1_24_and_text_alone_is_not():
    """A 1.23 daemon reads neither key: it would deliver the text WITHOUT
    the files and answer accepted -- a degraded call that reads as success."""
    c, sent = _client("1.23")
    with pytest.raises(ValueError, match="file_refs / text_attachments"):
        await c.send_session_message("s-b", "see", file_refs=["a.md"])
    with pytest.raises(ValueError, match="file_refs / text_attachments"):
        await c.send_session_message("s-b", "see",
                                     text_attachments=[{"text": "x"}])
    assert sent == []
    await c.send_session_message("s-b", "see")
    assert sent[0].payload == {"target": "s-b", "text": "see"}
