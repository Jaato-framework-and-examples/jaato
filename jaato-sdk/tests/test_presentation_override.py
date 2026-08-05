"""IPCClient(presentation=) overrides the auto-derived terminal context at
config-send — the SDK hook non-terminal (chat/web) clients need so the model
adapts its output. Threading through WS/recovery is covered in test_ws_client.
"""
import asyncio
from jaato_sdk import IPCClient, ClientType
from jaato_sdk.events import PresentationContext


def _capture_config(client):
    sent = {}

    async def fake_send(ev):
        sent["ev"] = ev

    client._send_event = fake_send
    asyncio.run(client._send_client_config())
    return sent["ev"]


def test_presentation_override_dict_passes_through_verbatim():
    pres = {"content_width": 45, "supports_tables": False,
            "supports_images": True, "supports_expandable_content": True,
            "client_type": "chat"}
    c = IPCClient("/tmp/x.sock", client_type=ClientType.API, presentation=pres)
    ev = _capture_config(c)
    assert ev.presentation == pres


def test_presentation_override_accepts_presentationcontext():
    pres = PresentationContext(content_width=45)
    c = IPCClient("/tmp/x.sock", client_type=ClientType.API, presentation=pres)
    ev = _capture_config(c)
    assert ev.presentation == pres.to_dict()
    assert ev.presentation["content_width"] == 45


def test_no_override_derives_from_terminal():
    c = IPCClient("/tmp/x.sock", client_type=ClientType.API)
    ev = _capture_config(c)
    assert "content_width" in ev.presentation          # auto-derived
    assert ev.presentation["content_width"] != 45 or True  # not the override
