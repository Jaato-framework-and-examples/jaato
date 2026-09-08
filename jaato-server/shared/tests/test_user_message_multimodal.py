"""User-message multimodal ferry — the two pure keystones:

- SDK ``IPCClient._normalize_attachments``: client-side expansion to the
  canonical wire shape
  ``{mime_type, data: base64-str, display_name, attachment_id}``.
- Runner-side ``JaatoSession._parts_from_user_message``: wire dicts → Parts
  (text + inline image bytes) for the multimodal loop.

Both are self-free so they test without a live client/session.

``attachment_id`` joined the shape in #850: an inbound attachment carried
no identifier of any kind, which is what made purging consumed audio from
history untraceable.  It is a digest of the payload, so both layers mint
the same value independently — see
``test_heard_audio_does_not_accumulate`` for what depends on that.
"""

import base64

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.media_identity import ATTACHMENT_ID_KEY, mint_attachment_id
from jaato_sdk.plugins.model_provider.types import Part
from shared.jaato_session import JaatoSession


# ---------------------------------------------- SDK normalization (Layer 1)

def test_normalize_dict_base64_passthrough():
    out = IPCClient._normalize_attachments(
        [{"mime_type": "image/png", "data": "QUJD", "display_name": "x.png"}])
    assert out == [{"mime_type": "image/png", "data": "QUJD",
                    "display_name": "x.png",
                    ATTACHMENT_ID_KEY: mint_attachment_id("QUJD")}]


def test_normalize_keeps_an_id_the_caller_supplied():
    """A caller with an id scheme of its own is not overwritten.

    The framework's digest is a default, not a claim of ownership: an
    integrator whose archive already keys recordings by its own reference
    needs THAT reference in the marker, not one jaato invented.
    """
    out = IPCClient._normalize_attachments(
        [{"mime_type": "audio/wav", "data": "QUJD",
          ATTACHMENT_ID_KEY: "call-2026-09-07-0031"}])
    assert out[0][ATTACHMENT_ID_KEY] == "call-2026-09-07-0031"


def test_normalize_dict_bytes_to_base64():
    out = IPCClient._normalize_attachments(
        [{"mime_type": "image/png", "data": b"ABC"}])
    assert out[0]["data"] == base64.b64encode(b"ABC").decode("ascii")


def test_normalize_path_to_dict(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(b"\x89PNG\r\n")
    out = IPCClient._normalize_attachments([str(p)])
    assert out[0]["mime_type"] == "image/png"
    assert base64.b64decode(out[0]["data"]) == b"\x89PNG\r\n"
    assert out[0]["display_name"] == "pic.png"


def test_normalize_skips_unknown_and_none():
    assert IPCClient._normalize_attachments([123, None]) == []
    assert IPCClient._normalize_attachments(None) == []


# ------------------------------------- runner-side Part building (Layer 6)

def test_parts_text_and_image():
    atts = [{"mime_type": "image/png",
             "data": base64.b64encode(b"PNGDATA").decode("ascii")}]
    parts = JaatoSession._parts_from_user_message(None, "describe this", atts)
    assert parts[0].text == "describe this"
    assert parts[1].inline_data["mime_type"] == "image/png"
    assert parts[1].inline_data["data"] == b"PNGDATA"   # base64 → raw bytes


def test_parts_image_only_when_no_text():
    atts = [{"mime_type": "image/jpeg",
             "data": base64.b64encode(b"X").decode("ascii")}]
    parts = JaatoSession._parts_from_user_message(None, "", atts)
    assert len(parts) == 1 and parts[0].inline_data is not None
