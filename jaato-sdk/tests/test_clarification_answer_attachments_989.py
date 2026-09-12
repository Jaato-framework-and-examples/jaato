"""The SDK refuses to answer a clarification with media a daemon will drop.

#845's rule 3, transferred verbatim (#989).  An additive optional field is
normally safe to send blind — an older peer ignores it and the call
degrades to what it always did.  That reasoning holds for a ``request_id``
and NOT for bytes.  Here the degraded call answers the agent's question
with the recording thrown away, and the agent acts on it: a clarification
answer reaches the model as a TOOL RESULT, which it reads as fact.  For a
voice-only answer it is worse still — the matching answer string is
legitimately ``""`` (the utterance IS the answer, #838), and the daemon's
``_parse_answer`` reads an empty response as ``free_text=""``.  So a
dropped payload becomes *a blank answer reported as a successful one*.

Hence: refused, not degraded, below protocol 1.6 — including against a
daemon whose version is UNKNOWN, because "I have not been told what this
peer can do" is not a licence to send bytes it may silently discard.

These guards also pin the normalisation, which is deliberately the SAME
encoder ``send_message(attachments=...)`` uses: a client that can attach
to a message attaches to an answer with no second code path, and a
file-path string is read here rather than daemon-side (the daemon cannot
reach a client's filesystem, least of all over WS).
"""

import base64

import pytest

from jaato_sdk.client.ipc import IPCClient


VOICE = b"RIFF----WAVE" + bytes(range(200))


class _Client(IPCClient):
    """An ``IPCClient`` that records events instead of writing a socket."""

    def __init__(self, protocol):
        self.sent = []
        self._server_protocol_version = protocol

    async def _send_event(self, event):  # type: ignore[override]
        self.sent.append(event)


def _wav_dict():
    return {"mime_type": "audio/wav",
            "data": base64.b64encode(VOICE).decode("ascii"),
            "display_name": "answer.wav"}


@pytest.mark.asyncio
async def test_a_voice_answer_reaches_the_wire_keyed_by_question():
    client = _Client("1.6")

    await client.respond_to_clarification_batch(
        "r1", ["", "yes"], answer_attachments={1: [_wav_dict()]},
    )

    event = client.sent[0]
    assert event.answers == ["", "yes"]
    assert list(event.answer_attachments) == ["1"]      # JSON-shaped key
    entry = event.answer_attachments["1"][0]
    assert base64.b64decode(entry["data"]) == VOICE
    # Minted by the SENDER, which is the side holding the file and
    # therefore the side that archives it (#850).
    assert entry["attachment_id"].startswith("att_")


@pytest.mark.asyncio
async def test_raw_bytes_are_base64_encoded_by_the_same_encoder_as_send_message():
    client = _Client("1.6")

    await client.respond_to_clarification_batch(
        "r1", [""], answer_attachments={"1": [{"mime_type": "audio/wav",
                                               "data": VOICE}]},
    )

    entry = client.sent[0].answer_attachments["1"][0]
    assert isinstance(entry["data"], str)
    assert base64.b64decode(entry["data"]) == VOICE


@pytest.mark.asyncio
async def test_a_file_path_is_read_client_side(tmp_path):
    path = tmp_path / "answer.wav"
    path.write_bytes(VOICE)
    client = _Client("1.6")

    await client.respond_to_clarification_batch(
        "r1", [""], answer_attachments={"1": [str(path)]},
    )

    entry = client.sent[0].answer_attachments["1"][0]
    assert base64.b64decode(entry["data"]) == VOICE
    assert entry["display_name"] == "answer.wav"


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["1.5", "1.0", None])
async def test_an_old_or_unknown_daemon_is_refused_not_degraded(protocol):
    client = _Client(protocol)

    with pytest.raises(ValueError) as exc:
        await client.respond_to_clarification_batch(
            "r1", [""], answer_attachments={"1": [_wav_dict()]},
        )

    assert "1.6" in str(exc.value)
    assert client.sent == []        # nothing was sent at all


@pytest.mark.asyncio
async def test_a_text_only_answer_is_unchanged_on_every_protocol():
    """The refusal must not become a floor on answering in text."""
    client = _Client("1.0")

    await client.respond_to_clarification_batch("r1", ["Dani"])

    assert client.sent[0].answers == ["Dani"]
    assert client.sent[0].answer_attachments == {}


@pytest.mark.asyncio
async def test_a_cancel_is_not_refused_for_carrying_stale_attachments():
    """``cancelled`` abandons the clarification; there is no answer to
    attach to, so the media is dropped rather than becoming a reason the
    cancel cannot get through — and a cancel that cannot get through is
    what blocks a turn forever (#704)."""
    client = _Client("1.5")

    await client.respond_to_clarification_batch(
        "r1", [], cancelled=True, answer_attachments={"1": [_wav_dict()]},
    )

    assert client.sent[0].cancelled is True
    assert client.sent[0].answer_attachments == {}
