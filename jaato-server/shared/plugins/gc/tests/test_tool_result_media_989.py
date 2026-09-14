"""GC can see the bytes a TOOL RESULT carries, not only a user message's.

THE GAP (#989, and #850 reopened).  ``message_media_bytes``,
``estimate_media_tokens`` and the eviction walk all read ``part.inline_data``
and nothing else; ``function_response`` appeared in ``gc/utils.py`` only for
message GROUPING.  So a ``ToolResult.attachments`` payload was:

* sized at the one-token floor, so no token threshold could fire on it;
* absent from ``context_usage["media_bytes"]``, so ``media_pressure_reason``
  could not fire on it either;
* invisible to ``evict_consumed_media``, so it rode every later request for
  the life of the session.

Pre-existing for the ``_multimodal`` image tools — and a clarification
answered by voice is where it RECURS, once per question, which is what
makes it a prerequisite for #989 rather than a tidy-up.

WHAT MUST NOT CHANGE WHEN THE BYTES GO.  A tool result is not a bag of
bytes: the model answered from its ``result``, and the ledger, the
completion processors and enrichment all read that structure.  So eviction
drops the attachment and appends the marker to ``model_suffix`` — the
existing model-facing-only channel — rather than replacing the part.

AND THE PER-MIME SPLIT IS #850's, UNCHANGED.  ``media_evict_mime_prefixes``
defaults to ``("audio/",)`` on the stated reasoning that "an image is
routinely re-examined across turns, a recording is not".  That is exactly
the right split for a clarification: the screenshot attached to a design
question survives the turns that discuss it, the voice note answering
"what's your name" does not.  No new knob — these guards assert the default
already does it.
"""

from jaato_sdk.media_identity import mint_attachment_id
from jaato_sdk.plugins.model_provider.types import (
    Attachment,
    Message,
    Part,
    Role,
    ToolResult,
)

from shared.plugins.gc.utils import (
    DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
    estimate_message_tokens,
    evict_consumed_media,
    history_media_bytes,
    message_media_bytes,
    message_media_tokens,
)


AUDIO = b"\x01\x02\x03\x04" * 40_000      # 160 kB, like a short utterance
IMAGE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 60_000


def _tool_result_message(*attachments: Attachment) -> Message:
    """One tool-result message carrying the given attachments."""
    return Message(
        role=Role.USER,
        parts=[Part.from_function_response(ToolResult(
            call_id="call_1",
            name="request_clarification",
            result={"responses": {"1": {"value": "", "type": "free_text"}}},
            attachments=list(attachments),
        ))],
    )


def _audio() -> Attachment:
    return Attachment(mime_type="audio/wav", data=AUDIO,
                      display_name="answer.wav")


def _image() -> Attachment:
    return Attachment(mime_type="image/png", data=IMAGE,
                      display_name="design.png")


# ---------------------------------------------------------------------
# 1. The bytes are counted
# ---------------------------------------------------------------------

def test_tool_result_attachment_counts_toward_media_bytes():
    msg = _tool_result_message(_audio())
    assert message_media_bytes(msg) == len(AUDIO)
    assert history_media_bytes([msg]) == len(AUDIO)


def test_both_kinds_are_counted_not_only_audio():
    msg = _tool_result_message(_audio(), _image())
    assert message_media_bytes(msg) == len(AUDIO) + len(IMAGE)


def test_a_text_only_tool_result_still_counts_zero():
    msg = Message(role=Role.USER, parts=[Part.from_function_response(
        ToolResult(call_id="c", name="t", result={"ok": True}),
    )])
    assert message_media_bytes(msg) == 0
    assert message_media_tokens(msg) == 0


# ---------------------------------------------------------------------
# 2. The bytes are sized in tokens, not scored at the floor
# ---------------------------------------------------------------------

def test_tool_result_audio_is_sized_above_the_one_token_floor():
    msg = _tool_result_message(_audio())
    # 160 kB of audio at the documented 4800 B/token anchor.
    assert message_media_tokens(msg) == len(AUDIO) // 4800
    # And the whole-message estimate carries it, which is the number every
    # strategy's percentage is computed from.
    text_only = Message(role=Role.USER, parts=[Part.from_function_response(
        ToolResult(call_id="call_1", name="request_clarification",
                   result={"responses": {"1": {"value": "",
                                               "type": "free_text"}}}),
    )])
    assert (estimate_message_tokens(msg)
            - estimate_message_tokens(text_only)) == len(AUDIO) // 4800


def test_tool_result_image_is_sized_too():
    msg = _tool_result_message(_image())
    assert message_media_tokens(msg) == len(IMAGE) // 1500
    assert estimate_message_tokens(msg) > len(IMAGE) // 1500


# ---------------------------------------------------------------------
# 3. Eviction reaches them — and obeys the per-mime default
# ---------------------------------------------------------------------

def test_audio_attachment_is_evicted_under_the_default_prefixes():
    history = [_tool_result_message(_audio())]
    new_history, reclaimed, evicted_ids = evict_consumed_media(
        history, DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
    )
    assert reclaimed == len(AUDIO)
    assert len(evicted_ids) == 1
    assert history_media_bytes(new_history) == 0


def test_image_attachment_survives_the_default_prefixes():
    """#850's split, applied to a tool result: a screenshot is re-examined."""
    history = [_tool_result_message(_image())]
    new_history, reclaimed, evicted_ids = evict_consumed_media(
        history, DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
    )
    assert (reclaimed, evicted_ids) == (0, [])
    assert new_history is history          # not even copied
    assert history_media_bytes(new_history) == len(IMAGE)


def test_a_mixed_result_keeps_the_image_and_loses_the_audio():
    history = [_tool_result_message(_audio(), _image())]
    new_history, reclaimed, _ = evict_consumed_media(
        history, DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
    )
    assert reclaimed == len(AUDIO)
    kept = new_history[0].parts[0].function_response.attachments
    assert [a.mime_type for a in kept] == ["image/png"]
    assert kept[0].data == IMAGE


# ---------------------------------------------------------------------
# 4. Purging, not deleting — and the structure survives
# ---------------------------------------------------------------------

def test_the_marker_names_an_id_recomputable_from_the_recording():
    history = [_tool_result_message(_audio())]
    new_history, _, _ = evict_consumed_media(
        history, DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
    )
    suffix = new_history[0].parts[0].function_response.model_suffix
    assert mint_attachment_id(AUDIO) in suffix
    assert "answer.wav" in suffix
    # The payload itself is never rendered into the marker.
    assert "\x01\x02\x03" not in suffix


def test_eviction_preserves_the_result_the_model_answered_from():
    history = [_tool_result_message(_audio())]
    new_history, _, _ = evict_consumed_media(
        history, DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
    )
    fr = new_history[0].parts[0].function_response
    assert fr.call_id == "call_1"
    assert fr.name == "request_clarification"
    assert fr.result == {"responses": {"1": {"value": "",
                                             "type": "free_text"}}}
    assert fr.attachments is None


def test_an_existing_model_suffix_is_not_clobbered():
    msg = _tool_result_message(_audio())
    msg.parts[0].function_response.model_suffix = "earlier steering"
    new_history, _, _ = evict_consumed_media(
        [msg], DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
    )
    suffix = new_history[0].parts[0].function_response.model_suffix
    assert suffix.startswith("earlier steering")
    assert mint_attachment_id(AUDIO) in suffix


def test_the_original_history_is_not_mutated():
    """Eviction copies the messages it rewrites, as it always has."""
    history = [_tool_result_message(_audio())]
    evict_consumed_media(history, DEFAULT_MEDIA_EVICT_MIME_PREFIXES)
    assert history[0].parts[0].function_response.attachments[0].data == AUDIO


def test_protect_last_messages_still_shields_a_tool_result():
    history = [_tool_result_message(_audio())]
    new_history, reclaimed, _ = evict_consumed_media(
        history, DEFAULT_MEDIA_EVICT_MIME_PREFIXES,
        protect_last_messages=1,
    )
    assert (new_history, reclaimed) == (history, 0)
