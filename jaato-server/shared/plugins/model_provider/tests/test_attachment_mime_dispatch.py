"""#829 — an OpenAI-shaped wire dispatches on mime instead of assuming image.

``_openai_compat/converters.py`` used to marshal *every* ``inline_data`` part
on a user message into an ``image_url`` block, defaulting a missing mime to
``image/png``.  A PDF went to the wire as ``image_url`` with
``data:application/pdf;base64,...``; audio as ``data:audio/wav;...``; a part
with no declared mime was *asserted* to be a PNG.  Nothing was refused,
logged, or stripped.

These tests pin the three halves of the fix:

* the shared dispatch in ``model_provider/_attachments`` (the policy),
* the ``_openai_compat`` converter shared by nim / vllm / lmstudio /
  tensorrt_llm / zhipuai_openai / triton / nebius / ovhcloud / doubleword,
  every one of which declares ``pdf_input=False``,
* the ``openrouter`` converter, which declares ``pdf_input=True`` and whose
  mime-dispatch shape the shared module was lifted from — it must still carry
  PDFs as ``file`` blocks after the lift.

The invariant across all of them: **no non-image ever appears inside an
``image_url`` block**.
"""

import base64
import json

import pytest

from jaato_sdk.plugins.model_provider.types import (
    Attachment,
    Message,
    Part,
    Role,
    ToolResult,
)

from shared.plugins.model_provider import _attachments
from shared.plugins.model_provider._attachments import (
    attachment_content_block,
    marshal_attachments,
    tool_result_followup_message,
    user_message_with_attachments,
    withheld_attachment_note,
)
from shared.plugins.model_provider._openai_compat.converters import (
    message_to_openai as compat_to_openai,
)
from shared.plugins.model_provider.openrouter.converters import (
    message_to_openai as openrouter_to_openai,
)
from shared.plugins.model_provider.nebius.converters import (
    message_to_openai as nebius_to_openai,
)

_PNG = b"\x89PNG\r\n\x1a\nFAKE-IMAGE-BYTES"
_PDF = b"%PDF-1.4 FAKE-DOCUMENT-BYTES %%EOF"
_WAV = b"RIFF....WAVEfmt FAKE-AUDIO-BYTES"

_PNG_B64 = base64.b64encode(_PNG).decode("utf-8")
_PDF_B64 = base64.b64encode(_PDF).decode("utf-8")
_WAV_B64 = base64.b64encode(_WAV).decode("utf-8")

# The two converters that share the OpenAI-compat policy (image-only).  Both
# reach every provider in the issue's scope: ``_openai_compat`` is what nim,
# vllm, lmstudio, tensorrt_llm, zhipuai_openai, triton, nebius, ovhcloud and
# doubleword actually run; ``nebius/converters.py`` is the in-tree duplicate
# the conformance registry points at, and carried the identical defect.
IMAGE_ONLY_CONVERTERS = [
    pytest.param(compat_to_openai, id="_openai_compat"),
    pytest.param(nebius_to_openai, id="nebius"),
]


def _user_msg(mime, data, text="look at this", display_name=None):
    inline = {"mime_type": mime, "data": data}
    if display_name is not None:
        inline["display_name"] = display_name
    parts = [Part(inline_data=inline)]
    if text:
        parts.insert(0, Part(text=text))
    return Message(role=Role.USER, parts=parts)


def _tool_msg(*attachments):
    return Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
        call_id="c1", name="readFile", result={"ok": True},
        attachments=list(attachments),
    ))])


def _image_urls(wire):
    """Every ``image_url`` data URL anywhere in a converted wire message."""
    urls = []
    for msg in wire:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "image_url":
                urls.append(block["image_url"]["url"])
    return urls


# ---------------------------------------------------------- the shared policy

class TestAttachmentContentBlock:
    """The dispatch itself: what each mime becomes, and what becomes nothing."""

    def test_image_becomes_an_image_url_block(self):
        block = attachment_content_block("image/png", _PNG)
        assert block == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"},
        }

    def test_pdf_is_withheld_unless_the_wire_declares_it(self):
        assert attachment_content_block("application/pdf", _PDF) is None

    def test_pdf_becomes_a_file_block_when_the_wire_declares_it(self):
        block = attachment_content_block(
            "application/pdf", _PDF, "report.pdf", pdf_as_file=True)
        assert block["type"] == "file"
        assert block["file"]["filename"] == "report.pdf"
        assert block["file"]["file_data"] == f"data:application/pdf;base64,{_PDF_B64}"

    @pytest.mark.parametrize("mime", ["audio/wav", "video/mp4",
                                      "application/octet-stream", "", None])
    def test_everything_else_is_withheld_on_both_wires(self, mime):
        assert attachment_content_block(mime, _WAV) is None
        assert attachment_content_block(mime, _WAV, pdf_as_file=True) is None

    def test_absent_mime_is_not_asserted_to_be_a_png(self):
        # The defect in one line: a part with no declared mime was sent as an
        # image.  Guessing is not a policy.
        assert attachment_content_block("", _PNG) is None

    def test_already_base64_payload_is_passed_through(self):
        block = attachment_content_block("image/jpeg", _PNG_B64)
        assert block["image_url"]["url"] == f"data:image/jpeg;base64,{_PNG_B64}"


class TestWithheldNote:
    """Withholding is stated, not silent — the model can see the gap."""

    def test_nothing_withheld_means_no_note(self):
        assert withheld_attachment_note([]) is None

    def test_note_names_the_mime(self):
        note = withheld_attachment_note(["application/pdf"])
        assert note.startswith("[Attachment withheld:")
        assert "application/pdf" in note

    def test_note_counts_repeats(self):
        note = withheld_attachment_note(["audio/wav", "audio/wav"])
        assert "audio/wav (x2)" in note

    def test_marshal_logs_a_warning_for_the_operator(self, caplog):
        with caplog.at_level("WARNING", logger=_attachments.__name__):
            blocks, withheld = marshal_attachments([("audio/wav", _WAV, None)])
        assert blocks == [] and withheld == ["audio/wav"]
        assert any("withheld" in r.message.lower() for r in caplog.records)


# --------------------------------------------- the OpenAI-compat wire (#829)

@pytest.mark.parametrize("convert", IMAGE_ONLY_CONVERTERS)
class TestOpenAICompatUserMessage:

    def test_image_still_reaches_the_wire(self, convert):
        # The fix must not cost the capability it protects.
        wire = convert(_user_msg("image/png", _PNG))
        assert _image_urls(wire) == [f"data:image/png;base64,{_PNG_B64}"]

    def test_pdf_is_not_sent_as_an_image(self, convert):
        wire = convert(_user_msg("application/pdf", _PDF, display_name="d.pdf"))
        assert _image_urls(wire) == []
        assert _PDF_B64 not in json.dumps(wire)

    def test_audio_is_not_sent_as_an_image(self, convert):
        wire = convert(_user_msg("audio/wav", _WAV))
        assert _image_urls(wire) == []
        assert _WAV_B64 not in json.dumps(wire)

    def test_mimeless_part_is_not_asserted_to_be_a_png(self, convert):
        wire = convert(_user_msg(None, _PNG))
        assert _image_urls(wire) == []

    def test_the_drop_is_stated_not_silent(self, convert):
        wire = convert(_user_msg("application/pdf", _PDF, text="summarize"))
        blob = json.dumps(wire)
        assert "Attachment withheld" in blob
        assert "application/pdf" in blob
        # ...and the user's own text still reaches the model.
        assert "summarize" in blob

    def test_withheld_only_turn_keeps_the_plain_string_wire_shape(self, convert):
        msg, = convert(_user_msg("audio/wav", _WAV, text="transcribe this"))
        assert msg["role"] == "user"
        assert isinstance(msg["content"], str)
        assert msg["content"].startswith("transcribe this")
        assert "Attachment withheld" in msg["content"]

    def test_mixed_turn_carries_the_image_and_reports_the_rest(self, convert):
        msg, = convert(Message(role=Role.USER, parts=[
            Part(text="compare these"),
            Part(inline_data={"mime_type": "image/png", "data": _PNG}),
            Part(inline_data={"mime_type": "application/pdf", "data": _PDF}),
        ]))
        blocks = msg["content"]
        assert [b["type"] for b in blocks] == ["text", "image_url", "text"]
        assert blocks[0]["text"] == "compare these"
        assert "application/pdf" in blocks[-1]["text"]

    def test_text_only_turn_is_byte_identical_to_before(self, convert):
        msg, = convert(Message(role=Role.USER, parts=[Part(text="just text")]))
        assert msg == {"role": "user", "content": "just text"}


@pytest.mark.parametrize("convert", IMAGE_ONLY_CONVERTERS)
class TestOpenAICompatToolResult:

    def test_image_attachment_still_surfaces_as_a_followup(self, convert):
        wire = convert(_tool_msg(
            Attachment(mime_type="image/png", data=_PNG, display_name="x.png")))
        assert wire[0]["role"] == "tool"
        assert wire[-1]["role"] == "user"
        assert _image_urls(wire) == [f"data:image/png;base64,{_PNG_B64}"]

    def test_pdf_attachment_is_withheld_and_reported(self, convert):
        wire = convert(_tool_msg(
            Attachment(mime_type="application/pdf", data=_PDF,
                       display_name="d.pdf")))
        assert _image_urls(wire) == []
        assert _PDF_B64 not in json.dumps(wire)
        # A model told "here is the document" and shown nothing invents its
        # contents, so the follow-up still exists — carrying only the note.
        assert "Attachment withheld" in json.dumps(wire)

    def test_result_with_no_attachments_emits_no_followup(self, convert):
        wire = convert(Message(role=Role.TOOL, parts=[Part(
            function_response=ToolResult(call_id="c1", name="grep",
                                         result={"matches": 3}))]))
        assert [m["role"] for m in wire] == ["tool"]


# -------------------------------------------------- the OpenRouter wire (PDF)

class TestOpenRouterKeepsItsPdfExtension:
    """The lift must not cost OpenRouter the capability it declares."""

    def test_pdf_user_message_still_becomes_a_file_block(self):
        msg, = openrouter_to_openai(
            _user_msg("application/pdf", _PDF, display_name="report.pdf"))
        files = [b for b in msg["content"] if b["type"] == "file"]
        assert len(files) == 1
        assert files[0]["file"]["filename"] == "report.pdf"
        assert _PDF_B64 in files[0]["file"]["file_data"]

    def test_pdf_tool_result_still_surfaces_as_a_followup(self):
        wire = openrouter_to_openai(_tool_msg(
            Attachment(mime_type="application/pdf", data=_PDF,
                       display_name="d.pdf")))
        assert wire[-1]["role"] == "user"
        assert _PDF_B64 in json.dumps(wire)

    def test_image_user_message_still_becomes_an_image_url(self):
        wire = openrouter_to_openai(_user_msg("image/png", _PNG))
        assert _image_urls(wire) == [f"data:image/png;base64,{_PNG_B64}"]

    def test_uncarriable_audio_is_withheld_and_reported_rather_than_dropped(self):
        # OpenRouter already declined non-carriable mimes, but silently.
        # #830 gave it ears for the containers the wire names; one it does
        # NOT name still has to be refused out loud rather than dropped.
        msg, = openrouter_to_openai(_user_msg("audio/opus", _WAV, text="hear"))
        assert isinstance(msg["content"], str)
        assert "Attachment withheld" in msg["content"]

    def test_mimeless_part_is_not_asserted_to_be_a_png(self):
        wire = openrouter_to_openai(_user_msg(None, _PNG))
        assert _image_urls(wire) == []


# ------------------------------------------------------- the shared assemblers

class TestSharedAssemblers:
    """The two helpers the three converters now delegate their tail to."""

    def test_user_message_with_no_attachments_stays_a_string(self):
        msg, = user_message_with_attachments("hi", [Part(text="hi")])
        assert msg == {"role": "user", "content": "hi"}

    def test_user_message_orders_text_then_media_then_note(self):
        msg, = user_message_with_attachments("q", [
            Part(text="q"),
            Part(inline_data={"mime_type": "image/png", "data": _PNG}),
            Part(inline_data={"mime_type": "audio/wav", "data": _WAV}),
        ])
        assert [b["type"] for b in msg["content"]] == \
            ["text", "image_url", "text"]

    def test_empty_text_withheld_only_turn_is_just_the_note(self):
        msg, = user_message_with_attachments("", [
            Part(inline_data={"mime_type": "audio/wav", "data": _WAV})])
        assert msg["content"].startswith("[Attachment withheld:")

    def test_followup_is_none_when_there_is_nothing_to_say(self):
        assert tool_result_followup_message(None) is None
        assert tool_result_followup_message([]) is None

    def test_followup_label_is_the_caller_s(self):
        msg = tool_result_followup_message(
            [Attachment(mime_type="image/png", data=_PNG,
                        display_name="x.png")],
            label="Image",
        )
        assert msg["content"][0]["text"] == \
            "[Image returned by tool call: x.png]"


# ------------------------------------------------------------ #830: the ears

class TestAudioInputBlocks:
    """#830 — ``audio/*`` reaches an audio-declaring wire as ``input_audio``.

    The framework grew a mouth in #824/#828 and had no ears: ``input_audio``
    appeared nowhere in the tree, so audio was withheld by the same clause
    that (correctly) withholds video, no matter what the model's catalog
    entry said it could hear.  These pin the three things that had to become
    true — the block shape, the closed ``format`` vocabulary, and the fact
    that a wire which does NOT opt in is completely unchanged.
    """

    def test_wav_becomes_an_input_audio_block(self):
        block = attachment_content_block(
            "audio/wav", _WAV, audio_as_input_audio=True)
        assert block == {
            "type": "input_audio",
            "input_audio": {"data": _WAV_B64, "format": "wav"},
        }

    def test_audio_is_still_withheld_where_the_wire_does_not_declare_it(self):
        # The default is off, so every _openai_compat sharer is untouched.
        assert attachment_content_block("audio/wav", _WAV) is None
        assert attachment_content_block(
            "audio/wav", _WAV, pdf_as_file=True) is None

    @pytest.mark.parametrize("mime,fmt", [
        ("audio/wav", "wav"),
        ("audio/x-wav", "wav"),          # what mimetypes.guess_type gives .wav
        ("audio/wave", "wav"),
        ("audio/mpeg", "mp3"),           # ... and .mp3
        ("audio/mp3", "mp3"),
        ("audio/aiff", "aiff"),
        ("audio/aac", "aac"),
        ("audio/ogg", "ogg"),
        ("audio/flac", "flac"),
        ("audio/x-m4a", "m4a"),          # ... and .m4a
        ("audio/mp4", "m4a"),
        ("audio/WAV", "wav"),            # case is not part of the type
        ("audio/wav; codecs=1", "wav"),  # nor are parameters
    ])
    def test_container_mimes_map_to_the_wire_vocabulary(self, mime, fmt):
        assert _attachments.audio_wire_format(mime) == fmt

    @pytest.mark.parametrize("mime", [
        "audio/opus",       # a real codec the ``format`` vocabulary omits
        "audio/webm",
        "audio/basic",
        "audio/l16",        # RFC 2586: BIG-endian. pcm16 is little-endian.
        "audio/L16;rate=24000",
    ])
    def test_unnameable_containers_are_refused_not_guessed(self, mime):
        assert _attachments.audio_wire_format(mime) is None
        assert attachment_content_block(
            mime, _WAV, audio_as_input_audio=True) is None

    def test_the_framework_s_own_output_mime_is_accepted_as_input(self):
        # What the mouth emits (_media_deltas.STREAM_AUDIO_MIME) the ears
        # must accept — otherwise a spoken turn cannot be replayed to a
        # model, which is the whole shape of the asymmetry #830 records.
        from shared.plugins.model_provider._media_deltas import (
            STREAM_AUDIO_MIME,
        )
        assert _attachments.audio_wire_format(STREAM_AUDIO_MIME) == "pcm16"

    @pytest.mark.parametrize("mime", [
        "audio/pcm",                                    # bare = agreement
        "audio/pcm;rate=24000",
        "audio/pcm;rate=24000;channels=1;encoding=s16le",
    ])
    def test_raw_pcm_matching_the_pcm16_assertion_is_carried(self, mime):
        assert _attachments.audio_wire_format(mime) == "pcm16"

    @pytest.mark.parametrize("mime", [
        "audio/pcm;rate=16000",          # not the rate pcm16 means
        "audio/pcm;channels=2",          # not mono
        "audio/pcm;encoding=f32le",      # not 16-bit signed
    ])
    def test_raw_pcm_contradicting_the_assertion_is_refused(self, mime):
        # Relabelling 16 kHz stereo float as ``pcm16`` produces noise, and
        # noise the model describes confidently is worse than a stated gap.
        assert _attachments.audio_wire_format(mime) is None

    def test_openrouter_user_message_carries_audio(self):
        msg, = openrouter_to_openai(_user_msg("audio/wav", _WAV, text="hear"))
        assert [b["type"] for b in msg["content"]] == ["text", "input_audio"]
        assert msg["content"][1]["input_audio"] == {
            "data": _WAV_B64, "format": "wav"}

    def test_openrouter_tool_result_audio_surfaces_as_a_followup(self):
        wire = openrouter_to_openai(_tool_msg(
            Attachment(mime_type="audio/mpeg", data=_WAV,
                       display_name="clip.mp3")))
        assert wire[0]["role"] == "tool"
        assert wire[-1]["role"] == "user"
        blocks = [b for b in wire[-1]["content"] if b["type"] == "input_audio"]
        assert blocks[0]["input_audio"]["format"] == "mp3"

    def test_audio_never_rides_inside_an_image_url(self):
        # The #829 invariant has to survive the #830 addition.
        wire = openrouter_to_openai(_user_msg("audio/wav", _WAV))
        assert _image_urls(wire) == []

    @pytest.mark.parametrize("convert", IMAGE_ONLY_CONVERTERS)
    def test_image_only_wires_are_byte_identical_after_830(self, convert):
        msg, = convert(_user_msg("audio/wav", _WAV, text="hear"))
        assert isinstance(msg["content"], str)
        assert "Attachment withheld" in msg["content"]
        assert _WAV_B64 not in json.dumps(msg)


class TestWithheldNoteNamesWhatThisWireAccepts:
    """The remedy the note suggests has to be available on THIS wire.

    The note said "text, or an image" on every wire, which was already
    wrong for OpenRouter (it carries PDFs) and would send a model that
    just had audio refused looking for a form the wire also refuses.
    """

    @staticmethod
    def _offered(note):
        """The parenthesised accepted-forms clause, isolated from the rest."""
        return note.split("wire accepts (", 1)[1].split(")", 1)[0]

    def test_image_only_wire_offers_text_and_images_only(self):
        offered = self._offered(withheld_attachment_note(["audio/wav"]))
        assert offered == "text, or images"

    def test_pdf_wire_offers_pdfs(self):
        offered = self._offered(
            withheld_attachment_note(["video/mp4"], pdf_as_file=True))
        assert offered == "text, images, or PDF documents"

    def test_audio_wire_offers_audio_and_names_the_formats(self):
        offered = self._offered(withheld_attachment_note(
            ["audio/opus"], pdf_as_file=True, audio_as_input_audio=True))
        assert offered.startswith("text, images, PDF documents, or audio (")
        for fmt in ("wav", "mp3", "pcm16", "flac"):
            assert fmt in offered

    def test_counts_survive_the_new_clause(self):
        note = withheld_attachment_note(["video/mp4", "video/mp4"])
        assert "video/mp4 (x2)" in note
