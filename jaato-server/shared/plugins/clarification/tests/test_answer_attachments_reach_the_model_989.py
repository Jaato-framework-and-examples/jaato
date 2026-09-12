"""A clarification answered with media puts BYTES in front of the model (#989).

The value of this feature is entirely at one end: a voice note the user
recorded has to arrive at the provider as audio the model can hear.  A
field that type-checks proves nothing, so these guards drive the whole
chain the daemon's answer actually travels --

    client submits  ->  relay result dict
                    ->  RunnerRPCClarificationChannel  (Answer.attachments)
                    ->  ClarificationPlugin._execute_clarification
                    ->  JaatoSession._build_tool_result  (ToolResult.attachments)
                    ->  the provider converter

-- and end on the wire block itself: an ``input_audio`` payload whose
base64 decodes back to the bytes the user recorded, and an ``image_url``
data URI for the screenshot case.

WHY THE TOOL-RESULT PATH.  A clarification answer IS a tool result; the
converters already marshal ``ToolResult.attachments`` on all three wire
families, and ``_gate_one_tool_result`` already gates them against the
active model.  Ferrying the bytes in as a separate user turn (the issue
body's first proposal) would put a message in history the user never
sent and leave the tool result claiming it carried nothing.

THE TWO KINDS ARE NOT THE SAME CASE, and both are here:

* a **voice note** answering a ``free_text`` question -- the answer
  string is legitimately ``""`` (#838: the utterance IS the message), and
  the framework must not read that as a skip;
* a **screenshot** attached to a CHOICE answer -- "1. you attach a
  screenshot" is the ordinal AND the image, two axes rather than
  alternatives.  This is the case the issue body called "N/A" and the
  design note corrected.
"""

import base64

import pytest

from jaato_sdk.plugins.model_provider.types import Attachment

from shared.plugins.clarification.channels import (
    RunnerRPCClarificationChannel,
)
from shared.plugins.clarification.models import (
    Choice,
    ClarificationRequest,
    Question,
    QuestionType,
)
from shared.plugins.clarification.plugin import ClarificationPlugin
from shared.plugins.model_provider._attachments import (
    tool_result_followup_message,
)
from shared.tool_result_builder import extract_multimodal_attachments


VOICE = b"RIFF----WAVEfmt " + bytes(range(256)) * 8
SHOT = b"\x89PNG\r\n\x1a\n" + b"pixels" * 100


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _relay(result):
    """A relay channel whose RPC answers with *result*, recording the payload."""
    sent = {}

    def rpc(payload):
        sent.update(payload)
        return result

    channel = RunnerRPCClarificationChannel(rpc, agent_id="main")
    return channel, sent


def _plugin(channel):
    plugin = ClarificationPlugin()
    plugin.initialize({})
    plugin._channel = channel
    return plugin


# =====================================================================
# 1. The relay channel turns wire dicts into Answer.attachments
# =====================================================================

def test_a_voice_answer_arrives_as_an_attachment_on_the_answer():
    channel, _ = _relay({
        "cancelled": False,
        "answers": [""],
        "answer_attachments": {"1": [{
            "mime_type": "audio/wav",
            "data": _b64(VOICE),
            "display_name": "answer.wav",
        }]},
    })
    request = ClarificationRequest(
        context="need a name",
        questions=[Question(text="What's your name?",
                            question_type=QuestionType.FREE_TEXT)],
    )

    response = channel.request_clarification(request)

    answer = response.answers[0]
    assert [a.mime_type for a in answer.attachments] == ["audio/wav"]
    assert answer.attachments[0].data == VOICE
    assert answer.attachments[0].display_name == "answer.wav"


def test_a_blank_voice_answer_to_an_optional_question_is_not_a_skip():
    """#838 inside a clarification: an attachment IS content.

    Without this, the NORMAL voice case -- no typed text, a recording --
    is reported to the model as "the user declined to answer" while the
    audio rides along beside the claim.
    """
    channel, _ = _relay({
        "cancelled": False,
        "answers": [""],
        "answer_attachments": {"1": [{"mime_type": "audio/wav",
                                      "data": _b64(VOICE)}]},
    })
    request = ClarificationRequest(
        context="optional",
        questions=[Question(text="Anything to add?",
                            question_type=QuestionType.FREE_TEXT,
                            required=False)],
    )

    answer = channel.request_clarification(request).answers[0]

    assert answer.skipped is False
    assert answer.free_text == ""
    assert answer.attachments


def test_a_blank_answer_with_NO_attachment_is_still_a_skip():
    """The un-skipping is caused by the bytes, not by the code path."""
    channel, _ = _relay({"cancelled": False, "answers": [""]})
    request = ClarificationRequest(
        context="optional",
        questions=[Question(text="Anything to add?",
                            question_type=QuestionType.FREE_TEXT,
                            required=False)],
    )

    assert channel.request_clarification(request).answers[0].skipped is True


def test_an_attachment_rides_a_CHOICE_answer_without_disturbing_the_ordinal():
    """The design note's case: option 1 is 'you attach a screenshot'."""
    channel, _ = _relay({
        "cancelled": False,
        "answers": ["1"],
        "answer_attachments": {"1": [{"mime_type": "image/png",
                                      "data": _b64(SHOT),
                                      "display_name": "design.png"}]},
    })
    request = ClarificationRequest(
        context="design",
        questions=[Question(
            text="How should we design this?",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=[Choice(text="You attach a screenshot",
                            expects_attachment=True),
                     Choice(text="We discuss the design")],
        )],
    )

    answer = channel.request_clarification(request).answers[0]

    assert answer.selected_choices == [1]
    assert answer.attachments[0].data == SHOT


def test_the_per_choice_affordance_reaches_the_client_payload():
    channel, sent = _relay({"cancelled": False, "answers": ["1"]})
    request = ClarificationRequest(
        context="design",
        questions=[Question(
            text="How should we design this?",
            choices=[Choice(text="You attach a screenshot",
                            expects_attachment=True),
                     Choice(text="We discuss the design")],
        )],
    )

    channel.request_clarification(request)

    choices = sent["questions"][0]["choices"]
    assert choices[0]["expects_attachment"] is True
    # Absent, not false, on a choice that wants nothing: a client that
    # does not know the key sees the dict it has always seen.
    assert "expects_attachment" not in choices[1]


def test_a_pre_989_daemon_reply_parses_exactly_as_before():
    """No ``answer_attachments`` key at all — the whole pre-#989 wire."""
    channel, _ = _relay({"cancelled": False, "answers": ["Dani"]})
    request = ClarificationRequest(
        context="c",
        questions=[Question(text="Name?",
                            question_type=QuestionType.FREE_TEXT)],
    )

    answer = channel.request_clarification(request).answers[0]

    assert answer.free_text == "Dani"
    assert answer.attachments == []


# =====================================================================
# 2. The plugin folds them onto the tool result
# =====================================================================

def _run_plugin(answers, media, questions):
    channel, _ = _relay({
        "cancelled": False, "answers": answers, "answer_attachments": media,
    })
    plugin = _plugin(channel)
    return plugin._execute_clarification({
        "context": "c", "questions": questions,
    })


def test_the_result_carries_the_bytes_for_the_session_to_pick_up():
    result = _run_plugin(
        [""], {"1": [{"mime_type": "audio/wav", "data": _b64(VOICE),
                      "display_name": "answer.wav"}]},
        [{"text": "Name?", "question_type": "free_text"}],
    )

    assert result["_multimodal"] is True
    assert result["_multimodal_type"] == "attachments"
    assert len(result["_multimodal_attachments"]) == 1


def test_the_result_names_each_attachment_beside_its_own_answer():
    """A batch with two media answers is otherwise unattributable: the
    wire's follow-up message names files in ONE lead line with no
    structural link back to a question."""
    result = _run_plugin(
        ["", "1"],
        {"1": [{"mime_type": "audio/wav", "data": _b64(VOICE),
                "display_name": "answer.wav"}],
         "2": [{"mime_type": "image/png", "data": _b64(SHOT),
                "display_name": "design.png"}]},
        [{"text": "Name?", "question_type": "free_text"},
         {"text": "Design?", "choices": ["attach a shot", "discuss"]}],
    )

    first = result["responses"]["1"]["attachments"]
    second = result["responses"]["2"]["attachments"]
    assert first[0]["mime_type"] == "audio/wav"
    assert first[0]["display_name"] == "answer.wav"
    assert second[0]["mime_type"] == "image/png"
    # The ordinal is untouched by the image riding beside it.
    assert result["responses"]["2"]["selected"] == 1


def test_the_descriptor_never_carries_the_payload():
    result = _run_plugin(
        [""], {"1": [{"mime_type": "audio/wav", "data": _b64(VOICE)}]},
        [{"text": "Name?", "question_type": "free_text"}],
    )

    descriptor = result["responses"]["1"]["attachments"][0]
    assert "data" not in descriptor
    assert "attachment_id" in descriptor
    assert _b64(VOICE) not in repr(descriptor)


def test_a_text_only_clarification_result_is_unchanged():
    result = _run_plugin(
        ["Dani"], None, [{"text": "Name?", "question_type": "free_text"}],
    )

    assert "_multimodal" not in result
    assert result["responses"]["1"] == {"value": "Dani", "type": "free_text"}


# =====================================================================
# 3. The session's extractor turns them into ToolResult.attachments
# =====================================================================

def test_the_session_extractor_rebuilds_the_bytes():
    result = _run_plugin(
        [""], {"1": [{"mime_type": "audio/wav", "data": _b64(VOICE),
                      "display_name": "answer.wav"}]},
        [{"text": "Name?", "question_type": "free_text"}],
    )

    attachments = extract_multimodal_attachments(result)

    assert isinstance(attachments[0], Attachment)
    assert attachments[0].data == VOICE
    assert attachments[0].mime_type == "audio/wav"


def test_the_extractor_carries_EVERY_answer_s_media_not_just_the_first():
    """The pre-#989 ``_multimodal`` shapes return exactly one attachment;
    a clarification BATCH may be answered with several."""
    result = _run_plugin(
        ["", "1"],
        {"1": [{"mime_type": "audio/wav", "data": _b64(VOICE)}],
         "2": [{"mime_type": "image/png", "data": _b64(SHOT)}]},
        [{"text": "Name?", "question_type": "free_text"},
         {"text": "Design?", "choices": ["a", "b"]}],
    )

    attachments = extract_multimodal_attachments(result)

    assert [a.mime_type for a in attachments] == ["audio/wav", "image/png"]


@pytest.mark.parametrize("entry", [
    {"mime_type": "audio/wav", "data": "not base64!!"},
    {"mime_type": "", "data": _b64(VOICE)},
    "not a dict",
])
def test_an_unusable_entry_is_dropped_not_turned_into_empty_bytes(entry):
    """A silently EMPTY attachment is worse than an absent one — the model
    is shown a file and sees nothing."""
    assert extract_multimodal_attachments({
        "_multimodal": True,
        "_multimodal_type": "attachments",
        "_multimodal_attachments": [entry],
    }) is None


# =====================================================================
# 3b. The real session seam, not a stand-in for it
# =====================================================================

def test_the_session_moves_them_onto_the_ToolResult_and_strips_the_scaffolding():
    """``JaatoSession._build_tool_result`` itself, because it does TWO
    things and the second is easy to get wrong: it lifts the attachments
    onto ``ToolResult.attachments`` AND strips every ``_multimodal*`` key
    from the dict the model reads.  A payload left behind in the result
    text would reach the model as base64 prose."""
    from types import SimpleNamespace

    from jaato_sdk.plugins.model_provider.types import FunctionCall
    from shared.jaato_session import JaatoSession

    result = _run_plugin(
        [""], {"1": [{"mime_type": "audio/wav", "data": _b64(VOICE),
                      "display_name": "answer.wav"}]},
        [{"text": "Name?", "question_type": "free_text"}],
    )

    session = JaatoSession.__new__(JaatoSession)
    session._runtime = SimpleNamespace(registry=None)
    tool_result = session._build_tool_result(
        FunctionCall(id="call_1", name="request_clarification", args={}),
        result,
    )

    assert tool_result.attachments[0].data == VOICE
    assert not [k for k in tool_result.result if k.startswith("_multimodal")]
    assert _b64(VOICE) not in repr(tool_result.result)
    # The descriptor stays: the model must know WHICH answer the audio
    # belongs to, and that it arrived as audio at all.
    assert tool_result.result["responses"]["1"]["attachments"][0][
        "mime_type"] == "audio/wav"


# =====================================================================
# 4. The wire: the bytes reach the provider
# =====================================================================

def _wire_blocks(result):
    attachments = extract_multimodal_attachments(result)
    message = tool_result_followup_message(
        attachments, pdf_as_file=True, audio_as_input_audio=True,
    )
    return message["content"]


def test_a_voice_answer_reaches_an_openai_shaped_wire_as_input_audio():
    """The end of the chain, and the only part that makes the feature real."""
    result = _run_plugin(
        [""], {"1": [{"mime_type": "audio/wav", "data": _b64(VOICE),
                      "display_name": "answer.wav"}]},
        [{"text": "Name?", "question_type": "free_text"}],
    )

    blocks = _wire_blocks(result)

    audio = [b for b in blocks if b.get("type") == "input_audio"]
    assert len(audio) == 1
    assert audio[0]["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio[0]["input_audio"]["data"]) == VOICE


def test_a_screenshot_answer_reaches_the_same_wire_as_an_image():
    result = _run_plugin(
        ["1"], {"1": [{"mime_type": "image/png", "data": _b64(SHOT),
                       "display_name": "design.png"}]},
        [{"text": "Design?", "choices": ["attach a shot", "discuss"]}],
    )

    blocks = _wire_blocks(result)

    images = [b for b in blocks if b.get("type") == "image_url"]
    assert len(images) == 1
    url = images[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == SHOT


def test_a_wire_without_audio_withholds_it_rather_than_relabelling_it():
    """#829's rule, unchanged by this feature: a wire that does not carry
    audio says so to the model instead of mislabelling it as an image."""
    result = _run_plugin(
        [""], {"1": [{"mime_type": "audio/wav", "data": _b64(VOICE)}]},
        [{"text": "Name?", "question_type": "free_text"}],
    )
    attachments = extract_multimodal_attachments(result)

    message = tool_result_followup_message(attachments, label="Image")

    text = message["content"][0]["text"]
    assert "withheld" in text.lower()
    assert not [b for b in message["content"] if b.get("type") == "image_url"]
