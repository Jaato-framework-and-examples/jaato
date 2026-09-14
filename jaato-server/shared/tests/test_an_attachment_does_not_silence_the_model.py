"""An attachment in the message must not turn streaming off.

THE GAP (#837).  #830 gave the framework ears and #824 gave it a voice,
and the two could not be used together.  A user message carrying an
attachment leaves :meth:`JaatoSession.send_message` for
:meth:`send_message_with_parts`, and the loop it lands in --
``_run_chat_loop_with_parts`` -- called the BATCHED provider method
unconditionally::

    self._provider.complete(...)          # no on_chunk, ever

``self._use_streaming`` was never consulted there.  The telemetry label
``streaming=False`` was accurate: it was the behaviour, not a mislabel.

Measured on one daemon, one model (``openai/gpt-audio-mini`` on
openrouter), one ``modalities: {audio: bidirectional}`` tier:

===========================  ==========================================
prompt, no attachment        works -- 30 media chunks, a spoken answer
prompt + ``audio/wav``       ``400 Audio output requires stream: true``
===========================  ==========================================

The upstream is right to refuse.  OpenAI-shaped wires emit audio ONLY
while streaming -- ``STREAM_AUDIO_MIME`` spells out the pcm16 parameters
for exactly that reason -- so a batched request that also asks for audio
output cannot be satisfied.  "Hear a question, answer aloud", the whole
point of bidirectional audio, was therefore unreachable: each half
worked alone, ``jaato-scaffold validate`` reported no findings, the
session started, and the first turn that used both directions failed.

WHY IT WENT UNNOTICED.  The path predates media output and was built for
images, where a batched vision turn is perfectly reasonable.  Audio input
is the first attachment kind whose REPLY may itself be audio, which is
what makes the two paths mutually exclusive.

THE CONTRACT, in the order these guards assert it:

1. an attachment turn dispatches with ``on_chunk`` whenever the session
   wants streaming and the provider supports it -- the same decision the
   text loop makes, now shared as ``_resolve_use_streaming``;
2. a model that speaks during such a turn reaches subscribed clients,
   which is the payoff and not merely the mechanism;
3. a provider that cannot stream still gets a batched call, so the fix
   does not break the vision turns the path was built for;
4. streamed text is delivered ONCE.  The loop used to emit each
   response's assembled text because nothing else had; doing that on top
   of the chunks would render every answer twice, which is the obvious
   way to "fix" this and be worse off.

WHY THE GUARDS DRIVE ``send_message(attachments=...)`` rather than
``send_message_with_parts``.  The bug is reached by having an attachment
in an ordinary message; asserting from the parts entry point would leave
the routing in :meth:`send_message` -- the actual trigger -- untested.
"""

import base64
import pathlib
from unittest.mock import MagicMock

from jaato_sdk.events import MODEL_MEDIA_CALL_ID
from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    MediaDelta,
    Part,
    ProviderResponse,
    TokenUsage,
    TurnResult,
)

from shared.jaato_session import JaatoSession
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


SESSION_PATH = pathlib.Path(__file__).resolve().parents[1] / "jaato_session.py"


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="        use_streaming = self._resolve_use_streaming()\n\n"
             "        turn_start = datetime.now()",
        replace="        use_streaming = False\n\n"
                "        turn_start = datetime.now()",
        test="test_an_attachment_turn_still_streams",
        because="an attachment in the message meaning the turn is "
                "dispatched batched, so a speaking tier is refused "
                "upstream with 'Audio output requires stream: true'",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""        text = response.get_text() or ''
        if text and on_output and not use_streaming:
            on_output("model", text, "write")
        return text""",
        replace="""        text = response.get_text() or ''
        if text and on_output:
            on_output("model", text, "write")
        return text""",
        test="test_streamed_text_is_delivered_once",
        because="a streamed attachment turn emitting its answer twice -- "
                "once chunk by chunk and again as the assembled response",
    ),
]


# ============================================================ fixtures

WAV = {
    "mime_type": "audio/wav",
    "data": base64.b64encode(b"RIFF....WAVEfmt ").decode("ascii"),
    "display_name": "question.wav",
}


def _recording_provider(streams: bool = True, media=(), chunks=()):
    """A provider that records HOW each turn was dispatched.

    ``calls`` collects one dict per ``complete()`` -- whether a chunk
    callback was handed over, and the cancel token that came with it --
    so a guard can assert on the dispatch rather than on an internal
    flag that could be right while the call is wrong.

    Args:
        streams: What ``supports_streaming()`` reports.
        media: ``MediaDelta`` objects to push through ``on_chunk``,
            standing in for a model speaking its answer.
        chunks: Text chunks to push through ``on_chunk`` before them.
    """
    calls = []

    def complete(messages, **kwargs):
        on_chunk = kwargs.get("on_chunk")
        calls.append({
            "streamed": on_chunk is not None,
            "cancel_token": kwargs.get("cancel_token"),
        })
        if on_chunk is not None:
            for chunk in chunks:
                on_chunk(chunk)
            for delta in media:
                on_chunk(delta)
        return TurnResult.from_provider_response(ProviderResponse(
            parts=[Part(text="".join(chunks) or "Here is your answer.")],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, output_tokens=5,
                             total_tokens=15),
        ))

    provider = MagicMock()
    provider.name = "fake"
    provider.supports_streaming.return_value = streams
    provider.get_context_limit.return_value = 0
    provider.complete.side_effect = complete
    provider.calls = calls
    return provider


def _live_session(provider):
    runtime = MagicMock()
    runtime.create_provider.return_value = provider
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.ledger = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    runtime.registry.enrich_prompt.side_effect = (
        lambda prompt, **_k: MagicMock(prompt=prompt, metadata={})
    )
    session = JaatoSession(runtime, "openai/gpt-audio-mini")
    session.configure()
    return session


# ============================================================ the guards

def test_an_attachment_turn_still_streams():
    """The issue itself: the dispatch, not a flag that describes it."""
    provider = _recording_provider()
    session = _live_session(provider)

    session.send_message("What did I just say?", lambda *a, **k: None,
                         attachments=[WAV])

    assert provider.calls, "the attachment turn never reached the provider"
    assert all(c["streamed"] for c in provider.calls), (
        "an attachment turn was dispatched batched. OpenAI-shaped wires "
        "emit audio only while streaming, so a speaking tier answers this "
        "with 400 'Audio output requires stream: true' (#837)."
    )


def test_a_spoken_answer_to_a_spoken_question_reaches_the_client():
    """The payoff, end to end.

    Streaming is the mechanism; audio arriving at a subscribed client is
    what #837 is actually about, and it is possible to restore the
    dispatch and still drop the bytes -- a chunk callback that treats a
    ``MediaDelta`` as text loses the audio and corrupts the transcript.
    """
    spoken = [
        MediaDelta(mime_type="audio/pcm;rate=24000;channels=1",
                   data=b"\x00\x01" * 8, sequence=0, final=False),
        MediaDelta(mime_type="audio/pcm;rate=24000;channels=1",
                   data=b"\x02\x03" * 8, sequence=1, final=True),
    ]
    provider = _recording_provider(media=spoken)
    session = _live_session(provider)
    hooks = MagicMock()
    session._ui_hooks = hooks

    transcript = []
    session.send_message("What did I just say?",
                         lambda source, text, mode: transcript.append(
                             (source, text)),
                         attachments=[WAV])

    delivered = [
        kw for _a, kw in hooks.on_tool_output.call_args_list
        if kw.get("call_id") == MODEL_MEDIA_CALL_ID
    ]
    assert len(delivered) == 2, (
        f"the model spoke 2 chunks and {len(delivered)} reached the "
        f"client -- an audio question got no audible answer (#837)"
    )
    assert delivered[-1]["final"] is True, (
        "no chunk was marked final, so a client cannot close its "
        "playback buffer"
    )
    assert all(b"\x00\x01" not in text.encode() for _s, text in transcript), (
        "raw audio bytes leaked into the text transcript; a MediaDelta is "
        "emphatically not text"
    )


def test_a_batched_provider_still_gets_a_batched_call():
    """The vision turns this path was built for are not collateral.

    ``_resolve_use_streaming`` consults the PROVIDER as well as the
    session, so "always stream now" would be as wrong as "never stream".
    """
    provider = _recording_provider(streams=False)
    session = _live_session(provider)

    session.send_message("Describe this", lambda *a, **k: None,
                         attachments=[WAV])

    assert provider.calls and not any(c["streamed"] for c in provider.calls), (
        "a provider that reports supports_streaming() False was handed a "
        "chunk callback anyway"
    )


def test_streamed_text_is_delivered_once():
    """Chunks OR the assembled response, never both.

    The loop emitted each response's whole text because, batched, nothing
    else did.  Left in place alongside the chunks it renders every answer
    twice -- the obvious way to fix #837 and come out behind.
    """
    provider = _recording_provider(chunks=("Hello ", "there."))
    session = _live_session(provider)

    emitted = []
    session.send_message(
        "What did I just say?",
        lambda source, text, mode: emitted.append((source, text, mode)),
        attachments=[WAV],
    )

    model_text = "".join(t for s, t, _m in emitted if s == "model")
    assert model_text == "Hello there.", (
        f"the streamed answer was delivered as {model_text!r} -- the "
        f"assembled response was emitted on top of the chunks"
    )
    assert [m for s, _t, m in emitted if s == "model"] == ["write", "append"], (
        "the first chunk must open a block and later ones append to it"
    )


def test_a_batched_attachment_turn_still_emits_its_answer():
    """Suppression is conditional on having streamed, not unconditional.

    Silencing the assembled-response emission outright would fix the
    double render by making a batched vision turn produce no text at all.
    """
    provider = _recording_provider(streams=False)
    session = _live_session(provider)

    emitted = []
    session.send_message(
        "Describe this",
        lambda source, text, mode: emitted.append((source, text)),
        attachments=[WAV],
    )

    assert [t for s, t in emitted if s == "model"] == [
        "Here is your answer."], (
        "a batched attachment turn emitted no model text; nothing else "
        "was going to"
    )


def test_the_streaming_decision_is_made_in_one_place():
    """Both loops ask the same question, and neither answers it inline.

    The defect was two dispatch sites disagreeing.  A future edit that
    re-inlines the decision in either loop can put them back out of step
    without any behavioural test noticing until a speaking tier is tried.
    """
    source = SESSION_PATH.read_text(encoding="utf-8")
    assert source.count("def _resolve_use_streaming") == 1
    assert source.count("use_streaming = self._resolve_use_streaming()") == 2, (
        "a chat loop stopped asking _resolve_use_streaming for its "
        "streaming decision"
    )
