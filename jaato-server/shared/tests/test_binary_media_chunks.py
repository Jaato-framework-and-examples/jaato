"""Delivery half of ``docs/design/binary-media-chunks.md``.

Covers getting bytes from a producer to a person: the widened chunk
primitive, the widened client event, audience routing, the content-gate
re-route, IPC backpressure, model-generated media, and the output-modality
capability the tier startup check probes.

The declaration half (direction-qualified tier ``modalities``) is tested
by ``test_tier_modalities.py``; the two meet at ``StreamChunk`` and
``ToolOutputEvent`` and nowhere else.
"""

import asyncio
import base64

import pytest

import server.ipc as ipc
from jaato_sdk.events import PresentationContext, ToolOutputEvent
from jaato_sdk.plugins.model_provider.types import MediaDelta
from shared.plugins.model_provider._openai_compat.base import (
    OpenAICompatProvider,
    _extract_audio_delta,
)
from shared.plugins.model_provider.base import (
    CAPABILITY_FIELDS,
    ModalityCapabilityMixin,
    ProviderCapabilities,
)
from shared.plugins.streaming import Audience, StreamChunk


# ==================== The chunk primitive ====================


class TestStreamChunk:
    """``StreamChunk`` carries bytes without disturbing text producers."""

    def test_positional_construction_still_works(self):
        """Every existing producer builds chunks positionally."""
        chunk = StreamChunk("hello", "match", 5)
        assert chunk.content == "hello"
        assert chunk.chunk_type == "match"
        assert chunk.sequence == 5

    def test_default_audience_is_model(self):
        """The default preserves the historical behaviour exactly."""
        assert StreamChunk("x").audience is Audience.MODEL

    def test_text_chunk_is_not_media(self):
        assert StreamChunk("x").is_media() is False
        assert StreamChunk("x").mime_type is None
        assert StreamChunk("x").data_b64() is None

    def test_media_chunk_exposes_mime_and_b64(self):
        chunk = StreamChunk(inline_data={"mime_type": "audio/wav", "data": b"\x00\x01"})
        assert chunk.is_media() is True
        assert chunk.mime_type == "audio/wav"
        assert chunk.data_b64() == base64.b64encode(b"\x00\x01").decode()

    def test_empty_data_is_not_media(self):
        """A declared mime type with no bytes is not a media chunk."""
        chunk = StreamChunk(inline_data={"mime_type": "audio/wav", "data": b""})
        assert chunk.is_media() is False

    def test_already_encoded_data_is_not_double_encoded(self):
        """A producer that encoded upstream must not be re-encoded."""
        chunk = StreamChunk(inline_data={"mime_type": "audio/wav", "data": "QUJD"})
        assert chunk.data_b64() == "QUJD"

    def test_to_dict_omits_inline_data_for_text(self):
        """The common text frame stays byte-identical to before media."""
        assert "inline_data" not in StreamChunk("hi").to_dict()

    def test_to_dict_base64_encodes_binary(self):
        chunk = StreamChunk(inline_data={"mime_type": "audio/wav", "data": b"\xff\xfe"})
        payload = chunk.to_dict()["inline_data"]
        assert payload["mime_type"] == "audio/wav"
        assert base64.b64decode(payload["data"]) == b"\xff\xfe"


class TestAudience:
    """Audience selects history entry, not publication."""

    @pytest.mark.parametrize(
        "audience,to_model,to_client",
        [
            (Audience.MODEL, True, False),
            (Audience.CLIENT, False, True),
            (Audience.BOTH, True, True),
        ],
    )
    def test_routing_predicates(self, audience, to_model, to_client):
        assert audience.reaches_model() is to_model
        assert audience.reaches_client() is to_client

    def test_serializes_as_plain_string(self):
        """The ``str`` mixin keeps JSON frames free of enum repr."""
        assert StreamChunk("x").to_dict()["audience"] == "model"


# ==================== The client-facing event ====================


class TestToolOutputEvent:
    """The event is widened, not replaced."""

    def test_text_event_is_not_media(self):
        assert ToolOutputEvent(chunk="hi").is_media() is False

    def test_media_event_round_trips(self):
        event = ToolOutputEvent(
            call_id="c1", mime_type="audio/wav", data_b64="AAE=",
            sequence=3, final=True, stream_id="s1",
        )
        assert event.is_media() is True
        rebuilt = ToolOutputEvent(**event.model_dump())
        assert rebuilt.mime_type == "audio/wav"
        assert rebuilt.sequence == 3
        assert rebuilt.final is True
        assert rebuilt.stream_id == "s1"

    def test_mime_without_data_is_not_media(self):
        """Half a payload is not a payload."""
        assert ToolOutputEvent(mime_type="audio/wav").is_media() is False
        assert ToolOutputEvent(data_b64="AAE=").is_media() is False

    def test_defaults_are_inert(self):
        """The new fields serialize (see the tool.output wire baseline),
        but their defaults mean nothing about an existing text chunk
        changes semantically."""
        event = ToolOutputEvent(agent_id="a", call_id="c", chunk="x")
        assert event.stream_id == ""
        assert event.sequence is None
        assert event.final is False


# ==================== IPC backpressure ====================


def _text_chunk(i):
    return ToolOutputEvent(call_id="c", chunk=f"t{i}")


def _media_chunk(i):
    return ToolOutputEvent(
        call_id="c", mime_type="audio/wav", data_b64="AA==", sequence=i
    )


class _Essential:
    """Stand-in for a lifecycle event: anything not a ToolOutputEvent."""


class TestBackpressure:
    """A bounded queue with a per-class drop policy (design §3.4)."""

    def test_tool_output_is_lossy_and_others_are_not(self):
        assert ipc._is_lossy_event(_text_chunk(0)) is True
        assert ipc._is_lossy_event(_Essential()) is False

    def test_media_is_evicted_before_text(self):
        """Media buys the most headroom per unit of lost output."""
        queue = asyncio.Queue()
        for event in (_text_chunk(0), _media_chunk(1), _text_chunk(2)):
            queue.put_nowait(event)

        assert ipc._evict_one_lossy(queue) is True

        remaining = [queue.get_nowait() for _ in range(queue.qsize())]
        assert [e.chunk for e in remaining] == ["t0", "t2"]

    def test_oldest_media_goes_first(self):
        """For a media stream, recency beats completeness."""
        queue = asyncio.Queue()
        for i in (1, 2):
            queue.put_nowait(_media_chunk(i))

        ipc._evict_one_lossy(queue)

        remaining = [queue.get_nowait() for _ in range(queue.qsize())]
        assert [e.sequence for e in remaining] == [2]

    def test_falls_back_to_oldest_text_when_no_media(self):
        queue = asyncio.Queue()
        for event in (_text_chunk(0), _text_chunk(1)):
            queue.put_nowait(event)

        assert ipc._evict_one_lossy(queue) is True

        remaining = [queue.get_nowait() for _ in range(queue.qsize())]
        assert [e.chunk for e in remaining] == ["t1"]

    def test_queue_of_essentials_is_left_untouched(self):
        """Dropping a lifecycle event desynchronises the client forever."""
        queue = asyncio.Queue()
        queue.put_nowait(_Essential())

        assert ipc._evict_one_lossy(queue) is False
        assert queue.qsize() == 1

    def test_eviction_preserves_relative_order(self):
        queue = asyncio.Queue()
        essential = _Essential()
        for event in (essential, _media_chunk(1), _text_chunk(2), _text_chunk(3)):
            queue.put_nowait(event)

        ipc._evict_one_lossy(queue)

        remaining = [queue.get_nowait() for _ in range(queue.qsize())]
        assert remaining[0] is essential
        assert [e.chunk for e in remaining[1:]] == ["t2", "t3"]

    def test_bound_defaults_and_rejects_nonsense(self, monkeypatch):
        """A typo must not silently reinstate the unbounded queue."""
        monkeypatch.setenv("JAATO_IPC_EVENT_QUEUE_MAX", "not-a-number")
        assert ipc._resolve_event_queue_max() == ipc._DEFAULT_EVENT_QUEUE_MAX

        monkeypatch.setenv("JAATO_IPC_EVENT_QUEUE_MAX", "0")
        assert ipc._resolve_event_queue_max() == ipc._DEFAULT_EVENT_QUEUE_MAX

        monkeypatch.setenv("JAATO_IPC_EVENT_QUEUE_MAX", "64")
        assert ipc._resolve_event_queue_max() == 64


# ==================== Model-generated media ====================


    def test_the_head_shortcut_keeps_the_policy(self):
        """The fast path must pick what the full search would have picked.

        ``_evict_one_lossy`` checks the HEAD before draining, because an
        audio flood is mostly media and the head is then already the
        preferred victim -- 1549us of drain-and-refill at the 2048 bound
        becomes ~18us.  A shortcut that changed WHICH event goes would be
        a policy change wearing a performance change's clothes, so this
        pins both arms: head-is-media takes the head, head-is-not falls
        through to the same search as before.
        """
        import asyncio
        from jaato_sdk.events import AgentOutputEvent, ToolOutputEvent

        def media(tag):
            return ToolOutputEvent(call_id="model-output", chunk=tag,
                                   mime_type="audio/pcm", data_b64="AA==")

        # head IS the victim -> shortcut
        queue = asyncio.Queue()
        for ev in (media("m1"), media("m2"), ToolOutputEvent(call_id="c", chunk="t")):
            queue.put_nowait(ev)
        assert ipc._evict_one_lossy(queue) is True
        assert [e.chunk for e in
                (queue.get_nowait() for _ in range(queue.qsize()))] == ["m2", "t"]

        # head is ESSENTIAL -> full search, media still preferred over text
        queue = asyncio.Queue()
        for ev in (AgentOutputEvent(text="e"), ToolOutputEvent(call_id="c", chunk="t1"),
                   media("m2")):
            queue.put_nowait(ev)
        assert ipc._evict_one_lossy(queue) is True
        left = [queue.get_nowait() for _ in range(queue.qsize())]
        assert isinstance(left[0], AgentOutputEvent), "an essential event was evicted"
        assert [getattr(e, "chunk", None) for e in left[1:]] == ["t1"]

class TestAudioDeltaExtraction:
    """``delta.audio`` is undocumented in the OpenAPI schema; read it defensively."""

    def test_attribute_style_delta(self):
        class _Audio:
            data = base64.b64encode(b"\x01\x02").decode()
            transcript = "hi"

        class _Delta:
            audio = _Audio()

        assert _extract_audio_delta(_Delta()) == (b"\x01\x02", "hi")

    def test_dict_style_delta(self):
        payload = {"audio": {"data": base64.b64encode(b"\x03").decode()}}
        assert _extract_audio_delta(payload) == (b"\x03", "")

    def test_model_extra_style_delta(self):
        """The SDK parks unknown fields in ``model_extra``."""

        class _Delta:
            model_extra = {"audio": {"data": base64.b64encode(b"\x04").decode()}}

        assert _extract_audio_delta(_Delta()) == (b"\x04", "")

    def test_text_only_delta_returns_none(self):
        class _Delta:
            content = "hello"

        assert _extract_audio_delta(_Delta()) is None

    def test_undecodable_payload_is_discarded_not_raised(self):
        """One bad chunk must not abort an otherwise healthy turn."""
        assert _extract_audio_delta({"audio": {"data": "!!!not-base64!!!"}}) is None

    def test_empty_payload_returns_none(self):
        assert _extract_audio_delta({"audio": {"data": ""}}) is None

    def test_stream_mime_names_the_pcm_parameters(self):
        """Headerless PCM carries no way to recover rate/channels."""
        mime = OpenAICompatProvider.STREAM_AUDIO_MIME
        assert mime.startswith("audio/pcm")
        assert "rate=24000" in mime
        assert "channels=1" in mime


class TestMediaDelta:
    def test_defaults(self):
        delta = MediaDelta(mime_type="audio/pcm", data=b"\x00")
        assert delta.sequence == 0
        assert delta.final is False
        assert delta.transcript == ""


# ==================== Capability declaration ====================


class TestOutputModality:
    """The name the tier startup check probes by ``getattr``."""

    def test_text_only_floor(self):
        class _Provider(ModalityCapabilityMixin):
            pass

        provider = _Provider()
        assert provider.output_modalities() == {"text"}
        assert provider.supports_output_modality("text") is True
        assert provider.supports_output_modality("audio") is False

    def test_probe_name_matches_the_startup_check(self):
        """``model_tiers`` looks this up by name; renaming it silently
        disables outbound verification."""
        assert hasattr(ModalityCapabilityMixin, "supports_output_modality")

    def test_accepts_the_model_keyword_the_check_passes(self):
        class _Provider(ModalityCapabilityMixin):
            pass

        assert _Provider().supports_output_modality("audio", model="gpt-audio") is False

    def test_override_is_honoured(self):
        class _Speaker(ModalityCapabilityMixin):
            def output_modalities(self, model=None):
                return {"text", "audio"}

        assert _Speaker().supports_output_modality("audio") is True

    def test_capability_column_defaults_off(self):
        assert ProviderCapabilities().output_media is False
        assert "output_media" in CAPABILITY_FIELDS


class TestForwardedApiParams:
    """OpenAI's ``modalities``/``audio`` were dropped by the allowlist."""

    def test_output_selectors_are_forwarded(self):
        assert "modalities" in OpenAICompatProvider._FORWARDED_API_PARAMS
        assert "audio" in OpenAICompatProvider._FORWARDED_API_PARAMS


# ==================== Client renderability ====================


class TestRenderableMedia:
    """The CLIENT capability axis, kept apart from the MODEL axis."""

    def test_default_client_renders_no_media(self):
        assert PresentationContext().renderable_media == []
        assert PresentationContext().can_render_media("audio/wav") is False

    def test_exact_match(self):
        context = PresentationContext(renderable_media=["image/png"])
        assert context.can_render_media("image/png") is True
        assert context.can_render_media("image/jpeg") is False

    def test_wildcard_match(self):
        context = PresentationContext(renderable_media=["audio/*"])
        assert context.can_render_media("audio/wav") is True

    def test_parameters_are_ignored_when_matching(self):
        """Parameters say how to play it, not whether it can be played."""
        context = PresentationContext(renderable_media=["audio/pcm"])
        assert context.can_render_media("audio/pcm;rate=24000;channels=1") is True

    def test_absent_mime_is_not_renderable(self):
        context = PresentationContext(renderable_media=["audio/*"])
        assert context.can_render_media(None) is False
        assert context.can_render_media("") is False


# ==================== The gate as a router, not a shredder ====================


class _RecordingHooks:
    """Captures ``on_tool_output`` calls a client would have received."""

    def __init__(self):
        self.calls = []

    def on_tool_output(self, **kwargs):
        self.calls.append(kwargs)


class _TextOnlyProvider:
    name = "openrouter"

    def supports_modality(self, kind, model=None):
        return kind == "text"


def _gating_session(hooks):
    from shared.jaato_session import JaatoSession

    session = JaatoSession.__new__(JaatoSession)
    session._provider = _TextOnlyProvider()
    session._model_name = "openai/gpt-5-mini"
    session._tier_config = None
    session._agent_id = "main"
    session._ui_hooks = hooks
    session._trace = lambda *a, **k: None
    return session


class TestGateReRoutesWithheldContent:
    """Content the model can't consume is what a client most wants (§5.2)."""

    def _result(self):
        from jaato_sdk.plugins.model_provider.types import Attachment, ToolResult

        return ToolResult(
            call_id="c1",
            name="screenshot",
            result="see attached",
            attachments=[
                Attachment(mime_type="image/png", data=b"\x89PNG", display_name="shot")
            ],
        )

    def test_withheld_attachment_reaches_the_client(self):
        hooks = _RecordingHooks()
        session = _gating_session(hooks)

        gated = session._gate_one_tool_result(self._result(), session._provider)

        # Still stripped from the model's copy...
        assert not gated.attachments
        # ...but no longer destroyed.
        assert len(hooks.calls) == 1
        call = hooks.calls[0]
        assert call["mime_type"] == "image/png"
        assert base64.b64decode(call["data_b64"]) == b"\x89PNG"
        assert call["call_id"] == "c1"
        assert call["final"] is True
        assert call["sequence"] == 0

    def test_supported_attachment_is_not_rerouted(self):
        """A model that can see the image needs no client copy."""

        class _VisionProvider:
            name = "openrouter"

            def supports_modality(self, kind, model=None):
                return kind in {"text", "image"}

        hooks = _RecordingHooks()
        session = _gating_session(hooks)
        session._provider = _VisionProvider()

        gated = session._gate_one_tool_result(self._result(), session._provider)

        assert gated.attachments
        assert hooks.calls == []

    def test_missing_hooks_is_not_an_error(self):
        """The gate runs on bare sessions with no UI-hooks attribute."""
        session = _gating_session(_RecordingHooks())
        del session._ui_hooks

        gated = session._gate_one_tool_result(self._result(), session._provider)

        assert not gated.attachments  # still gated, just not delivered

    def test_delivery_failure_does_not_fail_the_tool_result(self):
        class _ExplodingHooks:
            def on_tool_output(self, **kwargs):
                raise RuntimeError("client went away")

        session = _gating_session(_ExplodingHooks())

        gated = session._gate_one_tool_result(self._result(), session._provider)

        assert not gated.attachments
        assert gated.model_suffix  # the withheld note still reaches the model


# ==================== Runner -> daemon dispatch ====================


class TestDispatchToolOutput:
    """Media keys are forwarded only when bytes are actually present."""

    def _hooks(self):
        return _RecordingHooks()

    def test_text_payload_calls_the_original_arity(self):
        """A hooks implementation predating media must keep working."""
        from server.core import _dispatch_tool_output

        hooks = self._hooks()
        _dispatch_tool_output(hooks, {"call_id": "c1", "chunk": "hi"}, "main")

        assert hooks.calls == [
            {"agent_id": "main", "call_id": "c1", "chunk": "hi"}
        ]

    def test_media_payload_forwards_every_field(self):
        from server.core import _dispatch_tool_output

        hooks = self._hooks()
        _dispatch_tool_output(
            hooks,
            {
                "agent_id": "sub", "call_id": "c2", "chunk": "",
                "stream_id": "s1", "sequence": 7,
                "mime_type": "audio/pcm", "data_b64": "AAE=", "final": True,
            },
            "main",
        )

        call = hooks.calls[0]
        assert call["agent_id"] == "sub"
        assert call["mime_type"] == "audio/pcm"
        assert call["sequence"] == 7
        assert call["final"] is True

    def test_half_a_payload_is_treated_as_text(self):
        """A mime type with no data is not a media chunk."""
        from server.core import _dispatch_tool_output

        hooks = self._hooks()
        _dispatch_tool_output(
            hooks, {"call_id": "c", "chunk": "x", "mime_type": "audio/pcm"}, "main"
        )

        assert "mime_type" not in hooks.calls[0]

    def test_falls_back_to_the_default_agent_id(self):
        from server.core import _dispatch_tool_output

        hooks = self._hooks()
        _dispatch_tool_output(hooks, {"call_id": "c", "chunk": "x"}, "fallback")

        assert hooks.calls[0]["agent_id"] == "fallback"


# ==================== Model-generated media delivery ====================


class TestDeliverModelMedia:
    """Model audio rides the tool-output channel under a reserved call_id."""

    def _session(self, hooks):
        from shared.jaato_session import JaatoSession

        session = JaatoSession.__new__(JaatoSession)
        session._agent_id = "main"
        session._ui_hooks = hooks
        session._trace = lambda *a, **k: None
        return session

    def test_delivers_under_the_reserved_call_id(self):
        from shared.jaato_session import MODEL_MEDIA_CALL_ID

        hooks = _RecordingHooks()
        session = self._session(hooks)

        session._deliver_model_media(
            MediaDelta(
                mime_type="audio/pcm", data=b"\x01\x02",
                sequence=4, final=True, transcript="hey",
            )
        )

        call = hooks.calls[0]
        assert call["call_id"] == MODEL_MEDIA_CALL_ID
        assert base64.b64decode(call["data_b64"]) == b"\x01\x02"
        assert call["sequence"] == 4
        assert call["final"] is True
        assert call["chunk"] == "hey"
        # Per UTTERANCE, not per agent: a constant id collided every
        # utterance of a session into one stream, so a retried turn's
        # audio was spliced onto the first attempt's.
        assert call["stream_id"] == "model:main:1"

    def test_each_utterance_gets_its_own_stream_id(self):
        """The provider restarts `sequence` at 0 per turn, and that
        reset is the utterance boundary."""
        hooks = _RecordingHooks()
        session = self._session(hooks)
        for seq in (0, 1):        # first utterance
            session._deliver_model_media(
                MediaDelta(mime_type="audio/pcm", data=b"\x01", sequence=seq))
        for seq in (0, 1):        # second utterance, sequence restarts
            session._deliver_model_media(
                MediaDelta(mime_type="audio/pcm", data=b"\x02", sequence=seq))

        ids = [c["stream_id"] for c in hooks.calls]
        assert ids == ["model:main:1", "model:main:1",
                       "model:main:2", "model:main:2"]

    def test_empty_payload_is_not_delivered(self):
        hooks = _RecordingHooks()
        self._session(hooks)._deliver_model_media(
            MediaDelta(mime_type="audio/pcm", data=b"")
        )
        assert hooks.calls == []

    def test_delivery_failure_does_not_abort_generation(self):
        class _Exploding:
            def on_tool_output(self, **kwargs):
                raise RuntimeError("client gone")

        session = self._session(_Exploding())
        session._deliver_model_media(MediaDelta(mime_type="audio/pcm", data=b"\x01"))

    def test_missing_hooks_is_not_an_error(self):
        session = self._session(_RecordingHooks())
        del session._ui_hooks
        session._deliver_model_media(MediaDelta(mime_type="audio/pcm", data=b"\x01"))


# ==================== Audience never leaks into history ====================


class TestModelFacingText:
    """A CLIENT chunk must not appear in anything the model reads back."""

    def test_client_chunks_are_excluded(self):
        from shared.plugins.streaming.manager import _model_facing_text

        chunks = [
            StreamChunk("visible"),
            StreamChunk("secret", audience=Audience.CLIENT),
            StreamChunk("both", audience=Audience.BOTH),
        ]

        assert _model_facing_text(chunks) == "visible\nboth"

    def test_empty_content_is_skipped(self):
        from shared.plugins.streaming.manager import _model_facing_text

        assert _model_facing_text([StreamChunk(""), StreamChunk("x")]) == "x"

    def test_media_only_chunk_contributes_nothing(self):
        from shared.plugins.streaming.manager import _model_facing_text

        media = StreamChunk(
            inline_data={"mime_type": "audio/wav", "data": b"\x00"},
            audience=Audience.CLIENT,
        )
        assert _model_facing_text([media]) == ""


class TestTheAudienceBoundaryHoldsAtEverySite:
    """A CLIENT chunk must reach no surface the model reads back.

    Four places aggregate stream chunks for the model, and until this
    class existed exactly ONE of them was guarded.  Removing the filter
    from each in turn and running every media test left three of the four
    green -- including the history append, which is where a CLIENT chunk
    would become something the model was actually told.

    The commit that introduced the boundary asserted the property as
    established.  These are the assertions that make that true, so each
    test names the site it pins:

      * ``manager.py`` ``_model_facing_text``     -> the tool RESULT
      * ``_format_streaming_updates``             -> ``<streaming_updates>``
      * ``_execute_streaming_tool``'s ``on_chunk`` -> the ``on_output`` route
      * ``_execute_streaming_tool``'s initial loop -> ``initial_results``
    """

    def _updates(self):
        from shared.plugins.streaming.manager import StreamUpdate
        return [StreamUpdate(
            stream_id="s1", tool_name="speak", is_complete=False,
            new_chunks=[StreamChunk("model sees this"),
                        StreamChunk("VIEWERS ONLY", audience=Audience.CLIENT)],
            total_chunks=2, final_result=None)]

    def test_streaming_updates_block_excludes_client_chunks(self):
        from shared.jaato_session import JaatoSession
        session = JaatoSession.__new__(JaatoSession)
        rendered = JaatoSession._format_streaming_updates(session, self._updates())
        assert "model sees this" in rendered
        assert "VIEWERS ONLY" not in rendered, (
            "a CLIENT chunk reached the <streaming_updates> block, which is "
            "wrapped in <hidden> and handed to the model")

    def _drive_stream(self, chunks):
        """Run ``_execute_streaming_tool`` over ``chunks``; return what escaped.

        Doubles the collaborators rather than the method: the point is to
        exercise the REAL routing code, which is where the filters live.
        """
        from unittest.mock import MagicMock
        from shared.jaato_session import JaatoSession
        from jaato_sdk.plugins.model_provider.types import FunctionCall

        session = JaatoSession.__new__(JaatoSession)
        session._agent_id = "agent-1"
        session._ui_hooks = MagicMock()
        session._runtime = MagicMock()
        session._runtime.registry.get_base_tool_name.return_value = "speak"
        session._runtime.registry.get_streaming_plugin.return_value = MagicMock()

        handle = MagicMock(stream_id="s1", initial_chunks=chunks)
        handle.status.value = "running"

        def _start_stream(**kwargs):
            for chunk in chunks:                 # replay through the live callback
                kwargs["on_ui_chunk"](chunk)
            return handle

        session._stream_manager = MagicMock()
        session._stream_manager.start_stream.side_effect = _start_stream

        to_model = []
        ok, result = JaatoSession._execute_streaming_tool(
            session, FunctionCall(id="call_1", name="speak-stream", args={}),
            lambda source, text, mode: to_model.append(text))
        assert ok, result
        return to_model, result, session._ui_hooks

    def test_client_chunks_never_reach_on_output(self):
        """``on_output`` is the model's text channel and becomes history."""
        to_model, _, _ = self._drive_stream([
            StreamChunk("model sees this"),
            StreamChunk("VIEWERS ONLY", audience=Audience.CLIENT),
        ])
        joined = "".join(to_model)
        assert "model sees this" in joined
        assert "VIEWERS ONLY" not in joined

    def test_client_chunks_never_reach_initial_results(self):
        """``initial_results`` is returned in the tool result -> history."""
        _, result, _ = self._drive_stream([
            StreamChunk("model sees this"),
            StreamChunk("VIEWERS ONLY", audience=Audience.CLIENT),
        ])
        assert result["initial_results"] == ["model sees this"]

    def test_client_chunks_DO_reach_the_client(self):
        """The other half of the contract: withholding from the model is
        not discarding.  A boundary that dropped the chunk entirely would
        pass every test above and still be wrong."""
        _, _, hooks = self._drive_stream([
            StreamChunk("VIEWERS ONLY", audience=Audience.CLIENT),
        ])
        assert hooks.on_tool_output.called
        assert hooks.on_tool_output.call_args.kwargs["chunk"] == "VIEWERS ONLY"

    def test_both_audience_reaches_model_and_client(self):
        to_model, result, hooks = self._drive_stream([
            StreamChunk("shared", audience=Audience.BOTH),
        ])
        assert "shared" in "".join(to_model)
        assert result["initial_results"] == ["shared"]
        assert hooks.on_tool_output.call_args.kwargs["chunk"] == "shared"
