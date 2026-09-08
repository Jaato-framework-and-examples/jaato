"""The provider media-output contract, and the tier role that drives it.

Two things are asserted here that the delivery tests cannot see:

1. The contract is UNIVERSAL and the wire format is NOT.  A provider opts
   into model media by implementing hooks, not by inheriting one vendor's
   base class -- the defect this closes is that ten providers speak
   OpenAI's wire format while only five inherit ``OpenAICompatProvider``,
   so audio was unreachable through OpenRouter.
2. A tier's ``outbound`` role reaches the request body.  Without that
   wiring the role is a declaration nothing acts on.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.model_tiers import TierEntry
from shared.plugins.model_provider import _media_deltas
from shared.plugins.model_provider._media_deltas import (
    DEFAULT_AUDIO_OPTIONS,
    MEDIA_API_PARAMS,
    STREAM_AUDIO_MIME,
    OpenAIMediaOutputMixin,
    apply_output_modalities,
    emit_audio_delta,
    ensure_spoken_part,
    extract_audio_delta,
)
from shared.plugins.model_provider.base import ModalityCapabilityMixin
from jaato_sdk.plugins.model_provider.types import MediaDelta, Part


def _audio_delta(payload: bytes, transcript: str = ""):
    return {"audio": {"data": base64.b64encode(payload).decode(),
                      "transcript": transcript}}


# ==================== The universal contract ====================


class _BareProvider(ModalityCapabilityMixin):
    """A provider that implements nothing — the honest default."""


class TestContractDefaults:
    """"Leave it unimplemented" must be a working choice, not a crash."""

    def test_emit_media_delta_is_a_no_op(self):
        """A streaming loop calls this unconditionally; it must be cheap
        and must not consume a sequence number."""
        assert _BareProvider().emit_media_delta(object(), None, 7) == 7

    def test_request_output_modalities_is_a_no_op(self):
        assert _BareProvider().request_output_modalities({"audio"}) is None

    def test_output_floor_is_text_only(self):
        provider = _BareProvider()
        assert provider.output_modalities() == {"text"}
        assert provider.supports_output_modality("audio") is False

    def test_knob_raises_the_floor_and_always_keeps_text(self):
        """A model that speaks still writes."""
        provider = _BareProvider()
        provider._output_modalities_knob = ["audio"]
        assert provider.output_modalities() == {"text", "audio"}
        assert provider.supports_output_modality("audio") is True

    def test_knob_is_normalised(self):
        provider = _BareProvider()
        provider._output_modalities_knob = ["  AUDIO  "]
        assert provider.supports_output_modality("audio") is True


class TestContractIsUniversal:
    """Every provider answers the contract, whatever it inherits."""

    @pytest.mark.parametrize("module_name", [
        "anthropic", "google_genai", "openrouter", "chrome_ai",
        "claude_cli", "antigravity", "github_models",
    ])
    def test_provider_answers_output_capability(self, module_name):
        import importlib

        module = importlib.import_module(
            f"shared.plugins.model_provider.{module_name}.provider"
        )
        classes = [
            obj for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, ModalityCapabilityMixin)
            and obj is not ModalityCapabilityMixin
        ]
        assert classes, f"{module_name} exposes no provider class"
        for cls in classes:
            assert hasattr(cls, "output_modalities")
            assert hasattr(cls, "supports_output_modality")
            assert hasattr(cls, "emit_media_delta")
            assert hasattr(cls, "request_output_modalities")


# ==================== The OpenAI wire format ====================


class TestSharedDecoderIsNotInheritanceBound:
    """The regression that motivated the refactor."""

    def test_openrouter_decodes_audio_without_the_compat_base(self):
        from shared.plugins.model_provider._openai_compat.base import (
            OpenAICompatProvider,
        )
        from shared.plugins.model_provider.openrouter.provider import (
            OpenRouterProvider,
        )

        # The point: it does NOT inherit the compat base ...
        assert not issubclass(OpenRouterProvider, OpenAICompatProvider)
        # ... yet it decodes the same wire format.
        provider = OpenRouterProvider.__new__(OpenRouterProvider)
        emitted = []
        sequence = provider.emit_media_delta(
            _audio_delta(b"\x01\x02"), emitted.append, -1
        )
        assert sequence == 0
        assert emitted[0].data == b"\x01\x02"

    def test_compat_base_still_decodes_after_the_move(self):
        from shared.plugins.model_provider._openai_compat.base import (
            OpenAICompatProvider,
        )

        provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
        emitted = []
        provider.emit_media_delta(_audio_delta(b"\x03"), emitted.append, -1)
        assert emitted[0].data == b"\x03"

    def test_media_params_are_forwarded_by_the_compat_allowlist(self):
        from shared.plugins.model_provider._openai_compat.base import (
            OpenAICompatProvider,
        )

        assert MEDIA_API_PARAMS <= OpenAICompatProvider._FORWARDED_API_PARAMS


class TestExtractAudioDelta:
    """``delta.audio`` is undocumented upstream; read it defensively."""

    def test_dict_shape(self):
        assert extract_audio_delta(_audio_delta(b"\x01", "hi")) == (b"\x01", "hi")

    def test_attribute_shape(self):
        class _Audio:
            data = base64.b64encode(b"\x02").decode()
            transcript = "yo"

        class _Delta:
            audio = _Audio()

        assert extract_audio_delta(_Delta()) == (b"\x02", "yo")

    def test_model_extra_shape(self):
        class _Delta:
            model_extra = {"audio": {"data": base64.b64encode(b"\x03").decode()}}

        assert extract_audio_delta(_Delta()) == (b"\x03", "")

    def test_text_delta_is_none(self):
        class _Delta:
            content = "hello"

        assert extract_audio_delta(_Delta()) is None

    def test_undecodable_is_discarded_not_raised(self):
        assert extract_audio_delta({"audio": {"data": "!!!"}}) is None

    def test_empty_is_none(self):
        assert extract_audio_delta({"audio": {"data": ""}}) is None

    def test_stream_mime_names_the_pcm_parameters(self):
        """Headerless PCM carries no way to recover rate/channels."""
        assert "rate=24000" in STREAM_AUDIO_MIME
        assert "channels=1" in STREAM_AUDIO_MIME


class TestEmitAudioDelta:
    def test_sequence_only_advances_on_a_real_chunk(self):
        emitted = []
        seq = emit_audio_delta({"content": "x"}, emitted.append, 4)
        assert seq == 4 and emitted == []
        seq = emit_audio_delta(_audio_delta(b"\x01"), emitted.append, seq)
        assert seq == 5 and emitted[0].sequence == 5

    def test_mime_is_overridable(self):
        emitted = []
        emit_audio_delta(_audio_delta(b"\x01"), emitted.append, 0, "audio/wav")
        assert emitted[0].mime_type == "audio/wav"


# ==================== Tier role -> request body ====================


class TestApplyOutputModalities:
    def test_audio_request_stamps_both_fields(self):
        kwargs = {}
        apply_output_modalities(kwargs, {"audio"})
        assert kwargs["modalities"] == ["text", "audio"]
        assert kwargs["audio"] == DEFAULT_AUDIO_OPTIONS

    def test_default_format_is_pcm16_because_streaming_allows_nothing_else(self):
        assert DEFAULT_AUDIO_OPTIONS["format"] == "pcm16"

    def test_no_request_leaves_the_body_untouched(self):
        """A provider never asked for audio must send the same bytes as
        before this feature existed."""
        kwargs = {}
        apply_output_modalities(kwargs, set())
        assert kwargs == {}

    def test_profile_values_win_over_the_tier_default(self):
        """The tier says WHAT to emit; the profile says HOW."""
        kwargs = {"audio": {"voice": "cedar", "format": "pcm16"}}
        apply_output_modalities(kwargs, {"audio"})
        assert kwargs["audio"]["voice"] == "cedar"

    def test_non_audio_roles_are_ignored(self):
        kwargs = {}
        apply_output_modalities(kwargs, {"image"})
        assert kwargs == {}


class TestMixinRoundTrip:
    """Entering and leaving a speaking tier."""

    class _Provider(OpenAIMediaOutputMixin, ModalityCapabilityMixin):
        pass

    def test_entering_a_speaking_tier_requests_audio(self):
        provider = self._Provider()
        entry = TierEntry("m", outbound_modalities=frozenset({"audio"}))
        provider.request_output_modalities(entry.outbound_modalities)
        kwargs = {}
        provider.apply_requested_output_modalities(kwargs)
        assert kwargs["modalities"] == ["text", "audio"]

    def test_leaving_it_stops_requesting_audio(self):
        """The empty set is an instruction, not an absence of one."""
        provider = self._Provider()
        provider.request_output_modalities({"audio"})
        provider.request_output_modalities(TierEntry("m").outbound_modalities)
        kwargs = {}
        provider.apply_requested_output_modalities(kwargs)
        assert kwargs == {}

    def test_unrequested_provider_stamps_nothing(self):
        provider = self._Provider()
        kwargs = {}
        provider.apply_requested_output_modalities(kwargs)
        assert kwargs == {}


class TestSessionTierWiring:
    """``_request_tier_output_modalities`` is what makes the role act."""

    def _session(self, provider):
        from shared.jaato_session import JaatoSession

        session = JaatoSession.__new__(JaatoSession)
        session._provider = provider
        session._trace = lambda *a, **k: None
        return session

    class _Recorder(ModalityCapabilityMixin):
        def __init__(self):
            self.requested = None

        def request_output_modalities(self, kinds):
            self.requested = frozenset(kinds)

    def test_entered_tier_role_reaches_the_provider(self):
        provider = self._Recorder()
        session = self._session(provider)
        session._request_tier_output_modalities(
            TierEntry("m", outbound_modalities=frozenset({"audio"}))
        )
        assert provider.requested == frozenset({"audio"})

    def test_plain_tier_clears_the_request(self):
        provider = self._Recorder()
        session = self._session(provider)
        session._request_tier_output_modalities(TierEntry("m"))
        assert provider.requested == frozenset()

    def test_provider_without_the_hook_is_tolerated(self):
        class _Old:
            pass

        session = self._session(_Old())
        session._request_tier_output_modalities(TierEntry("m"))  # must not raise

    def test_a_raising_provider_never_fails_a_completed_switch(self):
        class _Exploding(ModalityCapabilityMixin):
            def request_output_modalities(self, kinds):
                raise RuntimeError("nope")

        session = self._session(_Exploding())
        session._request_tier_output_modalities(TierEntry("m"))  # must not raise

    def test_no_provider_is_tolerated(self):
        session = self._session(None)
        session._request_tier_output_modalities(TierEntry("m"))


    def test_a_session_that_starts_in_a_speaking_tier_requests_audio(self):
        """The initial tier never passes through ``_connect_tier_entry``
        (that is the SWITCH path), so it is stamped separately — without
        which an outbound role only took effect after the first
        ``enter_tier``."""
        from shared.model_tiers import ModelTierConfig

        provider = self._Recorder()
        session = self._session(provider)
        session._tier_config = ModelTierConfig(
            tiers={
                "executor": TierEntry("m1"),
                "planner": TierEntry(
                    "m2", outbound_modalities=frozenset({"audio"})),
            },
            initial_tier="planner",
            tier_fallback="executor",
        )
        session._active_tier = "planner"

        session._request_active_tier_output_modalities()

        assert provider.requested == frozenset({"audio"})

    def test_single_model_session_stamps_nothing(self):
        provider = self._Recorder()
        session = self._session(provider)
        session._tier_config = None
        session._active_tier = None

        session._request_active_tier_output_modalities()

        assert provider.requested is None


# ==================== Audio-only streams terminate ====================


def _termination_call(module_path: str) -> str:
    """Return the ``stream_terminated(...)`` call as one whitespace-normalised line.

    These assertions pin SOURCE TEXT because the regressions they guard
    are argument changes that would still typecheck -- passing `parts`
    back in, or dropping the usage signal.  Normalising whitespace keeps
    them honest about arguments while indifferent to line wrapping,
    which has broken them twice for no reason.
    """
    import importlib, inspect, re
    src = inspect.getsource(importlib.import_module(module_path))
    match = re.search(r"stream_terminated\((?:[^()]|\([^()]*\))*\)", src)
    assert match, f"{module_path} does not call stream_terminated at all"
    return re.sub(r"\s+", " ", match.group(0))


class TestAudioOnlyStreamTermination:
    """An audio-only stream carries no ``finish_reason`` (verified live).

    ``require_terminated_stream`` (#687) rejects a stream that stopped
    without saying why — correct for TEXT, where a missing finish_reason
    means the upstream died mid-generation.  Verified against
    ``openai/gpt-audio-mini`` via OpenRouter: the whole stream sends no
    ``finish_reason``, no ``native_finish_reason``, an empty-string
    ``content``, and a closing ``usage`` block.  Without this, every
    complete audio response was rejected as a fragment AND marked
    retryable, so the turn failed and retried forever.
    """

    def test_no_media_sentinel_is_not_arrival(self):
        assert _media_deltas.media_arrived(_media_deltas.NO_MEDIA_YET) is False

    def test_first_chunk_counts_as_arrival(self):
        """Sequence 0 is a real chunk — an off-by-one here would reject
        a single-chunk audio answer."""
        assert _media_deltas.media_arrived(0) is True
        assert _media_deltas.media_arrived(7) is True

    def test_sentinel_is_below_the_first_sequence(self):
        assert _media_deltas.NO_MEDIA_YET < 0

    def test_chunk_count_derives_from_the_sequence(self):
        """No separate tally to drift out of step with the counter."""
        assert _media_deltas.media_chunk_count(_media_deltas.NO_MEDIA_YET) == 0
        assert _media_deltas.media_chunk_count(0) == 1
        assert _media_deltas.media_chunk_count(4) == 5

    def test_stream_terminated_accepts_either_signal(self):
        no_media = _media_deltas.NO_MEDIA_YET
        assert _media_deltas.stream_terminated(True, no_media) is True   # text
        assert _media_deltas.stream_terminated(False, 0) is True         # audio
        assert _media_deltas.stream_terminated(False, no_media) is False # neither

    def test_emit_advances_past_the_sentinel(self):
        """The counter both providers feed to ``media_arrived``."""
        seq = _media_deltas.NO_MEDIA_YET
        emitted = []
        seq = emit_audio_delta({"content": "x"}, emitted.append, seq)
        assert _media_deltas.media_arrived(seq) is False, "text must not terminate"
        seq = emit_audio_delta(_audio_delta(b"\x01"), emitted.append, seq)
        assert _media_deltas.media_arrived(seq) is True

    @pytest.mark.parametrize("module_path", [
        "shared.plugins.model_provider.openrouter.provider",
        "shared.plugins.model_provider._openai_compat.base",
    ])
    def test_both_streaming_loops_use_the_signal(self, module_path):
        """openrouter owns its loop and _openai_compat owns the other, so
        the fix has to be in both or five providers keep the bug."""
        import importlib, inspect

        src = inspect.getsource(importlib.import_module(module_path))
        assert "terminal_seen, media_sequence" in _termination_call(module_path), \
            "termination signal missing"
        assert "NO_MEDIA_YET" in src, "sentinel not used"
        # audio counts as content — an audio-only turn reported
        # "no content arrived" while 31KB of speech had been decoded.
        assert "media_chunk_count(media_sequence)" in src, \
            "media not counted into the chunk total"


class TestMediaTerminatesStreamButToolCallsDoNot:
    """Decoded media is completion evidence; accumulated tool calls are not.

    An audio-modality request to OpenRouter carries NO finish reason --
    confirmed on the SSE wire and in OpenRouter's own billing export,
    where `finish_reason_raw` is empty for every such generation.  So a
    spoken turn needs a second signal, and decoded media is a sound one:
    a stream that died mid-generation cannot retroactively have
    delivered bytes that were already played.

    A completed tool call is NOT a sound one, and an earlier version of
    this branch wrongly accepted it.  Accumulated calls are precisely
    what a connection severed mid-``arguments`` leaves behind too -- the
    same bytes on the wire -- so treating them as completion is how a
    severed turn becomes an executed one.  That is what #687 exists to
    prevent, and it is a different problem from the missing finish
    reason on audio turns.
    """

    def test_a_usage_frame_terminates(self):
        """The only completion signal this upstream gives on some turns.

        OpenRouter's generation record for a turn that raised here:
        `finish_reason: null`, `native_finish_reason: null`,
        `cancelled: false`, `status: 200`, `tokens_completion: 27` -- a
        completed, billed generation naming no reason.  The request sets
        `stream_options.include_usage`, and that frame is the LAST of a
        finished stream, so its arrival is evidence of the end.
        """
        assert _media_deltas.stream_terminated(
            False, _media_deltas.NO_MEDIA_YET, usage_reported=True) is True

    def test_a_cut_stream_reports_no_usage_and_still_raises(self):
        """Fails closed: no finish reason, no media, no usage frame."""
        assert _media_deltas.stream_terminated(
            False, _media_deltas.NO_MEDIA_YET, usage_reported=False) is False

    def test_media_terminates(self):
        assert _media_deltas.stream_terminated(False, 0) is True

    def test_a_named_finish_reason_terminates(self):
        assert _media_deltas.stream_terminated(
            True, _media_deltas.NO_MEDIA_YET) is True

    def test_no_media_and_no_finish_reason_does_not(self):
        """The #687 guard, intact: nothing here says the turn ENDED."""
        assert _media_deltas.stream_terminated(
            False, _media_deltas.NO_MEDIA_YET) is False

    @pytest.mark.parametrize("module_path", [
        "shared.plugins.model_provider.openrouter.provider",
        "shared.plugins.model_provider._openai_compat.base",
    ])
    def test_neither_loop_passes_parts(self, module_path):
        """Both streaming loops must ask the narrow question.

        Pinned as source text because the regression this guards is a
        widening of the call itself -- passing `parts` back in would
        restore the behaviour #687 forbids, and would still typecheck.
        """
        call = _termination_call(module_path)
        assert "parts" not in call, (
            "a completed tool call is not evidence the stream ended -- "
            "that is #687's whole point")
        assert "usage_reported=usage.total_tokens > 0" in call, (
            "the usage frame is the only completion signal this upstream "
            "gives on an audio-modality request")


class TestTranscriptArrivesSeparately:
    """Bytes and transcript ride in DIFFERENT deltas.

    Measured against openai/gpt-audio-mini via OpenRouter, one turn:
    7 deltas carried data only, 11 carried transcript only, ZERO carried
    both.  An extractor that required `data` therefore discarded every
    word the model said — so a spoken turn produced no text, no Part and
    no history entry, and the framework nudged the model to finish work
    it had already done, costing a second generation every run.
    """

    def _b64(self, raw=b"\x01\x02"):
        return base64.b64encode(raw).decode()

    def test_transcript_only_delta_is_not_discarded(self):
        assert extract_audio_delta({"audio": {"transcript": "On a clear "}}) \
            == (None, "On a clear ")

    def test_data_only_delta_has_empty_transcript(self):
        assert extract_audio_delta({"audio": {"data": self._b64()}}) \
            == (b"\x01\x02", "")

    def test_audio_object_with_neither_is_nothing(self):
        assert extract_audio_delta({"audio": {}}) is None

    def test_transcript_only_does_not_consume_a_sequence(self):
        """Counting it would put a hole in the sequence a client uses to
        detect media dropped under backpressure."""
        emitted, said = [], []
        seq = _media_deltas.NO_MEDIA_YET
        seq = emit_audio_delta({"audio": {"transcript": "hi "}},
                               emitted.append, seq, transcript_sink=said)
        assert seq == _media_deltas.NO_MEDIA_YET
        assert emitted == []
        assert said == ["hi "]

    def test_transcript_only_emits_no_empty_chunk(self):
        """An empty MediaDelta would hand a client zero bytes to play."""
        emitted = []
        emit_audio_delta({"audio": {"transcript": "x"}}, emitted.append,
                         _media_deltas.NO_MEDIA_YET, transcript_sink=[])
        assert emitted == []

    def test_interleaved_stream_yields_words_and_contiguous_sequences(self):
        """The real wire shape: transcript and data alternate."""
        emitted, said = [], []
        seq = _media_deltas.NO_MEDIA_YET
        for delta in ({"audio": {"transcript": "On a clear day, "}},
                      {"audio": {"data": self._b64()}},
                      {"audio": {"transcript": "the sky is blue."}},
                      {"audio": {"data": self._b64()}}):
            seq = emit_audio_delta(delta, emitted.append, seq,
                                   transcript_sink=said)
        assert [m.sequence for m in emitted] == [0, 1]
        assert "".join(said) == "On a clear day, the sky is blue."

    def test_the_words_become_the_spoken_part(self):
        """End of the chain: what makes a spoken turn visible in history."""
        parts = []
        ensure_spoken_part(parts, "On a clear day, the sky is blue.")
        assert parts[0].text == "On a clear day, the sky is blue."

    def test_sink_is_optional(self):
        """A provider that does not want the words passes none."""
        emitted = []
        seq = emit_audio_delta({"audio": {"data": self._b64()}},
                               emitted.append, _media_deltas.NO_MEDIA_YET)
        assert seq == 0 and len(emitted) == 1


class TestFinalMarksTheLastChunk:
    """`final` is a FACT the upstream sends, not an inference.

    OpenAI closes an audio stream with a `delta.audio` carrying neither
    bytes nor transcript -- `{"expires_at": ...}` on the wire, a value it
    can only know once the audio object is complete.  Observed for
    openai/gpt-audio-mini through OpenRouter, frame 16 of 20:

        15  audio keys=['data']        6400B   <- last audio bytes
        16  audio keys=['expires_at']     0B   <- the marker
        19  [DONE]

    The decoder used to drop that frame -- `extract_audio_delta` returns
    None when a delta has neither bytes nor transcript -- so `final` was
    never set on model speech and every client had to GUESS where an
    utterance ended.  Each guess tried in this tree was wrong somewhere:
    "the turn ended" closed a player mid-utterance and started a second
    one over the first, heard as two answers at once.
    """

    def _frames(self):
        return [
            {"audio": {"transcript": "Sure"}},
            {"audio": {"data": base64.b64encode(b"one").decode()}},
            {"audio": {"data": base64.b64encode(b"two").decode()}},
            {"audio": {"data": base64.b64encode(b"three").decode()}},
            {"audio": {"expires_at": 1788}},
        ]

    def _run(self, frames, flush=True):
        got, seq, pending, sink = [], 0, [], []
        for f in frames:
            seq = _media_deltas.emit_audio_delta(
                f, got.append, seq, transcript_sink=sink, pending=pending)
        if flush:
            _media_deltas.flush_audio_stream(got.append, pending)
        return got, sink

    def test_the_marker_is_recognised(self):
        assert _media_deltas.is_end_of_audio({"audio": {"expires_at": 1}}) is True
        assert _media_deltas.is_end_of_audio({"audio": {"data": "AA=="}}) is False
        assert _media_deltas.is_end_of_audio({"audio": {"transcript": "x"}}) is False
        assert _media_deltas.is_end_of_audio({"content": "hi"}) is False

    def test_final_lands_on_the_last_chunk_only(self):
        got, _ = self._run(self._frames())
        assert [(c.data, c.final) for c in got] == [
            (b"one", False), (b"two", False), (b"three", True)]

    def test_sequence_is_unbroken(self):
        """A held chunk must not renumber: a client uses `sequence` to
        detect dropped media, so a gap would read as backpressure loss."""
        got, _ = self._run(self._frames())
        assert [c.sequence for c in got] == [1, 2, 3]

    def test_the_transcript_still_lands(self):
        _, sink = self._run(self._frames())
        assert sink == ["Sure"]

    def test_a_stream_with_no_marker_still_terminates(self):
        """Some upstream may send none; the stream ending is conclusive."""
        got, _ = self._run(self._frames()[:-1])
        assert [c.final for c in got] == [False, False, True]

    def test_without_flush_the_last_chunk_is_held(self):
        """Proves the flush is load-bearing rather than decorative: drop
        it and the final chunk is never delivered at all."""
        got, _ = self._run(self._frames()[:-1], flush=False)
        assert [c.data for c in got] == [b"one", b"two"]

    def test_opting_out_is_unchanged_behaviour(self):
        """A caller that passes no buffer gets exactly the old shape."""
        got, seq = [], 0
        for f in self._frames():
            seq = _media_deltas.emit_audio_delta(f, got.append, seq)
        assert [(c.data, c.final) for c in got] == [
            (b"one", False), (b"two", False), (b"three", False)]

    @pytest.mark.parametrize("module_path", [
        "shared.plugins.model_provider.openrouter.provider",
        "shared.plugins.model_provider._openai_compat.base",
    ])
    def test_both_loops_hold_and_flush(self, module_path):
        """Both streaming loops, or half the fleet never marks `final`."""
        import importlib, inspect
        src = inspect.getsource(importlib.import_module(module_path))
        assert "media_pending" in src, "no one-slot buffer in this loop"
        assert "flush_media_stream" in src, "a held chunk would never be delivered"


class TestTheWordsRideTheFinalChunk:
    """The transcript reaches the wire on the chunk marked `final` (#869).

    Every spoken turn produced a transcript inside the provider and no
    client could obtain it: `emit_audio_delta` put the words in a
    caller-owned sink, and the sink became a history Part only after
    the stream closed -- past every point that emits to a client.  A
    voice client watching a call saw 5.45 s of audio in 14 chunks and
    an `ask()` that returned `''`; a call log could record how long the
    agent spoke but not what it said.

    The `pending` buffer already holds the last chunk back so it can be
    marked `final`, and the transcript is complete at that moment, so
    the words ride that chunk.  Under the SAME rule `ensure_spoken_part`
    applies to history: a turn that wrote its own text already sent the
    words as `AGENT_OUTPUT`, and a client must never get them twice.
    """

    def _frames(self):
        return [
            {"audio": {"transcript": "On a clear day, "}},
            {"audio": {"data": base64.b64encode(b"one").decode()}},
            {"audio": {"transcript": "the sky is blue."}},
            {"audio": {"data": base64.b64encode(b"two").decode()}},
            {"audio": {"expires_at": 1788}},
        ]

    def _run(self, frames, wrote_text=None, flush=True):
        got, seq, pending, sink = [], 0, [], []
        for f in frames:
            seq = _media_deltas.emit_audio_delta(
                f, got.append, seq, transcript_sink=sink, pending=pending,
                wrote_text=wrote_text)
        if flush:
            _media_deltas.flush_audio_stream(
                got.append, pending, sink, wrote_text)
        return got, sink

    def test_the_final_chunk_carries_the_whole_transcript(self):
        got, _ = self._run(self._frames())
        assert [(c.final, c.transcript) for c in got] == [
            (False, ""), (True, "On a clear day, the sky is blue.")]

    def test_intermediate_chunks_stay_wordless(self):
        """Exactly once: a client logging `chunk` off every event must
        not see a growing prefix on each one."""
        got, _ = self._run(self._frames())
        assert all(c.transcript == "" for c in got if not c.final)

    def test_a_stream_with_no_marker_delivers_the_words_on_flush(self):
        """A provider that sends no end-of-audio marker must not also be
        one whose clients never learn what was said."""
        got, _ = self._run(self._frames()[:-1])
        assert got[-1].final is True
        assert got[-1].transcript == "On a clear day, the sky is blue."

    def test_a_turn_that_wrote_its_own_text_sends_no_transcript(self):
        """The `ensure_spoken_part` rule, carried to the wire: the words
        went out as AGENT_OUTPUT already."""
        got, _ = self._run(self._frames(), wrote_text=lambda: True)
        assert [c.final for c in got] == [False, True]
        assert got[-1].transcript == ""

    def test_wrote_text_is_read_at_the_marker_not_at_the_start(self):
        """Text can arrive after the first audio chunk; the answer is a
        live question, so the loop's accumulator is consulted THEN."""
        written = []
        frames = self._frames()
        got, seq, pending, sink = [], 0, [], []
        for i, f in enumerate(frames):
            if i == 3:              # the model writes, mid-utterance
                written.append("The sky is blue.")
            seq = _media_deltas.emit_audio_delta(
                f, got.append, seq, transcript_sink=sink, pending=pending,
                wrote_text=lambda: bool(written))
        assert got[-1].final is True and got[-1].transcript == ""

    def test_without_the_predicate_the_words_always_go_out(self):
        """A caller that has no text accumulator gets the transcript."""
        got, _ = self._run(self._frames(), wrote_text=None)
        assert got[-1].transcript == "On a clear day, the sky is blue."

    def test_a_transcript_only_stream_still_emits_no_chunk(self):
        """Acceptance: no playable chunk is invented to carry the words.
        A turn with words and no bytes keeps them for history only."""
        frames = [{"audio": {"transcript": "just words"}},
                  {"audio": {"expires_at": 1}}]
        got, sink = self._run(frames)
        assert got == []
        assert sink == ["just words"]

    def test_the_wire_and_history_agree(self):
        """One predicate behind both destinations, so a client's record
        and the session's record can never diverge."""
        from shared.plugins.model_provider._media_deltas import (
            ensure_spoken_part, model_wrote_text)

        for parts in ([], [Part.from_text("I wrote this")]):
            got, sink = self._run(
                self._frames(), wrote_text=lambda: model_wrote_text(parts))
            ensure_spoken_part(parts, "".join(sink))
            spoken_in_history = [
                p.text for p in parts
                if p.text == "On a clear day, the sky is blue."]
            assert bool(got[-1].transcript) == bool(spoken_in_history)

    def test_pending_text_counts_as_written(self):
        """The loop flushes text into a Part only at a tool-call boundary
        or the end of the stream, so at the marker a written answer is
        still in the accumulator -- which must therefore be asked too."""
        from shared.plugins.model_provider._media_deltas import model_wrote_text

        assert model_wrote_text([], ["Hello"]) is True
        assert model_wrote_text([], []) is False
        assert model_wrote_text([], [""]) is False
        assert model_wrote_text([Part.from_text("x")]) is True


    def test_the_words_reach_the_client_event(self):
        """End to end below the provider: decoder -> session delivery ->
        the `on_tool_output` hook that becomes `ToolOutputEvent`.  The
        transcript lands in `chunk` on the final media event, under the
        reserved model-output call id, and nowhere else."""
        from shared.jaato_session import JaatoSession, MODEL_MEDIA_CALL_ID

        calls = []

        class _Hooks:
            def on_tool_output(self, **kw):
                calls.append(kw)

        session = JaatoSession.__new__(JaatoSession)
        session._agent_id = "main"
        session._ui_hooks = _Hooks()
        session._trace = lambda *a, **k: None

        seq, pending, sink = 0, [], []
        for f in self._frames():
            seq = _media_deltas.emit_audio_delta(
                f, session._deliver_model_media, seq,
                transcript_sink=sink, pending=pending,
                wrote_text=lambda: False)
        _media_deltas.flush_audio_stream(
            session._deliver_model_media, pending, sink, lambda: False)

        assert [c["call_id"] for c in calls] == [MODEL_MEDIA_CALL_ID] * 2
        assert [(c["final"], c["chunk"]) for c in calls] == [
            (False, ""), (True, "On a clear day, the sky is blue.")]
        assert base64.b64decode(calls[-1]["data_b64"]) == b"two"


# ---- the streaming loops themselves --------------------------------------


def _wire_chunk(*, content=None, audio=None, finish_reason=None):
    """One chat-completions streaming chunk as both loops read it.

    Plain namespaces rather than ``MagicMock``: the decoder probes
    ``delta.audio`` with ``getattr(..., None)``, and a mock answers
    every attribute with a truthy mock, which would make every text
    chunk look like an audio object.
    """
    delta = SimpleNamespace(
        content=content, tool_calls=None, reasoning=None,
        reasoning_content=None, audio=audio, model_extra=None,
    )
    choice = SimpleNamespace(
        delta=delta, finish_reason=finish_reason,
        native_finish_reason=None, model_extra=None,
    )
    return SimpleNamespace(choices=[choice], usage=None, error=None,
                           model_extra=None)


def _openrouter_loop():
    from shared.plugins.model_provider.openrouter.provider import OpenRouterProvider

    provider = OpenRouterProvider()
    provider._client = MagicMock()
    provider._model_name = "openai/gpt-audio-mini"
    provider._enable_thinking = False
    return provider


def _compat_loop():
    from shared.plugins.model_provider.nebius.provider import NebiusProvider

    provider = NebiusProvider()
    provider._client = MagicMock()
    provider._model_name = "some/audio-model"
    provider._enable_thinking = False
    provider._trace = lambda _msg: None
    return provider


def _run_loop(provider, chunks):
    """Drive ``_stream_response`` over ``chunks``; return (media, response)."""
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    stream.response = None
    provider._client.chat.completions.create = lambda **kw: stream
    media = []

    def on_chunk(item):
        if isinstance(item, MediaDelta):
            media.append(item)

    response = provider._stream_response(messages=[], kwargs={}, on_chunk=on_chunk)
    return media, response


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest.mark.parametrize("loop", [_openrouter_loop, _compat_loop],
                         ids=["openrouter", "openai_compat"])
class TestTheLoopsHandTheDecoderALivePredicate:
    """The loop-level half of #869, guarded by BEHAVIOUR.

    `TestTheWordsRideTheFinalChunk` proves the decoder honours a
    `wrote_text` callable it is handed.  It says nothing about what the
    two streaming loops actually hand it -- and that is where the
    subtle part lives: the end-of-audio marker lands MID-stream, so a
    written answer may still be sitting in the loop's not-yet-flushed
    accumulator, and a predicate snapshotted at loop start (or a bool)
    would answer "spoke only" for a turn that also wrote.

    These tests drive each real `_stream_response` over a fake wire and
    read the result off the emitted chunks, so a refactor that keeps
    the property passes (lambda or def, either name) and one that loses
    it fails, whatever it is spelled like.
    """

    def test_text_written_after_audio_started_suppresses_the_transcript(self, loop):
        """The order that breaks a snapshot: audio first, text second,
        then the marker.  At the marker the text is in the accumulator
        and nowhere else; only a live question sees it."""
        media, response = _run_loop(loop(), [
            _wire_chunk(audio={"transcript": "The sky "}),
            _wire_chunk(audio={"data": _b64(b"one")}),
            _wire_chunk(content="The sky is blue."),      # written, mid-utterance
            _wire_chunk(audio={"transcript": "is blue."}),
            _wire_chunk(audio={"data": _b64(b"two")}),
            _wire_chunk(audio={"expires_at": 1788}),
        ])
        assert [(c.final, c.transcript) for c in media] == [(False, ""), (True, "")]
        # History agrees: the written text stands, and is not doubled
        # by a spoken Part carrying the same words.
        assert [p.text for p in response.parts if p.text] == ["The sky is blue."]

    def test_a_turn_that_only_spoke_delivers_the_words_through_the_loop(self, loop):
        media, response = _run_loop(loop(), [
            _wire_chunk(audio={"transcript": "The sky "}),
            _wire_chunk(audio={"data": _b64(b"one")}),
            _wire_chunk(audio={"transcript": "is blue."}),
            _wire_chunk(audio={"data": _b64(b"two")}),
            _wire_chunk(audio={"expires_at": 1788}),
        ])
        assert [(c.final, c.transcript) for c in media] == [
            (False, ""), (True, "The sky is blue.")]
        assert [p.text for p in response.parts if p.text] == ["The sky is blue."]

    def test_the_flush_path_carries_the_words_too(self, loop):
        """No marker: the stream just ends.  The loop's end-of-stream
        flush must release the held chunk WITH the transcript, or a
        provider that sends no marker leaves its clients deaf."""
        media, _ = _run_loop(loop(), [
            _wire_chunk(audio={"transcript": "Sure."}),
            _wire_chunk(audio={"data": _b64(b"one")}),
        ])
        assert [(c.final, c.transcript) for c in media] == [(True, "Sure.")]

    def test_the_flush_path_honours_written_text_too(self, loop):
        """The flush release is a second call site; the predicate must
        reach it as well, or a write-and-speak turn with no marker
        delivers the words twice."""
        media, response = _run_loop(loop(), [
            _wire_chunk(audio={"data": _b64(b"one")}),
            _wire_chunk(content="Written."),
            _wire_chunk(audio={"transcript": "Spoken."}),
        ])
        assert [(c.final, c.transcript) for c in media] == [(True, "")]
        assert [p.text for p in response.parts if p.text] == ["Written."]
