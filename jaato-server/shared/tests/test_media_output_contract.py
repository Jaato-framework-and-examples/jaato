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
    extract_audio_delta,
)
from shared.plugins.model_provider.base import ModalityCapabilityMixin


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
