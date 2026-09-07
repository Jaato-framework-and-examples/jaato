"""The inbound half of the modality content gate (#847).

``_gate_tool_results_for_active_modalities`` covered content coming back
*from a tool*.  Nothing covered content the caller *gave* the session — a
user utterance carrying ``audio/wav``, which #830 made possible — so it was
replayed on every later request, including the ones made after
``enter_tier`` moved the session onto a text-only model.  OpenRouter
answered those with ``404 No endpoints found that support input audio``,
which reads like a missing model and is really a refused request.

The requirement that shapes every test here is that the filter is
**per-request, not destructive**: the text tier must not see the audio, and
the audio tier must still see it afterwards.  A fix that stripped stored
history would repair the planner by permanently deafening the session, and
a test that only asserts "the text tier sent no audio" passes for that
broken fix too — so each case below checks the stored history as well.

Sessions are built with ``__new__`` (no full init): the gate reads only
``_provider`` / ``_model_name`` / ``_tier_config`` / ``_history`` /
``_trace``.
"""

from __future__ import annotations

import pytest

from shared.jaato_session import JaatoSession
from shared.model_tiers import ModelTierConfig, TierEntry
from shared.session_history import SessionHistory
from jaato_sdk.plugins.model_provider.types import (
    Attachment, Message, Part, Role, ToolResult,
)


_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "
_PNG = b"\x89PNG\r\n\x1a\n"


class _FakeProvider:
    """Minimal provider stand-in: a mutable input-modality set.

    ``set_modalities`` is how these tests spell a tier switch — the real
    ``_connect_tier_entry`` swaps the connected model underneath the same
    provider object, and ``modalities()`` then answers for the new model.
    """

    name = "openrouter"

    def __init__(self, modalities):
        self._modalities = set(modalities)

    def set_modalities(self, modalities) -> None:
        self._modalities = set(modalities)

    def supports_modality(self, kind: str, model=None) -> bool:
        return kind in self._modalities


_VOICE_TIERS = ModelTierConfig(
    tiers={
        "voice": TierEntry(
            "openai/gpt-audio-mini",
            description="Enter this tier to hear the caller.",
            inbound_modalities=frozenset({"audio"}),
            outbound_modalities=frozenset({"audio"}),
        ),
        "planner": TierEntry("openai/gpt-4o-mini"),
    },
    initial_tier="voice",
    tier_fallback="planner",
)


def _untouched(gated, session) -> bool:
    """Whether the gate copied nothing.

    ``SessionHistory.messages`` already hands out a fresh *list*, so list
    identity says nothing; what matters is that every ``Message`` in it is
    the stored object rather than a gated copy.
    """
    stored = session._history.messages_ref
    return len(gated) == len(stored) and all(
        g is m for g, m in zip(gated, stored)
    )


def _audio_part(mime="audio/wav"):
    return Part(inline_data={"mime_type": mime, "data": _WAV,
                             "display_name": "utterance.wav"})


def _session(provider, messages, model="openai/gpt-4o-mini",
             tier_config=_VOICE_TIERS, active_tier="planner"):
    s = JaatoSession.__new__(JaatoSession)
    s._provider = provider
    s._model_name = model
    s._tier_config = tier_config
    s._active_tier = active_tier
    s._trace = lambda *a, **k: None
    s._history = SessionHistory()
    s._history.replace(list(messages))
    return s


def _heard_one_utterance():
    """History of a session that heard audio, then did some text work."""
    return [
        Message(role=Role.USER, parts=[
            Part.from_text("please listen"), _audio_part(),
        ]),
        Message(role=Role.MODEL, parts=[Part.from_text("I heard you.")]),
    ]


class TestPartCarriesBinary:
    """The cheap pre-check that keeps text-only sessions out of the gate."""

    def test_text_and_calls_carry_nothing(self):
        f = JaatoSession._part_carries_binary
        assert f(Part.from_text("hi")) is False
        assert f(Part(function_response=ToolResult(
            call_id="c", name="t", result="ok"))) is False

    def test_inline_data_and_attachments_do(self):
        f = JaatoSession._part_carries_binary
        assert f(_audio_part()) is True
        assert f(Part(function_response=ToolResult(
            call_id="c", name="t", result="ok",
            attachments=[Attachment(mime_type="image/png", data=_PNG)],
        ))) is True


class TestHistoryGate:
    def test_text_model_does_not_receive_the_audio(self):
        s = _session(_FakeProvider({"text"}), _heard_one_utterance())
        gated = s._history_for_provider()
        first = gated[0]
        assert not any(p.inline_data for p in first.parts)
        assert first.parts[0].text == "please listen"   # the words survive
        note = first.parts[-1].text
        assert "withheld" in note.lower()
        assert "audio" in note

    def test_history_is_not_mutated(self):
        """The sharp requirement: filtering must not deafen the session."""
        s = _session(_FakeProvider({"text"}), _heard_one_utterance())
        s._history_for_provider()
        stored = s._history.messages[0]
        inline = [p.inline_data for p in stored.parts if p.inline_data]
        assert len(inline) == 1
        assert inline[0]["data"] == _WAV
        assert len(stored.parts) == 2                   # no note appended

    def test_switching_back_sees_the_audio_again(self):
        provider = _FakeProvider({"text"})
        s = _session(provider, _heard_one_utterance())
        s._history_for_provider()                       # planner turn
        provider.set_modalities({"text", "audio"})      # enter_tier("voice")
        s._model_name = "openai/gpt-audio-mini"
        s._active_tier = "voice"
        gated = s._history_for_provider()
        assert _untouched(gated, s)                     # no copies made
        inline = [p.inline_data for p in gated[0].parts if p.inline_data]
        assert inline and inline[0]["data"] == _WAV

    def test_capable_model_pays_nothing(self):
        """No copies, no note — the list handed over is the stored one."""
        s = _session(_FakeProvider({"text", "audio"}), _heard_one_utterance())
        assert _untouched(s._history_for_provider(), s)

    def test_text_only_history_is_untouched(self):
        s = _session(_FakeProvider({"text"}), [
            Message.from_text(Role.USER, "hello"),
            Message.from_text(Role.MODEL, "hi"),
        ])
        assert _untouched(s._history_for_provider(), s)

    def test_note_names_the_tier_that_can_hear(self):
        s = _session(_FakeProvider({"text"}), _heard_one_utterance())
        note = s._history_for_provider()[0].parts[-1].text
        assert 'enter_tier("voice")' in note
        # ...and NOT the tool-result remedy: nothing needs re-running, the
        # bytes are still in history and the voice tier's next request
        # carries them again.
        assert "re-run this tool" not in note

    def test_message_identity_fields_survive_the_copy(self):
        """GC's history-budget sync keys on ``message_id``."""
        s = _session(_FakeProvider({"text"}), _heard_one_utterance())
        original = s._history.messages[0]
        gated = s._history_for_provider()[0]
        assert gated is not original
        assert gated.message_id == original.message_id
        assert gated.role == original.role

    def test_unclassifiable_bytes_are_never_over_stripped(self):
        s = _session(_FakeProvider({"text"}), [
            Message(role=Role.USER, parts=[
                Part(inline_data={"mime_type": "application/octet-stream",
                                  "data": b"\x00\x01"}),
            ]),
        ])
        assert _untouched(s._history_for_provider(), s)

    def test_no_provider_is_a_no_op(self):
        s = _session(None, _heard_one_utterance())
        assert _untouched(s._history_for_provider(), s)

    def test_a_gated_message_is_never_contentless(self):
        """An audio-only turn keeps the note, or the model sees an empty turn."""
        s = _session(_FakeProvider({"text"}), [
            Message(role=Role.USER, parts=[_audio_part()]),
        ])
        parts = s._history_for_provider()[0].parts
        assert len(parts) == 1
        assert "withheld" in parts[0].text.lower()


class TestToolResultsInHistory:
    """A tool result gated in one tier is replayed in the next.

    ``_gate_one_tool_result`` runs when the result is produced, against the
    model that was active then — so an image a vision tier legitimately kept
    is still in history when the agent switches to a text tier, and the
    replay has the same problem the user utterance does.
    """

    def _history_with_image_result(self):
        return [
            Message(role=Role.MODEL, parts=[Part.from_text("reading")]),
            Message(role=Role.TOOL, parts=[Part.from_function_response(
                ToolResult(call_id="c1", name="readFile", result="bytes",
                           attachments=[Attachment(mime_type="image/png",
                                                   data=_PNG)]),
            )]),
        ]

    def test_image_attachment_is_withheld_from_a_text_model(self):
        s = _session(_FakeProvider({"text"}), self._history_with_image_result())
        gated = s._history_for_provider()
        result = gated[1].parts[0].function_response
        assert result.attachments is None
        assert result.result == "bytes"                 # structure preserved
        assert "withheld" in gated[1].parts[-1].text.lower()

    def test_stored_result_keeps_its_attachment(self):
        s = _session(_FakeProvider({"text"}), self._history_with_image_result())
        s._history_for_provider()
        stored = s._history.messages[1].parts[0].function_response
        assert stored.attachments and stored.attachments[0].data == _PNG

    def test_vision_model_passes_it_through(self):
        s = _session(_FakeProvider({"text", "image"}),
                     self._history_with_image_result())
        assert _untouched(s._history_for_provider(), s)


class TestTelemetryDescribesWhatWasSent:
    """A span describing the model's input must describe the real input.

    ``_record_input_messages_telemetry`` read the *stored* history, which
    was the same list as the request's until the gate existed.  Left alone
    it would report a text tier receiving the audio it was specifically not
    sent — the one reading that makes the 404 the gate prevents look
    impossible.
    """

    def _span_messages(self, session):
        captured = []

        class _Span:
            def set_input_messages(self, msgs):
                captured.append(msgs)

        session._system_instruction = "you are a helpdesk"
        session._record_input_messages_telemetry(_Span())
        return captured[0] if captured else []

    def test_span_reports_the_gated_turn(self):
        s = _session(_FakeProvider({"text"}), _heard_one_utterance())
        msgs = self._span_messages(s)
        user = [m for m in msgs if m.get("role") == "user"]
        assert user and "withheld" in str(user[0]["content"]).lower()

    def test_span_still_reports_audio_to_a_model_that_gets_it(self):
        """Guards the test above from passing for the wrong reason."""
        s = _session(_FakeProvider({"text", "audio"}),
                     _heard_one_utterance(), model="openai/gpt-audio-mini",
                     active_tier="voice")
        msgs = self._span_messages(s)
        assert not any("withheld" in str(m.get("content", "")).lower()
                       for m in msgs)


class TestOpenRouterWireHasNoInputAudio:
    """The 404's proximate cause, checked at the wire.

    ``openrouter/converters.py`` marshals ``audio/*`` as an ``input_audio``
    block for every model on the provider — correct for the 46 catalog
    models that advertise audio input, a refused request for the rest.  With
    the history gated, the block is simply not there to refuse.
    """

    def test_gated_history_emits_no_input_audio_block(self):
        from shared.plugins.model_provider.openrouter.converters import (
            history_to_openai,
        )
        s = _session(_FakeProvider({"text"}), _heard_one_utterance())
        wire = history_to_openai(s._history_for_provider())
        blocks = [b for m in wire if isinstance(m.get("content"), list)
                  for b in m["content"]]
        assert not [b for b in blocks if b.get("type") == "input_audio"]
        assert any("withheld" in str(b.get("text", "")).lower() for b in blocks
                   ) or any("withheld" in str(m.get("content", "")).lower()
                            for m in wire)

    def test_ungated_history_still_does(self):
        """Guards the test above from passing for the wrong reason."""
        from shared.plugins.model_provider.openrouter.converters import (
            history_to_openai,
        )
        wire = history_to_openai(_heard_one_utterance())
        blocks = [b for m in wire if isinstance(m.get("content"), list)
                  for b in m["content"]]
        assert [b for b in blocks if b.get("type") == "input_audio"]
