"""The inbound gate reads the ACTIVE TIER's declared roles too (#1001).

#847 gave the send path a modality gate keyed on one thing: the active
model's catalog INPUT capability.  That is the right bound and it is not
the only one.  A tier declares what it is *for* —

    voz:  {model: openai/gpt-audio, modalities: {audio: outbound}}

— and ``openai/gpt-audio`` lists ``audio`` among its catalog INPUT
modalities (verified against OpenRouter's live catalog: ``input:
['text', 'audio']``).  So a session that heard a Telegram voice note in a
``gemini-2.5-flash`` tier and then ``enter_tier("voz")`` to *speak* the
answer handed the recorded OGG to the speaking tier, which needed only the
text, and the upstream refused the whole request:

    Invalid value: 'ogg'. Supported values are: 'wav' and 'mp3'.
    param: messages[1].content[1].input_audio.format

The tier's own declaration — audio OUTBOUND, no inbound role — was never
consulted by the gate.

Three properties are load-bearing here and each has its own class:

* the declaration NARROWS and never widens (``TestNarrowNeverWiden``) — a
  tier claiming ``audio: inbound`` on a text-only model still withholds;
* a session that declared nothing behaves exactly as it did
  (``TestUndeclaredSessionsAreUnchanged``), which includes a ``vision``
  tier carrying only its IMPLICIT role;
* the model-facing note states the TRUE reason
  (``TestTheNoteDoesNotBlameTheModel``) — telling a model on
  ``openai/gpt-audio`` that the wire cannot carry audio is false, and a
  model with a true fact to contradict spends a turn contradicting it.

Sessions are built with ``__new__`` (no full init): the gate reads only
``_provider`` / ``_model_name`` / ``_tier_config`` / ``_active_tier`` /
``_history`` / ``_trace``.
"""

from __future__ import annotations

import pytest

from shared.jaato_session import JaatoSession
from shared.model_tiers import ModelTierConfig, TierEntry
from shared.session_history import SessionHistory
from jaato_sdk.plugins.model_provider.types import (
    Attachment, Message, Part, Role, ToolResult,
)


from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""        if not provider.supports_modality(kind):
            return "model"
        declared = self._active_tier_inbound_modalities()
        if declared is not None and kind not in declared:
            return "tier"
        return None""",
        replace="""        if not provider.supports_modality(kind):
            return "model"
        return None""",
        test="TestTheReportedFailure::test_the_speaking_tier_does_not_receive_the_audio",
        because="a tier declaring audio OUTBOUND still being handed the "
                "caller's recorded utterance, because the gate asked the "
                "model's catalog and never the tier's declared role",
    ),
    Reversion(
        target="jaato-server/shared/model_tiers.py",
        find="""        if not authored_in and not authored_out:
            return None
        return entry.inbound_modalities""",
        replace="""        if not entry.declares_any_modality:
            return None
        return entry.inbound_modalities""",
        test="TestUndeclaredSessionsAreUnchanged::test_an_implicit_vision_tier_still_receives_other_content",
        because="a tier named 'vision' that declared no modalities of its "
                "own arming the gate off its IMPLICIT image role, so the "
                "back-compat shim starts withholding content instead of "
                "keeping old profiles unchanged",
    ),
]


_OGG = b"OggS\x00\x02\x00\x00"
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


# The reported profile's shape: the tier that HEARS declares nothing (it
# is a general-purpose executor that happens to sit on a model accepting
# every modality), and the tier that SPEAKS declares audio outbound only.
_VOICE_OUT_TIERS = ModelTierConfig(
    tiers={
        "executor": TierEntry("google/gemini-2.5-flash"),
        "voz": TierEntry(
            "openai/gpt-audio",
            description="Enter this tier to speak the answer aloud.",
            outbound_modalities=frozenset({"audio"}),
        ),
    },
    initial_tier="executor",
    tier_fallback="executor",
)

# The same thing with somewhere to send the agent: a tier that DOES
# declare an inbound audio role, so the note can name it.
_VOICE_OUT_WITH_EARS = ModelTierConfig(
    tiers={
        "executor": TierEntry("google/gemini-2.5-flash"),
        "oido": TierEntry(
            "openai/gpt-audio",
            description="Enter this tier to hear the caller.",
            inbound_modalities=frozenset({"audio"}),
        ),
        "voz": TierEntry(
            "openai/gpt-audio",
            description="Enter this tier to speak the answer aloud.",
            outbound_modalities=frozenset({"audio"}),
        ),
    },
    initial_tier="executor",
    tier_fallback="executor",
)


def _audio_part(mime="audio/ogg"):
    return Part(inline_data={"mime_type": mime, "data": _OGG,
                             "display_name": "voice.ogg"})


def _image_part():
    return Part(inline_data={"mime_type": "image/png", "data": _PNG,
                             "display_name": "shot.png"})


def _session(provider, messages, model="openai/gpt-audio",
             tier_config=_VOICE_OUT_TIERS, active_tier="voz"):
    s = JaatoSession.__new__(JaatoSession)
    s._provider = provider
    s._model_name = model
    s._tier_config = tier_config
    s._active_tier = active_tier
    s._trace = lambda *a, **k: None
    s._history = SessionHistory()
    s._history.replace(list(messages))
    return s


def _heard_a_voice_note():
    """A turn in which the caller spoke and the executor tier answered."""
    return [
        Message(role=Role.USER, parts=[
            Part.from_text("summarise this"), _audio_part(),
        ]),
        Message(role=Role.MODEL, parts=[Part.from_text("Will do.")]),
    ]


def _inline_mimes(messages):
    return [p.inline_data.get("mime_type")
            for m in messages for p in (m.parts or []) if p.inline_data]


def _note_text(messages):
    return " ".join(p.text or "" for m in messages
                    for p in (m.parts or []) if p.text)


class TestGatingInboundModalities:
    """``ModelTierConfig.gating_inbound_modalities`` — the arming rule."""

    def test_a_tier_declaring_nothing_does_not_arm_the_gate(self):
        assert _VOICE_OUT_TIERS.gating_inbound_modalities("executor") is None

    def test_an_outbound_only_declaration_arms_it_with_an_empty_set(self):
        """The reported case: declared, and declared NO inbound role.

        ``frozenset()`` and ``None`` are the two answers that must not be
        confused — one means "this tier accepts no non-text input", the
        other "this tier has no opinion".
        """
        got = _VOICE_OUT_TIERS.gating_inbound_modalities("voz")
        assert got == frozenset()
        assert got is not None

    def test_an_inbound_declaration_arms_it_with_that_set(self):
        assert _VOICE_OUT_WITH_EARS.gating_inbound_modalities(
            "oido") == frozenset({"audio"})

    def test_an_unknown_tier_name_does_not_arm_it(self):
        assert _VOICE_OUT_TIERS.gating_inbound_modalities("nope") is None

    def test_a_purely_implicit_vision_role_does_not_arm_it(self):
        """The back-compat shim must not become a behaviour change.

        A tier literally named ``vision`` gets ``{"image"}`` inbound from
        ``IMPLICIT_TIER_MODALITIES`` so that profiles written before the
        ``modalities`` key behave unchanged.  Letting that implied role arm
        the gate would make the shim start withholding audio and PDFs from
        such a tier — the opposite of what it is for.
        """
        cfg = ModelTierConfig(
            tiers={"vision": TierEntry("openai/gpt-4o"),
                   "dispatcher": TierEntry("openai/gpt-4o-mini")},
            initial_tier="dispatcher", tier_fallback="dispatcher",
        )
        assert cfg.tiers["vision"].inbound_modalities == frozenset({"image"})
        assert cfg.gating_inbound_modalities("vision") is None

    def test_a_vision_tier_that_authors_a_role_is_armed_and_keeps_image(self):
        """Authoring any role arms it; the implicit role stays in the set."""
        cfg = ModelTierConfig(
            tiers={"vision": TierEntry(
                "openai/gpt-audio",
                outbound_modalities=frozenset({"audio"}))},
            initial_tier="vision", tier_fallback="vision",
        )
        assert cfg.gating_inbound_modalities("vision") == frozenset({"image"})


class TestTheReportedFailure:
    """A speaking tier is not handed the utterance it never asked for."""

    def test_the_speaking_tier_does_not_receive_the_audio(self):
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note())
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(gated) == []

    def test_the_model_would_have_accepted_it(self):
        """Pins the premise: this is NOT the #847 capability refusal.

        If the model were text-only the audio would have been withheld
        before #1001 and the test above would pass for the old reason.
        """
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note())
        assert s._provider.supports_modality("audio") is True
        assert s._modality_refusal("audio", s._provider) == "tier"

    def test_stored_history_still_has_the_bytes(self):
        """Per-request, never destructive — the #847 invariant, unchanged."""
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note())
        s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(s._history.messages_ref) == ["audio/ogg"]

    def test_the_hearing_tier_still_hears_it(self):
        """The other half: gating must not deafen the tier that listens."""
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note(),
                     tier_config=_VOICE_OUT_WITH_EARS, active_tier="oido")
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(gated) == ["audio/ogg"]

    def test_the_text_of_the_turn_survives(self):
        """A gated user turn must not become contentless."""
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note())
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert "summarise this" in _note_text(gated)

    def test_the_wire_carries_no_input_audio_block(self):
        """End to end at the converter: the 400 has nothing left to refuse."""
        from shared.plugins.model_provider.openrouter.converters import (
            history_to_openai,
        )
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note())
        wire = history_to_openai(s._history_for_provider())
        blocks = [b for m in wire if isinstance(m.get("content"), list)
                  for b in m["content"]]
        assert not [b for b in blocks if b.get("type") == "input_audio"]

    def test_the_wire_does_carry_it_without_the_gate(self):
        """Guards the test above from passing for the wrong reason.

        The same history, converted straight, produces exactly the
        ``input_audio`` block whose ``format: "ogg"`` the upstream refused.
        """
        from shared.plugins.model_provider.openrouter.converters import (
            history_to_openai,
        )
        wire = history_to_openai(_heard_a_voice_note())
        blocks = [b for m in wire if isinstance(m.get("content"), list)
                  for b in m["content"]]
        audio = [b for b in blocks if b.get("type") == "input_audio"]
        assert audio and audio[0]["input_audio"]["format"] == "ogg"


class TestNarrowNeverWiden:
    """A tier declaration constrains; it can never grant."""

    def test_declaring_audio_inbound_on_a_text_model_still_withholds(self):
        cfg = ModelTierConfig(
            tiers={"oido": TierEntry(
                "some/text-only-model",
                description="Enter this tier to hear the caller.",
                inbound_modalities=frozenset({"audio"}))},
            initial_tier="oido", tier_fallback="oido",
        )
        s = _session(_FakeProvider({"text"}), _heard_a_voice_note(),
                     tier_config=cfg, active_tier="oido")
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(gated) == []

    def test_the_model_refusal_is_reported_as_the_model_s(self):
        """Most-restrictive-wins, and the reason names the tighter bound."""
        cfg = ModelTierConfig(
            tiers={"oido": TierEntry(
                "some/text-only-model",
                description="Enter this tier to hear the caller.",
                inbound_modalities=frozenset({"audio"}))},
            initial_tier="oido", tier_fallback="oido",
        )
        s = _session(_FakeProvider({"text"}), [], tier_config=cfg,
                     active_tier="oido")
        assert s._modality_refusal("audio", s._provider) == "model"

    def test_a_declared_kind_the_model_accepts_passes(self):
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note(),
                     tier_config=_VOICE_OUT_WITH_EARS, active_tier="oido")
        assert s._modality_refusal("audio", s._provider) is None


class TestUndeclaredSessionsAreUnchanged:
    """The backwards-compatibility surface, stated as tests.

    Most sessions have no tiers at all and most tiers declare no
    modalities; if either arms the gate, every single-model session stops
    receiving images.
    """

    def test_no_tier_config_at_all(self):
        s = _session(_FakeProvider({"text", "audio", "image"}),
                     _heard_a_voice_note(), tier_config=None,
                     active_tier=None)
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(gated) == ["audio/ogg"]
        assert gated is s._history.messages or all(
            g is m for g, m in zip(gated, s._history.messages_ref))

    def test_a_tier_config_whose_active_tier_declares_nothing(self):
        s = _session(_FakeProvider({"text", "audio", "image"}),
                     _heard_a_voice_note(), active_tier="executor")
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(gated) == ["audio/ogg"]

    def test_an_active_tier_that_is_not_in_the_config(self):
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note(),
                     active_tier="not-a-tier")
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(gated) == ["audio/ogg"]

    def test_an_implicit_vision_tier_still_receives_other_content(self):
        cfg = ModelTierConfig(
            tiers={"vision": TierEntry("openai/gpt-4o"),
                   "dispatcher": TierEntry("openai/gpt-4o-mini")},
            initial_tier="dispatcher", tier_fallback="dispatcher",
        )
        s = _session(_FakeProvider({"text", "audio", "image"}),
                     _heard_a_voice_note(), tier_config=cfg,
                     active_tier="vision")
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(gated) == ["audio/ogg"]

    def test_object_identity_is_preserved_when_nothing_is_withheld(self):
        """The cheap path #847 documents: no copies, no allocation."""
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note(),
                     active_tier="executor")
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert all(g is m
                   for g, m in zip(gated, s._history.messages_ref))


class TestAnArmedTierIsGatedPerKind:
    """Declaring one role does not admit the others."""

    def _mixed_history(self):
        return [Message(role=Role.USER, parts=[
            Part.from_text("look and listen"), _image_part(), _audio_part(),
        ])]

    def _session_declaring_image(self):
        cfg = ModelTierConfig(
            tiers={"ojo": TierEntry(
                "openai/gpt-4o",
                description="Enter this tier to look at an image.",
                inbound_modalities=frozenset({"image"}))},
            initial_tier="ojo", tier_fallback="ojo",
        )
        return _session(_FakeProvider({"text", "image", "audio"}),
                        self._mixed_history(), tier_config=cfg,
                        active_tier="ojo")

    def test_the_declared_kind_is_kept(self):
        s = self._session_declaring_image()
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert _inline_mimes(gated) == ["image/png"]

    def test_the_undeclared_kind_is_withheld_though_the_model_reads_it(self):
        s = self._session_declaring_image()
        assert s._provider.supports_modality("audio") is True
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert "audio/ogg" not in _inline_mimes(gated)


class TestToolResultAttachments:
    """The same rule on the other gate, for content a TOOL produced."""

    def _result_with_audio(self):
        return ToolResult(
            call_id="c1", name="readFile", result="read one file",
            attachments=[Attachment(mime_type="audio/ogg", data=_OGG)],
        )

    def test_the_speaking_tier_does_not_receive_a_tool_s_audio(self):
        s = _session(_FakeProvider({"text", "audio"}), [])
        gated = s._gate_tool_results_for_active_modalities(
            [self._result_with_audio()])
        assert not gated[0].attachments

    def test_a_tier_declaring_nothing_receives_it(self):
        s = _session(_FakeProvider({"text", "audio"}), [],
                     active_tier="executor")
        gated = s._gate_tool_results_for_active_modalities(
            [self._result_with_audio()])
        assert len(gated[0].attachments) == 1

    def test_the_note_reaches_the_model_suffix(self):
        s = _session(_FakeProvider({"text", "audio"}), [])
        gated = s._gate_tool_results_for_active_modalities(
            [self._result_with_audio()])
        assert "withheld" in (gated[0].model_suffix or "").lower()


class TestTheNoteDoesNotBlameTheModel:
    """A false explanation costs a turn in argument."""

    def _note(self, tier_config=_VOICE_OUT_TIERS, active_tier="voz"):
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note(),
                     tier_config=tier_config, active_tier=active_tier)
        gated = s._gate_history_for_active_modalities(s._history.messages)
        return _note_text(gated)

    def test_it_does_not_claim_the_model_cannot_read_the_content(self):
        note = self._note()
        assert "can't view" not in note

    def test_it_names_the_tier_and_the_missing_inbound_role(self):
        note = self._note()
        assert "'voz'" in note and "does not accept audio as input" in note

    def test_it_says_the_model_itself_can_read_it(self):
        assert "can read audio" in self._note()

    def test_it_names_a_tier_that_declares_the_role_inbound(self):
        note = self._note(tier_config=_VOICE_OUT_WITH_EARS)
        assert 'enter_tier("oido")' in note

    def test_it_says_so_when_no_tier_declares_the_role(self):
        note = self._note()
        assert "no other tier" in note
        assert "enter_tier" not in note

    def test_a_model_refusal_still_gets_the_model_note(self):
        """The #847 wording is untouched for the refusal it describes."""
        s = _session(_FakeProvider({"text"}), _heard_a_voice_note(),
                     active_tier="executor")
        gated = s._gate_history_for_active_modalities(s._history.messages)
        assert "can't view" in _note_text(gated)

    def test_both_notes_appear_when_one_pass_made_both_refusals(self):
        """An image the model can't see beside audio the tier declined."""
        cfg = ModelTierConfig(
            tiers={"voz": TierEntry(
                "openai/gpt-audio",
                description="Enter this tier to speak the answer aloud.",
                outbound_modalities=frozenset({"audio"}))},
            initial_tier="voz", tier_fallback="voz",
        )
        history = [Message(role=Role.USER, parts=[
            Part.from_text("both"), _image_part(), _audio_part()])]
        s = _session(_FakeProvider({"text", "audio"}), history,
                     tier_config=cfg, active_tier="voz")
        gated = s._gate_history_for_active_modalities(s._history.messages)
        note = _note_text(gated)
        assert "can't view image" in note
        assert "does not accept audio as input" in note


class TestTheAuthoringSurfaceAgrees:
    """`explain tiers` must not assert the opposite of what the gate does.

    The surface carried one sentence saying an OUTBOUND role "is INERT
    unless the named provider declares `output_media`" — true of the
    EMISSION half, and after #1001 false of the role, which now arms the
    inbound gate.  An author reading it before writing
    ``voz: {audio: outbound}`` would be told the declaration does nothing,
    when that declaration is exactly what withholds the caller's audio.

    That is the silent-ignore family (#910, #925, #947, #950) arriving
    from the other direction: a surface claiming inertness for something
    load-bearing.  These pin the correction rather than the wording, so
    they fail if the claim comes back, not when a sentence is reworded.
    """

    def _explained(self):
        from shared.scaffold import explain
        return explain.tiers()

    def test_the_outbound_note_no_longer_calls_the_role_inert(self):
        data, _ = self._explained()
        note = data["outbound_is_inert"]
        assert "EMISSION half" in note
        assert "ARMS the inbound gate" in note

    def test_it_states_the_gating_rule(self):
        data, text = self._explained()
        rule = data["inbound_gate_is_armed_by_any_role"]
        assert "outbound" in rule and "STOP inbound audio" in rule
        assert "HOW YOU STOP" in text

    def test_it_states_the_arming_trigger_exactly(self):
        """Not "is the inbound set non-empty" — an empty set still arms."""
        rule = self._explained()[0]["inbound_gate_is_armed_by_any_role"]
        assert "did the author write a role" in rule
        assert "EMPTY" in rule

    def test_it_states_that_a_declaration_narrows_and_never_widens(self):
        rule = self._explained()[0]["inbound_gate_is_armed_by_any_role"]
        assert "NARROWS, never" in rule and "INTERSECTED" in rule

    def test_it_states_the_fallback_for_an_unconfigured_session(self):
        fallback = self._explained()[0]["inbound_gate_fallback"]
        assert "ALONE" in fallback
        assert "vision" in fallback and "IMPLICIT" in fallback

    def test_it_states_why_the_tier_is_the_right_level(self):
        why = self._explained()[0]["why_the_tier_and_not_the_format"]
        assert "CONTAINERS" in why
        assert "gpt-audio" in why

    def test_the_modality_vocabulary_is_derived_not_restated(self):
        """The rule text must track the framework's own vocabulary."""
        from shared.model_tiers import VALID_TIER_MODALITIES
        rule = self._explained()[0]["inbound_gate_is_armed_by_any_role"]
        for kind in VALID_TIER_MODALITIES:
            assert kind in rule

    def test_validate_does_not_call_the_whole_outbound_role_inert(self):
        """The warning is about emission; the role still bounds intake."""
        import inspect
        from shared.scaffold import validate as V
        src = inspect.getsource(V._check_modality_direction)
        assert "EMISSION" in src
        assert "bounds what this tier" in src


class TestTraceNamesWhichBoundRefused:
    """An operator reading the trace must be able to tell them apart."""

    def test_the_trace_names_the_tier_refusal(self):
        lines = []
        s = _session(_FakeProvider({"text", "audio"}), _heard_a_voice_note())
        s._trace = lambda msg, *a, **k: lines.append(str(msg))
        s._gate_history_for_active_modalities(s._history.messages)
        joined = " ".join(lines)
        assert "HISTORY_MODALITY_GATE" in joined
        assert "'voz' does not declare ['audio'] inbound" in joined
