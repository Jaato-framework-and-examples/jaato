"""Tests for the modality content-boundary gate (multimodal PR 3).

The gate (``JaatoSession._gate_tool_results_for_active_modalities``) is the
load-bearing correctness piece of multimodal-by-composition
(``docs/design/multimodal-model-support.md``): when a tool returns
attachment content the active model can't view (e.g. a ``readFile`` image
while in a text-only tier), the gate strips those attachments and appends
an actionable, self-correcting note rather than silently sending
unviewable bytes — so the turn continues and the agent can
``enter_tier("vision")`` and retry.

Constructed via ``__new__`` (no full session init): the gate methods only
read ``_provider`` / ``_model_name`` / ``_tier_config`` / ``_trace``.
"""

from __future__ import annotations

import pytest

from shared.jaato_session import JaatoSession
from shared.model_tiers import ModelTierConfig, ModelTierConfigError, TierEntry
from jaato_sdk.plugins.model_provider.types import Attachment, ToolResult


class _FakeProvider:
    name = "openrouter"

    def __init__(self, modalities):
        self._modalities = set(modalities)

    def supports_modality(self, kind: str, model=None) -> bool:
        return kind in self._modalities


def _session(provider, model="openai/gpt-5-mini", tier_config=None):
    s = JaatoSession.__new__(JaatoSession)
    s._provider = provider
    s._model_name = model
    s._tier_config = tier_config
    s._trace = lambda *a, **k: None
    return s


def _img_result(text="file bytes follow"):
    return ToolResult(
        call_id="c1",
        name="readFile",
        result=text,
        attachments=[Attachment(mime_type="image/png", data=b"\x89PNG...")],
    )


_VISION_TIERS = ModelTierConfig(
    tiers={
        "executor": TierEntry("openai/gpt-5-mini"),
        "vision": TierEntry("google/gemini-3-pro"),
    },
    initial_tier="executor",
    tier_fallback="executor",
)


class TestMimeToModality:
    def test_classification(self):
        f = JaatoSession._mime_to_modality
        assert f("image/png") == "image"
        assert f("image/jpeg; charset=binary") == "image"
        assert f("audio/wav") == "audio"
        assert f("video/mp4") == "video"
        assert f("application/pdf") == "file"
        assert f("application/octet-stream") is None  # unclassifiable
        assert f(None) is None
        assert f("") is None


class TestContentGate:
    def test_text_only_provider_withholds_image(self):
        s = _session(_FakeProvider({"text"}), tier_config=_VISION_TIERS)
        out = s._gate_tool_results_for_active_modalities([_img_result()])
        assert len(out) == 1
        r = out[0]
        assert r.attachments is None  # image stripped
        assert "file bytes follow" in r.result  # original result kept STRUCTURED
        # The withheld-attachment note is MODEL-FACING → model_suffix (appended
        # at serialization), not folded into the structured result.
        assert "withheld" in r.model_suffix.lower()
        assert 'enter_tier("vision")' in r.model_suffix  # actionable, vision tier exists

    def test_vision_provider_passes_image_through_unchanged(self):
        s = _session(_FakeProvider({"text", "image"}), model="google/gemini-3-pro")
        original = _img_result()
        out = s._gate_tool_results_for_active_modalities([original])
        assert out[0] is original  # identity: untouched

    def test_note_without_vision_tier_explains_no_tier(self):
        s = _session(_FakeProvider({"text"}), tier_config=None)
        out = s._gate_tool_results_for_active_modalities([_img_result()])
        r = out[0]
        assert r.attachments is None
        assert "no tier that can" in r.model_suffix.lower()
        assert "enter_tier" not in r.model_suffix  # nothing to switch into

    def test_mixed_attachments_keeps_supported_strips_unsupported(self):
        # An image (unsupported) alongside an unclassifiable blob (kept —
        # the gate never over-strips what it can't classify).
        s = _session(_FakeProvider({"text"}), tier_config=_VISION_TIERS)
        res = ToolResult(
            call_id="c2",
            name="readFile",
            result="data",
            attachments=[
                Attachment(mime_type="image/png", data=b"img"),
                Attachment(mime_type="application/octet-stream", data=b"blob"),
            ],
        )
        out = s._gate_tool_results_for_active_modalities([res])
        kept = out[0].attachments
        assert kept is not None and len(kept) == 1
        assert kept[0].mime_type == "application/octet-stream"

    def test_result_without_attachments_untouched(self):
        s = _session(_FakeProvider({"text"}))
        res = ToolResult(call_id="c3", name="cli", result="ok")
        out = s._gate_tool_results_for_active_modalities([res])
        assert out[0] is res

    def test_no_provider_is_noop(self):
        s = _session(None)
        results = [_img_result()]
        assert s._gate_tool_results_for_active_modalities(results) is results

    def test_audio_attachment_withheld_for_text_model(self):
        s = _session(_FakeProvider({"text", "image"}), tier_config=_VISION_TIERS)
        res = ToolResult(
            call_id="c4",
            name="readFile",
            result="clip",
            attachments=[Attachment(mime_type="audio/wav", data=b"RIFF")],
        )
        out = s._gate_tool_results_for_active_modalities([res])
        assert out[0].attachments is None
        assert "audio" in out[0].model_suffix.lower()
        # image-capable model + audio withheld + vision-only tier => the
        # generic note (vision tier is for image, not audio).
        assert "no tier that can" in out[0].model_suffix.lower() \
            or "enter_tier" not in out[0].model_suffix


class _ModelAwareProvider:
    """Provider whose image support depends on the *queried* model — mirrors
    a gateway serving mixed fleets (parameterized modalities(model=...))."""

    name = "openrouter"

    def __init__(self, vision_models):
        self._vision = set(vision_models)

    def supports_modality(self, kind: str, model=None) -> bool:
        return kind == "image" and model in self._vision


def _val_session(provider, tier_config):
    s = JaatoSession.__new__(JaatoSession)
    s._provider = provider
    s._tier_config = tier_config
    return s


def _tiers(vision_model=None):
    tiers = {"executor": TierEntry("openai/gpt-5-mini")}
    if vision_model is not None:
        tiers["vision"] = TierEntry(vision_model)
    return ModelTierConfig(
        tiers=tiers, initial_tier="executor", tier_fallback="executor"
    )


class TestVisionTierValidation:
    """``_validate_modality_tier_capabilities`` fails loud at
    provider-resolution time when a tier declaring a modality role maps to a
    model that can't accept that input — the earlier-warning over the
    runtime content gate.

    A tier named ``vision`` declares the ``image`` role implicitly, which is
    what these cases exercise; ``TestModalityRoleByKey`` covers the same
    machinery reached through an explicit ``modalities`` key on a
    differently-named tier.
    """

    def test_valid_vision_tier_passes(self):
        prov = _ModelAwareProvider({"google/gemini-3-pro"})
        s = _val_session(prov, _tiers(vision_model="google/gemini-3-pro"))
        s._validate_modality_tier_capabilities()  # no raise

    def test_text_only_vision_model_raises(self):
        prov = _ModelAwareProvider({"google/gemini-3-pro"})
        s = _val_session(prov, _tiers(vision_model="openai/gpt-5-mini"))
        with pytest.raises(ModelTierConfigError) as exc:
            s._validate_modality_tier_capabilities()
        msg = str(exc.value)
        assert "does not declare image input" in msg
        assert "openai/gpt-5-mini" in msg
        assert "modalities" in msg  # points at the assert-knob escape hatch

    def test_no_vision_tier_is_noop(self):
        prov = _ModelAwareProvider(set())
        s = _val_session(prov, _tiers(vision_model=None))
        s._validate_modality_tier_capabilities()  # no raise

    def test_missing_provider_or_config_is_noop(self):
        _val_session(None, None)._validate_modality_tier_capabilities()
        _val_session(
            _ModelAwareProvider(set()), None
        )._validate_modality_tier_capabilities()


class TestModalityRoleByKey:
    """The `modalities` key: a tier declares WHICH input roles it fills, and
    both the content gate and the startup capability check resolve the tier
    by that role rather than by the literal name ``vision``.

    The defect this closes: only a tier literally called ``vision`` could
    ever be the image tier.  The gate checked ``"vision" in tiers`` and the
    startup check looked up ``tiers["vision"]``, so a profile whose
    image-capable model sat on any other tier got neither the actionable
    note nor the config-time check — it confabulated about images instead of
    switching to a tier it already had.

    Uses ``planner`` as the role-bearing tier because tier NAMES are still a
    closed set (:data:`~shared.model_tiers.VALID_TIER_NAMES`); the role and
    the name are what this decouples, which is also the prerequisite for
    opening the name set up later.
    """

    def _custom(self, model="google/gemini-3-pro", modalities=("image",)):
        return ModelTierConfig(
            tiers={
                "executor": TierEntry("openai/gpt-5-mini"),
                "planner": TierEntry(model, inbound_modalities=frozenset(modalities)),
            },
            initial_tier="executor", tier_fallback="executor",
        )

    def test_gate_names_a_differently_named_image_tier(self):
        s = _session(_FakeProvider({"text"}), tier_config=self._custom())
        out = s._gate_tool_results_for_active_modalities([_img_result()])
        assert 'enter_tier("planner")' in out[0].model_suffix

    def test_startup_check_covers_a_differently_named_tier(self):
        prov = _ModelAwareProvider({"google/gemini-3-pro"})
        s = _val_session(prov, self._custom(model="openai/gpt-5-mini"))
        with pytest.raises(ModelTierConfigError) as exc:
            s._validate_modality_tier_capabilities()
        msg = str(exc.value)
        assert "'planner'" in msg          # names the tier, not "vision"
        assert "'image'" in msg
        assert "openai/gpt-5-mini" in msg

    def test_startup_check_passes_for_a_capable_custom_tier(self):
        prov = _ModelAwareProvider({"google/gemini-3-pro"})
        _val_session(prov, self._custom())._validate_modality_tier_capabilities()

    def test_gate_generalises_beyond_image(self):
        # An audio role on a custom tier makes the audio note actionable —
        # previously only `image` could ever produce an enter_tier hint.
        cfg = self._custom(modalities=("audio",))
        s = _session(_FakeProvider({"text", "image"}), tier_config=cfg)
        res = ToolResult(
            call_id="c9", name="readFile", result="clip",
            attachments=[Attachment(mime_type="audio/wav", data=b"RIFF")],
        )
        out = s._gate_tool_results_for_active_modalities([res])
        assert 'enter_tier("planner")' in out[0].model_suffix

    def test_tier_without_a_role_is_never_suggested(self):
        cfg = ModelTierConfig(
            tiers={"executor": TierEntry("openai/gpt-5-mini"),
                   "dispatcher": TierEntry("google/gemini-3-pro")},
            initial_tier="executor", tier_fallback="executor",
        )
        s = _session(_FakeProvider({"text"}), tier_config=cfg)
        out = s._gate_tool_results_for_active_modalities([_img_result()])
        # dispatcher's model CAN see, but the tier claims no role, so the
        # gate must not invent one — declaring the role is the contract.
        assert "enter_tier" not in out[0].model_suffix
        assert "no tier that can" in out[0].model_suffix.lower()
