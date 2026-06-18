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

from shared.jaato_session import JaatoSession
from shared.model_tiers import ModelTierConfig, TierEntry
from jaato_sdk.plugins.model_provider.types import Attachment, ToolResult


class _FakeProvider:
    def __init__(self, modalities):
        self._modalities = set(modalities)

    def supports_modality(self, kind: str) -> bool:
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
        assert "file bytes follow" in r.result  # original text preserved
        assert "withheld" in r.result.lower()
        assert 'enter_tier("vision")' in r.result  # actionable, vision tier exists

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
        assert "no vision tier" in r.result.lower()
        assert "enter_tier" not in r.result  # nothing to switch into

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
        assert "audio" in out[0].result.lower()
        # image-capable model + audio withheld + vision-only tier => the
        # generic note (vision tier is for image, not audio).
        assert "no vision tier" in out[0].result.lower() \
            or "enter_tier" not in out[0].result
