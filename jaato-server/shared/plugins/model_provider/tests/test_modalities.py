"""Unit tests for the input-modality capability primitive.

Covers :func:`resolve_modalities` (the sibling of
``resolve_context_window``), the ``"text"``-floor normalisation, and the
:class:`ModalityCapabilityMixin` default — the foundation of multimodal
support via model-tier roles (see
``docs/design/multimodal-model-support.md``).

PR 1 of the multimodal arc: capability layer only, no behaviour change.
The tier-role activation + content-boundary gate land in a follow-up.
"""

from __future__ import annotations

import pytest

from ..base import (
    MODALITY_IMAGE,
    MODALITY_TEXT,
    ModalityCapabilityMixin,
    _normalise_modalities,
    resolve_modalities,
)


class TestNormaliseModalities:
    def test_lowercases_strips_and_drops_blanks(self):
        assert _normalise_modalities([" IMAGE ", "Text", "", "  "]) == {
            "image",
            "text",
        }

    def test_always_re_adds_text_floor(self):
        # Every text-completion model accepts text — the floor is a
        # contract invariant, not a guessed fallback.
        assert _normalise_modalities(["audio"]) == {"audio", "text"}
        assert _normalise_modalities([]) == {"text"}


class TestResolveModalitiesPrecedence:
    def test_none_when_nothing_resolves(self):
        assert resolve_modalities() is None

    def test_detect_wins(self):
        assert resolve_modalities(
            detect=lambda: ["image", "text"],
            profile_value=["text"],
            table_value=["text"],
        ) == {"image", "text"}

    def test_falls_through_to_profile_knob_when_detect_empty(self):
        # An empty/None detect (catalog gap, unreachable endpoint) must
        # degrade to the manual tiers, not resolve to an empty set.
        assert resolve_modalities(
            detect=lambda: None, profile_value=["image"]
        ) == {"image", "text"}
        assert resolve_modalities(
            detect=lambda: [], profile_value=["image"]
        ) == {"image", "text"}

    def test_table_is_last_resort(self):
        assert resolve_modalities(table_value=["image"]) == {"image", "text"}

    def test_detect_failure_tolerant_returns_none_not_raise(self):
        # The helper trusts detect to be failure-tolerant; a detect that
        # returns None (its own try/except degraded) just falls through.
        assert resolve_modalities(detect=lambda: None) is None


class TestModalityCapabilityMixin:
    def test_default_is_text_only_floor(self):
        class P(ModalityCapabilityMixin):
            pass

        assert P().modalities() == {MODALITY_TEXT}

    def test_supports_modality_is_case_insensitive(self):
        class P(ModalityCapabilityMixin):
            pass

        p = P()
        assert p.supports_modality("TEXT") is True
        assert p.supports_modality(" text ") is True
        assert p.supports_modality(MODALITY_IMAGE) is False

    def test_override_changes_supports_modality(self):
        class Vision(ModalityCapabilityMixin):
            def modalities(self, model=None):
                return {MODALITY_TEXT, MODALITY_IMAGE}

        v = Vision()
        assert v.supports_modality("image") is True
        assert v.supports_modality("audio") is False


class TestImageCapableProviderTables:
    """The closed image-capable providers (no public modality catalog)
    declare vision via per-model static tables — the table tier of the
    resolve_modalities chain.  These pin that a known vision model resolves
    ``image`` and a known text-only model stays at the text floor, so the
    content-boundary gate (follow-up) doesn't regress existing
    image-to-anthropic/google usage.
    """

    def _bare(self, cls):
        # Construct without __init__ (no network/auth); set the two fields
        # modalities() reads.
        p = cls.__new__(cls)
        p._model_name = None
        p._modalities_knob = None
        return p

    def test_anthropic_claude_is_vision_legacy_is_text(self):
        from ..anthropic.provider import AnthropicProvider

        p = self._bare(AnthropicProvider)
        # Anthropic / Google declare {"text","image","file"} for PDF-capable models
        # (MODALITY_FILE = "PDFs / documents"); antigravity's table is image-only.
        p._model_name = "claude-opus-4-5-20251101"
        assert p.modalities() == {"text", "image", "file"}
        p._model_name = "claude-3-5-sonnet-20241022"
        assert p.modalities() == {"text", "image", "file"}
        p._model_name = "claude-2.1"  # legacy, text-only, absent from table
        assert p.modalities() == {"text"}

    def test_google_gemini_modern_is_vision_legacy_is_text(self):
        from ..google_genai.provider import GoogleGenAIProvider

        p = self._bare(GoogleGenAIProvider)
        for m in ("gemini-2.5-pro", "gemini-1.5-flash-latest",
                  "gemini-3-pro-preview", "gemini-2.0-flash"):
            p._model_name = m
            assert p.modalities() == {"text", "image", "file"}, m
        for m in ("gemini-1.0-pro", "gemini-pro"):
            p._model_name = m
            assert p.modalities() == {"text"}, m

    def test_antigravity_served_models_are_vision(self):
        from ..antigravity.provider import AntigravityProvider

        p = self._bare(AntigravityProvider)
        for m in ("antigravity-gemini-3-pro",
                  "antigravity-claude-sonnet-4-5", "gemini-2.5-flash"):
            p._model_name = m
            assert p.modalities() == {"text", "image"}, m
        p._model_name = "not-a-served-model"
        assert p.modalities() == {"text"}

    def test_modalities_accepts_explicit_model_for_other_than_active(self):
        # The active model is text-only, but a caller (vision-tier
        # validation) asks about a DIFFERENT model without switching.
        from ..anthropic.provider import AnthropicProvider

        p = self._bare(AnthropicProvider)
        p._model_name = "claude-2.1"  # active = text-only
        assert p.modalities() == {"text"}
        assert p.modalities(model="claude-opus-4-5") == {"text", "image", "file"}
        assert p.supports_modality("image", model="claude-3-5-sonnet") is True
        assert p.supports_modality("image", model="claude-2.1") is False

    def test_knob_overrides_table_absence(self):
        # The framework_overrides.modalities knob asserts a model the table
        # doesn't cover (forward-compat for new families).
        from ..anthropic.provider import AnthropicProvider

        p = self._bare(AnthropicProvider)
        p._model_name = "claude-5-future"
        assert p.modalities() == {"text"}  # not yet in table
        p._modalities_knob = ["text", "image"]
        assert p.modalities() == {"text", "image"}


def test_all_providers_expose_the_capability_contract():
    """Every shipped provider must answer ``modalities()`` /
    ``supports_modality()`` — the content-boundary gate (follow-up PR)
    calls them on whichever provider is active, so a missing method would
    AttributeError at runtime.  Providers inherit the mixin (text-only
    floor) or override it; this pins that none was missed.
    """
    import importlib

    provider_modules = [
        ("anthropic", "AnthropicProvider"),
        ("antigravity", "AntigravityProvider"),
        ("claude_cli", "ClaudeCLIProvider"),
        ("nim", "NIMProvider"),
        ("google_genai", "GoogleGenAIProvider"),
        ("github_models", "GitHubModelsProvider"),
        ("tensorrt_llm", "TensorRTLLMProvider"),
        ("lmstudio", "LMStudioProvider"),
        ("triton", "TritonProvider"),
        ("vllm", "VLLMProvider"),
        ("zhipuai_openai", "ZhipuAIOpenAIProvider"),
        ("openrouter", "OpenRouterProvider"),
        ("ollama", "OllamaProvider"),
        ("zhipuai", "ZhipuAIProvider"),
        ("nebius", "NebiusProvider"),
        ("ovhcloud", "OVHcloudProvider"),
        ("doubleword", "DoublewordProvider"),
        ("helmcode", "HelmcodeProvider"),
    ]
    for pkg, cls_name in provider_modules:
        mod = importlib.import_module(
            f"shared.plugins.model_provider.{pkg}.provider"
        )
        cls = getattr(mod, cls_name)
        assert hasattr(cls, "modalities"), f"{cls_name} missing modalities()"
        assert hasattr(
            cls, "supports_modality"
        ), f"{cls_name} missing supports_modality()"
