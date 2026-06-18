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
            def modalities(self):
                return {MODALITY_TEXT, MODALITY_IMAGE}

        v = Vision()
        assert v.supports_modality("image") is True
        assert v.supports_modality("audio") is False


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
