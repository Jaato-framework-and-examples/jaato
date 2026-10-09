"""A decision model is recognised by its OUTPUT modality.

Step 1 of docs/design/decision-models.md (§4): ``"decisions"`` joins the
modality vocabulary as an OUTPUT-only token, the text floor stops adding
``text`` to a set that is ``{"decisions"}`` alone, and ``openrouter``
reads ``architecture.output_modalities`` from its catalog before falling
back to the ``framework_overrides.output_modalities`` knob.

``typesafe/jev-1.13`` is the case that shapes the lookup: OpenRouter serves
it (``GET /api/v1/models/typesafe/jev-1.13/endpoints`` answers) but its
listing omits it, so a lookup that read only the listing would never see
it.  The fixtures below are that document as the API returned it on
2026-10-06, trimmed to the fields read.

No test touches the network: the listing is set on the instance and the
per-model document is served by a stub of ``httpx.get``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from jaato_server.shared.plugins.model_provider.base import (
    MODALITY_DECISIONS,
    MODALITY_TEXT,
    ModalityCapabilityMixin,
    is_decisions_only_set,
    normalise_output_modalities,
)
from jaato_server.shared.plugins.model_provider.openrouter.provider import (
    OpenRouterProvider,
)
from jaato_server.shared.tests.reversion import Reversion

_BASE = "jaato-server/jaato_server/shared/plugins/model_provider/base.py"
_OR = "jaato-server/jaato_server/shared/plugins/model_provider/openrouter/provider.py"

REVERSIONS = [
    Reversion(
        target=_BASE,
        find=(
            "    if normalised == {MODALITY_DECISIONS}:\n"
            "        return normalised\n"
        ),
        replace="",
        test="test_a_decisions_only_set_gets_no_text_floor",
        because="the text floor makes a decision model look like a chat model",
    ),
    Reversion(
        target=_OR,
        find="            detected = self._lookup_output_modalities(model)\n",
        replace="            detected = None\n",
        test="test_the_listing_answers_for_audio_without_a_knob",
        because="the catalog is not read, so output modalities need a knob again",
    ),
    Reversion(
        target=_OR,
        find="        doc = self._fetch_model_doc(model)\n        if doc is None:\n",
        replace="        doc = None\n        if doc is None:\n",
        test="test_an_unlisted_decision_model_is_found_through_its_endpoints_doc",
        because="jev-1.13 is not in the listing, so a listing-only lookup never sees it",
    ),
    Reversion(
        target=_OR,
        find="        cache[model] = doc\n",
        replace="",
        test="test_the_endpoints_doc_is_fetched_once",
        because="every output_modalities() call re-fetches the document",
    ),
]


JEV_DOC: Dict[str, Any] = {
    "data": {
        "id": "typesafe/jev-1.13",
        "name": "TypeSafe: Jev 1.13",
        "architecture": {
            "tokenizer": "Other",
            "modality": "text->decisions",
            "input_modalities": ["text"],
            "output_modalities": ["decisions"],
        },
        "endpoints": [
            {
                "name": "TypeSafe | typesafe/jev-1.13-20260917",
                "context_length": 32000,
                "supported_parameters": [],
            },
        ],
    },
}

LISTING: List[Dict[str, Any]] = [
    {
        "id": "openai/gpt-4o",
        "context_length": 128000,
        "architecture": {
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
    },
    {
        "id": "openai/gpt-audio-mini",
        "context_length": 128000,
        "architecture": {
            "input_modalities": ["text", "audio"],
            "output_modalities": ["text", "audio"],
        },
    },
    {"id": "acme/no-architecture", "context_length": 8000},
]


class _Response:
    def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self._status = status

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self) -> Dict[str, Any]:
        return self._payload


@pytest.fixture
def served(monkeypatch):
    """Stub ``httpx.get``: the jev endpoints doc, 404 for anything else."""
    import httpx

    calls: List[str] = []

    def fake_get(url, timeout=None):
        calls.append(url)
        if url.endswith("/models/typesafe/jev-1.13/endpoints"):
            return _Response(JEV_DOC)
        return _Response({}, status=404)

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def _provider(model: str = "openai/gpt-4o") -> OpenRouterProvider:
    provider = OpenRouterProvider()
    provider._catalog_cache = list(LISTING)
    provider._model_name = model
    return provider


# ---------------------------------------------------------------- base


def test_a_decisions_only_set_gets_no_text_floor():
    assert normalise_output_modalities(["decisions"]) == {MODALITY_DECISIONS}
    assert normalise_output_modalities([" Decisions "]) == {MODALITY_DECISIONS}


def test_every_other_set_keeps_the_text_floor():
    assert normalise_output_modalities([]) == {MODALITY_TEXT}
    assert normalise_output_modalities(["audio"]) == {"text", "audio"}
    assert normalise_output_modalities(["decisions", "audio"]) == {
        "text", "decisions", "audio",
    }


def test_is_decisions_only_means_exactly_decisions():
    assert is_decisions_only_set({"decisions"})
    assert not is_decisions_only_set({"text", "decisions"})
    assert not is_decisions_only_set({"text"})
    assert not is_decisions_only_set(set())


def test_the_mixin_knob_can_declare_a_decision_model():
    class Knobbed(ModalityCapabilityMixin):
        _output_modalities_knob = ["decisions"]

    provider = Knobbed()
    assert provider.output_modalities() == {MODALITY_DECISIONS}
    assert provider.is_decisions_only()
    assert provider.supports_output_modality("decisions")
    assert not provider.supports_output_modality("text")
    # Input is untouched: decisions is never an input modality.
    assert provider.modalities() == {MODALITY_TEXT}


def test_the_default_mixin_is_not_decisions_only():
    assert not ModalityCapabilityMixin().is_decisions_only()


# ---------------------------------------------------------- openrouter


def test_an_unlisted_decision_model_is_found_through_its_endpoints_doc(served):
    provider = _provider("typesafe/jev-1.13")
    assert provider.output_modalities() == {MODALITY_DECISIONS}
    assert provider.is_decisions_only()
    assert provider.modalities() == {MODALITY_TEXT}
    assert provider._lookup_context_length("typesafe/jev-1.13") == 32000


def test_the_model_argument_is_honoured(served):
    provider = _provider("openai/gpt-4o")
    assert not provider.is_decisions_only()
    assert provider.is_decisions_only("typesafe/jev-1.13")


def test_the_endpoints_doc_is_fetched_once(served):
    provider = _provider("typesafe/jev-1.13")
    provider.output_modalities()
    provider.modalities()
    provider._lookup_context_length("typesafe/jev-1.13")
    assert [u for u in served if "jev-1.13" in u] == [
        "https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints",
    ]


def test_a_failed_doc_fetch_is_not_cached(served):
    provider = _provider("acme/unknown")
    assert provider.output_modalities() == {MODALITY_TEXT}
    provider.output_modalities()
    assert len([u for u in served if "acme/unknown" in u]) == 2


def test_a_listed_model_never_fetches_the_doc(served):
    provider = _provider("openai/gpt-4o")
    assert provider.output_modalities() == {MODALITY_TEXT}
    assert served == []


def test_the_listing_answers_for_audio_without_a_knob(served):
    provider = _provider("openai/gpt-audio-mini")
    assert provider.output_modalities() == {"text", "audio"}
    assert provider.supports_output_modality("audio")


def test_the_catalog_outranks_the_knob(served):
    provider = _provider("openai/gpt-4o")
    provider._output_modalities_knob = ["audio"]
    assert provider.output_modalities() == {MODALITY_TEXT}


def test_the_knob_answers_when_the_catalog_does_not(served):
    provider = _provider("acme/no-architecture")
    provider._output_modalities_knob = ["audio"]
    assert provider.output_modalities() == {"text", "audio"}
    provider._output_modalities_knob = ["decisions"]
    assert provider.is_decisions_only()
