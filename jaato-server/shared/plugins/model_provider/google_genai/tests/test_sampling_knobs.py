"""Tests for the Google GenAI provider's ``api_params`` sampling knobs.

Audit follow-up to the vLLM temperature bug: a profile's
``plugin_configs.google_genai.api_params.temperature`` arrives NESTED at
``config.extra["api_params"]["temperature"]``.  The provider must read
from that nested layer (NOT the flat ``config.extra["temperature"]``) and
thread it into the ``GenerateContentConfig`` built in ``complete()`` so it
reaches the wire.

The ``temperature == 0.0`` case is load-bearing: 0.0 is falsy, so a
truthiness check would silently drop the determinism knob.
"""

from unittest.mock import MagicMock, patch

from ..provider import GoogleGenAIProvider
from shared.plugins.model_provider.base import ProviderConfig


def _make_provider_with_extra(extra: dict) -> GoogleGenAIProvider:
    """Initialize a Google GenAI provider with the given ``config.extra``,
    mocking the genai client so no network is hit."""
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.models.list.return_value = [MagicMock()]
        mock_client_class.return_value = mock_client
        provider = GoogleGenAIProvider()
        provider.initialize(
            ProviderConfig(
                api_key="test-key",
                use_vertex_ai=False,
                auth_method="api_key",
                extra=extra,
            )
        )
    return provider


class TestSamplingKnobs:
    def test_default_is_none(self):
        provider = _make_provider_with_extra({})
        assert provider._temperature is None
        assert provider._top_p is None
        assert provider._top_k is None
        assert provider._seed is None
        assert provider._max_output_tokens is None

    def test_temperature_zero_reaches_attr(self):
        provider = _make_provider_with_extra(
            {"api_params": {"temperature": 0.0}}
        )
        assert provider._temperature == 0.0

    def test_all_knobs_read_from_nested_layer(self):
        provider = _make_provider_with_extra(
            {
                "api_params": {
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "top_k": 32,
                    "seed": 7,
                    "max_output_tokens": 2048,
                }
            }
        )
        assert provider._temperature == 0.0
        assert provider._top_p == 0.9
        assert provider._top_k == 32
        assert provider._seed == 7
        assert provider._max_output_tokens == 2048

    def test_flat_layer_is_ignored(self):
        provider = _make_provider_with_extra({"temperature": 0.0})
        assert provider._temperature is None

    def test_temperature_zero_reaches_generation_config(self):
        """End-to-end: temperature=0.0 must reach the GenerateContentConfig
        passed to generate_content() (the non-cache batch path)."""
        provider = _make_provider_with_extra(
            {"api_params": {"temperature": 0.0, "top_p": 0.5, "seed": 3}}
        )
        captured: dict = {}

        def fake_generate_content(model, contents, config):
            captured["config"] = config
            resp = MagicMock()
            resp.candidates = []
            resp.usage_metadata = None
            return resp

        provider._client = MagicMock()
        provider._client.models.generate_content = fake_generate_content
        provider._model_name = "gemini-2.5-flash"
        provider._cache_plugin = None

        try:
            provider.complete(messages=[], system_instruction="sys")
        except Exception:
            # We only care about what reached the GenerateContentConfig.
            pass

        config = captured.get("config")
        assert config is not None
        assert config.temperature == 0.0
        assert config.top_p == 0.5
        assert config.seed == 3
