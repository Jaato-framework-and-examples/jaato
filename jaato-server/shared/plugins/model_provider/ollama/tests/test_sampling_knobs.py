"""Tests for the Ollama provider's ``api_params`` sampling knobs.

Audit follow-up to the vLLM temperature bug: a profile's
``plugin_configs.ollama.api_params.temperature`` arrives NESTED at
``config.extra["api_params"]["temperature"]``.  The provider must read
from that nested layer (NOT the flat ``config.extra["temperature"]``)
into ``self._temperature`` etc., which the inherited
``AnthropicProvider.complete()`` emits onto ``messages.create()``.

The ``temperature == 0.0`` case is load-bearing: 0.0 is falsy, so a
truthiness check would silently drop the determinism knob.
"""

from unittest.mock import MagicMock, patch

from ..provider import OllamaProvider
from shared.plugins.model_provider.base import ProviderConfig


def _make_provider_with_extra(extra: dict) -> OllamaProvider:
    """Initialize an Ollama provider with the given ``config.extra`` dict,
    mocking the anthropic SDK + connectivity check so no network is hit."""
    with patch("httpx.get") as mock_get, patch("anthropic.Anthropic"):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"models": []}
        provider = OllamaProvider()
        provider.initialize(ProviderConfig(extra=extra))
    return provider


class TestSamplingKnobs:
    def test_default_is_none(self):
        """Without api_params, sampling attrs stay None (omitted on the wire)."""
        provider = _make_provider_with_extra({})
        assert provider._temperature is None
        assert provider._top_p is None
        assert provider._top_k is None
        assert provider._max_tokens_override is None

    def test_temperature_zero_reaches_attr(self):
        """The NESTED api_params.temperature: 0.0 must survive (falsy)."""
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
                    "top_k": 40,
                    "max_tokens": 1234,
                }
            }
        )
        assert provider._temperature == 0.0
        assert provider._top_p == 0.9
        assert provider._top_k == 40
        assert provider._max_tokens_override == 1234

    def test_flat_layer_is_ignored(self):
        """The knobs are nested; a flat config.extra["temperature"] is NOT
        the api_params layer and must not set self._temperature."""
        provider = _make_provider_with_extra({"temperature": 0.0})
        assert provider._temperature is None

    def test_temperature_zero_reaches_messages_create(self):
        """End-to-end: temperature=0.0 must reach messages.create() kwargs via
        the inherited AnthropicProvider.complete()."""
        provider = _make_provider_with_extra(
            {"api_params": {"temperature": 0.0}}
        )
        captured: dict = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.content = []
            resp.stop_reason = "end_turn"
            resp.usage = MagicMock(input_tokens=0, output_tokens=0)
            return resp

        provider._client = MagicMock()
        provider._client.messages.create = fake_create
        provider._model_name = "qwen3:32b"

        try:
            provider.complete(messages=[], system_instruction="sys")
        except Exception:
            # We only care about what reached the create() kwargs.
            pass

        assert captured.get("temperature") == 0.0
