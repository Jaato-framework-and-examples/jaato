"""Tests for the GitHub Models provider's ``api_params`` sampling knobs.

Audit follow-up to the vLLM temperature bug: a profile's
``plugin_configs.github_models.api_params.temperature`` arrives NESTED at
``config.extra["api_params"]["temperature"]``.  The provider must read
from that nested layer (NOT the flat ``config.extra["temperature"]``) and
thread it into the Azure SDK ``complete()`` kwargs (and the Copilot
``complete*`` call sites) so it reaches the wire.

The ``temperature == 0.0`` case is load-bearing: 0.0 is falsy, so a
truthiness check would silently drop the determinism knob.
"""

from unittest.mock import MagicMock, patch

from ..provider import GitHubModelsProvider
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import Message, Role
from .test_provider import create_mock_response


def _make_provider_with_extra(mock_client_class, extra: dict) -> GitHubModelsProvider:
    """Initialize + connect a GitHub Models provider (Azure PAT path) with the
    given ``config.extra``, using the patched ChatCompletionsClient."""
    mock_client = MagicMock()
    mock_client.complete.return_value = create_mock_response(text="ok")
    mock_client_class.return_value = mock_client

    provider = GitHubModelsProvider()
    provider.initialize(ProviderConfig(api_key="ghp_test", extra=extra))
    provider.connect("openai/gpt-4o")
    return provider


class TestSamplingKnobs:
    @patch("azure.ai.inference.ChatCompletionsClient")
    def test_default_is_none(self, mock_client_class):
        provider = _make_provider_with_extra(mock_client_class, {})
        assert provider._temperature is None
        assert provider._top_p is None
        assert provider._seed is None
        assert provider._max_tokens_override is None

    @patch("azure.ai.inference.ChatCompletionsClient")
    def test_temperature_zero_reaches_attr(self, mock_client_class):
        provider = _make_provider_with_extra(
            mock_client_class, {"api_params": {"temperature": 0.0}}
        )
        assert provider._temperature == 0.0

    @patch("azure.ai.inference.ChatCompletionsClient")
    def test_all_knobs_read_from_nested_layer(self, mock_client_class):
        provider = _make_provider_with_extra(
            mock_client_class,
            {
                "api_params": {
                    "temperature": 0.0,
                    "top_p": 0.85,
                    "seed": 9,
                    "max_tokens": 512,
                }
            },
        )
        assert provider._temperature == 0.0
        assert provider._top_p == 0.85
        assert provider._seed == 9
        assert provider._max_tokens_override == 512

    @patch("azure.ai.inference.ChatCompletionsClient")
    def test_flat_layer_is_ignored(self, mock_client_class):
        provider = _make_provider_with_extra(
            mock_client_class, {"temperature": 0.0}
        )
        assert provider._temperature is None

    @patch("azure.ai.inference.ChatCompletionsClient")
    def test_temperature_zero_reaches_azure_complete(self, mock_client_class):
        """End-to-end: temperature=0.0 must reach the Azure SDK complete()
        kwargs (the batch path)."""
        provider = _make_provider_with_extra(
            mock_client_class,
            {"api_params": {"temperature": 0.0, "top_p": 0.85, "seed": 9}},
        )
        # Reset the call recorded during _verify_connectivity at init.
        provider._client.complete.reset_mock()
        provider._client.complete.return_value = create_mock_response(text="ok")

        provider.complete(messages=[Message.from_text(Role.USER, "Hi")])

        kwargs = provider._client.complete.call_args.kwargs
        assert kwargs.get("temperature") == 0.0
        assert kwargs.get("top_p") == 0.85
        assert kwargs.get("seed") == 9
