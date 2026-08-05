"""Tests for the Antigravity provider's ``api_params`` sampling knobs.

Audit follow-up to the vLLM temperature bug: a profile's
``plugin_configs.antigravity.api_params.temperature`` arrives NESTED at
``config.extra["api_params"]["temperature"]``.  The provider must read
from that nested layer (NOT the flat ``config.extra["temperature"]``) and
thread it through ``_build_generation_config()`` →
``build_generation_config()`` so it reaches the request's
``generationConfig``.

The ``temperature == 0.0`` case is load-bearing: 0.0 is falsy, so a
truthiness check would silently drop the determinism knob.
"""

from unittest.mock import patch

from shared.plugins.model_provider.antigravity import AntigravityProvider
from shared.plugins.model_provider.antigravity.oauth import (
    Account,
    AccountManager,
    OAuthTokens,
)
from shared.plugins.model_provider.base import ProviderConfig


def _make_connected_provider(extra: dict) -> AntigravityProvider:
    """Initialize + connect an Antigravity provider with the given
    ``config.extra``, mocking OAuth account loading so no network is hit."""
    tokens = OAuthTokens(
        access_token="test_token",
        refresh_token="test_refresh",
        expires_at=9999999999,
        email="test@example.com",
    )
    account = Account(email="test@example.com", tokens=tokens)
    manager = AccountManager(accounts=[account])

    provider = AntigravityProvider()
    with patch(
        "shared.plugins.model_provider.antigravity.provider.load_accounts"
    ) as mock_load:
        mock_load.return_value = manager
        provider.initialize(ProviderConfig(extra=extra))
    provider.connect("antigravity-gemini-3-flash")
    return provider


class TestSamplingKnobs:
    def test_default_is_none(self):
        provider = _make_connected_provider({})
        assert provider._temperature is None
        assert provider._top_p is None
        assert provider._top_k is None
        assert provider._seed is None

    def test_temperature_zero_reaches_attr(self):
        provider = _make_connected_provider(
            {"api_params": {"temperature": 0.0}}
        )
        assert provider._temperature == 0.0

    def test_all_knobs_read_from_nested_layer(self):
        provider = _make_connected_provider(
            {
                "api_params": {
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "top_k": 20,
                    "seed": 11,
                }
            }
        )
        assert provider._temperature == 0.0
        assert provider._top_p == 0.8
        assert provider._top_k == 20
        assert provider._seed == 11

    def test_flat_layer_is_ignored(self):
        provider = _make_connected_provider({"temperature": 0.0})
        assert provider._temperature is None

    def test_temperature_zero_reaches_generation_config(self):
        """End-to-end: temperature=0.0 must reach the generationConfig dict
        built by _build_generation_config()."""
        provider = _make_connected_provider(
            {"api_params": {"temperature": 0.0, "top_p": 0.8, "seed": 11}}
        )
        gen_config = provider._build_generation_config()
        assert gen_config["temperature"] == 0.0
        assert gen_config["topP"] == 0.8
        assert gen_config["seed"] == 11
