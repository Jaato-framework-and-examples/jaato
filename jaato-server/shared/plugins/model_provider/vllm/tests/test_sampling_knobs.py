"""vLLM provider sampling-determinism knobs threaded from the NESTED api_params layer.

A profile's `plugin_configs.vllm.api_params.temperature: 0.0` arrives as
`config.extra["api_params"]["temperature"]` (NESTED — NOT flat
`config.extra["temperature"]`). The OpenAI-compat base's `_read_api_params`
parses the nested layer into `self._api_params`; vLLM's custom `complete()` must
EMIT it. Pre-fix vLLM never emitted `self._api_params`, so the knob was dropped
and the stage ran at vLLM's ~1.0 default (the determinism bug).

The 0.0 case is load-bearing: it's falsy, so a truthiness check (or a flat-layer
read that never finds it) silently drops the determinism knob. The base's
membership-copy is falsy-safe by construction.
"""
from unittest.mock import MagicMock

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.vllm.provider import VLLMProvider


def _make_provider_with_extra(extra: dict) -> VLLMProvider:
    provider = VLLMProvider()
    provider._verify_connectivity = lambda *a, **k: None
    provider._trace = lambda _msg: None
    cfg = ProviderConfig(
        api_key="EMPTY",
        extra={"host": "http://test.invalid:8000", "context_length": 32768, **extra},
    )
    provider.initialize(cfg)
    return provider


def _capture_create_kwargs(provider) -> dict:
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        m = MagicMock()
        m.choices = []
        m.usage = None
        return m

    provider._client = MagicMock()
    provider._client.chat.completions.create = fake_create
    provider._model_name = "Qwen/Qwen3-14B"
    try:
        provider.complete(messages=[], system_instruction="sys")
    except Exception:
        pass
    return captured


class TestSamplingKnobsFromNestedApiParams:
    def test_temperature_zero_from_nested_api_params_reaches_wire(self):
        # THE load-bearing case: profile api_params.temperature=0.0 must reach create().
        p = _make_provider_with_extra(
            {"api_params": {"temperature": 0.0, "top_p": 0.95, "seed": 7}})
        kw = _capture_create_kwargs(p)
        assert kw.get("temperature") == 0.0   # greedy reaches the wire (was dropped)
        assert kw.get("top_p") == 0.95
        assert kw.get("seed") == 7

    def test_other_sampling_knobs_flow_too(self):
        p = _make_provider_with_extra(
            {"api_params": {"frequency_penalty": 0.5, "presence_penalty": 0.0}})
        kw = _capture_create_kwargs(p)
        assert kw.get("frequency_penalty") == 0.5
        assert kw.get("presence_penalty") == 0.0   # 0.0 survives

    def test_no_api_params_means_no_sampling_kwargs(self):
        p = _make_provider_with_extra({})
        kw = _capture_create_kwargs(p)
        assert "temperature" not in kw and "top_p" not in kw and "seed" not in kw

    def test_vllm_top_level_max_tokens_not_double_set_by_api_params_loop(self):
        # max_tokens stays vLLM's top_level knob; the api_params loop excludes it.
        p = _make_provider_with_extra(
            {"max_tokens": 2048, "api_params": {"temperature": 0.2}})
        kw = _capture_create_kwargs(p)
        assert kw.get("max_tokens") == 2048
        assert kw.get("temperature") == 0.2
