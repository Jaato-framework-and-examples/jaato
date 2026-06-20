"""Tests for the vLLM provider's ``parallel_tool_calls`` profile knob.

When set via ``plugin_configs.vllm.parallel_tool_calls: false``,
the kwarg must propagate to ``chat.completions.create`` so vLLM
returns at most one tool call per response.  Closes the small-model
parallel-batching failure mode (qwen3-14b + hermes parser emitting
N readFiles + signal_completion in one assistant message).

Verified against vLLM stable docs 2026-06-09 via context7: server
contract is "parallel_tool_calls=false → only zero or one tool call
is returned" — response-layer enforcement, not just a generation
hint.
"""

from unittest.mock import MagicMock

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.vllm.provider import VLLMProvider


def _make_provider_with_extra(extra: dict) -> VLLMProvider:
    """Build an initialized provider with the given config.extra dict.

    Mocks ``_verify_connectivity`` so initialize doesn't hit the
    network.  Uses the canonical context_length override path so the
    no-context-length error doesn't fire.
    """
    provider = VLLMProvider()
    provider._verify_connectivity = lambda *a, **k: None
    provider._trace = lambda _msg: None
    cfg = ProviderConfig(
        api_key=extra.get("api_token", "EMPTY"),
        extra={"host": "http://test.invalid:8000", "context_length": 32768, **extra},
    )
    provider.initialize(cfg)
    return provider


class TestParallelToolCallsKnob:
    def test_default_is_none_kwarg_not_set(self):
        """Without the knob in profile, _parallel_tool_calls stays None
        and the kwarg is absent from the create() call."""
        provider = _make_provider_with_extra({})
        assert provider._parallel_tool_calls is None

    def test_knob_false_propagates(self):
        provider = _make_provider_with_extra({"parallel_tool_calls": False})
        assert provider._parallel_tool_calls is False

    def test_knob_true_propagates(self):
        provider = _make_provider_with_extra({"parallel_tool_calls": True})
        assert provider._parallel_tool_calls is True

    def test_knob_false_appears_in_kwargs_at_complete(self):
        """The kwarg must reach chat.completions.create when the knob
        is set.  Mocks the client call + asserts parallel_tool_calls
        appears in the kwargs dict."""
        provider = _make_provider_with_extra({"parallel_tool_calls": False})

        captured_kwargs: dict = {}

        def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            # Return a minimal mock so complete() can flow through.
            mock_response = MagicMock()
            mock_response.choices = []
            mock_response.usage = None
            return mock_response

        provider._client = MagicMock()
        provider._client.chat.completions.create = fake_create
        provider._model_name = "Qwen/Qwen3-14B"

        try:
            provider.complete(messages=[], system_instruction="sys")
        except Exception:
            # complete() may raise downstream of the API call (e.g. on
            # the empty mock response); we only care about what reached
            # the create() kwarg dict.
            pass

        assert captured_kwargs.get("parallel_tool_calls") is False

    def test_knob_unset_means_kwarg_absent(self):
        """When the knob is not in the profile, the kwarg must NOT
        appear in the create() call — let vLLM apply its own default."""
        provider = _make_provider_with_extra({})

        captured_kwargs: dict = {}

        def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = []
            mock_response.usage = None
            return mock_response

        provider._client = MagicMock()
        provider._client.chat.completions.create = fake_create
        provider._model_name = "Qwen/Qwen3-14B"

        try:
            provider.complete(messages=[], system_instruction="sys")
        except Exception:
            pass

        assert "parallel_tool_calls" not in captured_kwargs
