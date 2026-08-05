"""Unit tests for cost stamping in JaatoSession._record_token_telemetry.

These pin the behavior added for observability-backend cost ingestion:
when a provider reports a per-call cost (``TokenUsage.cost_usd``), the LLM
span carries both ``gen_ai.usage.cost`` (Langfuse OTLP cost ingestion) and
``llm.cost.total`` (OpenInference / Arize Phoenix). When the provider does
not report a cost, neither attribute is set (cost is resolved downstream at
the daemon boundary or computed by the backend from model + token counts).

The method is exercised via a fake span so the test stays decoupled from the
OpenTelemetry SDK — it only asserts which attributes the session writes.
"""

from __future__ import annotations

from types import SimpleNamespace

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    ProviderResponse,
    TokenUsage,
)

from shared.jaato_session import JaatoSession


class _FakeSpan:
    """Records set_attribute / set_output_messages calls for assertions."""

    def __init__(self) -> None:
        self.attributes: dict = {}
        self.output_messages = None

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def set_output_messages(self, messages) -> None:
        self.output_messages = messages


def _stub_session() -> SimpleNamespace:
    """Minimal stand-in for `self` — the method only touches these two."""
    return SimpleNamespace(_current_turn_span=None, _trace=lambda *a, **k: None)


def _record(span, response) -> None:
    """Invoke the unbound method against a lightweight stub self."""
    JaatoSession._record_token_telemetry(_stub_session(), span, response)


class TestCostStamping:
    def test_provider_cost_sets_both_cost_attributes(self):
        span = _FakeSpan()
        response = ProviderResponse(
            parts=[],
            usage=TokenUsage(prompt_tokens=1500, output_tokens=200, cost_usd=0.0123),
            finish_reason=FinishReason.STOP,
        )
        _record(span, response)
        assert span.attributes["gen_ai.usage.cost"] == 0.0123
        assert span.attributes["llm.cost.total"] == 0.0123

    def test_no_cost_when_provider_omits_it(self):
        span = _FakeSpan()
        response = ProviderResponse(
            parts=[],
            usage=TokenUsage(prompt_tokens=1500, output_tokens=200, cost_usd=None),
            finish_reason=FinishReason.STOP,
        )
        _record(span, response)
        assert "gen_ai.usage.cost" not in span.attributes
        assert "llm.cost.total" not in span.attributes
        # Token counts are still recorded regardless of cost availability.
        assert span.attributes["llm.token_count.prompt"] == 1500
        assert span.attributes["llm.token_count.completion"] == 200

    def test_zero_cost_is_recorded(self):
        # cost_usd == 0.0 is a legitimate reported value (e.g. free-tier /
        # fully-cached turns) and must be distinguished from None.
        span = _FakeSpan()
        response = ProviderResponse(
            parts=[],
            usage=TokenUsage(prompt_tokens=10, output_tokens=5, cost_usd=0.0),
            finish_reason=FinishReason.STOP,
        )
        _record(span, response)
        assert span.attributes["gen_ai.usage.cost"] == 0.0
        assert span.attributes["llm.cost.total"] == 0.0
