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


def _stub_session(**overrides) -> SimpleNamespace:
    """Minimal stand-in for `self`.

    Defaults exercise the no-pricing-table path: ``_model_name`` set but the
    lazily-loaded ``_span_pricing`` already marked loaded and empty, so the
    pricing fallback resolves to None without touching disk. Override
    ``_span_pricing`` / ``_model_name`` / ``workspace_path`` per test.
    """
    stub = SimpleNamespace(
        _current_turn_span=None,
        _trace=lambda *a, **k: None,
        _model_name="test-model",
        _span_pricing=None,
        _span_pricing_loaded=True,  # empty table; no disk read
        workspace_path=None,
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    # Bind the real resolution logic to the stub so _record_token_telemetry's
    # self._resolve_span_cost(...) call exercises the actual precedence code.
    stub._resolve_span_cost = lambda usage: JaatoSession._resolve_span_cost(stub, usage)
    return stub


def _record(span, response, session=None) -> None:
    """Invoke the unbound method against a lightweight stub self."""
    JaatoSession._record_token_telemetry(session or _stub_session(), span, response)


class _FakePricing:
    """Stands in for shared.pricing.PricingTable."""

    def __init__(self, model, cost):
        self._model = model
        self._cost = cost

    def has(self, model_name):
        return model_name == self._model

    def cost_for_usage(self, model_name, **kwargs):
        return self._cost if model_name == self._model else None


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

    def test_pricing_table_used_when_provider_omits_cost(self):
        # Provider reports no cost, but the operator pricing table has an
        # entry for the model → the computed cost is stamped on the span.
        span = _FakeSpan()
        session = _stub_session(
            _model_name="priced-model",
            _span_pricing=_FakePricing("priced-model", 0.0042),
            _span_pricing_loaded=True,
        )
        response = ProviderResponse(
            parts=[],
            usage=TokenUsage(prompt_tokens=1000, output_tokens=100, cost_usd=None),
            finish_reason=FinishReason.STOP,
        )
        _record(span, response, session)
        assert span.attributes["gen_ai.usage.cost"] == 0.0042
        assert span.attributes["llm.cost.total"] == 0.0042

    def test_provider_cost_wins_over_pricing_table(self):
        # Provider-reported cost is fiscal truth — it beats any table estimate.
        span = _FakeSpan()
        session = _stub_session(
            _model_name="priced-model",
            _span_pricing=_FakePricing("priced-model", 0.0042),
            _span_pricing_loaded=True,
        )
        response = ProviderResponse(
            parts=[],
            usage=TokenUsage(prompt_tokens=1000, output_tokens=100, cost_usd=0.0099),
            finish_reason=FinishReason.STOP,
        )
        _record(span, response, session)
        assert span.attributes["gen_ai.usage.cost"] == 0.0099
        assert span.attributes["llm.cost.total"] == 0.0099

    def test_no_cost_when_model_absent_from_pricing_table(self):
        # Table present but no entry for this model → no cost stamped.
        span = _FakeSpan()
        session = _stub_session(
            _model_name="unlisted-model",
            _span_pricing=_FakePricing("some-other-model", 0.0042),
            _span_pricing_loaded=True,
        )
        response = ProviderResponse(
            parts=[],
            usage=TokenUsage(prompt_tokens=1000, output_tokens=100, cost_usd=None),
            finish_reason=FinishReason.STOP,
        )
        _record(span, response, session)
        assert "gen_ai.usage.cost" not in span.attributes
        assert "llm.cost.total" not in span.attributes
