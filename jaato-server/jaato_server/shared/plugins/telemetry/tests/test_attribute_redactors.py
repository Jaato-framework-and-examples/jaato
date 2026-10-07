"""Tests for the telemetry span-attribute redactor plug-in surface.

Seat 4 of the four-seat pseudonymization design — see
``project_backlog_pseudonymization_plugin_surface.md`` and
``jaato-premium/docs/design/pseudonymization-four-seat.md``.

The redactor chain runs in :class:`_SpanWrapper.set_attribute` and is
threaded through to every other set-attribute method on the wrapper
(set_input_messages, set_output_messages, set_metadata) so a single
registration covers every attribute path uniformly.
"""

from typing import Any, List, Tuple
from unittest.mock import MagicMock

from ..null_plugin import NullTelemetryPlugin
from ..otel_plugin import _SpanWrapper


def _wrap(redactors=None, redact_content: bool = False):
    """Build a _SpanWrapper around a MagicMock span for assertion."""
    span = MagicMock()
    return span, _SpanWrapper(span, redact_content, redactors or [])


# ─────────────────────────────────────────────────────────────────────


class TestAttributeRedactorChain:

    def test_unset_redactors_pass_value_unchanged(self):
        span, w = _wrap()
        w.set_attribute("custom.key", "raw")
        span.set_attribute.assert_called_once_with("custom.key", "raw")

    def test_single_redactor_runs(self):
        captured: List[Tuple[str, Any]] = []

        def redact(key: str, value: Any) -> Any:
            captured.append((key, value))
            return f"redacted:{value}"

        span, w = _wrap(redactors=[redact])
        w.set_attribute("custom.key", "raw")
        assert captured == [("custom.key", "raw")]
        span.set_attribute.assert_called_once_with("custom.key", "redacted:raw")

    def test_multiple_redactors_chain_in_registration_order(self):
        order: list = []

        def first(key, value):
            order.append("first")
            return f"[1]{value}"

        def second(key, value):
            order.append("second")
            assert value.startswith("[1]")  # second sees first's output
            return f"[2]{value}"

        span, w = _wrap(redactors=[first, second])
        w.set_attribute("k", "x")
        assert order == ["first", "second"]
        span.set_attribute.assert_called_once_with("k", "[2][1]x")

    def test_redactor_runs_after_built_in_sensitive_redaction(self):
        """Built-in length-preserving redaction for input.value /
        output.value runs first (when redact_content=True), then the
        registered redactor sees the already-redacted form."""
        seen: list = []

        def capture(key, value):
            seen.append((key, value))
            return value

        span, w = _wrap(redactors=[capture], redact_content=True)
        w.set_attribute("input.value", "the actual prompt")
        # Redactor saw the already-redacted form, not the raw prompt
        assert seen[0][0] == "input.value"
        assert "[REDACTED" in seen[0][1]
        assert "the actual prompt" not in seen[0][1]


class TestRedactorAppliesToAllAttributePaths:
    """Single chokepoint claim: redactors fire on every attribute set,
    including paths through set_input_messages / set_output_messages /
    set_metadata that previously bypassed the wrapper's own
    set_attribute."""

    def _capture_setup(self):
        captured: List[Tuple[str, Any]] = []

        def capture(key, value):
            captured.append((key, value))
            return value

        return _wrap(redactors=[capture]), captured

    def test_set_input_messages_routes_through_redactor(self):
        (span, w), captured = self._capture_setup()
        w.set_input_messages([
            {"role": "user", "content": "hello"},
        ])
        # Both attributes set via the message-flatten path went
        # through the redactor.
        keys = [k for k, _ in captured]
        assert "llm.input_messages.0.message.role" in keys
        assert "llm.input_messages.0.message.content" in keys

    def test_set_output_messages_routes_through_redactor(self):
        (span, w), captured = self._capture_setup()
        w.set_output_messages([{
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"name": "do_thing", "arguments": '{"x":1}'}],
        }])
        keys = [k for k, _ in captured]
        assert any("output_messages.0.message.role" in k for k in keys)
        assert any("output_messages.0.message.content" in k for k in keys)
        assert any("tool_calls.0.tool_call.function.name" in k for k in keys)
        assert any("tool_calls.0.tool_call.function.arguments" in k for k in keys)

    def test_set_metadata_routes_through_redactor(self):
        (span, w), captured = self._capture_setup()
        w.set_metadata({"foo": "bar"})
        assert captured[0][0] == "metadata"
        # set_metadata serialises to JSON; redactor sees the JSON string
        assert '"foo"' in captured[0][1]

    def test_redactor_can_rewrite_message_content(self):
        """End-to-end: a content-rewriting redactor catches every
        attribute path, including the message-flatten ones."""
        def rewrite(key, value):
            if isinstance(value, str) and "alice@example.com" in value:
                return value.replace("alice@example.com", "<EMAIL_1>")
            return value

        span, w = _wrap(redactors=[rewrite])
        w.set_input_messages([
            {"role": "user", "content": "email me at alice@example.com"},
        ])
        # The actual span saw the rewritten value
        recorded_values = [
            call.args[1] for call in span.set_attribute.call_args_list
        ]
        # Find the content attribute
        content_calls = [
            v for k, v in zip(
                [c.args[0] for c in span.set_attribute.call_args_list],
                recorded_values,
            ) if "content" in k
        ]
        assert any("<EMAIL_1>" in v for v in content_calls)
        assert not any("alice@example.com" in v for v in content_calls)


class TestNullPluginIgnoresRegistration:
    """NullTelemetryPlugin accepts the registration call (so consumers
    can register unconditionally) and ignores it (no spans produced)."""

    def test_register_is_noop(self):
        plugin = NullTelemetryPlugin()
        # Should not raise
        plugin.register_attribute_redactor(lambda key, value: "X")
        # No spans to test on; assertion is just that the call succeeded.
