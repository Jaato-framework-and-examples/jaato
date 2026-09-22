"""Guards for issue #766: ``finish_reason="error"`` must name its cause.

OpenRouter reports a mid-stream upstream failure in two shapes, and
only one of them used to survive contact with the framework:

============================================  ==========================
shape                                          old outcome
============================================  ==========================
top-level ``error`` object + ``"error"``       message preserved,
finish reason                                  **retryable**
``finish_reason: "error"`` alone, cause in     ``FinishReason.ERROR``,
``native_finish_reason``                       no message, **terminal**
============================================  ==========================

Shape 2 travelled back as an ordinary ``ProviderResponse`` whose finish
reason was ``ERROR``; ``JaatoSession._unwrap_turn_result`` then raised
``RuntimeError(turn_result.error_message or "Provider returned an
error")``, and because nothing on that path populates ``error_message``
the ``or`` always fired.  Eleven sweep arms died with that one string —
six of them indistinguishable from each other for days — while the
diagnosis (Gemini's ``MALFORMED_FUNCTION_CALL``: a function call the
model's own serialiser rejected) sat unread in a field the framework
had already parsed for #745.

Both halves are guarded here:

1.  **Diagnosis.**  The raised error names the native reason.  Revert
    the fix and the tests below see ``Provider returned an error``
    instead — the message that started the investigation.
2.  **Symmetry.**  The error is an :class:`InfrastructureError`, so
    ``classify_error`` routes it to ``with_retry`` exactly as shape 1
    already was.  The same upstream condition must not be retryable or
    fatal depending on which shape carried it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import FinishReason

from ..converters import read_response_native_finish_reason
from ..errors import InfrastructureError, OpenRouterError, UpstreamFinishError
from ..provider import OpenRouterProvider


# ==================== Fixtures ====================


def _make_chunk(
    *,
    content=None,
    finish_reason=None,
    native_finish_reason=None,
    native_in_model_extra=False,
    error=None,
):
    """Build a streaming chunk with an explicit native finish reason.

    ``native_in_model_extra`` puts the reason where the OpenAI SDK
    actually lands it on a real response (Pydantic extras) rather than
    as a declared attribute, so both read paths in
    ``read_native_finish_reason`` are exercised.
    """
    chunk = MagicMock()
    chunk.error = error
    chunk.model_extra = None
    chunk.usage = None

    choice = MagicMock()
    choice.finish_reason = finish_reason
    if native_in_model_extra:
        choice.native_finish_reason = None
        choice.model_extra = {"native_finish_reason": native_finish_reason}
    else:
        choice.native_finish_reason = native_finish_reason
        choice.model_extra = None

    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    delta.reasoning = None
    delta.reasoning_content = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _make_stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    stream.response = None
    return stream


def _build_provider(generation_id=None):
    provider = OpenRouterProvider()
    provider._client = MagicMock()
    provider._model_name = "google/gemini-2.5-flash"
    provider._enable_thinking = False
    provider._last_generation_id = generation_id
    return provider


def _stream(chunks, *, provider=None):
    """Run ``_stream_response`` over *chunks*, returning the response."""
    provider = provider or _build_provider()
    provider._client.chat.completions.create = lambda **kw: _make_stream(chunks)
    return provider._stream_response(
        messages=[],
        kwargs={},
        on_chunk=lambda _t: None,
    )


# ==================== The observed failure ====================


class TestBareErrorFinishNamesItsCause:
    """The exact wire shape recorded for generation ``gen-1788256220-...``.

    Three chunks, no top-level ``error`` anywhere in the stream, and a
    final choice carrying ``finish_reason="error"`` with the cause in
    ``native_finish_reason``.
    """

    def test_raises_naming_the_native_reason(self):
        chunks = [
            _make_chunk(content="I'll replace the entire content of "),
            _make_chunk(content="explain.py now."),
            _make_chunk(
                content="",
                finish_reason="error",
                native_finish_reason="MALFORMED_FUNCTION_CALL",
            ),
        ]

        with pytest.raises(UpstreamFinishError) as exc_info:
            _stream(chunks)

        assert "MALFORMED_FUNCTION_CALL" in str(exc_info.value)
        # The string that made six failures indistinguishable must not
        # be what an operator ends up reading.
        assert "Provider returned an error" not in str(exc_info.value)

    def test_native_reason_read_from_pydantic_extras(self):
        """A real SDK response lands the field in ``model_extra``."""
        chunks = [
            _make_chunk(
                content="",
                finish_reason="error",
                native_finish_reason="MALFORMED_FUNCTION_CALL",
                native_in_model_extra=True,
            ),
        ]

        with pytest.raises(UpstreamFinishError) as exc_info:
            _stream(chunks)

        assert "MALFORMED_FUNCTION_CALL" in str(exc_info.value)

    def test_generation_id_rides_along(self):
        """What makes an out-of-band OpenRouter lookup possible."""
        provider = _build_provider(generation_id="gen-1788256220-ONdq9ZX3n09AD1OCfYIu")
        chunks = [
            _make_chunk(
                content="",
                finish_reason="error",
                native_finish_reason="MALFORMED_FUNCTION_CALL",
            ),
        ]

        with pytest.raises(UpstreamFinishError) as exc_info:
            _stream(chunks, provider=provider)

        assert "gen-1788256220-ONdq9ZX3n09AD1OCfYIu" in str(exc_info.value)
        assert "google/gemini-2.5-flash" in str(exc_info.value)

    def test_stream_is_still_closed(self):
        """Teardown is not skipped by the new raise."""
        provider = _build_provider()
        captured = {}

        def capture_create(**kwargs):
            captured["stream"] = _make_stream(
                [
                    _make_chunk(
                        content="",
                        finish_reason="error",
                        native_finish_reason="MALFORMED_FUNCTION_CALL",
                    ),
                ]
            )
            return captured["stream"]

        provider._client.chat.completions.create = capture_create

        with pytest.raises(UpstreamFinishError):
            provider._stream_response(
                messages=[], kwargs={}, on_chunk=lambda _t: None,
            )

        captured["stream"].close.assert_called_once()

    def test_absent_native_reason_still_raises_but_says_so(self):
        """No diagnosis available is not the same as no failure."""
        chunks = [_make_chunk(content="", finish_reason="error")]

        with pytest.raises(UpstreamFinishError) as exc_info:
            _stream(chunks)

        assert "no native_finish_reason" in str(exc_info.value)


# ==================== Retry symmetry with shape 1 ====================


class TestRetrySymmetry:
    """The same upstream condition, retryable in either shape."""

    def test_is_an_infrastructure_error(self):
        assert issubclass(UpstreamFinishError, InfrastructureError)
        assert issubclass(UpstreamFinishError, OpenRouterError)

    def test_classified_transient(self):
        provider = _build_provider()
        assert provider.classify_error(
            UpstreamFinishError("MALFORMED_FUNCTION_CALL")
        ) == {"transient": True, "rate_limit": False, "infra": True}

    def test_handle_api_error_passes_it_through(self):
        """``_handle_api_error`` must not remap one of our own errors."""
        provider = _build_provider()
        exc = UpstreamFinishError("MALFORMED_FUNCTION_CALL")
        # Returns without raising anything else; the caller re-raises.
        assert provider._handle_api_error(exc) is None


# ==================== Outcomes that must not change ====================


class TestUnaffectedOutcomes:
    """A native reason is read on every turn; only ``error`` raises."""

    def test_unexpected_tool_call_does_not_raise(self):
        """The adjacency from the issue's second report.

        ``native_finish_reason="UNEXPECTED_TOOL_CALL"`` arrived with a
        normalised ``tool_calls`` one generation before a failure.  It
        maps to ``TOOL_USE`` and must keep doing so — surfacing native
        reasons is a diagnostic, not a new failure mode.
        """
        chunks = [
            _make_chunk(
                content="calling a tool",
                finish_reason="tool_calls",
                native_finish_reason="UNEXPECTED_TOOL_CALL",
            ),
        ]

        response = _stream(chunks)
        assert response.finish_reason == FinishReason.TOOL_USE

    def test_normal_stop_unaffected(self):
        chunks = [
            _make_chunk(content="all done"),
            _make_chunk(content="", finish_reason="stop"),
        ]

        response = _stream(chunks)
        assert response.finish_reason == FinishReason.STOP
        assert response.get_text() == "all done"

    def test_shape_one_still_reports_the_upstream_message(self):
        """Shape 1 keeps its own, richer diagnosis.

        A top-level ``error`` is read before the finish reason is ever
        resolved, so the message the upstream actually wrote wins over
        the generic native-reason phrasing.
        """
        chunks = [
            _make_chunk(content="partial "),
            _make_chunk(
                content="",
                finish_reason="error",
                error={"code": "server_error", "message": "Provider disconnected"},
            ),
        ]

        with pytest.raises(InfrastructureError) as exc_info:
            _stream(chunks)

        assert "Provider disconnected" in str(exc_info.value)
        assert not isinstance(exc_info.value, UpstreamFinishError)


# ==================== The non-streamed path ====================


class TestBatchPath:
    """A non-streamed turn carries the shape too — it is the upstream's
    verdict, not an artefact of SSE."""

    @staticmethod
    def _batch_response(finish_reason, native_finish_reason=None):
        message = SimpleNamespace(content="", tool_calls=None, reasoning=None)
        choice = SimpleNamespace(
            message=message,
            finish_reason=finish_reason,
            native_finish_reason=native_finish_reason,
        )
        return SimpleNamespace(choices=[choice], usage=None)

    def test_reads_the_native_reason_off_a_response(self):
        response = self._batch_response("error", "MALFORMED_FUNCTION_CALL")
        assert (
            read_response_native_finish_reason(response)
            == "MALFORMED_FUNCTION_CALL"
        )

    def test_none_when_nothing_reported(self):
        assert read_response_native_finish_reason(None) is None
        assert read_response_native_finish_reason(
            SimpleNamespace(choices=[])
        ) is None
        assert read_response_native_finish_reason(
            self._batch_response("stop")
        ) is None

    def test_complete_raises_on_a_bare_error_finish(self):
        provider = _build_provider()
        provider._client.chat.completions.create = (
            lambda **kw: self._batch_response("error", "MALFORMED_FUNCTION_CALL")
        )

        with pytest.raises(UpstreamFinishError) as exc_info:
            provider.complete(messages=[])

        assert "MALFORMED_FUNCTION_CALL" in str(exc_info.value)

    def test_complete_unaffected_by_a_normal_finish(self):
        provider = _build_provider()
        provider._client.chat.completions.create = (
            lambda **kw: self._batch_response("stop")
        )

        result = provider.complete(messages=[])
        assert result.finish_reason == FinishReason.STOP
