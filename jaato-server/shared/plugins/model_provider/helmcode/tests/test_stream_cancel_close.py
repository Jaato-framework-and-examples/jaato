"""Tests that the Helmcode provider explicitly closes the SDK ``Stream`` on
both cancellation and natural completion — mirror of
``nebius/tests/test_stream_cancel_close.py``.
"""

from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
)
from shared.plugins.model_provider.helmcode.provider import HelmcodeProvider


def _make_chunk(*, content=None, finish_reason=None, usage=None):
    chunk = MagicMock()
    chunk.usage = usage
    if content is None and finish_reason is None:
        chunk.choices = []
    else:
        choice = MagicMock()
        choice.finish_reason = finish_reason
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = None
        delta.reasoning_content = None
        choice.delta = delta
        chunk.choices = [choice]
    return chunk


def _make_stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    return stream


def _build_provider():
    provider = HelmcodeProvider()
    provider._client = MagicMock()
    provider._model_name = "deepseek-v4-flash"
    provider._enable_thinking = False
    provider._trace = lambda _msg: None
    return provider


class TestStreamCancellationClosesConnection:
    def test_cancel_closes_stream(self):
        provider = _build_provider()
        cancel = CancelToken()
        chunks = [
            _make_chunk(content="hello "),
            _make_chunk(content="world"),
            _make_chunk(content="!"),
        ]
        stream_ref = {}

        def capture_create(**kwargs):
            s = _make_stream(chunks)
            stream_ref["stream"] = s
            return s

        provider._client.chat.completions.create = capture_create

        collected = []

        def on_chunk(text: str) -> None:
            collected.append(text)
            if len(collected) == 1:
                cancel.cancel()

        result = provider._stream_response(
            messages=[],
            kwargs={},
            on_chunk=on_chunk,
            cancel_token=cancel,
        )

        assert result.finish_reason == FinishReason.CANCELLED
        stream_ref["stream"].close.assert_called_once()

    def test_close_called_on_normal_completion(self):
        provider = _build_provider()
        chunks = [
            _make_chunk(content="hi"),
            _make_chunk(finish_reason="stop"),
        ]
        captured = {}

        def capture_create(**kwargs):
            s = _make_stream(chunks)
            captured["stream"] = s
            return s

        provider._client.chat.completions.create = capture_create

        provider._stream_response(
            messages=[],
            kwargs={},
            on_chunk=lambda _t: None,
        )

        captured["stream"].close.assert_called_once()
