"""Tests that the Antigravity provider explicitly closes the httpx
``Response`` on both cancellation and natural completion.

``httpx.Response`` from ``client.send(req, stream=True)`` only sends
TCP-close at garbage-collection time, which on a cancelled turn can
leave the upstream model generating to completion.
"""

from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
)
from shared.plugins.model_provider.antigravity.provider import (
    AntigravityProvider,
)


def _make_response(sse_lines):
    response = MagicMock()
    response.iter_lines = lambda: iter(sse_lines)
    response.close = MagicMock()
    return response


def _build_provider():
    provider = AntigravityProvider()
    return provider


class TestStreamCancellationClosesConnection:
    def test_cancel_closes_response(self):
        provider = _build_provider()
        cancel = CancelToken()
        # First line carries one text chunk; cancel fires before iteration
        # picks up the next line.
        sse_lines = [
            'data: {"candidates":[{"content":{"parts":[{"text":"hello "}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"world"}]}}]}',
            'data: {"candidates":[{"finishReason":"STOP"}]}',
        ]
        response = _make_response(sse_lines)

        collected = []

        def on_chunk(text: str) -> None:
            collected.append(text)
            if len(collected) == 1:
                cancel.cancel()

        result = provider._process_stream(
            response=response,
            on_chunk=on_chunk,
            cancel_token=cancel,
        )

        assert result.finish_reason == FinishReason.CANCELLED
        response.close.assert_called_once()

    def test_close_called_on_normal_completion(self):
        provider = _build_provider()
        sse_lines = [
            'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}',
            'data: {"candidates":[{"finishReason":"STOP"}]}',
        ]
        response = _make_response(sse_lines)

        provider._process_stream(
            response=response,
            on_chunk=lambda _t: None,
        )

        response.close.assert_called_once()
