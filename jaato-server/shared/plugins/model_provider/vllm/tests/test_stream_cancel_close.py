"""Tests that the vLLM provider explicitly closes the SDK ``Stream`` on
both cancellation and natural completion.

Without this close, vLLM keeps generating output tokens up to
``max_tokens`` because the openai SDK ``Stream`` only sends TCP-close at
garbage-collection time — empirical observation 2026-06-09 on
192.168.50.154:8000 was 60-120s of wasted GPU work per cancelled call.

Mirror of openrouter's ``TestStreamCancellationClosesConnection``
(``openrouter/tests/test_openrouter_provider.py``) — same shape so the
contract is enforced symmetrically across OpenAI-SDK-based providers.
"""

from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
)
from shared.plugins.model_provider.vllm.provider import VLLMProvider


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
        choice.delta = delta
        chunk.choices = [choice]
    return chunk


def _make_stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    return stream


def _build_provider():
    provider = VLLMProvider()
    provider._client = MagicMock()
    provider._client.close = MagicMock()
    provider._model_name = "Qwen/Qwen3-14B"
    provider._max_tokens = 2048
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
        # Shape B (PR-260 falsification): on cancel, the openai client's
        # httpx pool must also be closed so TCP-FIN actually reaches vLLM.
        # Stream.close() alone does not propagate disconnect.
        provider._client.close.assert_called_once()

    def test_client_close_NOT_called_on_normal_completion(self):
        """Natural completion must NOT close the client — keep-alive
        is desirable for the next call from the same session."""
        provider = _build_provider()
        chunks = [
            _make_chunk(content="hi"),
            _make_chunk(finish_reason="stop"),
        ]

        def capture_create(**kwargs):
            return _make_stream(chunks)

        provider._client.chat.completions.create = capture_create

        provider._stream_response(
            messages=[],
            kwargs={},
            on_chunk=lambda _t: None,
        )

        provider._client.close.assert_not_called()

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
