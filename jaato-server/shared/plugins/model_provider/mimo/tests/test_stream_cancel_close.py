"""The MiMo provider closes the SDK ``Stream`` on cancel and on completion —
mirror of ``doubleword/tests/test_stream_cancel_close.py``, pinned so the
shared loop's contract is asserted for this inheritor too."""

from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import CancelToken, FinishReason
from shared.plugins.model_provider.mimo.provider import MiMoProvider


def _chunk(content=None, finish_reason=None):
    chunk = MagicMock()
    chunk.usage = None
    choice = MagicMock()
    choice.finish_reason = finish_reason
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    delta.reasoning_content = None
    delta.audio = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _provider(chunks):
    p = MiMoProvider()
    p._client = MagicMock()
    p._model_name = "mimo-v2.5-pro"
    p._trace = lambda _m: None
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    p._client.chat.completions.create = lambda **kw: stream
    return p, stream


def test_cancel_closes_stream():
    cancel = CancelToken()
    p, stream = _provider([_chunk("a"), _chunk("b")])
    seen = []

    def on_chunk(t):
        seen.append(t)
        cancel.cancel()

    r = p._stream_response(messages=[], kwargs={}, on_chunk=on_chunk, cancel_token=cancel)
    assert r.finish_reason == FinishReason.CANCELLED
    stream.close.assert_called_once()


def test_completion_closes_stream():
    p, stream = _provider([_chunk("a"), _chunk(finish_reason="stop")])
    r = p._stream_response(messages=[], kwargs={}, on_chunk=lambda _t: None)
    assert r.finish_reason == FinishReason.STOP
    stream.close.assert_called_once()
