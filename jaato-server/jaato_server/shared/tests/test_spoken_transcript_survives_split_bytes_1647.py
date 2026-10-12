"""A spoken transcript's accented characters survive the wire (#1647).

Reported: the transcript on a spoken turn's final media chunk (#869)
carried U+FFFD in place of some accented characters
(``est\\ufffdndar``).  The signature suggested a multi-byte UTF-8
sequence split across a boundary and decoded per chunk.

These tests locate it.  They drive the **real** streaming loop of two
providers through the **real** OpenAI SDK, over an HTTP response whose
body arrives in byte chunks that cut ``á`` in half -- the split that,
decoded per chunk, produces exactly the reported damage.  The words
arrive intact, so jaato's provider path does not produce it: the SDK
buffers bytes to a complete SSE line before decoding, and every later
hop (runner RPC, IPC, WS) decodes whole frames strictly, where a split
would raise rather than substitute.  A U+FFFD in a transcript delta
was therefore in the JSON the upstream sent.

What this change adds is the evidence for a given turn: each transcript
delta is traced as received (``AUDIO_TRANSCRIPT_DELTA``), and one that
carries U+FFFD is reported at WARNING.
"""

import json
import logging

import httpx
import pytest

pytest.importorskip("openai")
from openai import OpenAI  # noqa: E402

from jaato_sdk.plugins.model_provider.types import MediaDelta  # noqa: E402
from jaato_server.shared.plugins.model_provider import _media_deltas  # noqa: E402
from jaato_server.shared.plugins.model_provider.nim.provider import NIMProvider  # noqa: E402
from jaato_server.shared.plugins.model_provider.openrouter.provider import (  # noqa: E402
    OpenRouterProvider,
)
from jaato_server.shared.tests.reversion import Reversion  # noqa: E402

SPOKEN = "un resumen detallado del estándar RIP, su arquitectura y más"

_DELTAS = "jaato-server/jaato_server/shared/plugins/model_provider/_media_deltas.py"

REVERSIONS = [
    Reversion(
        target=_DELTAS,
        find="pending, wrote_text, on_transcript=self._observe_transcript_delta,",
        replace="pending, wrote_text,",
        test="TestEachDeltaIsTraced::test_every_transcript_delta_is_traced_as_received",
        because="no transcript delta is traced, so an affected turn leaves no evidence",
    ),
    Reversion(
        target=_DELTAS,
        find="        if damaged and not getattr(self, \"_transcript_damage_warned\", False):",
        replace="        if False:",
        test="TestDamageIsReported::test_a_replacement_character_is_reported_once",
        because="a damaged transcript reaches the client with nothing said",
    ),
]


def _sse(transcript_pieces):
    """The SSE body of one spoken turn: transcript deltas, audio, the end marker."""
    events = []
    for piece in transcript_pieces:
        events.append({"choices": [{"index": 0, "delta": {
            "audio": {"transcript": piece}}}]})
    events.append({"choices": [{"index": 0, "delta": {
        "audio": {"data": "AAEC"}}}]})
    events.append({"choices": [{"index": 0, "delta": {
        "audio": {"expires_at": 1}}}]})
    events.append({"choices": [{"index": 0, "delta": {},
                                "finish_reason": "stop"}]})
    events.append({"choices": [], "usage": {
        "prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}})
    body = "".join(
        "data: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode("utf-8")


def _split_inside_every_multibyte_char(raw: bytes):
    """Cut the body after the first byte of each multi-byte character."""
    cuts = [i + 1 for i, b in enumerate(raw) if b >= 0xC0]
    chunks, prev = [], 0
    for cut in cuts:
        chunks.append(raw[prev:cut])
        prev = cut
    chunks.append(raw[prev:])
    assert len(cuts) == 2  # "á" and "á" of "más": both cut in half
    for chunk in chunks[:-1]:
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError:
            continue
        raise AssertionError("a chunk ends on a character boundary")
    return chunks


def _client(chunks):
    def handler(_request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(b"") if not chunks else _Chunks(chunks))
    return OpenAI(api_key="test", base_url="http://upstream.test/v1",
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                  max_retries=0)


class _Chunks(httpx.SyncByteStream):
    """A response body delivered in exactly these byte chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks


def _build(cls, model, chunks, traces):
    provider = cls()
    provider._client = _client(chunks)
    provider._model_name = model
    provider._enable_thinking = False
    provider._trace = traces.append
    return provider


PROVIDERS = [
    pytest.param(NIMProvider, "meta/llama-3.3-70b-instruct", id="openai_compat"),
    pytest.param(OpenRouterProvider, "openai/gpt-audio-mini", id="openrouter"),
]


def _speak(provider):
    media = []

    def on_chunk(chunk):
        if isinstance(chunk, MediaDelta):
            media.append(chunk)

    provider.complete([], on_chunk=on_chunk)
    finals = [m for m in media if m.final]
    assert finals, "no final media chunk"
    return finals[-1].transcript


def _pieces():
    # Split the words into several deltas, one ending just before "á".
    cut = SPOKEN.index("á")
    return [SPOKEN[:cut], SPOKEN[cut:cut + 9], SPOKEN[cut + 9:]]


@pytest.mark.parametrize("cls,model", PROVIDERS)
class TestTheWordsArriveWhole:
    def test_a_character_split_across_byte_chunks_is_not_damaged(self, cls, model):
        chunks = _split_inside_every_multibyte_char(_sse(_pieces()))
        transcript = _speak(_build(cls, model, chunks, []))
        assert transcript == SPOKEN
        assert _media_deltas.REPLACEMENT_CHAR not in transcript


@pytest.mark.parametrize("cls,model", PROVIDERS)
class TestEachDeltaIsTraced:
    def test_every_transcript_delta_is_traced_as_received(self, cls, model):
        traces = []
        _speak(_build(cls, model, [_sse(_pieces())], traces))
        seen = [t for t in traces if t.startswith("AUDIO_TRANSCRIPT_DELTA")]
        assert len(seen) == len(_pieces())
        assert all("replacement_chars=0" in t for t in seen)
        assert repr(_pieces()[1]) in seen[1]


@pytest.mark.parametrize("cls,model", PROVIDERS)
class TestDamageIsReported:
    def test_a_replacement_character_is_reported_once(self, cls, model, caplog):
        damaged = ["del est�", "ndar RIP, y m�s"]
        traces = []
        provider = _build(cls, model, [_sse(damaged)], traces)
        with caplog.at_level(logging.WARNING, logger=_media_deltas.__name__):
            transcript = _speak(provider)
        # Passed through unchanged: nothing downstream can recover the bytes.
        assert transcript == "".join(damaged)
        warnings = [r for r in caplog.records if "U+FFFD" in r.getMessage()]
        assert len(warnings) == 1
        assert any("replacement_chars=1" in t for t in traces)


def test_a_raising_observer_does_not_cost_the_turn():
    class Delta:
        audio = {"transcript": "hola"}

    sink = []

    def boom(_text):
        raise RuntimeError("observer")

    seq = _media_deltas.emit_audio_delta(
        Delta(), lambda _c: None, 0, transcript_sink=sink, on_transcript=boom)
    assert seq == 0
    assert sink == ["hola"]
