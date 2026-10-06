"""Echo can take time to answer, so concurrency is measurable without a model.

Anything about how parallel sessions share runners, or how the pool behaves
while sessions wait on inference, measures nothing when every turn returns
instantly.  ``delay_ms`` / ``jitter_ms`` / ``token_delay_ms`` make a turn
wait; the jitter is seeded from the conversation, so a run is reproducible
while different sessions still wait differently.
"""

from __future__ import annotations

import threading
import time

import pytest

from jaato_server.shared.plugins.model_provider.base import ProviderConfig
from jaato_server.shared.plugins.model_provider.echo.provider import EchoProvider
from jaato_sdk.plugins.model_provider.types import (
    CancelledException, CancelToken, Message, Part, Role,
)


def _provider(**extra):
    p = EchoProvider()
    p.initialize(ProviderConfig(extra=extra))
    return p


def _msgs(*texts):
    return [Message(role=Role.USER, parts=[Part.from_text(t)]) for t in texts]


def _recorded_waits(provider, monkeypatch):
    waits = []
    monkeypatch.setattr(provider, "_wait",
                        lambda seconds, token: waits.append(seconds))
    return waits


def test_no_knob_means_no_wait(monkeypatch):
    p = _provider()
    waits = _recorded_waits(p, monkeypatch)
    p.complete(_msgs("hi"))
    assert waits == [0.0]


def test_fixed_delay_is_exact(monkeypatch):
    p = _provider(delay_ms=1500)
    waits = _recorded_waits(p, monkeypatch)
    p.complete(_msgs("a"))
    p.complete(_msgs("b"))
    assert waits == [1.5, 1.5]


def test_jitter_stays_in_range_and_varies_between_conversations(monkeypatch):
    p = _provider(delay_ms=3000, jitter_ms=2000, seed="s")
    waits = _recorded_waits(p, monkeypatch)
    for i in range(40):
        p.complete(_msgs(f"stage {i}"))
    assert all(1.0 <= w <= 5.0 for w in waits)
    assert len(set(waits)) > 30


def test_jitter_is_reproducible_for_the_same_conversation(monkeypatch):
    a, b = _provider(delay_ms=3000, jitter_ms=2000, seed="s"), \
        _provider(delay_ms=3000, jitter_ms=2000, seed="s")
    wa, wb = _recorded_waits(a, monkeypatch), _recorded_waits(b, monkeypatch)
    for p in (a, b):
        p.complete(_msgs("stage 3"))
        p.complete(_msgs("stage 3", "next"))
    assert wa == wb
    assert wa[0] != wa[1]  # a later turn of the same conversation draws anew


def test_the_seed_changes_the_draw(monkeypatch):
    a, b = _provider(delay_ms=3000, jitter_ms=2000, seed="x"), \
        _provider(delay_ms=3000, jitter_ms=2000, seed="y")
    wa, wb = _recorded_waits(a, monkeypatch), _recorded_waits(b, monkeypatch)
    a.complete(_msgs("same"))
    b.complete(_msgs("same"))
    assert wa != wb


def test_jitter_never_goes_below_zero(monkeypatch):
    p = _provider(delay_ms=10, jitter_ms=5000)
    waits = _recorded_waits(p, monkeypatch)
    for i in range(30):
        p.complete(_msgs(str(i)))
    assert min(waits) == 0.0


@pytest.mark.parametrize("value", [-1, "100", True, float("nan")])
def test_a_value_that_is_not_a_duration_is_refused(value):
    with pytest.raises(ValueError):
        _provider(delay_ms=value)


def test_the_wait_is_real_and_honours_cancel():
    p = _provider(delay_ms=5000)
    token = CancelToken()
    threading.Timer(0.1, token.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(CancelledException):
        p.complete(_msgs("hi"), cancel_token=token)
    assert time.monotonic() - t0 < 1.0


def test_token_delay_streams_word_by_word(monkeypatch):
    p = _provider(token_delay_ms=20, response="one two three")
    waits = _recorded_waits(p, monkeypatch)
    chunks = []
    result = p.complete(_msgs("hi"), on_chunk=chunks.append)
    assert chunks == ["one ", "two ", "three"]
    assert "".join(chunks) == "one two three"
    assert waits == [0.0, 0.02, 0.02]
    assert result is not None
