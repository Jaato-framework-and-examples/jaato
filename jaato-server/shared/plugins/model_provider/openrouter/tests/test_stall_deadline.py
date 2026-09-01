"""A stalled request must become a typed error, not an unbounded wait.

#732: agentic sessions against OpenRouter stopped mid-tool-loop and never
resumed — no exception, no ``finish_reason``, no retry.  Inspected live,
the process held an ESTABLISHED socket to OpenRouter with zero bytes
queued and a thread parked in ``do_poll``; the session burned wall-clock
until something *outside* the provider (a harness arm-timeout, a budget
ceiling) tore it down.  One arm's upstream eventually said
``APIError: Upstream idle timeout exceeded`` — proof the condition is
nameable, and that which side notices first was the only thing deciding
whether the caller saw an error or a hang.

Why a socket timeout can't be the fix: ``httpx``'s read timeout bounds the
gap between *bytes*, and OpenRouter keeps a stalled stream fed with
``: OPENROUTER PROCESSING`` SSE comments.  The OpenAI SDK's decoder drops
comment lines without yielding an event, so those bytes reset the byte
clock while the consumer's chunk loop never ticks.  The deadline therefore
has to be measured in payload — that is what
:class:`~..stall.StreamStallGuard` does.

The tests below use deadlines in the tens of milliseconds; the guard's
clock is ``time.monotonic`` and its wait is an ``Event``, so they are
fast and don't depend on wall-clock scheduling beyond "a sleep of 20x the
deadline outlasts the deadline".
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import Message, Role

from ..env import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_STREAM_IDLE_TIMEOUT,
    resolve_request_timeout,
    resolve_stream_idle_timeout,
)
from ..errors import InfrastructureError, StallTimeoutError
from ..provider import OpenRouterProvider, _resolve_deadline
from ..stall import StreamStallGuard


# ==================== The watchdog itself ====================


class TestStreamStallGuard:
    """The payload-idle watchdog, independent of any transport."""

    def test_fires_after_the_deadline_with_no_pings(self):
        fired = threading.Event()
        guard = StreamStallGuard(0.02, on_stall=fired.set)
        with guard:
            assert fired.wait(2.0), "watchdog never fired"
        assert guard.fired is True

    def test_pings_keep_it_from_firing(self):
        calls = []
        guard = StreamStallGuard(0.15, on_stall=lambda: calls.append(1))
        with guard:
            deadline = time.monotonic() + 0.6
            while time.monotonic() < deadline:
                guard.ping()
                time.sleep(0.01)
        assert calls == []
        assert guard.fired is False

    def test_stop_disarms_it(self):
        calls = []
        guard = StreamStallGuard(0.05, on_stall=lambda: calls.append(1))
        guard.start()
        guard.stop()
        time.sleep(0.2)
        assert calls == []
        assert guard.fired is False

    def test_zero_timeout_disables_the_guard(self):
        """``0`` is the documented opt-out, not a zero-length deadline."""
        calls = []
        guard = StreamStallGuard(0, on_stall=lambda: calls.append(1))
        with guard:
            time.sleep(0.1)
        assert guard.enabled is False
        assert guard.fired is False
        assert calls == []

    def test_a_raising_callback_still_marks_the_guard_fired(self):
        """Teardown is best-effort; the *fact* of the stall is not."""
        def boom():
            raise RuntimeError("close() blew up")

        guard = StreamStallGuard(0.02, on_stall=boom)
        with guard:
            time.sleep(0.4)
        assert guard.fired is True

    def test_fired_is_set_before_the_callback_runs(self):
        """A consumer unparked *by* the callback must not race the flag."""
        seen = {}
        guard = StreamStallGuard(0.02, on_stall=lambda: seen.update(
            fired_during_callback=guard.fired))
        with guard:
            time.sleep(0.4)
        assert seen == {"fired_during_callback": True}


# ==================== Knob resolution ====================


class TestDeadlineKnobs:
    """Profile > env > default, and a typo fails loud."""

    def test_defaults_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.delenv("JAATO_OPENROUTER_REQUEST_TIMEOUT", raising=False)
        monkeypatch.delenv("JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT", raising=False)
        assert resolve_request_timeout() == DEFAULT_REQUEST_TIMEOUT
        assert resolve_stream_idle_timeout() == DEFAULT_STREAM_IDLE_TIMEOUT

    def test_env_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv("JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT", "45")
        assert resolve_stream_idle_timeout() == 45.0

    def test_env_zero_disables(self, monkeypatch):
        monkeypatch.setenv("JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT", "0")
        assert resolve_stream_idle_timeout() == 0.0

    @pytest.mark.parametrize("raw", ["nonsense", "-5"])
    def test_env_garbage_falls_back_to_the_default(self, monkeypatch, raw):
        monkeypatch.setenv("JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT", raw)
        assert resolve_stream_idle_timeout() == DEFAULT_STREAM_IDLE_TIMEOUT

    def test_none_falls_through_to_the_fallback(self):
        assert _resolve_deadline(None, 12.0, "stream_idle_timeout") == 12.0

    def test_zero_is_accepted(self):
        assert _resolve_deadline(0, 12.0, "stream_idle_timeout") == 0.0

    @pytest.mark.parametrize("bad", ["soon", -1, [30]])
    def test_a_bad_profile_value_raises(self, bad):
        with pytest.raises(ValueError, match="stream_idle_timeout"):
            _resolve_deadline(bad, 12.0, "stream_idle_timeout")

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_profile_framework_overrides_win(self, mock_client_class, monkeypatch):
        monkeypatch.setenv("JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT", "45")
        mock_client_class.return_value = lambda **kw: MagicMock()

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"framework_overrides": {
                "connect_timeout": 5,
                "request_timeout": 120,
                "stream_idle_timeout": 90,
            }},
        ))

        assert provider._connect_timeout == 5.0
        assert provider._request_timeout == 120.0
        assert provider._stream_idle_timeout == 90.0

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_unconfigured_provider_gets_the_defaults(self, mock_client_class):
        mock_client_class.return_value = lambda **kw: MagicMock()

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))

        assert provider._connect_timeout == DEFAULT_CONNECT_TIMEOUT
        assert provider._request_timeout == DEFAULT_REQUEST_TIMEOUT
        assert provider._stream_idle_timeout == DEFAULT_STREAM_IDLE_TIMEOUT


class TestClientTransportDeadlines:
    """What actually reaches the OpenAI SDK constructor."""

    def _capture(self):
        captured = {}

        def fake_client_class(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        return captured, fake_client_class

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_timeout_and_retry_budget_are_explicit(self, mock_client_class):
        pytest.importorskip("httpx")
        captured, fake = self._capture()
        mock_client_class.return_value = fake

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"framework_overrides": {
                "connect_timeout": 5, "request_timeout": 120,
            }},
        ))

        timeout = captured["timeout"]
        assert timeout.connect == 5.0
        assert timeout.read == 120.0
        assert timeout.write == 120.0
        # The SDK's own retry budget would silently multiply every
        # deadline by three; retries belong to retry_utils.with_retry,
        # which is observable and honours OpenRouter's Retry-After (#720).
        assert captured["max_retries"] == 0

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_zero_request_timeout_means_no_byte_deadline(self, mock_client_class):
        pytest.importorskip("httpx")
        captured, fake = self._capture()
        mock_client_class.return_value = fake

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"framework_overrides": {"request_timeout": 0}},
        ))

        assert captured["timeout"].read is None


# ==================== The streaming path ====================


def _terminal_chunk():
    """A final chunk carrying ``finish_reason``, as a real stream sends.

    Fixtures that end without one now raise ``StreamInterruptedError``
    before any assertion here is reached (#687): "the stream stopped
    arriving" is a failure in its own right, so a fixture standing in
    for a HEALTHY stream has to terminate like one.
    """
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = None
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].delta.reasoning = None
    chunk.choices[0].finish_reason = "stop"
    chunk.usage = None
    chunk.error = None
    chunk.model_extra = {}
    return chunk


def _stalling_stream(stop: threading.Event):
    """A stream that yields nothing and blocks, like a parked read.

    Stands in for the observed condition: the socket is up, OpenRouter's
    keep-alive comments keep ``httpx`` happy, and the SDK iterator simply
    never produces an event.
    """
    class _Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            # Blocks until the watchdog's close() releases it — the
            # in-process analogue of the parked recv().
            stop.wait(5.0)
            raise RuntimeError("Connection closed while reading stream")

        def close(self):
            self.closed = True
            stop.set()

    return _Stream()


def _provider_with_stalling_stream(stop, idle_timeout=0.05, stream=None):
    """An initialized provider whose next stream call stalls."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = (
        stream if stream is not None else _stalling_stream(stop)
    )
    with patch(
        "shared.plugins.model_provider.openrouter.provider."
        "get_openai_client_class",
        return_value=lambda **kw: fake_client,
    ):
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"framework_overrides": {
                "stream_idle_timeout": idle_timeout,
                # connect() resolves the window from the catalog, which
                # a unit test can't reach; the knob is the documented
                # fallback.
                "context_length": 200000,
            }},
        ))
        provider.connect("openai/gpt-5-mini", skip_model_test=True)
        return provider, fake_client


class TestStreamingStallBecomesATypedError:

    def test_a_stalled_stream_raises_stall_timeout_error(self):
        stop = threading.Event()
        provider, _ = _provider_with_stalling_stream(stop)

        with pytest.raises(StallTimeoutError) as excinfo:
            provider.complete(
                [Message.from_text(Role.USER, "hi")],
                on_chunk=lambda text: None,
            )

        err = excinfo.value
        assert err.idle_timeout == 0.05
        assert err.chunks_received == 0
        assert err.model == "openai/gpt-5-mini"
        assert "stalled" in str(err).lower()
        assert "stream_idle_timeout" in str(err)

    def test_the_stall_error_is_retryable_infrastructure(self):
        """It composes with with_retry rather than needing its own wiring."""
        stop = threading.Event()
        provider, _ = _provider_with_stalling_stream(stop)
        err = StallTimeoutError(300.0)

        assert isinstance(err, InfrastructureError)
        assert provider.classify_error(err) == {
            "transient": True, "rate_limit": False, "infra": True,
        }
        # A stall has no Retry-After to read — the upstream never
        # answered — so the standard backoff applies (#720 covers the
        # case where a hint does exist).
        assert provider.get_retry_after(err) is None

    def test_the_watchdog_closes_the_transport(self):
        """Closing the pool is what unparks the read and stops billing."""
        stop = threading.Event()
        stream = _stalling_stream(stop)
        provider, fake_client = _provider_with_stalling_stream(
            stop, stream=stream)

        with pytest.raises(StallTimeoutError):
            provider.complete(
                [Message.from_text(Role.USER, "hi")],
                on_chunk=lambda text: None,
            )

        assert stream.closed is True
        assert fake_client.close.called

    def test_a_quietly_ending_stream_after_a_stall_is_not_a_finished_turn(self):
        """The silent half of #732: a truncated turn must not look done."""
        class _EndsQuietly:
            """Yields nothing, then returns — no exception at all."""

            def __init__(self):
                self.closed = False

            def __iter__(self):
                time.sleep(0.3)
                return iter(())

            def close(self):
                self.closed = True

        stop = threading.Event()
        provider, _ = _provider_with_stalling_stream(
            stop, idle_timeout=0.05, stream=_EndsQuietly())

        with pytest.raises(StallTimeoutError):
            provider.complete(
                [Message.from_text(Role.USER, "hi")],
                on_chunk=lambda text: None,
            )

    def test_a_healthy_stream_is_unaffected(self):
        """The guard must be invisible when payload keeps arriving."""
        class _Healthy:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                for text in ("Hel", "lo"):
                    chunk = MagicMock()
                    chunk.choices = [MagicMock()]
                    chunk.choices[0].delta.content = text
                    chunk.choices[0].delta.tool_calls = None
                    chunk.choices[0].delta.reasoning = None
                    chunk.choices[0].finish_reason = None
                    chunk.usage = None
                    chunk.error = None
                    chunk.model_extra = {}
                    yield chunk
                yield _terminal_chunk()

            def close(self):
                self.closed = True

        stop = threading.Event()
        provider, _ = _provider_with_stalling_stream(
            stop, idle_timeout=5.0, stream=_Healthy())

        chunks = []
        result = provider.complete(
            [Message.from_text(Role.USER, "hi")],
            on_chunk=chunks.append,
        )

        assert chunks == ["Hel", "lo"]
        assert result is not None

    def test_a_cancel_landing_with_the_deadline_stays_a_cancel(self):
        """A caller-requested stop must not be reported as a stall.

        Both conditions are true here at once: the watchdog has fired
        (it is what unblocks the stream) *and* the turn was cancelled.
        The turn must end as CANCELLED — reporting a stall instead would
        turn a clean stop into a retryable error and cost a real retry.
        """
        from jaato_sdk.plugins.model_provider.types import (
            CancelToken, FinishReason,
        )

        cancel_token = CancelToken()

        class _UnblocksOnClose:
            """Parked until the watchdog closes it, as a real read is."""

            def __init__(self):
                self.closed = False
                self._gate = threading.Event()

            def __iter__(self):
                self._gate.wait(5.0)
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = "partial"
                chunk.choices[0].delta.tool_calls = None
                chunk.choices[0].delta.reasoning = None
                chunk.choices[0].finish_reason = None
                chunk.usage = None
                chunk.error = None
                chunk.model_extra = {}
                yield chunk

            def close(self):
                self.closed = True
                # The cancel lands in the same instant the deadline does.
                cancel_token.cancel()
                self._gate.set()

        stream = _UnblocksOnClose()
        stop = threading.Event()
        provider, _ = _provider_with_stalling_stream(
            stop, idle_timeout=0.05, stream=stream)

        result = provider.complete(
            [Message.from_text(Role.USER, "hi")],
            on_chunk=lambda text: None,
            cancel_token=cancel_token,
        )

        assert stream.closed is True, "precondition: the watchdog fired"
        assert result.finish_reason == FinishReason.CANCELLED

    def test_a_disabled_deadline_restores_the_unbounded_wait(self):
        """``0`` opts out: the guard never arms, so nothing is torn down."""
        class _SlowButFine:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                time.sleep(0.2)
                # It is slow but FINE, so it terminates properly.  A
                # stream that ends with no finish reason is #687's
                # failure, not this test's subject.
                yield _terminal_chunk()

            def close(self):
                self.closed = True

        stop = threading.Event()
        stream = _SlowButFine()
        provider, _ = _provider_with_stalling_stream(
            stop, idle_timeout=0, stream=stream)

        result = provider.complete(
            [Message.from_text(Role.USER, "hi")],
            on_chunk=lambda text: None,
        )

        assert result is not None


# ==================== The client survives the stall ====================


def _healthy_stream():
    """A stream that yields two content chunks and ends."""
    class _Healthy:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            for text in ("Hel", "lo"):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                chunk.choices[0].delta.tool_calls = None
                chunk.choices[0].delta.reasoning = None
                chunk.choices[0].finish_reason = None
                chunk.usage = None
                chunk.error = None
                chunk.model_extra = {}
                yield chunk
            yield _terminal_chunk()

        def close(self):
            self.closed = True

    return _Healthy()


def _quietly_ending_stream():
    """A stream that outlasts the deadline and then just ends."""
    class _EndsQuietly:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            time.sleep(0.3)
            return iter(())

        def close(self):
            self.closed = True

    return _EndsQuietly()


class _PoolAwareClient:
    """A fake client whose pool death is observable, like the real one.

    ``MagicMock`` happily answers a request after ``close()``, which is
    what let the missing client rebuild go unnoticed.  Measured against
    the installed SDK: once ``client.close()`` has run, the next
    ``chat.completions.create`` raises
    ``openai.APIConnectionError: Connection error.``  This stand-in
    reproduces that, so a test can tell a live client from a dead one.
    """

    def __init__(self, stream_factory):
        self.closed = False
        self.requests = 0
        self._stream_factory = stream_factory
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if self.closed:
            import httpx
            import openai
            raise openai.APIConnectionError(request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"))
        self.requests += 1
        return self._stream_factory()

    def close(self):
        self.closed = True


class TestTheClientSurvivesAStall:
    """A stall must leave the provider able to make the next request.

    This is the precondition for the whole retry story, not cleanup.
    ``StallTimeoutError`` subclasses ``InfrastructureError`` so
    ``with_retry`` retries the turn — but the watchdog got the parked
    read unstuck by closing the client's httpx pool, and a closed client
    raises ``APIConnectionError: Connection error.`` on its next request
    (measured).  Without the rebuild, every retry of a stalled turn dies
    on a dead pool and reports a connection failure that never happened,
    burning the retry budget on a misleading error.  Wrong is worse than
    stuck.

    Both stall exits need this, so both are covered: the one where the
    torn-down read raises, and the one where the iterator just ends.
    The assertion is the *outcome* — a second turn reaches the transport
    — so it keeps its meaning if the rebuild is done some other way.
    """

    def _stalling_then_healthy(self, first_stream):
        """Serve ``first_stream`` to turn one, a healthy stream after."""
        streams = [lambda: first_stream, _healthy_stream]

        def next_stream():
            return streams.pop(0)() if len(streams) > 1 else streams[0]()

        return next_stream

    def _run_two_turns(self, first_stream):
        """Stall a turn, then take another.  Returns (result, clients)."""
        clients = []
        next_stream = self._stalling_then_healthy(first_stream)

        def fake_client_class(**kwargs):
            client = _PoolAwareClient(next_stream)
            clients.append(client)
            return client

        # The patch spans BOTH turns: the rebuild goes through
        # get_openai_client_class() too, and outside the patch it would
        # build a real client pointed at the real endpoint.
        with patch(
            "shared.plugins.model_provider.openrouter.provider."
            "get_openai_client_class",
            return_value=fake_client_class,
        ):
            provider = OpenRouterProvider()
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"framework_overrides": {
                    "stream_idle_timeout": 0.05,
                    "context_length": 200000,
                }},
            ))
            provider.connect("openai/gpt-5-mini", skip_model_test=True)

            with pytest.raises(StallTimeoutError):
                provider.complete(
                    [Message.from_text(Role.USER, "hi")],
                    on_chunk=lambda text: None,
                )

            # Disarm the deadline before turn two.  Turn one has made its
            # point, and 50ms is not a budget a HEALTHY turn can be held
            # to: the guard is stopped in the ``finally`` AFTER the chunk
            # loop, so a consumer thread descheduled between the last
            # chunk and ``guard.stop()`` fires the watchdog on a stream
            # that already finished, and the second turn dies as
            # "stalled ... after 2 content chunk(s)" — the two chunks
            # _healthy_stream yields.  Observed on CI, not locally.
            # What this test asserts is the REBUILD (turn two reaches the
            # transport at all), which the deadline plays no part in.
            provider._stream_idle_timeout = 0.0

            # The turn with_retry would take next.  It must reach the
            # transport rather than dying on the pool the guard closed.
            result = provider.complete(
                [Message.from_text(Role.USER, "again")],
                on_chunk=lambda text: None,
            )

        return result, clients

    def test_after_a_stall_that_raises(self):
        stop = threading.Event()
        result, clients = self._run_two_turns(_stalling_stream(stop))

        assert result is not None
        assert clients[0].closed is True, "precondition: the guard closed it"
        assert len(clients) == 2, (
            "the stalled turn left the provider on a closed client — the "
            "next retry would raise APIConnectionError instead of retrying"
        )
        assert clients[1].requests == 1

    def test_after_a_stall_that_ends_quietly(self):
        result, clients = self._run_two_turns(_quietly_ending_stream())

        assert result is not None
        assert clients[0].closed is True, "precondition: the guard closed it"
        assert len(clients) == 2, (
            "the quiet-end stall path left the provider on a closed client"
        )
        assert clients[1].requests == 1
