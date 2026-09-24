"""Payload-idle watchdog for OpenRouter streaming turns.

WHY A WATCHDOG AND NOT A SOCKET TIMEOUT.  ``httpx``'s read timeout bounds
the gap between *bytes*, and OpenRouter keeps a stalled stream fed with
SSE keep-alive comments (``: OPENROUTER PROCESSING``).  The OpenAI SDK's
SSE decoder drops comment lines without yielding an event, so those bytes
reset the read clock while the caller's ``for chunk in stream`` loop never
ticks.  A session can therefore sit on an ESTABLISHED socket with zero
bytes queued, in ``do_poll``, for as long as the upstream feels like it —
which is exactly what #732 observed (20+ minutes, no exception, no
``finish_reason``, no retry).

So the deadline this module enforces is measured in *payload*, not bytes:
the consumer :meth:`~StreamStallGuard.ping`\\ s on every event the SDK
actually yields, and if no ping arrives within ``timeout`` seconds the
guard fires its ``on_stall`` callback — the provider closes the stream and
the HTTP client, which is what unblocks the parked read — and records that
it did, so the consumer can convert whatever the torn-down connection
raises into a typed, retryable error rather than a truncated turn.

The guard is deliberately transport-agnostic: it knows nothing about
``openai``, ``httpx`` or the provider, which keeps it unit-testable
without either package installed.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class StreamStallGuard:
    """Fire a callback when a stream goes ``timeout`` seconds without payload.

    Lifecycle::

        guard = StreamStallGuard(300.0, on_stall=abort)
        with guard:                     # starts the watchdog thread
            for event in stream:
                guard.ping()            # payload arrived — reset the clock
                ...
        if guard.fired:                 # the watchdog tore the stream down
            raise StallTimeoutError(...)

    A ``timeout`` of ``0`` (or negative) disables the guard entirely: no
    thread is started, :attr:`fired` stays ``False``, and the caller keeps
    the pre-#732 unbounded behaviour.  That is the documented opt-out for
    operators whose models legitimately think for longer than any deadline
    they are willing to set.

    Thread-safety: :meth:`ping` and :meth:`stop` are safe to call from the
    consumer thread while the watchdog thread is running.  ``on_stall``
    runs on the *watchdog* thread, so it must only do work that is safe
    off the consumer thread (closing an HTTP stream / client is; mutating
    the half-built response is not).
    """

    def __init__(
        self,
        timeout: float,
        on_stall: Callable[[], None],
        *,
        name: str = "openrouter-stall-guard",
    ) -> None:
        self._timeout = float(timeout or 0.0)
        self._on_stall = on_stall
        self._name = name
        self._lock = threading.Lock()
        self._last_ping = time.monotonic()
        self._stop = threading.Event()
        self._fired = False
        self._thread: Optional[threading.Thread] = None

    # ---------------- properties ----------------

    @property
    def enabled(self) -> bool:
        """True when a positive deadline was configured."""
        return self._timeout > 0

    @property
    def timeout(self) -> float:
        """The configured idle deadline in seconds (0 = disabled)."""
        return self._timeout

    @property
    def fired(self) -> bool:
        """True once the deadline expired and ``on_stall`` was invoked.

        Set *before* the callback runs, so a consumer woken by the
        callback's teardown always sees it.  Never reset — a guard that
        fired stays fired for the life of the object.
        """
        return self._fired

    # ---------------- lifecycle ----------------

    def start(self) -> "StreamStallGuard":
        """Start the watchdog thread (no-op when disabled or started)."""
        if not self.enabled or self._thread is not None:
            return self
        self._last_ping = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True,
        )
        self._thread.start()
        return self

    def ping(self) -> None:
        """Record that payload arrived, resetting the deadline."""
        with self._lock:
            self._last_ping = time.monotonic()

    def stop(self) -> None:
        """Stop the watchdog.  Idempotent; safe after firing."""
        self._stop.set()

    def __enter__(self) -> "StreamStallGuard":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ---------------- internals ----------------

    def _run(self) -> None:
        """Watchdog body: sleep until the deadline, re-check, then fire.

        ``Event.wait`` is not woken by :meth:`ping` — it does not need to
        be.  Waking at the *old* deadline and recomputing from the latest
        ping costs one extra wakeup per idle period and keeps ``ping`` (the
        hot path, once per SSE event) down to a lock and a clock read.
        """
        while not self._stop.is_set():
            with self._lock:
                remaining = (self._last_ping + self._timeout) - time.monotonic()
            if remaining > 0:
                self._stop.wait(remaining)
                continue
            self._fired = True
            try:
                self._on_stall()
            except Exception:  # pragma: no cover - teardown is best effort
                pass
            return
