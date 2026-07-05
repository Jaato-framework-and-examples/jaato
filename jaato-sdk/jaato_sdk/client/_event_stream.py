"""Synchronously-subscribed event stream — the public ``open_event_stream()``.

``client.events()`` is an async generator whose subscription (the
``_subscribe_events()`` call) runs lazily on the FIRST ``__anext__`` — so
``it = client.events()`` registers nothing until it is first driven.  A client
that must guarantee its queue is subscribed BEFORE it triggers server-side
output (e.g. subscribe, THEN ``attach`` a session the daemon will immediately
drive a woken turn on — recovery clients have no zero-subscriber replay, so a
not-yet-subscribed queue drops that output) otherwise has to reach into the
private ``_subscribe_events`` / ``_event_subscribers`` internals to force + prove
the registration.

``open_event_stream()`` closes that gap: it calls ``_subscribe_events()``
**synchronously at call time** and returns this iterator over the already-
registered queue.  No first-``__anext__`` deferral, no barrier dance, no private
coupling:

    stream = client.open_event_stream()   # queue registered NOW
    await client.attach(session_id)        # drive can't outrun the subscription
    async for ev in stream:
        ...

Lifetime is caller-managed (this is a plain iterator, not a generator, precisely
so the subscribe is eager — so it carries no generator-``finally`` auto-cleanup):
the queue is unsubscribed on the disconnect ``None`` sentinel, on ``aclose()``,
or on ``async with`` exit.  A long-lived consumer should ``aclose()`` at teardown;
otherwise the only leak is dropping the iterator while still connected without
closing it.
"""

from __future__ import annotations

from typing import Any


class _SyncSubscribedStream:
    """Async iterator over a queue subscribed EAGERLY at construction.

    Yields events from ``queue`` until the drain/pump pushes the ``None``
    disconnect sentinel (then raises ``StopAsyncIteration`` cleanly), mirroring
    ``events()`` semantics.  ``_unsubscribe_events`` is called exactly once (on
    the sentinel, ``aclose()``, or ``async with`` exit) and is idempotent.
    """

    def __init__(self, client: Any, queue: Any) -> None:
        self._client = client
        self._queue = queue
        self._closed = False

    def __aiter__(self) -> "_SyncSubscribedStream":
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        event = await self._queue.get()
        if event is None:  # drain loop / pump signalled disconnect
            await self.aclose()
            raise StopAsyncIteration
        return event

    async def aclose(self) -> None:
        """Unsubscribe the queue.  Idempotent; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._client._unsubscribe_events(self._queue)

    async def __aenter__(self) -> "_SyncSubscribedStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
