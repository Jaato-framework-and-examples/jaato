"""Internal handler registry shared by ``IPCClient`` and ``IPCRecoveryClient``.

Holds typed + catchall event subscribers and dispatches incoming events
to them with full exception isolation. Sync handlers run inline; async
handlers are scheduled fire-and-forget on the current event loop.

Both client classes embed an instance of ``_HandlerRegistry`` and
expose its ``subscribe*`` methods through thin forwarders. Keeping the
state local to each client (rather than sharing a singleton) means
each client/connection has its own independent set of subscribers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Union

from jaato_sdk.events import Event, EventType

logger = logging.getLogger(__name__)


# A handler may be sync (returns None) or async (returns an Awaitable).
EventHandler = Callable[[Event], Union[None, Awaitable[None]]]
Unsubscribe = Callable[[], None]


@dataclass
class _HandlerEntry:
    """Internal record of a registered handler.

    Lives in ``_HandlerRegistry._handlers`` keyed by ``EventType.value``
    (typed subscription) or ``None`` (catchall). ``handler_id`` is unique
    across all entries and used by the unsubscribe closure to find and
    remove this record.
    """

    handler: EventHandler
    once: bool
    handler_id: int


class _HandlerRegistry:
    """Typed event subscription + dispatch.

    Mutated only on the asyncio event loop where the owning client's
    receive loop runs — no cross-thread access is supported.
    ``dispatch`` snapshots each bucket before iterating so subscribe /
    unsubscribe calls made during dispatch only take effect on the
    next event.
    """

    def __init__(self) -> None:
        self._handlers: Dict[Optional[str], List[_HandlerEntry]] = {}
        self._handler_id_counter: int = 0

    # ── Public subscription API ────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Unsubscribe:
        """Subscribe to events of a specific type."""
        return self._add_handler(event_type.value, handler, once=False)

    def subscribe_once(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Unsubscribe:
        """Subscribe to one event of the given type, then auto-unsubscribe."""
        return self._add_handler(event_type.value, handler, once=True)

    def subscribe_all(self, handler: EventHandler) -> Unsubscribe:
        """Subscribe to every event regardless of type (catchall firehose)."""
        return self._add_handler(None, handler, once=False)

    def subscribe_many(
        self,
        handlers: Dict[EventType, EventHandler],
    ) -> Unsubscribe:
        """Register multiple typed handlers in one call.

        Returns a single unsubscribe that removes all of them atomically.
        """
        unsubs = [self.subscribe(et, h) for et, h in handlers.items()]

        def unsub_all() -> None:
            for u in unsubs:
                u()

        return unsub_all

    # ── Dispatch (called by owning client's receive loop) ──────────────

    def dispatch(self, event: Event) -> None:
        """Deliver ``event`` to typed handlers + catchall handlers.

        ``once`` handlers are removed from the bucket *before* being
        invoked, so a handler that re-subscribes itself fires on the
        next matching event rather than this one. Sync exceptions and
        async rejections are logged and swallowed.
        """
        type_key: Optional[str]
        if isinstance(event.type, EventType):
            type_key = event.type.value
        else:
            type_key = str(event.type)

        typed_snapshot = list(self._handlers.get(type_key, []))
        catchall_snapshot = list(self._handlers.get(None, []))

        for entry in typed_snapshot:
            if entry.once:
                self._remove_handler_id(type_key, entry.handler_id)
            self._invoke_handler(entry.handler, event)

        for entry in catchall_snapshot:
            if entry.once:
                self._remove_handler_id(None, entry.handler_id)
            self._invoke_handler(entry.handler, event)

    # ── Internals ──────────────────────────────────────────────────────

    def _add_handler(
        self,
        key: Optional[str],
        handler: EventHandler,
        *,
        once: bool,
    ) -> Unsubscribe:
        self._handler_id_counter += 1
        hid = self._handler_id_counter
        entry = _HandlerEntry(handler=handler, once=once, handler_id=hid)
        self._handlers.setdefault(key, []).append(entry)

        def unsub() -> None:
            self._remove_handler_id(key, hid)

        return unsub

    def _remove_handler_id(self, key: Optional[str], hid: int) -> None:
        bucket = self._handlers.get(key)
        if not bucket:
            return
        for i, entry in enumerate(bucket):
            if entry.handler_id == hid:
                bucket.pop(i)
                return

    def _invoke_handler(self, handler: EventHandler, event: Event) -> None:
        """Run a single handler with full exception isolation."""
        try:
            result = handler(event)
        except Exception:
            logger.exception(
                "Event handler raised for %s", type(event).__name__
            )
            return

        if asyncio.iscoroutine(result):
            task = asyncio.create_task(result)
            task.add_done_callback(self._log_async_handler_failure)

    @staticmethod
    def _log_async_handler_failure(task: "asyncio.Task[object]") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Async event handler failed: %s", exc, exc_info=exc
            )
