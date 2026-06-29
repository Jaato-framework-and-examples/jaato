"""In-process convenience facade — Shape 1, PR1 tracer-bullet.

Run the ``jaato_sdk`` ``ask`` / ``complete`` / ``stream`` facade against an
*embedded* ``jaato.JaatoClient`` — no daemon, no runner, no socket. See
``docs/design/in-process-facade.md``.

``InProcessClient`` is the embedded analog of ``jaato_sdk``'s ``IPCClient``: it
implements the small client contract the facade's ``Session`` rides on
(``subscribe`` / ``subscribe_once`` / ``send_message`` /
``respond_to_permission`` / ``connect`` / ``create_session`` / ``disconnect``)
by wrapping the embedded ``JaatoClient`` and translating its callbacks into the
same typed events the daemon emits. The facade (``Session.ask`` / ``.complete``
/ ``.stream``) is reused **unchanged** — the whole point of Shape 1.

This slice wires the two clean seams:

* ``AGENT_OUTPUT`` — from the embedded ``send_message(on_output=...)`` callback
  (the streaming path proven by the dual-mode examples).
* ``TURN_COMPLETED`` — the ``send_message`` return is the turn-done signal.

Deferred to later slices: the ``InProcessChannel`` permission bridge
(``PERMISSION_REQUESTED``, PR2), the profile-load chain + ``AGENT_COMPLETED``
(PR3, via ``AgentUIHooks.on_agent_completed``), and ``stream`` polish (PR4).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.client.convenience import Session
from jaato_sdk.events import AgentOutputEvent, TurnCompletedEvent


def _default_embedded_factory() -> Any:
    """Build the real embedded runtime.

    Imported lazily so only embedded use pulls in the server runtime — callers
    that never embed pay nothing for importing this module.
    """
    from shared import JaatoClient
    return JaatoClient()


class InProcessEventEmitter:
    """Thread-safe in-process pub/sub matching the facade's subscribe contract.

    The no-socket analog of the IPC client's event fan-out. The facade rides on
    exactly three methods:

    * ``subscribe(event_type, cb) -> unsubscribe``
    * ``subscribe_once(event_type, cb) -> unsubscribe`` (auto-unsubscribes
      before dispatching, so a re-entrant ``emit`` cannot double-fire)
    * ``emit(event)`` — dispatch to subscribers keyed on ``event.type``
    """

    def __init__(self) -> None:
        self._subs: Dict[Any, List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(
        self, event_type: Any, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        with self._lock:
            self._subs.setdefault(event_type, []).append(cb)

        def _unsub() -> None:
            with self._lock:
                lst = self._subs.get(event_type)
                if lst and cb in lst:
                    lst.remove(cb)

        return _unsub

    def subscribe_once(
        self, event_type: Any, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        box: Dict[str, Callable[[], None]] = {}

        def _wrapper(ev: Any) -> None:
            unsub = box.get("unsub")
            if unsub is not None:
                unsub()
            cb(ev)

        unsub = self.subscribe(event_type, _wrapper)
        box["unsub"] = unsub
        return unsub

    def emit(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        with self._lock:
            subs = list(self._subs.get(event_type, []))
        for cb in subs:
            cb(event)


class InProcessClient:
    """Embedded analog of ``IPCClient`` — the facade's ``Session`` rides on it.

    Wraps an embedded ``jaato.JaatoClient`` and exposes the facade client
    contract over an in-process event emitter (no socket). Construct via
    :meth:`session` (the ``async with`` entry point), not directly.

    The embedded runtime runs the blocking agent loop, so ``send_message`` is
    dispatched to a worker thread via ``asyncio.to_thread``; its
    ``on_output`` callback fires on that worker thread and marshals
    ``AGENT_OUTPUT`` events back onto the event loop with
    ``loop.call_soon_threadsafe``.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        embedded_factory: Optional[Callable[[], Any]] = None,
        **_ignored: Any,
    ) -> None:
        self._model = model
        self._project = project
        self._location = location
        # Test seam: inject a fake embedded client. Production uses the real
        # ``jaato.JaatoClient`` via the default factory.
        self._embedded_factory = embedded_factory or _default_embedded_factory
        self._embedded: Any = None
        self._emitter = InProcessEventEmitter()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_id: Optional[str] = None

    # ---- facade contract: events ----
    def subscribe(
        self, event_type: Any, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        return self._emitter.subscribe(event_type, cb)

    def subscribe_once(
        self, event_type: Any, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        return self._emitter.subscribe_once(event_type, cb)

    # ---- facade contract: lifecycle ----
    async def connect(self, timeout: Optional[float] = None) -> bool:
        self._loop = asyncio.get_running_loop()
        self._embedded = self._embedded_factory()
        await asyncio.to_thread(
            self._embedded.connect, self._project, self._location, self._model
        )
        return True

    async def create_session(self, **_kwargs: Any) -> str:
        # PR1: ex01 needs no profile/tools. The profile-load chain
        # (discover_profiles -> resolve_secret_uri -> configure_tools) lands in
        # the profile slice (PR3).
        self._session_id = "in-process"
        return self._session_id

    async def send_message(
        self,
        prompt: str,
        *,
        parallel_tools: Optional[bool] = None,
        attachments: Optional[list] = None,
        sources: Any = None,
    ) -> str:
        loop = self._loop
        emitter = self._emitter

        def on_output(source: str, text: str, mode: str) -> None:
            # Called on the worker thread — marshal onto the loop.
            loop.call_soon_threadsafe(
                emitter.emit,
                AgentOutputEvent(
                    agent_id="main", source=source, text=text, mode=mode
                ),
            )

        final_text = await asyncio.to_thread(
            self._embedded.send_message, prompt, on_output
        )
        # Drain the AGENT_OUTPUT callbacks queued from the worker thread before
        # the terminal, so the facade sees all output ahead of TURN_COMPLETED.
        await asyncio.sleep(0)
        emitter.emit(TurnCompletedEvent(agent_id="main"))
        return final_text

    async def respond_to_permission(self, request_id: str, response: str) -> None:
        # The InProcessChannel permission bridge lands in PR2. With no channel
        # wired yet, a session that requests permission can't reach here.
        return None

    async def disconnect(self) -> None:
        if self._embedded is not None and hasattr(self._embedded, "close_session"):
            await asyncio.to_thread(self._embedded.close_session)

    # ---- entry point ----
    @classmethod
    def session(cls, **kwargs: Any) -> "_InProcessSessionContext":
        """Open an embedded session.

        ``async with InProcessClient.session(model=..., profile=...) as s:``
        yields the same :class:`~jaato_sdk.client.convenience.Session` the IPC
        path yields, so ``await s.ask(...)`` / ``.complete`` / ``.stream`` work
        identically.
        """
        return _InProcessSessionContext(cls, kwargs)


class _InProcessSessionContext:
    """Async context manager mirroring ``IPCClient.session``'s ``_SessionContext``.

    Separates connection kwargs (forwarded to ``InProcessClient.__init__``)
    from ``create_session`` kwargs and ``on_permission`` (handed to the facade
    ``Session``), then connects, creates the session, and tears down on exit.
    """

    _CREATE_KEYS = ("profile", "agent", "agent_params", "cascade_driver_id")

    def __init__(self, client_cls: type, kwargs: Dict[str, Any]) -> None:
        self._client_cls = client_cls
        self._kwargs = dict(kwargs)
        self._client: Optional[InProcessClient] = None

    async def __aenter__(self) -> Session:
        on_permission = self._kwargs.pop("on_permission", None)
        create_kwargs = {
            k: self._kwargs.pop(k) for k in self._CREATE_KEYS if k in self._kwargs
        }
        self._client = self._client_cls(**self._kwargs)
        await self._client.connect()
        session_id = await self._client.create_session(**create_kwargs)
        return Session(self._client, session_id, on_permission)

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.disconnect()
