"""Typed wake-primitive client methods — shared by IPCClient + IPCRecoveryClient.

The daemon exposes wake binding + cascade observation as string-keyed commands
(``session.bind_wake`` / ``session.unbind_wake`` / ``cascade.register``) whose
bind result arrives asynchronously as a :class:`WakeBindResultEvent`.  A client
otherwise hand-wires ``execute_command(...)`` + ``subscribe_once(...)`` + a
manual future to correlate that round-trip — the same string-RPC boilerplate
``open_event_stream`` replaced for subscriptions.  These helpers give typed
``await``-the-result methods instead.

Both clients expose the two primitives these use — a SYNCHRONOUS ``subscribe``
(registers on the registry immediately, so the result handler is armed BEFORE
the command is sent → no subscribe-after-send race, which matters on the
recovery client with no replay buffer) and ``execute_command`` — so one
implementation serves both.  Keys ride as positional ``args`` (the server
accepts ``[wake_ref, key...]``); that is the path already proven live.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Union


async def bind_wake(
    client: Any,
    wake_ref: str,
    trust_keys: List[str],
    *,
    timeout: float = 30.0,
) -> Any:
    """Declare a wake binding for the caller's OWN session; await the result.

    Sends ``session.bind_wake`` and returns the daemon's
    :class:`~jaato_sdk.events.WakeBindResultEvent` for ``wake_ref`` (inspect
    ``.outcome`` — ``"ok"`` on success, else ``"unauthorized"`` / ``"no_keys"``
    / ``"no_session"`` / ...; ``.expires_at`` + ``.endpoint`` on success).
    Raises ``asyncio.TimeoutError`` if no result arrives within ``timeout``.
    """
    from jaato_sdk.events import EventType, WakeBindResultEvent  # noqa: F401
    loop = asyncio.get_event_loop()
    fut: "asyncio.Future" = loop.create_future()

    def _on_result(ev: Any) -> None:
        # Filter by wake_ref so concurrent binds don't cross results.
        if getattr(ev, "wake_ref", "") == wake_ref and not fut.done():
            fut.set_result(ev)

    unsub = client.subscribe(EventType.WAKE_BIND_RESULT, _on_result)  # armed BEFORE send
    try:
        await client.execute_command("session.bind_wake", [wake_ref, *trust_keys])
        return await asyncio.wait_for(fut, timeout)
    finally:
        unsub()


async def unbind_wake(
    client: Any,
    wake_ref: str,
    *,
    timeout: float = 30.0,
) -> Any:
    """Remove the caller's own wake binding (owner-guarded); await the result.

    Sends ``session.unbind_wake`` and returns the
    :class:`~jaato_sdk.events.WakeBindResultEvent` for ``wake_ref``.  Raises
    ``asyncio.TimeoutError`` if no result arrives within ``timeout``.
    """
    from jaato_sdk.events import EventType, WakeBindResultEvent  # noqa: F401
    loop = asyncio.get_event_loop()
    fut: "asyncio.Future" = loop.create_future()

    def _on_result(ev: Any) -> None:
        if getattr(ev, "wake_ref", "") == wake_ref and not fut.done():
            fut.set_result(ev)

    unsub = client.subscribe(EventType.WAKE_BIND_RESULT, _on_result)
    try:
        await client.execute_command("session.unbind_wake", [wake_ref])
        return await asyncio.wait_for(fut, timeout)
    finally:
        unsub()


async def cascade_register(
    client: Any,
    cascade_driver_id: str,
    role: str = "observer",
    event_types: Union[List[Any], None] = None,
) -> None:
    """Register the caller as a cascade owner/observer for ``cascade_driver_id``.

    Fire-and-forget (the daemon sends no response for ``cascade.register``).
    ``event_types`` items may be event CLASSES or their names — the class form
    prevents the class-name-string footgun (the cascade filter matches
    ``type(event).__name__``, so a value like ``"session.woken"`` silently never
    matches; pass ``SessionWokenEvent`` and the right name is derived).
    """
    names: List[str] = []
    for et in (event_types or []):
        names.append(et if isinstance(et, str) else getattr(et, "__name__", str(et)))
    await client.execute_command(
        "cascade.register", [cascade_driver_id, role, *names])
