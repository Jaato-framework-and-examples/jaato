"""One definition of how an ``ExternalEventRequest`` becomes a bus event.

An external event is the host's way of poking a running session from the
outside: ``{name, data, timestamp}`` arrives as a typed client request,
becomes an :class:`jaato_sdk.event_bus.Event` of type ``EXTERNAL_EVENT`` on
the session's ``EventBus``, and from there reaches both the agents that
called ``subscribeToEvents(event_types=['external_event'])`` and -- through
``SessionManager.reactor_bus_sink`` -- the daemon-wide reactor bus.

**Why a module and not a method on either transport.**  Until issue #1167
the translation lived inside ``JaatoWSServer._handle_external_event`` and
was reachable from WebSocket only: ``session_manager.py``,
``command_router.py`` and ``ipc.py`` contained no occurrence of
``ExternalEventRequest`` at all, so an IPC client's request fell through
every ``isinstance`` arm of ``SessionManager.handle_request`` and was
answered ``Unknown request type: ExternalEventRequest``.  An IPC-only
deployment therefore had a reactor engine nothing could externally trigger
(the ``webhook`` plugin's listener was the only way in, at the cost of a
bound port, TLS, an allowlist and a signed route).

Giving IPC its own copy of the translation would have made *two* answers to
"what does an external event look like on the bus" -- the shape this tree
treats as a defect in its own right, because each copy masks the other and
neither can be shown to do anything (the ``on_unmetered`` validation the
reversion meta-guard correctly reported as decorative, #688).  So the
transports keep what is genuinely theirs -- resolving *which* session the
caller is driving -- and share this.

**What a caller still owns.**  Two things, and they are the two a transport
knows and this module cannot:

* the ``JaatoServer`` whose runtime holds the bus.  WebSocket resolves it
  from its client->session map (daemon mode) or from ``self._jaato_server``
  (standalone mode); ``SessionManager`` has the ``Session`` in hand already.
* the ``source`` label.  It is rendered to the model as ``Source: <x>`` by
  ``shared.event_bus_tools._format_event_notification``, so it must name the
  transport the event really came in on.  There is no default: a wrong
  source is a claim about provenance, and a module that guessed one would be
  making that claim on every caller's behalf.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# The ``source_agent`` every externally-injected bus event carries.  Bus
# subscribers drop events whose ``source_agent`` is their own name, so this
# must never collide with an agent id; the leading underscore is what keeps
# it out of that namespace.  The ``webhook`` plugin uses ``webhook:<route>``
# for the same reason.
EXTERNAL_EVENT_SOURCE_AGENT = "_external"


@dataclass(frozen=True)
class ExternalEventDelivery:
    """What became of one publish attempt.

    ``ok`` is the only field a caller must read.  It is ``False`` when the
    session had no ``EventBus`` to publish onto -- a real state rather than
    a defensive check: a session whose server has not finished initialising,
    or one whose runtime was torn down, has none.

    Attributes:
        ok: Whether the event reached a bus.  When ``False``, ``error`` says
            why and ``notified`` is 0.
        notified: How many subscribers the bus notified.  **Zero is a
            success**, and a common one -- an external event published before
            any agent has called ``subscribeToEvents`` notifies nobody and is
            still correctly delivered.  Callers must not read 0 as a failure.
        error: The human sentence; empty when ``ok``.
        event_id: The id stamped on the published bus event; empty when not
            ``ok``.  Returned so a caller can correlate a log line with the
            bus record rather than re-deriving the id.
    """

    ok: bool
    notified: int = 0
    error: str = ""
    event_id: str = ""


def publish_external_event(
    server: Any,
    *,
    name: str,
    data: Optional[Dict[str, Any]] = None,
    timestamp: str = "",
    source: str,
) -> ExternalEventDelivery:
    """Publish one external event onto ``server``'s session ``EventBus``.

    Args:
        server: The ``JaatoServer`` owning the target session.  Duck-typed
            rather than imported: ``server.core`` pulls in far more than this
            module needs, and the one attribute read -- ``_runtime.event_bus``
            -- is the same read ``JaatoServer.event_bus`` makes.  A ``None``
            server, or one with no runtime, is *answered* rather than raised
            on, because both are reachable states of a session being built or
            torn down.
        name: The event name the host chose, e.g. ``order.placed``.  Lands in
            the payload as ``event_type`` -- the key
            ``EventBusTools._execute_subscribe`` filters ``event_names``
            against, and the key ``_format_event_notification`` renders as
            ``Type:``.
        data: The host's arbitrary payload.  ``None`` is normalised to ``{}``:
            a name with no data is a legitimate ping, and a ``None`` here
            would reach the model as ``Data: None``.
        timestamp: ISO 8601, as the client sent it.  Empty means "the client
            did not stamp one" and this module stamps UTC now -- never the
            reverse, because the host's clock is the one that knows when the
            thing it is reporting happened.
        source: Which transport the request arrived on (``"websocket"`` /
            ``"ipc"``).  Keyword-only and without a default on purpose; see
            the module docstring.

    Returns:
        An :class:`ExternalEventDelivery`.  Never raises for a missing bus --
        both callers answer their client with the ``error`` string, and a
        transport handler that raised would take the connection down over a
        session that simply has no bus yet.
    """
    from jaato_sdk.event_bus import Event as BusEvent, EventType as BusEventType

    runtime = getattr(server, "_runtime", None) if server is not None else None
    bus = getattr(runtime, "event_bus", None) if runtime is not None else None
    if bus is None:
        return ExternalEventDelivery(
            ok=False, error="Session event bus not available",
        )

    stamp = timestamp or datetime.now(timezone.utc).isoformat()
    event_id = f"ext_{stamp}"
    bus_event = BusEvent(
        event_id=event_id,
        event_type=BusEventType.EXTERNAL_EVENT,
        timestamp=stamp,
        source_agent=EXTERNAL_EVENT_SOURCE_AGENT,
        payload={
            "source": source,
            "event_type": name,
            "data": data if data is not None else {},
        },
    )
    return ExternalEventDelivery(
        ok=True, notified=bus.publish(bus_event), event_id=event_id,
    )
