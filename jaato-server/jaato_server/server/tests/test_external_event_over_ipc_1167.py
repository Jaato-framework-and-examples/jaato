"""An external event reaches the bus on BOTH transports, from one definition (#1167).

THE GAP.  ``ExternalEventRequest`` was handled in exactly one place in the
tree -- ``JaatoWSServer._handle_external_event``.  ``session_manager.py``,
``command_router.py`` and ``ipc.py`` contained zero occurrences, so an IPC
client's request fell through every ``isinstance`` arm of
``SessionManager.handle_request`` and hit the final ``else``::

    ErrorEvent(error="Unknown request type: ExternalEventRequest",
               error_type="RequestError")

Loud rather than silent, which is the good direction and bounds the
severity -- this is a missing feature that announces itself, not one of the
silent-ignore family (#910 / #925 / #947 / #950 / #1133).  What it cost is
still real: ``_handle_external_event``'s publish was **the only**
``EXTERNAL_EVENT`` producer under ``server/``, and each per-session bus
sinks into ``SessionManager.reactor_event_bus``, so an IPC-only deployment
had a daemon-wide reactor engine nothing could externally trigger.  The one
workaround -- the ``webhook`` plugin's listener, the tree's other producer
-- costs a bound port, TLS, an allowlist and a signed route to deliver an
event from a process already holding an authenticated socket to the same
daemon.

WHY THE FIX IS TWO PIECES.  Adding the IPC dispatch alone would have given
the tree TWO answers to "what does an external event look like on the bus",
which it treats as a defect in its own right: each copy masks the other, so
neither can be shown to do anything (the ``on_unmetered`` validation the
reversion meta-guard correctly called decorative, #688).  So the translation
moved to :mod:`server.external_event` and both transports call it, keeping
only what is genuinely theirs -- resolving WHICH session the caller drives.

WHAT EACH TEST WOULD MISS IF NEUTERED
-------------------------------------

* :func:`test_an_external_event_over_ipc_reaches_the_session_bus` is the
  issue itself.  Neuter it -- assert on the absence of an ``ErrorEvent``
  rather than on the bus -- and a dispatch that routes somewhere harmless
  passes while no reactor is ever triggered.  It asserts the PUBLISHED
  event, because reaching the arm is not the thing that was missing.
* :func:`test_the_publish_has_exactly_one_definition_under_server` is the
  source guard, and it is the one that cannot be replaced by behaviour: a
  second, correct copy in ``websocket.py`` passes every behavioural test
  here by construction.  That is what makes duplication expensive and
  invisible at once.
* :func:`test_both_transports_publish_the_same_shape` is what stops the
  source guard passing for the wrong reason.  One definition that the two
  transports call with different arguments is one definition in name only,
  so the two bus events are compared field by field -- ``source`` excepted,
  which is the one thing they are SUPPOSED to disagree about.
* :func:`test_websocket_still_intercepts_before_the_router` pins the
  invariant that makes ``source="ipc"`` honest on a transport-agnostic
  method.  ``SessionManager.handle_request`` serves both transports; WS is
  the only one that never reaches it with an ``ExternalEventRequest``,
  because ``_handle_message`` returns first.  Remove that interception and
  WS traffic starts arriving at the IPC arm and being labelled ``ipc`` to
  the model, in the one payload field whose job is to say where the event
  came from.
* :func:`test_a_session_with_no_bus_is_refused_by_name` pins the failure
  answer.  Without it, "the request is dispatched" is satisfiable by a
  dispatch that raises on the IPC executor thread and takes the connection
  with it.

VERIFIED NON-VACUOUS: each registered reversion below fails its named test
and no other.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from jaato_sdk.event_bus import EventType as BusEventType
from jaato_sdk.events import ExternalEventRequest
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.tests.reversion import Reversion


_SESSION_MANAGER = "jaato-server/jaato_server/server/session_manager.py"
_WEBSOCKET = "jaato-server/jaato_server/server/websocket.py"

#: The IPC dispatch arm, and its absence -- which is ``main`` before #1167.
#: Removing it does not break the file: the request simply falls through to
#: the ``Unknown request type`` else, exactly as it did.
_IPC_ARM_FIXED = """        elif isinstance(event, ExternalEventRequest):
            self._handle_external_event_request(client_id, session, event)

"""

#: The WS handler's one call into the shared translation, and a local
#: re-implementation of it.  The replacement is BEHAVIOURALLY IDENTICAL --
#: that is the point of the source guard it trips, since a second correct
#: copy is exactly what no behavioural test can see.
_WS_CALL_FIXED = """        delivery = publish_external_event(
            owner,
            name=event.name,
            data=event.data,
            timestamp=event.timestamp,
            source="websocket",
        )
"""
_WS_CALL_BROKEN = """        from jaato_sdk.event_bus import (
            Event as BusEvent, EventType as BusEventType,
        )
        from jaato_server.server.external_event import ExternalEventDelivery
        _runtime = getattr(owner, "_runtime", None)
        _bus = getattr(_runtime, "event_bus", None) if _runtime else None
        if _bus is None:
            delivery = ExternalEventDelivery(
                ok=False, error="Session event bus not available")
        else:
            _stamp = event.timestamp or datetime.now(timezone.utc).isoformat()
            delivery = ExternalEventDelivery(ok=True, notified=_bus.publish(
                BusEvent(
                    event_id=f"ext_{_stamp}",
                    event_type=BusEventType.EXTERNAL_EVENT,
                    timestamp=_stamp,
                    source_agent="_external",
                    payload={
                        "source": "websocket",
                        "event_type": event.name,
                        "data": event.data,
                    },
                )))
"""

#: The WS interception, and its absence.  Without it an
#: ``ExternalEventRequest`` falls through to ``_handle_message_daemon`` and
#: reaches ``SessionManager.handle_request`` -- which now serves it, and
#: labels it ``source="ipc"``.
_WS_INTERCEPT_FIXED = """        from jaato_sdk.events import ExternalEventRequest
        if isinstance(event, ExternalEventRequest):
            await self._handle_external_event(client_id, event)
            return
"""
_WS_INTERCEPT_BROKEN = """        from jaato_sdk.events import ExternalEventRequest  # noqa: F401
"""


REVERSIONS = [
    Reversion(
        target=_SESSION_MANAGER,
        find=_IPC_ARM_FIXED,
        replace="",
        test="test_an_external_event_over_ipc_reaches_the_session_bus",
        because=(
            "an IPC client having no route to publish an EXTERNAL_EVENT, so "
            "an IPC-only deployment runs a reactor engine nothing can "
            "externally trigger"
        ),
    ),
    Reversion(
        target=_SESSION_MANAGER,
        find=_IPC_ARM_FIXED,
        replace="",
        test="test_a_session_with_no_bus_is_refused_by_name",
        because=(
            "the IPC dispatch being gone, so a session with no bus is "
            "answered 'Unknown request type' rather than by name"
        ),
    ),
    Reversion(
        target=_WEBSOCKET,
        find=_WS_CALL_FIXED,
        replace=_WS_CALL_BROKEN,
        test="test_the_publish_has_exactly_one_definition_under_server",
        because=(
            "a second, behaviourally identical definition of what an "
            "external event looks like on the bus -- the shape no "
            "behavioural test can see"
        ),
    ),
    Reversion(
        target=_WEBSOCKET,
        find=_WS_INTERCEPT_FIXED,
        replace=_WS_INTERCEPT_BROKEN,
        test="test_websocket_still_intercepts_before_the_router",
        because=(
            "WS traffic falling through to the transport-agnostic IPC arm, "
            "which labels every external event source='ipc'"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _RecordingBus:
    """The one thing under test: what was published, verbatim.

    ``publish`` returns a subscriber count so the production code's
    ``notified`` plumbing is exercised rather than stubbed to zero -- the
    difference between "no subscriber" and "not delivered" is a distinction
    :class:`~server.external_event.ExternalEventDelivery` documents, and a
    double returning a constant 0 would let a regression in it pass.
    """

    def __init__(self, notified: int = 2) -> None:
        self.published: List[Any] = []
        self._notified = notified

    def publish(self, event: Any) -> int:
        self.published.append(event)
        return self._notified


def _session(bus: Any) -> SimpleNamespace:
    """A ``Session``-shaped double whose server's runtime holds ``bus``.

    ``bus=None`` models a session whose runtime has no ``EventBus`` -- being
    built, or torn down -- which is the refusal path, not a contrived one.
    """
    runtime = SimpleNamespace(event_bus=bus)
    server = SimpleNamespace(_runtime=runtime, get_all_session_env=lambda: {})
    return SimpleNamespace(
        server=server,
        session_id="sess-1",
        workspace_path="/ws",
        is_dirty=False,
        last_activity="",
    )


def _manager(emitted: List[Tuple[str, Any]], bus: Any) -> SessionManager:
    """A bare ``SessionManager`` that resolves one session holding ``bus``."""
    sm = SessionManager.__new__(SessionManager)
    sm._event_callback = lambda cid, ev: emitted.append((cid, ev))
    sm._client_to_session = {}
    sm._save_session = lambda session: None
    sm.get_session = lambda sid: _session(bus)
    return sm


def _drive_ipc(bus: Any, **kwargs: Any) -> List[Tuple[str, Any]]:
    """Put one ``ExternalEventRequest`` through the real IPC entry point."""
    emitted: List[Tuple[str, Any]] = []
    sm = _manager(emitted, bus)
    sm.handle_request("client-1", "sess-1", ExternalEventRequest(**kwargs))
    return emitted


def _ws_server(bus: Any, router_calls: List[Any]) -> Any:
    """A ``JaatoWSServer``-shaped double in DAEMON mode.

    Daemon mode is the configuration that matters here: it is the one with a
    ``CommandRouter`` behind it, so it is the one where a lost interception
    would silently reroute WS traffic onto the IPC arm.
    """
    from jaato_server.server.websocket import JaatoWSServer

    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._lock = asyncio.Lock()
    ws._clients = {}
    ws._message_handlers = {}
    ws._event_sink_adapter = SimpleNamespace(_client_sessions={"client-1": "sess-1"})
    ws._jaato_server = None
    ws._command_router = SimpleNamespace(
        _session_manager=SimpleNamespace(get_session=lambda sid: _session(bus)),
    )
    ws._workspace_manager = None

    sent: List[str] = []
    ws._send_error = lambda cid, msg: _async_none(sent.append(msg))
    ws._handle_message_daemon = lambda *a, **kw: _async_none(
        router_calls.append((a, kw))
    )
    ws._ws_errors = sent
    return ws


async def _async_none(_ignored: Any = None) -> None:
    """Await-able no-op, so a double can stand in for a coroutine method."""
    return None


# ---------------------------------------------------------------------------
# The issue
# ---------------------------------------------------------------------------


def test_an_external_event_over_ipc_reaches_the_session_bus():
    """#1167 -- the request now publishes instead of being refused.

    Asserts the PUBLISHED bus event rather than the absence of an error: a
    dispatch that reaches the arm and routes somewhere harmless would pass
    the weaker assertion while no reactor is ever triggered, which is the
    whole complaint.
    """
    bus = _RecordingBus()
    emitted = _drive_ipc(
        bus, name="order.placed", data={"id": 7},
        timestamp="2026-09-20T12:00:00+00:00",
    )

    assert [type(ev).__name__ for _, ev in emitted] == [], (
        "an ExternalEventRequest over IPC was answered rather than "
        f"published: {[(c, getattr(e, 'error', e)) for c, e in emitted]}"
    )
    assert len(bus.published) == 1, (
        "nothing reached the session's EventBus, so no reactor was triggered"
    )
    (published,) = bus.published
    assert published.event_type is BusEventType.EXTERNAL_EVENT
    assert published.payload["event_type"] == "order.placed"
    assert published.payload["data"] == {"id": 7}
    # The transport names itself, because ``_format_event_notification``
    # renders this to the model as ``Source: <x>``.
    assert published.payload["source"] == "ipc"


def test_a_session_with_no_bus_is_refused_by_name():
    """The failure answer is an ``ExternalEventError``, not a raise.

    This runs on the IPC executor thread; a handler that raised would cost
    the connection rather than the request.
    """
    emitted = _drive_ipc(None, name="order.placed")

    kinds = [(type(ev).__name__, getattr(ev, "error_type", "")) for _, ev in emitted]
    assert kinds == [("ErrorEvent", "ExternalEventError")], kinds


def test_an_absent_payload_is_published_as_an_empty_dict():
    """``data=None`` reaches the model as ``{}``, never as ``Data: None``.

    A name with no data is a legitimate ping, so this is the normal shape of
    a signalling event rather than an edge case.
    """
    bus = _RecordingBus()
    _drive_ipc(bus, name="build.finished")

    (published,) = bus.published
    assert published.payload["data"] == {}


# ---------------------------------------------------------------------------
# One definition
# ---------------------------------------------------------------------------


def _external_event_publish_sites() -> Dict[str, int]:
    """Every module under ``server/`` naming ``EventType.EXTERNAL_EVENT``.

    An AST walk rather than a text scan, so the module's own prose -- which
    says the words repeatedly -- does not count as a producer.  Test modules
    are excluded: a guard asserting the shape has to be able to name it.
    """
    root = Path(__file__).resolve().parents[1]
    sites: Dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "EXTERNAL_EVENT"
        )
        if hits:
            sites[str(path.relative_to(root))] = hits
    return sites


def test_the_publish_has_exactly_one_definition_under_server():
    """Only ``external_event.py`` may say what an external event looks like.

    The source-level guard, and the only one of these that a second CORRECT
    copy cannot satisfy: duplication here changes no behaviour at all, which
    is precisely why it survives and why the two copies then drift.
    """
    sites = _external_event_publish_sites()
    assert set(sites) == {"external_event.py"}, (
        "EventType.EXTERNAL_EVENT is constructed outside "
        "server/external_event.py -- that is a second answer to what an "
        f"external event looks like on the bus: {sites}"
    )


def test_both_transports_publish_the_same_shape():
    """One definition called two ways is one definition in name only.

    Everything but ``source`` must agree, because ``source`` is the one
    thing the two transports are supposed to disagree about.
    """
    ipc_bus = _RecordingBus()
    _drive_ipc(
        ipc_bus, name="ticket.assigned", data={"n": 1},
        timestamp="2026-09-20T12:00:00+00:00",
    )

    ws_bus = _RecordingBus()
    router_calls: List[Any] = []
    ws = _ws_server(ws_bus, router_calls)
    asyncio.run(ws._handle_external_event(
        "client-1",
        ExternalEventRequest(
            name="ticket.assigned", data={"n": 1},
            timestamp="2026-09-20T12:00:00+00:00",
        ),
    ))

    (over_ipc,) = ipc_bus.published
    (over_ws,) = ws_bus.published

    assert over_ipc.event_id == over_ws.event_id
    assert over_ipc.event_type is over_ws.event_type
    assert over_ipc.timestamp == over_ws.timestamp
    assert over_ipc.source_agent == over_ws.source_agent
    assert {k: v for k, v in over_ipc.payload.items() if k != "source"} == {
        k: v for k, v in over_ws.payload.items() if k != "source"
    }
    assert (over_ipc.payload["source"], over_ws.payload["source"]) == (
        "ipc", "websocket",
    )


def test_websocket_still_intercepts_before_the_router():
    """What makes ``source="ipc"`` honest on a transport-agnostic method.

    ``SessionManager.handle_request`` serves both transports and cannot ask
    which one it is on.  It may label the event ``ipc`` only because WS
    returns before delegating to the ``CommandRouter``, so that ordering is
    an invariant rather than an observation.
    """
    bus = _RecordingBus()
    router_calls: List[Any] = []
    ws = _ws_server(bus, router_calls)

    from jaato_sdk.events import serialize_event

    asyncio.run(ws._handle_message(
        "client-1",
        serialize_event(ExternalEventRequest(name="order.placed")),
    ))

    assert router_calls == [], (
        "an ExternalEventRequest reached the CommandRouter, so it will be "
        "served by the IPC arm and labelled source='ipc' to the model"
    )
    assert len(bus.published) == 1, ws._ws_errors
    assert bus.published[0].payload["source"] == "websocket"
