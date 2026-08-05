"""Daemon-wide reactor EventBus foundation: per-session buses SINK into one
daemon-wide bus, exposed on _ExtensionContext.event_bus.

Reactors subscribe ONCE to the daemon-wide bus (instead of per-loaded-session),
so they receive events from ALL sessions and survive session unloads. See
docs/design/reactor-bus-session-scope.md.
"""
from shared.event_bus import EventBus
from jaato_sdk.event_bus import EventFilter, Event, EventType


def test_reactor_sink_forwards_per_session_events_to_daemon_wide_bus():
    session_bus = EventBus()
    reactor_bus = EventBus()
    # Wire the sink exactly as SessionManager does at session build:
    session_bus.subscribe(
        subscriber_name="reactor_bus_sink", filter=EventFilter(),
        callback=reactor_bus.publish, replay_history=False)
    # A reactor subscribes ONCE to the daemon-wide bus:
    received = []
    reactor_bus.subscribe(
        subscriber_name="reactor", filter=EventFilter(),
        callback=received.append, replay_history=False)
    # An event on the per-session bus is forwarded to the daemon-wide bus:
    evt = Event.create(next(iter(EventType)), "session-A", {"k": "v"})
    session_bus.publish(evt)
    assert len(received) == 1 and received[0] is evt


def test_two_session_buses_both_sink_into_one_reactor_bus():
    reactor_bus = EventBus()
    got = []
    reactor_bus.subscribe("reactor", EventFilter(), callback=got.append,
                          replay_history=False)
    buses = [EventBus(), EventBus()]
    for b in buses:
        b.subscribe("reactor_bus_sink", EventFilter(),
                    callback=reactor_bus.publish, replay_history=False)
    et = next(iter(EventType))
    buses[0].publish(Event.create(et, "A", {}))
    buses[1].publish(Event.create(et, "B", {}))
    # One daemon-wide subscription sees events from BOTH sessions.
    assert sorted(e.source_agent for e in got) == ["A", "B"]


def test_extension_context_carries_event_bus():
    from server.__main__ import _ExtensionContext
    bus = EventBus()
    ctx = _ExtensionContext(
        session_manager=None, ws_server=None, web_socket=None, ipc_socket=None,
        server_name=None, dashboard_port=None, event_bus=bus)
    assert ctx.event_bus is bus


def test_session_manager_creates_a_reactor_event_bus():
    from server.session_manager import SessionManager
    sm = SessionManager()
    assert isinstance(sm.reactor_event_bus, EventBus)
