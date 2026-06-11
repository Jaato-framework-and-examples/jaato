"""Pin test for the SlotReusableEvent → EventBus bridge.

The framework emits ``SlotReusableEvent`` at
``JaatoServer.shutdown`` → ``PoolManager.return_slot_after_session`` for
cascade sessions, so a cascade reactor can gate the next stage's spawn on
warm-slot readiness (guaranteeing reuse instead of cold-spawn).  For that to
work the event must reach the bus (reactor rules subscribe on the bus) — i.e.
``EventType.SLOT_REUSABLE`` must be in ``_SERVER_TO_BUS``, exactly like the
SessionTerminatedEvent Bug-C fix.  This pin guards that wiring.
"""

from __future__ import annotations


def test_slot_reusable_maps_to_bus_event():
    """``EventType.SLOT_REUSABLE`` is in ``_SERVER_TO_BUS`` and the bridge
    returns a non-None BusEvent carrying the slot/cascade context."""
    from jaato_sdk.events import SlotReusableEvent, EventType
    from jaato_sdk.event_bus import EventType as BusEventType
    from server.core import _SERVER_TO_BUS, _server_event_to_bus_event

    assert EventType.SLOT_REUSABLE in _SERVER_TO_BUS, (
        "EventType.SLOT_REUSABLE missing from _SERVER_TO_BUS — the event "
        "bypasses the bus and cascade reactor rules won't fire (same class "
        "of bug as SESSION_TERMINATED Bug C)."
    )
    assert _SERVER_TO_BUS[EventType.SLOT_REUSABLE] == BusEventType.SLOT_REUSABLE

    sdk_event = SlotReusableEvent(
        session_id="20260611_120000",
        cascade_driver_id="cascade-abc",
        pool_slot_pid=4242,
    )
    bus_event = _server_event_to_bus_event(sdk_event)
    assert bus_event is not None, (
        "_server_event_to_bus_event returned None for SlotReusableEvent — "
        "bus bridge missing."
    )
    assert bus_event.event_type == BusEventType.SLOT_REUSABLE
    # Payload carries the correlation keys a reactor's where-clause needs.
    assert bus_event.payload["session_id"] == "20260611_120000"
    assert bus_event.payload["cascade_driver_id"] == "cascade-abc"
    assert bus_event.payload["pool_slot_pid"] == 4242


def test_slot_reusable_serializes_with_type_marker():
    """SDK event round-trips with the ``slot.reusable`` type marker so wire
    clients + the deserialization registry recognize it."""
    from jaato_sdk.events import SlotReusableEvent, EventType

    d = SlotReusableEvent(
        session_id="s", cascade_driver_id="c", pool_slot_pid=9,
    ).to_dict()
    assert d["type"] == EventType.SLOT_REUSABLE.value == "slot.reusable"
    assert d["cascade_driver_id"] == "c" and d["pool_slot_pid"] == 9


def test_slot_reusable_bus_event_satisfies_cascade_correlation():
    """End-to-end: the BusEvent satisfies a reactor where-clause correlating
    on ``cascade_driver_id`` (the gating predicate a cascade reactor uses to
    recognize *its* slot is ready)."""
    from jaato_sdk.events import SlotReusableEvent
    from server.core import _server_event_to_bus_event

    try:
        from jaato_premium.reactors.matcher import (
            build_merged_view,
            matches_where,
        )
    except ImportError:
        import pytest
        pytest.skip("jaato_premium not installed; skipping end-to-end")

    bus_event = _server_event_to_bus_event(SlotReusableEvent(
        session_id="s1", cascade_driver_id="cascade-xyz", pool_slot_pid=7,
    ))
    assert bus_event is not None
    view = build_merged_view(bus_event)
    assert view["event_type"] == "slot.reusable"
    assert view["cascade_driver_id"] == "cascade-xyz"
    assert matches_where("cascade_driver_id == 'cascade-xyz'", view) is True
    assert matches_where("cascade_driver_id == 'other'", view) is False
