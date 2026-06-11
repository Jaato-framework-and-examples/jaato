"""Pin test for the SlotSettledEvent → EventBus bridge.

The framework emits ``SlotSettledEvent`` once per cascade stage at the end of
``JaatoServer.shutdown`` — on ALL teardown paths (warm slot returned, slot torn
down on error, cold-spawned) — so a cascade reactor can gate the next stage's
spawn on one universal, stall-proof event (no timeout).  For reactor rules
(which subscribe on the bus) to see it, ``EventType.SLOT_SETTLED`` must be in
``_SERVER_TO_BUS`` — this pin guards that wiring + the payload contract.

Supersedes the warm-only ``SlotReusableEvent`` (which didn't fire for
cold-spawned stages and could stall a pure-reactor handoff).
"""

from __future__ import annotations


def test_slot_settled_maps_to_bus_event():
    """``EventType.SLOT_SETTLED`` is in ``_SERVER_TO_BUS`` and the bridge
    returns a non-None BusEvent carrying the full handoff context."""
    from jaato_sdk.events import SlotSettledEvent, EventType
    from jaato_sdk.event_bus import EventType as BusEventType
    from server.core import _SERVER_TO_BUS, _server_event_to_bus_event

    assert EventType.SLOT_SETTLED in _SERVER_TO_BUS, (
        "EventType.SLOT_SETTLED missing from _SERVER_TO_BUS — the event "
        "bypasses the bus and cascade reactor rules won't fire."
    )
    assert _SERVER_TO_BUS[EventType.SLOT_SETTLED] == BusEventType.SLOT_SETTLED

    bus_event = _server_event_to_bus_event(SlotSettledEvent(
        session_id="20260611_120000",
        agent_id="codegen",
        cascade_driver_id="cascade-abc",
        was_warm=True,
        pool_slot_pid=4242,
    ))
    assert bus_event is not None
    assert bus_event.event_type == BusEventType.SLOT_SETTLED
    p = bus_event.payload
    assert p["session_id"] == "20260611_120000"
    assert p["agent_id"] == "codegen"
    assert p["cascade_driver_id"] == "cascade-abc"
    assert p["was_warm"] is True
    assert p["pool_slot_pid"] == 4242


def test_slot_settled_serializes_with_type_marker():
    """SDK event round-trips with the ``slot.settled`` type marker."""
    from jaato_sdk.events import SlotSettledEvent, EventType

    d = SlotSettledEvent(
        session_id="s", agent_id="discovery", cascade_driver_id="c",
        was_warm=False, pool_slot_pid=0,
    ).to_dict()
    assert d["type"] == EventType.SLOT_SETTLED.value == "slot.settled"
    assert d["was_warm"] is False and d["pool_slot_pid"] == 0
    assert d["agent_id"] == "discovery"


def test_slot_settled_cold_path_still_carries_correlation():
    """The COLD case (was_warm=False, no pool slot) — the exact stages that
    would stall a warm-only event — still carries session_id/agent_id/
    cascade_driver_id so the reactor's gate fires + routes correctly."""
    from jaato_sdk.events import SlotSettledEvent
    from server.core import _server_event_to_bus_event

    bus_event = _server_event_to_bus_event(SlotSettledEvent(
        session_id="s-cold", agent_id="discovery",
        cascade_driver_id="cascade-xyz", was_warm=False, pool_slot_pid=0,
    ))
    assert bus_event is not None
    p = bus_event.payload
    assert p["cascade_driver_id"] == "cascade-xyz"
    assert p["agent_id"] == "discovery"
    assert p["was_warm"] is False


def test_slot_settled_bus_event_satisfies_cascade_correlation():
    """End-to-end: the BusEvent satisfies a reactor where-clause correlating
    on cascade_driver_id (the gating predicate a cascade reactor uses)."""
    from jaato_sdk.events import SlotSettledEvent
    from server.core import _server_event_to_bus_event

    try:
        from jaato_premium.reactors.matcher import (
            build_merged_view,
            matches_where,
        )
    except ImportError:
        import pytest
        pytest.skip("jaato_premium not installed; skipping end-to-end")

    bus_event = _server_event_to_bus_event(SlotSettledEvent(
        session_id="s1", agent_id="codegen",
        cascade_driver_id="cascade-xyz", was_warm=True, pool_slot_pid=7,
    ))
    assert bus_event is not None
    view = build_merged_view(bus_event)
    assert view["event_type"] == "slot.settled"
    assert view["cascade_driver_id"] == "cascade-xyz"
    assert matches_where("cascade_driver_id == 'cascade-xyz'", view) is True
    assert matches_where("cascade_driver_id == 'other'", view) is False
