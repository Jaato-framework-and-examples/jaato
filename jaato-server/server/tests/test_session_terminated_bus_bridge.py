"""Pin test for the SessionTerminatedEvent → EventBus bridge (Bug C).

Server 0.6.162+.  Pre-fix the ``_SERVER_TO_BUS`` mapping at
``core.py:127`` did NOT include ``EventType.SESSION_TERMINATED``,
so ``server.emit(SessionTerminatedEvent(...))`` bypassed the bus
entirely and reactor rules subscribing on the bus never saw it.
The third gap in the visibility-on-stoppage chain after Bug A
(premium 0.1.188 ``build_merged_view`` top-level pydantic hoist)
and Bug B (PR #188 cascade GC sweep).

Empirical motivation (peer 7:1 retry-40 iter-4, 2026-05-27 21:48:01):
build_judge session 20260527_214649 raised APIStatusError →
SessionTerminatedEvent(reason="error") emitted → default policy
unloaded session correctly → BUT
``orchestrator.cascade.on_session_error`` reactor rule never fired
→ ``cascade_aborted.json`` never written → cascade dead-end.

This pin verifies:
1. SessionTerminatedEvent is bridged to a BusEvent (not None).
2. The bus event carries ``reason`` / ``error_summary`` /
   ``error_type`` in its payload so reactor rules + matcher's
   payload-hoist see them.
3. End-to-end: the bus event's merged view satisfies
   ``where: reason == 'error'`` per the premium 0.1.188 matcher.
"""

from __future__ import annotations


def test_session_terminated_maps_to_bus_event():
    """``EventType.SESSION_TERMINATED`` is in ``_SERVER_TO_BUS``
    and ``_server_event_to_bus_event`` returns a non-None BusEvent
    carrying the terminal-error context."""
    from jaato_sdk.events import SessionTerminatedEvent
    from jaato_sdk.event_bus import EventType as BusEventType
    from server.core import _SERVER_TO_BUS, _server_event_to_bus_event

    # The map MUST contain the SESSION_TERMINATED entry.
    from jaato_sdk.events import EventType
    assert EventType.SESSION_TERMINATED in _SERVER_TO_BUS, (
        "EventType.SESSION_TERMINATED missing from _SERVER_TO_BUS — "
        "events will bypass the bus and reactor rules won't fire.  "
        "See Bug C diagnosis (server 0.6.162)."
    )
    assert _SERVER_TO_BUS[EventType.SESSION_TERMINATED] == (
        BusEventType.SESSION_TERMINATED
    )

    # The bridge function returns a non-None bus event.
    sdk_event = SessionTerminatedEvent(
        session_id="20260527_214649",
        agent_id="build_judge",
        reason="error",
        error_summary="APIStatusError from z.ai API",
        error_type="APIStatusError",
    )
    bus_event = _server_event_to_bus_event(sdk_event)
    assert bus_event is not None, (
        "_server_event_to_bus_event returned None for "
        "SessionTerminatedEvent — bus bridge missing."
    )
    assert bus_event.event_type == BusEventType.SESSION_TERMINATED

    # Payload carries the terminal-error context — reactor rule
    # matcher's payload-hoist needs these to evaluate where clauses.
    assert bus_event.payload["reason"] == "error"
    assert bus_event.payload["session_id"] == "20260527_214649"
    assert bus_event.payload["error_summary"] == (
        "APIStatusError from z.ai API"
    )
    assert bus_event.payload["error_type"] == "APIStatusError"


def test_session_terminated_bus_event_satisfies_reactor_where_clause():
    """End-to-end: the BusEvent produced by the bridge satisfies
    the canonical reactor rule pattern
    ``event_type=session.terminated where reason=='error'``.

    Validates the full chain: SDK event → bridge → bus event → matcher
    merged view → JMESPath evaluation.  If this passes, the
    cascade_on_session_error reactor rule will fire in production.
    """
    from jaato_sdk.events import SessionTerminatedEvent
    from server.core import _server_event_to_bus_event

    # Try to import the premium matcher; skip if premium isn't
    # installed in the test environment.
    try:
        from jaato_premium.reactors.matcher import (
            build_merged_view,
            matches_where,
        )
    except ImportError:
        import pytest
        pytest.skip("jaato_premium not installed; skipping end-to-end")

    sdk_event = SessionTerminatedEvent(
        agent_id="build_judge",
        reason="error",
        error_summary="some-error",
        error_type="APIStatusError",
    )
    bus_event = _server_event_to_bus_event(sdk_event)
    assert bus_event is not None

    view = build_merged_view(bus_event)
    # event_type comes from the bus envelope.
    assert view["event_type"] == "session.terminated"
    # reason comes from the bus payload (tier 3 hoist).
    assert view["reason"] == "error"
    # The canonical reactor rule's where clause must evaluate True.
    assert matches_where("reason == 'error'", view) is True
    # Sanity: a different reason doesn't match.
    sdk_event_natural = SessionTerminatedEvent(reason="natural")
    bus_event_natural = _server_event_to_bus_event(sdk_event_natural)
    assert bus_event_natural is not None
    view_natural = build_merged_view(bus_event_natural)
    assert matches_where("reason == 'error'", view_natural) is False
