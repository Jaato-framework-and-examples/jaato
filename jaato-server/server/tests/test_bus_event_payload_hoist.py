"""Tests for typed-payload hoisting in ``_server_event_to_bus_event``.

Cascade typed-handoff fix.  ``AgentCompletedEvent.payload`` carries the
validated ``signal_completion`` payload (when the producing stage declared a
``completion_payload_schema``).  Reactor consumers read it as flat fields on
the merged view (``event.get("facts")``).  But ``to_dict()`` leaves that
typed payload nested under a ``payload`` key, and ``build_merged_view`` hoists
only one level (``view.update(event.payload)``) — so on the bus hop the
reactor actually receives, ``event.get("facts")`` returned None even though a
validated payload was present.  This was misdiagnosed as a "bus serialize
drop"; the data wasn't dropped, just nested one level too deep.

The fix hoists the nested typed payload to the bus-event top level (without
clobbering envelope identity fields), so a single ``view.update`` surfaces it.
"""

from __future__ import annotations

import pytest

from server.core import _server_event_to_bus_event
from jaato_sdk.events import AgentCompletedEvent


def test_hoists_typed_payload_to_top_level() -> None:
    bus = _server_event_to_bus_event(
        AgentCompletedEvent(agent_id="extract", payload={"facts": "X"}),
    )
    assert bus is not None
    # Hoisted → reactor's event.get("facts") path works.
    assert bus.payload["facts"] == "X"
    # Nested payload preserved for back-compat.
    assert bus.payload["payload"] == {"facts": "X"}
    # Envelope identity intact.
    assert bus.payload["agent_id"] == "extract"
    assert bus.source_agent == "extract"


def test_no_typed_payload_is_noop() -> None:
    """payload=None (no completion_payload_schema declared) → no hoist,
    no crash; nothing spurious appears at the top level."""
    bus = _server_event_to_bus_event(
        AgentCompletedEvent(agent_id="extract", payload=None),
    )
    assert bus is not None
    assert bus.payload["payload"] is None
    assert "facts" not in bus.payload


def test_typed_payload_does_not_clobber_envelope_fields() -> None:
    """A typed-payload key colliding with an envelope field (e.g. the kb
    author names a payload field ``success``) must NOT overwrite the event's
    own envelope value — setdefault keeps the envelope authoritative; the
    nested payload still carries the author's value."""
    bus = _server_event_to_bus_event(
        AgentCompletedEvent(
            agent_id="extract",
            success=True,
            payload={"success": False, "facts": "X"},
        ),
    )
    assert bus.payload["success"] is True          # envelope wins
    assert bus.payload["facts"] == "X"             # non-colliding field hoisted
    assert bus.payload["payload"]["success"] is False  # nested intact


def test_multiple_typed_fields_all_hoisted() -> None:
    bus = _server_event_to_bus_event(
        AgentCompletedEvent(
            agent_id="extract",
            payload={"facts": "X", "confidence": 0.9, "sources": ["a", "b"]},
        ),
    )
    assert bus.payload["facts"] == "X"
    assert bus.payload["confidence"] == 0.9
    assert bus.payload["sources"] == ["a", "b"]


def test_end_to_end_merged_view_surfaces_typed_payload() -> None:
    """The full cascade path: bus event → reactor build_merged_view →
    ``event.get("facts")``.  This is the exact path that returned None before
    the fix.  Skips when jaato-premium (the reactor engine) isn't installed."""
    try:
        from jaato_premium.reactors.matcher import build_merged_view
    except ImportError:
        pytest.skip("jaato-premium not installed")
    bus = _server_event_to_bus_event(
        AgentCompletedEvent(agent_id="extract", payload={"facts": "X"}),
    )
    view = build_merged_view(bus)
    assert view.get("facts") == "X"
    # The where-predicate path benefits too (e.g. where: facts != null).
    assert view.get("source_agent") == "extract"
