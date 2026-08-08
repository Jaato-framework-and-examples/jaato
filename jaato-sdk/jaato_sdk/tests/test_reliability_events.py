"""Reliability reactor event types are present + resolvable on the bus enum.

These are the emit substrate for the tenant-authored reliability reactor (the
event-driven successor to the in-process reliability plugin — see
docs/design/reliability-event-driven-migration.md).  The reactor publishes them
via ``ActionContext.emit_event(event_type_str, payload)``, which resolves the
string against ``jaato_sdk.event_bus.EventType`` by ``.value`` and DROPS the
event with a warning if no member matches.  A reactor rule subscribes by the
same ``.value`` string.  So the contract these tests pin is: the member exists
AND its value is exactly the wire string.
"""

from jaato_sdk.event_bus import EventType


def _resolve_by_value(value: str):
    """Mirror ActionContext.emit_event's lookup (member whose .value matches)."""
    for member in EventType:
        if member.value == value:
            return member
    return None


def test_reliability_escalated_present_and_resolvable():
    assert EventType.RELIABILITY_ESCALATED.value == "reliability.escalated"
    assert _resolve_by_value("reliability.escalated") is EventType.RELIABILITY_ESCALATED


def test_reliability_pattern_detected_present_and_resolvable():
    assert EventType.RELIABILITY_PATTERN_DETECTED.value == "reliability.pattern_detected"
    assert (
        _resolve_by_value("reliability.pattern_detected")
        is EventType.RELIABILITY_PATTERN_DETECTED
    )


def test_reliability_values_are_unique_on_the_bus_enum():
    # No accidental collision with an existing member (which would shadow the
    # reactor's emit/subscribe resolution).
    values = [m.value for m in EventType]
    assert values.count("reliability.escalated") == 1
    assert values.count("reliability.pattern_detected") == 1
