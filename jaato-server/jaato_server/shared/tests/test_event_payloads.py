"""Verify payload TypedDicts stay aligned with server event models.

The TypedDict payload schemas in ``jaato_sdk.event_payloads`` are the
single source of truth for bus event payloads.  Bridged events (server →
bus) are produced by flattening a server event's fields (minus
``type`` and ``timestamp``) into the bus event's ``payload`` dict.

This test ensures every key in the server event has a corresponding
key in the payload TypedDict, and vice versa.  If someone adds a
field to a server event but forgets to update the TypedDict (or
the reverse), this test fails.
"""

from typing import get_type_hints

import pytest

from jaato_sdk.event_payloads import BRIDGED_PAYLOAD_SOURCES
import jaato_sdk.events as events_module


# Fields on the base Event model that are NOT part of the payload.
_BASE_EVENT_FIELDS = {"type", "timestamp"}


def _get_server_event_payload_fields(event_class: type) -> set:
    """Extract payload-relevant field names from a server event model.

    Returns all pydantic field names except the base Event fields
    (type, timestamp) which are stripped during bus bridging.
    """
    return {
        name
        for name in event_class.model_fields
        if name not in _BASE_EVENT_FIELDS
    }


def _get_typed_dict_keys(td_class: type) -> set:
    """Extract all keys from a TypedDict (including inherited)."""
    return set(get_type_hints(td_class).keys())


@pytest.mark.parametrize(
    "payload_class,event_class_name",
    list(BRIDGED_PAYLOAD_SOURCES.items()),
    ids=[cls.__name__ for cls in BRIDGED_PAYLOAD_SOURCES.keys()],
)
def test_bridged_payload_matches_server_event(payload_class, event_class_name):
    """Payload TypedDict keys must match server event dataclass fields."""
    event_class = getattr(events_module, event_class_name)
    event_fields = _get_server_event_payload_fields(event_class)
    payload_keys = _get_typed_dict_keys(payload_class)

    missing_in_payload = event_fields - payload_keys
    extra_in_payload = payload_keys - event_fields

    errors = []
    if missing_in_payload:
        errors.append(
            f"Server event {event_class_name} has fields not in "
            f"{payload_class.__name__}: {sorted(missing_in_payload)}"
        )
    if extra_in_payload:
        errors.append(
            f"{payload_class.__name__} has keys not in server event "
            f"{event_class_name}: {sorted(extra_in_payload)}"
        )

    if errors:
        pytest.fail("\n".join(errors))
