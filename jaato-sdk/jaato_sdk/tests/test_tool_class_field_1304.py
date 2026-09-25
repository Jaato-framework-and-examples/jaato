"""``tool_class`` on ``ToolCallStartEvent`` / ``PermissionRequestedEvent``
(the final rollout item from jaato/#1304, "Permissions + diffs" phase 3).

Both events already existed; this field is additive, and the tree's own
documented lesson (CLAUDE.md's #823 / "Three Rows the Web Client Drew
That Nobody Sent") is that a daemon-emitted key an SDK model does not
DECLARE is silently dropped on ingest -- so the load-bearing property to
pin here is not "the field exists on the Python model" but "a value set
on the wire survives serialize -> deserialize -> re-serialize", which is
exactly the failure mode that bit ``WorkspaceCreatedEvent.workspace`` and
``ToolOutputEvent.mime_type`` before it.
"""

from __future__ import annotations

import json

from jaato_sdk.events import (
    Event,
    PermissionRequestedEvent,
    ToolCallStartEvent,
    deserialize_event,
    serialize_event,
)


def _round_trip(event: Event) -> Event:
    return deserialize_event(serialize_event(event))


def test_tool_call_start_event_declares_tool_class() -> None:
    """A pre-#823-shaped SDK model silently drops an undeclared field;
    the field must actually exist on the pydantic model, not merely on
    the daemon's wire dict."""
    assert "tool_class" in ToolCallStartEvent.model_fields


def test_permission_requested_event_declares_tool_class() -> None:
    assert "tool_class" in PermissionRequestedEvent.model_fields


def test_tool_call_start_event_tool_class_round_trips() -> None:
    original = ToolCallStartEvent(
        agent_id="main",
        tool_name="createPlan",
        tool_args={},
        call_id="call_1",
        tool_class="housekeeping",
    )
    restored = _round_trip(original)
    assert isinstance(restored, ToolCallStartEvent)
    assert restored.tool_class == "housekeeping"


def test_permission_requested_event_tool_class_round_trips() -> None:
    original = PermissionRequestedEvent(
        agent_id="main",
        request_id="r-1",
        tool_name="writeNewFile",
        tool_args={"path": "x.py"},
        response_options=[{"key": "y", "label": "Allow", "action": "allow"}],
        tool_class="write",
    )
    restored = _round_trip(original)
    assert isinstance(restored, PermissionRequestedEvent)
    assert restored.tool_class == "write"


def test_tool_class_defaults_to_none_and_survives_absence() -> None:
    """A daemon that predates this field (or a call site that never
    classified the tool) must not have the field manufactured as a
    string -- absence stays absence through the round trip."""
    original = ToolCallStartEvent(agent_id="main", tool_name="mystery")
    assert original.tool_class is None
    restored = _round_trip(original)
    assert isinstance(restored, ToolCallStartEvent)
    assert restored.tool_class is None


def test_an_older_daemon_payload_with_no_tool_class_key_still_deserializes() -> None:
    """Simulates the wire payload an OLDER daemon (below this field)
    would send: no ``tool_class`` key at all, rather than an explicit
    ``null``.  Must degrade to ``None``, not raise."""
    payload = {
        "type": "tool.call_start",
        "agent_id": "main",
        "tool_name": "cli_based_tool",
        "tool_args": {},
        "call_id": "call_2",
    }
    event = deserialize_event(json.dumps(payload))
    assert isinstance(event, ToolCallStartEvent)
    assert event.tool_class is None
