"""Event bus types for cross-agent and cross-plugin event coordination.

This module defines the core types for the shared event bus that enables:
- Cross-agent task coordination (plan/step lifecycle events)
- External event ingestion (webhooks, WebSocket streams)
- Plugin-to-plugin communication

These types are consumed by the ``EventBus`` class in
``jaato-server/shared/event_bus.py`` and by any plugin or session code
that publishes or subscribes to events.

Previously these types lived in ``jaato_sdk.plugins.todo.models``.
They were extracted because the event bus is cross-cutting infrastructure,
not a todo-plugin concern.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    """Types of events published to the shared event bus.

    Includes both task lifecycle events (published by the todo plugin
    when plan/step state changes) and external events (published by
    plugins like webhook and ws_client for inbound data).
    """
    # Plan lifecycle events
    PLAN_CREATED = "plan_created"
    PLAN_STARTED = "plan_started"
    PLAN_COMPLETED = "plan_completed"
    PLAN_FAILED = "plan_failed"
    PLAN_CANCELLED = "plan_cancelled"

    # Step lifecycle events
    STEP_ADDED = "step_added"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_SKIPPED = "step_skipped"

    # Collaboration events
    STEP_BLOCKED = "step_blocked"      # Step waiting on dependencies
    STEP_UNBLOCKED = "step_unblocked"  # All dependencies satisfied

    # Agent lifecycle events (bridged from server events)
    AGENT_CREATED = "agent.created"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_COMPLETED = "agent.completed"
    AGENT_ERROR = "agent.error"            # Recoverable terminal error (post Layer-1 exhaustion)
    AGENT_OUTPUT = "agent.output"          # Model text, system text, plugin text

    # Session lifecycle events (bridged from server events)
    SESSION_TERMINATED = "session.terminated"  # Server 0.6.162+ / SDK 0.14.3+

    # Cascade stage settled (warm-return / torn-down / cold) — universal
    # per-stage handoff signal; reactors gate next-stage spawn on it.
    SLOT_SETTLED = "slot.settled"

    # HandoffGate release (bridged from server events) — lets reactor rules
    # (e.g. reliability's T3 gate-park resume) fire when a parked session's
    # gate is approved/denied/timed-out. gate.announced stays a daemon-wide
    # client-discovery broadcast (not bridged here). See
    # jaato-premium docs/design/gate-released-bus-delivery.md.
    GATE_RELEASED = "gate.released"

    # Tool execution events (bridged from server events)
    TOOL_CALL_STARTED = "tool.call_started"
    TOOL_CALL_COMPLETED = "tool.call_completed"
    TOOL_OUTPUT = "tool.output"

    # Context & turn events (bridged from server events)
    TURN_COMPLETED = "turn.completed"
    TURN_PROGRESS = "turn.progress"
    CONTEXT_UPDATED = "context.updated"

    # Plan display events (bridged from server events)
    PLAN_STEP_UPDATED = "plan.step_updated"

    # Permission events (bridged from server events)
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_RESOLVED = "permission.resolved"

    # Drift monitor events (published by drift monitor plugin)
    DRIFT_MEASURED = "drift.measured"

    # Reliability reactor events (published by a tenant-authored reliability
    # reactor — the event-driven successor to the in-process reliability
    # plugin; see docs/design/reliability-event-driven-migration.md).
    # Observability: emitted on every escalation / behavioral-pattern
    # detection so traces see the signal even when no steer fires.  A reactor
    # may also SUBSCRIBE to these (e.g. a T2/T3 headless-approval reactor
    # reacting to reliability.escalated).
    RELIABILITY_ESCALATED = "reliability.escalated"
    RELIABILITY_PATTERN_DETECTED = "reliability.pattern_detected"

    # External events (published by plugins, not plan/step lifecycle)
    EXTERNAL_EVENT = "external_event"  # Webhook, WebSocket, or other external ingress


@dataclass
class EventFilter:
    """Filter for subscribing to specific events on the bus.

    All fields are optional — ``None`` means "match any".
    Empty ``event_types`` list means "match all event types".
    """
    agent_id: Optional[str] = None       # None or "*" = any agent
    plan_id: Optional[str] = None        # None = any plan
    step_id: Optional[str] = None        # None = any step
    event_types: List[EventType] = field(default_factory=list)  # Empty = all

    def matches(self, event: 'Event') -> bool:
        """Check if an event matches this filter.

        Plan and step IDs are read from ``event.payload`` (not top-level
        fields), since those are domain-specific data.

        Args:
            event: The Event to check.

        Returns:
            True if the event matches all specified filter criteria.
        """
        if self.agent_id and self.agent_id != "*":
            if event.source_agent != self.agent_id:
                return False

        if self.plan_id and event.payload.get("plan_id") != self.plan_id:
            return False

        if self.step_id and event.payload.get("step_id") != self.step_id:
            return False

        if self.event_types and event.event_type not in self.event_types:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "event_types": [e.value for e in self.event_types],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventFilter':
        """Create from dictionary."""
        event_types = []
        for et in data.get("event_types", []):
            try:
                event_types.append(EventType(et))
            except ValueError:
                pass  # Skip invalid event types
        return cls(
            agent_id=data.get("agent_id"),
            plan_id=data.get("plan_id"),
            step_id=data.get("step_id"),
            event_types=event_types,
        )


@dataclass
class Subscription:
    """A subscription to events on the bus.

    When events matching the filter are published, the subscription's
    action is executed (typically a callback that injects the event
    into the subscriber's session).
    """
    subscription_id: str
    subscriber_name: str    # Identifier of the subscriber (agent, plugin, etc.)
    filter: EventFilter

    # Action when event matches
    action_type: str = "callback"  # "callback", "unblock_step", "inject_message"
    action_target: Optional[str] = None  # step_id to unblock, callback name, etc.

    # Lifecycle
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z")
    expires_after: Optional[int] = None   # Auto-remove after N matches (None = persistent)
    match_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subscription_id": self.subscription_id,
            "subscriber_name": self.subscriber_name,
            "filter": self.filter.to_dict(),
            "action_type": self.action_type,
            "action_target": self.action_target,
            "created_at": self.created_at,
            "expires_after": self.expires_after,
            "match_count": self.match_count,
        }


@dataclass
class Event:
    """Universal event on the bus.

    Minimal envelope carrying event identity and source. All domain-specific
    data (plan context, tool args, text content, token counts) lives in
    ``payload``. This keeps the envelope stable as new event types are added.
    """
    event_id: str
    event_type: EventType
    timestamp: str  # ISO8601
    source_agent: str
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source_agent: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> 'Event':
        """Create a new Event with auto-generated ID and timestamp.

        Args:
            event_type: Type of event.
            source_agent: ID of the agent or plugin publishing the event.
            payload: Event-specific data. For plan/step events, include
                plan_id, plan_title, step_id, etc. here.

        Returns:
            A new Event instance.
        """
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            source_agent=source_agent,
            payload=payload or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create from dictionary."""
        try:
            event_type = EventType(data.get("event_type", "step_completed"))
        except ValueError:
            event_type = EventType.STEP_COMPLETED

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat() + "Z"),
            source_agent=data.get("source_agent", ""),
            payload=data.get("payload", {}),
        )


# === Backward-compatible alias ===

TaskEventType = EventType
"""Alias for ``EventType``. Use ``EventType`` in new code."""
