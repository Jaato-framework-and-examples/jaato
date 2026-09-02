"""Event bus subscription tools — registered as core tools in JaatoSession.

These tools allow the model to subscribe to events on the shared
``EventBus`` and receive them via ``inject_prompt()``. They are
session-level infrastructure, not plugin-specific.

Tools provided:
- ``subscribeToEvents``: Subscribe with push delivery via inject_prompt()
- ``getEvents``: Poll/long-poll for recent events
- ``listSubscriptions``: List active subscriptions
- ``unsubscribe``: Remove a subscription

Backward-compatible aliases (``subscribeToTasks``, ``getTaskEvents``,
etc.) are registered alongside the new names so existing prompts and
system instructions continue to work.
"""

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from jaato_sdk.event_bus import Event, EventFilter, EventType
from jaato_sdk.plugins.model_provider.types import ToolSchema, DISCOVERABILITY_EAGER

if TYPE_CHECKING:
    from shared.jaato_session import JaatoSession

logger = logging.getLogger(__name__)

# All event type values for the tool schema enum
_ALL_EVENT_TYPES = [e.value for e in EventType]


def _format_event_notification(event: Event) -> str:
    """Format an event as a notification message for injection.

    Produces the same ``[SUBAGENT event=...]`` format used by the
    todo plugin's subscription callbacks.

    Args:
        event: The Event to format.

    Returns:
        Formatted string suitable for injection into agent conversation.
    """
    parts = [f"[SUBAGENT event={event.event_type.value}]"]
    parts.append(f"From: {event.source_agent}")

    if event.payload.get("plan_title", ""):
        parts.append(f"Plan: {event.payload.get('plan_title', '')}")

    # Event-specific details
    if event.event_type == EventType.PLAN_CREATED:
        steps = event.payload.get("steps", [])
        if steps:
            parts.append("Steps:")
            for s in steps:
                parts.append(f"  - {s.get('step_id')}: {s.get('description')}")

    elif event.event_type == EventType.STEP_COMPLETED:
        if event.payload.get("step_id"):
            parts.append(f"Step: {event.payload.get('step_id')}")
        output = event.payload.get("output")
        if output:
            parts.append(f"Output: {json.dumps(output)}")
        result = event.payload.get("result")
        if result:
            parts.append(f"Result: {result}")

    elif event.event_type in (EventType.PLAN_COMPLETED, EventType.PLAN_FAILED):
        summary = event.payload.get("summary")
        if summary:
            parts.append(f"Summary: {summary}")
        progress = event.payload.get("progress")
        if progress:
            parts.append(
                f"Progress: {progress.get('completed', 0)}/{progress.get('total', 0)} steps"
            )

    elif event.event_type == EventType.EXTERNAL_EVENT:
        source = event.payload.get("source", "")
        etype = event.payload.get("event_type", "")
        if source:
            parts.append(f"Source: {source}")
        if etype:
            parts.append(f"Type: {etype}")
        # Include payload data if present
        data = event.payload.get("payload") or event.payload.get("data")
        if data:
            if isinstance(data, dict):
                parts.append(f"Data: {json.dumps(data)}")
            else:
                parts.append(f"Data: {data}")

    return "\n".join(parts)


class EventBusTools:
    """Provides event bus subscription tools for a session.

    Created per-session. Captures the session reference at construction
    time for ``inject_prompt()`` delivery — no thread-local dance needed
    since the session owns this instance.

    Lifecycle:
        1. Created during ``JaatoSession.configure()``
        2. Tools registered as core tools via ``registry.register_core_tool()``
        3. Lives for the lifetime of the session
        4. ``shutdown()`` unsubscribes all active subscriptions
    """

    def __init__(self, session: 'JaatoSession'):
        """Initialize event bus tools for a session.

        Args:
            session: The owning JaatoSession. Used for inject_prompt()
                    delivery and agent_id identification.
        """
        self._session = session
        self._subscriptions: List[str] = []  # subscription IDs for cleanup
        self._bus: Optional[Any] = None  # lazy import to avoid circular deps
        # Optional callback invoked after a successful subscription.
        # Set by the server to emit EventsSubscribedEvent to WS clients.
        self._on_subscribed: Optional[Callable[[str, List[str]], None]] = None

    def _get_bus(self):
        """Get the session's EventBus instance from the runtime."""
        if self._bus is None:
            self._bus = self._session._runtime.event_bus
        return self._bus

    def _get_agent_name(self) -> str:
        """Get the current agent name from the session."""
        return getattr(self._session, 'agent_id', None) or "main"

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for event bus operations."""
        return [
            ToolSchema(
                name="subscribeToEvents",
                description=(
                    "Subscribe to events from other agents or external sources. "
                    "Events arrive as [SUBAGENT event=X] inline messages — no polling needed.\n\n"
                    "RECOMMENDED: Subscribe early in your workflow:\n"
                    "  subscribeToEvents(event_types=['plan_created', 'step_completed'])\n\n"
                    "For external events (webhooks, WebSocket):\n"
                    "  subscribeToEvents(event_types=['external_event'])\n\n"
                    "Events are delivered via inject_prompt() — the session wakes automatically."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Agent to subscribe to. Use '*' for any agent. Omit for any.",
                        },
                        "plan_id": {
                            "type": "string",
                            "description": "Specific plan ID to filter (optional)",
                        },
                        "step_id": {
                            "type": "string",
                            "description": "Specific step ID to filter (optional)",
                        },
                        "event_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": _ALL_EVENT_TYPES,
                            },
                            "description": "Event types to subscribe to",
                        },
                        "event_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Filter external events by name (e.g., 'editor.change', "
                                "'order.placed'). Only meaningful when event_types includes "
                                "'external_event'. Omit to receive all external events."
                            ),
                        },
                    },
                    "required": ["event_types"],
                },
                category="coordination",
                discoverability=DISCOVERABILITY_EAGER,
            ),
            ToolSchema(
                name="getEvents",
                description=(
                    "Review recent cross-agent and external events.\n\n"
                    "LONG-POLL: Set timeout (1-30) to block until events arrive.\n"
                    "Combine with after_event for incremental consumption.\n\n"
                    "IMPORTANT: timeout requires at least one of agent_id, event_types, "
                    "or after_event."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Filter by source agent (optional)",
                        },
                        "event_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by event types (optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum events to return (default: 20)",
                        },
                        "timeout": {
                            "type": "number",
                            "description": (
                                "Long-poll timeout in seconds (0-30). Blocks until "
                                "events matching filters arrive or timeout expires. "
                                "Default: 0 (no wait)."
                            ),
                        },
                        "after_event": {
                            "type": "string",
                            "description": (
                                "Only return events after this event ID. "
                                "Pass the last event_id for incremental consumption."
                            ),
                        },
                    },
                    "required": [],
                },
                category="coordination",
                discoverability=DISCOVERABILITY_EAGER,
            ),
            ToolSchema(
                name="listSubscriptions",
                description="List your active event subscriptions.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                category="coordination",
                discoverability=DISCOVERABILITY_EAGER,
            ),
            ToolSchema(
                name="unsubscribe",
                description="Remove an event subscription.",
                parameters={
                    "type": "object",
                    "properties": {
                        "subscription_id": {
                            "type": "string",
                            "description": "ID of the subscription to remove",
                        },
                    },
                    "required": ["subscription_id"],
                },
                category="coordination",
                discoverability=DISCOVERABILITY_EAGER,
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executors for event bus tools, including backward-compat aliases."""
        executors = {
            "subscribeToEvents": self._execute_subscribe,
            "getEvents": self._execute_get_events,
            "listSubscriptions": self._execute_list_subscriptions,
            "unsubscribe": self._execute_unsubscribe,
            # Backward-compatible aliases
            "subscribeToTasks": self._execute_subscribe,
            "getTaskEvents": self._execute_get_events,
        }
        return executors

    def get_auto_approved_tools(self) -> List[str]:
        """Tools that don't require permission prompts."""
        return ["listSubscriptions"]

    def shutdown(self) -> None:
        """Unsubscribe all active subscriptions for this session."""
        if not self._subscriptions:
            return
        try:
            bus = self._get_bus()
            for sub_id in self._subscriptions:
                bus.unsubscribe(sub_id)
            logger.debug(
                "Cleaned up %d event subscriptions for %s",
                len(self._subscriptions), self._get_agent_name(),
            )
        except Exception:
            logger.debug("Failed to clean up subscriptions", exc_info=True)
        self._subscriptions.clear()

    # === Tool executors ===

    def _execute_subscribe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subscribeToEvents / subscribeToTasks."""
        event_types_raw = args.get("event_types", [])
        agent_id = args.get("agent_id")
        plan_id = args.get("plan_id")
        step_id = args.get("step_id")
        event_names_raw = args.get("event_names", [])

        if not event_types_raw:
            return {"error": "event_types is required"}

        # Some providers pass array parameters as JSON strings rather
        # than parsed lists.  Deserialize if needed so iteration yields
        # individual event type strings, not characters.
        if isinstance(event_types_raw, str):
            try:
                event_types_raw = json.loads(event_types_raw)
            except json.JSONDecodeError:
                event_types_raw = [event_types_raw]

        if isinstance(event_names_raw, str):
            try:
                event_names_raw = json.loads(event_names_raw)
            except json.JSONDecodeError:
                event_names_raw = [event_names_raw]

        # Parse event types
        event_types = []
        for et in event_types_raw:
            try:
                event_types.append(EventType(et))
            except ValueError:
                return {"error": f"Invalid event type: {et}"}

        # event_names filter: only applies to external_event subscriptions
        event_names_set = set(event_names_raw) if event_names_raw else None

        bus = self._get_bus()
        subscriber_name = self._get_agent_name()
        # Capture session reference at subscription time (not thread-local)
        session = self._session

        # Create filter
        filter = EventFilter(
            agent_id=agent_id,
            plan_id=plan_id,
            step_id=step_id,
            event_types=event_types,
        )

        # Callback that injects events into the subscriber's session.
        # Uses SourceType.EVENT so subscribed events are delivered mid-turn
        # (high priority), unlike passive CHILD status updates which wait
        # until idle.  The agent explicitly asked for these events, so they
        # should arrive at the next natural pause point (between tool calls).
        def on_event(event: Event) -> None:
            # Don't deliver events from self
            if event.source_agent == subscriber_name:
                return

            # Apply event_names filter for external events
            if event_names_set and event.event_type == EventType.EXTERNAL_EVENT:
                name = event.payload.get("event_type", "")
                if name not in event_names_set:
                    return

            event_message = _format_event_notification(event)

            from shared.message_queue import SourceType
            try:
                session.inject_prompt(
                    text=event_message,
                    source_id=event.source_agent,
                    source_type=SourceType.EVENT,
                )
            except Exception as e:
                logger.debug("Failed to inject event: %s", e)

        sub_id = bus.subscribe(
            subscriber_name=subscriber_name,
            filter=filter,
            callback=on_event,
        )
        self._subscriptions.append(sub_id)

        # Notify WS clients only for external_event subscriptions so the
        # host page knows the agent is ready to receive external events.
        has_external = EventType.EXTERNAL_EVENT in event_types
        if has_external and self._on_subscribed:
            names = list(event_names_raw) if event_names_raw else ["*"]
            try:
                self._on_subscribed(subscriber_name, names)
            except Exception:
                logger.debug("Failed to emit subscription notification", exc_info=True)

        return {
            "subscription_id": sub_id,
            "filter": filter.to_dict(),
            "message": f"Subscribed to {len(event_types)} event type(s). Events will be delivered inline.",
        }

    def _execute_get_events(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute getEvents / getTaskEvents."""
        agent_id = args.get("agent_id")
        event_types_raw = args.get("event_types", [])
        limit = args.get("limit", 20)
        poll_timeout = args.get("timeout", 0)
        after_event = args.get("after_event")

        bus = self._get_bus()

        # Handle event_types passed as JSON string (see _execute_subscribe)
        if isinstance(event_types_raw, str):
            try:
                event_types_raw = json.loads(event_types_raw)
            except json.JSONDecodeError:
                event_types_raw = [event_types_raw]

        # Parse event types
        event_types = None
        if event_types_raw:
            event_types = []
            for et in event_types_raw:
                try:
                    event_types.append(EventType(et))
                except ValueError:
                    pass

        # Clamp timeout
        try:
            poll_timeout = float(poll_timeout)
        except (TypeError, ValueError):
            poll_timeout = 0
        poll_timeout = min(max(poll_timeout, 0), 30)

        # Guard: timeout without narrowing is useless
        if poll_timeout > 0 and not agent_id and not event_types and not after_event:
            return {
                "error": (
                    "timeout requires at least one of: agent_id, "
                    "event_types, or after_event."
                )
            }

        if poll_timeout > 0:
            events = bus.wait_for_events(
                timeout=poll_timeout,
                agent_id=agent_id,
                event_types=event_types,
                after_event_id=after_event,
                limit=limit,
            )
        else:
            events = bus.get_recent_events(
                agent_id=agent_id,
                event_types=event_types,
                after_event_id=after_event,
                limit=limit,
            )

        result: Dict[str, Any] = {
            "events": [e.to_dict() for e in events],
            "count": len(events),
        }
        if events:
            result["last_event_id"] = events[-1].event_id

        return result

    def _execute_list_subscriptions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute listSubscriptions."""
        bus = self._get_bus()
        subs = bus.get_subscriptions(agent_id=self._get_agent_name())
        return {
            "subscriptions": [s.to_dict() for s in subs],
            "count": len(subs),
        }

    def _execute_unsubscribe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unsubscribe."""
        sub_id = args.get("subscription_id", "")
        if not sub_id:
            return {"error": "subscription_id is required"}

        bus = self._get_bus()
        success = bus.unsubscribe(sub_id)

        if success and sub_id in self._subscriptions:
            self._subscriptions.remove(sub_id)

        return {
            "success": success,
            "subscription_id": sub_id,
            "message": "Subscription removed" if success else "Subscription not found",
        }
