"""Per-runtime event bus for cross-agent and cross-plugin event coordination.

Each ``JaatoRuntime`` creates its own ``EventBus`` instance, ensuring
session isolation. Agents within a single runtime (main + subagents)
share the same bus for plan coordination and event delivery.

The event bus enables:
- Cross-agent task coordination (plan/step lifecycle events)
- External event ingestion (webhooks, WebSocket streams)
- Plugin-to-plugin communication

Task-specific features (dependency tracking, resolution callbacks)
remain in the todo plugin's ``TaskEventBus`` which wraps this bus.

Usage:
    bus = runtime.event_bus  # from JaatoRuntime
    sub_id = bus.subscribe(
        subscriber_name="main",
        filter=EventFilter(event_types=[EventType.EXTERNAL_EVENT]),
        callback=lambda event: print(event),
    )
    bus.publish(Event.create(
        event_type=EventType.EXTERNAL_EVENT,
        source_agent="webhook:github",
        payload={"event_type": "push", ...},
    ))
"""

import logging
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

from jaato_sdk.event_bus import (
    Event,
    EventFilter,
    EventType,
    Subscription,
)

logger = logging.getLogger(__name__)


class EventBus:
    """Central event bus for cross-agent and cross-plugin coordination.

    Each JaatoRuntime owns its own EventBus instance, ensuring session
    isolation when multiple sessions share the same server process.
    Agents within a single runtime (main agent + subagents) share the
    same bus for plan coordination and event delivery.

    It enables:
    - Publishing events (plan state changes, external ingress, etc.)
    - Subscribing to events with filters
    - Long-poll waiting for events

    Thread-safe for concurrent access from multiple agents running
    in different threads.

    Lifecycle:
        Created by JaatoRuntime during initialization. Lives for
        the lifetime of the runtime.
    """

    def __init__(self):
        """Initialize the event bus."""
        # Subscription storage: subscription_id -> Subscription
        self._subscriptions: Dict[str, Subscription] = {}

        # Callbacks for subscriptions: subscription_id -> callback function
        self._callbacks: Dict[str, Callable[[Event], None]] = {}

        # Event history for late subscribers and debugging
        self._event_history: List[Event] = []
        self._max_history = 1000

        # Lock for thread-safe access
        self._sub_lock = threading.Lock()

        # Condition variable for long-poll notifications.
        # Signaled every time a new event is published, allowing
        # wait_for_events() callers to wake up without busy-polling.
        self._event_condition = threading.Condition()

    def subscribe(
        self,
        subscriber_name: str,
        filter: EventFilter,
        callback: Optional[Callable[[Event], None]] = None,
        action_type: str = "callback",
        action_target: Optional[str] = None,
        expires_after: Optional[int] = None,
        replay_history: bool = True,
    ) -> str:
        """Subscribe to events matching the filter.

        Args:
            subscriber_name: Identifier of the subscriber (agent, plugin, etc.).
            filter: EventFilter specifying which events to receive.
            callback: Function called when matching event is published.
                      Required if action_type is "callback".
            action_type: Type of action to take on match:
                - "callback": Call the callback function
                - "inject_message": Inject message to subscriber's queue
            action_target: Target for the action (callback name, etc.)
            expires_after: Auto-remove subscription after N matches.
                          None means persistent until explicitly unsubscribed.
            replay_history: If True (default), replay matching historical events
                          to the new subscriber. This prevents race conditions
                          where events are published before the subscription
                          is created.

        Returns:
            Subscription ID for later unsubscription.
        """
        sub_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=sub_id,
            subscriber_name=subscriber_name,
            filter=filter,
            action_type=action_type,
            action_target=action_target,
            expires_after=expires_after,
        )

        # Register subscription and snapshot history atomically.
        # This ensures no events are lost between the history scan and
        # subscription activation. Combined with the atomic lock in
        # publish(), this guarantees exactly-once delivery: an event
        # is either in the history snapshot (replayed) or delivered
        # via the live subscription, never both and never neither.
        events_to_replay: List[Event] = []
        with self._sub_lock:
            self._subscriptions[sub_id] = subscription
            if callback:
                self._callbacks[sub_id] = callback

            # Snapshot matching historical events for replay
            if replay_history and callback:
                events_to_replay = [
                    e for e in self._event_history
                    if filter.matches(e)
                ]

        logger.debug(
            "Subscription created: %s by %s for %s from %s (replay=%d events)",
            sub_id[:8], subscriber_name,
            [e.value for e in filter.event_types] or "all",
            filter.agent_id or "any",
            len(events_to_replay),
        )

        # Replay historical events outside the lock to avoid deadlocks.
        for event in events_to_replay:
            subscription.match_count += 1
            try:
                callback(event)
            except Exception as e:
                logger.exception(
                    "Error replaying historical event to %s: %s",
                    sub_id[:8], e,
                )
            # Check expiration during replay
            if expires_after and subscription.match_count >= expires_after:
                self.unsubscribe(sub_id)
                break

        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription.

        Args:
            subscription_id: The ID returned by subscribe().

        Returns:
            True if subscription was found and removed.
        """
        with self._sub_lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                self._callbacks.pop(subscription_id, None)
                logger.debug("Subscription removed: %s", subscription_id[:8])
                return True
        return False

    def publish(self, event: Event) -> int:
        """Publish an event to all matching subscribers.

        Args:
            event: The Event to publish.

        Returns:
            Number of subscribers notified.
        """
        # Atomically store in history AND snapshot subscriptions.
        with self._sub_lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            subscriptions = list(self._subscriptions.items())

        notified = 0
        to_remove = []

        for sub_id, sub in subscriptions:
            if sub.filter.matches(event):
                notified += 1
                sub.match_count += 1

                # Execute callback
                callback = self._callbacks.get(sub.subscription_id)
                if callback:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.exception(
                            "Error in subscription callback %s: %s",
                            sub.subscription_id[:8], e,
                        )

                # Check expiration
                if sub.expires_after and sub.match_count >= sub.expires_after:
                    to_remove.append(sub_id)

        # Clean up expired subscriptions
        for sub_id in to_remove:
            self.unsubscribe(sub_id)

        # Wake up any long-poll waiters
        with self._event_condition:
            self._event_condition.notify_all()

        logger.debug(
            "Published %s from %s: notified %d subscribers",
            event.event_type.value, event.source_agent, notified,
        )

        return notified

    def get_subscriptions(
        self,
        agent_id: Optional[str] = None,
    ) -> List[Subscription]:
        """Get all subscriptions, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID to filter by.

        Returns:
            List of Subscription objects.
        """
        with self._sub_lock:
            subs = list(self._subscriptions.values())
        if agent_id:
            subs = [s for s in subs if s.subscriber_name == agent_id]
        return subs

    def get_recent_events(
        self,
        agent_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        after_event_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Get recent events from history, optionally filtered.

        Args:
            agent_id: Filter by source agent ID.
            event_types: Filter by event types.
            after_event_id: Cursor — only return events published after this
                            event ID.
            limit: Maximum number of events to return.

        Returns:
            List of Event objects, most recent last.
        """
        with self._sub_lock:
            events = list(self._event_history)

        # Advance past the cursor
        if after_event_id:
            idx = None
            for i, e in enumerate(events):
                if e.event_id == after_event_id:
                    idx = i
                    break
            if idx is not None:
                events = events[idx + 1:]

        if agent_id:
            events = [e for e in events if e.source_agent == agent_id]
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        return events[-limit:]

    def wait_for_events(
        self,
        timeout: float,
        agent_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        after_event_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Wait for events, returning early when they arrive.

        Implements long-polling: if events already exist (given the cursor
        and filters), returns immediately. Otherwise blocks up to
        ``timeout`` seconds for new events to be published.

        Args:
            timeout: Maximum seconds to wait (capped at 30).
            agent_id: Filter by source agent ID.
            event_types: Filter by event types.
            after_event_id: Cursor — only return events after this event ID.
            limit: Maximum number of events to return.

        Returns:
            List of matching Event objects, most recent last.
        """
        timeout = min(max(timeout, 0), 30)

        def _poll() -> List[Event]:
            return self.get_recent_events(
                agent_id=agent_id,
                event_types=event_types,
                after_event_id=after_event_id,
                limit=limit,
            )

        # Fast path: events already exist.
        result = _poll()
        if result or timeout <= 0:
            return result

        # Slow path: wait for the condition to be signaled by publish().
        deadline = time.monotonic() + timeout
        with self._event_condition:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._event_condition.wait(timeout=remaining)
                result = _poll()
                if result:
                    return result

        # Final check after timeout.
        return _poll()

    def clear_history(self) -> int:
        """Clear the event history.

        Returns:
            Number of events cleared.
        """
        with self._sub_lock:
            count = len(self._event_history)
            self._event_history.clear()
        return count

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the event bus.

        Returns:
            Dict with counts for subscriptions, events.
        """
        with self._sub_lock:
            return {
                "subscriptions": len(self._subscriptions),
                "events_in_history": len(self._event_history),
            }

