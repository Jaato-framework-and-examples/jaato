"""TaskEventBus for cross-agent task collaboration.

This module provides a central event bus that enables agents to:
- Publish task events (plan created, step completed, etc.)
- Subscribe to events from other agents
- Coordinate dependencies across agent boundaries

The TaskEventBus is a singleton shared across all agents in a runtime.
"""

import logging
import threading
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from jaato_sdk.plugins.todo.models import (
    TaskEvent, TaskEventType, EventFilter, Subscription, TaskRef
)

if TYPE_CHECKING:
    from jaato_sdk.plugins.todo.models import TodoPlan, TodoStep

logger = logging.getLogger(__name__)


class TaskEventBus:
    """Central event bus for cross-agent task coordination.

    This is a singleton shared across all agents in the runtime.
    It enables:
    - Publishing task events (plan created, step completed, etc.)
    - Subscribing to events with filters
    - Automatic dependency resolution and notification

    Thread-safe for concurrent access from multiple agents running
    in different threads.

    Usage:
        # Get the singleton instance
        bus = TaskEventBus.get_instance()

        # Subscribe to events
        sub_id = bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED]),
            callback=lambda event: print(f"New plan: {event.source_plan_title}")
        )

        # Publish events
        bus.publish(TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="subagent_1",
            plan=my_plan
        ))

        # Unsubscribe
        bus.unsubscribe(sub_id)
    """

    _instance: Optional['TaskEventBus'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'TaskEventBus':
        """Get or create the singleton instance.

        Returns:
            The TaskEventBus singleton.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.debug("TaskEventBus singleton created")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing).

        This clears all subscriptions and event history.
        """
        with cls._lock:
            cls._instance = None
            logger.debug("TaskEventBus singleton reset")

    def __init__(self):
        """Initialize the event bus.

        Note: Use get_instance() instead of direct construction.
        """
        # Subscription storage: subscription_id -> Subscription
        self._subscriptions: Dict[str, Subscription] = {}

        # Callbacks for subscriptions: subscription_id -> callback function
        self._callbacks: Dict[str, Callable[[TaskEvent], None]] = {}

        # Event history for late subscribers and debugging
        self._event_history: List[TaskEvent] = []
        self._max_history = 1000

        # Lock for thread-safe access
        self._sub_lock = threading.Lock()

        # Condition variable for long-poll notifications.
        # Signaled every time a new event is published, allowing
        # wait_for_events() callers to wake up without busy-polling.
        self._event_condition = threading.Condition()

        # Dependency tracking: maps TaskRef URIs to waiting steps
        # Key: TaskRef.to_uri(), Value: list of (subscriber_agent, plan_id, step_id)
        self._dependency_waiters: Dict[str, List[tuple]] = {}

        # Callback for dependency resolution (set by TodoPlugin)
        self._dependency_resolver: Optional[Callable[[str, str, str, TaskEvent], None]] = None

    def subscribe(
        self,
        subscriber_agent: str,
        filter: EventFilter,
        callback: Optional[Callable[[TaskEvent], None]] = None,
        action_type: str = "callback",
        action_target: Optional[str] = None,
        expires_after: Optional[int] = None,
        replay_history: bool = True
    ) -> str:
        """Subscribe to task events matching the filter.

        Args:
            subscriber_agent: Agent ID of the subscriber.
            filter: EventFilter specifying which events to receive.
            callback: Function called when matching event is published.
                      Required if action_type is "callback".
            action_type: Type of action to take on match:
                - "callback": Call the callback function
                - "unblock_step": Signal dependency resolution
                - "inject_message": Inject message to subscriber's queue
            action_target: Target for the action (step_id for unblock_step, etc.)
            expires_after: Auto-remove subscription after N matches.
                          None means persistent until explicitly unsubscribed.
            replay_history: If True (default), replay matching historical events
                          to the new subscriber. This prevents race conditions
                          where events are published before the subscription
                          is created.

        Returns:
            Subscription ID for later unsubscription.

        Example:
            # Subscribe to all plan_created events from any agent
            sub_id = bus.subscribe(
                subscriber_agent="main",
                filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED]),
                callback=lambda e: print(f"Plan created: {e.source_plan_title}")
            )
        """
        import uuid
        sub_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=sub_id,
            subscriber_agent=subscriber_agent,
            filter=filter,
            action_type=action_type,
            action_target=action_target,
            expires_after=expires_after
        )

        # Register subscription and snapshot history atomically.
        # This ensures no events are lost between the history scan and
        # subscription activation. Combined with the atomic lock in
        # publish(), this guarantees exactly-once delivery: an event
        # is either in the history snapshot (replayed) or delivered
        # via the live subscription, never both and never neither.
        events_to_replay: List[TaskEvent] = []
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
            sub_id[:8], subscriber_agent,
            [e.value for e in filter.event_types] or "all",
            filter.agent_id or "any",
            len(events_to_replay)
        )

        # Replay historical events outside the lock to avoid deadlocks.
        # This is safe because publish() takes the lock atomically for
        # both appending to history AND snapshotting subscriptions, so
        # any event published after our lock section above will find
        # this subscription and deliver normally (not duplicated).
        for event in events_to_replay:
            subscription.match_count += 1
            try:
                callback(event)
            except Exception as e:
                logger.exception(
                    "Error replaying historical event to %s: %s",
                    sub_id[:8], e
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

    def publish(self, event: TaskEvent) -> int:
        """Publish an event to all matching subscribers.

        Args:
            event: The TaskEvent to publish.

        Returns:
            Number of subscribers notified.
        """
        # Atomically store in history AND snapshot subscriptions.
        # This single lock section is critical for correctness with
        # subscribe(replay_history=True): it ensures that an event is
        # either in the history when a new subscriber snapshots it, or
        # the subscriber is in our snapshot here — never both, never neither.
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

                # Execute action
                self._execute_subscription_action(sub, event)

                # Check expiration
                if sub.expires_after and sub.match_count >= sub.expires_after:
                    to_remove.append(sub_id)

        # Clean up expired subscriptions
        for sub_id in to_remove:
            self.unsubscribe(sub_id)

        # Handle dependency resolution for step_completed events
        if event.event_type == TaskEventType.STEP_COMPLETED:
            self._resolve_dependencies(event)

        # Wake up any long-poll waiters so they can check for new events.
        with self._event_condition:
            self._event_condition.notify_all()

        logger.debug(
            "Published %s from %s: notified %d subscribers",
            event.event_type.value, event.source_agent, notified
        )

        return notified

    def _execute_subscription_action(
        self,
        sub: Subscription,
        event: TaskEvent
    ) -> None:
        """Execute the action associated with a subscription.

        Args:
            sub: The Subscription that matched.
            event: The TaskEvent that triggered the subscription.
        """
        if sub.action_type == "callback":
            callback = self._callbacks.get(sub.subscription_id)
            if callback:
                try:
                    callback(event)
                except Exception as e:
                    logger.exception(
                        "Error in subscription callback %s: %s",
                        sub.subscription_id[:8], e
                    )

        elif sub.action_type == "unblock_step":
            # This is handled via the dependency resolver
            if self._dependency_resolver and sub.action_target:
                try:
                    self._dependency_resolver(
                        sub.subscriber_agent,
                        sub.action_target,  # plan_id
                        sub.action_target,  # step_id (reusing for now)
                        event
                    )
                except Exception as e:
                    logger.exception(
                        "Error in dependency resolver for %s: %s",
                        sub.subscription_id[:8], e
                    )

        elif sub.action_type == "inject_message":
            # Message injection requires session access
            # This is typically handled by the subscriber's callback
            logger.debug(
                "inject_message action for %s (handled by callback)",
                sub.subscription_id[:8]
            )

    def register_dependency(
        self,
        dependency_ref: TaskRef,
        waiting_agent: str,
        waiting_plan_id: str,
        waiting_step_id: str
    ) -> None:
        """Register that a step is waiting on a dependency.

        When the dependency completes, the waiting step will be notified
        via the dependency resolver callback.

        Args:
            dependency_ref: The TaskRef being waited on.
            waiting_agent: Agent that owns the waiting step.
            waiting_plan_id: Plan containing the waiting step.
            waiting_step_id: The step that is blocked.
        """
        key = dependency_ref.to_uri()
        with self._sub_lock:
            if key not in self._dependency_waiters:
                self._dependency_waiters[key] = []
            self._dependency_waiters[key].append(
                (waiting_agent, waiting_plan_id, waiting_step_id)
            )

        logger.debug(
            "Registered dependency: %s:%s waits on %s",
            waiting_agent, waiting_step_id, key
        )

    def _resolve_dependencies(self, event: TaskEvent) -> None:
        """Resolve dependencies when a step completes.

        Called automatically when a step_completed event is published.
        Notifies all steps that were waiting on this dependency.

        Waiters are only removed after the resolver callback succeeds.
        If the resolver raises, the waiter is re-added so it can be
        retried on the next matching event.

        Args:
            event: The step_completed event.
        """
        if not event.source_step_id:
            return

        # Build possible keys for this completed step
        # (with and without plan_id)
        keys_to_check = [
            f"{event.source_agent}:{event.source_plan_id}/{event.source_step_id}",
            f"{event.source_agent}:*/{event.source_step_id}",
        ]

        # Collect waiters with their originating keys (for re-adding on failure)
        waiters_with_keys: list = []
        with self._sub_lock:
            for key in keys_to_check:
                if key in self._dependency_waiters:
                    for waiter in self._dependency_waiters.pop(key):
                        waiters_with_keys.append((key, waiter))

        if not waiters_with_keys:
            return

        logger.debug(
            "Resolving dependencies for %s:%s - %d waiters",
            event.source_agent, event.source_step_id, len(waiters_with_keys)
        )

        # Notify waiters via the dependency resolver.
        # On failure, re-add the waiter so it can be retried.
        if self._dependency_resolver:
            for key, (waiting_agent, waiting_plan_id, waiting_step_id) in waiters_with_keys:
                try:
                    self._dependency_resolver(
                        waiting_agent,
                        waiting_plan_id,
                        waiting_step_id,
                        event
                    )
                except Exception as e:
                    logger.warning(
                        "Dependency resolution failed for %s:%s (%s), "
                        "re-adding waiter for retry: %s",
                        waiting_agent, waiting_step_id, key, e
                    )
                    # Re-add the waiter so the next matching event retries
                    with self._sub_lock:
                        if key not in self._dependency_waiters:
                            self._dependency_waiters[key] = []
                        self._dependency_waiters[key].append(
                            (waiting_agent, waiting_plan_id, waiting_step_id)
                        )

    def set_dependency_resolver(
        self,
        resolver: Callable[[str, str, str, TaskEvent], None]
    ) -> None:
        """Set the callback for resolving dependencies.

        The resolver is called when a dependency completes, with:
        - waiting_agent: Agent that was waiting
        - waiting_plan_id: Plan containing the waiting step
        - waiting_step_id: Step that was blocked
        - event: The completion event with output data

        Args:
            resolver: Callback function for dependency resolution.
        """
        self._dependency_resolver = resolver

    def get_dependency_waiters(self, completed_ref: TaskRef) -> List[tuple]:
        """Get all steps waiting on a specific dependency.

        Args:
            completed_ref: The TaskRef to check.

        Returns:
            List of (agent_id, plan_id, step_id) tuples for waiting steps.
        """
        key = completed_ref.to_uri()
        with self._sub_lock:
            return list(self._dependency_waiters.get(key, []))

    def get_subscriptions(
        self,
        agent_id: Optional[str] = None
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
            subs = [s for s in subs if s.subscriber_agent == agent_id]
        return subs

    def get_recent_events(
        self,
        agent_id: Optional[str] = None,
        event_types: Optional[List[TaskEventType]] = None,
        after_event_id: Optional[str] = None,
        limit: int = 50
    ) -> List[TaskEvent]:
        """Get recent events from history, optionally filtered.

        Args:
            agent_id: Filter by source agent ID.
            event_types: Filter by event types.
            after_event_id: Cursor — only return events published after this
                            event ID.  Pass the last ``event_id`` you received
                            to consume the stream incrementally.  If the ID has
                            been evicted from the rolling history window, all
                            available events are returned.
            limit: Maximum number of events to return.

        Returns:
            List of TaskEvent objects, most recent last.
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
            # If the event_id wasn't found the cursor has been evicted
            # from the rolling window — return everything available.

        if agent_id:
            events = [e for e in events if e.source_agent == agent_id]
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        return events[-limit:]

    def wait_for_events(
        self,
        timeout: float,
        agent_id: Optional[str] = None,
        event_types: Optional[List[TaskEventType]] = None,
        after_event_id: Optional[str] = None,
        limit: int = 50
    ) -> List[TaskEvent]:
        """Wait for events, returning early when they arrive.

        Implements long-polling: if events already exist (given the cursor
        and filters), returns immediately.  Otherwise blocks up to
        ``timeout`` seconds for new events to be published.

        Delegates all filtering and cursor logic to
        :meth:`get_recent_events`; this method only adds the blocking
        wait on top.

        Args:
            timeout: Maximum seconds to wait (capped at 30).
            agent_id: Filter by source agent ID.
            event_types: Filter by event types.
            after_event_id: Cursor — only return events after this event ID.
            limit: Maximum number of events to return.

        Returns:
            List of matching TaskEvent objects, most recent last.
        """
        timeout = min(max(timeout, 0), 30)

        def _poll() -> List[TaskEvent]:
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
        import time
        deadline = time.monotonic() + timeout
        with self._event_condition:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._event_condition.wait(timeout=remaining)
                # Re-check after wakeup.
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
            Dict with counts for subscriptions, events, waiters.
        """
        with self._sub_lock:
            return {
                "subscriptions": len(self._subscriptions),
                "events_in_history": len(self._event_history),
                "dependency_waiters": sum(
                    len(w) for w in self._dependency_waiters.values()
                ),
            }


def get_event_bus() -> TaskEventBus:
    """Convenience function to get the TaskEventBus singleton.

    Returns:
        The TaskEventBus singleton instance.
    """
    return TaskEventBus.get_instance()
