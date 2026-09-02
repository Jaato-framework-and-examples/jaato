"""Task-specific event bus wrapper with dependency tracking.

Wraps ``EventBus`` (from ``shared.event_bus``) with task-specific
dependency tracking and resolution.

The generic event infrastructure (subscribe, publish, long-poll,
history) lives in ``EventBus``. This module adds:
- Dependency registration (``register_dependency``)
- Automatic dependency resolution on ``step_completed`` events
- Task-specific ``get_stats`` with dependency waiter counts
"""

import logging
import threading
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from jaato_sdk.event_bus import (
    Event,
    EventType,
    EventFilter,
    Subscription,
)
from jaato_sdk.plugins.todo.models import TaskRef, TaskEvent, TaskEventType  # noqa: F401
from shared.event_bus import EventBus

if TYPE_CHECKING:
    from jaato_sdk.plugins.todo.models import TodoPlan, TodoStep

logger = logging.getLogger(__name__)


class TaskEventBus:
    """Task-aware event bus that wraps an ``EventBus`` instance.

    Delegates all generic event operations (subscribe, publish, poll,
    history) to the wrapped ``EventBus``. Adds task-specific behavior:

    - **Dependency tracking**: ``register_dependency()`` records that a
      step is waiting on another agent's step.
    - **Automatic resolution**: When a ``step_completed`` event is
      published, ``_resolve_dependencies()`` notifies all waiting steps
      via the dependency resolver callback.

    Each session's TodoPlugin creates its own TaskEventBus wrapping the
    per-runtime EventBus from ``JaatoRuntime.event_bus``.

    Usage:
        bus = TaskEventBus(event_bus=runtime.event_bus)

        # All EventBus methods are available:
        bus.subscribe(...)
        bus.publish(...)
        bus.get_recent_events(...)

        # Plus task-specific methods:
        bus.register_dependency(ref, agent, plan_id, step_id)
        bus.set_dependency_resolver(callback)
    """

    def __init__(self, event_bus: EventBus):
        """Initialize the task event bus wrapper.

        Args:
            event_bus: The EventBus instance to wrap (from JaatoRuntime).
        """
        self._bus = event_bus

        # Task-specific: dependency tracking
        self._dependency_waiters: Dict[str, List[tuple]] = {}
        self._dependency_resolver: Optional[Callable[[str, str, str, Event], None]] = None
        self._dep_lock = threading.Lock()

        # Hook into publish for dependency resolution
        self._original_publish = self._bus.publish

    # === Delegated methods (pass through to EventBus) ===

    def subscribe(self, *args, **kwargs) -> str:
        """Subscribe to events. See ``EventBus.subscribe``."""
        return self._bus.subscribe(*args, **kwargs)

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe. See ``EventBus.unsubscribe``."""
        return self._bus.unsubscribe(subscription_id)

    def publish(self, event: Event) -> int:
        """Publish an event, then resolve any task dependencies.

        Delegates to ``EventBus.publish()`` for subscription delivery,
        then checks for dependency resolution on ``step_completed`` events.
        """
        notified = self._bus.publish(event)

        # Task-specific: resolve dependencies on step completion
        if event.event_type == EventType.STEP_COMPLETED:
            self._resolve_dependencies(event)

        return notified

    def get_subscriptions(self, agent_id: Optional[str] = None) -> List[Subscription]:
        """Get subscriptions. See ``EventBus.get_subscriptions``."""
        return self._bus.get_subscriptions(agent_id=agent_id)

    def get_recent_events(self, **kwargs) -> List[Event]:
        """Get recent events. See ``EventBus.get_recent_events``."""
        return self._bus.get_recent_events(**kwargs)

    def wait_for_events(self, **kwargs) -> List[Event]:
        """Wait for events. See ``EventBus.wait_for_events``."""
        return self._bus.wait_for_events(**kwargs)

    def clear_history(self) -> int:
        """Clear event history. See ``EventBus.clear_history``."""
        return self._bus.clear_history()

    # === Task-specific methods ===

    def register_dependency(
        self,
        dependency_ref: TaskRef,
        waiting_agent: str,
        waiting_plan_id: str,
        waiting_step_id: str,
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
        with self._dep_lock:
            if key not in self._dependency_waiters:
                self._dependency_waiters[key] = []
            self._dependency_waiters[key].append(
                (waiting_agent, waiting_plan_id, waiting_step_id)
            )

        logger.debug(
            "Registered dependency: %s:%s waits on %s",
            waiting_agent, waiting_step_id, key,
        )

    def _resolve_dependencies(self, event: Event) -> None:
        """Resolve dependencies when a step completes.

        Called automatically when a ``step_completed`` event is published.
        Notifies all steps that were waiting on this dependency.

        Waiters are only removed after the resolver callback succeeds.
        If the resolver raises, the waiter is re-added for retry.
        """
        step_id = event.payload.get("step_id")
        if not step_id:
            return

        plan_id = event.payload.get("plan_id", "")
        keys_to_check = [
            f"{event.source_agent}:{plan_id}/{step_id}",
            f"{event.source_agent}:*/{step_id}",
        ]

        waiters_with_keys: list = []
        with self._dep_lock:
            for key in keys_to_check:
                if key in self._dependency_waiters:
                    for waiter in self._dependency_waiters.pop(key):
                        waiters_with_keys.append((key, waiter))

        if not waiters_with_keys:
            return

        logger.debug(
            "Resolving dependencies for %s:%s - %d waiters",
            event.source_agent, step_id, len(waiters_with_keys),
        )

        if self._dependency_resolver:
            for key, (waiting_agent, waiting_plan_id, waiting_step_id) in waiters_with_keys:
                try:
                    self._dependency_resolver(
                        waiting_agent,
                        waiting_plan_id,
                        waiting_step_id,
                        event,
                    )
                except Exception as e:
                    logger.warning(
                        "Dependency resolution failed for %s:%s (%s), "
                        "re-adding waiter for retry: %s",
                        waiting_agent, waiting_step_id, key, e,
                    )
                    with self._dep_lock:
                        if key not in self._dependency_waiters:
                            self._dependency_waiters[key] = []
                        self._dependency_waiters[key].append(
                            (waiting_agent, waiting_plan_id, waiting_step_id)
                        )

    def set_dependency_resolver(
        self,
        resolver: Callable[[str, str, str, Event], None],
    ) -> None:
        """Set the callback for resolving dependencies.

        The resolver is called when a dependency completes, with:
        - waiting_agent: Agent that was waiting
        - waiting_plan_id: Plan containing the waiting step
        - waiting_step_id: Step that was blocked
        - event: The completion event with output data
        """
        self._dependency_resolver = resolver

    def get_dependency_waiters(self, completed_ref: TaskRef) -> List[tuple]:
        """Get all steps waiting on a specific dependency."""
        key = completed_ref.to_uri()
        with self._dep_lock:
            return list(self._dependency_waiters.get(key, []))

    def get_stats(self) -> Dict[str, int]:
        """Get statistics including task-specific dependency counts."""
        stats = self._bus.get_stats()
        with self._dep_lock:
            stats["dependency_waiters"] = sum(
                len(w) for w in self._dependency_waiters.values()
            )
        return stats
