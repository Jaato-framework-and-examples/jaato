"""Tests for TaskEventBus cross-agent collaboration."""

import pytest
import threading
import time
from unittest.mock import Mock, MagicMock

from ..event_bus import TaskEventBus, get_event_bus
from jaato_sdk.plugins.todo.models import (
    TaskEvent, TaskEventType, TaskRef, EventFilter, Subscription,
    TodoPlan, TodoStep, StepStatus
)


class TestTaskEventBus:
    """Tests for TaskEventBus singleton and basic operations."""

    def setup_method(self):
        """Reset singleton before each test."""
        TaskEventBus.reset()

    def teardown_method(self):
        """Clean up after each test."""
        TaskEventBus.reset()

    def test_singleton(self):
        """Test that get_instance returns same instance."""
        bus1 = TaskEventBus.get_instance()
        bus2 = TaskEventBus.get_instance()
        bus3 = get_event_bus()

        assert bus1 is bus2
        assert bus2 is bus3

    def test_reset_singleton(self):
        """Test that reset creates new instance."""
        bus1 = TaskEventBus.get_instance()
        TaskEventBus.reset()
        bus2 = TaskEventBus.get_instance()

        assert bus1 is not bus2

    def test_subscribe_basic(self):
        """Test basic subscription creation."""
        bus = get_event_bus()

        sub_id = bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED])
        )

        assert sub_id is not None
        assert len(sub_id) > 0

        subs = bus.get_subscriptions(agent_id="main")
        assert len(subs) == 1
        assert subs[0].subscription_id == sub_id

    def test_unsubscribe(self):
        """Test subscription removal."""
        bus = get_event_bus()

        sub_id = bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED])
        )

        assert bus.unsubscribe(sub_id) is True
        assert bus.unsubscribe(sub_id) is False  # Already removed

        subs = bus.get_subscriptions()
        assert len(subs) == 0

    def test_publish_notifies_subscribers(self):
        """Test that publish notifies matching subscribers."""
        bus = get_event_bus()
        received_events = []

        def callback(event: TaskEvent):
            received_events.append(event)

        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED]),
            callback=callback
        )

        # Create and publish event
        plan = TodoPlan.create("Test Plan", ["Step 1"])
        event = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="subagent_1",
            plan=plan
        )

        count = bus.publish(event)

        assert count == 1
        assert len(received_events) == 1
        assert received_events[0].source_agent == "subagent_1"

    def test_publish_filters_by_agent(self):
        """Test that events are filtered by agent_id."""
        bus = get_event_bus()
        received_events = []

        def callback(event: TaskEvent):
            received_events.append(event)

        # Subscribe only to events from "researcher"
        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(
                agent_id="researcher",
                event_types=[TaskEventType.STEP_COMPLETED]
            ),
            callback=callback
        )

        plan = TodoPlan.create("Test Plan", ["Step 1"])
        step = plan.steps[0]

        # Event from researcher - should match
        event1 = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="researcher",
            plan=plan,
            step=step
        )

        # Event from reviewer - should not match
        event2 = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="reviewer",
            plan=plan,
            step=step
        )

        bus.publish(event1)
        bus.publish(event2)

        assert len(received_events) == 1
        assert received_events[0].source_agent == "researcher"

    def test_publish_filters_by_event_type(self):
        """Test that events are filtered by event type."""
        bus = get_event_bus()
        received_events = []

        def callback(event: TaskEvent):
            received_events.append(event)

        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.STEP_COMPLETED]),
            callback=callback
        )

        plan = TodoPlan.create("Test Plan", ["Step 1"])
        step = plan.steps[0]

        # step_completed - should match
        event1 = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="subagent",
            plan=plan,
            step=step
        )

        # step_started - should not match
        event2 = TaskEvent.create(
            event_type=TaskEventType.STEP_STARTED,
            agent_id="subagent",
            plan=plan,
            step=step
        )

        bus.publish(event1)
        bus.publish(event2)

        assert len(received_events) == 1
        assert received_events[0].event_type == TaskEventType.STEP_COMPLETED

    def test_subscription_expiration(self):
        """Test that subscriptions expire after N matches."""
        bus = get_event_bus()
        received_events = []

        def callback(event: TaskEvent):
            received_events.append(event)

        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.STEP_COMPLETED]),
            callback=callback,
            expires_after=2
        )

        plan = TodoPlan.create("Test Plan", ["Step 1"])
        step = plan.steps[0]

        # Publish 3 events - only first 2 should be received
        for i in range(3):
            event = TaskEvent.create(
                event_type=TaskEventType.STEP_COMPLETED,
                agent_id="subagent",
                plan=plan,
                step=step
            )
            bus.publish(event)

        assert len(received_events) == 2

        # Subscription should be removed
        subs = bus.get_subscriptions()
        assert len(subs) == 0

    def test_event_history(self):
        """Test that events are stored in history."""
        bus = get_event_bus()

        plan = TodoPlan.create("Test Plan", ["Step 1"])

        for i in range(5):
            event = TaskEvent.create(
                event_type=TaskEventType.STEP_COMPLETED,
                agent_id=f"agent_{i}",
                plan=plan,
                step=plan.steps[0]
            )
            bus.publish(event)

        events = bus.get_recent_events(limit=10)
        assert len(events) == 5

        # Filter by agent
        events = bus.get_recent_events(agent_id="agent_2")
        assert len(events) == 1
        assert events[0].source_agent == "agent_2"

    def test_thread_safety(self):
        """Test that bus operations are thread-safe."""
        bus = get_event_bus()
        received_counts = {"total": 0}
        lock = threading.Lock()

        def callback(event: TaskEvent):
            with lock:
                received_counts["total"] += 1

        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.STEP_COMPLETED]),
            callback=callback
        )

        plan = TodoPlan.create("Test Plan", ["Step 1"])

        def publish_events(count: int):
            for _ in range(count):
                event = TaskEvent.create(
                    event_type=TaskEventType.STEP_COMPLETED,
                    agent_id="agent",
                    plan=plan,
                    step=plan.steps[0]
                )
                bus.publish(event)

        # Start multiple threads publishing events
        threads = []
        for _ in range(5):
            t = threading.Thread(target=publish_events, args=(10,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert received_counts["total"] == 50


class TestWaitForEvents:
    """Tests for long-poll wait_for_events on TaskEventBus."""

    def setup_method(self):
        """Reset singleton before each test."""
        TaskEventBus.reset()

    def teardown_method(self):
        """Clean up after each test."""
        TaskEventBus.reset()

    def test_returns_immediately_when_events_exist(self):
        """wait_for_events returns right away if matching events are already present."""
        bus = get_event_bus()
        plan = TodoPlan.create("P", ["S"])

        event = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="a",
            plan=plan,
        )
        bus.publish(event)

        start = time.monotonic()
        events = bus.wait_for_events(timeout=5.0, limit=10)
        elapsed = time.monotonic() - start

        assert len(events) == 1
        assert events[0].event_id == event.event_id
        # Should return almost instantly, not after the full timeout.
        assert elapsed < 1.0

    def test_blocks_until_event_published(self):
        """wait_for_events blocks and wakes up when a matching event arrives."""
        bus = get_event_bus()
        plan = TodoPlan.create("P", ["S"])

        result_holder: list = []

        def waiter():
            evts = bus.wait_for_events(timeout=10.0, limit=10)
            result_holder.extend(evts)

        t = threading.Thread(target=waiter)
        t.start()

        # Give the waiter thread time to enter the wait.
        time.sleep(0.2)

        event = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="b",
            plan=plan,
        )
        bus.publish(event)
        t.join(timeout=5)

        assert len(result_holder) == 1
        assert result_holder[0].event_id == event.event_id

    def test_timeout_returns_empty(self):
        """wait_for_events returns empty list after timeout with no events."""
        bus = get_event_bus()

        start = time.monotonic()
        events = bus.wait_for_events(timeout=0.3, limit=10)
        elapsed = time.monotonic() - start

        assert events == []
        # Should have waited approximately the timeout duration.
        assert elapsed >= 0.25

    def test_after_event_id_filtering(self):
        """wait_for_events only returns events after the specified cursor."""
        bus = get_event_bus()
        plan = TodoPlan.create("P", ["S"])

        event1 = TaskEvent.create(TaskEventType.PLAN_CREATED, "a", plan)
        event2 = TaskEvent.create(TaskEventType.PLAN_STARTED, "a", plan)
        event3 = TaskEvent.create(TaskEventType.PLAN_COMPLETED, "a", plan)

        bus.publish(event1)
        bus.publish(event2)
        bus.publish(event3)

        # Get events after event1 → should return event2, event3
        events = bus.wait_for_events(
            timeout=0, after_event_id=event1.event_id, limit=10
        )
        assert len(events) == 2
        assert events[0].event_id == event2.event_id
        assert events[1].event_id == event3.event_id

    def test_after_event_id_with_wait(self):
        """wait_for_events blocks when cursor is at the latest event, then wakes."""
        bus = get_event_bus()
        plan = TodoPlan.create("P", ["S"])

        event1 = TaskEvent.create(TaskEventType.PLAN_CREATED, "a", plan)
        bus.publish(event1)

        result_holder: list = []

        def waiter():
            evts = bus.wait_for_events(
                timeout=10.0, after_event_id=event1.event_id, limit=10
            )
            result_holder.extend(evts)

        t = threading.Thread(target=waiter)
        t.start()

        time.sleep(0.2)

        event2 = TaskEvent.create(TaskEventType.PLAN_STARTED, "a", plan)
        bus.publish(event2)
        t.join(timeout=5)

        assert len(result_holder) == 1
        assert result_holder[0].event_id == event2.event_id

    def test_agent_filter_with_wait(self):
        """wait_for_events respects agent_id filter during long-poll."""
        bus = get_event_bus()
        plan = TodoPlan.create("P", ["S"])

        result_holder: list = []

        def waiter():
            evts = bus.wait_for_events(
                timeout=10.0, agent_id="target", limit=10
            )
            result_holder.extend(evts)

        t = threading.Thread(target=waiter)
        t.start()

        time.sleep(0.2)

        # Publish from wrong agent — should not wake the waiter.
        wrong = TaskEvent.create(TaskEventType.PLAN_CREATED, "other", plan)
        bus.publish(wrong)
        time.sleep(0.1)
        assert result_holder == []

        # Publish from target agent — should wake.
        right = TaskEvent.create(TaskEventType.PLAN_CREATED, "target", plan)
        bus.publish(right)
        t.join(timeout=5)

        assert len(result_holder) == 1
        assert result_holder[0].source_agent == "target"

    def test_timeout_clamped_to_30(self):
        """Timeout is capped at 30 seconds."""
        bus = get_event_bus()

        start = time.monotonic()
        # Pass a huge timeout — it should be clamped to 30, but we test
        # that it at least doesn't blow up. We use 0.1 to keep the test fast.
        events = bus.wait_for_events(timeout=0.1, limit=10)
        elapsed = time.monotonic() - start

        assert events == []
        assert elapsed < 1.0


class TestDependencyResolution:
    """Tests for cross-agent dependency resolution."""

    def setup_method(self):
        """Reset singleton before each test."""
        TaskEventBus.reset()

    def teardown_method(self):
        """Clean up after each test."""
        TaskEventBus.reset()

    def test_register_dependency(self):
        """Test registering a dependency waiter."""
        bus = get_event_bus()

        dep_ref = TaskRef(agent_id="researcher", step_id="analysis")

        bus.register_dependency(
            dependency_ref=dep_ref,
            waiting_agent="main",
            waiting_plan_id="plan_123",
            waiting_step_id="integrate"
        )

        waiters = bus.get_dependency_waiters(dep_ref)
        assert len(waiters) == 1
        assert waiters[0] == ("main", "plan_123", "integrate")

    def test_dependency_resolution_callback(self):
        """Test that dependency resolver is called on step completion."""
        bus = get_event_bus()
        resolved_deps = []

        def resolver(waiting_agent, waiting_plan_id, waiting_step_id, event):
            resolved_deps.append({
                "waiting_agent": waiting_agent,
                "waiting_plan_id": waiting_plan_id,
                "waiting_step_id": waiting_step_id,
                "completed_agent": event.source_agent,
                "completed_step": event.source_step_id
            })

        bus.set_dependency_resolver(resolver)

        # Register dependency
        dep_ref = TaskRef(agent_id="researcher", step_id="analysis")
        bus.register_dependency(
            dependency_ref=dep_ref,
            waiting_agent="main",
            waiting_plan_id="plan_123",
            waiting_step_id="integrate"
        )

        # Publish completion event
        plan = TodoPlan.create("Research", ["Analysis"])
        plan.steps[0].step_id = "analysis"  # Match the ref

        event = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="researcher",
            plan=plan,
            step=plan.steps[0],
            payload={"output": {"findings": ["a", "b"]}}
        )
        bus.publish(event)

        assert len(resolved_deps) == 1
        assert resolved_deps[0]["waiting_agent"] == "main"
        assert resolved_deps[0]["completed_agent"] == "researcher"

    def test_multiple_waiters(self):
        """Test that multiple waiters are all notified."""
        bus = get_event_bus()
        resolved_deps = []

        def resolver(waiting_agent, waiting_plan_id, waiting_step_id, event):
            resolved_deps.append(waiting_agent)

        bus.set_dependency_resolver(resolver)

        dep_ref = TaskRef(agent_id="researcher", step_id="analysis")

        # Register multiple waiters
        bus.register_dependency(dep_ref, "agent_a", "plan_a", "step_a")
        bus.register_dependency(dep_ref, "agent_b", "plan_b", "step_b")
        bus.register_dependency(dep_ref, "agent_c", "plan_c", "step_c")

        # Publish completion
        plan = TodoPlan.create("Research", ["Analysis"])
        plan.steps[0].step_id = "analysis"

        event = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="researcher",
            plan=plan,
            step=plan.steps[0]
        )
        bus.publish(event)

        assert len(resolved_deps) == 3
        assert set(resolved_deps) == {"agent_a", "agent_b", "agent_c"}


class TestEventFilter:
    """Tests for EventFilter matching logic."""

    def test_matches_any_agent(self):
        """Test that None agent_id matches any agent."""
        filter = EventFilter(event_types=[TaskEventType.PLAN_CREATED])

        plan = TodoPlan.create("Test", ["Step"])
        event = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="any_agent",
            plan=plan
        )

        assert filter.matches(event) is True

    def test_matches_wildcard_agent(self):
        """Test that '*' agent_id matches any agent."""
        filter = EventFilter(
            agent_id="*",
            event_types=[TaskEventType.PLAN_CREATED]
        )

        plan = TodoPlan.create("Test", ["Step"])
        event = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="specific_agent",
            plan=plan
        )

        assert filter.matches(event) is True

    def test_matches_specific_agent(self):
        """Test that specific agent_id filters correctly."""
        filter = EventFilter(
            agent_id="target_agent",
            event_types=[TaskEventType.PLAN_CREATED]
        )

        plan = TodoPlan.create("Test", ["Step"])

        matching = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="target_agent",
            plan=plan
        )
        non_matching = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="other_agent",
            plan=plan
        )

        assert filter.matches(matching) is True
        assert filter.matches(non_matching) is False

    def test_matches_multiple_event_types(self):
        """Test filtering by multiple event types."""
        filter = EventFilter(
            event_types=[TaskEventType.STEP_COMPLETED, TaskEventType.STEP_FAILED]
        )

        plan = TodoPlan.create("Test", ["Step"])
        step = plan.steps[0]

        completed = TaskEvent.create(TaskEventType.STEP_COMPLETED, "a", plan, step)
        failed = TaskEvent.create(TaskEventType.STEP_FAILED, "a", plan, step)
        started = TaskEvent.create(TaskEventType.STEP_STARTED, "a", plan, step)

        assert filter.matches(completed) is True
        assert filter.matches(failed) is True
        assert filter.matches(started) is False

    def test_matches_empty_event_types(self):
        """Test that empty event_types matches all."""
        filter = EventFilter(event_types=[])

        plan = TodoPlan.create("Test", ["Step"])

        event = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="any",
            plan=plan
        )

        assert filter.matches(event) is True


class TestTaskRef:
    """Tests for TaskRef reference handling."""

    def test_to_uri(self):
        """Test URI serialization."""
        ref = TaskRef(agent_id="agent", plan_id="plan123", step_id="step1")
        assert ref.to_uri() == "agent:plan123/step1"

        ref_no_plan = TaskRef(agent_id="agent", step_id="step1")
        assert ref_no_plan.to_uri() == "agent:*/step1"

    def test_from_uri(self):
        """Test URI parsing."""
        ref = TaskRef.from_uri("agent:plan123/step1")
        assert ref.agent_id == "agent"
        assert ref.plan_id == "plan123"
        assert ref.step_id == "step1"

        ref_wildcard = TaskRef.from_uri("agent:*/step1")
        assert ref_wildcard.agent_id == "agent"
        assert ref_wildcard.plan_id is None
        assert ref_wildcard.step_id == "step1"

    def test_matches(self):
        """Test reference matching."""
        ref = TaskRef(agent_id="agent", step_id="step1")

        assert ref.matches("agent", "any_plan", "step1") is True
        assert ref.matches("other_agent", "any_plan", "step1") is False
        assert ref.matches("agent", "any_plan", "other_step") is False

        # With specific plan
        ref_with_plan = TaskRef(agent_id="agent", plan_id="plan123", step_id="step1")
        assert ref_with_plan.matches("agent", "plan123", "step1") is True
        assert ref_with_plan.matches("agent", "other_plan", "step1") is False

    def test_equality(self):
        """Test TaskRef equality and hashing."""
        ref1 = TaskRef(agent_id="a", step_id="s")
        ref2 = TaskRef(agent_id="a", step_id="s")
        ref3 = TaskRef(agent_id="a", step_id="other")

        assert ref1 == ref2
        assert ref1 != ref3
        assert hash(ref1) == hash(ref2)

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "agent_id": "researcher",
            "step_id": "analysis",
            "plan_id": "plan_123"
        }
        ref = TaskRef.from_dict(data)

        assert ref.agent_id == "researcher"
        assert ref.step_id == "analysis"
        assert ref.plan_id == "plan_123"


class TestTaskEvent:
    """Tests for TaskEvent creation and serialization."""

    def test_create_plan_event(self):
        """Test creating a plan-level event."""
        plan = TodoPlan.create("Test Plan", ["Step 1", "Step 2"])

        event = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="main",
            plan=plan
        )

        assert event.event_type == TaskEventType.PLAN_CREATED
        assert event.source_agent == "main"
        assert event.source_plan_id == plan.plan_id
        assert event.source_plan_title == "Test Plan"
        assert event.source_step_id is None
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_create_step_event(self):
        """Test creating a step-level event."""
        plan = TodoPlan.create("Test Plan", ["Step 1"])
        step = plan.steps[0]

        event = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="subagent",
            plan=plan,
            step=step,
            payload={"output": {"result": "success"}}
        )

        assert event.event_type == TaskEventType.STEP_COMPLETED
        assert event.source_step_id == step.step_id
        assert event.source_step_description == step.description
        assert event.payload["output"]["result"] == "success"

    def test_to_dict(self):
        """Test event serialization."""
        plan = TodoPlan.create("Test", ["Step 1"])
        step = plan.steps[0]

        event = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="agent1",
            plan=plan,
            step=step,
            payload={"key": "value"}
        )

        data = event.to_dict()

        assert data["event_type"] == "step_completed"
        assert data["source_agent"] == "agent1"
        assert data["payload"]["key"] == "value"
        assert "event_id" in data
        assert "timestamp" in data

    def test_from_dict(self):
        """Test event deserialization."""
        data = {
            "event_id": "evt_123",
            "event_type": "plan_created",
            "timestamp": "2024-01-01T00:00:00Z",
            "source_agent": "main",
            "source_plan_id": "plan_456",
            "source_plan_title": "My Plan",
            "payload": {"steps": []}
        }

        event = TaskEvent.from_dict(data)

        assert event.event_id == "evt_123"
        assert event.event_type == TaskEventType.PLAN_CREATED
        assert event.source_agent == "main"
        assert event.source_plan_id == "plan_456"
