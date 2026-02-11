"""Regression tests for plan locking and dependency resolution fixes.

These tests cover root causes from coordination bugs:
1. Concurrent read-modify-write race condition (lost update)
2. Event bus pop-before-resolve losing waiters on failure
3. register_dependency firing before save_plan
4. Late subscriber missing plan_created events (no history replay)
"""

import threading
import time
import tempfile
from unittest.mock import Mock

import pytest

from ..plugin import TodoPlugin
from ..models import (
    StepStatus, TaskEvent, TaskEventType, TaskRef, TodoPlan, EventFilter
)
from ..event_bus import TaskEventBus
from ..storage import FileStorage, InMemoryStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plugin(storage_type="memory", storage_path=None):
    """Create and initialize a TodoPlugin with the given storage backend."""
    plugin = TodoPlugin()
    config = {"storage_type": storage_type}
    if storage_path:
        config["storage_path"] = storage_path
    plugin.initialize(config)
    return plugin


def _create_started_plan(plugin, title="Test Plan", steps=None):
    """Create and start a plan, returning the plan object."""
    steps = steps or ["Step 1", "Step 2", "Step 3"]
    result = plugin._execute_create_plan({"title": title, "steps": steps})
    assert "error" not in result, result.get("error")
    plugin._execute_start_plan({"message": "go"})
    return plugin._get_current_plan()


# ---------------------------------------------------------------------------
# Root Cause 1: Concurrent read-modify-write race condition
# ---------------------------------------------------------------------------

class TestPlanLockingConcurrency:
    """Test that per-plan locking prevents lost updates under parallel tool execution."""

    def setup_method(self):
        TaskEventBus.reset()

    def teardown_method(self):
        TaskEventBus.reset()

    def test_concurrent_add_step_and_update_step_with_file_storage(self):
        """Regression: parallel addDependentStep + updateStep should not lose the added step.

        With FileStorage, each get_plan returns a new deserialized object.
        Without locking, the last save_plan wins and the other's changes are lost.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = _make_plugin("file", f"{tmpdir}/plans")

            plan = _create_started_plan(plugin, steps=["A", "B", "C", "D", "E", "F"])
            plan_id = plan.plan_id
            step_ids = [s.step_id for s in plan.steps]

            barrier = threading.Barrier(2)
            errors = []

            def add_step_thread():
                try:
                    barrier.wait(timeout=5)
                    result = plugin._execute_add_step({
                        "description": "New step G",
                    })
                    if result.get("error"):
                        errors.append(f"addStep error: {result['error']}")
                except Exception as e:
                    errors.append(f"addStep exception: {e}")

            def update_step_thread():
                try:
                    barrier.wait(timeout=5)
                    result = plugin._execute_update_step({
                        "step_id": step_ids[0],
                        "status": "completed",
                        "result": "done",
                    })
                    if result.get("error"):
                        errors.append(f"updateStep error: {result['error']}")
                except Exception as e:
                    errors.append(f"updateStep exception: {e}")

            t1 = threading.Thread(target=add_step_thread)
            t2 = threading.Thread(target=update_step_thread)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert not errors, f"Thread errors: {errors}"

            # Both changes must be visible
            final_plan = plugin._storage.get_plan(plan_id)
            assert len(final_plan.steps) == 7, (
                f"Expected 7 steps (6 original + 1 added), got {len(final_plan.steps)}"
            )
            completed = [s for s in final_plan.steps if s.status == StepStatus.COMPLETED]
            assert len(completed) == 1, f"Expected 1 completed step, got {len(completed)}"

    def test_concurrent_update_steps_with_file_storage(self):
        """Two concurrent updateStep calls should both persist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = _make_plugin("file", f"{tmpdir}/plans")

            plan = _create_started_plan(plugin, steps=["A", "B", "C"])
            plan_id = plan.plan_id
            step_ids = [s.step_id for s in plan.steps]

            barrier = threading.Barrier(2)
            errors = []

            def complete_step(idx):
                try:
                    barrier.wait(timeout=5)
                    result = plugin._execute_update_step({
                        "step_id": step_ids[idx],
                        "status": "completed",
                        "result": f"done-{idx}",
                    })
                    if result.get("error"):
                        errors.append(f"updateStep[{idx}] error: {result['error']}")
                except Exception as e:
                    errors.append(f"updateStep[{idx}] exception: {e}")

            t1 = threading.Thread(target=complete_step, args=(0,))
            t2 = threading.Thread(target=complete_step, args=(1,))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert not errors, f"Thread errors: {errors}"

            final_plan = plugin._storage.get_plan(plan_id)
            completed = [s for s in final_plan.steps if s.status == StepStatus.COMPLETED]
            assert len(completed) == 2, (
                f"Expected 2 completed steps, got {len(completed)}: "
                f"{[(s.step_id[:8], s.status.value) for s in final_plan.steps]}"
            )

    def test_plan_lock_is_per_plan(self):
        """Locks for different plans should be independent."""
        plugin = _make_plugin()
        lock_a = plugin._get_plan_lock("plan-a")
        lock_b = plugin._get_plan_lock("plan-b")
        assert lock_a is not lock_b

        # Same plan_id returns same lock
        lock_a2 = plugin._get_plan_lock("plan-a")
        assert lock_a is lock_a2


# ---------------------------------------------------------------------------
# Root Cause 2: Event bus pop-before-resolve
# ---------------------------------------------------------------------------

class TestDependencyResolverRetry:
    """Test that failed dependency resolution retains waiters for retry."""

    def setup_method(self):
        TaskEventBus.reset()

    def teardown_method(self):
        TaskEventBus.reset()

    def test_failed_resolver_retains_waiter(self):
        """If the resolver callback raises, the waiter should be re-added."""
        bus = TaskEventBus.get_instance()

        call_count = [0]

        def failing_then_succeeding_resolver(agent, plan_id, step_id, event):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Plan not found yet")
            # Second call succeeds

        bus.set_dependency_resolver(failing_then_succeeding_resolver)

        # Register a dependency waiter
        dep_ref = TaskRef(agent_id="subagent_1", step_id="step-4", plan_id="sub-plan")
        bus.register_dependency(
            dependency_ref=dep_ref,
            waiting_agent="main",
            waiting_plan_id="parent-plan",
            waiting_step_id="step-7",
        )

        # First completion event — resolver fails, waiter should be retained
        event1 = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="subagent_1",
            plan=TodoPlan.create("Sub Plan", ["s1"]),
            step=None,
        )
        event1.source_step_id = "step-4"
        event1.source_plan_id = "sub-plan"
        bus.publish(event1)

        assert call_count[0] == 1, "Resolver should have been called once"

        # Waiter should still exist
        key = dep_ref.to_uri()
        assert key in bus._dependency_waiters, "Waiter should be retained after failure"

        # Second completion event — resolver succeeds
        bus.publish(event1)
        assert call_count[0] == 2, "Resolver should have been called again"

        # Now the waiter should be gone (resolved successfully)
        assert key not in bus._dependency_waiters, "Waiter should be removed after success"

    def test_successful_resolver_removes_waiter(self):
        """If the resolver succeeds, the waiter should not be re-added."""
        bus = TaskEventBus.get_instance()

        resolved = []

        def success_resolver(agent, plan_id, step_id, event):
            resolved.append(step_id)

        bus.set_dependency_resolver(success_resolver)

        dep_ref = TaskRef(agent_id="sub", step_id="s1", plan_id="p1")
        bus.register_dependency(
            dependency_ref=dep_ref,
            waiting_agent="main",
            waiting_plan_id="parent",
            waiting_step_id="waiting-step",
        )

        event = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="sub",
            plan=TodoPlan.create("Plan", ["x"]),
            step=None,
        )
        event.source_step_id = "s1"
        event.source_plan_id = "p1"
        bus.publish(event)

        assert resolved == ["waiting-step"]
        key = dep_ref.to_uri()
        assert key not in bus._dependency_waiters


# ---------------------------------------------------------------------------
# Root Cause 3: register_dependency before save_plan timing window
# ---------------------------------------------------------------------------

class TestDependencyRegistrationOrder:
    """Test that dependencies are registered after the plan is saved."""

    def setup_method(self):
        TaskEventBus.reset()

    def teardown_method(self):
        TaskEventBus.reset()

    def test_add_dependent_step_saves_before_registering(self):
        """addDependentStep must save the plan before registering the dependency waiter.

        We verify this by checking that the step (with its new dependency)
        exists in storage at the time the event bus register_dependency is called.
        """
        plugin = _make_plugin()
        plan = _create_started_plan(plugin, steps=["Step 1", "Step 2"])
        target_step_id = plan.steps[1].step_id  # "Step 2"

        # Patch register_dependency to verify plan state at call time
        original_register = plugin._event_bus.register_dependency
        step_in_storage_at_register_time = [None]

        def checking_register(dependency_ref, waiting_agent, waiting_plan_id, waiting_step_id):
            # At this point, the plan should already be saved with the dependency
            current_plan = plugin._storage.get_plan(waiting_plan_id)
            step = current_plan.get_step_by_id(waiting_step_id) if current_plan else None
            step_in_storage_at_register_time[0] = step is not None and len(step.depends_on) > 0
            return original_register(dependency_ref, waiting_agent, waiting_plan_id, waiting_step_id)

        plugin._event_bus.register_dependency = checking_register

        # Add a dependency to an existing step
        dep_ref = {"agent_id": "sub", "step_id": "sub-step-1"}
        result = plugin._execute_add_dependent_step({
            "step_id": target_step_id,
            "depends_on": [dep_ref],
        })

        assert "error" not in result, result.get("error")
        assert step_in_storage_at_register_time[0] is True, (
            "Step with dependency should be in storage when register_dependency is called"
        )

    def test_dependency_resolved_raises_on_missing_plan(self):
        """_on_dependency_resolved should raise (not silently return) when plan is missing.

        This allows the event bus to retain the waiter for retry.
        """
        plugin = _make_plugin()

        event = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="sub",
            plan=TodoPlan.create("Plan", ["x"]),
            step=None,
        )
        event.source_step_id = "some-step"
        event.source_plan_id = "some-plan"

        with pytest.raises(RuntimeError, match="Plan.*not found"):
            plugin._on_dependency_resolved(
                waiting_agent="main",
                waiting_plan_id="nonexistent-plan-id",
                waiting_step_id="nonexistent-step-id",
                completion_event=event,
            )

    def test_dependency_resolved_raises_on_missing_step(self):
        """_on_dependency_resolved should raise when step is missing from plan."""
        plugin = _make_plugin()
        plan = _create_started_plan(plugin)

        event = TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="sub",
            plan=TodoPlan.create("Plan", ["x"]),
            step=None,
        )
        event.source_step_id = "some-step"
        event.source_plan_id = "some-plan"

        with pytest.raises(RuntimeError, match="Step.*not found"):
            plugin._on_dependency_resolved(
                waiting_agent="main",
                waiting_plan_id=plan.plan_id,
                waiting_step_id="nonexistent-step-id",
                completion_event=event,
            )


# ---------------------------------------------------------------------------
# Integration: end-to-end addDependentStep visibility
# ---------------------------------------------------------------------------

class TestAddDependentStepVisibility:
    """End-to-end test for the original bug: addDependentStep step should
    be visible in getPlanStatus."""

    def setup_method(self):
        TaskEventBus.reset()

    def teardown_method(self):
        TaskEventBus.reset()

    def test_dependent_step_visible_in_plan_status(self):
        """A step with dependencies added by addDependentStep must show as blocked in getPlanStatus."""
        plugin = _make_plugin()
        plan = _create_started_plan(plugin, steps=[
            "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"
        ])
        target_step_id = plan.steps[5].step_id  # "Step 6"

        # Add a dependency to an existing step
        result = plugin._execute_add_dependent_step({
            "step_id": target_step_id,
            "depends_on": [{"agent_id": "subagent_1", "step_id": "sub-step-4"}],
        })
        assert "error" not in result
        # No new step created — still 6 steps
        assert result["total_steps"] == 6

        # Verify getPlanStatus sees 6 steps with 1 blocked
        status = plugin._execute_get_plan_status({})
        assert len(status["steps"]) == 6, (
            f"Expected 6 steps in getPlanStatus, got {len(status['steps'])}"
        )
        assert status["progress"]["total"] == 6
        assert status["progress"]["blocked"] == 1

    def test_dependent_step_visible_after_concurrent_update(self):
        """The dependency added to an existing step must survive a concurrent step completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = _make_plugin("file", f"{tmpdir}/plans")
            plan = _create_started_plan(plugin, steps=[
                "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"
            ])
            step_ids = [s.step_id for s in plan.steps]
            target_step_id = step_ids[5]  # "Step 6" — will get the dependency

            # Mark step 1 as in_progress
            plugin._execute_update_step({
                "step_id": step_ids[0], "status": "in_progress"
            })

            barrier = threading.Barrier(2)
            errors = []

            def add_dependent():
                try:
                    barrier.wait(timeout=5)
                    result = plugin._execute_add_dependent_step({
                        "step_id": target_step_id,
                        "depends_on": [{"agent_id": "sub", "step_id": "s1"}],
                    })
                    if result.get("error"):
                        errors.append(f"addDependentStep: {result['error']}")
                except Exception as e:
                    errors.append(f"addDependentStep: {e}")

            def complete_step():
                try:
                    barrier.wait(timeout=5)
                    result = plugin._execute_update_step({
                        "step_id": step_ids[0],
                        "status": "completed",
                        "result": "done",
                    })
                    if result.get("error"):
                        errors.append(f"updateStep: {result['error']}")
                except Exception as e:
                    errors.append(f"updateStep: {e}")

            t1 = threading.Thread(target=add_dependent)
            t2 = threading.Thread(target=complete_step)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert not errors, f"Thread errors: {errors}"

            # The dependency on step 6 and the completed step 1 must both be visible
            status = plugin._execute_get_plan_status({})
            assert len(status["steps"]) == 6, (
                f"Expected 6 steps, got {len(status['steps'])}"
            )
            assert status["progress"]["total"] == 6
            assert status["progress"]["blocked"] == 1
            assert status["progress"]["completed"] == 1


# ---------------------------------------------------------------------------
# Root Cause 4: Late subscriber misses plan_created events (no history replay)
# ---------------------------------------------------------------------------

class TestLateSubscriberReplay:
    """Test that subscribing after events are published still delivers them.

    This covers the race condition where a parent agent subscribes to
    plan_created events AFTER subagents have already created their plans.
    The event bus must replay matching historical events to late subscribers.
    """

    def setup_method(self):
        TaskEventBus.reset()

    def teardown_method(self):
        TaskEventBus.reset()

    def test_subscribe_after_publish_replays_history(self):
        """Events published before subscribe() should be replayed to the callback."""
        bus = TaskEventBus.get_instance()

        # Subagent publishes plan_created BEFORE parent subscribes
        plan = TodoPlan.create("Subagent Plan", ["step1", "step2"])
        event = TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="subagent_1",
            plan=plan,
        )
        bus.publish(event)

        # Parent subscribes AFTER the event was published
        received = []
        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED]),
            callback=lambda e: received.append(e),
        )

        # The historical event should have been replayed
        assert len(received) == 1, (
            f"Expected 1 replayed event, got {len(received)}"
        )
        assert received[0].source_agent == "subagent_1"
        assert received[0].event_type == TaskEventType.PLAN_CREATED

    def test_replay_respects_filter(self):
        """Only events matching the filter should be replayed."""
        bus = TaskEventBus.get_instance()

        plan = TodoPlan.create("Plan", ["s1"])

        # Publish two different event types
        bus.publish(TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="sub1",
            plan=plan,
        ))
        bus.publish(TaskEvent.create(
            event_type=TaskEventType.STEP_COMPLETED,
            agent_id="sub1",
            plan=plan,
        ))

        # Subscribe only to plan_created
        received = []
        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED]),
            callback=lambda e: received.append(e),
        )

        assert len(received) == 1
        assert received[0].event_type == TaskEventType.PLAN_CREATED

    def test_replay_respects_agent_filter(self):
        """Only events from the specified agent should be replayed."""
        bus = TaskEventBus.get_instance()

        plan = TodoPlan.create("Plan", ["s1"])

        bus.publish(TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="sub1",
            plan=plan,
        ))
        bus.publish(TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="sub2",
            plan=plan,
        ))

        # Subscribe only to events from sub1
        received = []
        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(
                agent_id="sub1",
                event_types=[TaskEventType.PLAN_CREATED],
            ),
            callback=lambda e: received.append(e),
        )

        assert len(received) == 1
        assert received[0].source_agent == "sub1"

    def test_replay_disabled_skips_history(self):
        """replay_history=False should not replay anything."""
        bus = TaskEventBus.get_instance()

        plan = TodoPlan.create("Plan", ["s1"])
        bus.publish(TaskEvent.create(
            event_type=TaskEventType.PLAN_CREATED,
            agent_id="sub1",
            plan=plan,
        ))

        received = []
        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED]),
            callback=lambda e: received.append(e),
            replay_history=False,
        )

        assert len(received) == 0

    def test_no_duplicate_delivery_concurrent(self):
        """An event should be delivered exactly once, not both via replay and live.

        This tests the critical invariant: the atomic lock in publish() ensures
        that an event is either in the history snapshot (replayed by subscribe)
        OR delivered via the live subscription, never both.
        """
        bus = TaskEventBus.get_instance()

        plan = TodoPlan.create("Plan", ["s1"])
        received = []
        lock = threading.Lock()

        def on_event(e):
            with lock:
                received.append(e)

        # Run many concurrent publish + subscribe pairs to stress the locking
        errors = []
        barrier = threading.Barrier(2)

        def publisher():
            try:
                barrier.wait(timeout=5)
                for i in range(20):
                    bus.publish(TaskEvent.create(
                        event_type=TaskEventType.PLAN_CREATED,
                        agent_id=f"sub_{i}",
                        plan=plan,
                    ))
            except Exception as e:
                errors.append(f"publisher: {e}")

        def subscriber():
            try:
                barrier.wait(timeout=5)
                time.sleep(0.01)  # Let some events publish first
                bus.subscribe(
                    subscriber_agent="main",
                    filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED]),
                    callback=on_event,
                )
            except Exception as e:
                errors.append(f"subscriber: {e}")

        t1 = threading.Thread(target=publisher)
        t2 = threading.Thread(target=subscriber)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

        # All 20 events should be received exactly once
        agent_ids = [e.source_agent for e in received]
        unique_agents = set(agent_ids)
        assert len(agent_ids) == len(unique_agents), (
            f"Duplicate events detected! Got {len(agent_ids)} events "
            f"but only {len(unique_agents)} unique agents. "
            f"Duplicates: {[a for a in agent_ids if agent_ids.count(a) > 1]}"
        )
        assert len(received) == 20, (
            f"Expected 20 events, got {len(received)}"
        )

    def test_replay_counts_toward_expires_after(self):
        """Replayed events should count toward expires_after limit."""
        bus = TaskEventBus.get_instance()

        plan = TodoPlan.create("Plan", ["s1"])

        # Publish 3 events before subscribing
        for i in range(3):
            bus.publish(TaskEvent.create(
                event_type=TaskEventType.PLAN_CREATED,
                agent_id=f"sub_{i}",
                plan=plan,
            ))

        received = []
        bus.subscribe(
            subscriber_agent="main",
            filter=EventFilter(event_types=[TaskEventType.PLAN_CREATED]),
            callback=lambda e: received.append(e),
            expires_after=2,  # Only want 2 events total
        )

        # Should only get 2 events (expired after that)
        assert len(received) == 2

        # Subscription should have been removed
        assert len(bus.get_subscriptions("main")) == 0
