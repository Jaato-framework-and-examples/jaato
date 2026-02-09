"""Regression tests for plan locking and dependency resolution fixes.

These tests cover three root causes from the addDependentStep visibility bug:
1. Concurrent read-modify-write race condition (lost update)
2. Event bus pop-before-resolve losing waiters on failure
3. register_dependency firing before save_plan
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

        We verify this by checking that the step exists in storage at the
        time the event bus register_dependency is called.
        """
        plugin = _make_plugin()
        plan = _create_started_plan(plugin, steps=["Step 1", "Step 2"])

        # Patch register_dependency to verify plan state at call time
        original_register = plugin._event_bus.register_dependency
        step_in_storage_at_register_time = [None]

        def checking_register(dependency_ref, waiting_agent, waiting_plan_id, waiting_step_id):
            # At this point, the plan should already be saved with the new step
            current_plan = plugin._storage.get_plan(waiting_plan_id)
            step = current_plan.get_step_by_id(waiting_step_id) if current_plan else None
            step_in_storage_at_register_time[0] = step is not None
            return original_register(dependency_ref, waiting_agent, waiting_plan_id, waiting_step_id)

        plugin._event_bus.register_dependency = checking_register

        # Create a dependent step
        dep_ref = {"agent_id": "sub", "step_id": "sub-step-1"}
        result = plugin._execute_add_dependent_step({
            "description": "Wait for subagent",
            "depends_on": [dep_ref],
        })

        assert "error" not in result, result.get("error")
        assert step_in_storage_at_register_time[0] is True, (
            "Step should be in storage when register_dependency is called"
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
        """The step created by addDependentStep must appear in getPlanStatus."""
        plugin = _make_plugin()
        plan = _create_started_plan(plugin, steps=[
            "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"
        ])

        # Add a dependent step
        result = plugin._execute_add_dependent_step({
            "description": "Wait for subagent results",
            "depends_on": [{"agent_id": "subagent_1", "step_id": "sub-step-4"}],
        })
        assert "error" not in result
        assert result["total_steps"] == 7

        # Verify getPlanStatus sees 7 steps
        status = plugin._execute_get_plan_status({})
        assert len(status["steps"]) == 7, (
            f"Expected 7 steps in getPlanStatus, got {len(status['steps'])}"
        )
        assert status["progress"]["total"] == 7
        assert status["progress"]["blocked"] == 1

    def test_dependent_step_visible_after_concurrent_update(self):
        """The dependent step must survive a concurrent step completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = _make_plugin("file", f"{tmpdir}/plans")
            plan = _create_started_plan(plugin, steps=[
                "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"
            ])
            step_ids = [s.step_id for s in plan.steps]

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
                        "description": "Wait for subagent",
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

            # Both the dependent step AND the completed step must be visible
            status = plugin._execute_get_plan_status({})
            assert len(status["steps"]) == 7, (
                f"Expected 7 steps, got {len(status['steps'])}"
            )
            assert status["progress"]["total"] == 7
            assert status["progress"]["blocked"] == 1
            assert status["progress"]["completed"] == 1
