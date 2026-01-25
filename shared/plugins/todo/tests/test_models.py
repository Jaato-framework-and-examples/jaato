"""Tests for TODO plugin data models."""

import pytest

from ..models import (
    StepStatus,
    PlanStatus,
    TodoStep,
    TodoPlan,
    ProgressEvent,
)


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_status_values(self):
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"


class TestPlanStatus:
    """Tests for PlanStatus enum."""

    def test_status_values(self):
        assert PlanStatus.ACTIVE.value == "active"
        assert PlanStatus.COMPLETED.value == "completed"
        assert PlanStatus.FAILED.value == "failed"
        assert PlanStatus.CANCELLED.value == "cancelled"


class TestTodoStep:
    """Tests for TodoStep dataclass."""

    def test_create_step(self):
        step = TodoStep.create(1, "First step")

        assert step.sequence == 1
        assert step.description == "First step"
        assert step.status == StepStatus.PENDING
        assert step.step_id is not None
        assert len(step.step_id) > 0

    def test_step_start(self):
        step = TodoStep.create(1, "Test")
        assert step.started_at is None

        step.start()

        assert step.status == StepStatus.IN_PROGRESS
        assert step.started_at is not None

    def test_step_complete(self):
        step = TodoStep.create(1, "Test")
        step.start()
        step.complete("Done successfully")

        assert step.status == StepStatus.COMPLETED
        assert step.completed_at is not None
        assert step.result == "Done successfully"

    def test_step_fail(self):
        step = TodoStep.create(1, "Test")
        step.start()
        step.fail("Something went wrong")

        assert step.status == StepStatus.FAILED
        assert step.completed_at is not None
        assert step.error == "Something went wrong"

    def test_step_skip(self):
        step = TodoStep.create(1, "Test")
        step.skip("Not needed")

        assert step.status == StepStatus.SKIPPED
        assert step.completed_at is not None
        assert step.result == "Not needed"

    def test_step_to_dict(self):
        step = TodoStep.create(1, "Test step")
        step.start()
        step.complete("Done")

        data = step.to_dict()

        assert data["sequence"] == 1
        assert data["description"] == "Test step"
        assert data["status"] == "completed"
        assert data["result"] == "Done"
        assert "step_id" in data
        assert "started_at" in data
        assert "completed_at" in data

    def test_step_from_dict(self):
        data = {
            "step_id": "abc123",
            "sequence": 2,
            "description": "Second step",
            "status": "in_progress",
            "started_at": "2024-01-01T00:00:00Z",
        }

        step = TodoStep.from_dict(data)

        assert step.step_id == "abc123"
        assert step.sequence == 2
        assert step.description == "Second step"
        assert step.status == StepStatus.IN_PROGRESS

    def test_step_from_dict_invalid_status(self):
        data = {
            "step_id": "abc123",
            "sequence": 1,
            "description": "Test",
            "status": "invalid_status",
        }

        step = TodoStep.from_dict(data)

        # Should default to PENDING for invalid status
        assert step.status == StepStatus.PENDING


class TestTodoPlan:
    """Tests for TodoPlan dataclass."""

    def test_create_plan(self):
        plan = TodoPlan.create(
            title="Test Plan",
            step_descriptions=["Step 1", "Step 2", "Step 3"]
        )

        assert plan.title == "Test Plan"
        assert len(plan.steps) == 3
        assert plan.status == PlanStatus.ACTIVE
        assert plan.plan_id is not None
        assert plan.created_at is not None

    def test_create_plan_with_context(self):
        context = {"session_id": "abc123", "user": "test"}
        plan = TodoPlan.create(
            title="Test",
            step_descriptions=["Step 1"],
            context=context
        )

        assert plan.context == context

    def test_get_step_by_id(self):
        plan = TodoPlan.create("Test", ["A", "B", "C"])
        step_id = plan.steps[1].step_id

        found = plan.get_step_by_id(step_id)

        assert found is not None
        assert found.description == "B"

    def test_get_step_by_id_not_found(self):
        plan = TodoPlan.create("Test", ["A"])

        found = plan.get_step_by_id("nonexistent")

        assert found is None

    def test_get_step_by_sequence(self):
        plan = TodoPlan.create("Test", ["A", "B", "C"])

        found = plan.get_step_by_sequence(2)

        assert found is not None
        assert found.description == "B"

    def test_get_step_by_sequence_not_found(self):
        plan = TodoPlan.create("Test", ["A"])

        found = plan.get_step_by_sequence(99)

        assert found is None

    def test_get_current_step(self):
        plan = TodoPlan.create("Test", ["A", "B"])
        plan.current_step = 2

        current = plan.get_current_step()

        assert current is not None
        assert current.description == "B"

    def test_get_current_step_none(self):
        plan = TodoPlan.create("Test", ["A"])

        current = plan.get_current_step()

        assert current is None

    def test_get_next_pending_step(self):
        plan = TodoPlan.create("Test", ["A", "B", "C"])
        plan.steps[0].complete()

        next_step = plan.get_next_pending_step()

        assert next_step is not None
        assert next_step.description == "B"

    def test_get_next_pending_step_all_done(self):
        plan = TodoPlan.create("Test", ["A", "B"])
        plan.steps[0].complete()
        plan.steps[1].complete()

        next_step = plan.get_next_pending_step()

        assert next_step is None

    def test_get_progress(self):
        plan = TodoPlan.create("Test", ["A", "B", "C", "D", "E"])
        plan.steps[0].complete()
        plan.steps[1].complete()
        plan.steps[2].fail()
        plan.steps[3].skip()

        progress = plan.get_progress()

        assert progress["total"] == 5
        assert progress["completed"] == 2
        assert progress["failed"] == 1
        assert progress["skipped"] == 1
        assert progress["in_progress"] == 0
        assert progress["pending"] == 1
        assert progress["percent"] == 40.0

    def test_get_progress_empty(self):
        plan = TodoPlan.create("Test", [])

        progress = plan.get_progress()

        assert progress["total"] == 0
        assert progress["percent"] == 0

    def test_complete_plan(self):
        plan = TodoPlan.create("Test", ["A"])
        plan.complete_plan("All done")

        assert plan.status == PlanStatus.COMPLETED
        assert plan.completed_at is not None
        assert plan.summary == "All done"

    def test_fail_plan(self):
        plan = TodoPlan.create("Test", ["A"])
        plan.fail_plan("Something broke")

        assert plan.status == PlanStatus.FAILED
        assert plan.summary == "Something broke"

    def test_cancel_plan(self):
        plan = TodoPlan.create("Test", ["A"])
        plan.cancel_plan("User cancelled")

        assert plan.status == PlanStatus.CANCELLED
        assert plan.summary == "User cancelled"

    def test_plan_to_dict(self):
        plan = TodoPlan.create("Test Plan", ["Step 1", "Step 2"])
        plan.steps[0].complete()

        data = plan.to_dict()

        assert data["title"] == "Test Plan"
        assert len(data["steps"]) == 2
        assert data["status"] == "active"
        assert "plan_id" in data
        assert "created_at" in data

    def test_plan_from_dict(self):
        data = {
            "plan_id": "plan123",
            "title": "Restored Plan",
            "created_at": "2024-01-01T00:00:00Z",
            "status": "completed",
            "steps": [
                {"step_id": "s1", "sequence": 1, "description": "A", "status": "completed"}
            ],
            "summary": "Done",
        }

        plan = TodoPlan.from_dict(data)

        assert plan.plan_id == "plan123"
        assert plan.title == "Restored Plan"
        assert plan.status == PlanStatus.COMPLETED
        assert len(plan.steps) == 1
        assert plan.summary == "Done"


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_create_event(self):
        plan = TodoPlan.create("Test", ["A", "B"])

        event = ProgressEvent.create("plan_created", plan)

        assert event.event_type == "plan_created"
        assert event.plan_id == plan.plan_id
        assert event.plan_title == "Test"
        assert event.progress is not None
        assert event.timestamp is not None

    def test_create_event_with_step(self):
        plan = TodoPlan.create("Test", ["A"])
        step = plan.steps[0]

        event = ProgressEvent.create("step_completed", plan, step)

        assert event.step is not None
        assert event.step.step_id == step.step_id

    def test_event_to_dict(self):
        plan = TodoPlan.create("Test", ["A"])
        event = ProgressEvent.create("plan_created", plan)

        data = event.to_dict()

        assert data["event_type"] == "plan_created"
        assert "event_id" in data
        assert "timestamp" in data
        assert "progress" in data


# === Cross-Agent Collaboration Tests ===

from ..models import TaskRef, TaskEventType


class TestStepStatusBlocked:
    """Tests for the new BLOCKED step status."""

    def test_blocked_status_value(self):
        assert StepStatus.BLOCKED.value == "blocked"

    def test_create_step_with_dependencies(self):
        """Test creating a step with dependencies sets it to BLOCKED."""
        deps = [
            TaskRef(agent_id="researcher", step_id="analysis"),
            TaskRef(agent_id="reviewer", step_id="review")
        ]
        step = TodoStep.create(1, "Integrate findings", depends_on=deps)

        assert step.status == StepStatus.BLOCKED
        assert len(step.depends_on) == 2
        assert len(step.blocked_by) == 2

    def test_create_step_without_dependencies(self):
        """Test creating a step without dependencies is PENDING."""
        step = TodoStep.create(1, "Simple step")

        assert step.status == StepStatus.PENDING
        assert len(step.depends_on) == 0
        assert len(step.blocked_by) == 0


class TestTodoStepDependencies:
    """Tests for TodoStep dependency management."""

    def test_is_blocked(self):
        """Test is_blocked helper."""
        deps = [TaskRef(agent_id="a", step_id="s")]
        step = TodoStep.create(1, "Test", depends_on=deps)

        assert step.is_blocked() is True

        step.blocked_by = []
        assert step.is_blocked() is False

    def test_add_dependency(self):
        """Test adding a dependency after creation."""
        step = TodoStep.create(1, "Test")
        assert step.status == StepStatus.PENDING

        step.add_dependency(TaskRef(agent_id="a", step_id="s"))

        assert step.status == StepStatus.BLOCKED
        assert len(step.depends_on) == 1
        assert len(step.blocked_by) == 1

    def test_add_dependency_no_duplicates(self):
        """Test that duplicate dependencies are not added."""
        step = TodoStep.create(1, "Test")
        ref = TaskRef(agent_id="a", step_id="s")

        step.add_dependency(ref)
        step.add_dependency(ref)  # Same ref

        assert len(step.depends_on) == 1

    def test_resolve_dependency_unblocks(self):
        """Test resolving all dependencies unblocks the step."""
        dep = TaskRef(agent_id="researcher", step_id="analysis")
        step = TodoStep.create(1, "Integrate", depends_on=[dep])

        assert step.status == StepStatus.BLOCKED

        is_unblocked = step.resolve_dependency(
            dep,
            output={"findings": ["a", "b"]},
            provides_name="research_output"
        )

        assert is_unblocked is True
        assert step.status == StepStatus.PENDING
        assert len(step.blocked_by) == 0
        assert "researcher:analysis" in step.received_outputs
        assert "research_output" in step.received_outputs

    def test_resolve_dependency_partial(self):
        """Test resolving one of multiple dependencies."""
        dep1 = TaskRef(agent_id="a", step_id="s1")
        dep2 = TaskRef(agent_id="b", step_id="s2")
        step = TodoStep.create(1, "Test", depends_on=[dep1, dep2])

        is_unblocked = step.resolve_dependency(dep1, {"result": 1})

        assert is_unblocked is False
        assert step.status == StepStatus.BLOCKED
        assert len(step.blocked_by) == 1
        assert "a:s1" in step.received_outputs

    def test_get_blocking_refs(self):
        """Test getting list of blocking dependencies."""
        deps = [
            TaskRef(agent_id="a", step_id="s1"),
            TaskRef(agent_id="b", step_id="s2")
        ]
        step = TodoStep.create(1, "Test", depends_on=deps)

        blocking = step.get_blocking_refs()

        assert len(blocking) == 2
        # Should be copies, not the same objects
        assert blocking is not step.blocked_by

    def test_complete_with_output(self):
        """Test completing a step with structured output."""
        step = TodoStep.create(1, "Analysis", provides="analysis_result")
        step.start()
        step.complete(result="Done", output={"findings": ["x", "y"]})

        assert step.status == StepStatus.COMPLETED
        assert step.result == "Done"
        assert step.output == {"findings": ["x", "y"]}
        assert step.provides == "analysis_result"

    def test_step_to_dict_with_collaboration_fields(self):
        """Test serialization includes collaboration fields."""
        deps = [TaskRef(agent_id="a", step_id="s1")]
        step = TodoStep.create(1, "Test", depends_on=deps, provides="output")
        step.received_outputs = {"a:s1": {"data": 123}}

        data = step.to_dict()

        assert "depends_on" in data
        assert len(data["depends_on"]) == 1
        assert data["depends_on"][0]["agent_id"] == "a"
        assert data["provides"] == "output"
        assert data["received_outputs"] == {"a:s1": {"data": 123}}

    def test_step_from_dict_with_collaboration_fields(self):
        """Test deserialization handles collaboration fields."""
        data = {
            "step_id": "step_123",
            "sequence": 1,
            "description": "Test",
            "status": "blocked",
            "depends_on": [{"agent_id": "a", "step_id": "s1"}],
            "blocked_by": [{"agent_id": "a", "step_id": "s1"}],
            "provides": "my_output",
            "output": {"result": 42},
            "received_outputs": {"a:s1": {"data": "received"}}
        }

        step = TodoStep.from_dict(data)

        assert step.status == StepStatus.BLOCKED
        assert len(step.depends_on) == 1
        assert step.depends_on[0].agent_id == "a"
        assert step.provides == "my_output"
        assert step.output == {"result": 42}
        assert step.received_outputs == {"a:s1": {"data": "received"}}


class TestTodoPlanDependencies:
    """Tests for TodoPlan dependency-related methods."""

    def test_get_blocked_steps(self):
        """Test getting all blocked steps."""
        plan = TodoPlan.create("Test", ["A", "B", "C"])

        # Block step B
        plan.steps[1].add_dependency(TaskRef(agent_id="x", step_id="y"))

        blocked = plan.get_blocked_steps()

        assert len(blocked) == 1
        assert blocked[0].description == "B"

    def test_get_blocked_steps_none(self):
        """Test no blocked steps returns empty list."""
        plan = TodoPlan.create("Test", ["A", "B"])

        blocked = plan.get_blocked_steps()

        assert blocked == []

    def test_add_step_with_dependencies(self):
        """Test adding a step with dependencies."""
        plan = TodoPlan.create("Test", ["Step 1"])
        plan.started = True

        deps = [TaskRef(agent_id="researcher", step_id="done")]
        new_step = plan.add_step(
            description="Dependent step",
            depends_on=deps,
            provides="my_output"
        )

        assert new_step.status == StepStatus.BLOCKED
        assert len(new_step.depends_on) == 1
        assert new_step.provides == "my_output"

    def test_resolve_dependencies_from(self):
        """Test resolving dependencies from a completed step."""
        plan = TodoPlan.create("Test", ["Step 1"])
        plan.started = True

        # Add dependent step
        dep = TaskRef(agent_id="researcher", step_id="analysis")
        step2 = plan.add_step("Integrate", depends_on=[dep])

        assert step2.status == StepStatus.BLOCKED

        # Simulate completion of dependency
        unblocked = plan.resolve_dependencies_from(
            completed_agent="researcher",
            completed_plan_id="plan_123",
            completed_step_id="analysis",
            output={"findings": ["a", "b"]},
            provides_name="research_output"
        )

        assert len(unblocked) == 1
        assert unblocked[0] is step2
        assert step2.status == StepStatus.PENDING
        assert "researcher:analysis" in step2.received_outputs

    def test_get_progress_includes_blocked(self):
        """Test that progress stats include blocked count."""
        plan = TodoPlan.create("Test", ["A", "B", "C"])
        plan.steps[1].status = StepStatus.BLOCKED

        progress = plan.get_progress()

        assert progress["blocked"] == 1
        assert progress["pending"] == 2

    def test_get_step_by_provides(self):
        """Test finding a step by its provides name."""
        plan = TodoPlan.create("Test", ["Step 1"])
        plan.started = True
        plan.add_step("Analysis", provides="analysis_output")

        found = plan.get_step_by_provides("analysis_output")

        assert found is not None
        assert found.provides == "analysis_output"

    def test_get_step_by_provides_not_found(self):
        """Test that missing provides name returns None."""
        plan = TodoPlan.create("Test", ["Step 1"])

        found = plan.get_step_by_provides("nonexistent")

        assert found is None


class TestTaskEventTypeEnum:
    """Tests for TaskEventType enum."""

    def test_plan_events(self):
        assert TaskEventType.PLAN_CREATED.value == "plan_created"
        assert TaskEventType.PLAN_STARTED.value == "plan_started"
        assert TaskEventType.PLAN_COMPLETED.value == "plan_completed"
        assert TaskEventType.PLAN_FAILED.value == "plan_failed"
        assert TaskEventType.PLAN_CANCELLED.value == "plan_cancelled"

    def test_step_events(self):
        assert TaskEventType.STEP_ADDED.value == "step_added"
        assert TaskEventType.STEP_STARTED.value == "step_started"
        assert TaskEventType.STEP_COMPLETED.value == "step_completed"
        assert TaskEventType.STEP_FAILED.value == "step_failed"
        assert TaskEventType.STEP_SKIPPED.value == "step_skipped"

    def test_collaboration_events(self):
        assert TaskEventType.STEP_BLOCKED.value == "step_blocked"
        assert TaskEventType.STEP_UNBLOCKED.value == "step_unblocked"
