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
