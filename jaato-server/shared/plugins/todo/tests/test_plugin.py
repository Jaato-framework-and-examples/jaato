"""Tests for the TODO plugin integration."""

import json
import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch

import pytest

from ..plugin import TodoPlugin, create_plugin
from ..event_bus import TaskEventBus, get_event_bus
from jaato_sdk.plugins.todo.models import (
    StepStatus, PlanStatus, TaskEvent, TaskEventType, TodoPlan,
)


class TestTodoPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, TodoPlugin)

    def test_plugin_name(self):
        plugin = TodoPlugin()
        assert plugin.name == "todo"

    def test_initialize_without_config(self):
        plugin = TodoPlugin()
        plugin.initialize()
        assert plugin._initialized is True

    def test_initialize_with_reporter_type(self):
        plugin = TodoPlugin()
        plugin.initialize({"reporter_type": "console"})
        assert plugin._reporter is not None
        assert plugin._reporter.name == "console"

    def test_initialize_with_storage_type(self):
        plugin = TodoPlugin()
        plugin.initialize({"storage_type": "memory"})
        assert plugin._storage is not None

    def test_shutdown(self):
        plugin = TodoPlugin()
        plugin.initialize()
        plugin.shutdown()

        assert plugin._initialized is False
        # Note: _storage and _reporter are intentionally preserved across shutdown
        # for cross-agent collaboration - plans should persist when one agent
        # shuts down while others continue using the shared plugin


class TestTodoPluginToolSchemas:
    """Tests for tool schemas."""

    def test_get_tool_schemas(self):
        plugin = TodoPlugin()
        schemas = plugin.get_tool_schemas()

        # Core tools (always present)
        names = {s.name for s in schemas}
        assert "createPlan" in names
        assert "setStepStatus" in names
        assert "getPlanStatus" in names
        assert "completePlan" in names
        # Additional tools for cross-agent collaboration
        assert "startPlan" in names
        assert "addStep" in names
        assert "subscribeToTasks" in names
        assert "addDependentStep" in names
        assert "completeStepWithOutput" in names
        assert "getBlockedSteps" in names
        assert "getTaskEvents" in names
        assert "listSubscriptions" in names
        assert "unsubscribe" in names
        assert len(schemas) == 13

    def test_createPlan_schema(self):
        plugin = TodoPlugin()
        schemas = plugin.get_tool_schemas()
        create_plan = next(s for s in schemas if s.name == "createPlan")
        schema = create_plan.parameters

        assert schema["type"] == "object"
        assert "title" in schema["properties"]
        assert "steps" in schema["properties"]
        assert "title" in schema["required"]
        assert "steps" in schema["required"]

    def test_setStepStatus_schema(self):
        plugin = TodoPlugin()
        schemas = plugin.get_tool_schemas()
        set_step_status = next(s for s in schemas if s.name == "setStepStatus")
        schema = set_step_status.parameters

        assert "step_id" in schema["properties"]
        assert "status" in schema["properties"]
        assert "result" in schema["properties"]
        assert "error" in schema["properties"]


class TestTodoPluginExecutors:
    """Tests for executor methods."""

    def test_get_executors(self):
        plugin = TodoPlugin()
        executors = plugin.get_executors()

        assert "createPlan" in executors
        assert "setStepStatus" in executors
        assert "getPlanStatus" in executors
        assert "completePlan" in executors
        for executor in executors.values():
            assert callable(executor)


class TestCreatePlanExecutor:
    """Tests for createPlan executor."""

    def test_create_plan_success(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["createPlan"]({
            "title": "Test Plan",
            "steps": ["Step 1", "Step 2", "Step 3"]
        })

        assert "plan_id" in result
        assert result["title"] == "Test Plan"
        assert result["status"] == "active"
        assert len(result["steps"]) == 3
        assert "progress" in result

    def test_create_plan_missing_title(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["createPlan"]({
            "steps": ["Step 1"]
        })

        assert "error" in result

    def test_create_plan_missing_steps(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["createPlan"]({
            "title": "Test"
        })

        assert "error" in result

    def test_create_plan_empty_steps(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["createPlan"]({
            "title": "Test",
            "steps": []
        })

        assert "error" in result

    def test_create_plan_sets_current_plan(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })

        # Plugin now uses dict mapping agent_name -> plan_id for multi-agent support
        # With no session set, _get_agent_name() defaults to "main"
        assert plugin._current_plan_ids.get("main") == result["plan_id"]

    def test_create_plan_reports_creation(self):
        plugin = TodoPlugin()
        plugin.initialize()

        mock_reporter = Mock()
        plugin._reporter = mock_reporter

        executors = plugin.get_executors()
        executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })

        mock_reporter.report_plan_created.assert_called_once()


class TestSetStepStatusExecutor:
    """Tests for setStepStatus executor."""

    def test_set_step_status_to_in_progress(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        # Create and start plan first (setStepStatus requires plan to be started)
        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1", "Step 2"]
        })
        executors["startPlan"]({})  # Start the plan
        step_id = create_result["steps"][0]["step_id"]

        # Update step
        result = executors["setStepStatus"]({
            "step_id": step_id,
            "status": "in_progress"
        })

        assert result["status"] == "in_progress"
        assert result["step_id"] == step_id

    def test_set_step_status_to_completed(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        executors["startPlan"]({})  # Start the plan
        step_id = create_result["steps"][0]["step_id"]

        result = executors["setStepStatus"]({
            "step_id": step_id,
            "status": "completed",
            "result": "Done successfully"
        })

        assert result["status"] == "completed"
        assert result["result"] == "Done successfully"

    def test_set_step_status_to_failed(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        executors["startPlan"]({})  # Start the plan
        step_id = create_result["steps"][0]["step_id"]

        result = executors["setStepStatus"]({
            "step_id": step_id,
            "status": "failed",
            "error": "Something went wrong"
        })

        assert result["status"] == "failed"
        assert result["error"] == "Something went wrong"

    def test_set_step_status_to_skipped(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        executors["startPlan"]({})  # Start the plan
        step_id = create_result["steps"][0]["step_id"]

        result = executors["setStepStatus"]({
            "step_id": step_id,
            "status": "skipped",
            "result": "Not needed"
        })

        assert result["status"] == "skipped"

    def test_set_step_status_no_plan(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["setStepStatus"]({
            "step_id": "some-id",
            "status": "completed"
        })

        assert "error" in result
        assert "No active plan" in result["error"]

    def test_set_step_status_not_found(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        executors["startPlan"]({})  # Start the plan

        result = executors["setStepStatus"]({
            "step_id": "nonexistent",
            "status": "completed"
        })

        assert "error" in result
        assert "Step not found" in result["error"]

    def test_set_step_status_invalid_status(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        step_id = create_result["steps"][0]["step_id"]

        result = executors["setStepStatus"]({
            "step_id": step_id,
            "status": "invalid_status"
        })

        assert "error" in result

    def test_set_step_status_reports_update(self):
        plugin = TodoPlugin()
        plugin.initialize()

        mock_reporter = Mock()
        plugin._reporter = mock_reporter

        executors = plugin.get_executors()
        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })
        executors["startPlan"]({})  # Start the plan
        step_id = create_result["steps"][0]["step_id"]

        mock_reporter.reset_mock()
        executors["setStepStatus"]({
            "step_id": step_id,
            "status": "completed"
        })

        mock_reporter.report_step_update.assert_called_once()


class TestGetPlanStatusExecutor:
    """Tests for getPlanStatus executor."""

    def test_get_status_current_plan(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test Plan",
            "steps": ["A", "B"]
        })

        result = executors["getPlanStatus"]({})

        assert result["plan_id"] == create_result["plan_id"]
        assert result["title"] == "Test Plan"
        assert len(result["steps"]) == 2
        assert "progress" in result

    def test_get_status_by_id(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })

        result = executors["getPlanStatus"]({
            "plan_id": create_result["plan_id"]
        })

        assert result["plan_id"] == create_result["plan_id"]

    def test_get_status_no_plan(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["getPlanStatus"]({})

        assert "error" in result


class TestCompletePlanExecutor:
    """Tests for completePlan executor."""

    def test_complete_plan_success(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })
        executors["startPlan"]({})  # Must start before completing

        result = executors["completePlan"]({
            "status": "completed",
            "summary": "All done"
        })

        assert result["status"] == "completed"
        assert result["summary"] == "All done"
        assert result["completed_at"] is not None

    def test_complete_plan_failed(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })
        executors["startPlan"]({})  # Must start before failing

        result = executors["completePlan"]({
            "status": "failed",
            "summary": "Something broke"
        })

        assert result["status"] == "failed"

    def test_complete_plan_cancelled(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })

        result = executors["completePlan"]({
            "status": "cancelled"
        })

        assert result["status"] == "cancelled"

    def test_complete_plan_clears_current(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })

        # Plugin now uses dict mapping agent_name -> plan_id for multi-agent support
        # With no session set, _get_agent_name() defaults to "main"
        assert plugin._current_plan_ids.get("main") is not None

        executors["startPlan"]({})  # Must start before completing
        executors["completePlan"]({"status": "completed"})

        assert plugin._current_plan_ids.get("main") is None

    def test_complete_plan_no_plan(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["completePlan"]({
            "status": "completed"
        })

        assert "error" in result

    def test_complete_plan_invalid_status(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })

        result = executors["completePlan"]({
            "status": "invalid"
        })

        assert "error" in result

    def test_complete_plan_reports_completion(self):
        plugin = TodoPlugin()
        plugin.initialize()

        mock_reporter = Mock()
        plugin._reporter = mock_reporter

        executors = plugin.get_executors()
        executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })
        executors["startPlan"]({})  # Must start before completing

        mock_reporter.reset_mock()
        executors["completePlan"]({"status": "completed"})

        mock_reporter.report_plan_completed.assert_called_once()


class TestTodoPluginProgrammaticAPI:
    """Tests for programmatic API methods."""

    def test_create_plan_method(self):
        plugin = TodoPlugin()
        plugin.initialize()

        plan = plugin.create_plan(
            title="Test",
            steps=["A", "B"],
            context={"user": "test"}
        )

        assert plan.title == "Test"
        assert len(plan.steps) == 2
        assert plan.context["user"] == "test"

    def test_set_step_status_method(self):
        plugin = TodoPlugin()
        plugin.initialize()

        plan = plugin.create_plan("Test", ["A"])
        step_id = plan.steps[0].step_id

        step = plugin.set_step_status(step_id, StepStatus.COMPLETED, result="Done")

        assert step is not None
        assert step.status == StepStatus.COMPLETED
        assert step.result == "Done"

    def test_set_step_status_no_plan(self):
        plugin = TodoPlugin()
        plugin.initialize()

        step = plugin.set_step_status("some-id", StepStatus.COMPLETED)

        assert step is None

    def test_get_current_plan_method(self):
        plugin = TodoPlugin()
        plugin.initialize()

        plugin.create_plan("Test", ["A"])
        plan = plugin.get_current_plan()

        assert plan is not None
        assert plan.title == "Test"

    def test_get_all_plans_method(self):
        plugin = TodoPlugin()
        plugin.initialize()

        plugin.create_plan("Plan 1", ["A"])
        plugin.create_plan("Plan 2", ["B"])

        plans = plugin.get_all_plans()

        assert len(plans) >= 2


class TestTodoPluginWithConfig:
    """Tests for plugin configuration."""

    def test_initialize_from_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "version": "1.0",
                "reporter": {
                    "type": "console"
                },
                "storage": {
                    "type": "memory"
                }
            }
            json.dump(config, f)
            f.flush()

            try:
                plugin = TodoPlugin()
                plugin.initialize({"config_path": f.name})
                assert plugin._initialized is True
            finally:
                os.unlink(f.name)


class TestTodoPluginWorkflow:
    """Tests for complete plan workflows."""

    def test_full_workflow(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        # Create plan
        create_result = executors["createPlan"]({
            "title": "Deploy Feature",
            "steps": ["Run tests", "Build", "Deploy", "Verify"]
        })

        assert create_result["progress"]["total"] == 4
        assert create_result["progress"]["pending"] == 4

        # Start the plan (required before setStepStatus)
        executors["startPlan"]({})

        # Start first step
        step1_id = create_result["steps"][0]["step_id"]
        executors["setStepStatus"]({
            "step_id": step1_id,
            "status": "in_progress"
        })

        # Complete first step
        executors["setStepStatus"]({
            "step_id": step1_id,
            "status": "completed",
            "result": "All tests passed"
        })

        # Check progress
        status = executors["getPlanStatus"]({})
        assert status["progress"]["completed"] == 1
        assert status["progress"]["pending"] == 3

        # Complete remaining steps
        for i in range(1, 4):
            step_id = create_result["steps"][i]["step_id"]
            executors["setStepStatus"]({
                "step_id": step_id,
                "status": "completed"
            })

        # Complete plan
        result = executors["completePlan"]({
            "status": "completed",
            "summary": "Feature deployed successfully"
        })

        assert result["status"] == "completed"
        assert result["progress"]["percent"] == 100.0

    def test_workflow_with_failure(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Risky Task",
            "steps": ["Step 1", "Step 2"]
        })

        # Start the plan (required before setStepStatus)
        executors["startPlan"]({})

        # Complete first step
        executors["setStepStatus"]({
            "step_id": create_result["steps"][0]["step_id"],
            "status": "completed"
        })

        # Fail second step
        executors["setStepStatus"]({
            "step_id": create_result["steps"][1]["step_id"],
            "status": "failed",
            "error": "Network timeout"
        })

        # Fail plan
        result = executors["completePlan"]({
            "status": "failed",
            "summary": "Task failed due to network issues"
        })

        assert result["status"] == "failed"
        assert result["progress"]["completed"] == 1
        assert result["progress"]["failed"] == 1


class TestPersistence:
    """Tests for plugin state persistence."""

    def test_get_persistence_state_empty(self):
        """Test get_persistence_state with no plans."""
        plugin = TodoPlugin()
        plugin.initialize({"storage_type": "memory"})

        state = plugin.get_persistence_state()
        assert state == {"agent_plan_ids": {}}

    def test_get_persistence_state_with_plans(self):
        """Test get_persistence_state with active plans."""
        plugin = TodoPlugin()
        plugin.initialize({"storage_type": "memory"})

        # Create a plan
        executors = plugin.get_executors()
        result = executors["createPlan"]({
            "title": "Test Plan",
            "steps": ["Step 1", "Step 2"]
        })
        plan_id = result["plan_id"]

        state = plugin.get_persistence_state()
        # None key becomes "__none__" in serialization
        assert "__none__" in state["agent_plan_ids"] or "main" in state["agent_plan_ids"]

    def test_restore_persistence_state(self):
        """Test restoring plugin state."""
        plugin = TodoPlugin()
        plugin.initialize({"storage_type": "memory"})

        # Create a plan
        executors = plugin.get_executors()
        result = executors["createPlan"]({
            "title": "Test Plan",
            "steps": ["Step 1", "Step 2"]
        })
        plan_id = result["plan_id"]

        # Get state and clear plugin
        state = plugin.get_persistence_state()
        plugin._current_plan_ids.clear()

        # Restore
        plugin.restore_persistence_state(state)

        # Verify plan mapping restored
        assert len(plugin._current_plan_ids) > 0

    def test_persistence_state_roundtrip_json(self):
        """Test that persistence state survives JSON serialization."""
        plugin = TodoPlugin()
        plugin.initialize({"storage_type": "memory"})

        executors = plugin.get_executors()
        executors["createPlan"]({
            "title": "Test Plan",
            "steps": ["Step 1"]
        })

        # Serialize and deserialize
        state = plugin.get_persistence_state()
        json_str = json.dumps(state)
        restored_state = json.loads(json_str)

        # Clear and restore
        plugin._current_plan_ids.clear()
        plugin.restore_persistence_state(restored_state)

        # Should have restored the plan mapping
        assert len(plugin._current_plan_ids) > 0


class TestGetTaskEventsExecutor:
    """Tests for getTaskEvents executor with long-poll support."""

    def setup_method(self):
        """Reset event bus singleton before each test."""
        TaskEventBus.reset()

    def teardown_method(self):
        """Clean up after each test."""
        TaskEventBus.reset()

    def _make_plugin(self) -> TodoPlugin:
        """Create an initialized plugin with event bus wired up."""
        plugin = TodoPlugin()
        plugin.initialize()
        plugin._event_bus = get_event_bus()
        return plugin

    def _publish_event(self, agent_id: str = "sub") -> TaskEvent:
        """Publish a dummy event and return it."""
        bus = get_event_bus()
        plan = TodoPlan.create("P", ["S"])
        event = TaskEvent.create(TaskEventType.PLAN_CREATED, agent_id, plan)
        bus.publish(event)
        return event

    def test_basic_get_events(self):
        """getTaskEvents returns events without timeout."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()
        event = self._publish_event()

        result = executors["getTaskEvents"]({})

        assert result["count"] == 1
        assert result["events"][0]["event_id"] == event.event_id

    def test_after_event_filters_old_events(self):
        """after_event parameter excludes events up to and including the cursor."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        event1 = self._publish_event("a")
        event2 = self._publish_event("b")

        result = executors["getTaskEvents"]({"after_event": event1.event_id})

        assert result["count"] == 1
        assert result["events"][0]["event_id"] == event2.event_id

    def test_last_event_id_in_result(self):
        """Result includes last_event_id for cursor-based consumption."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        event1 = self._publish_event()
        event2 = self._publish_event()

        result = executors["getTaskEvents"]({})

        assert result["last_event_id"] == event2.event_id

    def test_no_last_event_id_when_empty(self):
        """Result has no last_event_id key when there are no events."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        result = executors["getTaskEvents"]({})

        assert result["count"] == 0
        assert "last_event_id" not in result

    def test_timeout_returns_immediately_with_events(self):
        """timeout does not delay when events already exist."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()
        self._publish_event("sub")

        start = time.monotonic()
        result = executors["getTaskEvents"]({
            "timeout": 5,
            "agent_id": "sub",
        })
        elapsed = time.monotonic() - start

        assert result["count"] == 1
        assert elapsed < 1.0

    def test_timeout_blocks_until_event(self):
        """timeout blocks and returns when a new event is published."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        result_holder: list = []

        def call_tool():
            r = executors["getTaskEvents"]({
                "timeout": 10,
                "agent_id": "sub",
            })
            result_holder.append(r)

        t = threading.Thread(target=call_tool)
        t.start()

        # Give the tool time to enter the wait.
        time.sleep(0.3)

        event = self._publish_event("sub")
        t.join(timeout=5)

        assert len(result_holder) == 1
        assert result_holder[0]["count"] == 1
        assert result_holder[0]["events"][0]["event_id"] == event.event_id

    def test_timeout_returns_empty(self):
        """timeout returns empty after timeout with no events."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        start = time.monotonic()
        result = executors["getTaskEvents"]({
            "timeout": 0.3,
            "agent_id": "nobody",
        })
        elapsed = time.monotonic() - start

        assert result["count"] == 0
        assert elapsed >= 0.25

    def test_timeout_with_after_event(self):
        """timeout combined with after_event for incremental consumption."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        event1 = self._publish_event()

        result_holder: list = []

        def call_tool():
            r = executors["getTaskEvents"]({
                "timeout": 10,
                "after_event": event1.event_id,
            })
            result_holder.append(r)

        t = threading.Thread(target=call_tool)
        t.start()
        time.sleep(0.3)

        event2 = self._publish_event()
        t.join(timeout=5)

        assert len(result_holder) == 1
        assert result_holder[0]["count"] == 1
        assert result_holder[0]["events"][0]["event_id"] == event2.event_id

    def test_timeout_clamped(self):
        """timeout is clamped to [0, 30]."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        # Negative → treated as 0 (immediate return)
        start = time.monotonic()
        result = executors["getTaskEvents"]({"timeout": -5})
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

        # Invalid type → treated as 0
        result = executors["getTaskEvents"]({"timeout": "not_a_number"})
        assert result["count"] == 0

    def test_timeout_requires_narrowing(self):
        """timeout without any filters or cursor returns an error."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        result = executors["getTaskEvents"]({"timeout": 5})
        assert "error" in result
        assert "timeout requires" in result["error"]

    def test_timeout_accepted_with_agent_id(self):
        """timeout is accepted when agent_id narrows the query."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        # Should not error — agent_id provides narrowing.
        start = time.monotonic()
        result = executors["getTaskEvents"]({
            "timeout": 0.2,
            "agent_id": "sub",
        })
        elapsed = time.monotonic() - start

        assert "error" not in result
        # Should have waited (no events from "sub").
        assert elapsed >= 0.15

    def test_timeout_accepted_with_event_types(self):
        """timeout is accepted when event_types narrows the query."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()

        start = time.monotonic()
        result = executors["getTaskEvents"]({
            "timeout": 0.2,
            "event_types": ["step_completed"],
        })
        elapsed = time.monotonic() - start

        assert "error" not in result
        assert elapsed >= 0.15

    def test_timeout_accepted_with_after_event(self):
        """timeout is accepted when after_event provides a cursor."""
        plugin = self._make_plugin()
        executors = plugin.get_executors()
        event = self._publish_event()

        start = time.monotonic()
        result = executors["getTaskEvents"]({
            "timeout": 0.2,
            "after_event": event.event_id,
        })
        elapsed = time.monotonic() - start

        assert "error" not in result
        assert elapsed >= 0.15
