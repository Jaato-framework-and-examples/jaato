"""Tests for the TODO plugin integration."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from ..plugin import TodoPlugin, create_plugin
from ..models import StepStatus, PlanStatus


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
        assert plugin._storage is None
        assert plugin._reporter is None


class TestTodoPluginFunctionDeclarations:
    """Tests for function declarations."""

    def test_get_function_declarations(self):
        plugin = TodoPlugin()
        declarations = plugin.get_function_declarations()

        assert len(declarations) == 4
        names = {d.name for d in declarations}
        assert "createPlan" in names
        assert "updateStep" in names
        assert "getPlanStatus" in names
        assert "completePlan" in names

    def test_createPlan_schema(self):
        plugin = TodoPlugin()
        declarations = plugin.get_function_declarations()
        create_plan = next(d for d in declarations if d.name == "createPlan")
        schema = create_plan.parameters_json_schema

        assert schema["type"] == "object"
        assert "title" in schema["properties"]
        assert "steps" in schema["properties"]
        assert "title" in schema["required"]
        assert "steps" in schema["required"]

    def test_updateStep_schema(self):
        plugin = TodoPlugin()
        declarations = plugin.get_function_declarations()
        update_step = next(d for d in declarations if d.name == "updateStep")
        schema = update_step.parameters_json_schema

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
        assert "updateStep" in executors
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

        assert plugin._current_plan_id == result["plan_id"]

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


class TestUpdateStepExecutor:
    """Tests for updateStep executor."""

    def test_update_step_to_in_progress(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        # Create plan first
        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1", "Step 2"]
        })
        step_id = create_result["steps"][0]["step_id"]

        # Update step
        result = executors["updateStep"]({
            "step_id": step_id,
            "status": "in_progress"
        })

        assert result["status"] == "in_progress"
        assert result["step_id"] == step_id

    def test_update_step_to_completed(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        step_id = create_result["steps"][0]["step_id"]

        result = executors["updateStep"]({
            "step_id": step_id,
            "status": "completed",
            "result": "Done successfully"
        })

        assert result["status"] == "completed"
        assert result["result"] == "Done successfully"

    def test_update_step_to_failed(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        step_id = create_result["steps"][0]["step_id"]

        result = executors["updateStep"]({
            "step_id": step_id,
            "status": "failed",
            "error": "Something went wrong"
        })

        assert result["status"] == "failed"
        assert result["error"] == "Something went wrong"

    def test_update_step_to_skipped(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        step_id = create_result["steps"][0]["step_id"]

        result = executors["updateStep"]({
            "step_id": step_id,
            "status": "skipped",
            "result": "Not needed"
        })

        assert result["status"] == "skipped"

    def test_update_step_no_plan(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        result = executors["updateStep"]({
            "step_id": "some-id",
            "status": "completed"
        })

        assert "error" in result
        assert "No active plan" in result["error"]

    def test_update_step_not_found(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })

        result = executors["updateStep"]({
            "step_id": "nonexistent",
            "status": "completed"
        })

        assert "error" in result
        assert "Step not found" in result["error"]

    def test_update_step_invalid_status(self):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["Step 1"]
        })
        step_id = create_result["steps"][0]["step_id"]

        result = executors["updateStep"]({
            "step_id": step_id,
            "status": "invalid_status"
        })

        assert "error" in result

    def test_update_step_reports_update(self):
        plugin = TodoPlugin()
        plugin.initialize()

        mock_reporter = Mock()
        plugin._reporter = mock_reporter

        executors = plugin.get_executors()
        create_result = executors["createPlan"]({
            "title": "Test",
            "steps": ["A"]
        })
        step_id = create_result["steps"][0]["step_id"]

        mock_reporter.reset_mock()
        executors["updateStep"]({
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

        assert plugin._current_plan_id is not None

        executors["completePlan"]({"status": "completed"})

        assert plugin._current_plan_id is None

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

    def test_update_step_method(self):
        plugin = TodoPlugin()
        plugin.initialize()

        plan = plugin.create_plan("Test", ["A"])
        step_id = plan.steps[0].step_id

        step = plugin.update_step(step_id, StepStatus.COMPLETED, result="Done")

        assert step is not None
        assert step.status == StepStatus.COMPLETED
        assert step.result == "Done"

    def test_update_step_no_plan(self):
        plugin = TodoPlugin()
        plugin.initialize()

        step = plugin.update_step("some-id", StepStatus.COMPLETED)

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

        # Start first step
        step1_id = create_result["steps"][0]["step_id"]
        executors["updateStep"]({
            "step_id": step1_id,
            "status": "in_progress"
        })

        # Complete first step
        executors["updateStep"]({
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
            executors["updateStep"]({
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

        # Complete first step
        executors["updateStep"]({
            "step_id": create_result["steps"][0]["step_id"],
            "status": "completed"
        })

        # Fail second step
        executors["updateStep"]({
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
