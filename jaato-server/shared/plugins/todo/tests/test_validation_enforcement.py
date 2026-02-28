"""Tests for validation enforcement on plan steps.

Validation steps (auto-detected from description patterns or explicitly marked)
cannot be completed via setStepStatus unless they have received_outputs from a
subagent.  This prevents models from bypassing delegated validation by manually
marking validation steps as completed.
"""

import pytest

from ..plugin import TodoPlugin, _is_validation_step
from jaato_sdk.plugins.todo.models import (
    StepStatus, TodoStep, TodoPlan, TaskRef,
)


# ---------------------------------------------------------------------------
# _is_validation_step pattern detection
# ---------------------------------------------------------------------------


class TestIsValidationStep:
    """Tests for the _is_validation_step helper."""

    def test_tier_patterns(self):
        assert _is_validation_step("Execute Tier 1 validation")
        assert _is_validation_step("Run tier-2 checks")
        assert _is_validation_step("Tier 3: pattern compliance")
        assert _is_validation_step("Execute tier-4 CI/CD validation")
        assert _is_validation_step("tier 1 universal check")

    def test_validate_patterns(self):
        assert _is_validation_step("Validate output against standards")
        assert _is_validation_step("validate schema correctness")
        assert _is_validation_step("Validate the generated code")

    def test_verification_patterns(self):
        assert _is_validation_step("Run verification checks")
        assert _is_validation_step("Verification of generated output")

    def test_run_validation_patterns(self):
        assert _is_validation_step("Run validation checks")
        assert _is_validation_step("run validation suite")

    def test_execute_validation_patterns(self):
        assert _is_validation_step("Execute validation suite")
        assert _is_validation_step("execute validation on output")

    def test_non_validation_steps(self):
        """Steps that should NOT be detected as validation."""
        assert not _is_validation_step("Create a valid configuration")
        assert not _is_validation_step("Build the project")
        assert not _is_validation_step("Deploy to staging")
        assert not _is_validation_step("Run tests")
        assert not _is_validation_step("Write code for the feature")
        assert not _is_validation_step("Analyze requirements")
        assert not _is_validation_step("Check configuration is valid")

    def test_empty_and_edge_cases(self):
        assert not _is_validation_step("")
        assert not _is_validation_step("   ")
        assert not _is_validation_step("tier")  # No digit after tier
        assert not _is_validation_step("a valid tier")  # "valid" has no word boundary match


# ---------------------------------------------------------------------------
# TodoStep.validation_required field
# ---------------------------------------------------------------------------


class TestTodoStepValidationRequired:
    """Tests for the validation_required field on TodoStep."""

    def test_default_is_false(self):
        step = TodoStep.create(1, "Build the project")
        assert step.validation_required is False

    def test_can_be_set_explicitly(self):
        step = TodoStep.create(1, "Custom check")
        step.validation_required = True
        assert step.validation_required is True

    def test_serialization_when_false(self):
        """validation_required=False should be omitted from dict."""
        step = TodoStep.create(1, "Build")
        d = step.to_dict()
        assert "validation_required" not in d

    def test_serialization_when_true(self):
        """validation_required=True should be included in dict."""
        step = TodoStep.create(1, "Validate output")
        step.validation_required = True
        d = step.to_dict()
        assert d["validation_required"] is True

    def test_deserialization_with_field(self):
        step = TodoStep.create(1, "Validate output")
        step.validation_required = True
        d = step.to_dict()
        restored = TodoStep.from_dict(d)
        assert restored.validation_required is True

    def test_deserialization_without_field(self):
        """Old serialized data without the field should default to False."""
        d = {
            "step_id": "test-id",
            "sequence": 1,
            "description": "Old step",
            "status": "pending",
        }
        step = TodoStep.from_dict(d)
        assert step.validation_required is False


# ---------------------------------------------------------------------------
# createPlan auto-detection
# ---------------------------------------------------------------------------


class TestCreatePlanAutoDetection:
    """Tests that createPlan auto-detects validation steps."""

    def _create_plan(self, steps):
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()
        return executors["createPlan"]({"title": "Test", "steps": steps})

    def test_validation_steps_are_flagged(self):
        result = self._create_plan([
            "Implement feature",
            "Execute Tier 1 validation",
            "Execute Tier 2 validation: Java/Spring",
            "Deploy",
        ])

        assert "error" not in result
        steps = result["steps"]
        assert len(steps) == 4

        # Non-validation steps should not have the flag
        assert "validation_required" not in steps[0]
        assert "validation_required" not in steps[3]

        # Validation steps should be flagged
        assert steps[1].get("validation_required") is True
        assert steps[2].get("validation_required") is True

    def test_non_validation_plan(self):
        result = self._create_plan([
            "Build project",
            "Run tests",
            "Deploy",
        ])

        assert "error" not in result
        for step in result["steps"]:
            assert "validation_required" not in step

    def test_validate_keyword_detected(self):
        result = self._create_plan([
            "Write code",
            "Validate output against standards",
        ])

        assert result["steps"][1].get("validation_required") is True


# ---------------------------------------------------------------------------
# setStepStatus hard gate enforcement
# ---------------------------------------------------------------------------


class TestValidationEnforcement:
    """Tests that validation steps cannot be completed without subagent evidence."""

    def _setup_plan_with_validation(self, validation_desc="Execute Tier 1 validation"):
        """Create a started plan with a validation step."""
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Test Plan",
            "steps": [
                "Implement feature",
                validation_desc,
                "Deploy",
            ],
        })

        executors["startPlan"]({})

        return plugin, executors, create_result

    def test_validation_step_blocks_completion_without_evidence(self):
        """Completing a validation step without received_outputs should be rejected."""
        plugin, executors, create_result = self._setup_plan_with_validation()

        validation_step = create_result["steps"][1]
        assert validation_step.get("validation_required") is True

        # Try to complete the validation step directly
        result = executors["setStepStatus"]({
            "step_id": validation_step["step_id"],
            "status": "completed",
            "result": "Looks good to me",
        })

        # Should be rejected
        assert "error" in result
        assert "validation_required" in result["error"]
        assert result.get("validation_required") is True
        assert result.get("has_received_outputs") is False

    def test_validation_step_allows_in_progress(self):
        """Setting a validation step to in_progress should always work."""
        plugin, executors, create_result = self._setup_plan_with_validation()

        validation_step = create_result["steps"][1]

        result = executors["setStepStatus"]({
            "step_id": validation_step["step_id"],
            "status": "in_progress",
        })

        # result dict includes "error": None (step.error field), so check value
        assert result.get("error") is None
        assert result["status"] == "in_progress"

    def test_validation_step_allows_failed(self):
        """Setting a validation step to failed should always work."""
        plugin, executors, create_result = self._setup_plan_with_validation()

        validation_step = create_result["steps"][1]

        result = executors["setStepStatus"]({
            "step_id": validation_step["step_id"],
            "status": "failed",
            "error": "Validator subagent did not return",
        })

        # Failed status should work — error field contains the step's error, not a rejection
        assert result["status"] == "failed"
        assert result["error"] == "Validator subagent did not return"

    def test_validation_step_allows_skipped(self):
        """Setting a validation step to skipped should always work."""
        plugin, executors, create_result = self._setup_plan_with_validation()

        validation_step = create_result["steps"][1]

        result = executors["setStepStatus"]({
            "step_id": validation_step["step_id"],
            "status": "skipped",
            "result": "Validation not needed for this change",
        })

        assert result.get("error") is None
        assert result["status"] == "skipped"

    def test_non_validation_step_completes_normally(self):
        """Non-validation steps should complete without restriction."""
        plugin, executors, create_result = self._setup_plan_with_validation()

        # First step is "Implement feature" — not a validation step
        impl_step = create_result["steps"][0]
        assert "validation_required" not in impl_step

        result = executors["setStepStatus"]({
            "step_id": impl_step["step_id"],
            "status": "completed",
            "result": "Feature implemented",
        })

        assert result.get("error") is None
        assert result["status"] == "completed"

    def test_validation_step_completes_with_received_outputs(self):
        """Validation step should complete once it has received_outputs."""
        plugin, executors, create_result = self._setup_plan_with_validation()

        validation_step_id = create_result["steps"][1]["step_id"]

        # Simulate subagent providing output by directly setting received_outputs
        # on the step (normally done by resolve_dependency via event bus)
        plan = plugin._get_current_plan()
        step = plan.get_step_by_id(validation_step_id)
        step.received_outputs["validator:final"] = {"passed": True, "errors": []}

        # Save updated plan
        if plugin._storage:
            plugin._storage.save_plan(plan)

        # Now completing should work
        result = executors["setStepStatus"]({
            "step_id": validation_step_id,
            "status": "completed",
            "result": "All validations passed",
        })

        assert result.get("error") is None
        assert result["status"] == "completed"

    def test_completeStepWithOutput_bypasses_gate(self):
        """completeStepWithOutput should work on validation steps.

        completeStepWithOutput IS the subagent evidence mechanism — a subagent
        calling it is proof that validation actually ran.
        """
        plugin, executors, create_result = self._setup_plan_with_validation()

        validation_step_id = create_result["steps"][1]["step_id"]

        # completeStepWithOutput should work even without received_outputs
        result = executors["completeStepWithOutput"]({
            "step_id": validation_step_id,
            "output": {"passed": True, "errors": []},
            "result": "All checks passed",
        })

        assert result.get("error") is None
        assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Integration with cross-agent dependency system
# ---------------------------------------------------------------------------


class TestValidationWithDependencies:
    """Tests that validation enforcement integrates with the dependency system."""

    def test_validation_step_with_resolved_dependency_completes(self):
        """After dependency resolves (providing received_outputs), completion works."""
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Validated Deploy",
            "steps": [
                "Implement code",
                "Validate implementation",
                "Deploy",
            ],
        })
        executors["startPlan"]({})

        validation_step_id = create_result["steps"][1]["step_id"]

        # Simulate what happens when a subagent completes:
        # The dependency resolution puts output in received_outputs
        plan = plugin._get_current_plan()
        step = plan.get_step_by_id(validation_step_id)

        # Add a dependency and then resolve it (mimicking event bus flow)
        dep_ref = TaskRef(agent_id="validator-tier1", step_id="final-check")
        step.add_dependency(dep_ref)
        step.resolve_dependency(
            dep_ref,
            output={"passed": True, "results": ["No issues found"]},
            provides_name="tier1_results",
        )

        if plugin._storage:
            plugin._storage.save_plan(plan)

        # Now the step has received_outputs — completion should work
        result = executors["setStepStatus"]({
            "step_id": validation_step_id,
            "status": "completed",
            "result": "Tier 1 validation passed",
        })

        assert result.get("error") is None
        assert result["status"] == "completed"

    def test_validation_step_with_unresolved_dependency_blocks(self):
        """Validation step with unresolved dependency still blocks completion."""
        plugin = TodoPlugin()
        plugin.initialize()
        executors = plugin.get_executors()

        create_result = executors["createPlan"]({
            "title": "Validated Deploy",
            "steps": [
                "Implement code",
                "Validate implementation",
                "Deploy",
            ],
        })
        executors["startPlan"]({})

        validation_step_id = create_result["steps"][1]["step_id"]

        # Add a dependency but DON'T resolve it
        plan = plugin._get_current_plan()
        step = plan.get_step_by_id(validation_step_id)
        dep_ref = TaskRef(agent_id="validator-tier1", step_id="final-check")
        step.add_dependency(dep_ref)

        if plugin._storage:
            plugin._storage.save_plan(plan)

        # Still no received_outputs — completion should be rejected
        result = executors["setStepStatus"]({
            "step_id": validation_step_id,
            "status": "completed",
        })

        assert "error" in result
        assert "validation_required" in result["error"]
