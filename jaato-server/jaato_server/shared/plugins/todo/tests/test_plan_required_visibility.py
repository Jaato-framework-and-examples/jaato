"""Tests for ``TodoPlugin.is_tool_visible`` — plan-required tool gating.

Pin the visibility-filter behavior added 2026-06-07 to close the
class of bug surfaced during the vLLM Llama-3.1-8B-AWQ smoke run:

- Persona instructs the model to call ``signal_completion``.
- 30 tools are exposed including all 8 plan-required todo tools
  (``startPlan``, ``setStepStatus``, ..., ``completeStepWithOutput``).
- Each plan-required tool's runtime check returns
  ``{"error": "No active plan..."}`` — but the model has already
  picked one because the description of
  ``completeStepWithOutput`` ("Complete a step AND pass structured
  data") overlaps semantically with the persona's "call
  signal_completion with payload".
- Model loops trying plan-required tools, each fails with the
  "No active plan" rejection.

The fix is to NOT expose the 8 plan-required tools to the model
when no plan exists.  Same shape as
``LifecycleTools._should_hide_signal_completion`` — gate at the
schema-export layer, not the call layer.

These tests pin the ``is_tool_visible`` contract; the
session-side integration (per-turn filter at
``_get_tools_for_provider``) is tested separately in
``shared/tests/test_jaato_session.py``.
"""

import pytest

from ..plugin import PLAN_REQUIRED_TOOLS, TodoPlugin


class TestPlanRequiredToolsConstant:
    """Pin: the 8 documented plan-required tool names are in the
    frozenset; ``createPlan`` (the entry point) is not."""

    def test_constant_is_frozenset(self):
        assert isinstance(PLAN_REQUIRED_TOOLS, frozenset)

    def test_eight_documented_tools_present(self):
        documented = {
            "startPlan",
            "setStepStatus",
            "getPlanStatus",
            "completePlan",
            "addStep",
            "addDependentStep",
            "completeStepWithOutput",
            "getBlockedSteps",
        }
        assert documented <= PLAN_REQUIRED_TOOLS

    def test_user_command_alias_plan_is_present(self):
        """``plan`` is the user-command alias for ``getPlanStatus``
        (see ``get_executors`` mapping in plugin.py)."""
        assert "plan" in PLAN_REQUIRED_TOOLS

    def test_createPlan_not_in_set(self):
        """``createPlan`` is the entry point — it cannot be
        plan-required (you need it to make a plan exist).  Must NOT
        appear in the gated set."""
        assert "createPlan" not in PLAN_REQUIRED_TOOLS


class TestIsToolVisibleNoPlan:
    """Pin: when no plan exists, the 8 (+1 alias) plan-required
    tools are hidden; everything else stays visible."""

    def setup_method(self):
        self.plugin = TodoPlugin()
        self.plugin.initialize()
        # Confirm precondition: no plan has been created yet.
        assert self.plugin._get_current_plan() is None

    def test_createPlan_visible_when_no_plan(self):
        assert self.plugin.is_tool_visible("createPlan") is True

    def test_startPlan_hidden_when_no_plan(self):
        assert self.plugin.is_tool_visible("startPlan") is False

    def test_setStepStatus_hidden_when_no_plan(self):
        assert self.plugin.is_tool_visible("setStepStatus") is False

    def test_getPlanStatus_hidden_when_no_plan(self):
        assert self.plugin.is_tool_visible("getPlanStatus") is False

    def test_completePlan_hidden_when_no_plan(self):
        assert self.plugin.is_tool_visible("completePlan") is False

    def test_addStep_hidden_when_no_plan(self):
        assert self.plugin.is_tool_visible("addStep") is False

    def test_addDependentStep_hidden_when_no_plan(self):
        assert self.plugin.is_tool_visible("addDependentStep") is False

    def test_completeStepWithOutput_hidden_when_no_plan(self):
        """Smoking-gun case: this was the tool the vLLM smoke
        misrouted to."""
        assert self.plugin.is_tool_visible("completeStepWithOutput") is False

    def test_getBlockedSteps_hidden_when_no_plan(self):
        assert self.plugin.is_tool_visible("getBlockedSteps") is False

    def test_plan_alias_hidden_when_no_plan(self):
        assert self.plugin.is_tool_visible("plan") is False

    def test_unrelated_tool_visible_regardless(self):
        """Predicate has no opinion on tools owned by other plugins."""
        assert self.plugin.is_tool_visible("cli_based_tool") is True
        assert self.plugin.is_tool_visible("signal_completion") is True
        assert self.plugin.is_tool_visible("readFile") is True


class TestIsToolVisibleWithPlan:
    """Pin: when a plan IS active, all plan-required tools become
    visible.  ``createPlan`` remains visible (always visible)."""

    def setup_method(self):
        self.plugin = TodoPlugin()
        self.plugin.initialize()
        executors = self.plugin.get_executors()
        # Bring the plugin into "plan exists" state via the public
        # createPlan executor — same path the model would take.
        result = executors["createPlan"]({
            "title": "Visibility-gating test plan",
            "steps": ["Step 1", "Step 2"],
        })
        assert "plan_id" in result, f"createPlan failed: {result}"
        # Confirm precondition: a plan now exists.
        assert self.plugin._get_current_plan() is not None

    def test_createPlan_still_visible(self):
        assert self.plugin.is_tool_visible("createPlan") is True

    def test_all_plan_required_tools_visible(self):
        """When a plan exists, every member of PLAN_REQUIRED_TOOLS
        is visible."""
        for tool_name in PLAN_REQUIRED_TOOLS:
            assert self.plugin.is_tool_visible(tool_name) is True, (
                f"{tool_name!r} should be visible with an active plan"
            )

    def test_unrelated_tool_still_visible(self):
        assert self.plugin.is_tool_visible("cli_based_tool") is True


class TestIsToolVisibleFailsOpenOnLookupError:
    """Pin: a buggy ``_get_current_plan`` must not break the turn.
    The predicate fails OPEN (treat tool as visible) so the runtime
    check inside ``_execute_*`` still surfaces a clean error if the
    plan really is missing.
    """

    def test_get_current_plan_raises_returns_true(self, monkeypatch):
        plugin = TodoPlugin()
        plugin.initialize()

        def boom():
            raise RuntimeError("storage layer exploded")

        monkeypatch.setattr(plugin, "_get_current_plan", boom)
        # Plan-required tool name — would normally be False without
        # a plan, but the lookup error fails-open to True.
        assert plugin.is_tool_visible("setStepStatus") is True
        # Non-plan-required tools take the fast path and never
        # touch _get_current_plan — also True.
        assert plugin.is_tool_visible("createPlan") is True
        assert plugin.is_tool_visible("cli_based_tool") is True
