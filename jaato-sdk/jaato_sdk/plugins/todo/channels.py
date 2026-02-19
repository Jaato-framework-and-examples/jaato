"""TodoReporter abstract base class and generic callback-based reporter.

This module contains the ABC for progress reporters and a generic
callback-based implementation (LivePlanReporter) used by both the server
and TUI. Concrete transport-specific implementations (Console, Webhook,
File, Memory, Multi) live in the server-side ``shared.plugins.todo.channels``
module.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from .models import TodoPlan, TodoStep, StepStatus


class TodoReporter(ABC):
    """Base class for progress reporters.

    Reporters handle different transport protocols for reporting
    plan progress to external systems or users.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this reporter type."""
        ...

    @abstractmethod
    def report_plan_created(self, plan: TodoPlan, agent_id: Optional[str] = None) -> None:
        """Report that a new plan was created.

        Args:
            plan: The newly created plan.
            agent_id: Optional agent identifier for multi-agent tracking.
        """
        ...

    @abstractmethod
    def report_step_update(self, plan: TodoPlan, step: TodoStep, agent_id: Optional[str] = None) -> None:
        """Report that a step's status changed.

        Args:
            plan: The plan containing the step.
            step: The step that was updated.
            agent_id: Optional agent identifier for multi-agent tracking.
        """
        ...

    @abstractmethod
    def report_plan_completed(self, plan: TodoPlan, agent_id: Optional[str] = None) -> None:
        """Report that a plan was completed/failed/cancelled.

        Args:
            plan: The completed plan.
            agent_id: Optional agent identifier for multi-agent tracking.
        """
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the reporter with optional configuration."""
        pass

    def shutdown(self) -> None:
        """Clean up any resources used by the reporter."""
        pass


class LivePlanReporter(TodoReporter):
    """Callback-based TodoReporter for live plan updates.

    Converts plan lifecycle events into callback invocations. Used by both
    the server (to emit events to clients) and the TUI (to update the
    sticky plan panel).

    Callbacks:
        update_callback(plan_data: Dict, agent_id: Optional[str]) — called
            with a display-friendly dict whenever the plan changes.
        clear_callback(agent_id: Optional[str]) — called to clear the panel.
        output_callback(source: str, text: str, mode: str) — called for
            supplementary text output (completion messages, errors).
    """

    def __init__(self):
        self._update_callback: Optional[Callable[[Dict[str, Any], Optional[str]], None]] = None
        self._clear_callback: Optional[Callable[[Optional[str]], None]] = None
        self._output_callback: Optional[Callable[[str, str, str], None]] = None

    @property
    def name(self) -> str:
        return "live_panel"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with callbacks.

        Config keys:
            update_callback: Callable[[Dict, Optional[str]], None]
            clear_callback: Callable[[Optional[str]], None]
            output_callback: Callable[[str, str, str], None]
        """
        if config:
            self._update_callback = config.get("update_callback")
            self._clear_callback = config.get("clear_callback")
            self._output_callback = config.get("output_callback")

    def _emit_plan_update(self, plan: TodoPlan, agent_id: Optional[str] = None) -> None:
        """Emit plan update to the display."""
        if self._update_callback:
            plan_data = self._plan_to_display_dict(plan)
            self._update_callback(plan_data, agent_id)

    def _plan_to_display_dict(self, plan: TodoPlan) -> Dict[str, Any]:
        """Convert TodoPlan to display-friendly dict."""
        steps = []
        for step in plan.steps:
            step_data = {
                "step_id": step.step_id,
                "sequence": step.sequence,
                "description": step.description,
                "status": step.status.value,
                "result": step.result,
                "error": step.error,
            }
            if step.blocked_by:
                step_data["blocked_by"] = [
                    ref.to_dict() for ref in step.blocked_by
                ]
            if step.depends_on:
                step_data["depends_on"] = [
                    ref.to_dict() for ref in step.depends_on
                ]
            if step.received_outputs:
                step_data["received_outputs"] = step.received_outputs
            steps.append(step_data)

        return {
            "plan_id": plan.plan_id,
            "title": plan.title,
            "status": plan.status.value,
            "started": plan.started,
            "steps": steps,
            "progress": plan.get_progress(),
            "summary": plan.summary,
        }

    def _emit_output(self, source: str, text: str, mode: str = "write") -> None:
        """Emit supplementary output to the scrolling panel."""
        if self._output_callback:
            self._output_callback(source, text, mode)

    def report_plan_created(self, plan: TodoPlan, agent_id: Optional[str] = None) -> None:
        """Report new plan creation - update the sticky panel."""
        self._emit_plan_update(plan, agent_id)
        self._emit_output("plan", f"Plan created: {plan.title}", "write")

    def report_step_update(self, plan: TodoPlan, step: TodoStep, agent_id: Optional[str] = None) -> None:
        """Report step status change - update the sticky panel."""
        self._emit_plan_update(plan, agent_id)

        if step.status == StepStatus.COMPLETED and step.result:
            self._emit_output(
                "plan",
                f"[{step.sequence}] {step.description}: {step.result}",
                "write"
            )
        elif step.status == StepStatus.FAILED and step.error:
            self._emit_output(
                "plan",
                f"[{step.sequence}] FAILED: {step.error}",
                "write"
            )

    def report_plan_completed(self, plan: TodoPlan, agent_id: Optional[str] = None) -> None:
        """Report plan completion - update panel and emit summary."""
        self._emit_plan_update(plan, agent_id)

        progress = plan.get_progress()
        status_emoji = {
            "completed": "\u2705",
            "failed": "\u274c",
            "cancelled": "\u26a0\ufe0f",
        }.get(plan.status.value, "\U0001f4cb")

        summary = (
            f"{status_emoji} Plan {plan.status.value}: {plan.title} "
            f"({progress['completed']}/{progress['total']} completed"
        )
        if progress['failed'] > 0:
            summary += f", {progress['failed']} failed"
        summary += ")"

        self._emit_output("plan", summary, "write")

        if plan.summary:
            self._emit_output("plan", f"Summary: {plan.summary}", "write")

    def shutdown(self) -> None:
        """Clean up - don't auto-clear to let user see final state."""
        pass


def create_live_reporter(
    update_callback: Callable[[Dict[str, Any], Optional[str]], None],
    clear_callback: Optional[Callable[[Optional[str]], None]] = None,
    output_callback: Optional[Callable[[str, str, str], None]] = None,
) -> LivePlanReporter:
    """Factory to create a configured LivePlanReporter.

    Args:
        update_callback: Called with (plan_data, agent_id) to update the panel.
        clear_callback: Called with agent_id to clear the panel.
        output_callback: Called with (source, text, mode) for output.

    Returns:
        Configured LivePlanReporter instance.
    """
    reporter = LivePlanReporter()
    reporter.initialize({
        "update_callback": update_callback,
        "clear_callback": clear_callback,
        "output_callback": output_callback,
    })
    return reporter
