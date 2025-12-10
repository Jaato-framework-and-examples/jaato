"""Plan reporter that updates the sticky plan panel.

A TodoReporter implementation that integrates with PTDisplay
to update the sticky plan panel in real-time.
"""

from typing import Any, Callable, Dict, Optional

import sys
import pathlib

# Add project root to path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.plugins.todo.channels import TodoReporter
from shared.plugins.todo.models import TodoPlan, TodoStep, StepStatus


class LivePlanReporter(TodoReporter):
    """Reporter that updates the display's sticky plan panel.

    Instead of printing to console, this reporter calls back to the
    display to update the sticky plan panel in-place.
    """

    def __init__(self):
        self._update_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._clear_callback: Optional[Callable[[], None]] = None
        self._output_callback: Optional[Callable[[str, str, str], None]] = None

    @property
    def name(self) -> str:
        return "live_panel"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with callbacks to display.

        Config options:
            update_callback: Callable[[Dict], None] - called with plan data to update panel
            clear_callback: Callable[[], None] - called to clear the panel
            output_callback: Callable[[str, str, str], None] - for supplementary output
        """
        if config:
            self._update_callback = config.get("update_callback")
            self._clear_callback = config.get("clear_callback")
            self._output_callback = config.get("output_callback")

    def _emit_plan_update(self, plan: TodoPlan) -> None:
        """Emit plan update to the display."""
        if self._update_callback:
            plan_data = self._plan_to_display_dict(plan)
            self._update_callback(plan_data)

    def _plan_to_display_dict(self, plan: TodoPlan) -> Dict[str, Any]:
        """Convert TodoPlan to display-friendly dict."""
        steps = []
        for step in plan.steps:
            steps.append({
                "sequence": step.sequence,
                "description": step.description,
                "status": step.status.value,
                "result": step.result,
                "error": step.error,
            })

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

    def report_plan_created(self, plan: TodoPlan) -> None:
        """Report new plan creation - update the sticky panel."""
        self._emit_plan_update(plan)
        # Also emit a message to the output panel
        self._emit_output("plan", f"Plan created: {plan.title}", "write")

    def report_step_update(self, plan: TodoPlan, step: TodoStep) -> None:
        """Report step status change - update the sticky panel."""
        self._emit_plan_update(plan)

        # Emit step details to output for completed/failed steps
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

    def report_plan_completed(self, plan: TodoPlan) -> None:
        """Report plan completion - update panel and emit summary."""
        self._emit_plan_update(plan)

        # Emit completion message to output
        progress = plan.get_progress()
        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "cancelled": "⚠️",
        }.get(plan.status.value, "📋")

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
        """Clean up - optionally clear the panel."""
        # Don't auto-clear on shutdown to let user see final state
        pass


def create_live_reporter(
    update_callback: Callable[[Dict[str, Any]], None],
    clear_callback: Optional[Callable[[], None]] = None,
    output_callback: Optional[Callable[[str, str, str], None]] = None,
) -> LivePlanReporter:
    """Factory to create a configured LivePlanReporter.

    Args:
        update_callback: Called with plan data dict to update the panel.
        clear_callback: Called to clear the panel.
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
