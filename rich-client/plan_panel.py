"""Plan panel rendering for the sticky plan display.

Renders the current plan status as a Rich Table/Panel for display
in the sticky header region of the TUI.
"""

from typing import Any, Dict, List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group


class PlanPanel:
    """Renders plan status as Rich renderables for the sticky panel."""

    # Status symbols with colors
    STATUS_SYMBOLS = {
        "pending": ("○", "dim"),
        "in_progress": ("◐", "blue"),
        "completed": ("●", "green"),
        "failed": ("✗", "red"),
        "skipped": ("⊘", "yellow"),
    }

    PLAN_STATUS_DISPLAY = {
        "pending": ("📋", "PENDING", "dim"),
        "in_progress": ("🔄", "IN PROGRESS", "blue"),
        "completed": ("✅", "COMPLETED", "green"),
        "failed": ("❌", "FAILED", "red"),
        "cancelled": ("⚠️", "CANCELLED", "yellow"),
    }

    def __init__(self):
        self._plan_data: Optional[Dict[str, Any]] = None

    def update_plan(self, plan_data: Dict[str, Any]) -> None:
        """Update the plan data to render.

        Args:
            plan_data: Plan status dict with title, status, steps, progress.
        """
        self._plan_data = plan_data

    def clear(self) -> None:
        """Clear the current plan."""
        self._plan_data = None

    @property
    def has_plan(self) -> bool:
        """Check if there's an active plan to display."""
        return self._plan_data is not None

    def render(self) -> Panel:
        """Render the plan panel.

        Returns:
            Rich Panel containing the plan status display.
        """
        if not self._plan_data:
            return self._render_no_plan()

        return self._render_plan()

    def _render_no_plan(self) -> Panel:
        """Render empty state when no plan is active."""
        content = Text("No active plan", style="dim italic")
        return Panel(
            content,
            title="[bold]Plan[/bold]",
            border_style="dim",
            height=3,
        )

    def _render_plan(self) -> Panel:
        """Render the current plan."""
        plan = self._plan_data
        title = plan.get("title", "Untitled Plan")
        status = plan.get("status", "pending")
        progress = plan.get("progress", {})
        steps = plan.get("steps", [])

        # Build the content
        elements = []

        # Progress bar
        progress_bar = self._render_progress_bar(progress)
        elements.append(progress_bar)
        elements.append(Text(""))  # Spacer

        # Steps table
        if steps:
            steps_table = self._render_steps_table(steps)
            elements.append(steps_table)

        # Get status display
        emoji, status_text, color = self.PLAN_STATUS_DISPLAY.get(
            status, ("📋", status.upper(), "white")
        )

        panel_title = f"[bold]{emoji} {title}[/bold]"
        if status != "pending":
            panel_title += f" [{color}]({status_text})[/{color}]"

        return Panel(
            Group(*elements),
            title=panel_title,
            border_style=color,
        )

    def _render_progress_bar(self, progress: Dict[str, Any]) -> Text:
        """Render the progress bar."""
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        in_prog = progress.get("in_progress", 0)
        percent = progress.get("percent", 0)

        if total == 0:
            return Text("No steps defined", style="dim")

        # Build progress bar
        bar_width = 30
        filled = int(bar_width * percent / 100)

        bar = Text()
        bar.append("━" * filled, style="green")
        if filled < bar_width:
            bar.append("━" * (bar_width - filled), style="dim")

        # Stats
        stats = Text()
        stats.append(f" {completed}", style="green")
        stats.append(f"/{total}")
        if in_prog > 0:
            stats.append(f" ({in_prog} running)", style="blue")
        if failed > 0:
            stats.append(f" ({failed} failed)", style="red")
        stats.append(f" [{percent:.0f}%]", style="bold")

        result = Text()
        result.append_text(bar)
        result.append_text(stats)
        return result

    def _render_steps_table(self, steps: List[Dict[str, Any]]) -> Table:
        """Render the steps as a compact table."""
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("status", width=2)
        table.add_column("seq", width=3)
        table.add_column("description", ratio=1)

        # Sort by sequence
        sorted_steps = sorted(steps, key=lambda s: s.get("sequence", 0))

        # Find current step (first in_progress, or first pending if none in_progress)
        current_step_seq = None
        for step in sorted_steps:
            if step.get("status") == "in_progress":
                current_step_seq = step.get("sequence")
                break

        for step in sorted_steps:
            seq = step.get("sequence", "?")
            desc = step.get("description", "")
            step_status = step.get("status", "pending")
            result = step.get("result", "")
            error = step.get("error", "")

            # Get symbol and style
            symbol, style = self.STATUS_SYMBOLS.get(step_status, ("?", "white"))

            # Highlight current step
            is_current = seq == current_step_seq
            desc_style = "bold" if is_current else ""

            # Add step row
            table.add_row(
                Text(symbol, style=style),
                Text(f"{seq}.", style="dim"),
                Text(desc, style=desc_style),
            )

            # Add result/error on next line if present
            if step_status == "completed" and result:
                table.add_row(
                    Text(""),
                    Text(""),
                    Text(f"→ {result}", style="dim green"),
                )
            elif step_status == "failed" and error:
                table.add_row(
                    Text(""),
                    Text(""),
                    Text(f"✗ {error}", style="dim red"),
                )

        return table
