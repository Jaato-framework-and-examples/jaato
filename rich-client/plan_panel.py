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
        self._collapsed: bool = False
        self._hidden: bool = False
        self._prev_started: bool = False

    def update_plan(self, plan_data: Dict[str, Any]) -> None:
        """Update the plan data to render.

        Args:
            plan_data: Plan status dict with title, status, steps, progress.
        """
        # Auto-collapse when plan transitions from not-started to started
        current_started = plan_data.get("started", False)
        if current_started and not self._prev_started:
            self._collapsed = True

        self._prev_started = current_started
        self._plan_data = plan_data

    def clear(self) -> None:
        """Clear the current plan."""
        self._plan_data = None
        self._collapsed = False
        self._hidden = False
        self._prev_started = False

    def toggle_collapsed(self) -> None:
        """Toggle between collapsed and expanded view (F1)."""
        self._collapsed = not self._collapsed

    def toggle_hidden(self) -> None:
        """Toggle panel visibility (Ctrl+F1)."""
        self._hidden = not self._hidden

    @property
    def has_plan(self) -> bool:
        """Check if there's an active plan to display."""
        return self._plan_data is not None

    @property
    def is_visible(self) -> bool:
        """Check if panel should be visible (has plan and not hidden)."""
        return self._plan_data is not None and not self._hidden

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

        # Get status display
        emoji, status_text, color = self.PLAN_STATUS_DISPLAY.get(
            status, ("📋", status.upper(), "white")
        )

        # Build panel title with F1 hint
        panel_title = f"[bold]{emoji} {title}[/bold] [dim]\\[F1][/dim]"
        if status != "pending":
            panel_title += f" [{color}]({status_text})[/{color}]"

        # Determine what to show based on state
        in_progress_count = progress.get("in_progress", 0)
        pending_count = progress.get("pending", 0)
        total = progress.get("total", 0)
        completed_count = progress.get("completed", 0)
        failed_count = progress.get("failed", 0)

        # Check if plan is complete (no pending, no in_progress)
        is_complete = (status == "completed" or
                       (total > 0 and in_progress_count == 0 and pending_count == 0))

        # Check if all tasks are still pending (none started yet)
        all_pending = (total > 0 and pending_count == total)

        # Build the content
        elements = []

        if is_complete:
            # Completed view: just show completion summary, no task list
            summary = plan.get("summary", "")
            done_text = Text()
            done_text.append(f"✓ {completed_count}/{total} completed", style="green")
            if failed_count > 0:
                done_text.append(f", {failed_count} failed", style="red")
            elements.append(done_text)
            if summary:
                elements.append(Text(""))
                elements.append(Text(summary, style="dim italic"))
        elif all_pending or not self._collapsed:
            # Expanded view: show progress bar and all steps
            progress_bar = self._render_progress_bar(progress)
            elements.append(progress_bar)
            elements.append(Text(""))  # Spacer
            if steps:
                steps_table = self._render_steps_table(steps)
                elements.append(steps_table)
        else:
            # Collapsed view: show progress bar and current step only
            progress_bar = self._render_progress_bar(progress)
            elements.append(progress_bar)
            elements.append(Text(""))  # Spacer
            if steps:
                current_step = self._render_current_step(steps)
                if current_step:
                    elements.append(current_step)

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

    def _render_current_step(self, steps: List[Dict[str, Any]]) -> Optional[Text]:
        """Render the current in-progress step and previous step for collapsed view."""
        # Sort by sequence
        sorted_steps = sorted(steps, key=lambda s: s.get("sequence", 0))

        # Find current step (first in_progress) and its index
        current_idx = None
        for i, step in enumerate(sorted_steps):
            if step.get("status") == "in_progress":
                current_idx = i
                break

        if current_idx is None:
            return None

        text = Text()

        # Show previous step if exists (dimmed, with result/error)
        if current_idx > 0:
            prev_step = sorted_steps[current_idx - 1]
            prev_seq = prev_step.get("sequence", "?")
            prev_desc = prev_step.get("description", "")
            prev_status = prev_step.get("status", "pending")
            prev_result = prev_step.get("result", "")
            prev_error = prev_step.get("error", "")
            prev_symbol, _ = self.STATUS_SYMBOLS.get(prev_status, ("○", "dim"))

            text.append(prev_symbol, style="dim")
            text.append(f" {prev_seq}. ", style="dim")
            text.append(prev_desc, style="dim")
            text.append("\n")

            # Show result/error line for previous step
            if prev_status == "completed" and prev_result:
                text.append("     → ", style="dim")
                text.append(prev_result, style="dim green")
                text.append("\n")
            elif prev_status == "failed" and prev_error:
                text.append("     ✗ ", style="dim")
                text.append(prev_error, style="dim red")
                text.append("\n")

        # Show current step (bold)
        current_step = sorted_steps[current_idx]
        seq = current_step.get("sequence", "?")
        desc = current_step.get("description", "")
        symbol, style = self.STATUS_SYMBOLS.get("in_progress", ("◐", "blue"))

        text.append(symbol, style=style)
        text.append(f" {seq}. ", style="dim")
        text.append(desc, style="bold")

        return text

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
