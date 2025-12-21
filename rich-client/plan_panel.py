"""Plan panel rendering for status bar symbols and popup overlay.

Provides compact plan progress display in the status bar (symbols only)
with a detailed popup overlay accessible via Ctrl+P.
"""

from typing import Any, Dict, List, Optional

from rich.panel import Panel
from rich.text import Text
from rich.console import Group


class PlanPanel:
    """Renders plan status as symbols for status bar and popup overlay."""

    # Status symbols with colors
    STATUS_SYMBOLS = {
        "pending": ("○", "dim"),
        "in_progress": ("◐", "blue"),
        "completed": ("●", "green"),
        "failed": ("✗", "red"),
        "skipped": ("⊘", "yellow"),
    }

    def __init__(self):
        self._plan_data: Optional[Dict[str, Any]] = None
        self._popup_visible: bool = False

    def update_plan(self, plan_data: Dict[str, Any]) -> None:
        """Update the plan data to render.

        Args:
            plan_data: Plan status dict with title, status, steps, progress.
        """
        self._plan_data = plan_data

    def clear(self) -> None:
        """Clear the current plan."""
        self._plan_data = None
        self._popup_visible = False

    @property
    def has_plan(self) -> bool:
        """Check if there's an active plan to display."""
        return self._plan_data is not None

    @property
    def is_popup_visible(self) -> bool:
        """Check if popup is visible."""
        return self._popup_visible

    def toggle_popup(self) -> None:
        """Toggle popup visibility (Ctrl+P)."""
        self._popup_visible = not self._popup_visible

    def get_status_symbols(self) -> str:
        """Get plan progress as compact symbols for status bar.

        Returns a string like "●●●◐○○" representing step states,
        or empty string if no plan.
        """
        if not self._plan_data:
            return ""

        steps = self._plan_data.get("steps", [])
        if not steps:
            return ""

        # Sort by sequence and build symbol string
        sorted_steps = sorted(steps, key=lambda s: s.get("sequence", 0))
        symbols = []
        for step in sorted_steps:
            status = step.get("status", "pending")
            symbol, _ = self.STATUS_SYMBOLS.get(status, ("○", "dim"))
            symbols.append(symbol)

        return "".join(symbols)

    def get_status_symbols_formatted(self) -> List[tuple]:
        """Get plan progress as formatted text tuples for prompt_toolkit.

        Returns list of (style, symbol) tuples for colored display.
        """
        if not self._plan_data:
            return []

        steps = self._plan_data.get("steps", [])
        if not steps:
            return []

        # Map our styles to prompt_toolkit style classes
        style_map = {
            "dim": "class:plan.pending",
            "blue": "class:plan.in-progress",
            "green": "class:plan.completed",
            "red": "class:plan.failed",
            "yellow": "class:plan.skipped",
        }

        # Sort by sequence and build formatted tuples
        sorted_steps = sorted(steps, key=lambda s: s.get("sequence", 0))
        result = []
        for step in sorted_steps:
            status = step.get("status", "pending")
            symbol, color = self.STATUS_SYMBOLS.get(status, ("○", "dim"))
            style = style_map.get(color, "class:plan.pending")
            result.append((style, symbol))

        return result

    def render_popup(self, width: int = 50) -> Panel:
        """Render the plan popup overlay.

        Args:
            width: Width of the popup panel.

        Returns:
            Rich Panel containing the full plan details.
        """
        if not self._plan_data:
            return Panel(
                Text("No active plan", style="dim italic"),
                title="[bold]Plan[/bold]",
                border_style="dim",
                width=width,
            )

        plan = self._plan_data
        title = plan.get("title", "Untitled Plan")
        steps = plan.get("steps", [])
        progress = plan.get("progress", {})

        # Build content
        elements = []

        # Steps list
        if steps:
            sorted_steps = sorted(steps, key=lambda s: s.get("sequence", 0))
            for step in sorted_steps:
                seq = step.get("sequence", "?")
                desc = step.get("description", "")
                step_status = step.get("status", "pending")
                result = step.get("result", "")
                error = step.get("error", "")

                symbol, color = self.STATUS_SYMBOLS.get(step_status, ("○", "dim"))

                # Step line
                line = Text()
                line.append(f" {symbol} ", style=color)
                line.append(f"{seq}. ", style="dim")
                line.append(desc, style="bold" if step_status == "in_progress" else "")
                elements.append(line)

                # Result/error line
                if step_status == "completed" and result:
                    result_line = Text()
                    result_line.append("    → ", style="dim")
                    result_line.append(result, style="dim green")
                    elements.append(result_line)
                elif step_status == "failed" and error:
                    error_line = Text()
                    error_line.append("    ✗ ", style="dim")
                    error_line.append(error, style="dim red")
                    elements.append(error_line)

        # Separator
        elements.append(Text("─" * (width - 4), style="dim"))

        # Progress line
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        percent = progress.get("percent", 0)
        progress_text = Text()
        progress_text.append(f" {percent:.0f}% ", style="bold")
        progress_text.append(f"({completed}/{total})", style="dim")
        # Calculate padding to right-align [Ctrl+P]
        left_content = f" {percent:.0f}% ({completed}/{total})"
        right_content = "[Ctrl+P]"
        padding = max(1, width - 4 - len(left_content) - len(right_content))
        progress_text.append(" " * padding)
        progress_text.append(right_content, style="dim")
        elements.append(progress_text)

        return Panel(
            Group(*elements),
            title=f"[bold]{title}[/bold]",
            border_style="blue",
            width=width,
        )
