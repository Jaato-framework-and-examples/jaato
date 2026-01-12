"""Plan panel rendering for status bar symbols and popup overlay.

Provides compact plan progress display in the status bar (symbols only)
with a detailed popup overlay accessible via the configured toggle key.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text
from rich.console import Group

from keybindings import KeyBinding, format_key_for_display

if TYPE_CHECKING:
    from theme import ThemeConfig


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

    def __init__(self, toggle_key: Optional[KeyBinding] = None):
        self._plan_data: Optional[Dict[str, Any]] = None
        self._popup_visible: bool = False
        self._popup_scroll_offset: int = 0  # Scroll position in popup
        self._popup_max_visible_steps: int = 10  # Max steps visible at once
        self._toggle_key = toggle_key or "c-p"
        self._theme: Optional["ThemeConfig"] = None

    def set_theme(self, theme: "ThemeConfig") -> None:
        """Set the theme configuration for styling.

        Args:
            theme: ThemeConfig instance for Rich style lookups.
        """
        self._theme = theme

    def _get_status_style(self, status: str) -> str:
        """Get style for a plan status from theme or fallback.

        Args:
            status: Status name (pending, in_progress, completed, failed, skipped).

        Returns:
            Rich style string.
        """
        if self._theme:
            semantic_name = f"plan_{status}"
            style = self._theme.get_rich_style(semantic_name)
            if style:
                return style
        # Fallback to STATUS_SYMBOLS
        _, fallback = self.STATUS_SYMBOLS.get(status, ("?", ""))
        return fallback

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
        self._popup_scroll_offset = 0

    @property
    def has_plan(self) -> bool:
        """Check if there's an active plan to display."""
        return self._plan_data is not None

    @property
    def is_popup_visible(self) -> bool:
        """Check if popup is visible."""
        return self._popup_visible

    def toggle_popup(self) -> None:
        """Toggle popup visibility."""
        self._popup_visible = not self._popup_visible
        if not self._popup_visible:
            self._popup_scroll_offset = 0  # Reset scroll when closing

    def scroll_popup_up(self, plan_data: Optional[Dict[str, Any]] = None) -> bool:
        """Scroll popup up by one step.

        Args:
            plan_data: Optional plan data (for per-agent plans).

        Returns:
            True if scrolled, False if already at top.
        """
        if self._popup_scroll_offset > 0:
            self._popup_scroll_offset -= 1
            return True
        return False

    def scroll_popup_down(self, plan_data: Optional[Dict[str, Any]] = None) -> bool:
        """Scroll popup down by one step.

        Args:
            plan_data: Optional plan data (for per-agent plans).

        Returns:
            True if scrolled, False if already at bottom.
        """
        data = plan_data if plan_data is not None else self._plan_data
        if not data:
            return False

        steps = data.get("steps", [])
        max_offset = max(0, len(steps) - self._popup_max_visible_steps)
        if self._popup_scroll_offset < max_offset:
            self._popup_scroll_offset += 1
            return True
        return False

    def set_popup_max_visible_steps(self, max_steps: int) -> None:
        """Set the maximum number of steps visible in the popup.

        Args:
            max_steps: Maximum number of steps to show at once.
        """
        self._popup_max_visible_steps = max(3, max_steps)  # At least 3 steps

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

    def render_popup(self, width: int = 50, plan_data: Optional[Dict[str, Any]] = None) -> Panel:
        """Render the plan popup overlay.

        Args:
            width: Width of the popup panel.
            plan_data: Optional plan data to render. If None, uses self._plan_data.

        Returns:
            Rich Panel containing the full plan details.
        """
        # Use provided plan_data or fall back to instance data
        data = plan_data if plan_data is not None else self._plan_data

        if not data:
            return Panel(
                Text("No active plan", style="dim italic"),
                title="[bold]Plan[/bold]",
                border_style="dim",
                width=width,
            )

        plan = data
        title = plan.get("title", "Untitled Plan")
        steps = plan.get("steps", [])
        progress = plan.get("progress", {})

        # Build content
        elements = []

        # Steps list with scroll support
        if steps:
            sorted_steps = sorted(steps, key=lambda s: s.get("sequence", 0))
            total_steps = len(sorted_steps)

            # Calculate visible window
            start_idx = self._popup_scroll_offset
            end_idx = min(start_idx + self._popup_max_visible_steps, total_steps)

            # Ensure scroll offset is valid
            max_offset = max(0, total_steps - self._popup_max_visible_steps)
            if self._popup_scroll_offset > max_offset:
                self._popup_scroll_offset = max_offset
                start_idx = self._popup_scroll_offset
                end_idx = min(start_idx + self._popup_max_visible_steps, total_steps)

            # Show "more above" indicator
            if start_idx > 0:
                more_above = Text()
                more_above.append(f" ↑ {start_idx} more above ", style="dim italic")
                more_above.append("[↑/↓ to scroll]", style="dim")
                elements.append(more_above)

            # Render visible steps
            visible_steps = sorted_steps[start_idx:end_idx]
            for step in visible_steps:
                seq = step.get("sequence", "?")
                desc = step.get("description", "")
                step_status = step.get("status", "pending")
                result = step.get("result", "")
                error = step.get("error", "")

                symbol, _ = self.STATUS_SYMBOLS.get(step_status, ("○", "dim"))
                color = self._get_status_style(step_status)

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

            # Show "more below" indicator
            remaining = total_steps - end_idx
            if remaining > 0:
                more_below = Text()
                more_below.append(f" ↓ {remaining} more below ", style="dim italic")
                more_below.append("[↑/↓ to scroll]", style="dim")
                elements.append(more_below)

        # Separator
        elements.append(Text("─" * (width - 4), style="dim"))

        # Progress line with close hint
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        percent = progress.get("percent", 0)
        progress_text = Text()
        progress_text.append(f" {percent:.0f}% ", style="bold")
        progress_text.append(f"({completed}/{total})", style="dim")
        # Calculate padding to right-align hint
        left_content = f" {percent:.0f}% ({completed}/{total})"
        right_content = f"[{format_key_for_display(self._toggle_key)} to close]"
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
