"""Live display manager for the Rich TUI.

Manages the Rich Live context with a split layout:
- Top: Sticky plan panel (fixed height)
- Bottom: Scrolling output panel (flexible)
"""

import sys
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from plan_panel import PlanPanel
from output_buffer import OutputBuffer

if TYPE_CHECKING:
    from input_handler import InputHandler


class LiveDisplay:
    """Manages the Rich Live display with sticky plan and scrolling output.

    The display is split into two regions:
    - Plan panel at top (sticky, updates in place)
    - Output panel below (scrolls with new content)

    This class manages the Live context and provides methods to update
    each region independently.
    """

    def __init__(self, refresh_rate: int = 4):
        """Initialize the live display.

        Args:
            refresh_rate: Screen refresh rate per second.
        """
        self._console = Console()
        self._refresh_rate = refresh_rate
        self._live: Optional[Live] = None

        # Display components
        self._plan_panel = PlanPanel()
        self._output_buffer = OutputBuffer()

        # Layout configuration
        self._plan_height = 12  # Fixed height for plan panel
        self._layout: Optional[Layout] = None

    def _check_tty(self) -> None:
        """Verify we're running in a TTY."""
        if not sys.stdout.isatty():
            sys.exit(
                "Error: rich-client requires an interactive terminal.\n"
                "Use simple-client for non-TTY environments (pipes, scripts, etc.)"
            )

    def _build_layout(self) -> Layout:
        """Build the split layout."""
        layout = Layout()

        # Split into plan (top, fixed) and output (bottom, flexible)
        layout.split_column(
            Layout(name="plan", size=self._plan_height),
            Layout(name="output"),
        )

        return layout

    def _update_layout(self) -> None:
        """Update layout content."""
        if not self._layout:
            return

        # Update plan panel
        self._layout["plan"].update(self._plan_panel.render())

        # Calculate available height for output
        # Console height minus plan panel minus some margin
        try:
            available_height = self._console.height - self._plan_height - 4
            available_height = max(10, available_height)
        except Exception:
            available_height = 20

        # Update output panel
        self._layout["output"].update(
            self._output_buffer.render_panel(height=available_height)
        )

    def start(self) -> None:
        """Start the live display.

        Enters the alternate screen buffer and begins live updates.
        """
        self._check_tty()
        self._layout = self._build_layout()
        self._update_layout()

        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=self._refresh_rate,
            screen=True,  # Use alternate screen buffer
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display.

        Exits the alternate screen buffer.
        """
        if self._live:
            self._live.stop()
            self._live = None

    def refresh(self) -> None:
        """Force a display refresh."""
        if self._live:
            self._update_layout()
            self._live.refresh()

    # Plan panel methods

    def update_plan(self, plan_data: Dict[str, Any]) -> None:
        """Update the plan panel with new plan data.

        Args:
            plan_data: Plan status dict with title, status, steps, progress.
        """
        self._plan_panel.update_plan(plan_data)
        self.refresh()

    def clear_plan(self) -> None:
        """Clear the plan panel."""
        self._plan_panel.clear()
        self.refresh()

    @property
    def has_plan(self) -> bool:
        """Check if there's an active plan."""
        return self._plan_panel.has_plan

    # Output buffer methods

    def append_output(self, source: str, text: str, mode: str) -> None:
        """Append output to the scrolling panel.

        Args:
            source: Source of output ("model", plugin name, etc.)
            text: The output text.
            mode: "write" for new block, "append" to continue.
        """
        self._output_buffer.append(source, text, mode)
        self.refresh()

    def add_system_message(self, message: str, style: str = "dim") -> None:
        """Add a system message to the output.

        Args:
            message: The system message.
            style: Rich style for the message.
        """
        self._output_buffer.add_system_message(message, style)
        self.refresh()

    def clear_output(self) -> None:
        """Clear the output buffer."""
        self._output_buffer.clear()
        self.refresh()

    # Context manager support

    def __enter__(self) -> "LiveDisplay":
        """Enter context - start live display."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context - stop live display."""
        self.stop()

    # Input handling

    def get_input(
        self,
        prompt: str = "You> ",
        input_handler: Optional["InputHandler"] = None
    ) -> str:
        """Get user input while pausing live updates.

        The live display is temporarily suspended to allow clean input,
        then resumed after input is received.

        Args:
            prompt: The input prompt string.
            input_handler: Optional InputHandler for completion support.
                          If provided, uses prompt_toolkit with completions.
                          Otherwise falls back to basic Rich console input.

        Returns:
            The user's input string.
        """
        if self._live:
            # Temporarily stop live updates and exit alternate screen
            self._live.stop()

        try:
            if input_handler:
                # Use InputHandler with full completion support
                user_input = input_handler.get_input(prompt)
            else:
                # Fallback to basic Rich console input
                user_input = self._console.input(f"[green]{prompt}[/green]")
            return user_input.strip()
        finally:
            if self._live:
                # Resume live updates
                self._live.start()
                self.refresh()

    def print_static(self, *args, **kwargs) -> None:
        """Print static content (outside of live display).

        Useful for messages that should persist after live display stops.
        """
        if self._live:
            self._live.console.print(*args, **kwargs)
        else:
            self._console.print(*args, **kwargs)
