"""Output buffer for the scrolling output panel.

Manages a ring buffer of output lines for display in the scrolling
region of the TUI.
"""

from collections import deque
from typing import List, Optional, Tuple

from rich.console import Console, RenderableType
from rich.text import Text
from rich.panel import Panel


class OutputBuffer:
    """Manages output lines for the scrolling panel.

    Stores output in a ring buffer and renders to Rich Text
    for display in the output panel.
    """

    def __init__(self, max_lines: int = 1000):
        """Initialize the output buffer.

        Args:
            max_lines: Maximum number of lines to retain.
        """
        self._lines: deque[Tuple[str, str, str]] = deque(maxlen=max_lines)
        self._current_block: Optional[Tuple[str, List[str]]] = None

    def append(self, source: str, text: str, mode: str) -> None:
        """Append output to the buffer.

        Args:
            source: Source of the output ("model", plugin name, etc.)
            text: The output text.
            mode: "write" for new block, "append" to continue.
        """
        if mode == "write":
            # Start a new block
            self._flush_current_block()
            self._current_block = (source, [text])
        elif mode == "append" and self._current_block:
            # Append to current block
            self._current_block[1].append(text)
        else:
            # Standalone line
            self._flush_current_block()
            for line in text.split('\n'):
                self._lines.append((source, line, "line"))

    def _flush_current_block(self) -> None:
        """Flush the current block to lines."""
        if self._current_block:
            source, parts = self._current_block
            full_text = ''.join(parts)
            for line in full_text.split('\n'):
                if line:  # Skip empty lines from split
                    self._lines.append((source, line, "line"))
            self._current_block = None

    def add_system_message(self, message: str, style: str = "dim") -> None:
        """Add a system message to the buffer.

        Args:
            message: The system message.
            style: Rich style for the message.
        """
        self._flush_current_block()
        self._lines.append(("system", message, style))

    def clear(self) -> None:
        """Clear all output."""
        self._lines.clear()
        self._current_block = None

    def render(self, height: Optional[int] = None) -> RenderableType:
        """Render the output buffer as Rich Text.

        Args:
            height: Optional height limit (shows last N lines).

        Returns:
            Rich renderable for the output panel.
        """
        self._flush_current_block()

        if not self._lines:
            return Text("Waiting for output...", style="dim italic")

        # Get lines to display
        lines_to_show = list(self._lines)
        if height and len(lines_to_show) > height:
            lines_to_show = lines_to_show[-height:]

        # Build output text
        output = Text()
        for i, (source, text, style_or_mode) in enumerate(lines_to_show):
            if i > 0:
                output.append("\n")

            if source == "system":
                # System messages use their style directly
                output.append(text, style=style_or_mode)
            elif source == "model":
                # Model output with prefix
                output.append("Model> ", style="bold cyan")
                output.append(text)
            elif source == "tool":
                # Tool output
                output.append(f"[{source}] ", style="dim yellow")
                output.append(text, style="dim")
            else:
                # Other plugin output
                output.append(f"[{source}] ", style="dim magenta")
                output.append(text)

        return output

    def render_panel(self, height: Optional[int] = None) -> Panel:
        """Render as a Panel.

        Args:
            height: Optional height limit.

        Returns:
            Panel containing the output.
        """
        return Panel(
            self.render(height),
            title="[bold]Output[/bold]",
            border_style="blue",
        )
