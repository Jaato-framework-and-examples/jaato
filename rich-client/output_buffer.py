"""Output buffer for the scrolling output panel.

Manages a ring buffer of output lines for display in the scrolling
region of the TUI.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rich.console import Console, RenderableType
from rich.text import Text
from rich.panel import Panel
from rich.align import Align


@dataclass
class OutputLine:
    """A single line of output with metadata."""
    source: str
    text: str
    style: str
    display_lines: int = 1  # How many terminal lines this takes when rendered
    is_turn_start: bool = False  # True if this is the first line of a new turn


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
        self._lines: deque[OutputLine] = deque(maxlen=max_lines)
        self._current_block: Optional[Tuple[str, List[str]]] = None
        self._measure_console: Optional[Console] = None
        self._console_width: int = 80
        self._last_source: Optional[str] = None  # Track source for turn detection
        self._scroll_offset: int = 0  # Lines scrolled up from bottom (0 = at bottom)

    def set_width(self, width: int) -> None:
        """Set the console width for measuring line wrapping.

        Args:
            width: Console width in characters.
        """
        self._console_width = width
        self._measure_console = Console(width=width, force_terminal=True)

    def _measure_display_lines(self, source: str, text: str, is_turn_start: bool = False) -> int:
        """Measure how many display lines a piece of text will take.

        Args:
            source: Source of the output.
            text: The text content.
            is_turn_start: Whether this is the first line of a turn (shows prefix).

        Returns:
            Number of display lines when rendered.
        """
        if not self._measure_console:
            self._measure_console = Console(width=self._console_width, force_terminal=True)

        # Build the text as it will be rendered
        rendered = Text()
        if source == "model":
            if is_turn_start:
                rendered.append("Model> ", style="bold cyan")
            rendered.append(text)
        elif source == "system":
            rendered.append(text)
        else:
            if is_turn_start:
                rendered.append(f"[{source}] ", style="dim magenta")
            rendered.append(text)

        # Measure by capturing output
        with self._measure_console.capture() as capture:
            self._measure_console.print(rendered, end='')

        output = capture.get()
        if not output:
            return 1
        return output.count('\n') + 1

    def _add_line(self, source: str, text: str, style: str, is_turn_start: bool = False) -> None:
        """Add a line to the buffer with measured display lines.

        Args:
            source: Source of the output.
            text: The text content.
            style: Style for the line.
            is_turn_start: Whether this is the first line of a new turn.
        """
        display_lines = self._measure_display_lines(source, text, is_turn_start)
        self._lines.append(OutputLine(source, text, style, display_lines, is_turn_start))

    def append(self, source: str, text: str, mode: str) -> None:
        """Append output to the buffer.

        Args:
            source: Source of the output ("model", plugin name, etc.)
            text: The output text.
            mode: "write" for new block, "append" to continue.
        """
        if mode == "write":
            # Start a new block - this is a new turn
            self._flush_current_block()
            self._current_block = (source, [text], True)  # True = is new turn
        elif mode == "append" and self._current_block:
            # Append to current block
            self._current_block[1].append(text)
        else:
            # Standalone line
            self._flush_current_block()
            is_new_turn = self._last_source != source
            for i, line in enumerate(text.split('\n')):
                self._add_line(source, line, "line", is_turn_start=(i == 0 and is_new_turn))
            self._last_source = source

    def _flush_current_block(self) -> None:
        """Flush the current block to lines."""
        if self._current_block:
            source, parts, is_new_turn = self._current_block
            full_text = ''.join(parts)
            lines = full_text.split('\n')
            for i, line in enumerate(lines):
                if line:  # Skip empty lines from split
                    # Only first line of a new turn gets the prefix
                    self._add_line(source, line, "line", is_turn_start=(i == 0 and is_new_turn))
            self._last_source = source
            self._current_block = None

    def add_system_message(self, message: str, style: str = "dim") -> None:
        """Add a system message to the buffer.

        Args:
            message: The system message.
            style: Rich style for the message.
        """
        self._flush_current_block()
        self._add_line("system", message, style)

    def clear(self) -> None:
        """Clear all output."""
        self._lines.clear()
        self._current_block = None
        self._scroll_offset = 0

    def scroll_up(self, lines: int = 5) -> bool:
        """Scroll up (view older content).

        Args:
            lines: Number of display lines to scroll.

        Returns:
            True if scroll position changed.
        """
        # Calculate total display lines
        total_display_lines = sum(line.display_lines for line in self._lines)
        max_offset = max(0, total_display_lines - 1)

        old_offset = self._scroll_offset
        self._scroll_offset = min(self._scroll_offset + lines, max_offset)
        return self._scroll_offset != old_offset

    def scroll_down(self, lines: int = 5) -> bool:
        """Scroll down (view newer content).

        Args:
            lines: Number of display lines to scroll.

        Returns:
            True if scroll position changed.
        """
        old_offset = self._scroll_offset
        self._scroll_offset = max(0, self._scroll_offset - lines)
        return self._scroll_offset != old_offset

    def scroll_to_bottom(self) -> bool:
        """Scroll to the bottom (most recent content).

        Returns:
            True if scroll position changed.
        """
        old_offset = self._scroll_offset
        self._scroll_offset = 0
        return self._scroll_offset != old_offset

    @property
    def is_at_bottom(self) -> bool:
        """Check if scrolled to the bottom."""
        return self._scroll_offset == 0

    def render(self, height: Optional[int] = None, width: Optional[int] = None) -> RenderableType:
        """Render the output buffer as Rich Text.

        Args:
            height: Optional height limit (in display lines).
            width: Optional width for calculating line wrapping.

        Returns:
            Rich renderable for the output panel.
        """
        self._flush_current_block()

        if not self._lines:
            return Text("Waiting for output...", style="dim italic")

        # Update width if provided
        if width and width != self._console_width:
            self.set_width(width)

        # Work backwards from the end, using stored display line counts
        # First skip _scroll_offset lines, then collect 'height' lines
        all_lines = list(self._lines)
        lines_to_show: List[OutputLine] = []

        if height:
            # Skip scroll_offset display lines from the bottom
            display_lines_skipped = 0
            start_index = len(all_lines)

            for i in range(len(all_lines) - 1, -1, -1):
                line = all_lines[i]
                if display_lines_skipped + line.display_lines <= self._scroll_offset:
                    display_lines_skipped += line.display_lines
                    start_index = i
                else:
                    break

            # Now collect 'height' display lines going backwards from start_index
            display_lines_used = 0
            for i in range(start_index - 1, -1, -1):
                line = all_lines[i]
                if display_lines_used + line.display_lines <= height:
                    lines_to_show.insert(0, line)
                    display_lines_used += line.display_lines
                else:
                    break
        else:
            lines_to_show = all_lines

        # Build output text
        output = Text()

        for i, line in enumerate(lines_to_show):
            if i > 0:
                output.append("\n")

            if line.source == "system":
                # System messages use their style directly
                output.append(line.text, style=line.style)
            elif line.source == "user":
                # User input - show with You> prefix on turn start
                if line.is_turn_start:
                    output.append("You> ", style="bold green")
                output.append(line.text)
            elif line.source == "model":
                # Model output - only show prefix on turn start
                if line.is_turn_start:
                    output.append("Model> ", style="bold cyan")
                output.append(line.text)
            elif line.source == "tool":
                # Tool output
                if line.is_turn_start:
                    output.append(f"[{line.source}] ", style="dim yellow")
                output.append(line.text, style="dim")
            else:
                # Other plugin output
                if line.is_turn_start:
                    output.append(f"[{line.source}] ", style="dim magenta")
                output.append(line.text)

        return output

    def render_panel(self, height: Optional[int] = None, width: Optional[int] = None) -> Panel:
        """Render as a Panel.

        Args:
            height: Optional height limit (for content lines, not panel height).
            width: Optional width for calculating line wrapping.

        Returns:
            Panel containing the output.
        """
        # Account for panel border (2 lines) and padding when calculating content height
        content_height = (height - 2) if height else None
        # Account for panel borders (2 chars each side) when calculating content width
        content_width = (width - 4) if width else None

        content = self.render(content_height, content_width)

        # Use Align to push content to bottom of panel
        aligned_content = Align(content, vertical="bottom")

        # Show scroll indicator in title when not at bottom
        if self._scroll_offset > 0:
            title = f"[bold]Output[/bold] [dim](↑{self._scroll_offset} lines)[/dim]"
        else:
            title = "[bold]Output[/bold]"

        return Panel(
            aligned_content,
            title=title,
            border_style="blue",
            height=height,  # Constrain panel to exact height
        )
