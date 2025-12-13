"""Output buffer for the scrolling output panel.

Manages a ring buffer of output lines for display in the scrolling
region of the TUI.
"""

import textwrap
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


@dataclass
class ActiveToolCall:
    """Represents an actively executing tool call."""
    name: str
    args_summary: str  # Truncated string representation of args


class OutputBuffer:
    """Manages output lines for the scrolling panel.

    Stores output in a ring buffer and renders to Rich Text
    for display in the output panel.
    """

    # Spinner animation frames
    SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

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
        self._spinner_active: bool = False
        self._spinner_index: int = 0
        self._active_tools: List[ActiveToolCall] = []  # Currently executing tools

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
        # Skip plan messages - they're shown in the sticky plan panel
        if source == "plan":
            return

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
            # Join parts with newlines - each append is a separate line
            full_text = '\n'.join(parts)
            lines = full_text.split('\n')
            for i, line in enumerate(lines):
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
        self._spinner_active = False
        self._active_tools.clear()

    def start_spinner(self) -> None:
        """Start showing spinner in the output."""
        self._spinner_active = True
        self._spinner_index = 0

    def stop_spinner(self) -> None:
        """Stop showing spinner and clear active tools."""
        self._spinner_active = False
        self._active_tools.clear()  # Clear active tools when spinner stops

    def advance_spinner(self) -> None:
        """Advance spinner to next frame."""
        if self._spinner_active:
            self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)

    @property
    def spinner_active(self) -> bool:
        """Check if spinner is currently active."""
        return self._spinner_active

    def add_active_tool(self, tool_name: str, tool_args: dict) -> None:
        """Add a tool to the active tools list.

        Args:
            tool_name: Name of the tool being executed.
            tool_args: Arguments passed to the tool.
        """
        # Create a summary of args (truncated for display)
        args_str = str(tool_args)
        if len(args_str) > 60:
            args_str = args_str[:57] + "..."

        # Don't add duplicates
        for tool in self._active_tools:
            if tool.name == tool_name:
                return

        self._active_tools.append(ActiveToolCall(name=tool_name, args_summary=args_str))

    def remove_active_tool(self, tool_name: str) -> None:
        """Remove a tool from the active tools list.

        Args:
            tool_name: Name of the tool that finished.
        """
        self._active_tools = [t for t in self._active_tools if t.name != tool_name]

    def clear_active_tools(self) -> None:
        """Clear all active tools."""
        self._active_tools.clear()

    @property
    def active_tools(self) -> List[ActiveToolCall]:
        """Get list of currently active tools."""
        return list(self._active_tools)

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

        # If buffer is empty but spinner is active, show only spinner
        if not self._lines:
            if self._spinner_active:
                output = Text()
                frame = self.SPINNER_FRAMES[self._spinner_index]
                output.append(f"Model> {frame} ", style="bold cyan")
                output.append("thinking...", style="dim italic")
                return output
            return Text("Waiting for output...", style="dim italic")

        # Update width if provided
        if width and width != self._console_width:
            self.set_width(width)

        # Work backwards from the end, using stored display line counts
        # First skip _scroll_offset lines, then collect 'height' lines
        all_lines = list(self._lines)
        lines_to_show: List[OutputLine] = []

        if height:
            # Calculate total display lines
            total_display_lines = sum(line.display_lines for line in all_lines)

            # Find the end position (bottom of visible window)
            # scroll_offset=0 means show the most recent content
            # scroll_offset>0 means we've scrolled up, showing older content
            end_display_line = total_display_lines - self._scroll_offset
            start_display_line = max(0, end_display_line - height)

            # Collect lines that fall within the visible range
            current_display_line = 0
            for line in all_lines:
                line_end = current_display_line + line.display_lines
                # Include line if it overlaps with visible range
                if line_end > start_display_line and current_display_line < end_display_line:
                    lines_to_show.append(line)
                current_display_line = line_end
                # Stop if we've passed the visible range
                if current_display_line >= end_display_line:
                    break
        else:
            lines_to_show = all_lines

        # Build output text with wrapping
        output = Text()

        # Calculate available width for content (accounting for prefixes)
        wrap_width = self._console_width if self._console_width > 20 else 80

        for i, line in enumerate(lines_to_show):
            if i > 0:
                output.append("\n")

            # Wrap text to fit console width
            def wrap_text(text: str, prefix_width: int = 0) -> List[str]:
                """Wrap text to console width, accounting for prefix."""
                available = max(20, wrap_width - prefix_width)
                if len(text) <= available:
                    return [text]
                # Use textwrap for clean word-based wrapping
                return textwrap.wrap(text, width=available, break_long_words=True, break_on_hyphens=False)

            if line.source == "system":
                # System messages use their style directly
                wrapped = wrap_text(line.text)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    output.append(wrapped_line, style=line.style)
            elif line.source == "user":
                # User input - show with You> prefix on turn start
                prefix_width = 5 if line.is_turn_start else 0  # "You> " = 5 chars
                wrapped = wrap_text(line.text, prefix_width)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    if j == 0 and line.is_turn_start:
                        output.append("You> ", style="bold green")
                    elif j > 0 and line.is_turn_start:
                        output.append("     ")  # Indent continuation lines
                    output.append(wrapped_line)
            elif line.source == "model":
                # Model output - only show prefix on turn start
                prefix_width = 7 if line.is_turn_start else 0  # "Model> " = 7 chars
                wrapped = wrap_text(line.text, prefix_width)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    if j == 0 and line.is_turn_start:
                        output.append("Model> ", style="bold cyan")
                    elif j > 0 and line.is_turn_start:
                        output.append("       ")  # Indent continuation lines
                    output.append(wrapped_line)
            elif line.source == "tool":
                # Tool output
                prefix_width = len(f"[{line.source}] ") if line.is_turn_start else 0
                wrapped = wrap_text(line.text, prefix_width)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    if j == 0 and line.is_turn_start:
                        output.append(f"[{line.source}] ", style="dim yellow")
                    elif j > 0 and line.is_turn_start:
                        output.append(" " * (len(f"[{line.source}] ")))  # Indent continuation
                    output.append(wrapped_line, style="dim")
            elif line.source == "permission":
                # Permission prompts - wrap but preserve ANSI codes
                text = line.text
                if "[askPermission]" in text:
                    text = text.replace("[askPermission]", "")
                    wrapped = wrap_text(text, 16)  # "[askPermission] " = 16 chars
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n                ")  # Indent continuation
                        if j == 0:
                            output.append("[askPermission] ", style="bold yellow")
                        output.append_text(Text.from_ansi(wrapped_line))
                elif "Options:" in text or text.startswith(("===", "─", "=")) or "Enter choice" in text:
                    # Special lines - wrap normally
                    wrapped = wrap_text(text)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        if "Options:" in text:
                            output.append_text(Text.from_ansi(wrapped_line, style="cyan"))
                        elif text.startswith(("===", "─", "=")):
                            output.append(wrapped_line, style="dim")
                        else:
                            output.append_text(Text.from_ansi(wrapped_line, style="cyan"))
                else:
                    # Preserve ANSI codes with wrapping
                    wrapped = wrap_text(text)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        output.append_text(Text.from_ansi(wrapped_line))
            elif line.source == "clarification":
                # Clarification prompts - wrap but preserve ANSI codes
                text = line.text
                wrapped = wrap_text(text)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    if "Clarification Needed" in wrapped_line:
                        output.append_text(Text.from_ansi(wrapped_line, style="bold cyan"))
                    elif wrapped_line.startswith(("===", "─", "=")):
                        output.append(wrapped_line, style="dim")
                    elif "Enter choice" in wrapped_line:
                        output.append_text(Text.from_ansi(wrapped_line, style="cyan"))
                    elif wrapped_line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                        output.append_text(Text.from_ansi(wrapped_line, style="cyan"))
                    elif "Question" in wrapped_line and "/" in wrapped_line:
                        output.append_text(Text.from_ansi(wrapped_line, style="bold"))
                    elif "[*required]" in wrapped_line:
                        wrapped_line = wrapped_line.replace("[*required]", "")
                        output.append_text(Text.from_ansi(wrapped_line))
                        output.append("[*required]", style="yellow")
                    else:
                        output.append_text(Text.from_ansi(wrapped_line))
            else:
                # Other plugin output - wrap and preserve ANSI codes
                prefix_width = len(f"[{line.source}] ") if line.is_turn_start else 0
                wrapped = wrap_text(line.text, prefix_width)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    if j == 0 and line.is_turn_start:
                        output.append(f"[{line.source}] ", style="dim magenta")
                    elif j > 0 and line.is_turn_start:
                        output.append(" " * (len(f"[{line.source}] ")))  # Indent continuation
                    output.append_text(Text.from_ansi(wrapped_line))

        # Add spinner at the bottom if active
        if self._spinner_active:
            if lines_to_show:
                output.append("\n")
            frame = self.SPINNER_FRAMES[self._spinner_index]
            output.append(f"Model> {frame} ", style="bold cyan")
            output.append("thinking...", style="dim italic")

            # Show active tool calls below spinner
            if self._active_tools:
                for i, tool in enumerate(self._active_tools):
                    output.append("\n")
                    is_last = (i == len(self._active_tools) - 1)
                    prefix = "└─" if is_last else "├─"
                    output.append(f"       {prefix} ", style="dim")
                    output.append(tool.name, style="yellow")
                    if tool.args_summary and tool.args_summary != "{}":
                        output.append(f"({tool.args_summary})", style="dim")

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
            width=width,  # Constrain panel to exact width (preserves right border)
        )
