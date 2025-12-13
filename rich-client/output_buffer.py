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
    """Represents an actively executing or completed tool call."""
    name: str
    args_summary: str  # Truncated string representation of args
    completed: bool = False  # True when tool execution finished
    success: bool = True  # Whether the tool succeeded (only valid when completed)
    duration_seconds: Optional[float] = None  # Execution time (only valid when completed)
    error_message: Optional[str] = None  # Error message if tool failed
    # Permission tracking
    permission_state: Optional[str] = None  # None, "pending", "granted", "denied"
    permission_method: Optional[str] = None  # "yes", "always", "once", "never", "whitelist", "blacklist"
    permission_prompt_lines: Optional[List[str]] = None  # Expanded prompt while pending
    permission_truncated: bool = False  # True if prompt is truncated
    # Clarification tracking (per-question progressive display)
    clarification_state: Optional[str] = None  # None, "pending", "resolved"
    clarification_prompt_lines: Optional[List[str]] = None  # Current question lines
    clarification_truncated: bool = False  # True if prompt is truncated
    clarification_current_question: int = 0  # Current question index (1-based)
    clarification_total_questions: int = 0  # Total number of questions
    clarification_answered: Optional[List[Tuple[int, str]]] = None  # List of (question_index, answer_summary)


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

        # Skip permission messages - they're shown inline under tool calls in the tree
        if source == "permission":
            return

        # Skip clarification messages - they're shown inline under tool calls in the tree
        if source == "clarification":
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
        """Stop showing spinner. Tool history is preserved for display."""
        self._spinner_active = False
        # Don't clear active tools - they remain visible as completed tools

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

    def mark_tool_completed(self, tool_name: str, success: bool = True,
                            duration_seconds: Optional[float] = None,
                            error_message: Optional[str] = None) -> None:
        """Mark a tool as completed (keeps it in the tree with completion status).

        Args:
            tool_name: Name of the tool that finished.
            success: Whether the tool execution succeeded.
            duration_seconds: How long the tool took to execute.
            error_message: Error message if the tool failed.
        """
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.completed = True
                tool.success = success
                tool.duration_seconds = duration_seconds
                tool.error_message = error_message
                return

    def remove_active_tool(self, tool_name: str) -> None:
        """Remove a tool from the active tools list (legacy, now marks as completed).

        Args:
            tool_name: Name of the tool that finished.
        """
        # Instead of removing, mark as completed to keep the tree visible
        self.mark_tool_completed(tool_name)

    def clear_active_tools(self) -> None:
        """Clear all active tools."""
        self._active_tools.clear()

    @property
    def active_tools(self) -> List[ActiveToolCall]:
        """Get list of currently active tools."""
        return list(self._active_tools)

    def set_tool_permission_pending(self, tool_name: str, prompt_lines: List[str]) -> None:
        """Mark a tool as awaiting permission with the prompt to display.

        Args:
            tool_name: Name of the tool awaiting permission.
            prompt_lines: Lines of the permission prompt to display.
        """
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.permission_state = "pending"
                tool.permission_prompt_lines = prompt_lines
                # Scroll to bottom to show the prompt
                self._scroll_offset = 0
                return

    def set_tool_permission_resolved(self, tool_name: str, granted: bool,
                                      method: str) -> None:
        """Mark a tool's permission as resolved.

        Args:
            tool_name: Name of the tool.
            granted: Whether permission was granted.
            method: How permission was resolved (yes, always, once, never, whitelist, etc.)
        """
        for tool in self._active_tools:
            if tool.name == tool_name:
                tool.permission_state = "granted" if granted else "denied"
                tool.permission_method = method
                tool.permission_prompt_lines = None  # Clear expanded prompt
                return

    def set_tool_clarification_pending(self, tool_name: str, prompt_lines: List[str]) -> None:
        """Mark a tool as awaiting clarification (initial context only).

        Args:
            tool_name: Name of the tool awaiting clarification.
            prompt_lines: Initial context lines (not the questions).
        """
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.clarification_state = "pending"
                tool.clarification_prompt_lines = prompt_lines
                tool.clarification_answered = []  # Initialize answered list
                # Scroll to bottom to show the prompt
                self._scroll_offset = 0
                return

    def set_tool_clarification_question(
        self,
        tool_name: str,
        question_index: int,
        total_questions: int,
        question_lines: List[str]
    ) -> None:
        """Set the current question being displayed for clarification.

        Args:
            tool_name: Name of the tool.
            question_index: Current question number (1-based).
            total_questions: Total number of questions.
            question_lines: Lines for this question's prompt.
        """
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.clarification_state = "pending"
                tool.clarification_current_question = question_index
                tool.clarification_total_questions = total_questions
                tool.clarification_prompt_lines = question_lines
                if tool.clarification_answered is None:
                    tool.clarification_answered = []
                # Scroll to bottom to show the question
                self._scroll_offset = 0
                return

    def set_tool_question_answered(
        self,
        tool_name: str,
        question_index: int,
        answer_summary: str
    ) -> None:
        """Mark a clarification question as answered.

        Args:
            tool_name: Name of the tool.
            question_index: Question number that was answered (1-based).
            answer_summary: Brief summary of the answer.
        """
        for tool in self._active_tools:
            if tool.name == tool_name:
                if tool.clarification_answered is None:
                    tool.clarification_answered = []
                tool.clarification_answered.append((question_index, answer_summary))
                # Clear prompt lines since question is answered
                tool.clarification_prompt_lines = None
                return

    def set_tool_clarification_resolved(self, tool_name: str) -> None:
        """Mark a tool's clarification as fully resolved.

        Args:
            tool_name: Name of the tool.
        """
        for tool in self._active_tools:
            if tool.name == tool_name:
                tool.clarification_state = "resolved"
                tool.clarification_prompt_lines = None
                tool.clarification_current_question = 0
                tool.clarification_total_questions = 0
                return

    def get_pending_prompt_for_pager(self) -> Optional[Tuple[str, List[str]]]:
        """Get the pending prompt that's awaiting user input for pager display.

        Returns:
            Tuple of (type, lines) where type is "permission" or "clarification",
            or None if no prompt is pending.
        """
        for tool in self._active_tools:
            if tool.permission_state == "pending" and tool.permission_prompt_lines:
                return ("permission", tool.permission_prompt_lines)
            if tool.clarification_state == "pending" and tool.clarification_prompt_lines:
                return ("clarification", tool.clarification_prompt_lines)
        return None

    def has_truncated_pending_prompt(self) -> bool:
        """Check if there's a truncated prompt awaiting user input.

        Returns:
            True if a truncated permission or clarification prompt is pending.
        """
        for tool in self._active_tools:
            if tool.permission_state == "pending" and tool.permission_truncated:
                return True
            if tool.clarification_state == "pending" and tool.clarification_truncated:
                return True
        return False

    def has_pending_prompt(self) -> bool:
        """Check if there's any pending prompt awaiting user input.

        Returns:
            True if any permission or clarification prompt is pending.
        """
        for tool in self._active_tools:
            if tool.permission_state == "pending" and tool.permission_prompt_lines:
                return True
            if tool.clarification_state == "pending" and tool.clarification_prompt_lines:
                return True
        return False

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

        # Add tool call tree at the bottom (persistent)
        if self._active_tools:
            if lines_to_show:
                output.append("\n")

            # Check if waiting for user input (permission or clarification)
            awaiting_input = any(
                tool.permission_state == "pending" or tool.clarification_state == "pending"
                for tool in self._active_tools
            )

            # Show header based on state
            # Awaiting user input takes precedence over other states
            if awaiting_input:
                # Waiting for user response
                output.append("Model> ⏳ ", style="bold yellow")
                output.append("Awaiting...", style="dim italic")
            elif self._spinner_active:
                # Model is processing
                frame = self.SPINNER_FRAMES[self._spinner_index]
                output.append(f"Model> {frame} ", style="bold cyan")
                output.append("thinking...", style="dim italic")
            else:
                # All tools completed - show "Processed" header
                output.append("Model> ✓ ", style="bold green")
                output.append("Processed", style="dim italic")

            # Show tool calls below header
            for i, tool in enumerate(self._active_tools):
                output.append("\n")
                is_last = (i == len(self._active_tools) - 1)
                prefix = "└─" if is_last else "├─"
                continuation = "   " if is_last else "│  "
                output.append(f"       {prefix} ", style="dim")

                if tool.completed:
                    # Completed tool - show with checkmark and duration
                    status_icon = "✓" if tool.success else "✗"
                    status_style = "green" if tool.success else "red"
                    output.append(f"{status_icon} ", style=status_style)
                    output.append(tool.name, style="dim yellow")
                    if tool.duration_seconds is not None:
                        output.append(f" ({tool.duration_seconds:.2f}s)", style="dim")
                    # Add collapsed permission result on same line
                    if tool.permission_state in ("granted", "denied"):
                        output.append(" ", style="dim")
                        if tool.permission_state == "granted":
                            output.append("✓ ", style="green")
                            method_label = tool.permission_method or "allowed"
                            if method_label == "whitelist":
                                output.append("auto-approved (whitelist)", style="dim green")
                            else:
                                output.append(f"allowed ({method_label})", style="dim green")
                        else:
                            output.append("✗ ", style="red")
                            method_label = tool.permission_method or "denied"
                            if method_label == "blacklist":
                                output.append("blocked (blacklist)", style="dim red")
                            else:
                                output.append(f"denied ({method_label})", style="dim red")
                    # Add collapsed clarification result on same line
                    if tool.clarification_state == "resolved":
                        output.append(" ", style="dim")
                        output.append("✓ ", style="cyan")
                        output.append("answered", style="dim cyan")
                    # Show error message on next line for failed tools
                    if not tool.success and tool.error_message:
                        output.append("\n")
                        output.append(f"       {continuation}     ", style="dim")
                        output.append("⚠ ", style="red")
                        # Truncate error message if too long
                        error_msg = tool.error_message
                        if len(error_msg) > 60:
                            error_msg = error_msg[:57] + "..."
                        output.append(error_msg, style="dim red")
                else:
                    # Active tool - show with spinner indicator
                    output.append("○ ", style="yellow")
                    output.append(tool.name, style="yellow")
                    if tool.args_summary and tool.args_summary != "{}":
                        output.append(f"({tool.args_summary})", style="dim")

                # Show permission info under this tool (only when pending)
                if tool.permission_state == "pending" and tool.permission_prompt_lines:
                    # Expanded permission prompt
                    output.append("\n")
                    output.append(f"       {continuation}     ", style="dim")
                    output.append("⚠ ", style="yellow")
                    output.append("Permission required", style="yellow")

                    # Limit lines to show (keep options visible at end)
                    max_prompt_lines = 18
                    prompt_lines = tool.permission_prompt_lines
                    total_lines = len(prompt_lines)
                    truncated = False
                    hidden_count = 0

                    if total_lines > max_prompt_lines:
                        # Show first (max - 2) lines, then "...N more...", then last line (options)
                        truncated = True
                        tool.permission_truncated = True
                        hidden_count = total_lines - max_prompt_lines + 1
                        # First part + placeholder + last line
                        lines_to_render = prompt_lines[:max_prompt_lines - 2]
                    else:
                        tool.permission_truncated = False
                        lines_to_render = prompt_lines

                    # Draw box around permission prompt
                    box_width = max(len(line) for line in prompt_lines) + 4
                    box_width = min(box_width, 60)  # Cap width
                    output.append("\n")
                    output.append(f"       {continuation}     ┌" + "─" * box_width + "┐", style="dim")

                    for prompt_line in lines_to_render:
                        output.append("\n")
                        # Truncate long lines
                        display_line = prompt_line[:box_width - 2] if len(prompt_line) > box_width - 2 else prompt_line
                        padding = box_width - len(display_line) - 2
                        output.append(f"       {continuation}     │ ", style="dim")
                        # Color diff lines appropriately
                        if display_line.startswith('+') and not display_line.startswith('+++'):
                            output.append(display_line, style="green")
                        elif display_line.startswith('-') and not display_line.startswith('---'):
                            output.append(display_line, style="red")
                        elif display_line.startswith('@@'):
                            output.append(display_line, style="cyan")
                        else:
                            output.append(display_line)
                        output.append(" " * padding + " │", style="dim")

                    # Show truncation indicator if needed
                    if truncated:
                        output.append("\n")
                        truncation_msg = f"[...{hidden_count} more - 'v' to view...]"
                        padding = box_width - len(truncation_msg) - 2
                        output.append(f"       {continuation}     │ ", style="dim")
                        output.append(truncation_msg, style="dim italic cyan")
                        output.append(" " * padding + " │", style="dim")
                        # Show last line (usually options)
                        last_line = prompt_lines[-1]
                        output.append("\n")
                        display_line = last_line[:box_width - 2] if len(last_line) > box_width - 2 else last_line
                        padding = box_width - len(display_line) - 2
                        output.append(f"       {continuation}     │ ", style="dim")
                        output.append(display_line, style="cyan")  # Options styled cyan
                        output.append(" " * padding + " │", style="dim")

                    output.append("\n")
                    output.append(f"       {continuation}     └" + "─" * box_width + "┘", style="dim")

                # Show clarification info under this tool
                if tool.clarification_state == "pending":
                    # Show header with progress
                    output.append("\n")
                    output.append(f"       {continuation}     ", style="dim")
                    output.append("❓ ", style="cyan")
                    if tool.clarification_total_questions > 0:
                        output.append(f"Clarification ({tool.clarification_current_question}/{tool.clarification_total_questions})", style="cyan")
                    else:
                        output.append("Clarification needed", style="cyan")

                    # Show previously answered questions (collapsed)
                    if tool.clarification_answered:
                        for q_idx, answer_summary in tool.clarification_answered:
                            output.append("\n")
                            output.append(f"       {continuation}     ", style="dim")
                            output.append("  ✓ ", style="green")
                            output.append(f"Q{q_idx}: ", style="dim")
                            output.append(answer_summary, style="dim green")

                    # Show current question prompt (if any)
                    if tool.clarification_prompt_lines:
                        # Limit lines to show
                        max_prompt_lines = 18
                        prompt_lines = tool.clarification_prompt_lines
                        total_lines = len(prompt_lines)
                        truncated = False
                        hidden_count = 0

                        if total_lines > max_prompt_lines:
                            truncated = True
                            tool.clarification_truncated = True
                            hidden_count = total_lines - max_prompt_lines + 1
                            lines_to_render = prompt_lines[:max_prompt_lines - 2]
                        else:
                            tool.clarification_truncated = False
                            lines_to_render = prompt_lines

                        # Draw box around current question
                        box_width = max(len(line) for line in prompt_lines) + 4
                        box_width = min(box_width, 60)  # Cap width
                        output.append("\n")
                        output.append(f"       {continuation}     ┌" + "─" * box_width + "┐", style="dim")

                        for prompt_line in lines_to_render:
                            output.append("\n")
                            display_line = prompt_line[:box_width - 2] if len(prompt_line) > box_width - 2 else prompt_line
                            padding = box_width - len(display_line) - 2
                            output.append(f"       {continuation}     │ ", style="dim")
                            output.append(display_line)
                            output.append(" " * padding + " │", style="dim")

                        # Show truncation indicator if needed
                        if truncated:
                            output.append("\n")
                            truncation_msg = f"[...{hidden_count} more - 'v' to view...]"
                            padding = box_width - len(truncation_msg) - 2
                            output.append(f"       {continuation}     │ ", style="dim")
                            output.append(truncation_msg, style="dim italic cyan")
                            output.append(" " * padding + " │", style="dim")
                            last_line = prompt_lines[-1]
                            output.append("\n")
                            display_line = last_line[:box_width - 2] if len(last_line) > box_width - 2 else last_line
                            padding = box_width - len(display_line) - 2
                            output.append(f"       {continuation}     │ ", style="dim")
                            output.append(display_line, style="cyan")
                            output.append(" " * padding + " │", style="dim")

                        output.append("\n")
                        output.append(f"       {continuation}     └" + "─" * box_width + "┘", style="dim")
        elif self._spinner_active:
            # Spinner active but no tools yet
            if lines_to_show:
                output.append("\n")
            frame = self.SPINNER_FRAMES[self._spinner_index]
            output.append(f"Model> {frame} ", style="bold cyan")
            output.append("thinking...", style="dim italic")

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
