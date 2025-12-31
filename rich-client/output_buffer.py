"""Output buffer for the scrolling output panel.

Manages a ring buffer of output lines for display in the scrolling
region of the TUI.
"""

import os
import tempfile
import textwrap
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    trace_path = os.environ.get(
        'JAATO_TRACE_LOG',
        os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
    )
    if trace_path:
        try:
            with open(trace_path, "a") as f:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] [OutputBuffer] {msg}\n")
                f.flush()
        except (IOError, OSError):
            pass

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
class ToolBlock:
    """A block of completed tools stored inline in the output buffer.

    This allows tool blocks to be rendered dynamically with expand/collapse
    state while maintaining their position in the output flow.
    """
    tools: List['ActiveToolCall']  # The tools in this block
    expanded: bool = True  # Whether the block is in expanded view
    selected_index: Optional[int] = None  # Which tool is selected (when navigating)


@dataclass
class ActiveToolCall:
    """Represents an actively executing or completed tool call."""
    name: str
    args_summary: str  # Truncated string representation of args
    call_id: Optional[str] = None  # Unique ID for correlating start/end of same tool call
    completed: bool = False  # True when tool execution finished
    success: bool = True  # Whether the tool succeeded (only valid when completed)
    duration_seconds: Optional[float] = None  # Execution time (only valid when completed)
    error_message: Optional[str] = None  # Error message if tool failed
    # Permission tracking
    permission_state: Optional[str] = None  # None, "pending", "granted", "denied"
    permission_method: Optional[str] = None  # "yes", "always", "once", "never", "whitelist", "blacklist"
    permission_prompt_lines: Optional[List[str]] = None  # Expanded prompt while pending
    permission_truncated: bool = False  # True if prompt is truncated
    permission_format_hint: Optional[str] = None  # "diff" for colored diff display
    # Clarification tracking (per-question progressive display)
    clarification_state: Optional[str] = None  # None, "pending", "resolved"
    clarification_prompt_lines: Optional[List[str]] = None  # Current question lines
    clarification_truncated: bool = False  # True if prompt is truncated
    clarification_current_question: int = 0  # Current question index (1-based)
    clarification_total_questions: int = 0  # Total number of questions
    clarification_answered: Optional[List[Tuple[int, str]]] = None  # List of (question_index, answer_summary)
    # Live output tracking (tail -f style preview)
    output_lines: Optional[List[str]] = None  # Rolling buffer of recent output lines
    output_max_lines: int = 30  # Max lines to keep in buffer
    output_display_lines: int = 5  # Max lines to show at once when expanded
    output_scroll_offset: int = 0  # Scroll position (0 = show most recent lines)
    # Per-tool expand state for navigation
    expanded: bool = False  # Whether this tool's output is expanded


class OutputBuffer:
    """Manages output lines for the scrolling panel.

    Stores output in a ring buffer and renders to Rich Text
    for display in the output panel.
    """

    # Spinner animation frames
    SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    # Type alias for items that can be stored in the line buffer
    LineItem = Union[OutputLine, ToolBlock]

    def __init__(self, max_lines: int = 1000, agent_type: str = "main"):
        """Initialize the output buffer.

        Args:
            max_lines: Maximum number of lines to retain.
            agent_type: Type of agent ("main" or "subagent") for label display.
        """
        self._lines: deque[Union[OutputLine, ToolBlock]] = deque(maxlen=max_lines)
        self._current_block: Optional[Tuple[str, List[str], bool]] = None
        self._measure_console: Optional[Console] = None
        self._console_width: int = 80
        self._visible_height: int = 20  # Last known visible height for auto-scroll
        self._last_source: Optional[str] = None  # Track source for turn detection
        self._last_turn_source: Optional[str] = None  # Track user/model turns (ignores system/tool)
        self._scroll_offset: int = 0  # Lines scrolled up from bottom (0 = at bottom)
        self._spinner_active: bool = False
        self._spinner_index: int = 0
        self._active_tools: List[ActiveToolCall] = []  # Currently executing tools
        self._tools_expanded: bool = False  # Toggle between collapsed/expanded tool view
        self._rendering: bool = False  # Guard against flushes during render
        self._agent_type: str = agent_type  # "main" or "subagent" for user label
        # Tool navigation state
        self._tool_nav_active: bool = False  # True when navigating tools
        self._selected_block_index: Optional[int] = None  # Which ToolBlock is selected
        self._selected_tool_index: Optional[int] = None  # Which tool within selected block
        # Keybinding config for dynamic UI hints
        self._keybinding_config: Optional[Any] = None

    def set_width(self, width: int) -> None:
        """Set the console width for measuring line wrapping.

        Args:
            width: Console width in characters.
        """
        self._console_width = width
        self._measure_console = Console(width=width, force_terminal=True)

    def set_keybinding_config(self, config: Any) -> None:
        """Set the keybinding config for dynamic UI hints.

        Args:
            config: KeybindingConfig instance from keybindings module.
        """
        self._keybinding_config = config

    def _format_key_hint(self, action: str) -> str:
        """Format a keybinding for display in UI hints.

        Args:
            action: The action name (e.g., "tool_expand", "nav_up")

        Returns:
            Human-readable key representation (e.g., "→", "↑", "Ctrl+N")
        """
        if not self._keybinding_config:
            # Fallback defaults if no config
            defaults = {
                "nav_up": "↑", "nav_down": "↓",
                "tool_expand": "→", "tool_collapse": "←", "tool_exit": "Esc",
                "tool_output_up": "↑", "tool_output_down": "↓",
                "tool_nav_enter": "Ctrl+N", "toggle_tools": "Ctrl+T",
            }
            return defaults.get(action, action)

        try:
            key = getattr(self._keybinding_config, action, None)
            if key is None:
                return action

            # Convert key to human-readable format
            if isinstance(key, list):
                key = " ".join(key)

            # Map common keys to symbols/readable names
            key_map = {
                "up": "↑", "down": "↓", "left": "←", "right": "→",
                "escape": "Esc", "enter": "Enter", "space": "Space",
                "pageup": "PgUp", "pagedown": "PgDn",
                "home": "Home", "end": "End", "tab": "Tab",
            }

            # Handle control keys
            if key.startswith("c-"):
                letter = key[2:].upper()
                return f"Ctrl+{letter}"

            return key_map.get(key, key)
        except Exception:
            return action

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
                # Header line takes full width, actual text on next line
                return 2  # Header + first content line (minimum)
            rendered.append(text)
        elif source == "user":
            if is_turn_start:
                return 2  # Header + first content line
            rendered.append(text)
        elif source == "system":
            rendered.append(text)
        elif source == "enrichment":
            # Enrichment notifications are pre-formatted single lines
            # after _flush_current_block splits by newline
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

        # If this is new user input, exit navigation mode and finalize any remaining tools
        if source == "user" and mode == "write":
            self._exit_navigation_mode()
            # Finalize any remaining active tools
            if self._active_tools:
                self._finalize_completed_tools()

        # If this is new model text and there are completed tools, finalize the tree first
        # This ensures the tool tree appears BEFORE the new response, not after
        if source == "model" and mode == "write" and self._active_tools:
            all_completed = all(tool.completed for tool in self._active_tools)
            any_pending = any(
                tool.permission_state == "pending" or tool.clarification_state == "pending"
                for tool in self._active_tools
            )
            if all_completed and not any_pending:
                self.finalize_tool_tree()

        if mode == "write":
            # Start a new block
            self._flush_current_block()
            # Only show header when switching between user and model turns
            # (not for every model response within same agentic loop)
            is_new_turn = False
            if source in ("user", "model"):
                is_new_turn = (self._last_turn_source != source)
                self._last_turn_source = source
            self._current_block = (source, [text], is_new_turn)
        elif mode == "append":
            # Append mode (streaming)
            if self._current_block and self._current_block[0] == source:
                # Append to existing block from same source
                self._current_block[1].append(text)
            else:
                # First streaming chunk or source changed - create new block
                self._flush_current_block()
                is_new_turn = False
                if source in ("user", "model"):
                    is_new_turn = (self._last_turn_source != source)
                    self._last_turn_source = source
                self._current_block = (source, [text], is_new_turn)
        elif mode == "flush":
            # Flush mode: process pending output but don't add any content
            # Used to synchronize output before UI events like tool tree rendering
            self._flush_current_block()
        else:
            # Standalone line (unknown mode)
            self._flush_current_block()
            is_new_turn = self._last_source != source
            for i, line in enumerate(text.split('\n')):
                self._add_line(source, line, "line", is_turn_start=(i == 0 and is_new_turn))
            self._last_source = source

    def _flush_current_block(self) -> None:
        """Flush the current block to lines."""
        # Guard: don't flush during render cycles (prompt_toolkit calls render frequently)
        if self._rendering:
            return
        if self._current_block:
            source, parts, is_new_turn = self._current_block
            # Concatenate streaming chunks directly (no separator)
            # Then split by newlines for display
            full_text = ''.join(parts)
            lines = full_text.split('\n')
            for i, line in enumerate(lines):
                # Only first line of a new turn gets the prefix
                self._add_line(source, line, "line", is_turn_start=(i == 0 and is_new_turn))
            self._last_source = source
            self._current_block = None

    def _get_current_block_lines(self) -> List[OutputLine]:
        """Get lines from current block without flushing it.

        This allows render() to display streaming content without
        breaking the append chain.
        """
        if not self._current_block:
            return []

        source, parts, is_new_turn = self._current_block
        # Concatenate streaming chunks directly (no separator)
        full_text = ''.join(parts)
        lines_text = full_text.split('\n')

        result = []
        for i, line_text in enumerate(lines_text):
            is_turn_start_line = (i == 0 and is_new_turn)
            display_lines = self._measure_display_lines(source, line_text, is_turn_start_line)
            result.append(OutputLine(
                source=source,
                text=line_text,
                style="line",
                display_lines=display_lines,
                is_turn_start=is_turn_start_line
            ))
        return result

    def add_system_message(self, message: str, style: str = "dim") -> None:
        """Add a system message to the buffer.

        Args:
            message: The system message.
            style: Rich style for the message.
        """
        self._flush_current_block()
        # Handle multi-line messages
        for line in message.split('\n'):
            self._add_line("system", line, style)

    def update_last_system_message(self, message: str, style: str = "dim") -> bool:
        """Update the last system message in the buffer.

        Used for progressive updates like init progress that show "..." then "OK".

        Args:
            message: The new message text.
            style: Rich style for the message.

        Returns:
            True if a system message was found and updated, False otherwise.
        """
        # Search backwards for the last system message
        for i in range(len(self._lines) - 1, -1, -1):
            if self._lines[i].source == "system":
                # Update the line
                display_lines = self._measure_display_lines("system", message, False)
                self._lines[i] = OutputLine(
                    source="system",
                    text=message,
                    style=style,
                    display_lines=display_lines,
                    is_turn_start=False
                )
                return True
        return False

    def clear(self) -> None:
        """Clear all output."""
        self._lines.clear()
        self._current_block = None
        self._last_source = None
        self._last_turn_source = None
        self._scroll_offset = 0
        self._spinner_active = False
        self._active_tools.clear()
        self._tool_nav_active = False
        self._selected_block_index = None
        self._selected_tool_index = None

    def start_spinner(self) -> None:
        """Start showing spinner in the output."""
        self._spinner_active = True
        self._spinner_index = 0

    def stop_spinner(self) -> None:
        """Stop showing spinner and finalize tool tree if complete."""
        self._spinner_active = False
        # Flush any pending streaming text before finalizing the turn
        self._flush_current_block()
        # Convert tool tree to scrollable lines if all tools are done
        self.finalize_tool_tree()

    def advance_spinner(self) -> None:
        """Advance spinner to next frame."""
        if self._spinner_active:
            self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)

    @property
    def spinner_active(self) -> bool:
        """Check if spinner is currently active."""
        return self._spinner_active

    def add_active_tool(self, tool_name: str, tool_args: dict,
                        call_id: Optional[str] = None) -> None:
        """Add a tool to the active tools list.

        Args:
            tool_name: Name of the tool being executed.
            tool_args: Arguments passed to the tool.
            call_id: Unique identifier for this tool call (for correlation).
        """
        # If all existing tools are completed, finalize them as a ToolBlock
        # This ensures proper ordering of text and tool output
        if self._active_tools and all(t.completed for t in self._active_tools):
            self._finalize_completed_tools()

        # Create a summary of args (truncated for display)
        args_str = str(tool_args)
        if len(args_str) > 60:
            args_str = args_str[:57] + "..."

        # Don't add duplicates - check by call_id if provided, otherwise by name
        for tool in self._active_tools:
            if call_id and tool.call_id == call_id:
                return
            # Fall back to name-based check only if no call_id provided
            if not call_id and tool.name == tool_name and not tool.call_id:
                return

        self._active_tools.append(ActiveToolCall(
            name=tool_name, args_summary=args_str, call_id=call_id
        ))

    def mark_tool_completed(self, tool_name: str, success: bool = True,
                            duration_seconds: Optional[float] = None,
                            error_message: Optional[str] = None,
                            call_id: Optional[str] = None) -> None:
        """Mark a tool as completed (keeps it in the tree with completion status).

        Args:
            tool_name: Name of the tool that finished.
            success: Whether the tool execution succeeded.
            duration_seconds: How long the tool took to execute.
            error_message: Error message if the tool failed.
            call_id: Unique identifier for this tool call (for correlation).
        """
        for tool in self._active_tools:
            # Match by call_id if provided, otherwise by name (for backwards compatibility)
            if call_id:
                if tool.call_id == call_id and not tool.completed:
                    tool.completed = True
                    tool.success = success
                    tool.duration_seconds = duration_seconds
                    tool.error_message = error_message
                    return
            elif tool.name == tool_name and not tool.completed and not tool.call_id:
                tool.completed = True
                tool.success = success
                tool.duration_seconds = duration_seconds
                tool.error_message = error_message
                return

    def append_tool_output(self, call_id: str, chunk: str) -> None:
        """Append output chunk to a running tool's output buffer.

        Used for live "tail -f" style output preview in the tool tree.

        Args:
            call_id: Unique identifier for the tool call.
            chunk: Output text chunk (may contain newlines).
        """
        for tool in self._active_tools:
            if tool.call_id == call_id and not tool.completed:
                # Initialize output_lines if needed
                if tool.output_lines is None:
                    tool.output_lines = []

                # Split chunk by newlines and add to buffer
                lines = chunk.splitlines()
                for line in lines:
                    # Skip empty lines from split
                    if line or chunk == "\n":
                        tool.output_lines.append(line)

                # Trim to max size (keep most recent lines)
                if len(tool.output_lines) > tool.output_max_lines:
                    tool.output_lines = tool.output_lines[-tool.output_max_lines:]
                return

    def remove_active_tool(self, tool_name: str) -> None:
        """Remove a tool from the active tools list (legacy, now marks as completed).

        Args:
            tool_name: Name of the tool that finished.
        """
        # Instead of removing, mark as completed to keep the tree visible
        self.mark_tool_completed(tool_name)

    def clear_active_tools(self) -> None:
        """Clear all active tools and reset navigation state."""
        self._active_tools.clear()
        self._tool_nav_active = False
        self._selected_tool_index = None

    def toggle_tools_expanded(self) -> bool:
        """Toggle between collapsed and expanded tool view.

        Returns:
            True if now expanded, False if now collapsed.
        """
        self._tools_expanded = not self._tools_expanded
        return self._tools_expanded

    @property
    def tools_expanded(self) -> bool:
        """Check if tool view is currently expanded."""
        return self._tools_expanded

    def _get_approval_indicator(self, method: str) -> Optional[str]:
        """Convert a permission method to a short display indicator.

        Args:
            method: The permission method (whitelist, session_whitelist, etc.)

        Returns:
            Short indicator string like "[pre]", "[once]", "[session]", or None.
        """
        # Pre-approved via config whitelist
        if method in ("whitelist", "policy"):
            return "[pre]"
        # Approved for entire session
        if method in ("session_whitelist", "always"):
            return "[session]"
        # User chose to approve all future requests
        if method == "allow_all":
            return "[all]"
        # Single approval by user
        if method in ("user_approved", "yes", "once"):
            return "[once]"
        # Other methods - no indicator needed
        return None

    # Tool navigation methods
    def _get_tool_blocks(self) -> List[Tuple[int, ToolBlock]]:
        """Get all ToolBlocks in the line buffer with their indices.

        Returns:
            List of (index, ToolBlock) tuples.
        """
        blocks = []
        for i, item in enumerate(self._lines):
            if isinstance(item, ToolBlock):
                blocks.append((i, item))
        return blocks

    def _get_selected_block(self) -> Optional[ToolBlock]:
        """Get the currently selected ToolBlock."""
        if self._selected_block_index is None:
            return None
        blocks = self._get_tool_blocks()
        if 0 <= self._selected_block_index < len(blocks):
            return blocks[self._selected_block_index][1]
        return None

    def _exit_navigation_mode(self) -> None:
        """Exit navigation mode and clear all selection state."""
        self._tool_nav_active = False
        self._selected_block_index = None
        self._selected_tool_index = None
        # Clear selection from all blocks
        for _, block in self._get_tool_blocks():
            block.selected_index = None

    def enter_tool_navigation(self) -> bool:
        """Enter tool navigation mode.

        Returns:
            True if entered successfully, False if no tools available.
        """
        # First check active tools (currently executing)
        if self._active_tools:
            self._tool_nav_active = True
            self._selected_block_index = None  # No block, using active tools
            self._selected_tool_index = 0
            self._tools_expanded = True
            return True

        # Then check tool blocks in the buffer
        blocks = self._get_tool_blocks()
        if not blocks:
            return False

        # Select last tool in last block (most recent) for more intuitive navigation
        self._tool_nav_active = True
        last_block_idx = len(blocks) - 1
        self._selected_block_index = last_block_idx
        last_block = blocks[last_block_idx][1]
        last_tool_idx = len(last_block.tools) - 1
        self._selected_tool_index = last_tool_idx
        last_block.selected_index = last_tool_idx
        last_block.expanded = True
        self._tools_expanded = True  # Ensure tools are in expanded mode
        # Scroll to bottom to show the selected block
        self._scroll_offset = 0
        return True

    def exit_tool_navigation(self) -> None:
        """Exit tool navigation mode."""
        self._exit_navigation_mode()

    @property
    def tool_nav_active(self) -> bool:
        """Check if tool navigation is active."""
        return self._tool_nav_active

    @property
    def selected_tool_index(self) -> Optional[int]:
        """Get currently selected tool index."""
        return self._selected_tool_index

    def _get_all_tools_flat(self) -> List[Tuple[Optional[int], int, ActiveToolCall]]:
        """Get all tools flattened with (block_index, tool_index, tool).

        Returns:
            List of (block_index, tool_index, tool) where block_index is None for active tools.
        """
        result = []
        # Active tools first (block_index = None)
        for i, tool in enumerate(self._active_tools):
            result.append((None, i, tool))
        # Then tools from blocks
        for block_idx, (_, block) in enumerate(self._get_tool_blocks()):
            for tool_idx, tool in enumerate(block.tools):
                result.append((block_idx, tool_idx, tool))
        return result

    def select_next_tool(self) -> bool:
        """Select next tool in list (across all blocks).

        Returns:
            True if selection changed, False if already at end.
        """
        if not self._tool_nav_active:
            return False

        all_tools = self._get_all_tools_flat()
        if not all_tools:
            return False

        # Find current position
        current_flat_idx = None
        for i, (block_idx, tool_idx, _) in enumerate(all_tools):
            if block_idx == self._selected_block_index and tool_idx == self._selected_tool_index:
                current_flat_idx = i
                break

        if current_flat_idx is None:
            # No selection, select first
            block_idx, tool_idx, _ = all_tools[0]
            self._select_tool(block_idx, tool_idx)
            return True

        if current_flat_idx < len(all_tools) - 1:
            # Move to next
            block_idx, tool_idx, _ = all_tools[current_flat_idx + 1]
            self._select_tool(block_idx, tool_idx)
            return True

        return False  # Already at end

    def select_prev_tool(self) -> bool:
        """Select previous tool in list (across all blocks).

        Returns:
            True if selection changed, False if already at start.
        """
        if not self._tool_nav_active:
            return False

        all_tools = self._get_all_tools_flat()
        if not all_tools:
            return False

        # Find current position
        current_flat_idx = None
        for i, (block_idx, tool_idx, _) in enumerate(all_tools):
            if block_idx == self._selected_block_index and tool_idx == self._selected_tool_index:
                current_flat_idx = i
                break

        if current_flat_idx is None:
            # No selection, select first
            block_idx, tool_idx, _ = all_tools[0]
            self._select_tool(block_idx, tool_idx)
            return True

        if current_flat_idx > 0:
            # Move to previous
            block_idx, tool_idx, _ = all_tools[current_flat_idx - 1]
            self._select_tool(block_idx, tool_idx)
            return True

        return False  # Already at start

    def _select_tool(self, block_idx: Optional[int], tool_idx: int) -> None:
        """Select a specific tool."""
        # Clear selection from old block
        old_block = self._get_selected_block()
        if old_block:
            old_block.selected_index = None

        self._selected_block_index = block_idx
        self._selected_tool_index = tool_idx

        # Set selection on new block
        if block_idx is not None:
            blocks = self._get_tool_blocks()
            if 0 <= block_idx < len(blocks):
                new_block = blocks[block_idx][1]
                new_block.selected_index = tool_idx
                new_block.expanded = True

        # Auto-scroll to keep selected tool visible
        self._scroll_to_selected_tool()

    def _scroll_to_selected_tool(self) -> None:
        """Scroll the output panel to keep the selected tool visible."""
        if not self._tool_nav_active:
            return

        # For active tools (block_idx is None), they're always at the bottom
        if self._selected_block_index is None:
            self._scroll_offset = 0
            return

        # For ToolBlocks, calculate the position and scroll if needed
        blocks = self._get_tool_blocks()
        if not blocks or self._selected_block_index >= len(blocks):
            return

        # Calculate display lines up to the selected block
        display_line = 0
        for i, item in enumerate(self._lines):
            if isinstance(item, ToolBlock):
                # Find which block index this is
                block_list_idx = None
                for bi, (line_idx, _) in enumerate(blocks):
                    if line_idx == i:
                        block_list_idx = bi
                        break

                if block_list_idx == self._selected_block_index:
                    # Found the selected block, add lines for tools before selected
                    # Block header (separator + header line)
                    display_line += 2
                    # Add lines for each tool before the selected one
                    block = blocks[block_list_idx][1]
                    for ti in range(self._selected_tool_index or 0):
                        display_line += 1  # Tool line
                        tool = block.tools[ti]
                        if tool.expanded and tool.output_lines:
                            display_line += min(len(tool.output_lines), tool.output_display_lines)
                            if len(tool.output_lines) > tool.output_display_lines:
                                display_line += 2  # Scroll indicators
                    break

            display_line += self._get_item_display_lines(item)

        # Calculate total display lines
        total_lines = sum(self._get_item_display_lines(item) for item in self._lines)

        # Calculate the scroll offset needed to show this position
        # scroll_offset is "lines from bottom", so higher = older content shown
        # We want the selected tool roughly in the middle of the visible area
        target_from_bottom = total_lines - display_line
        margin = self._visible_height // 3  # Keep some margin

        # If tool would be above visible area, scroll up (increase offset)
        if target_from_bottom > self._scroll_offset + self._visible_height - margin:
            self._scroll_offset = max(0, target_from_bottom - self._visible_height + margin)
        # If tool would be below visible area, scroll down (decrease offset)
        elif target_from_bottom < self._scroll_offset + margin:
            self._scroll_offset = max(0, target_from_bottom - margin)

    def toggle_selected_tool_expanded(self) -> bool:
        """Toggle expand/collapse for selected tool's output.

        Returns:
            New expanded state of the tool, False if no selection.
        """
        tool = self.get_selected_tool()
        if tool is None:
            return False
        tool.expanded = not tool.expanded
        return tool.expanded

    def expand_selected_tool(self) -> bool:
        """Expand the selected tool's output.

        Returns:
            True if tool was expanded (or already expanded), False if no selection.
        """
        tool = self.get_selected_tool()
        if tool is None:
            return False
        tool.expanded = True
        return True

    def collapse_selected_tool(self) -> bool:
        """Collapse the selected tool's output.

        Returns:
            True if tool was collapsed (or already collapsed), False if no selection.
        """
        tool = self.get_selected_tool()
        if tool is None:
            return False
        tool.expanded = False
        return True

    def get_selected_tool(self) -> Optional[ActiveToolCall]:
        """Get the currently selected tool."""
        if self._selected_tool_index is None:
            return None

        # Check active tools first
        if self._selected_block_index is None:
            if 0 <= self._selected_tool_index < len(self._active_tools):
                return self._active_tools[self._selected_tool_index]
            return None

        # Check tool blocks
        block = self._get_selected_block()
        if block and 0 <= self._selected_tool_index < len(block.tools):
            return block.tools[self._selected_tool_index]
        return None

    def scroll_selected_tool_up(self) -> bool:
        """Scroll up within the selected tool's output.

        Returns:
            True if scroll position changed, False if at top or no tool selected.
        """
        tool = self.get_selected_tool()
        if not tool or not tool.expanded or not tool.output_lines:
            return False
        # Scroll offset is from the end, so scrolling "up" means increasing offset
        max_offset = max(0, len(tool.output_lines) - tool.output_display_lines)
        if tool.output_scroll_offset < max_offset:
            tool.output_scroll_offset += 1
            return True
        return False

    def scroll_selected_tool_down(self) -> bool:
        """Scroll down within the selected tool's output.

        Returns:
            True if scroll position changed, False if at bottom or no tool selected.
        """
        tool = self.get_selected_tool()
        if not tool or not tool.expanded or not tool.output_lines:
            return False
        if tool.output_scroll_offset > 0:
            tool.output_scroll_offset -= 1
            return True
        return False

    def finalize_tool_tree(self) -> None:
        """Mark turn as complete and finalize active tools as a ToolBlock.

        Called when a turn is complete. Active tools are stored as a ToolBlock
        that can be navigated and expanded.
        """
        # Flush any pending streaming text first to ensure proper ordering
        self._flush_current_block()

        # Finalize any remaining active tools as a ToolBlock
        if self._active_tools:
            self._finalize_completed_tools()

    def _finalize_completed_tools(self) -> None:
        """Store completed tools as a ToolBlock in the line buffer.

        Called internally when a new tool is added while previous tools are
        all completed. Stores a ToolBlock that can be navigated and expanded.
        """
        if not self._active_tools:
            return

        # Create a copy of the tools for the ToolBlock
        import copy
        tools_copy = copy.deepcopy(self._active_tools)

        # Create ToolBlock with the completed tools
        tool_block = ToolBlock(
            tools=tools_copy,
            expanded=self._tools_expanded,
            selected_index=None
        )

        # Add separator and the tool block to the buffer
        self._add_line("system", "", "")
        self._lines.append(tool_block)
        self._add_line("system", "", "")

        # Clear active tools so they don't render separately anymore
        self._active_tools.clear()

    @property
    def active_tools(self) -> List[ActiveToolCall]:
        """Get list of currently active tools."""
        return list(self._active_tools)

    def set_tool_permission_pending(
        self,
        tool_name: str,
        prompt_lines: List[str],
        format_hint: Optional[str] = None
    ) -> None:
        """Mark a tool as awaiting permission with the prompt to display.

        Args:
            tool_name: Name of the tool awaiting permission.
            prompt_lines: Lines of the permission prompt to display.
            format_hint: Optional hint for display format ("diff" for colored diff).
        """
        _trace(f"set_tool_permission_pending: looking for tool={tool_name}")
        _trace(f"set_tool_permission_pending: active_tools={[(t.name, t.completed) for t in self._active_tools]}")
        for tool in self._active_tools:
            if tool.name == tool_name and not tool.completed:
                tool.permission_state = "pending"
                tool.permission_prompt_lines = prompt_lines
                tool.permission_format_hint = format_hint
                # Scroll to bottom to show the prompt
                self._scroll_offset = 0
                _trace(f"set_tool_permission_pending: FOUND and set pending for {tool_name}")
                return
        _trace(f"set_tool_permission_pending: NO MATCH for {tool_name}")

    def set_tool_permission_resolved(self, tool_name: str, granted: bool,
                                      method: str) -> None:
        """Mark a tool's permission as resolved.

        Args:
            tool_name: Name of the tool.
            granted: Whether permission was granted.
            method: How permission was resolved (yes, always, once, never, whitelist, etc.)
        """
        for tool in self._active_tools:
            # Match the tool with pending permission state (only one at a time due to blocking)
            if tool.name == tool_name and tool.permission_state == "pending":
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
        # Calculate total display lines (handles both OutputLine and ToolBlock)
        total_display_lines = sum(self._get_item_display_lines(item) for item in self._lines)
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

    def _calculate_tool_tree_height(self) -> int:
        """Calculate the approximate height of the tool tree in display lines.

        This is used to reserve space when selecting which stored lines to display,
        ensuring the tool tree doesn't push content off the visible area.

        Note: This only calculates height for active tools. ToolBlocks in the
        line buffer are accounted for via _get_item_display_lines().

        Returns:
            Number of display lines the tool tree will occupy.
        """
        if not self._active_tools and not self._spinner_active:
            return 0

        height = 0

        if self._active_tools:
            if self._tools_expanded:
                # Expanded view: header + each tool on its own line
                height += 1  # Header line

                for tool in self._active_tools:
                    height += 1  # Tool line

                    # Output preview lines (persists after tool completes)
                    if tool.output_lines:
                        height += len(tool.output_lines)

                    # Permission denied message
                    if tool.permission_state == "denied" and tool.permission_method:
                        height += 1
                    # Error message (if failed and has error, but not permission denied)
                    elif tool.completed and not tool.success and tool.error_message:
                        height += 1
            else:
                # Collapsed view: just one summary line
                height += 1

            # Check for pending prompts (same logic for both views)
            for tool in self._active_tools:
                # Permission prompt (if pending)
                if tool.permission_state == "pending" and tool.permission_prompt_lines:
                    height += 1  # "Permission required" header

                    # Box calculation
                    prompt_lines = tool.permission_prompt_lines
                    max_prompt_lines = 18
                    max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
                    box_width = min(max_box_width, max(len(line) for line in prompt_lines) + 4)
                    content_width = box_width - 4

                    # First count ALL wrapped lines to decide truncation (matches render logic)
                    total_wrapped_lines = 0
                    for prompt_line in prompt_lines:
                        if len(prompt_line) > content_width:
                            wrapped = textwrap.wrap(prompt_line, width=content_width, break_long_words=True)
                            total_wrapped_lines += len(wrapped) if wrapped else 1
                        else:
                            total_wrapped_lines += 1

                    if total_wrapped_lines > max_prompt_lines:
                        # Truncation triggered - render shows:
                        # - max_lines_before_truncation content lines
                        # - 1 truncation message
                        # - Last line (may wrap to multiple lines)
                        max_lines_before_truncation = max_prompt_lines - 3

                        # Calculate wrapped lines for last line
                        last_line = prompt_lines[-1]
                        if len(last_line) > content_width:
                            last_wrapped = textwrap.wrap(last_line, width=content_width, break_long_words=True)
                            last_line_count = len(last_wrapped) if last_wrapped else 1
                        else:
                            last_line_count = 1

                        rendered_lines = max_lines_before_truncation + 1 + last_line_count  # content + truncation + last
                    else:
                        # No truncation - show all wrapped lines
                        rendered_lines = total_wrapped_lines

                    height += 1  # top border
                    height += rendered_lines
                    height += 1  # bottom border

                # Clarification prompt (if pending)
                if tool.clarification_state == "pending":
                    height += 1  # header ("Clarification needed" or progress)

                    # Previously answered questions
                    if tool.clarification_answered:
                        height += len(tool.clarification_answered)

                    # Current question prompt box
                    if tool.clarification_prompt_lines:
                        prompt_lines = tool.clarification_prompt_lines
                        max_prompt_lines = 18
                        max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
                        box_width = min(max_box_width, max(len(line) for line in prompt_lines) + 4)
                        content_width = box_width - 4

                        # First count ALL wrapped lines to decide truncation (matches render logic)
                        total_wrapped_lines = 0
                        for prompt_line in prompt_lines:
                            if len(prompt_line) > content_width:
                                wrapped = textwrap.wrap(prompt_line, width=content_width, break_long_words=True)
                                total_wrapped_lines += len(wrapped) if wrapped else 1
                            else:
                                total_wrapped_lines += 1

                        if total_wrapped_lines > max_prompt_lines:
                            # Truncation triggered - render shows:
                            # - max_lines_before_truncation content lines
                            # - 1 truncation message
                            # - Last line (may wrap to multiple lines)
                            max_lines_before_truncation = max_prompt_lines - 3

                            # Calculate wrapped lines for last line
                            last_line = prompt_lines[-1]
                            if len(last_line) > content_width:
                                last_wrapped = textwrap.wrap(last_line, width=content_width, break_long_words=True)
                                last_line_count = len(last_wrapped) if last_wrapped else 1
                            else:
                                last_line_count = 1

                            rendered_lines = max_lines_before_truncation + 1 + last_line_count  # content + truncation + last
                        else:
                            # No truncation - show all wrapped lines
                            rendered_lines = total_wrapped_lines

                        height += 1  # top border
                        height += rendered_lines
                        height += 1  # bottom border

        elif self._spinner_active:
            # Spinner alone (no tools) - just 1 line
            height += 1

        return height

    def _get_item_display_lines(self, item: Union[OutputLine, ToolBlock]) -> int:
        """Get display line count for a line item (OutputLine or ToolBlock)."""
        if isinstance(item, OutputLine):
            return item.display_lines
        elif isinstance(item, ToolBlock):
            return self._calculate_tool_block_height(item)
        return 1

    def _calculate_tool_block_height(self, block: ToolBlock) -> int:
        """Calculate display height of a ToolBlock."""
        height = 1  # Separator line
        if block.expanded:
            height += 1  # Header
            for tool in block.tools:
                height += 1  # Tool line
                # Output preview if tool has output and is expanded
                if tool.expanded and tool.output_lines:
                    display_count = min(len(tool.output_lines), tool.output_display_lines)
                    height += display_count
                    # Scroll indicators
                    if len(tool.output_lines) > tool.output_display_lines:
                        height += 2
                # Error message
                if not tool.success and tool.error_message:
                    height += 1
        else:
            height += 1  # Collapsed summary line
        return height

    def _render_tool_block(self, block: ToolBlock, output: Text, wrap_width: int) -> None:
        """Render a ToolBlock inline in the output."""
        tool_count = len(block.tools)

        # Separator
        output.append("  ───", style="dim")

        # Navigation hint if in nav mode and this block is selected
        if self._tool_nav_active and block.selected_index is not None:
            all_tools = self._get_all_tools_flat()
            # Find position in all tools
            pos = 0
            for i, (bidx, tidx, _) in enumerate(all_tools):
                if bidx == self._selected_block_index and tidx == self._selected_tool_index:
                    pos = i + 1
                    break
            total = len(all_tools)
            # Dynamic hint based on selected tool's expanded state and actual keybindings
            selected_tool = block.tools[block.selected_index]
            nav_up = self._format_key_hint("nav_up")
            nav_down = self._format_key_hint("nav_down")
            expand_key = self._format_key_hint("tool_expand")
            collapse_key = self._format_key_hint("tool_collapse")
            exit_key = self._format_key_hint("tool_exit")
            has_output = selected_tool.output_lines and len(selected_tool.output_lines) > 0
            if selected_tool.expanded and has_output:
                # When expanded: arrows scroll output, left collapses
                output.append(f"  {nav_up}/{nav_down} scroll, {collapse_key} collapse, {exit_key} exit [{pos}/{total}]", style="dim")
            elif has_output:
                # When collapsed but has output: arrows navigate, right expands
                output.append(f"  {nav_up}/{nav_down} nav, {expand_key} expand, {exit_key} exit [{pos}/{total}]", style="dim")
            else:
                # No output: just navigation hints
                output.append(f"  {nav_up}/{nav_down} nav, {exit_key} exit [{pos}/{total}]", style="dim")

        output.append("\n")

        if block.expanded:
            # Expanded view - each tool on its own line
            output.append(f"  ▾ {tool_count} tool{'s' if tool_count != 1 else ''}:", style="dim")

            for i, tool in enumerate(block.tools):
                is_last = (i == len(block.tools) - 1)
                is_selected = (block.selected_index == i)
                connector = "└─" if is_last else "├─"

                status_icon = "✓" if tool.success else "✗"
                status_style = "green" if tool.success else "red"

                # Expand indicator for tool output (only if tool has output)
                if tool.output_lines:
                    expand_icon = "▾" if tool.expanded else "▸"
                else:
                    expand_icon = " "

                # Selection highlight
                row_style = "reverse" if is_selected else "dim"

                output.append("\n")
                output.append(f"  {expand_icon} {connector} ", style=row_style)
                output.append(tool.name, style=row_style)
                if tool.args_summary:
                    output.append(f"({tool.args_summary})", style=row_style)
                output.append(f" {status_icon}", style=status_style)

                # Approval indicator
                if tool.permission_state == "granted" and tool.permission_method:
                    indicator = self._get_approval_indicator(tool.permission_method)
                    if indicator:
                        output.append(f" {indicator}", style="dim cyan")

                # Duration
                if tool.duration_seconds is not None:
                    output.append(f" ({tool.duration_seconds:.1f}s)", style="dim")

                # Expanded output
                if tool.expanded and tool.output_lines:
                    continuation = "   " if is_last else "│  "
                    prefix = "    "
                    total_lines = len(tool.output_lines)
                    display_count = tool.output_display_lines

                    end_idx = total_lines - tool.output_scroll_offset
                    start_idx = max(0, end_idx - display_count)

                    lines_above = start_idx
                    lines_below = tool.output_scroll_offset

                    if lines_above > 0:
                        output.append("\n")
                        output.append(f"{prefix}{continuation}   ", style="dim")
                        scroll_up_key = self._format_key_hint("nav_up")
                        output.append(f"▲ {lines_above} more line{'s' if lines_above != 1 else ''} ({scroll_up_key} to scroll)", style="dim italic")

                    for output_line in tool.output_lines[start_idx:end_idx]:
                        output.append("\n")
                        output.append(f"{prefix}{continuation}   ", style="dim")
                        max_line_width = max(40, self._console_width - 20) if self._console_width > 60 else 40
                        if len(output_line) > max_line_width:
                            display_line = output_line[:max_line_width - 3] + "..."
                        else:
                            display_line = output_line
                        output.append(display_line, style="#87D7D7 italic")

                    if lines_below > 0:
                        output.append("\n")
                        output.append(f"{prefix}{continuation}   ", style="dim")
                        scroll_down_key = self._format_key_hint("nav_down")
                        output.append(f"▼ {lines_below} more line{'s' if lines_below != 1 else ''} ({scroll_down_key} to scroll)", style="dim italic")

                # Error message
                if not tool.success and tool.error_message:
                    output.append("\n")
                    continuation = "   " if is_last else "│  "
                    output.append(f"    {continuation}   ⚠ {tool.error_message}", style="red dim")
        else:
            # Collapsed view
            tool_summaries = []
            for tool in block.tools:
                status_icon = "✓" if tool.success else "✗"
                summary = f"{tool.name} {status_icon}"
                if tool.permission_state == "granted" and tool.permission_method:
                    indicator = self._get_approval_indicator(tool.permission_method)
                    if indicator:
                        summary += f" {indicator}"
                tool_summaries.append(summary)

            output.append(f"  ▸ {tool_count} tool{'s' if tool_count != 1 else ''}: ", style="dim")
            output.append(" ".join(tool_summaries), style="dim")

    def render(self, height: Optional[int] = None, width: Optional[int] = None) -> RenderableType:
        """Render the output buffer as Rich Text.

        Args:
            height: Optional height limit (in display lines).
            width: Optional width for calculating line wrapping.

        Returns:
            Rich renderable for the output panel.
        """
        # Set rendering guard to prevent flush during render cycle
        self._rendering = True
        try:
            return self._render_impl(height, width)
        finally:
            self._rendering = False

    def _render_impl(self, height: Optional[int] = None, width: Optional[int] = None) -> RenderableType:
        """Internal render implementation (called within rendering guard)."""
        # Get current block lines without flushing (preserves streaming state)
        current_block_lines = self._get_current_block_lines()

        # If buffer is empty but spinner is active, show only spinner
        if not self._lines and not current_block_lines:
            if self._spinner_active:
                output = Text()
                frame = self.SPINNER_FRAMES[self._spinner_index]
                output.append(f"  {frame} ", style="cyan")
                output.append("thinking...", style="dim italic")
                return output
            return Text("Waiting for output...", style="dim italic")

        # Update width if provided
        if width and width != self._console_width:
            self.set_width(width)

        # Store visible height for auto-scroll calculations
        if height:
            self._visible_height = height

        # Work backwards from the end, using stored display line counts
        # First skip _scroll_offset lines, then collect 'height' lines
        # Include current block lines (streaming content) at the end
        all_items: List[Union[OutputLine, ToolBlock]] = list(self._lines) + current_block_lines
        items_to_show: List[Union[OutputLine, ToolBlock]] = []

        if height:
            # Calculate how much space the active tool tree will take (including separator)
            tool_tree_height = self._calculate_tool_tree_height()
            if tool_tree_height > 0 and all_items:
                # Separator adds: blank line (\n\n) + separator text + \n = 2 visual lines
                # Add +3 safety margin for edge cases (line wrapping, alignment)
                tool_tree_height += 5  # 2 for separator + 3 safety margin

            # Adjust available height for stored lines
            available_for_lines = max(1, height - tool_tree_height)

            # Calculate total display lines (accounting for ToolBlocks)
            total_display_lines = sum(self._get_item_display_lines(item) for item in all_items)

            # Find the end position (bottom of visible window)
            # scroll_offset=0 means show the most recent content
            # scroll_offset>0 means we've scrolled up, showing older content
            end_display_line = total_display_lines - self._scroll_offset
            start_display_line = max(0, end_display_line - available_for_lines)

            # Collect items that fall within the visible range
            current_display_line = 0
            for item in all_items:
                item_height = self._get_item_display_lines(item)
                line_end = current_display_line + item_height
                # Include item if it overlaps with visible range
                if line_end > start_display_line and current_display_line < end_display_line:
                    items_to_show.append(item)
                current_display_line = line_end
                # Stop if we've passed the visible range
                if current_display_line >= end_display_line:
                    break
        else:
            items_to_show = all_items

        # Build output text with wrapping
        output = Text()

        # Calculate available width for content (accounting for prefixes)
        wrap_width = self._console_width if self._console_width > 20 else 80

        # Define wrap_text helper before the loops
        def wrap_text(text: str, prefix_width: int = 0) -> List[str]:
            """Wrap text to console width, accounting for prefix.

            Handles multi-line text by splitting on newlines first,
            then wrapping each line individually.
            """
            available = max(20, wrap_width - prefix_width)

            # Handle literal \n in text (escaped newlines) - convert to actual newlines
            text = text.replace('\\n', '\n')

            # Split on actual newlines first, then wrap each line
            result = []
            for paragraph in text.split('\n'):
                if not paragraph.strip():
                    # Preserve empty lines (paragraph breaks)
                    result.append('')
                elif len(paragraph) <= available:
                    result.append(paragraph)
                else:
                    # Use textwrap for clean word-based wrapping
                    wrapped = textwrap.wrap(paragraph, width=available, break_long_words=True, break_on_hyphens=False)
                    result.extend(wrapped)
            return result if result else ['']

        # Render items (OutputLines and ToolBlocks)
        for i, item in enumerate(items_to_show):
            if i > 0:
                output.append("\n")

            # Handle ToolBlocks specially
            if isinstance(item, ToolBlock):
                self._render_tool_block(item, output, wrap_width)
                continue

            # For OutputLine items, render based on source
            line = item
            if line.source == "system":
                # System messages use their style directly
                wrapped = wrap_text(line.text)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    output.append(wrapped_line, style=line.style)
            elif line.source == "user":
                # User input - use header line for turn start
                if line.is_turn_start:
                    # Add blank line before header for visual separation (if not first line)
                    if i > 0:
                        output.append("\n")
                    # Render header line: ── User/Parent ───────────────────
                    user_label = "Parent" if self._agent_type == "subagent" else "User"
                    header_prefix = f"── {user_label} "
                    remaining = max(0, wrap_width - len(header_prefix))
                    output.append(header_prefix, style="bold green")
                    output.append("─" * remaining, style="dim green")
                    output.append("\n")
                    # Then render the text content
                    wrapped = wrap_text(line.text, 0)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        output.append(wrapped_line)
                else:
                    # Non-turn-start - just render text
                    wrapped = wrap_text(line.text, 0)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        output.append(wrapped_line)
            elif line.source == "model":
                # Model output - use header line for turn start
                if line.is_turn_start:
                    # Add blank line before header for visual separation (if not first line)
                    if i > 0:
                        output.append("\n")
                    # Render header line: ── Model ─────────────────
                    header_prefix = "── Model "
                    remaining = max(0, wrap_width - len(header_prefix))
                    output.append(header_prefix, style="bold cyan")
                    output.append("─" * remaining, style="dim cyan")
                    output.append("\n")
                    # Then render the text content (no prefix needed)
                    wrapped = wrap_text(line.text, 0)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
                        output.append(wrapped_line)
                else:
                    # Non-turn-start - just render text, no prefix
                    wrapped = wrap_text(line.text, 0)
                    for j, wrapped_line in enumerate(wrapped):
                        if j > 0:
                            output.append("\n")
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
            elif line.source == "enrichment":
                # Enrichment notifications - render dimmed with proper wrapping
                # The formatter pre-aligns continuation lines, so we wrap each line
                wrapped = wrap_text(line.text)
                for j, wrapped_line in enumerate(wrapped):
                    if j > 0:
                        output.append("\n")
                    output.append(wrapped_line, style="dim")
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

        # Add tool call summary (after regular lines)
        if self._active_tools:
            if items_to_show:
                output.append("\n\n")  # Extra blank line for visual separation
                # Add separator line with toggle/navigation hint
                if self._tool_nav_active:
                    # Show navigation hints when in tool nav mode using actual keybindings
                    pos = (self._selected_tool_index or 0) + 1
                    total = len(self._active_tools)
                    selected_tool = self._active_tools[self._selected_tool_index or 0]
                    nav_up = self._format_key_hint("nav_up")
                    nav_down = self._format_key_hint("nav_down")
                    expand_key = self._format_key_hint("tool_expand")
                    collapse_key = self._format_key_hint("tool_collapse")
                    exit_key = self._format_key_hint("tool_exit")
                    has_output = selected_tool.output_lines and len(selected_tool.output_lines) > 0
                    if selected_tool.expanded and has_output:
                        # When expanded: arrows scroll output, left collapses
                        output.append(f"  ───  {nav_up}/{nav_down} scroll, {collapse_key} collapse, {exit_key} exit [{pos}/{total}]", style="dim")
                    elif has_output:
                        # When collapsed but has output: arrows navigate, right expands
                        output.append(f"  ───  {nav_up}/{nav_down} nav, {expand_key} expand, {exit_key} exit [{pos}/{total}]", style="dim")
                    else:
                        # No output: just navigation hints
                        output.append(f"  ───  {nav_up}/{nav_down} nav, {exit_key} exit [{pos}/{total}]", style="dim")
                elif self._tools_expanded:
                    toggle_tools = self._format_key_hint("toggle_tools")
                    tool_nav = self._format_key_hint("tool_nav_enter")
                    output.append(f"  ───  {toggle_tools} to collapse, {tool_nav} to navigate", style="dim")
                else:
                    toggle_tools = self._format_key_hint("toggle_tools")
                    output.append(f"  ───  {toggle_tools} to expand", style="dim")
                output.append("\n")

            # Check if waiting for user input (permission or clarification)
            pending_tool = None
            for tool in self._active_tools:
                if tool.permission_state == "pending" or tool.clarification_state == "pending":
                    pending_tool = tool
                    break

            # Trace for debugging permission display issues
            if self._active_tools:
                tool_states = [(t.name, t.permission_state, t.completed) for t in self._active_tools]
                _trace(f"render_tool_tree: active_tools={tool_states}, pending_tool={pending_tool.name if pending_tool else None}")

            tool_count = len(self._active_tools)
            # Check if any tools are still executing (not completed)
            any_uncompleted = any(not tool.completed for tool in self._active_tools)
            # Show spinner if: spinner is active OR any tool is still executing
            # Also show spinner if all tools completed but not finalized yet (turn still in progress)
            all_completed = all(tool.completed for tool in self._active_tools)
            show_spinner = self._spinner_active or any_uncompleted or (all_completed and not pending_tool)

            if self._tools_expanded:
                # Expanded view - show each tool on its own line
                if pending_tool:
                    output.append("  ⏳ ", style="bold yellow")
                elif show_spinner:
                    frame = self.SPINNER_FRAMES[self._spinner_index]
                    output.append(f"  {frame} ", style="cyan")
                else:
                    output.append("  ▾ ", style="dim")
                output.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}:", style="dim")

                # Show each tool on its own line
                for i, tool in enumerate(self._active_tools):
                    is_last = (i == len(self._active_tools) - 1)
                    is_selected = (self._tool_nav_active and i == self._selected_tool_index)
                    connector = "└─" if is_last else "├─"

                    if tool.completed:
                        status_icon = "✓" if tool.success else "✗"
                        status_style = "green" if tool.success else "red"
                    else:
                        status_icon = "○"
                        status_style = "dim"

                    # Determine expand indicator (only in nav mode)
                    if self._tool_nav_active:
                        expand_icon = "▾" if tool.expanded else "▸"
                    else:
                        expand_icon = ""

                    # Selection highlight style
                    if is_selected:
                        row_style = "reverse"
                    else:
                        row_style = "dim"

                    output.append("\n")
                    if self._tool_nav_active:
                        output.append(f"  {expand_icon} {connector} ", style=row_style)
                    else:
                        output.append(f"    {connector} ", style=row_style)
                    output.append(tool.name, style=row_style)
                    if tool.args_summary:
                        output.append(f"({tool.args_summary})", style=row_style)
                    output.append(f" {status_icon}", style=status_style)

                    # Show approval indicator for granted permissions
                    if tool.permission_state == "granted" and tool.permission_method:
                        indicator = self._get_approval_indicator(tool.permission_method)
                        if indicator:
                            output.append(f" {indicator}", style="dim cyan")

                    # Show duration if available
                    if tool.completed and tool.duration_seconds is not None:
                        output.append(f" ({tool.duration_seconds:.1f}s)", style="dim")

                    # Show output only if:
                    # - In nav mode: tool.expanded is True
                    # - Not in nav mode: always show (legacy behavior)
                    show_output = tool.expanded if self._tool_nav_active else True

                    if show_output and tool.output_lines:
                        continuation = "   " if is_last else "│  "
                        prefix = "    "
                        total_lines = len(tool.output_lines)
                        display_count = tool.output_display_lines

                        # Calculate visible window (offset is from end, 0 = most recent)
                        # end_idx is exclusive, start_idx is inclusive
                        end_idx = total_lines - tool.output_scroll_offset
                        start_idx = max(0, end_idx - display_count)

                        lines_above = start_idx
                        lines_below = tool.output_scroll_offset

                        # Show "more above" indicator
                        if lines_above > 0:
                            output.append("\n")
                            output.append(f"{prefix}{continuation}   ", style="dim")
                            scroll_up_key = self._format_key_hint("nav_up")
                            output.append(f"▲ {lines_above} more line{'s' if lines_above != 1 else ''} ({scroll_up_key} to scroll)", style="dim italic")

                        # Show visible lines
                        for output_line in tool.output_lines[start_idx:end_idx]:
                            output.append("\n")
                            output.append(f"{prefix}{continuation}   ", style="dim")
                            # Truncate long lines
                            max_line_width = max(40, self._console_width - 20) if self._console_width > 60 else 40
                            if len(output_line) > max_line_width:
                                display_line = output_line[:max_line_width - 3] + "..."
                            else:
                                display_line = output_line
                            output.append(display_line, style="#87D7D7 italic")

                        # Show "more below" indicator
                        if lines_below > 0:
                            output.append("\n")
                            output.append(f"{prefix}{continuation}   ", style="dim")
                            scroll_down_key = self._format_key_hint("nav_down")
                            output.append(f"▼ {lines_below} more line{'s' if lines_below != 1 else ''} ({scroll_down_key} to scroll)", style="dim italic")

                    # Show permission denied info (when permission was denied)
                    if tool.permission_state == "denied" and tool.permission_method:
                        output.append("\n")
                        continuation = "   " if is_last else "│  "
                        output.append(f"    {continuation} ", style="dim")
                        output.append(f"  ⊘ Permission denied: User chose: {tool.permission_method}", style="red dim")
                    # Show error message if failed (but not for permission denied - already shown above)
                    elif tool.completed and not tool.success and tool.error_message:
                        output.append("\n")
                        continuation = "   " if is_last else "│  "
                        output.append(f"    {continuation} ", style="dim")
                        # Show full error message without truncation
                        output.append(f"  ⚠ {tool.error_message}", style="red dim")
            else:
                # Collapsed view - all tools on one line
                tool_summaries = []
                for tool in self._active_tools:
                    if tool.completed:
                        status_icon = "✓" if tool.success else "✗"
                    else:
                        status_icon = "○"
                    summary = f"{tool.name} {status_icon}"
                    # Add approval indicator for granted permissions
                    if tool.permission_state == "granted" and tool.permission_method:
                        indicator = self._get_approval_indicator(tool.permission_method)
                        if indicator:
                            summary += f" {indicator}"
                    tool_summaries.append(summary)

                if pending_tool:
                    output.append("  ⏳ ", style="bold yellow")
                    output.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}: ", style="dim")
                    output.append(" ".join(tool_summaries), style="dim")
                elif show_spinner:
                    frame = self.SPINNER_FRAMES[self._spinner_index]
                    output.append(f"  {frame} ", style="cyan")
                    output.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}: ", style="dim")
                    output.append(" ".join(tool_summaries), style="dim")
                else:
                    output.append("  ▸ ", style="dim")
                    output.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}: ", style="dim")
                    output.append(" ".join(tool_summaries), style="dim")

            # Show permission/clarification prompt for pending tool (expanded)
            if pending_tool:
                continuation = "   "  # No tree structure needed

                # Show permission info (only when pending)
                if pending_tool.permission_state == "pending" and pending_tool.permission_prompt_lines:
                    tool = pending_tool
                    # Expanded permission prompt
                    output.append("\n")
                    output.append(f"  {continuation}     ", style="dim")
                    output.append("⚠ ", style="yellow")
                    output.append("Permission required", style="yellow")

                    # Limit lines to show (keep options visible at end)
                    max_prompt_lines = 18
                    prompt_lines = tool.permission_prompt_lines
                    truncated = False
                    hidden_count = 0

                    # Draw box around permission prompt
                    # Box prefix is ~18 chars ("       │       │ "), leave room for border
                    max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
                    box_width = min(max_box_width, max(len(line) for line in prompt_lines) + 4)
                    content_width = box_width - 4  # Space for "│ " and " │"

                    # Count lines after wrapping to properly truncate
                    wrapped_line_count = 0
                    for line in prompt_lines:
                        if len(line) > content_width:
                            wrapped = textwrap.wrap(line, width=content_width, break_long_words=True)
                            wrapped_line_count += len(wrapped) if wrapped else 1
                        else:
                            wrapped_line_count += 1

                    if wrapped_line_count > max_prompt_lines:
                        truncated = True
                        tool.permission_truncated = True
                        hidden_count = wrapped_line_count - max_prompt_lines + 1
                    else:
                        tool.permission_truncated = False

                    output.append("\n")
                    output.append(f"  {continuation}     ┌" + "─" * (box_width - 2) + "┐", style="dim")

                    # Track rendered lines to enforce truncation
                    rendered_lines = 0
                    truncation_triggered = False
                    # Reserve space for truncation message and last line (options)
                    max_lines_before_truncation = max_prompt_lines - 3 if truncated else max_prompt_lines

                    for prompt_line in prompt_lines[:-1] if truncated else prompt_lines:
                        if truncation_triggered:
                            break
                        # Wrap long lines instead of truncating
                        if len(prompt_line) > content_width:
                            wrapped = textwrap.wrap(prompt_line, width=content_width, break_long_words=True)
                            if not wrapped:
                                wrapped = [prompt_line[:content_width]]
                        else:
                            wrapped = [prompt_line]

                        for display_line in wrapped:
                            if rendered_lines >= max_lines_before_truncation:
                                truncation_triggered = True
                                break
                            output.append("\n")
                            padding = box_width - len(display_line) - 4
                            output.append(f"  {continuation}     │ ", style="dim")
                            # Color diff lines appropriately
                            if display_line.startswith('+') and not display_line.startswith('+++'):
                                output.append(display_line, style="green")
                            elif display_line.startswith('-') and not display_line.startswith('---'):
                                output.append(display_line, style="red")
                            elif display_line.startswith('@@'):
                                output.append(display_line, style="cyan")
                            # Color options line (contains [y]es, [n]o, etc.)
                            elif display_line.strip().startswith('[') and ']' in display_line:
                                output.append(display_line, style="cyan")
                            else:
                                output.append(display_line)
                            output.append(" " * max(0, padding) + " │", style="dim")
                            rendered_lines += 1

                    # Show truncation indicator if needed
                    if truncated:
                        output.append("\n")
                        truncation_msg = f"[...{hidden_count} more - 'v' to view...]"
                        padding = box_width - len(truncation_msg) - 4
                        output.append(f"  {continuation}     │ ", style="dim")
                        output.append(truncation_msg, style="dim italic cyan")
                        output.append(" " * max(0, padding) + " │", style="dim")
                        # Show last line (usually options) - wrap if needed
                        last_line = prompt_lines[-1]
                        if len(last_line) > content_width:
                            last_wrapped = textwrap.wrap(last_line, width=content_width, break_long_words=True)
                        else:
                            last_wrapped = [last_line]
                        for display_line in last_wrapped:
                            output.append("\n")
                            padding = box_width - len(display_line) - 4
                            output.append(f"  {continuation}     │ ", style="dim")
                            output.append(display_line, style="cyan")  # Options styled cyan
                            output.append(" " * max(0, padding) + " │", style="dim")

                    output.append("\n")
                    output.append(f"  {continuation}     └" + "─" * (box_width - 2) + "┘", style="dim")

                # Show clarification info for pending tool
                if pending_tool.clarification_state == "pending":
                    tool = pending_tool
                    # Show header with progress
                    output.append("\n")
                    output.append(f"  {continuation}     ", style="dim")
                    output.append("❓ ", style="cyan")
                    if tool.clarification_total_questions > 0:
                        output.append(f"Clarification ({tool.clarification_current_question}/{tool.clarification_total_questions})", style="cyan")
                    else:
                        output.append("Clarification needed", style="cyan")

                    # Show previously answered questions (collapsed)
                    if tool.clarification_answered:
                        for q_idx, answer_summary in tool.clarification_answered:
                            output.append("\n")
                            output.append(f"  {continuation}     ", style="dim")
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

                        # Pre-calculate total rendered lines after wrapping
                        max_box_width = max(60, self._console_width - 22) if self._console_width > 40 else 60
                        box_width = min(max_box_width, max(len(line) for line in prompt_lines) + 4)
                        content_width = box_width - 4

                        # Count lines after wrapping
                        wrapped_line_count = 0
                        for line in prompt_lines:
                            if len(line) > content_width:
                                wrapped = textwrap.wrap(line, width=content_width, break_long_words=True)
                                wrapped_line_count += len(wrapped) if wrapped else 1
                            else:
                                wrapped_line_count += 1

                        if wrapped_line_count > max_prompt_lines:
                            truncated = True
                            tool.clarification_truncated = True
                            hidden_count = wrapped_line_count - max_prompt_lines + 1
                        else:
                            tool.clarification_truncated = False

                        # Draw box around current question
                        output.append("\n")
                        output.append(f"  {continuation}     ┌" + "─" * (box_width - 2) + "┐", style="dim")

                        # Track rendered lines to enforce truncation
                        rendered_lines = 0
                        truncation_triggered = False

                        for prompt_line in prompt_lines:
                            if truncation_triggered:
                                break
                            # Wrap long lines
                            if len(prompt_line) > content_width:
                                wrapped = textwrap.wrap(prompt_line, width=content_width, break_long_words=True)
                                if not wrapped:
                                    wrapped = [prompt_line[:content_width]]
                            else:
                                wrapped = [prompt_line]
                            for display_line in wrapped:
                                if truncated and rendered_lines >= max_prompt_lines - 2:
                                    truncation_triggered = True
                                    break
                                output.append("\n")
                                padding = box_width - len(display_line) - 4
                                output.append(f"  {continuation}     │ ", style="dim")
                                output.append(display_line)
                                output.append(" " * max(0, padding) + " │", style="dim")
                                rendered_lines += 1

                        # Show truncation indicator if needed
                        if truncated:
                            output.append("\n")
                            truncation_msg = f"[...{hidden_count} more - 'v' to view...]"
                            padding = box_width - len(truncation_msg) - 4
                            output.append(f"  {continuation}     │ ", style="dim")
                            output.append(truncation_msg, style="dim italic cyan")
                            output.append(" " * max(0, padding) + " │", style="dim")
                            last_line = prompt_lines[-1]
                            if len(last_line) > content_width:
                                last_wrapped = textwrap.wrap(last_line, width=content_width, break_long_words=True)
                            else:
                                last_wrapped = [last_line]
                            for display_line in last_wrapped:
                                output.append("\n")
                                padding = box_width - len(display_line) - 4
                                output.append(f"  {continuation}     │ ", style="dim")
                                output.append(display_line, style="cyan")
                                output.append(" " * max(0, padding) + " │", style="dim")

                        output.append("\n")
                        output.append(f"  {continuation}     └" + "─" * (box_width - 2) + "┘", style="dim")
        elif self._spinner_active:
            # Spinner active but no tools yet - show model header first
            if items_to_show:
                output.append("\n\n")  # Blank line before header
                # Show model header if this is a new turn
                if self._last_turn_source != "model":
                    header_prefix = "── Model "
                    remaining = max(0, wrap_width - len(header_prefix))
                    output.append(header_prefix, style="bold cyan")
                    output.append("─" * remaining, style="dim cyan")
                    output.append("\n")
            frame = self.SPINNER_FRAMES[self._spinner_index]
            output.append(f"  {frame} ", style="cyan")
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

    # -------------------------------------------------------------------------
    # Chrome-free text extraction (for clipboard operations)
    # -------------------------------------------------------------------------

    def _should_include_source(self, source: str, sources: Optional[set] = None) -> bool:
        """Check if a source should be included based on filter.

        Args:
            source: The source type to check.
            sources: Set of sources to include. If None, includes all.

        Returns:
            True if the source should be included.
        """
        if sources is None:
            return True
        if source in sources:
            return True
        # Handle prefix matching (e.g., "tool" matches "tool:grep")
        if ":" in source:
            prefix = source.split(":")[0]
            if prefix in sources:
                return True
        return False

    def get_last_response_text(self, sources: Optional[set] = None) -> Optional[str]:
        """Get text from the last model response (chrome-free).

        Extracts text from the most recent contiguous block of model output,
        excluding turn headers and other UI chrome.

        Args:
            sources: Set of sources to include (default: {"model"}).

        Returns:
            The extracted text, or None if no matching content.
        """
        if sources is None:
            sources = {"model"}

        # Get all items including current streaming block
        all_items = list(self._lines) + self._get_current_block_lines()

        if not all_items:
            return None

        # Find the last contiguous block of matching source lines
        result_lines: List[str] = []
        in_block = False

        # Walk backwards to find the last matching block
        for item in reversed(all_items):
            # Skip ToolBlocks - they don't have text content
            if isinstance(item, ToolBlock):
                if in_block:
                    # We've hit a ToolBlock while in a block, stop
                    break
                continue
            # item is an OutputLine
            if self._should_include_source(item.source, sources):
                result_lines.append(item.text)
                in_block = True
            elif in_block:
                # We've exited the block, stop
                break

        if not result_lines:
            return None

        # Reverse since we walked backwards
        result_lines.reverse()

        # Join and clean up
        text = "\n".join(result_lines)
        return text.strip() if text.strip() else None

    def get_text_in_line_range(
        self,
        start_line: int,
        end_line: int,
        sources: Optional[set] = None
    ) -> Optional[str]:
        """Get text from a range of display lines (chrome-free).

        Used for mouse selection - maps screen coordinates to content.

        Args:
            start_line: Start display line (0-indexed from top of buffer).
            end_line: End display line (inclusive).
            sources: Set of sources to include. If None, includes all.

        Returns:
            The extracted text, or None if no matching content.
        """
        all_items = list(self._lines) + self._get_current_block_lines()

        if not all_items:
            return None

        result_lines: List[str] = []
        current_display_line = 0

        for item in all_items:
            item_height = self._get_item_display_lines(item)
            line_end = current_display_line + item_height

            # Check if this item overlaps with the selection range
            if line_end > start_line and current_display_line <= end_line:
                # Only include OutputLine items (skip ToolBlocks)
                if isinstance(item, OutputLine):
                    if self._should_include_source(item.source, sources):
                        result_lines.append(item.text)

            current_display_line = line_end

            # Stop if we've passed the selection range
            if current_display_line > end_line:
                break

        if not result_lines:
            return None

        text = "\n".join(result_lines)
        return text.strip() if text.strip() else None

    def get_all_text(self, sources: Optional[set] = None) -> Optional[str]:
        """Get all text from the buffer (chrome-free).

        Args:
            sources: Set of sources to include. If None, includes all.

        Returns:
            The extracted text, or None if no matching content.
        """
        all_items = list(self._lines) + self._get_current_block_lines()

        if not all_items:
            return None

        result_lines: List[str] = []

        for item in all_items:
            # Only include OutputLine items (skip ToolBlocks)
            if isinstance(item, OutputLine):
                if self._should_include_source(item.source, sources):
                    result_lines.append(item.text)

        if not result_lines:
            return None

        text = "\n".join(result_lines)
        return text.strip() if text.strip() else None

    def get_visible_line_range(self, visible_height: int) -> Tuple[int, int]:
        """Get the display line range currently visible on screen.

        Used for mouse selection to map screen Y coordinates to buffer lines.

        Args:
            visible_height: Height of the visible area in display lines.

        Returns:
            Tuple of (start_line, end_line) display line indices.
        """
        all_items = list(self._lines) + self._get_current_block_lines()
        total_display_lines = sum(self._get_item_display_lines(item) for item in all_items)

        # Calculate visible range based on scroll offset
        end_line = total_display_lines - self._scroll_offset
        start_line = max(0, end_line - visible_height)

        return (start_line, end_line)
