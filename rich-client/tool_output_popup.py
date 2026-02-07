"""Tool output popup panel for live tool output visualization.

Renders a floating popup panel (right-aligned) that shows real-time output
from running tools. Supports auto-follow (tail -f), tab-switching between
concurrent tools, and auto-dismiss on tool completion.

The popup automatically appears when an expanded tool starts producing output
and dismisses when the tool finishes. For continuation groups (e.g.,
interactive shell sessions), the popup stays open across multiple tool calls
sharing the same continuation_id.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text
from rich.console import Group

from keybindings import KeyBinding, format_key_for_display

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from output_buffer import ActiveToolCall, OutputBuffer
    from theme import ThemeConfig


class ToolOutputPopup:
    """Renders a floating popup for live tool output visualization.

    Features:
    - Auto-popup when tools are expanded and a running tool has output
    - Auto-follow (tail -f style, always showing latest output)
    - Tab-switching between concurrent running tools
    - Auto-dismiss when the tracked tool completes
    - Continuation-aware: stays open across tools in the same session
    - Right-aligned positioning with border
    """

    def __init__(self, tab_key: Optional[KeyBinding] = None):
        self._visible: bool = False
        self._tracked_popup_key: Optional[str] = None  # call_id or continuation_id
        self._scroll_offset: int = 0  # 0 = auto-follow (bottom), >0 = scrolled up
        self._tab_key = tab_key or "c-o"
        self._theme: Optional["ThemeConfig"] = None
        self._max_lines: int = 20  # Max visible lines in popup

    def set_theme(self, theme: "ThemeConfig") -> None:
        """Set the theme configuration for styling."""
        self._theme = theme

    def _style(self, semantic_name: str, fallback: str = "") -> str:
        """Get a Rich style string from the theme."""
        if self._theme:
            style = self._theme.get_rich_style(semantic_name)
            if style:
                return style
        return fallback

    @property
    def is_visible(self) -> bool:
        """Check if popup is currently visible."""
        return self._visible

    @property
    def tracked_call_id(self) -> Optional[str]:
        """Get the key of the tool currently being tracked (call_id or continuation_id)."""
        return self._tracked_popup_key

    def _get_popup_key(self, tool: "ActiveToolCall") -> Optional[str]:
        """Get the popup tracking key for a tool (continuation_id or call_id)."""
        return tool.continuation_id or tool.call_id

    def _get_running_tools(self, buffer: "OutputBuffer") -> List["ActiveToolCall"]:
        """Get list of running, backgrounded, or continuation tools with output.

        Checks both active_tools (pre-finalization) and popup_tools
        (post-finalization backgrounded/continuation tools).
        """
        result = [
            t for t in buffer.active_tools
            if (not t.completed or t.backgrounded or t.continuation_id)
            and t.output_lines and t.call_id
        ]
        seen = {self._get_popup_key(t) for t in result}
        for key, tool in buffer.popup_tools.items():
            if key not in seen and (tool.backgrounded or tool.continuation_id) and tool.output_lines:
                result.append(tool)
        return result

    def _get_tracked_tool(self, buffer: "OutputBuffer") -> Optional["ActiveToolCall"]:
        """Get the tool currently being tracked, if still active.

        Searches by popup key (call_id or continuation_id) across:
        1. Active tools — prefer running tool matching key
        2. Popup tools — completed continuation/backgrounded tool
        3. Active tools fallback — any tool matching key
        """
        if not self._tracked_popup_key:
            return None
        key = self._tracked_popup_key
        # Check active_tools for a running tool matching key
        for tool in buffer.active_tools:
            if not tool.completed and (tool.call_id == key or tool.continuation_id == key):
                return tool
        # Check popup_tools by key (continuation or backgrounded)
        popup_tool = buffer.popup_tools.get(key)
        if popup_tool:
            return popup_tool
        # Fall back to any active tool matching key (completed continuation, etc.)
        for tool in buffer.active_tools:
            if tool.call_id == key or tool.continuation_id == key:
                return tool
        return None

    def update(self, buffer: "OutputBuffer") -> None:
        """Update popup state based on current tool state.

        Called on each refresh cycle. Handles:
        - Auto-popup when tools are expanded and a running tool produces output
        - Auto-dismiss when tracked tool completes (unless continuation)
        - Continuation: stay open when tracked tool completes with continuation_id
        - Tracking the first available tool if none tracked
        """
        tracked = self._get_tracked_tool(buffer)

        if self._visible:
            # Auto-dismiss: tracked tool completed (and not backgrounded/continuation) or disappeared
            if tracked is None:
                # Tool gone entirely — try switching
                running = self._get_running_tools(buffer)
                if running:
                    self._tracked_popup_key = self._get_popup_key(running[0])
                    self._scroll_offset = 0
                else:
                    self._visible = False
                    self._tracked_popup_key = None
                    self._scroll_offset = 0
                return

            if tracked.completed and not tracked.backgrounded:
                if tracked.continuation_id:
                    # Session still alive — switch tracking to continuation_id, stay open
                    self._tracked_popup_key = tracked.continuation_id
                    return
                # Normal completion — try switch or dismiss
                running = self._get_running_tools(buffer)
                if running:
                    self._tracked_popup_key = self._get_popup_key(running[0])
                    self._scroll_offset = 0
                else:
                    self._visible = False
                    self._tracked_popup_key = None
                    self._scroll_offset = 0
                return

        if not self._visible:
            # Auto-popup: show when tools are expanded and a running/backgrounded tool has output
            if not buffer.tools_expanded:
                return
            running = self._get_running_tools(buffer)
            if running:
                self._visible = True
                self._tracked_popup_key = self._get_popup_key(running[0])
                self._scroll_offset = 0
                return

    def cycle_tool(self, buffer: "OutputBuffer", forward: bool = True) -> bool:
        """Cycle to the next/previous running tool.

        Args:
            buffer: The output buffer to get tools from.
            forward: True for next, False for previous.

        Returns:
            True if cycled to a different tool.
        """
        running = self._get_running_tools(buffer)
        if len(running) <= 1:
            return False

        # Find current index
        current_idx = 0
        for i, tool in enumerate(running):
            if self._get_popup_key(tool) == self._tracked_popup_key:
                current_idx = i
                break

        # Cycle
        if forward:
            next_idx = (current_idx + 1) % len(running)
        else:
            next_idx = (current_idx - 1) % len(running)

        self._tracked_popup_key = self._get_popup_key(running[next_idx])
        self._scroll_offset = 0  # Reset to auto-follow on switch
        return True

    def scroll_up(self, lines: int = 1) -> bool:
        """Scroll up in the popup (away from bottom).

        Returns:
            True if scrolled.
        """
        self._scroll_offset += lines
        return True

    def scroll_down(self, lines: int = 1) -> bool:
        """Scroll down in the popup (toward bottom).

        Returns:
            True if scrolled.
        """
        if self._scroll_offset > 0:
            self._scroll_offset = max(0, self._scroll_offset - lines)
            return True
        return False

    def scroll_to_bottom(self) -> None:
        """Reset to auto-follow mode (scroll to bottom)."""
        self._scroll_offset = 0

    def dismiss(self) -> None:
        """Manually dismiss the popup."""
        self._visible = False
        self._tracked_popup_key = None
        self._scroll_offset = 0

    def get_content_width(self, buffer: "OutputBuffer", min_width: int = 40, max_width: int = 120) -> int:
        """Calculate optimal popup width based on tracked tool's output content.

        Measures the max visible width of the output lines and returns a
        width that fits the content, clamped between min_width and max_width.

        Args:
            buffer: Output buffer containing tool data.
            min_width: Minimum popup width.
            max_width: Maximum popup width (typically based on terminal width).

        Returns:
            Optimal popup width in characters.
        """
        tracked = self._get_tracked_tool(buffer)
        if not tracked or not tracked.output_lines:
            return min_width

        # Panel borders: │ + space on each side = 4 chars
        border_overhead = 4

        # Measure max content width from output lines
        max_line_width = 0
        for line in tracked.output_lines:
            if '\x1b[' in line:
                text = Text.from_ansi(line)
                width = text.cell_len
            else:
                width = len(line)
            if width > max_line_width:
                max_line_width = width

        # Also consider sticky header (args summary)
        args_text = tracked.display_args_summary if tracked.display_args_summary is not None else tracked.args_summary
        if args_text:
            # "❯ " prefix = 2 chars
            max_line_width = max(max_line_width, len(args_text) + 2)

        optimal = max_line_width + border_overhead
        return max(min_width, min(max_width, optimal))

    def render(self, buffer: "OutputBuffer", width: int = 60, max_height: int = 20) -> Panel:
        """Render the popup panel.

        Args:
            buffer: Output buffer containing tool data.
            width: Width of the popup panel.
            max_height: Maximum height for the popup content.

        Returns:
            Rich Panel with tool output.
        """
        self._max_lines = max_height

        tracked = self._get_tracked_tool(buffer)
        if not tracked or not tracked.output_lines:
            return Panel(
                Text("No output", style="dim"),
                title="[bold]Tool Output[/bold]",
                border_style=self._style("tool_output_popup_border", "cyan"),
                width=width,
            )

        # Build title with tool name and tab indicator
        running = self._get_running_tools(buffer)
        if len(running) > 1:
            # Find current index among running tools
            current_idx = 0
            for i, tool in enumerate(running):
                if self._get_popup_key(tool) == self._tracked_popup_key:
                    current_idx = i
                    break
            tab_hint = f" [{current_idx + 1}/{len(running)}]"
        else:
            tab_hint = ""

        tool_name = tracked.display_name or tracked.name
        title = f"[bold]{tool_name}{tab_hint}[/bold]"

        # Get output lines
        lines = tracked.output_lines

        # Calculate visible window — reserve for borders, sticky header, separator, hints
        total_lines = len(lines)
        visible_lines = min(total_lines, max_height - 5)  # borders + header + separator + hint

        if self._scroll_offset == 0:
            # Auto-follow: show last N lines
            start = max(0, total_lines - visible_lines)
            end = total_lines
        else:
            # Manual scroll
            end = max(0, total_lines - self._scroll_offset)
            start = max(0, end - visible_lines)
            # Clamp scroll offset
            if self._scroll_offset > total_lines - visible_lines:
                self._scroll_offset = max(0, total_lines - visible_lines)
                start = 0
                end = min(visible_lines, total_lines)

        display_lines = lines[start:end]

        # Build content
        elements = []

        # Sticky header: command/args summary
        content_width = width - 4  # Panel borders
        args_text = tracked.display_args_summary if tracked.display_args_summary is not None else tracked.args_summary
        if args_text:
            header = Text()
            header.append("❯ ", style=self._style("tool_output_popup_border", "cyan"))
            header.append(args_text, style=self._style("tool_output_popup_header", "bold"))
            header.truncate(content_width)
            elements.append(header)
            elements.append(Text("─" * content_width, style=self._style("tool_output_popup_scroll", "dim")))

        # "More above" indicator
        if start > 0:
            above = Text()
            above.append(f" ↑ {start} more above", style=self._style("tool_output_popup_scroll", "dim"))
            elements.append(above)

        # Content lines (strip to fit width, preserve ANSI)
        for line in display_lines:
            line_text = Text.from_ansi(line) if '\x1b[' in line else Text(line)
            line_text.truncate(content_width)
            elements.append(line_text)

        # "More below" indicator
        lines_below = total_lines - end
        if lines_below > 0:
            below = Text()
            below.append(f" ↓ {lines_below} more below", style=self._style("tool_output_popup_scroll", "dim"))
            elements.append(below)

        # Hint line
        hint = Text()
        if self._scroll_offset > 0:
            hint.append(" [scrolled] ", style=self._style("tool_output_popup_hint", "dim italic"))
        if len(running) > 1:
            tab_display = format_key_for_display(self._tab_key)
            hint.append(f"[{tab_display}: switch tool]", style=self._style("tool_output_popup_hint", "dim italic"))
        elements.append(hint)

        border_style = self._style("tool_output_popup_border", "cyan")

        return Panel(
            Group(*elements),
            title=title,
            border_style=border_style,
            style=self._style("tool_output_popup_background", ""),
            width=width,
        )
