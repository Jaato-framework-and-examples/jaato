"""Tool output popup panel for live tool output visualization.

Renders a floating popup panel (right-aligned) that shows real-time output
from running tools. Supports auto-follow (tail -f), tab-switching between
concurrent tools, and auto-dismiss on tool completion.

The popup automatically appears when an expanded tool starts producing output
and dismisses when the tool finishes.
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
    - Auto-popup when an expanded running tool has output
    - Auto-follow (tail -f style, always showing latest output)
    - Tab-switching between concurrent running tools
    - Auto-dismiss when the tracked tool completes
    - Right-aligned positioning with border
    """

    def __init__(self, tab_key: Optional[KeyBinding] = None):
        self._visible: bool = False
        self._tracked_call_id: Optional[str] = None  # Which tool we're showing
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
        """Get the call_id of the tool currently being tracked."""
        return self._tracked_call_id

    def _get_running_tools(self, buffer: "OutputBuffer") -> List["ActiveToolCall"]:
        """Get list of currently running (non-completed) tools with output."""
        return [
            t for t in buffer.active_tools
            if not t.completed and t.output_lines and t.call_id
        ]

    def _get_tracked_tool(self, buffer: "OutputBuffer") -> Optional["ActiveToolCall"]:
        """Get the tool currently being tracked, if still running."""
        if not self._tracked_call_id:
            return None
        for tool in buffer.active_tools:
            if tool.call_id == self._tracked_call_id:
                return tool
        return None

    def update(self, buffer: "OutputBuffer") -> None:
        """Update popup state based on current tool state.

        Called on each refresh cycle. Handles:
        - Auto-popup when an expanded running tool produces output
        - Auto-dismiss when tracked tool completes
        - Tracking the first available tool if none tracked
        """
        tracked = self._get_tracked_tool(buffer)

        if self._visible:
            # Auto-dismiss: tracked tool completed or disappeared
            if tracked is None or tracked.completed:
                # Try to switch to another running tool
                running = self._get_running_tools(buffer)
                if running:
                    self._tracked_call_id = running[0].call_id
                    self._scroll_offset = 0
                else:
                    self._visible = False
                    self._tracked_call_id = None
                    self._scroll_offset = 0
                return

        if not self._visible:
            # Auto-popup: check for expanded running tools with output
            for tool in buffer.active_tools:
                if (not tool.completed
                        and tool.expanded
                        and tool.output_lines
                        and tool.call_id):
                    self._visible = True
                    self._tracked_call_id = tool.call_id
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
            if tool.call_id == self._tracked_call_id:
                current_idx = i
                break

        # Cycle
        if forward:
            next_idx = (current_idx + 1) % len(running)
        else:
            next_idx = (current_idx - 1) % len(running)

        self._tracked_call_id = running[next_idx].call_id
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
        self._tracked_call_id = None
        self._scroll_offset = 0

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
                if tool.call_id == self._tracked_call_id:
                    current_idx = i
                    break
            tab_hint = f" [{current_idx + 1}/{len(running)}]"
        else:
            tab_hint = ""

        tool_name = tracked.display_name or tracked.name
        title = f"[bold]{tool_name}{tab_hint}[/bold]"

        # Get output lines
        lines = tracked.output_lines

        # Calculate visible window
        total_lines = len(lines)
        visible_lines = min(total_lines, max_height - 3)  # Reserve for borders + hints

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

        # "More above" indicator
        if start > 0:
            above = Text()
            above.append(f" ↑ {start} more above", style=self._style("tool_output_popup_scroll", "dim"))
            elements.append(above)

        # Content lines (strip to fit width, preserve ANSI)
        content_width = width - 4  # Panel borders
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
