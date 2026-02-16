"""Agent tab bar for displaying agents as horizontal tabs.

This module provides the AgentTabBar class for rendering agents as
compact horizontal tabs at the top of the UI, replacing the vertical
side panel.
"""

import time
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_registry import AgentRegistry, AgentInfo

from keybindings import KeyBinding, format_key_for_display


# Spinner frames for processing status (same as output panel)
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Status symbols
STATUS_SYMBOLS = {
    "active": None,  # Uses animated spinner
    "idle": "⏸",
    "waiting": "⏸",
    "pending": "⏸",
    "done": "🏁",
    "error": "💣",
    "permission": "?",
}

# Status colors (prompt_toolkit style classes)
STATUS_STYLES = {
    "active": "class:agent-tab.processing",
    "idle": "class:agent-tab.awaiting",
    "waiting": "class:agent-tab.awaiting",
    "pending": "class:agent-tab.awaiting",
    "done": "class:agent-tab.finished",
    "error": "class:agent-tab.error",
    "permission": "class:agent-tab.permission",
}


class AgentTabBar:
    """Horizontal tab bar for displaying agents.

    Renders agents as compact tabs with status symbols:
    - Processing: animated spinner (blue)
    - Awaiting: ⏸ (cyan)
    - Finished: 🏁 (dim)
    - Error: 💣 (red)
    - Permission: ? (yellow)

    Selected tab is shown with reverse video (like permission prompt TAB-cycling).
    """

    def __init__(
        self,
        agent_registry: "AgentRegistry",
        cycle_key: Optional[KeyBinding] = None,
    ):
        """Initialize the agent tab bar.

        Args:
            agent_registry: Registry managing all agents.
            cycle_key: Keybinding for cycling agents (for display hint).
        """
        self._registry = agent_registry
        self._cycle_key = cycle_key or "c-a"
        self._width = 80  # Updated dynamically

        # Spinner state
        self._spinner_frame = 0

        # Popup state
        self._popup_visible = False
        self._popup_hide_time: Optional[float] = None
        self._popup_auto_hide_delay = 2.5  # seconds

        # Scroll state for overflow
        self._scroll_offset = 0

    def set_width(self, width: int) -> None:
        """Set available width for the tab bar.

        Args:
            width: Available width in characters.
        """
        self._width = max(40, width)

    def advance_spinner(self) -> None:
        """Advance the spinner animation frame."""
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)

    def get_spinner_char(self) -> str:
        """Get the current spinner character."""
        return SPINNER_FRAMES[self._spinner_frame]

    def show_popup(self) -> None:
        """Show the details popup and schedule auto-hide."""
        self._popup_visible = True
        self._popup_hide_time = time.time() + self._popup_auto_hide_delay

    def hide_popup(self) -> None:
        """Hide the details popup."""
        self._popup_visible = False
        self._popup_hide_time = None

    def check_popup_timeout(self) -> bool:
        """Check if popup should auto-hide.

        Returns:
            True if popup was hidden, False otherwise.
        """
        if self._popup_visible and self._popup_hide_time:
            if time.time() >= self._popup_hide_time:
                self.hide_popup()
                return True
        return False

    @property
    def is_popup_visible(self) -> bool:
        """Check if popup is currently visible."""
        return self._popup_visible

    def get_status_symbol(self, status: str) -> str:
        """Get the status symbol for an agent status.

        Args:
            status: Agent status string.

        Returns:
            Status symbol character.
        """
        if status == "active":
            return self.get_spinner_char()
        return STATUS_SYMBOLS.get(status, "⏸")

    def get_status_style(self, status: str, is_selected: bool) -> str:
        """Get the style for a status symbol.

        Args:
            status: Agent status string.
            is_selected: Whether this agent is selected.

        Returns:
            Style class string.
        """
        if is_selected:
            return "class:agent-tab.selected"
        return STATUS_STYLES.get(status, "class:agent-tab.awaiting")

    def _build_agent_tabs(self, agents, selected_id) -> List[Tuple[str, str, int]]:
        """Build individual tab entries with their widths.

        Each tab shows the agent_id as its label. The selected tab is
        rendered in reverse video; unselected tabs are dim.

        Args:
            agents: List of AgentInfo objects.
            selected_id: ID of the selected agent.

        Returns:
            List of (tabs, agent_index, width) tuples where tabs is a list
            of (style, text) tuples for that agent's tab.
        """
        result = []

        for i, agent in enumerate(agents):
            is_selected = (agent.agent_id == selected_id)
            tab_parts: List[Tuple[str, str]] = []

            # Get status symbol (1-2 chars depending on emoji)
            symbol = self.get_status_symbol(agent.status)

            # Determine styles: selected uses reverse video, unselected is dim
            if is_selected:
                name_style = "class:agent-tab.selected reverse"
            else:
                name_style = "class:agent-tab.dim"
            symbol_style = self._get_symbol_color_style(agent.status)

            # Use agent_id as the tab label
            max_name_len = 15
            label = agent.agent_id
            if len(label) > max_name_len:
                label = label[:max_name_len - 1] + "…"

            # Build tab parts
            tab_parts.append((symbol_style, symbol))
            tab_parts.append(("", " "))
            if is_selected:
                # Pad with spaces for reverse-video readability
                tab_parts.append((name_style, f" {label} "))
            else:
                tab_parts.append((name_style, label))

            # Calculate width (symbol + space + label + optional padding)
            # Note: some symbols like 🏁 and 💣 may render as 2 chars wide
            symbol_width = 2 if symbol in ("🏁", "💣") else 1
            label_display_len = len(label) + 2 if is_selected else len(label)
            width = symbol_width + 1 + label_display_len

            result.append((tab_parts, i, width))

        return result

    def render(self) -> List[Tuple[str, str]]:
        """Render the tab bar as prompt_toolkit formatted text.

        Handles overflow by showing scroll indicators (◀ ▶) when there
        are more tabs than can fit in the available width.

        Returns:
            List of (style, text) tuples for prompt_toolkit.
        """
        agents = self._registry.get_all_agents()
        selected_id = self._registry.get_selected_agent_id()

        if not agents:
            # No agents - show placeholder
            return [("class:agent-tab.dim", " No agents ")]

        # Build individual tab entries
        tab_entries = self._build_agent_tabs(agents, selected_id)

        # Find selected agent index
        selected_index = 0
        for i, agent in enumerate(agents):
            if agent.agent_id == selected_id:
                selected_index = i
                break

        # Calculate total width needed
        separator_width = 3  # " │ "
        total_width = sum(w for _, _, w in tab_entries)
        total_width += separator_width * (len(tab_entries) - 1)

        # Reserve space for hint and possible scroll indicators
        # Only show cycle hint when there are multiple agents
        if len(agents) > 1:
            key_hint = format_key_for_display(self._cycle_key)
            hint_text = f"({key_hint}: cycle)"
            reserved_width = 4 + len(hint_text)  # leading/trailing space + hint
        else:
            hint_text = ""
            reserved_width = 4  # just leading/trailing space
        available_width = self._width - reserved_width

        # Check if we need overflow handling
        needs_overflow = total_width > available_width

        if needs_overflow:
            # Reserve space for scroll indicators
            scroll_indicator_width = 4  # "◀ " or " ▶"
            available_width -= scroll_indicator_width * 2

            # Adjust scroll offset to keep selected tab visible
            self._adjust_scroll_for_selection(tab_entries, selected_index, available_width, separator_width)

        # Build visible tabs
        tabs: List[Tuple[str, str]] = []
        tabs.append(("", " "))  # Leading space

        if needs_overflow:
            # Show left scroll indicator if there are hidden tabs on the left
            if self._scroll_offset > 0:
                tabs.append(("class:agent-tab.scroll", "◀ "))
            else:
                tabs.append(("", "  "))

        # Render visible tabs
        current_width = 0
        visible_start = self._scroll_offset if needs_overflow else 0
        first_visible = True

        for tab_parts, idx, width in tab_entries:
            if idx < visible_start:
                continue

            # Check if this tab fits
            tab_total_width = width + (0 if first_visible else separator_width)
            if needs_overflow and current_width + tab_total_width > available_width:
                break

            # Add separator (except for first visible)
            if not first_visible:
                tabs.append(("class:agent-tab.separator", " │ "))
                current_width += separator_width

            # Add tab parts
            tabs.extend(tab_parts)
            current_width += width
            first_visible = False

        if needs_overflow:
            # Show right scroll indicator if there are hidden tabs on the right
            visible_end = visible_start
            for tab_parts, idx, width in tab_entries:
                if idx >= visible_start:
                    visible_end = idx + 1
            if visible_end < len(tab_entries):
                tabs.append(("class:agent-tab.scroll", " ▶"))
            else:
                tabs.append(("", "  "))

        # Add trailing space and hint
        tabs.append(("", "  "))
        tabs.append(("class:agent-tab.hint", hint_text))
        tabs.append(("", " "))

        return tabs

    def _adjust_scroll_for_selection(
        self,
        tab_entries: List[Tuple[List[Tuple[str, str]], int, int]],
        selected_index: int,
        available_width: int,
        separator_width: int
    ) -> None:
        """Adjust scroll offset to ensure selected tab is visible.

        Args:
            tab_entries: List of (tab_parts, index, width) tuples.
            selected_index: Index of the selected tab.
            available_width: Available width for tabs.
            separator_width: Width of separator between tabs.
        """
        # If selected is before scroll offset, scroll left
        if selected_index < self._scroll_offset:
            self._scroll_offset = selected_index
            return

        # Calculate width from scroll_offset to selected (inclusive)
        width_to_selected = 0
        for tab_parts, idx, width in tab_entries:
            if idx < self._scroll_offset:
                continue
            if idx > selected_index:
                break
            if idx > self._scroll_offset:
                width_to_selected += separator_width
            width_to_selected += width

        # If selected tab doesn't fit, scroll right
        while width_to_selected > available_width and self._scroll_offset < selected_index:
            # Remove first visible tab's width
            for tab_parts, idx, width in tab_entries:
                if idx == self._scroll_offset:
                    width_to_selected -= width
                    if self._scroll_offset > 0:
                        width_to_selected -= separator_width
                    break
            self._scroll_offset += 1

    def _get_symbol_color_style(self, status: str) -> str:
        """Get color style for status symbol (not affected by selection).

        Args:
            status: Agent status string.

        Returns:
            Style class for the symbol color.
        """
        style_map = {
            "active": "class:agent-tab.symbol.processing",
            "idle": "class:agent-tab.symbol.awaiting",
            "waiting": "class:agent-tab.symbol.awaiting",
            "pending": "class:agent-tab.symbol.awaiting",
            "done": "class:agent-tab.symbol.finished",
            "error": "class:agent-tab.symbol.error",
            "permission": "class:agent-tab.symbol.permission",
        }
        return style_map.get(status, "class:agent-tab.symbol.awaiting")

    def render_popup(self, agent: "AgentInfo") -> List[Tuple[str, str]]:
        """Render a minimal tooltip showing only the subagent name.

        The tooltip is a single-line box displayed briefly when cycling
        agents with C-A, giving the user a quick glance at the agent's
        display name without the heavier icon/status popup.

        Args:
            agent: Agent to show details for.

        Returns:
            List of (style, text) tuples for prompt_toolkit.
        """
        name = agent.name
        # Pad name for visual breathing room
        content = f" {name} "
        return [("class:agent-popup.name reverse bold", content)]

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max_width.

        Args:
            text: Text to wrap.
            max_width: Maximum width per line.

        Returns:
            List of wrapped lines.
        """
        if len(text) <= max_width:
            return [text]

        lines = []
        remaining = text

        while remaining:
            if len(remaining) <= max_width:
                lines.append(remaining)
                break

            # Find a good break point (prefer breaking at - or .)
            break_at = max_width
            for sep in ['-', '.', '_']:
                pos = remaining.rfind(sep, 0, max_width)
                if pos > max_width // 2:  # Only break if separator is past halfway
                    break_at = pos + 1
                    break

            lines.append(remaining[:break_at])
            remaining = remaining[break_at:]

        return lines

    def _get_popup_status_style(self, status: str) -> str:
        """Get style for status in popup.

        Args:
            status: Agent status string.

        Returns:
            Style class for the status text.
        """
        style_map = {
            "active": "class:agent-popup.status.processing",
            "idle": "class:agent-popup.status.awaiting",
            "waiting": "class:agent-popup.status.awaiting",
            "pending": "class:agent-popup.status.awaiting",
            "done": "class:agent-popup.status.finished",
            "error": "class:agent-popup.status.error",
        }
        return style_map.get(status, "class:agent-popup.status.awaiting")
