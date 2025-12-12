"""Agent panel for visualizing active and completed agents.

This module provides the AgentPanel class for rendering the agent list
in the rich client's side panel (20% width).
"""

from typing import List, Optional
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.style import Style

from agent_registry import AgentRegistry, AgentInfo


class AgentPanel:
    """Panel for displaying agent list with selection indicator.

    Renders agents as cards with ASCII art icons, labels, and status.
    The selected agent is highlighted.
    """

    def __init__(self, agent_registry: AgentRegistry):
        """Initialize the agent panel.

        Args:
            agent_registry: Registry managing all agents.
        """
        self._registry = agent_registry
        self._panel_width = 24  # Default width, updated dynamically

    def set_width(self, width: int) -> None:
        """Set panel width.

        Args:
            width: Panel width in characters.
        """
        self._panel_width = max(12, width)  # Minimum 12 chars

    def render(self, available_height: int) -> Panel:
        """Render the agent panel.

        Args:
            available_height: Available vertical space for the panel.

        Returns:
            Rich Panel containing the agent list.
        """
        agents = self._registry.get_all_agents()
        selected_id = self._registry.get_selected_agent_id()

        if not agents:
            # No agents - show placeholder
            placeholder = Text("No agents", style="dim")
            return Panel(
                placeholder,
                title="Agents",
                border_style="dim",
                width=self._panel_width
            )

        # Render each agent card
        agent_renderables = []
        for agent in agents:
            is_selected = (agent.agent_id == selected_id)
            card = self._render_agent_card(agent, is_selected)
            agent_renderables.append(card)

        # Combine into group
        agent_group = Group(*agent_renderables)

        # Create panel
        panel = Panel(
            agent_group,
            title="[bold]Agents[/bold] [dim](F2: cycle)[/dim]",
            border_style="cyan",
            width=self._panel_width,
            padding=(0, 1)
        )

        return panel

    def _render_agent_card(self, agent: AgentInfo, is_selected: bool) -> Group:
        """Render a single agent card.

        Args:
            agent: Agent information.
            is_selected: Whether this agent is currently selected.

        Returns:
            Rich Group containing the card elements.
        """
        card_width = self._panel_width - 4  # Account for panel padding/border

        # Map internal status to user-friendly display labels
        status_labels = {
            "active": "Processing",
            "waiting": "Awaiting",
            "done": "Finished",
            "error": "Error",
            "pending": "Awaiting"
        }
        display_status = status_labels.get(agent.status, agent.status.capitalize())

        # Determine styles based on status and selection
        if is_selected:
            border_char = "═"
            name_style = "bold cyan"
            status_style = "cyan"
        else:
            border_char = "─"
            if agent.status == "done":
                name_style = "dim"
                status_style = "dim"
            elif agent.status == "error":
                name_style = "red"
                status_style = "red"
            elif agent.status == "waiting":
                name_style = "white"
                status_style = "green"
            else:  # active
                name_style = "white"
                status_style = "yellow"

        lines = []

        # Top border
        if is_selected:
            top_border = Text("╔" + border_char * (card_width - 2) + "╗", style="cyan")
        else:
            top_border = Text("┌" + border_char * (card_width - 2) + "┐", style="dim")
        lines.append(top_border)

        # Icon lines (3 lines)
        for icon_line in agent.icon_lines:
            # Center the icon
            padded_icon = icon_line.center(card_width - 2)
            if is_selected:
                line_text = Text("║", style="cyan") + Text(padded_icon, style=name_style) + Text("║", style="cyan")
            else:
                line_text = Text("│", style="dim") + Text(padded_icon, style=name_style) + Text("│", style="dim")
            lines.append(line_text)

        # Agent name (may wrap if too long)
        name_display = self._truncate_name(agent.name, card_width - 4)
        name_text = Text(name_display.center(card_width - 2), style=name_style)
        if is_selected:
            name_line = Text("║", style="cyan") + name_text + Text("║", style="cyan")
        else:
            name_line = Text("│", style="dim") + name_text + Text("│", style="dim")
        lines.append(name_line)

        # Status line
        status_text = Text(f"({display_status})".center(card_width - 2), style=status_style)
        if is_selected:
            status_line = Text("║", style="cyan") + status_text + Text("║", style="cyan")
        else:
            status_line = Text("│", style="dim") + status_text + Text("│", style="dim")
        lines.append(status_line)

        # Bottom border with selection indicator
        if is_selected:
            bottom_border = Text("╚" + border_char * (card_width - 2) + "╝", style="cyan")
            # Add selection arrow
            lines.append(bottom_border)
            lines.append(Text("◄ SEL", style="bold cyan"))
        else:
            bottom_border = Text("└" + border_char * (card_width - 2) + "┘", style="dim")
            lines.append(bottom_border)

        # Add spacing between cards
        lines.append(Text(""))

        return Group(*lines)

    def _truncate_name(self, name: str, max_width: int) -> str:
        """Truncate agent name to fit width.

        For nested agents (e.g., "parent.child"), may truncate middle.

        Args:
            name: Agent name.
            max_width: Maximum width.

        Returns:
            Truncated name.
        """
        if len(name) <= max_width:
            return name

        # If name contains dot (nested agent), try to preserve both parts
        if "." in name:
            parts = name.split(".", 1)
            if len(parts) == 2:
                parent, child = parts
                # Try to show "parent...child"
                if len(parent) + len(child) + 3 <= max_width:
                    return f"{parent}...{child}"

        # Simple truncation with ellipsis
        return name[:max_width - 3] + "..."
