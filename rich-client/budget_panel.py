"""Budget panel for instruction source token tracking.

Displays token usage by instruction source layer with drill-down capability
for Plugin (per-tool) and Conversation (per-turn) breakdowns.

Supports multi-agent views: Total (aggregated), Main, and subagents.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.table import Table

from keybindings import KeyBinding, format_key_for_display

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from theme import ThemeConfig


class BudgetPanel:
    """Renders instruction budget as a table with drill-down support."""

    # Source display names
    SOURCE_NAMES = {
        "system": "System",
        "plugin": "Plugin",
        "enrichment": "Enrichment",
        "conversation": "Conversation",
    }

    def __init__(self, toggle_key: Optional[KeyBinding] = None):
        # Budget data keyed by agent_id
        self._budgets: Dict[str, Dict[str, Any]] = {}
        self._visible: bool = False
        self._toggle_key = toggle_key or "c-b"
        self._theme: Optional["ThemeConfig"] = None

        # View state
        self._selected_agent_index: int = 0  # 0 = Total, 1 = Main, 2+ = subagents
        self._agent_order: List[str] = []  # ["_total", "main", "subagent_1", ...]
        self._drill_down_source: Optional[str] = None  # None = top level, or source name
        self._selected_row: int = 0  # Currently selected row in the table
        self._source_order: List[str] = ["system", "plugin", "enrichment", "conversation"]

    def set_theme(self, theme: "ThemeConfig") -> None:
        """Set the theme configuration for styling."""
        self._theme = theme

    def _style(self, semantic_name: str) -> str:
        """Get a Rich style string from the theme."""
        if self._theme:
            style = self._theme.get_rich_style(semantic_name)
            if style:
                return style
        return ""

    def _get_border_style(self) -> str:
        """Get style for panel border from theme."""
        return self._theme.get_rich_style("budget_panel_border") if self._theme else ""

    def _get_header_style(self) -> str:
        """Get style for table headers from theme."""
        return self._theme.get_rich_style("budget_header") if self._theme else ""

    def _get_gc_style(self, policy: str) -> str:
        """Get style for GC policy indicator from theme."""
        return self._theme.get_rich_style(f"budget_gc_{policy}") if self._theme else ""

    def _get_popup_background_style(self) -> str:
        """Get style for popup background from theme."""
        return self._theme.get_rich_style("budget_popup_background") if self._theme else ""

    @property
    def is_visible(self) -> bool:
        """Whether the budget panel is currently visible."""
        return self._visible

    def toggle(self) -> None:
        """Toggle panel visibility."""
        self._visible = not self._visible
        if self._visible:
            # Reset to top-level view when opening
            self._drill_down_source = None
            self._selected_row = 0

    def hide(self) -> None:
        """Hide the panel."""
        self._visible = False

    def update_budget(self, agent_id: str, budget_snapshot: Dict[str, Any]) -> None:
        """Update budget data for an agent.

        Args:
            agent_id: Agent identifier ("main", "subagent_1", etc.)
            budget_snapshot: Budget snapshot from InstructionBudget.snapshot()
        """
        self._budgets[agent_id] = budget_snapshot
        self._rebuild_agent_order()

    def clear_budget(self, agent_id: str) -> None:
        """Remove budget data for an agent."""
        if agent_id in self._budgets:
            del self._budgets[agent_id]
            self._rebuild_agent_order()

    def _rebuild_agent_order(self) -> None:
        """Rebuild the agent order list for tab navigation."""
        self._agent_order = ["_total"]  # Total always first
        if "main" in self._budgets:
            self._agent_order.append("main")
        # Add subagents in sorted order
        subagents = sorted(k for k in self._budgets.keys() if k != "main")
        self._agent_order.extend(subagents)

        # Clamp selection index
        if self._selected_agent_index >= len(self._agent_order):
            self._selected_agent_index = max(0, len(self._agent_order) - 1)

    def cycle_agent(self, forward: bool = True) -> None:
        """Cycle through agent views (Total, Main, subagents).

        Args:
            forward: True for next agent, False for previous.
        """
        if not self._agent_order:
            return
        if forward:
            self._selected_agent_index = (self._selected_agent_index + 1) % len(self._agent_order)
        else:
            self._selected_agent_index = (self._selected_agent_index - 1) % len(self._agent_order)
        # Reset view state when switching agents
        self._drill_down_source = None
        self._selected_row = 0

    def drill_down(self) -> bool:
        """Enter drill-down view for the selected source.

        Returns:
            True if drill-down was entered, False if not applicable.
        """
        # Only Plugin and Conversation support drill-down
        if self._drill_down_source:
            # Already in drill-down, can't go deeper
            return False

        selected_source = self._get_selected_source()
        if selected_source in ("system", "plugin", "conversation"):
            # Check if this source has children
            selected_agent = self._get_selected_agent_id()
            if selected_agent == "_total":
                budget = self._get_aggregated_budget()
            else:
                budget = self._budgets.get(selected_agent, {})

            entries = budget.get("entries", {})
            source_entry = entries.get(selected_source, {})
            children = source_entry.get("children", {})

            if children:
                self._drill_down_source = selected_source
                self._selected_row = 0  # Reset row selection for drill-down view
                return True
        return False

    def drill_up(self) -> bool:
        """Exit drill-down view, return to top level.

        Returns:
            True if exited drill-down, False if already at top level.
        """
        if self._drill_down_source:
            self._drill_down_source = None
            self._selected_row = 0  # Reset to first row
            return True
        return False

    def nav_up(self) -> None:
        """Navigate to previous row."""
        if self._selected_row > 0:
            self._selected_row -= 1

    def nav_down(self) -> None:
        """Navigate to next row."""
        max_row = self._get_max_row()
        if self._selected_row < max_row:
            self._selected_row += 1

    def _get_selected_source(self) -> Optional[str]:
        """Get the source name at the currently selected row."""
        if self._selected_row < len(self._source_order):
            return self._source_order[self._selected_row]
        return None

    def _get_max_row(self) -> int:
        """Get the maximum valid row index for current view."""
        if self._drill_down_source:
            # In drill-down view, count children
            selected_agent = self._get_selected_agent_id()
            if selected_agent == "_total":
                budget = self._get_aggregated_budget()
            else:
                budget = self._budgets.get(selected_agent, {})

            entries = budget.get("entries", {})
            source_entry = entries.get(self._drill_down_source, {})
            children = source_entry.get("children", {})
            # Count only children with non-zero tokens (matches drill-down filtering)
            non_zero_count = sum(1 for v in children.values() if v.get("tokens", 0) > 0)
            return max(0, non_zero_count - 1)
        else:
            # Top level - 5 sources
            return len(self._source_order) - 1

    # Keep old method names for compatibility but redirect to new ones
    def scroll_up(self, lines: int = 1) -> None:
        """Navigate up (legacy name for compatibility)."""
        self.nav_up()

    def scroll_down(self, lines: int = 1) -> None:
        """Navigate down (legacy name for compatibility)."""
        self.nav_down()

    def _get_selected_agent_id(self) -> Optional[str]:
        """Get the currently selected agent ID."""
        if not self._agent_order or self._selected_agent_index >= len(self._agent_order):
            return None
        return self._agent_order[self._selected_agent_index]

    def _get_aggregated_budget(self) -> Dict[str, Any]:
        """Get aggregated budget across all agents (for Total view)."""
        if not self._budgets:
            return {}

        # Aggregate entries by source
        aggregated_entries = {}
        total_tokens = 0

        for budget in self._budgets.values():
            entries = budget.get("entries", {})
            for source, entry in entries.items():
                if source not in aggregated_entries:
                    aggregated_entries[source] = {
                        "source": source,
                        "tokens": 0,
                        "total_tokens": 0,
                        "gc_policy": entry.get("gc_policy", "partial"),
                        "gc_eligible_tokens": 0,
                        "indicator": entry.get("indicator", "◐"),
                    }
                aggregated_entries[source]["tokens"] += entry.get("tokens", 0)
                aggregated_entries[source]["total_tokens"] += entry.get("total_tokens", 0)
                aggregated_entries[source]["gc_eligible_tokens"] += entry.get("gc_eligible_tokens", 0)
            total_tokens += budget.get("total_tokens", 0)

        return {
            "agent_id": "_total",
            "total_tokens": total_tokens,
            "entries": aggregated_entries,
            # No context_limit or utilization for aggregated view
        }

    def _render_usage_bar(self, tokens: int, max_tokens: int, width: int = 18) -> Text:
        """Render a usage bar visualization.

        Args:
            tokens: Current token count.
            max_tokens: Maximum tokens (for percentage calculation).
            width: Bar width in characters.

        Returns:
            Rich Text object with usage bar.
        """
        if max_tokens <= 0:
            # For aggregated view, show distribution bar instead
            return Text("░" * width, style="dim")

        filled = int((tokens / max_tokens) * width) if max_tokens > 0 else 0
        filled = min(filled, width)

        bar = Text()
        bar.append("█" * filled, style="cyan")
        bar.append("░" * (width - filled), style="dim")
        return bar

    def _format_tokens(self, tokens: int) -> str:
        """Format token count for display."""
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens / 1_000:.1f}K"
        return str(tokens)

    def render(self, available_height: int, available_width: int) -> Panel:
        """Render the budget panel.

        Args:
            available_height: Maximum height in lines.
            available_width: Maximum width in characters.

        Returns:
            Rich Panel containing the budget display.
        """
        # Handle empty state
        if not self._budgets:
            return Panel(
                Text("No token usage data yet.\nSend a message to start tracking.", style="dim italic"),
                title="[bold]Token Usage[/bold]",
                border_style=self._get_border_style(),
                style=self._get_popup_background_style(),
                padding=(0, 1),
            )

        selected_agent = self._get_selected_agent_id()

        if selected_agent == "_total":
            budget = self._get_aggregated_budget()
            title_suffix = "Total"
        else:
            budget = self._budgets.get(selected_agent, {})
            title_suffix = selected_agent or "Unknown"

        # Build title with token count
        total_tokens = budget.get("total_tokens", 0)
        context_limit = budget.get("context_limit", 0)
        utilization = budget.get("utilization_percent", 0)

        if context_limit > 0 and selected_agent != "_total":
            title = f"Token Usage ({self._format_tokens(total_tokens)} / {self._format_tokens(context_limit)} = {utilization:.1f}%)"
        else:
            title = f"Token Usage (Total: {self._format_tokens(total_tokens)} tokens)"

        # Build content
        if self._drill_down_source:
            content = self._render_drill_down(budget, available_width - 4)
        else:
            content = self._render_top_level(budget, available_width - 4)

        # Build agent tabs
        tabs = self._render_agent_tabs()

        # Combine content with tabs
        full_content = Group(content, Text(""), tabs)

        return Panel(
            full_content,
            title=f"[bold]{title}[/bold]",
            subtitle=f"[dim]{title_suffix}[/dim]",
            border_style=self._get_border_style(),
            style=self._get_popup_background_style(),
            padding=(0, 1),
        )

    def _render_top_level(self, budget: Dict[str, Any], width: int) -> Table:
        """Render top-level source breakdown."""
        table = Table(
            show_header=True,
            header_style=self._get_header_style(),
            box=None,
            padding=(0, 1),
            expand=True,
        )

        table.add_column("Source", style="bold", width=14)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("GC", justify="center", width=4)
        table.add_column("Usage", width=20)

        entries = budget.get("entries", {})
        total_tokens = budget.get("total_tokens", 1)  # Avoid div by zero
        context_limit = budget.get("context_limit", 0)

        for row_idx, source in enumerate(self._source_order):
            entry = entries.get(source, {})
            if not entry:
                continue

            is_selected = row_idx == self._selected_row
            name = self.SOURCE_NAMES.get(source, source.title())
            tokens = entry.get("total_tokens", entry.get("tokens", 0))
            gc_policy = entry.get("gc_policy", "partial")
            indicator = entry.get("indicator", "?")

            # Usage bar - relative to context_limit if available, else total
            max_for_bar = context_limit if context_limit > 0 else total_tokens
            usage_bar = self._render_usage_bar(tokens, max_for_bar)

            # Check if has children (drillable)
            children = entry.get("children", {})
            drill_indicator = " ▸" if children else ""

            # Apply selection highlighting
            if is_selected:
                name_text = Text(f"▶ {name}{drill_indicator}", style="bold reverse")
                tokens_text = Text(self._format_tokens(tokens), style="reverse")
                gc_text = Text(indicator, style=f"{self._get_gc_style(gc_policy)} reverse")
            else:
                name_text = Text(f"  {name}{drill_indicator}")
                tokens_text = Text(self._format_tokens(tokens))
                gc_text = Text(indicator, style=self._get_gc_style(gc_policy))

            table.add_row(name_text, tokens_text, gc_text, usage_bar)

        # Add legend
        table.add_row("", "", "", "")
        legend = Text()
        legend.append("🔒", style=self._get_gc_style("locked"))
        legend.append(" locked  ", style="dim")
        legend.append("◐", style=self._get_gc_style("partial"))
        legend.append(" partial  ", style="dim")
        legend.append("○", style=self._get_gc_style("ephemeral"))
        legend.append(" ephemeral", style="dim")
        table.add_row(legend, "", "", "")

        # Navigation hint
        hint = Text("↑↓ navigate  ", style="dim")
        hint.append("Enter", style="dim bold")
        hint.append(" drill down  ", style="dim")
        hint.append("ESC", style="dim bold")
        hint.append(" close", style="dim")
        table.add_row(hint, "", "", "")

        return table

    def _render_drill_down(self, budget: Dict[str, Any], width: int) -> Table:
        """Render drill-down view for a specific source."""
        entries = budget.get("entries", {})
        source_entry = entries.get(self._drill_down_source, {})
        children = source_entry.get("children", {})

        source_name = self.SOURCE_NAMES.get(self._drill_down_source, self._drill_down_source)
        source_tokens = source_entry.get("total_tokens", 0)

        table = Table(
            show_header=True,
            header_style=self._get_header_style(),
            box=None,
            padding=(0, 1),
            expand=True,
            title=f"[bold]{source_name} Breakdown ({self._format_tokens(source_tokens)} tokens)[/bold]",
        )

        if self._drill_down_source == "plugin":
            table.add_column("Tool", style="bold", width=20)
        else:
            table.add_column("Turn", style="bold", width=20)

        table.add_column("Tokens", justify="right", width=8)
        table.add_column("GC", justify="center", width=4)
        table.add_column("Usage", width=18)

        # Sort children by tokens descending, filter out 0-token entries
        sorted_children = sorted(
            ((k, v) for k, v in children.items() if v.get("tokens", 0) > 0),
            key=lambda x: x[1].get("tokens", 0),
            reverse=True
        )

        for row_idx, (child_id, child) in enumerate(sorted_children):
            is_selected = row_idx == self._selected_row
            label = child.get("label", child_id)
            tokens = child.get("tokens", 0)
            gc_policy = child.get("gc_policy", "ephemeral")
            indicator = child.get("indicator", "?")

            usage_bar = self._render_usage_bar(tokens, source_tokens)

            # Apply selection highlighting
            if is_selected:
                label_text = Text(f"▶ {label}", style="bold reverse")
                tokens_text = Text(self._format_tokens(tokens), style="reverse")
                gc_text = Text(indicator, style=f"{self._get_gc_style(gc_policy)} reverse")
            else:
                label_text = Text(f"  {label}")
                tokens_text = Text(self._format_tokens(tokens))
                gc_text = Text(indicator, style=self._get_gc_style(gc_policy))

            table.add_row(label_text, tokens_text, gc_text, usage_bar)

        # Add legend (same as top level)
        table.add_row("", "", "", "")
        legend = Text()
        legend.append("🔒", style=self._get_gc_style("locked"))
        legend.append(" locked  ", style="dim")
        legend.append("◐", style=self._get_gc_style("partial"))
        legend.append(" partial  ", style="dim")
        legend.append("○", style=self._get_gc_style("ephemeral"))
        legend.append(" ephemeral", style="dim")
        table.add_row(legend, "", "", "")

        # Navigation hint with ordering explanation
        hint = Text("Sorted by token count (0-token entries hidden)  ", style="dim italic")
        table.add_row(hint, "", "", "")
        nav_hint = Text("↑↓ navigate  ", style="dim")
        nav_hint.append("ESC", style="dim bold")
        nav_hint.append(" back", style="dim")
        table.add_row(nav_hint, "", "", "")

        return table

    def _render_agent_tabs(self) -> Text:
        """Render agent selection tabs."""
        tabs = Text()

        for i, agent_id in enumerate(self._agent_order):
            is_selected = i == self._selected_agent_index

            if agent_id == "_total":
                label = "Total"
            elif agent_id == "main":
                label = "Main"
            else:
                label = agent_id

            if is_selected:
                tabs.append(f"[{label}]", style="bold reverse")
            else:
                tabs.append(f" {label} ", style="dim")

            tabs.append(" ")

        # Add navigation hint
        tabs.append("      TAB →", style="dim")

        return tabs

    def has_data(self) -> bool:
        """Check if there's any budget data to display."""
        return bool(self._budgets)

    def get_hint_text(self) -> str:
        """Get hint text for status bar."""
        key_display = format_key_for_display(self._toggle_key)
        return f"{key_display}:budget"
