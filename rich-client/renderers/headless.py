# rich-client/renderers/headless.py
"""Headless file-based renderer for jaato client.

Writes output to per-agent files with ANSI formatting (viewable with `less -R`).
Plans are printed inline with full reprint on each update.
All permissions are auto-approved.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .base import Renderer


# Plan status symbols (same as plan_panel.py)
STATUS_SYMBOLS = {
    "pending": ("○", "dim"),
    "in_progress": ("◐", "yellow"),
    "completed": ("●", "green"),
    "failed": ("✗", "red"),
    "blocked": ("◌", "dim red"),
}


class HeadlessFileRenderer(Renderer):
    """Renderer that writes output to per-agent files.

    Creates files in {workspace}/jaato-headless-client-agents/:
    - main.log
    - {agent_id}_{name}.log (for subagents)

    Output format matches rich client's expanded tool blocks with ANSI colors.
    Plans are printed inline (no popup).
    """

    def __init__(self, workspace: Path, flush_immediately: bool = True):
        """Initialize the headless renderer.

        Args:
            workspace: Root workspace directory.
            flush_immediately: Whether to flush after each write (real-time output).
        """
        self.workspace = workspace
        self.output_dir = workspace / "jaato-headless-client-agents"
        self.flush_immediately = flush_immediately

        # Per-agent state
        self._files: Dict[str, TextIO] = {}
        self._consoles: Dict[str, Console] = {}
        self._agent_names: Dict[str, str] = {}  # agent_id -> name
        self._current_plans: Dict[str, Dict[str, Any]] = {}  # agent_id -> plan_data

        # Track active tools for output formatting
        self._active_tools: Dict[str, Dict[str, Any]] = {}  # call_id -> tool info
        # Step ID → step number mapping for human-readable display in tool args
        self._step_id_to_number: Dict[str, int] = {}

    # ==================== Lifecycle ====================

    def start(self) -> None:
        """Create output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def shutdown(self) -> None:
        """Close all open files."""
        for agent_id, file in self._files.items():
            try:
                # Write session end marker
                console = self._consoles.get(agent_id)
                if console:
                    console.print()
                    console.rule("[dim]Session ended[/dim]")
                file.close()
            except Exception:
                pass
        self._files.clear()
        self._consoles.clear()

    # ==================== File Management ====================

    def _get_console(self, agent_id: str) -> Console:
        """Get or create Rich Console for an agent."""
        if agent_id not in self._consoles:
            # Determine filename
            name = self._agent_names.get(agent_id, "")
            if agent_id == "main":
                filename = "main.log"
            elif name:
                # Sanitize name for filename
                safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
                filename = f"{agent_id}_{safe_name}.log"
            else:
                filename = f"{agent_id}.log"

            filepath = self.output_dir / filename
            file = open(filepath, "w", encoding="utf-8")
            self._files[agent_id] = file

            # Create Rich Console that writes to file with ANSI codes
            console = Console(
                file=file,
                force_terminal=True,  # Enable ANSI codes even for file output
                width=120,  # Fixed width for consistent formatting
                color_system="truecolor",
            )
            self._consoles[agent_id] = console

            # Write session header
            display_name = self._agent_names.get(agent_id, agent_id)
            console.rule(f"[bold]Agent: {display_name}[/bold]")
            console.print(f"[dim]Started: {datetime.now().isoformat()}[/dim]")
            console.print()

        return self._consoles[agent_id]

    def _flush(self, agent_id: str) -> None:
        """Flush the file buffer for an agent."""
        if self.flush_immediately and agent_id in self._files:
            self._files[agent_id].flush()

    # ==================== Agent Management ====================

    def on_agent_created(
        self,
        agent_id: str,
        agent_type: str,
        name: Optional[str] = None,
        profile_name: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
    ) -> None:
        """Handle creation of a new agent."""
        # Store name for file naming
        if name:
            self._agent_names[agent_id] = name

        console = self._get_console(agent_id)

        # Print agent info
        info_parts = [f"Type: {agent_type}"]
        if name:
            info_parts.append(f"Name: {name}")
        if profile_name:
            info_parts.append(f"Profile: {profile_name}")
        if parent_agent_id:
            parent_name = self._agent_names.get(parent_agent_id, parent_agent_id)
            info_parts.append(f"Parent: {parent_name}")

        console.print(Panel(
            "\n".join(info_parts),
            title="[bold cyan]Agent Created[/bold cyan]",
            border_style="cyan",
        ))
        console.print()
        self._flush(agent_id)

    def on_agent_status_changed(self, agent_id: str, status: str) -> None:
        """Handle agent status change."""
        console = self._get_console(agent_id)

        status_styles = {
            "active": ("▶", "green"),
            "done": ("✓", "green"),
            "error": ("✗", "red"),
        }
        symbol, style = status_styles.get(status, ("?", "yellow"))

        # Ensure we're on a new line (streaming output may not end with newline)
        console.print()
        console.print(f"[{style}]{symbol} Agent status: {status}[/{style}]")
        self._flush(agent_id)

    def on_agent_completed(self, agent_id: str) -> None:
        """Handle agent completion."""
        console = self._get_console(agent_id)
        console.print()
        console.rule("[dim]Agent completed[/dim]")
        self._flush(agent_id)

    # ==================== Output ====================

    def on_agent_output(
        self,
        agent_id: str,
        source: str,
        text: str,
        mode: str,
    ) -> None:
        """Handle agent output text."""
        console = self._get_console(agent_id)

        # Style based on source
        if source == "model":
            # Model output - render as-is (may contain markdown/ANSI)
            console.print(text, end="", markup=False, highlight=False)
        elif source == "tool":
            # Tool output - dim
            console.print(f"[dim]{text}[/dim]", end="")
        elif source == "system":
            # System messages
            console.print(f"[italic]{text}[/italic]", end="")
        else:
            # Default
            console.print(text, end="")

        self._flush(agent_id)

    def on_system_message(
        self, message: str, style: str = "system_info", agent_id: str = "main"
    ) -> None:
        """Handle system messages.

        Args:
            message: The message to display.
            style: Style name for formatting.
            agent_id: Target agent's log (default: main).
        """
        console = self._get_console(agent_id)

        style_map = {
            "system_info": "cyan",
            "system_warning": "yellow",
            "system_error": "red",
            "system_progress": "dim cyan",
            "system_version": "bold cyan",
            "system_init_error": "bold red",
        }
        rich_style = style_map.get(style, "dim")

        console.print(f"[{rich_style}]{message}[/{rich_style}]")
        self._flush(agent_id)

    # ==================== Tool Execution ====================

    def on_tool_start(
        self,
        agent_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> None:
        """Handle tool execution start."""
        console = self._get_console(agent_id)

        # Track active tool
        if call_id:
            self._active_tools[call_id] = {
                "agent_id": agent_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
            }

        # Format tool call with expanded args (always expanded in headless)
        console.print()
        console.print(f"[bold yellow]┌─ {tool_name}[/bold yellow]")

        # Print args (replace step_id UUIDs with human-readable step numbers)
        for key, value in tool_args.items():
            display_value = value
            if key == "step_id" and isinstance(value, str) and value in self._step_id_to_number:
                display_value = f"Step #{self._step_id_to_number[value]}"
            # Truncate long values
            str_value = str(display_value)
            if len(str_value) > 200:
                str_value = str_value[:200] + "..."
            console.print(f"[dim]│ {key}: {str_value}[/dim]")

        self._flush(agent_id)

    def on_tool_end(
        self,
        agent_id: str,
        tool_name: str,
        success: bool,
        duration_seconds: float,
        error_message: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> None:
        """Handle tool execution completion."""
        console = self._get_console(agent_id)

        # Remove from active tools
        if call_id and call_id in self._active_tools:
            del self._active_tools[call_id]

        # Ensure we're on a new line (tool output may not end with newline)
        console.print()

        # Format result
        if success:
            console.print(f"[bold green]└─ ✓ {tool_name}[/bold green] [dim]({duration_seconds:.2f}s)[/dim]")
        else:
            console.print(f"[bold red]└─ ✗ {tool_name}[/bold red] [dim]({duration_seconds:.2f}s)[/dim]")
            if error_message:
                console.print(f"[red]   Error: {error_message}[/red]")

        console.print()
        self._flush(agent_id)

    def on_tool_output(self, agent_id: str, call_id: str, chunk: str) -> None:
        """Handle live tool output chunk."""
        console = self._get_console(agent_id)
        # Print as dim indented output
        console.print(f"[dim]│ {chunk}[/dim]", end="")
        self._flush(agent_id)

    # ==================== Permissions ====================

    def on_permission_requested(
        self,
        agent_id: str,
        request_id: str,
        tool_name: str,
        call_id: Optional[str] = None,
        response_options: Optional[List[str]] = None,
    ) -> None:
        """Handle permission request (log only - headless auto-approves)."""
        console = self._get_console(agent_id)
        console.print(f"[yellow]⚡ Permission requested: {tool_name} (auto-approved)[/yellow]")
        self._flush(agent_id)

    def on_permission_resolved(
        self,
        agent_id: str,
        tool_name: str,
        granted: bool,
        method: str,
    ) -> None:
        """Handle permission resolution."""
        console = self._get_console(agent_id)
        if granted:
            console.print(f"[green]✓ Permission granted: {tool_name} ({method})[/green]")
        else:
            console.print(f"[red]✗ Permission denied: {tool_name} ({method})[/red]")
        self._flush(agent_id)

    # ==================== Plan Management ====================

    def on_plan_updated(
        self,
        agent_id: Optional[str],
        plan_data: Dict[str, Any],
    ) -> None:
        """Handle plan update - print full plan inline."""
        agent_id = agent_id or "main"
        console = self._get_console(agent_id)

        # Store current plan for reference
        self._current_plans[agent_id] = plan_data

        # Build step_id → step number mapping for tool args display
        for step in plan_data.get("steps", []):
            sid = step.get("step_id")
            seq = step.get("sequence")
            if sid and seq is not None:
                self._step_id_to_number[sid] = seq

        # Build plan display
        title = plan_data.get("title", "Plan")
        steps = plan_data.get("steps", [])
        progress = plan_data.get("progress", {})

        # Create step table
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("seq", style="dim", width=3)
        table.add_column("status", width=2)
        table.add_column("description")

        for step in steps:
            seq = str(step.get("sequence", ""))
            status = step.get("status", "pending")
            desc = step.get("description", "")

            # Get status symbol and style
            symbol, style = STATUS_SYMBOLS.get(status, ("?", "dim"))

            # Use active_form if in_progress
            if status == "in_progress" and step.get("active_form"):
                desc = step["active_form"]

            table.add_row(seq, f"[{style}]{symbol}[/{style}]", desc)

        # Progress info
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        percent = progress.get("percent", 0)
        progress_text = f"{completed}/{total} ({percent:.0f}%)"

        # Print as panel
        console.print()
        console.print(Panel(
            table,
            title=f"[bold]{title}[/bold]",
            subtitle=f"[dim]{progress_text}[/dim]",
            border_style="blue",
        ))
        console.print()
        self._flush(agent_id)

    def on_plan_cleared(self, agent_id: Optional[str]) -> None:
        """Handle plan being cleared."""
        agent_id = agent_id or "main"
        console = self._get_console(agent_id)

        if agent_id in self._current_plans:
            del self._current_plans[agent_id]

        console.print("[dim]Plan cleared[/dim]")
        self._flush(agent_id)

    # ==================== Context ====================

    def on_context_updated(
        self,
        agent_id: str,
        total_tokens: int,
        prompt_tokens: int,
        output_tokens: int,
        turns: int,
        percent_used: float,
    ) -> None:
        """Handle context usage update (logged periodically, not every update)."""
        # Only log if significant change (e.g., > 10% change)
        # For now, we'll skip context updates in headless to reduce noise
        pass

    # ==================== Errors & Retries ====================

    def on_error(self, message: str, details: Optional[str] = None) -> None:
        """Handle error event."""
        console = self._get_console("main")
        console.print(f"[bold red]ERROR: {message}[/bold red]")
        if details:
            console.print(f"[red]{details}[/red]")
        self._flush("main")

    def on_retry(
        self,
        attempt: int,
        max_attempts: int,
        reason: str,
        delay_seconds: float,
    ) -> None:
        """Handle retry event."""
        console = self._get_console("main")
        console.print(
            f"[yellow]Retry {attempt}/{max_attempts}: {reason} "
            f"(waiting {delay_seconds:.1f}s)[/yellow]"
        )
        self._flush("main")
