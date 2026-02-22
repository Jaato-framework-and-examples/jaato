"""Agent registry for managing agent state and isolation.

This module provides the central registry for tracking all agents (main and subagents)
with their isolated output buffers, conversation history, and accounting data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading

from output_buffer import OutputBuffer
from agent_icons import get_icon


@dataclass
class AgentInfo:
    """Information about a single agent (main or subagent).

    Each agent maintains completely isolated state:
    - Output buffer (display history)
    - Conversation history (messages)
    - Turn accounting (per-turn token usage)
    - Context usage (cumulative metrics)
    """

    # Identity
    agent_id: str
    name: str
    agent_type: str  # "main" | "subagent"
    profile_name: Optional[str]
    parent_agent_id: Optional[str]

    # Visual
    icon_lines: List[str]
    status: str  # "active" | "done" | "error"

    # Isolated state (per-agent)
    output_buffer: OutputBuffer
    history: List[Any] = field(default_factory=list)  # List[Message]
    turn_accounting: List[Dict[str, Any]] = field(default_factory=list)
    context_usage: Dict[str, Any] = field(default_factory=dict)
    plan_data: Optional[Dict[str, Any]] = None  # Plan state for this agent

    # GC configuration (per-agent)
    gc_threshold: Optional[float] = None  # GC trigger threshold percentage
    gc_strategy: Optional[str] = None  # GC strategy name (e.g., "truncate", "hybrid")
    gc_target_percent: Optional[float] = None  # Target usage after GC
    gc_continuous_mode: bool = False  # True if GC runs after every turn

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class AgentRegistry:
    """Registry for managing all agents and their state.

    Maintains a collection of AgentInfo objects with isolated state per agent.
    Provides selection management for UI navigation (F2 cycling).

    Thread-safe for concurrent access from main thread and background subagent threads.
    """

    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, AgentInfo] = {}
        self._agent_order: List[str] = []  # Maintains display order
        self._selected_agent_id: str = "main"
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._formatter_pipeline: Any = None  # Shared formatter pipeline for all agents
        # Pending events for agents not yet created (handles race conditions)
        # Each entry is a tuple: (event_type, *event_data)
        # - ("output", source, text, mode)
        # - ("tool_start", tool_name, tool_args, call_id)
        # - ("tool_end", tool_name, success, duration_seconds, error_message, call_id)
        # - ("tool_output", call_id, chunk)
        self._pending_events: Dict[str, List[tuple]] = {}  # agent_id -> [events...]

    def create_agent(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        profile_name: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
        icon_lines: Optional[List[str]] = None,
        created_at: Optional[datetime] = None
    ) -> None:
        """Create a new agent entry with isolated state.

        Args:
            agent_id: Unique identifier (e.g., "main", "subagent_1", "parent.child").
            name: Display name (e.g., "main", "code-assist").
            agent_type: "main" or "subagent".
            profile_name: Profile name if subagent, None for main.
            parent_agent_id: Parent agent ID if nested, None otherwise.
            icon_lines: Custom icon (3 lines) or None for default.
            created_at: Creation timestamp (defaults to now).
        """
        with self._lock:
            # Skip if agent already exists (prevent duplicates)
            if agent_id in self._agents:
                return

            # Resolve icon
            if icon_lines is None:
                icon_lines = get_icon(agent_type, profile_name)

            # Create agent info with isolated state
            agent_info = AgentInfo(
                agent_id=agent_id,
                name=name,
                agent_type=agent_type,
                profile_name=profile_name,
                parent_agent_id=parent_agent_id,
                status="active",
                icon_lines=icon_lines,
                output_buffer=OutputBuffer(agent_type=agent_type),  # Dedicated buffer
                history=[],  # Isolated history
                turn_accounting=[],  # Isolated accounting
                context_usage={},  # Isolated context metrics
                created_at=created_at or datetime.now(),
                completed_at=None
            )

            self._agents[agent_id] = agent_info
            self._agent_order.append(agent_id)

            # Apply formatter pipeline if set
            if self._formatter_pipeline and agent_info.output_buffer:
                agent_info.output_buffer.set_formatter_pipeline(self._formatter_pipeline)

            # Apply theme if set
            if self._theme and agent_info.output_buffer:
                agent_info.output_buffer.set_theme(self._theme)

            # If this is the first agent (main), select it
            if len(self._agents) == 1:
                self._selected_agent_id = agent_id

            # Flush any pending events for this agent (handles race condition where
            # events arrived before AgentCreatedEvent)
            if agent_id in self._pending_events:
                pending = self._pending_events.pop(agent_id)
                for event in pending:
                    event_type = event[0]
                    if event_type == "output":
                        _, source, text, mode = event
                        agent_info.output_buffer.append(source, text, mode)
                    elif event_type == "tool_start":
                        _, tool_name, tool_args, call_id = event
                        agent_info.output_buffer.add_active_tool(tool_name, tool_args, call_id=call_id)
                    elif event_type == "tool_end":
                        _, tool_name, success, duration_seconds, error_message, call_id, continuation_id, show_output, show_popup = event
                        agent_info.output_buffer.mark_tool_completed(
                            tool_name, success, duration_seconds, error_message,
                            call_id=call_id, continuation_id=continuation_id,
                            show_output=show_output, show_popup=show_popup,
                        )
                    elif event_type == "tool_output":
                        _, call_id, chunk = event
                        agent_info.output_buffer.append_tool_output(call_id, chunk)

    def queue_output(self, agent_id: str, source: str, text: str, mode: str) -> None:
        """Queue output for an agent that hasn't been created yet.

        Used when AgentOutputEvent arrives before AgentCreatedEvent (race condition).
        The queued output will be flushed to the agent's buffer when it's created.

        Args:
            agent_id: Target agent identifier.
            source: Source of the output ("model", plugin name, etc.)
            text: The output text.
            mode: "write" for new block, "append" to continue.
        """
        with self._lock:
            if agent_id not in self._pending_events:
                self._pending_events[agent_id] = []
            self._pending_events[agent_id].append(("output", source, text, mode))

    def queue_tool_start(self, agent_id: str, tool_name: str, tool_args: Dict, call_id: Optional[str] = None) -> None:
        """Queue tool start event for an agent that hasn't been created yet."""
        with self._lock:
            if agent_id not in self._pending_events:
                self._pending_events[agent_id] = []
            self._pending_events[agent_id].append(("tool_start", tool_name, tool_args, call_id))

    def queue_tool_end(
        self,
        agent_id: str,
        tool_name: str,
        success: bool,
        duration_seconds: float,
        error_message: Optional[str],
        call_id: Optional[str] = None,
        continuation_id: Optional[str] = None,
        show_output: Optional[bool] = None,
        show_popup: Optional[bool] = None,
    ) -> None:
        """Queue tool end event for an agent that hasn't been created yet."""
        with self._lock:
            if agent_id not in self._pending_events:
                self._pending_events[agent_id] = []
            self._pending_events[agent_id].append(("tool_end", tool_name, success, duration_seconds, error_message, call_id, continuation_id, show_output, show_popup))

    def queue_tool_output(self, agent_id: str, call_id: str, chunk: str) -> None:
        """Queue tool output event for an agent that hasn't been created yet."""
        with self._lock:
            if agent_id not in self._pending_events:
                self._pending_events[agent_id] = []
            self._pending_events[agent_id].append(("tool_output", call_id, chunk))

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent by ID.

        Args:
            agent_id: Agent identifier.

        Returns:
            AgentInfo or None if not found.
        """
        with self._lock:
            return self._agents.get(agent_id)

    def get_all_agents(self) -> List[AgentInfo]:
        """Get all agents in display order.

        Returns:
            List of AgentInfo objects ordered as: main first, then subagents chronologically.
        """
        with self._lock:
            return [self._agents[agent_id] for agent_id in self._agent_order if agent_id in self._agents]

    def set_keybinding_config_all(self, config: Any) -> None:
        """Set keybinding config on all agent output buffers.

        Args:
            config: KeybindingConfig instance for dynamic UI hints.
        """
        with self._lock:
            for agent in self._agents.values():
                if agent.output_buffer:
                    agent.output_buffer.set_keybinding_config(config)

    def set_formatter_pipeline_all(self, pipeline: Any) -> None:
        """Set formatter pipeline on all agent output buffers.

        Args:
            pipeline: FormatterPipeline instance for output processing.
        """
        self._formatter_pipeline = pipeline
        with self._lock:
            for agent in self._agents.values():
                if agent.output_buffer:
                    agent.output_buffer.set_formatter_pipeline(pipeline)

    # Legacy alias for backwards compatibility
    def set_output_formatter_all(self, formatter: Any) -> None:
        """Deprecated: Use set_formatter_pipeline_all instead."""
        self.set_formatter_pipeline_all(formatter)

    def set_theme_all(self, theme: Any) -> None:
        """Set theme on all agent output buffers.

        Args:
            theme: ThemeConfig instance for styling.
        """
        self._theme = theme
        with self._lock:
            for agent in self._agents.values():
                if agent.output_buffer:
                    agent.output_buffer.set_theme(theme)

    def format_text(self, text: str, format_hint: Optional[str] = None) -> str:
        """Format text through the formatter pipeline.

        Args:
            text: Text to format.
            format_hint: Optional hint for formatting (e.g., "diff").

        Returns:
            Formatted text with ANSI codes, or original text if no pipeline.
        """
        if self._formatter_pipeline and hasattr(self._formatter_pipeline, 'format'):
            return self._formatter_pipeline.format(text, format_hint=format_hint)
        return text

    def get_selected_agent(self) -> Optional[AgentInfo]:
        """Get currently selected agent.

        Returns:
            Selected AgentInfo or None if no agents.
        """
        with self._lock:
            return self._agents.get(self._selected_agent_id)

    def select_agent(self, agent_id: str) -> bool:
        """Select an agent by ID.

        Args:
            agent_id: Agent ID to select.

        Returns:
            True if agent was selected, False if agent doesn't exist.
        """
        with self._lock:
            if agent_id in self._agents:
                self._selected_agent_id = agent_id
                return True
            return False

    def cycle_selection(self) -> Optional[str]:
        """Cycle to next agent in list (for cycle_agents keybinding).

        Cycles: main → subagent1 → subagent2 → ... → main

        Returns:
            New selected agent_id or None if no agents.
        """
        with self._lock:
            if not self._agent_order:
                return None

            try:
                current_idx = self._agent_order.index(self._selected_agent_id)
                next_idx = (current_idx + 1) % len(self._agent_order)
                self._selected_agent_id = self._agent_order[next_idx]
                return self._selected_agent_id
            except ValueError:
                # Current selection not in list - select first
                self._selected_agent_id = self._agent_order[0]
                return self._selected_agent_id

    def update_status(self, agent_id: str, status: str) -> None:
        """Update agent's status.

        Args:
            agent_id: Which agent to update.
            status: New status ("active", "done", "error", "idle").
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.status = status

    def mark_completed(self, agent_id: str, completed_at: Optional[datetime] = None) -> None:
        """Mark agent as completed.

        Args:
            agent_id: Which agent completed.
            completed_at: Completion timestamp (defaults to now).
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.status = "done"
                agent.completed_at = completed_at or datetime.now()

    def get_buffer(self, agent_id: str) -> Optional[OutputBuffer]:
        """Get agent's output buffer.

        Args:
            agent_id: Which agent's buffer to get.

        Returns:
            OutputBuffer or None if agent not found.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            return agent.output_buffer if agent else None

    def update_turn_accounting(
        self,
        agent_id: str,
        turn_number: int,
        prompt_tokens: int,
        output_tokens: int,
        total_tokens: int,
        duration_seconds: float,
        function_calls: List[Dict[str, Any]],
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
    ) -> None:
        """Update agent's turn accounting.

        Args:
            agent_id: Which agent's accounting to update.
            turn_number: Turn index (0-based).
            prompt_tokens: Tokens in prompt.
            output_tokens: Tokens in response.
            total_tokens: Sum of prompt + output.
            duration_seconds: Turn duration.
            function_calls: List of function call stats.
            cache_read_tokens: Tokens read from prompt cache (reduced cost).
                None when the provider does not support caching.
            cache_creation_tokens: Tokens written to prompt cache.
                None when the provider does not support caching.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return

            turn_data = {
                'turn': turn_number,
                'prompt': prompt_tokens,
                'output': output_tokens,
                'total': total_tokens,
                'duration_seconds': duration_seconds,
                'function_calls': function_calls
            }
            if cache_read_tokens is not None:
                turn_data['cache_read'] = cache_read_tokens
            if cache_creation_tokens is not None:
                turn_data['cache_creation'] = cache_creation_tokens

            # Ensure list is long enough
            while len(agent.turn_accounting) <= turn_number:
                agent.turn_accounting.append({})

            agent.turn_accounting[turn_number] = turn_data

    def update_context_usage(
        self,
        agent_id: str,
        total_tokens: int,
        prompt_tokens: int,
        output_tokens: int,
        turns: int,
        percent_used: float
    ) -> None:
        """Update agent's context usage metrics.

        Args:
            agent_id: Which agent's context to update.
            total_tokens: Total tokens used.
            prompt_tokens: Cumulative prompt tokens.
            output_tokens: Cumulative output tokens.
            turns: Number of turns.
            percent_used: Percentage of context window used.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return

            agent.context_usage = {
                'total_tokens': total_tokens,
                'prompt_tokens': prompt_tokens,
                'output_tokens': output_tokens,
                'turns': turns,
                'percent_used': percent_used
            }

    def update_history(self, agent_id: str, history: List[Any]) -> None:
        """Update agent's conversation history.

        Args:
            agent_id: Which agent's history to update.
            history: Complete conversation history (List[Message]).
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return

            # Store a snapshot (copy to avoid reference issues)
            agent.history = list(history)

    # Convenience methods for selected agent

    def get_selected_buffer(self) -> Optional[OutputBuffer]:
        """Get selected agent's output buffer."""
        with self._lock:
            agent = self.get_selected_agent()
            return agent.output_buffer if agent else None

    def get_selected_history(self) -> List[Any]:
        """Get selected agent's conversation history."""
        with self._lock:
            agent = self.get_selected_agent()
            return agent.history if agent else []

    def get_selected_context_usage(self) -> Dict[str, Any]:
        """Get selected agent's context usage."""
        with self._lock:
            agent = self.get_selected_agent()
            return agent.context_usage if agent else {}

    def update_gc_config(
        self,
        agent_id: str,
        threshold: Optional[float],
        strategy: Optional[str] = None,
        target_percent: Optional[float] = None,
        continuous_mode: bool = False
    ) -> None:
        """Update an agent's GC configuration.

        Args:
            agent_id: The agent to update
            threshold: GC trigger threshold percentage (e.g., 80.0)
            strategy: GC strategy name (e.g., "truncate", "hybrid")
            target_percent: Target usage after GC (e.g., 60.0)
            continuous_mode: True if GC runs after every turn
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.gc_threshold = threshold
                agent.gc_strategy = strategy
                agent.gc_target_percent = target_percent
                agent.gc_continuous_mode = continuous_mode

    def get_selected_gc_config(self) -> tuple[Optional[float], Optional[str], Optional[float], bool]:
        """Get selected agent's GC configuration.

        Returns:
            Tuple of (gc_threshold, gc_strategy, gc_target_percent, gc_continuous_mode).
        """
        with self._lock:
            agent = self.get_selected_agent()
            if agent:
                return agent.gc_threshold, agent.gc_strategy, agent.gc_target_percent, agent.gc_continuous_mode
            return None, None, None, False

    def get_selected_turn_accounting(self) -> List[Dict[str, Any]]:
        """Get selected agent's turn accounting."""
        with self._lock:
            agent = self.get_selected_agent()
            return agent.turn_accounting if agent else []

    def get_selected_agent_id(self) -> str:
        """Get currently selected agent's ID."""
        with self._lock:
            return self._selected_agent_id

    def get_selected_agent_name(self) -> str:
        """Get currently selected agent's name."""
        with self._lock:
            agent = self.get_selected_agent()
            return agent.name if agent else "main"

    def get_selected_plan_data(self) -> Optional[Dict[str, Any]]:
        """Get selected agent's plan data."""
        with self._lock:
            agent = self.get_selected_agent()
            return agent.plan_data if agent else None

    def update_plan(self, agent_id: str, plan_data: Optional[Dict[str, Any]]) -> None:
        """Update agent's plan data.

        Args:
            agent_id: Which agent's plan to update.
            plan_data: Plan status dict, or None to clear.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return
            agent.plan_data = plan_data

    def clear_plan(self, agent_id: str) -> None:
        """Clear agent's plan data.

        Args:
            agent_id: Which agent's plan to clear.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.plan_data = None

    def get_all_agent_ids(self) -> List[str]:
        """Get all agent IDs in display order.

        Returns:
            List of agent IDs.
        """
        with self._lock:
            return list(self._agent_order)

    def find_agent_id_by_name(self, name: str) -> Optional[str]:
        """Find agent_id by profile/agent name.

        This is useful for mapping from TodoPlugin's agent_name (profile name)
        to the actual agent_id in the registry.

        Args:
            name: Agent/profile name to search for.

        Returns:
            agent_id if found, None otherwise.
        """
        with self._lock:
            for agent_id, agent in self._agents.items():
                if agent.name == name:
                    return agent_id
            return None
