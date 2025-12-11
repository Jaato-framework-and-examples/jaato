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
                output_buffer=OutputBuffer(),  # Dedicated buffer
                history=[],  # Isolated history
                turn_accounting=[],  # Isolated accounting
                context_usage={},  # Isolated context metrics
                created_at=created_at or datetime.now(),
                completed_at=None
            )

            self._agents[agent_id] = agent_info
            self._agent_order.append(agent_id)

            # If this is the first agent (main), select it
            if len(self._agents) == 1:
                self._selected_agent_id = agent_id

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

    def get_selected_agent(self) -> Optional[AgentInfo]:
        """Get currently selected agent.

        Returns:
            Selected AgentInfo or None if no agents.
        """
        with self._lock:
            return self._agents.get(self._selected_agent_id)

    def cycle_selection(self) -> Optional[str]:
        """Cycle to next agent in list (for F2 key).

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
            status: New status ("active", "done", "error").
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
        function_calls: List[Dict[str, Any]]
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
