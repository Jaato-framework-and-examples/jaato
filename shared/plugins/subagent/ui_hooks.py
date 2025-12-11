"""UI hooks protocol for agent lifecycle integration.

This module defines the protocol for integrating rich terminal UIs with the
agent system, enabling visualization of main agent and subagent execution.

The hooks allow UIs to:
- Track agent creation and lifecycle
- Capture per-agent output in isolated buffers
- Monitor per-agent token usage and context consumption
- Maintain per-agent conversation history

Both main agent and subagents use the same hook interface.
"""

from typing import Protocol, Dict, Any, Optional, List
from datetime import datetime


class AgentUIHooks(Protocol):
    """Protocol for UI integration with agent lifecycle.

    These hooks allow the UI to track agent creation, execution, and completion
    for visualization purposes (e.g., agent panel in rich client).

    Hooks are called from both main agent and subagent execution paths.
    All hooks are optional - if not implemented, agents run normally without
    UI integration.

    Thread Safety:
        All hooks may be called from background threads (especially for subagents).
        Implementations must be thread-safe.
    """

    def on_agent_created(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        profile_name: Optional[str],
        parent_agent_id: Optional[str],
        icon_lines: Optional[List[str]],
        created_at: datetime
    ) -> None:
        """Called when a new agent is created.

        Args:
            agent_id: Unique identifier.
                     Format: "main" for main agent,
                            "subagent_1", "subagent_2" for top-level subagents,
                            "parent.child" for nested subagents.
            agent_name: Display name (e.g., "main", "code-assist", "code-assist.analyzer").
            agent_type: "main" or "subagent".
            profile_name: Profile name if subagent (e.g., "code_assistant"), None for main.
            parent_agent_id: Parent agent's ID if nested subagent, None for main or top-level.
            icon_lines: Custom ASCII art icon (3 lines) or None for default.
            created_at: Creation timestamp.
        """
        ...

    def on_agent_output(
        self,
        agent_id: str,
        source: str,
        text: str,
        mode: str
    ) -> None:
        """Called when agent produces output.

        Args:
            agent_id: Which agent produced this output.
            source: Output source: "model" for model responses, "user" for user input,
                   or plugin name for tool output (e.g., "cli", "mcp").
            text: Output text content.
            mode: "write" for new output block, "append" to continue previous block.
        """
        ...

    def on_agent_status_changed(
        self,
        agent_id: str,
        status: str,
        error: Optional[str] = None
    ) -> None:
        """Called when agent status changes.

        Args:
            agent_id: Which agent's status changed.
            status: New status: "active", "done", or "error".
            error: Error message if status is "error", None otherwise.
        """
        ...

    def on_agent_completed(
        self,
        agent_id: str,
        completed_at: datetime,
        success: bool,
        token_usage: Optional[Dict[str, int]] = None,
        turns_used: Optional[int] = None
    ) -> None:
        """Called when agent completes execution.

        Args:
            agent_id: Which agent completed.
            completed_at: Completion timestamp.
            success: True if agent succeeded, False if errored.
            token_usage: Dict with "prompt_tokens", "output_tokens", "total_tokens".
                        None if not available.
            turns_used: Number of conversation turns used. None if not available.
        """
        ...

    def on_agent_turn_completed(
        self,
        agent_id: str,
        turn_number: int,
        prompt_tokens: int,
        output_tokens: int,
        total_tokens: int,
        duration_seconds: float,
        function_calls: List[Dict[str, Any]]
    ) -> None:
        """Called after each conversation turn completes.

        Enables per-agent, per-turn token accounting.

        Args:
            agent_id: Which agent completed the turn.
            turn_number: Turn index (0-based).
            prompt_tokens: Tokens consumed by the prompt.
            output_tokens: Tokens generated in the response.
            total_tokens: Sum of prompt_tokens + output_tokens.
            duration_seconds: Time taken for the turn.
            function_calls: List of function calls made during the turn,
                          each with 'name' and 'duration_seconds' keys.
        """
        ...

    def on_agent_context_updated(
        self,
        agent_id: str,
        total_tokens: int,
        prompt_tokens: int,
        output_tokens: int,
        turns: int,
        percent_used: float
    ) -> None:
        """Called when agent's context usage changes.

        Enables per-agent context tracking.

        Args:
            agent_id: Which agent's context updated.
            total_tokens: Total tokens used.
            prompt_tokens: Cumulative prompt tokens.
            output_tokens: Cumulative output tokens.
            turns: Number of turns.
            percent_used: Percentage of context window used.
        """
        ...

    def on_agent_history_updated(
        self,
        agent_id: str,
        history: List[Any]
    ) -> None:
        """Called when agent's conversation history changes (after each turn).

        Enables per-agent history isolation.

        Args:
            agent_id: Which agent's history updated.
            history: Complete conversation history snapshot (List[Message]).
        """
        ...
