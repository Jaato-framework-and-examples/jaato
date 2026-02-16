# jaato-tui/renderers/base.py
"""Abstract base class for client renderers.

Defines the interface that all renderers (TUI, headless, etc.) must implement.
This enables the event handler to be renderer-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Renderer(ABC):
    """Abstract renderer interface for client output.

    Renderers receive structured events from the event handler and produce
    output in their target format (TUI, file, etc.).
    """

    # ==================== Lifecycle ====================

    @abstractmethod
    def start(self) -> None:
        """Initialize the renderer and prepare for output."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources and finalize output."""
        pass

    # ==================== Agent Management ====================

    @abstractmethod
    def on_agent_created(
        self,
        agent_id: str,
        agent_type: str,
        name: Optional[str] = None,
        profile_name: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
    ) -> None:
        """Handle creation of a new agent."""
        pass

    @abstractmethod
    def on_agent_status_changed(
        self,
        agent_id: str,
        status: str,  # "active", "done", "error"
    ) -> None:
        """Handle agent status change."""
        pass

    @abstractmethod
    def on_agent_completed(self, agent_id: str) -> None:
        """Handle agent completion (cleanup)."""
        pass

    # ==================== Output ====================

    @abstractmethod
    def on_agent_output(
        self,
        agent_id: str,
        source: str,
        text: str,
        mode: str,  # "write", "append", "replace"
    ) -> None:
        """Handle agent output text."""
        pass

    @abstractmethod
    def on_system_message(
        self,
        message: str,
        style: str = "system_info",
    ) -> None:
        """Handle system messages (initialization, errors, etc.)."""
        pass

    # ==================== Tool Execution ====================

    @abstractmethod
    def on_tool_start(
        self,
        agent_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> None:
        """Handle tool execution start."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def on_tool_output(
        self,
        agent_id: str,
        call_id: str,
        chunk: str,
    ) -> None:
        """Handle live tool output chunk."""
        pass

    # ==================== Permissions ====================

    @abstractmethod
    def on_permission_requested(
        self,
        agent_id: str,
        request_id: str,
        tool_name: str,
        call_id: Optional[str] = None,
        response_options: Optional[List[str]] = None,
    ) -> None:
        """Handle permission request (for display/logging)."""
        pass

    @abstractmethod
    def on_permission_resolved(
        self,
        agent_id: str,
        tool_name: str,
        granted: bool,
        method: str,  # How it was resolved (auto, user, etc.)
    ) -> None:
        """Handle permission resolution."""
        pass

    # ==================== Plan Management ====================

    @abstractmethod
    def on_plan_updated(
        self,
        agent_id: Optional[str],
        plan_data: Dict[str, Any],
    ) -> None:
        """Handle plan update.

        plan_data format:
        {
            "title": str,
            "steps": [
                {
                    "description": str,
                    "status": str,  # "pending", "in_progress", "completed", "failed"
                    "active_form": Optional[str],
                    "sequence": int,
                    "blocked_by": Optional[...],
                    "depends_on": Optional[...],
                }
            ],
            "progress": {
                "total": int,
                "completed": int,
                "percent": float,
            }
        }
        """
        pass

    @abstractmethod
    def on_plan_cleared(self, agent_id: Optional[str]) -> None:
        """Handle plan being cleared."""
        pass

    # ==================== Context ====================

    @abstractmethod
    def on_context_updated(
        self,
        agent_id: str,
        total_tokens: int,
        prompt_tokens: int,
        output_tokens: int,
        turns: int,
        percent_used: float,
    ) -> None:
        """Handle context usage update."""
        pass

    # ==================== Errors & Retries ====================

    @abstractmethod
    def on_error(self, message: str, details: Optional[str] = None) -> None:
        """Handle error event."""
        pass

    @abstractmethod
    def on_retry(
        self,
        attempt: int,
        max_attempts: int,
        reason: str,
        delay_seconds: float,
    ) -> None:
        """Handle retry event."""
        pass

    # ==================== Refresh ====================

    def refresh(self) -> None:
        """Refresh the display (no-op for non-interactive renderers)."""
        pass
