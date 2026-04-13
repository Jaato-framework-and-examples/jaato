"""Lifecycle plugin — implementation.

Single tool: ``signal_completion``.  When called, emits
``AgentCompletedEvent`` for the calling agent via the session's UI
hooks — the same mechanism subagents use when they finish.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from jaato_sdk.plugins.model_provider.types import ToolSchema

logger = logging.getLogger(__name__)


class LifecyclePlugin:
    """Provides agent lifecycle signaling tools.

    Auto-wired via ``set_session()`` during configure.  The tool
    accesses the session's ``_ui_hooks`` to emit the completion event.
    """

    @property
    def name(self) -> str:
        return "lifecycle"

    def initialize(self, config: Any = None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [
            ToolSchema(
                name="signal_completion",
                description=(
                    "Signal that you have finished all your work and have "
                    "nothing left to do.  This triggers downstream agents "
                    "(e.g. memory curator) and allows the session to be "
                    "cleaned up.  Call this as your very last action."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": (
                                "Brief summary of what was accomplished."
                            ),
                        },
                    },
                    "required": ["summary"],
                },
                discoverability="core",
            ),
        ]

    def get_executors(self) -> Dict[str, Any]:
        return {"signal_completion": self._execute_signal_completion}

    def get_auto_approved_tools(self) -> List[str]:
        return ["signal_completion"]

    def get_system_instructions(self) -> str:
        return ""

    def get_user_commands(self) -> List:
        return []

    def _execute_signal_completion(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Emit AgentCompletedEvent for the calling agent.

        Reads the session from the ContextVar and calls
        on_agent_completed on its UI hooks — the same path subagents
        use.
        """
        summary = args.get("summary", "")

        from shared.session_context import get_current_session

        try:
            session = get_current_session()
        except LookupError:
            return {"error": "No active session context"}

        hooks = getattr(session, '_ui_hooks', None)
        if not hooks or not hasattr(hooks, 'on_agent_completed'):
            return {"error": "No UI hooks available"}

        agent_id = getattr(session, '_agent_id', 'main')
        usage = session.get_context_usage() if hasattr(session, 'get_context_usage') else {}

        hooks.on_agent_completed(
            agent_id=agent_id,
            completed_at=datetime.now(),
            success=True,
            token_usage=usage.get('total_tokens'),
            turns_used=usage.get('turns'),
        )

        logger.info(
            "Agent %s signaled completion: %s",
            agent_id, summary[:80] if summary else "(no summary)",
        )

        return {
            "status": "completed",
            "agent_id": agent_id,
            "summary": summary,
        }


def create_plugin() -> LifecyclePlugin:
    return LifecyclePlugin()
