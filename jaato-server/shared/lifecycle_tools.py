"""Agent lifecycle tools — registered as core tools in JaatoSession.

Provides ``signal_completion``, which lets the main agent declare its
work is done.  This emits ``AgentCompletedEvent`` through the
session's UI hooks — the same mechanism subagents use — enabling
downstream reactors (e.g. memory-advisor) to trigger.

Subagents get completion signaling for free from the subagent plugin
which controls their lifecycle.  The main agent has no host, so it
needs this tool to signal explicitly.

Registered as a core tool (not a plugin) so it is available regardless
of the profile's plugin list.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, TYPE_CHECKING

from jaato_sdk.plugins.model_provider.types import ToolSchema

if TYPE_CHECKING:
    from shared.jaato_session import JaatoSession

logger = logging.getLogger(__name__)


class LifecycleTools:
    """Agent lifecycle signaling tools.

    Instantiated per-session in ``JaatoSession.configure()`` and
    registered via ``registry.register_core_tool()``.

    Args:
        session: The owning JaatoSession.
    """

    def __init__(self, session: 'JaatoSession') -> None:
        self._session = session

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

    def _execute_signal_completion(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Emit AgentCompletedEvent for the calling agent.

        Calls on_agent_completed on the session's UI hooks — the same
        path subagents use when they finish.
        """
        summary = args.get("summary", "")

        hooks = getattr(self._session, '_ui_hooks', None)
        if not hooks or not hasattr(hooks, 'on_agent_completed'):
            return {"error": "No UI hooks available"}

        agent_id = getattr(self._session, '_agent_id', 'main')
        usage = (
            self._session.get_context_usage()
            if hasattr(self._session, 'get_context_usage')
            else {}
        )

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
