"""Streaming control plugin providing dismiss_stream tool.

This plugin provides the dismiss_stream tool that allows the model to
stop receiving streaming updates when it has seen enough results.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from ..base import ToolPlugin
from ..model_provider.types import ToolSchema
from .manager import StreamManager

logger = logging.getLogger(__name__)


class StreamingControlPlugin(ToolPlugin):
    """Plugin providing streaming control tools.

    Currently provides:
    - dismiss_stream: Stop receiving updates from a streaming tool
    """

    def __init__(self):
        """Initialize the streaming control plugin."""
        self._manager: Optional[StreamManager] = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "streaming_control"

    def set_stream_manager(self, manager: StreamManager) -> None:
        """Set the stream manager instance.

        Called by the session to wire up the manager.

        Args:
            manager: The StreamManager instance.
        """
        self._manager = manager

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin."""
        self._initialized = True
        logger.info("StreamingControlPlugin initialized")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._initialized = False
        logger.info("StreamingControlPlugin shutdown")

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the tool schemas."""
        return [
            ToolSchema(
                name="dismiss_stream",
                description=(
                    "Stop receiving streaming updates from a tool. "
                    "Call this when you have enough results from a streaming tool "
                    "and don't need more updates. Use stream_id='*' to dismiss all active streams."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "stream_id": {
                            "type": "string",
                            "description": (
                                "ID of the stream to dismiss (from the streaming tool response), "
                                "or '*' to dismiss all active streams."
                            ),
                        },
                    },
                    "required": ["stream_id"],
                },
                category="system",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor functions."""
        return {
            "dismiss_stream": self._execute_dismiss_stream,
        }

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that don't require permission."""
        return ["dismiss_stream"]

    def _execute_dismiss_stream(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the dismiss_stream tool.

        Args:
            args: Tool arguments containing stream_id.

        Returns:
            Dict with success status and message.
        """
        if not self._manager:
            return {
                "success": False,
                "error": "Stream manager not available",
            }

        stream_id = args.get("stream_id", "")
        if not stream_id:
            return {
                "success": False,
                "error": "stream_id is required",
            }

        dismissed = self._manager.dismiss_stream(stream_id)

        if stream_id == "*":
            return {
                "success": True,
                "message": "All active streams dismissed" if dismissed else "No active streams to dismiss",
            }
        else:
            if dismissed:
                return {
                    "success": True,
                    "message": f"Stream {stream_id} dismissed",
                }
            else:
                return {
                    "success": False,
                    "error": f"Stream {stream_id} not found or already completed",
                }


def create_plugin() -> StreamingControlPlugin:
    """Factory function for plugin discovery.

    Returns:
        A new StreamingControlPlugin instance.
    """
    return StreamingControlPlugin()


__all__ = ["StreamingControlPlugin", "create_plugin"]
