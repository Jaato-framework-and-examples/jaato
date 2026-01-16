"""Data models for the Waypoint plugin.

Waypoints mark significant moments in your coding journey with the model,
allowing you to return to previous states when the path ahead becomes uncertain.

Unlike version control which captures code state, waypoints capture the full
context of your collaboration - both the code changes and the conversation
that led to them.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Waypoint:
    """A marked point in your coding journey.

    Waypoints capture the state of both your code and conversation at a
    specific moment, allowing you to return if the path ahead leads astray.

    Attributes:
        id: Short identifier (w0, w1, w2, ...). w0 is the implicit initial state.
        description: User or model-provided description of this waypoint.
        created_at: When the waypoint was created.
        turn_index: Which conversation turn this waypoint was created at.
        is_implicit: True only for waypoint w0 (initial state).
        history_snapshot: Serialized conversation history at this point.
        message_count: Number of messages in the conversation at this point.
        user_message_preview: Preview of the last user message for context.
    """

    id: str
    description: str
    created_at: datetime
    turn_index: int
    is_implicit: bool = False
    history_snapshot: Optional[str] = None  # JSON serialized history
    message_count: int = 0
    user_message_preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "turn_index": self.turn_index,
            "is_implicit": self.is_implicit,
            "history_snapshot": self.history_snapshot,
            "message_count": self.message_count,
            "user_message_preview": self.user_message_preview,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Waypoint":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            turn_index=data["turn_index"],
            is_implicit=data.get("is_implicit", False),
            history_snapshot=data.get("history_snapshot"),
            message_count=data.get("message_count", 0),
            user_message_preview=data.get("user_message_preview"),
        )


@dataclass
class RestoreResult:
    """Result of restoring to a waypoint.

    Attributes:
        success: Whether the restore operation succeeded.
        waypoint_id: The waypoint that was restored to.
        files_restored: List of files that were restored.
        message: Human-readable summary of what was restored.
        error: Error message if success is False.
    """

    success: bool
    waypoint_id: str
    files_restored: List[str] = field(default_factory=list)
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "success": self.success,
            "waypoint_id": self.waypoint_id,
            "files_restored": self.files_restored,
            "message": self.message,
        }
        # Only include error key if there's an actual error
        if self.error:
            result["error"] = self.error
        return result


# Initial waypoint ID - represents the state at session start
INITIAL_WAYPOINT_ID = "w0"

# Description for the implicit initial waypoint
INITIAL_WAYPOINT_DESCRIPTION = "session start"
