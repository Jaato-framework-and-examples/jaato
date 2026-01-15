"""Waypoint plugin - mark your journey, return when needed.

Every coding session is a journey - you and the model exploring solutions together,
making discoveries, sometimes taking wrong turns. Waypoints let you mark significant
moments along this journey, creating safe points you can return to if the path ahead
becomes treacherous.

Unlike version control which captures code state, waypoints capture the full context
of your collaboration - both the code changes and the conversation that led to them.

Usage:
    waypoint                    # List all waypoints
    waypoint create             # Create with auto-generated description
    waypoint create "desc"      # Create with custom description
    waypoint restore w1         # Restore both code and conversation
    waypoint restore w1 code    # Restore code only
    waypoint restore w1 conv    # Restore conversation only
    waypoint delete w1          # Delete a waypoint
    waypoint delete all         # Delete all user waypoints
    waypoint info w1            # Show waypoint details
"""

# Plugin kind for discovery by PluginRegistry
PLUGIN_KIND = "tool"

from .plugin import WaypointPlugin, create_plugin
from .models import Waypoint, RestoreMode, RestoreResult, INITIAL_WAYPOINT_ID
from .manager import WaypointManager
from .suggester import DescriptionSuggester, create_suggester

__all__ = [
    "PLUGIN_KIND",
    "WaypointPlugin",
    "create_plugin",
    "Waypoint",
    "RestoreMode",
    "RestoreResult",
    "WaypointManager",
    "DescriptionSuggester",
    "create_suggester",
    "INITIAL_WAYPOINT_ID",
]
