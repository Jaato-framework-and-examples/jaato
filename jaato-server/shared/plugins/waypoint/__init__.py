"""Waypoint plugin - mark your journey, return when needed.

Every coding session is a journey - you and the model exploring solutions together,
making discoveries, sometimes taking wrong turns. Waypoints let you mark significant
moments along this journey, creating safe points you can return to if the path ahead
becomes treacherous.

Usage:
    waypoint                    # List all waypoints
    waypoint create "desc"      # Create with description
    waypoint restore w1         # Restore files to waypoint state
    waypoint delete w1          # Delete a waypoint
    waypoint delete all         # Delete all user waypoints
    waypoint info w1            # Show waypoint details
"""

# Plugin kind for discovery by PluginRegistry
PLUGIN_KIND = "tool"

from .plugin import WaypointPlugin, create_plugin
from .models import Waypoint, RestoreResult, INITIAL_WAYPOINT_ID
from .manager import WaypointManager

__all__ = [
    "PLUGIN_KIND",
    "WaypointPlugin",
    "create_plugin",
    "Waypoint",
    "RestoreResult",
    "WaypointManager",
    "INITIAL_WAYPOINT_ID",
]
