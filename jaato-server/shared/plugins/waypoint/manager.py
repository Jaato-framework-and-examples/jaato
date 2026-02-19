"""Waypoint manager for tracking and restoring session states.

Every coding session is a journey - you and the model exploring solutions together,
making discoveries, sometimes taking wrong turns. The WaypointManager lets you mark
significant moments along this journey, creating safe points you can return to if
the path ahead becomes treacherous.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .models import (
    Waypoint,
    WaypointOwner,
    RestoreResult,
    INITIAL_WAYPOINT_ID,
    INITIAL_WAYPOINT_DESCRIPTION,
)

if TYPE_CHECKING:
    from ..file_edit.backup import BackupManager
    from jaato_sdk.plugins.model_provider.types import Message


class WaypointManager:
    """Manages waypoints for a session.

    Waypoints capture the state of both code and conversation at specific
    moments, allowing restoration to previous states when needed.

    The manager works with BackupManager to tag file backups with waypoint
    IDs, enabling code restoration without duplicating backup storage.
    """

    def __init__(
        self,
        backup_manager: "BackupManager",
        storage_path: Optional[Path] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the waypoint manager.

        Args:
            backup_manager: The backup manager for file state tracking.
            storage_path: Path to store waypoint data. If not provided,
                         uses .jaato/sessions/{session_id}/waypoints.json
                         or .jaato/waypoints.json if no session_id.
            session_id: Session identifier for scoping waypoints to a session.
        """
        self._backup_manager = backup_manager
        self._session_id = session_id

        # Determine storage path with session scoping
        if storage_path:
            self._storage_path = storage_path
        elif session_id:
            self._storage_path = Path(f".jaato/sessions/{session_id}/waypoints.json").resolve()
        else:
            self._storage_path = Path(".jaato/waypoints.json").resolve()

        self._waypoints: Dict[str, Waypoint] = {}
        self._next_id: int = 1  # Next ID to assign (monotonically increasing)

        # Callbacks for session info (set by plugin)
        # Used to capture history snapshots for waypoint metadata
        self._get_history: Optional[Callable[[], List["Message"]]] = None
        self._serialize_history: Optional[Callable[[List["Message"]], str]] = None

        # Load existing waypoints (also loads _next_id)
        self._load()

        # Ensure implicit initial waypoint exists
        self._ensure_initial_waypoint()

    def set_history_callbacks(
        self,
        get_history: Callable[[], List["Message"]],
        serialize_history: Callable[[List["Message"]], str],
    ) -> None:
        """Set callbacks for capturing session history snapshots.

        These callbacks enable capturing conversation metadata when creating
        waypoints (message count, preview). The snapshots are stored for
        informational purposes but not used for restoration.

        Args:
            get_history: Returns current conversation history.
            serialize_history: Converts history to JSON string.
        """
        self._get_history = get_history
        self._serialize_history = serialize_history

    def _ensure_initial_waypoint(self) -> None:
        """Ensure the implicit initial waypoint (w0) exists."""
        if INITIAL_WAYPOINT_ID not in self._waypoints:
            self._waypoints[INITIAL_WAYPOINT_ID] = Waypoint(
                id=INITIAL_WAYPOINT_ID,
                description=INITIAL_WAYPOINT_DESCRIPTION,
                created_at=datetime.now(),
                turn_index=0,
                is_implicit=True,
                history_snapshot=None,  # Empty history at start
                message_count=0,
            )
            self._save()

    def _load(self) -> None:
        """Load waypoints and next_id counter from storage."""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, 'r') as f:
                data = json.load(f)

            for wp_data in data.get("waypoints", []):
                waypoint = Waypoint.from_dict(wp_data)
                self._waypoints[waypoint.id] = waypoint

            # Load next_id counter (defaults to 1 if not present)
            self._next_id = data.get("next_id", 1)

        except (json.JSONDecodeError, IOError, KeyError):
            # If storage is corrupted, start fresh (but keep w0)
            self._waypoints = {}
            self._next_id = 1

    def _save(self) -> None:
        """Save waypoints and next_id counter to storage."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                "waypoints": [wp.to_dict() for wp in self._waypoints.values()],
                "next_id": self._next_id,
            }
            with open(self._storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass

    def _generate_id(self) -> str:
        """Generate the next sequential waypoint ID.

        IDs are monotonically increasing (w1, w2, w3, ...) and never reused.
        This ensures IDs always increase chronologically - a higher ID always
        means a waypoint was created later, making trees easier to understand.

        Returns:
            The next ID (uses and increments the counter).
        """
        wp_id = f"w{self._next_id}"
        self._next_id += 1
        return wp_id

    def create(
        self,
        description: str,
        turn_index: Optional[int] = None,
        user_message_preview: Optional[str] = None,
        owner: WaypointOwner = "user",
    ) -> Waypoint:
        """Create a new waypoint marking the current state.

        This establishes a new node in the waypoint tree:
        - parent_id is set to the current waypoint
        - Pending backups are tagged with this waypoint's ID
        - Future backups will diverge from this waypoint

        Args:
            description: User or model-provided description.
            turn_index: Current turn in the conversation (optional).
            user_message_preview: Preview of recent user message (optional).
            owner: Who is creating this waypoint - "user" or "model".
                   Tracked for permission rules (model can only delete
                   waypoints it owns).

        Returns:
            The newly created Waypoint.
        """
        wp_id = self._generate_id()

        # Parent is the current waypoint (where we are in the tree)
        parent_id = self._backup_manager.current_waypoint

        # Tag pending backups with this waypoint ID (closes the edit sequence)
        self._backup_manager.tag_pending_backups(wp_id)

        # Capture conversation history snapshot
        history_snapshot = None
        message_count = 0

        if self._get_history and self._serialize_history:
            history = self._get_history()
            message_count = len(history)
            history_snapshot = self._serialize_history(history)

        waypoint = Waypoint(
            id=wp_id,
            description=description,
            created_at=datetime.now(),
            turn_index=turn_index or 0,
            is_implicit=False,
            history_snapshot=history_snapshot,
            message_count=message_count,
            user_message_preview=user_message_preview,
            owner=owner,
            parent_id=parent_id,
        )

        self._waypoints[wp_id] = waypoint

        # Update BackupManager - future backups diverge from this waypoint
        self._backup_manager.set_current_waypoint(wp_id)

        self._save()

        return waypoint

    def list(self, include_implicit: bool = True) -> List[Waypoint]:
        """List all waypoints.

        Args:
            include_implicit: Whether to include the implicit w0 waypoint.

        Returns:
            List of waypoints, sorted by creation time (oldest first).
        """
        waypoints = list(self._waypoints.values())

        if not include_implicit:
            waypoints = [wp for wp in waypoints if not wp.is_implicit]

        # Sort by creation time
        waypoints.sort(key=lambda wp: wp.created_at)

        return waypoints

    def get(self, waypoint_id: str) -> Optional[Waypoint]:
        """Get a waypoint by ID.

        Args:
            waypoint_id: The waypoint ID (e.g., "w0", "w1").

        Returns:
            The Waypoint if found, None otherwise.
        """
        return self._waypoints.get(waypoint_id)

    def delete(self, waypoint_id: str) -> bool:
        """Delete a waypoint.

        Cannot delete the implicit initial waypoint (w0).
        Children of the deleted waypoint are reparented to its parent.

        Args:
            waypoint_id: The waypoint ID to delete.

        Returns:
            True if deleted, False if not found or cannot delete.
        """
        if waypoint_id == INITIAL_WAYPOINT_ID:
            return False  # Cannot delete implicit waypoint

        if waypoint_id not in self._waypoints:
            return False

        # Reparent children to the deleted waypoint's parent
        deleted_wp = self._waypoints[waypoint_id]
        parent_id = deleted_wp.parent_id

        for wp in self._waypoints.values():
            if wp.parent_id == waypoint_id:
                wp.parent_id = parent_id

        del self._waypoints[waypoint_id]
        self._save()

        return True

    def delete_all(self) -> int:
        """Delete all user-created waypoints.

        The implicit w0 waypoint is preserved.

        Returns:
            Number of waypoints deleted.
        """
        to_delete = [
            wp_id for wp_id, wp in self._waypoints.items()
            if not wp.is_implicit
        ]

        for wp_id in to_delete:
            del self._waypoints[wp_id]

        # Reset BackupManager to initial waypoint
        self._backup_manager.set_current_waypoint(INITIAL_WAYPOINT_ID)

        self._save()

        return len(to_delete)

    def restore(self, waypoint_id: str) -> RestoreResult:
        """Restore files to their state at a waypoint.

        If there are uncommitted edits (pending backups), a ceiling waypoint
        is automatically created to preserve them before restoring. This
        ensures you can always navigate back to where you were.

        Restores only file changes. The model is notified of the restoration
        through prompt enrichment, so conversation history is not modified.

        Args:
            waypoint_id: The waypoint ID to restore to.

        Returns:
            RestoreResult with details of what was restored.
        """
        waypoint = self._waypoints.get(waypoint_id)

        if not waypoint:
            return RestoreResult(
                success=False,
                waypoint_id=waypoint_id,
                error=f"Waypoint not found: {waypoint_id}",
            )

        # Auto-create ceiling waypoint if there are uncommitted edits
        ceiling_waypoint = None
        if self._backup_manager.has_pending_backups():
            ceiling_waypoint = self.create(
                description=f"auto-saved before restore to {waypoint_id}",
                owner="user",  # System-created, but user-owned for safety
            )

        # Restore files to their state at the waypoint
        files_restored = self._restore_code(waypoint_id)

        # After restoration, set BackupManager to tag future backups
        # with this waypoint (we're now diverging from it again)
        self._backup_manager.set_current_waypoint(waypoint_id)

        # Build result message
        parts = []
        if ceiling_waypoint:
            parts.append(f"auto-saved to {ceiling_waypoint.id}")
        if files_restored:
            parts.append(f"restored {len(files_restored)} file(s)")
        else:
            parts.append("no files to restore")

        message = f"Returned to waypoint {waypoint_id}: {', '.join(parts)}"

        return RestoreResult(
            success=True,
            waypoint_id=waypoint_id,
            files_restored=files_restored,
            message=message,
        )

    def _restore_code(self, waypoint_id: str) -> List[str]:
        """Restore files to their state at the given waypoint.

        Uses backups that were tagged as diverging from this waypoint.
        The first backup of each file after the waypoint contains the
        file's state AT the waypoint (before any divergent edits).

        Args:
            waypoint_id: The waypoint ID to restore to.

        Returns:
            List of file paths that were restored.
        """
        # Get first backup per file that diverged from this waypoint
        backups_to_restore = self._backup_manager.get_first_backup_per_file_by_waypoint(
            waypoint_id
        )

        restored_files: List[str] = []

        for original_path, backup_info in backups_to_restore.items():
            success = self._backup_manager.restore_from_backup(
                Path(original_path),
                backup_info.backup_path,
            )
            if success:
                restored_files.append(original_path)

        return restored_files

    def get_info(self, waypoint_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a waypoint.

        Args:
            waypoint_id: The waypoint ID.

        Returns:
            Dictionary with waypoint details, or None if not found.
        """
        waypoint = self._waypoints.get(waypoint_id)
        if not waypoint:
            return None

        # Get file changes since this waypoint
        backups = self._backup_manager.get_backups_by_waypoint(waypoint_id)
        unique_files = set(b.original_path for b in backups)

        # Get children for tree navigation
        children = self.get_children(waypoint_id)

        return {
            "id": waypoint.id,
            "description": waypoint.description,
            "created_at": waypoint.created_at.isoformat(),
            "turn_index": waypoint.turn_index,
            "is_implicit": waypoint.is_implicit,
            "message_count": waypoint.message_count,
            "user_message_preview": waypoint.user_message_preview,
            "owner": waypoint.owner,
            "parent_id": waypoint.parent_id,
            "children": children,
            "files_changed_since": list(unique_files),
            "total_backups_since": len(backups),
        }

    @property
    def current_waypoint(self) -> str:
        """Get the current waypoint ID (what new backups are tagged with)."""
        return self._backup_manager.current_waypoint

    # ==================== Tree Navigation ====================

    def get_ancestors(self, waypoint_id: str) -> List[str]:
        """Get the path from a waypoint to the root (w0).

        Args:
            waypoint_id: The waypoint ID to start from.

        Returns:
            List of waypoint IDs from the given waypoint to root (inclusive).
            Example: ["w3", "w2", "w1", "w0"]
        """
        path = []
        current_id = waypoint_id

        while current_id is not None:
            waypoint = self._waypoints.get(current_id)
            if waypoint is None:
                break
            path.append(current_id)
            current_id = waypoint.parent_id

        return path

    def get_children(self, waypoint_id: str) -> List[str]:
        """Get the direct children of a waypoint.

        Args:
            waypoint_id: The parent waypoint ID.

        Returns:
            List of waypoint IDs that have this waypoint as their parent.
        """
        return [
            wp.id for wp in self._waypoints.values()
            if wp.parent_id == waypoint_id
        ]

    def find_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """Find the path between two waypoints in the tree.

        The path goes through the common ancestor:
        from_id → ... → common_ancestor → ... → to_id

        Args:
            from_id: Starting waypoint ID.
            to_id: Target waypoint ID.

        Returns:
            List of waypoint IDs representing the path, or None if no path exists.
            The list includes both endpoints.
        """
        if from_id == to_id:
            return [from_id]

        # Get ancestors of both waypoints
        from_ancestors = self.get_ancestors(from_id)
        to_ancestors = self.get_ancestors(to_id)

        if not from_ancestors or not to_ancestors:
            return None

        # Convert to sets for fast lookup
        from_ancestor_set = set(from_ancestors)
        to_ancestor_set = set(to_ancestors)

        # Find common ancestor (first ancestor of 'from' that's also ancestor of 'to')
        common_ancestor = None
        for ancestor in from_ancestors:
            if ancestor in to_ancestor_set:
                common_ancestor = ancestor
                break

        if common_ancestor is None:
            return None  # No path exists (shouldn't happen in a valid tree)

        # Build path: from_id → common_ancestor (backward)
        backward_path = []
        for ancestor in from_ancestors:
            backward_path.append(ancestor)
            if ancestor == common_ancestor:
                break

        # Build path: common_ancestor → to_id (forward)
        forward_path = []
        for ancestor in to_ancestors:
            if ancestor == common_ancestor:
                break
            forward_path.append(ancestor)

        # Combine: backward path + reversed forward path
        # backward_path ends with common_ancestor, forward_path excludes it
        forward_path.reverse()
        full_path = backward_path + forward_path

        return full_path

    def get_tree_structure(self) -> Dict[str, Any]:
        """Get the full waypoint tree structure.

        Returns:
            Dictionary with tree information:
            - current: The current waypoint ID
            - root: The root waypoint ID (w0)
            - nodes: Dict mapping waypoint IDs to their info including children
        """
        nodes = {}
        for wp in self._waypoints.values():
            children = self.get_children(wp.id)
            nodes[wp.id] = {
                "id": wp.id,
                "description": wp.description,
                "owner": wp.owner,
                "parent_id": wp.parent_id,
                "children": children,
                "is_implicit": wp.is_implicit,
                "created_at": wp.created_at.isoformat(),
            }

        return {
            "current": self.current_waypoint,
            "root": INITIAL_WAYPOINT_ID,
            "nodes": nodes,
        }
