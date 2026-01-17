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
    from ..model_provider.types import Message


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

        # Callbacks for session info (set by plugin)
        # Used to capture history snapshots for waypoint metadata
        self._get_history: Optional[Callable[[], List["Message"]]] = None
        self._serialize_history: Optional[Callable[[List["Message"]], str]] = None

        # Load existing waypoints
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
        """Load waypoints from storage."""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, 'r') as f:
                data = json.load(f)

            for wp_data in data.get("waypoints", []):
                waypoint = Waypoint.from_dict(wp_data)
                self._waypoints[waypoint.id] = waypoint

        except (json.JSONDecodeError, IOError, KeyError):
            # If storage is corrupted, start fresh (but keep w0)
            self._waypoints = {}

    def _save(self) -> None:
        """Save waypoints to storage."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                "waypoints": [wp.to_dict() for wp in self._waypoints.values()],
            }
            with open(self._storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass

    def _generate_id(self, owner: WaypointOwner = "user") -> str:
        """Generate the lowest available waypoint ID for the given owner.

        IDs are reused after deletion to keep numbering compact.
        User waypoints use w-prefix (w1, w2, ...), model waypoints use
        m-prefix (m1, m2, ...).

        Args:
            owner: Who will own the waypoint - "user" or "model".

        Returns:
            The next available ID for the owner type.
        """
        prefix = "w" if owner == "user" else "m"

        # Find existing numeric IDs with this prefix
        existing_nums = set()
        for wp_id in self._waypoints.keys():
            if wp_id.startswith(prefix) and wp_id[1:].isdigit():
                num = int(wp_id[1:])
                if num > 0:  # Skip w0 (implicit initial)
                    existing_nums.add(num)

        # Find lowest available ID starting from 1
        next_num = 1
        while next_num in existing_nums:
            next_num += 1

        return f"{prefix}{next_num}"

    def create(
        self,
        description: str,
        turn_index: Optional[int] = None,
        user_message_preview: Optional[str] = None,
        owner: WaypointOwner = "user",
    ) -> Waypoint:
        """Create a new waypoint marking the current state.

        Args:
            description: User or model-provided description.
            turn_index: Current turn in the conversation (optional).
            user_message_preview: Preview of recent user message (optional).
            owner: Who is creating this waypoint - "user" or "model".
                   Determines the ID prefix and permission rules.

        Returns:
            The newly created Waypoint.
        """
        wp_id = self._generate_id(owner)

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
        )

        self._waypoints[wp_id] = waypoint

        # Update BackupManager to tag future backups with this waypoint
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

        Args:
            waypoint_id: The waypoint ID to delete.

        Returns:
            True if deleted, False if not found or cannot delete.
        """
        if waypoint_id == INITIAL_WAYPOINT_ID:
            return False  # Cannot delete implicit waypoint

        if waypoint_id not in self._waypoints:
            return False

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

        # Restore files to their state at the waypoint
        files_restored = self._restore_code(waypoint_id)

        # After restoration, set BackupManager to tag future backups
        # with this waypoint (we're now diverging from it again)
        self._backup_manager.set_current_waypoint(waypoint_id)

        # Build result message
        if files_restored:
            message = f"Returned to waypoint {waypoint_id}: restored {len(files_restored)} file(s)"
        else:
            message = f"Returned to waypoint {waypoint_id}: no files to restore"

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

        return {
            "id": waypoint.id,
            "description": waypoint.description,
            "created_at": waypoint.created_at.isoformat(),
            "turn_index": waypoint.turn_index,
            "is_implicit": waypoint.is_implicit,
            "message_count": waypoint.message_count,
            "user_message_preview": waypoint.user_message_preview,
            "owner": waypoint.owner,
            "files_changed_since": list(unique_files),
            "total_backups_since": len(backups),
        }

    @property
    def current_waypoint(self) -> str:
        """Get the current waypoint ID (what new backups are tagged with)."""
        return self._backup_manager.current_waypoint
