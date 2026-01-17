"""Waypoint plugin for marking and restoring session states.

Mark your journey, return when needed.

Every coding session is a journey - you and the model exploring solutions together,
making discoveries, sometimes taking wrong turns. Waypoints let you mark significant
moments along this journey, creating safe points you can return to if the path ahead
becomes treacherous.

Unlike version control which captures code state, waypoints capture the full context
of your collaboration - both the code changes and the conversation that led to them.

This plugin supports BOTH user commands AND model tools with an ownership model:
- User-owned waypoints (w-prefix): Created via user commands, model needs permission
  to restore, cannot delete
- Model-owned waypoints (m-prefix): Created via model tools, model has full control
  (create, restore, delete without permission)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .models import INITIAL_WAYPOINT_ID, WaypointOwner
from .manager import WaypointManager
from ..base import (
    UserCommand,
    CommandCompletion,
    CommandParameter,
    PromptEnrichmentResult,
)
from ..model_provider.types import ToolSchema

if TYPE_CHECKING:
    from ..file_edit.backup import BackupManager
    from ..model_provider.types import Message


class WaypointPlugin:
    """Plugin for marking and restoring session waypoints.

    This plugin provides BOTH user-facing commands AND model tools with an
    ownership-based permission model.

    Ownership Model:
        - User waypoints (w-prefix): Created by user commands. Model can list/view
          but needs permission to restore, cannot delete.
        - Model waypoints (m-prefix): Created by model tools. Model has full control
          (create, restore, delete) without needing permission.

    User Commands:
        waypoint              - List all waypoints
        waypoint create "desc" - Create user-owned waypoint
        waypoint restore N    - Restore files to waypoint N
        waypoint delete N     - Delete waypoint N (user-owned only)
        waypoint delete all   - Delete all user waypoints
        waypoint info N       - Show detailed info about waypoint N

    Model Tools:
        list_waypoints        - List all waypoints with ownership info
        waypoint_info         - Get detailed info about a waypoint
        create_waypoint       - Create a model-owned waypoint
        restore_waypoint      - Restore to a waypoint (permission for user-owned)
        delete_waypoint       - Delete a model-owned waypoint
    """

    def __init__(self):
        """Initialize the waypoint plugin."""
        self._manager: Optional[WaypointManager] = None
        self._backup_manager: Optional["BackupManager"] = None
        self._plugin_registry = None
        self._storage_path: Optional[Path] = None
        self._session_id: Optional[str] = None
        self._initialized = False

        # Session callbacks (set via set_session_callbacks)
        self._get_history: Optional[Callable[[], List["Message"]]] = None
        self._serialize_history: Optional[Callable[[List["Message"]], str]] = None
        self._get_turn_index: Optional[Callable[[], int]] = None

        # Pending restore notification for prompt enrichment
        self._pending_restore_notification: Optional[Dict[str, Any]] = None

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        import os
        import tempfile
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] [WAYPOINT] {msg}\n")
            except Exception:
                pass

    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        return "waypoint"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the waypoint plugin.

        Note: The manager is lazily created when first needed, after
        set_plugin_registry() has been called by the registry.

        Args:
            config: Optional configuration dict:
                - backup_manager: BackupManager instance (optional, can be
                  obtained from file_edit plugin via registry)
                - storage_path: Path for waypoint data storage
                - session_id: Session identifier for scoping waypoints
        """
        config = config or {}

        # Store config for lazy initialization
        self._backup_manager = config.get("backup_manager")
        self._session_id = config.get("session_id")
        storage_path = config.get("storage_path")
        if storage_path:
            self._storage_path = Path(storage_path)

        self._initialized = True

    def set_plugin_registry(self, registry) -> None:
        """Set the plugin registry for accessing backup_manager.

        Called by PluginRegistry after initialize(). This is when we can
        access the file_edit plugin's backup_manager.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry

    def _ensure_manager(self) -> bool:
        """Ensure the manager is created, lazily initializing if needed.

        Returns:
            True if manager is available, False otherwise.
        """
        if self._manager is not None:
            self._trace("_ensure_manager: manager already exists")
            return True

        self._trace(f"_ensure_manager: creating manager, backup_manager={self._backup_manager}, registry={self._plugin_registry}")

        # Get backup_manager from file_edit plugin if not directly configured
        if not self._backup_manager and self._plugin_registry:
            file_edit_plugin = self._plugin_registry.get_plugin("file_edit")
            self._trace(f"_ensure_manager: got file_edit_plugin={file_edit_plugin}")
            if file_edit_plugin and hasattr(file_edit_plugin, 'backup_manager'):
                self._backup_manager = file_edit_plugin.backup_manager
                self._trace(f"_ensure_manager: got backup_manager={self._backup_manager}")

        if not self._backup_manager:
            self._trace("_ensure_manager: no backup_manager, returning False")
            return False

        self._manager = WaypointManager(
            backup_manager=self._backup_manager,
            storage_path=self._storage_path,
            session_id=self._session_id,
        )
        self._trace(f"_ensure_manager: created manager={self._manager}")

        # Wire up history callbacks if they were set before manager was created
        if self._get_history and self._serialize_history:
            self._manager.set_history_callbacks(
                get_history=self._get_history,
                serialize_history=self._serialize_history,
            )

        return True

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._manager = None
        self._plugin_registry = None
        self._initialized = False

    def set_session_callbacks(
        self,
        get_history: Callable[[], List["Message"]],
        serialize_history: Callable[[List["Message"]], str],
        get_turn_index: Optional[Callable[[], int]] = None,
    ) -> None:
        """Set callbacks for session state access.

        This must be called after initialize() to enable capturing
        conversation metadata when creating waypoints.

        Args:
            get_history: Returns current conversation history.
            serialize_history: Converts history to JSON string.
            get_turn_index: Returns current turn index (optional).
        """
        self._get_history = get_history
        self._serialize_history = serialize_history
        self._get_turn_index = get_turn_index

        if self._manager:
            self._manager.set_history_callbacks(
                get_history=get_history,
                serialize_history=serialize_history,
            )

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for model waypoint access.

        The model has access to waypoint tools with ownership-based permissions:
        - Full control over model-owned waypoints (m-prefix)
        - Read access to user-owned waypoints (w-prefix)
        - Restore access to user-owned waypoints with permission
        - No delete access to user-owned waypoints
        """
        return [
            ToolSchema(
                name="list_waypoints",
                description=(
                    "List all waypoints in the session. Waypoints are checkpoints "
                    "that capture file state at significant moments. User-owned "
                    "waypoints (w-prefix) are created by the user; model-owned "
                    "waypoints (m-prefix) are created by you."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                category="planning",
            ),
            ToolSchema(
                name="waypoint_info",
                description=(
                    "Get detailed information about a specific waypoint, including "
                    "files changed since that waypoint."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "waypoint_id": {
                            "type": "string",
                            "description": (
                                "The waypoint ID (e.g., 'w0', 'w1', 'm1'). "
                                "w-prefix = user-owned, m-prefix = model-owned."
                            ),
                        },
                    },
                    "required": ["waypoint_id"],
                },
                category="planning",
            ),
            ToolSchema(
                name="create_waypoint",
                description=(
                    "Create a new model-owned waypoint to mark the current file state. "
                    "Use this before risky operations so you can restore if needed. "
                    "Model waypoints use m-prefix (m1, m2, ...) and you have full "
                    "control over them."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": (
                                "A brief description of why this waypoint is being "
                                "created (e.g., 'before auth refactor', "
                                "'pre-optimization checkpoint')."
                            ),
                        },
                    },
                    "required": ["description"],
                },
                category="planning",
            ),
            ToolSchema(
                name="restore_waypoint",
                description=(
                    "Restore all files to their state at a waypoint, undoing changes "
                    "made after that point. You can restore model-owned waypoints "
                    "(m-prefix) freely. Restoring user-owned waypoints (w-prefix) "
                    "requires user permission."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "waypoint_id": {
                            "type": "string",
                            "description": (
                                "The waypoint ID to restore to (e.g., 'w1', 'm1'). "
                                "m-prefix waypoints can be restored without permission."
                            ),
                        },
                    },
                    "required": ["waypoint_id"],
                },
                category="planning",
            ),
            ToolSchema(
                name="delete_waypoint",
                description=(
                    "Delete a model-owned waypoint. You can only delete waypoints "
                    "you created (m-prefix). User-owned waypoints (w-prefix) cannot "
                    "be deleted by the model."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "waypoint_id": {
                            "type": "string",
                            "description": (
                                "The model-owned waypoint ID to delete (must be "
                                "m-prefix, e.g., 'm1', 'm2')."
                            ),
                        },
                    },
                    "required": ["waypoint_id"],
                },
                category="planning",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return command executors for both user commands and model tools."""
        return {
            # User command
            "waypoint": self._execute_waypoint,
            # Model tools
            "list_waypoints": self._execute_list_waypoints,
            "waypoint_info": self._execute_waypoint_info,
            "create_waypoint": self._execute_create_waypoint,
            "restore_waypoint": self._execute_restore_waypoint,
            "delete_waypoint": self._execute_delete_waypoint,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions explaining waypoint ownership model."""
        return """## Waypoints

Waypoints are checkpoints that capture file state at significant moments, allowing
restoration if changes don't work out.

**Ownership Model:**
- **User waypoints (w-prefix)**: Created by the user. You can view and restore them
  (with user permission), but cannot delete them.
- **Model waypoints (m-prefix)**: Created by you. You have full control - create,
  restore, and delete without needing permission.

**Restore Behavior:**
- Restoring reverts file **contents** to their state at the waypoint
- Files **created after** the waypoint are NOT deleted (to avoid data loss)
- Files **deleted after** the waypoint are NOT recreated
- Only files that were modified and had backups are restored

**When to use waypoints:**
- Before attempting risky refactoring or experimental changes
- Before making multiple related file changes that might need rollback
- When exploring different implementation approaches

**Example workflow:**
1. `create_waypoint("before auth refactor")` - creates m1
2. Make changes across multiple files
3. If changes don't work: `restore_waypoint("m1")` - reverts all files
4. If changes work: optionally `delete_waypoint("m1")` to clean up"""

    def get_auto_approved_tools(self) -> List[str]:
        """Return auto-approved tools.

        Auto-approved:
        - waypoint: user command (always trusted)
        - list_waypoints, waypoint_info: read-only operations
        - create_waypoint: creates model-owned waypoint (safe)
        - delete_waypoint: executor enforces model-owned only

        Requires permission:
        - restore_waypoint: can affect user-owned waypoints
        """
        return [
            "waypoint",  # User command
            "list_waypoints",
            "waypoint_info",
            "create_waypoint",
            "delete_waypoint",  # Executor enforces ownership
        ]

    # ==================== Prompt Enrichment Protocol ====================

    def subscribes_to_prompt_enrichment(self) -> bool:
        """Subscribe to prompt enrichment to notify model of waypoint restores."""
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Add waypoint restore notification to the prompt if a restore occurred.

        When the user restores files to a previous waypoint, this informs the model
        that file state has changed since its last actions. This prevents the model
        from having stale context about files it previously modified.

        Args:
            prompt: The user's original prompt text.

        Returns:
            PromptEnrichmentResult with restore notification prepended if applicable.
        """
        if not self._pending_restore_notification:
            return PromptEnrichmentResult(prompt=prompt)

        # Consume the pending notification
        notification = self._pending_restore_notification
        self._pending_restore_notification = None

        # Build the notification message
        waypoint_id = notification["waypoint_id"]
        description = notification["description"]
        files_restored = notification.get("files_restored", [])

        if files_restored:
            files_list = ", ".join(files_restored)
            restore_info = (
                f"<hidden><waypoint-restore>\n"
                f"The user has restored files to waypoint {waypoint_id} ({description}).\n"
                f"Files restored to their previous state: {files_list}\n"
                f"Any changes you made to these files after this waypoint have been undone.\n"
                f"</waypoint-restore></hidden>\n\n"
            )
        else:
            restore_info = (
                f"<hidden><waypoint-restore>\n"
                f"The user has restored to waypoint {waypoint_id} ({description}).\n"
                f"File state has been reverted to this waypoint.\n"
                f"</waypoint-restore></hidden>\n\n"
            )

        self._trace(f"enrich_prompt: adding waypoint restore notification")

        return PromptEnrichmentResult(
            prompt=restore_info + prompt,
            metadata={
                "waypoint_restore": {
                    "waypoint_id": waypoint_id,
                    "description": description,
                    "files_restored": files_restored,
                }
            }
        )

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing waypoint commands."""
        return [
            UserCommand(
                name="waypoint",
                description="Mark your journey, return when needed",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: create, restore, delete, info, or list (default)",
                        required=False,
                    ),
                    CommandParameter(
                        name="target",
                        description="Waypoint ID (e.g., w1), 'all' for delete, or description for create (required)",
                        required=False,
                        capture_rest=True,
                    ),
                ],
            ),
        ]

    def get_command_completions(
        self,
        command: str,
        args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for waypoint command arguments."""
        if command != "waypoint":
            return []

        # No args yet - suggest actions (doesn't require manager)
        if not args:
            return [
                CommandCompletion("create", "Create a new waypoint"),
                CommandCompletion("restore", "Restore to a waypoint"),
                CommandCompletion("delete", "Delete waypoint(s)"),
                CommandCompletion("info", "Show waypoint details"),
                CommandCompletion("list", "List all waypoints"),
            ]

        action = args[0].lower() if args else ""
        known_actions = ["create", "restore", "delete", "info", "list"]

        # Completing first arg (action) - only if not an exact match
        if len(args) == 1 and not args[0].startswith("w") and action not in known_actions:
            return [
                CommandCompletion(a, f"{a.capitalize()} waypoint(s)")
                for a in known_actions
                if a.startswith(action)
            ]

        # Action complete, suggest waypoint IDs or description
        if len(args) >= 1:
            action = args[0].lower()

            if action == "create":
                # Description is required for create
                if len(args) == 1:
                    return [
                        CommandCompletion(
                            '"',
                            "Enter description (required)"
                        )
                    ]
                return []

            elif action in ("restore", "delete", "info"):
                # Suggest waypoint IDs (requires manager)
                if not self._ensure_manager():
                    return []
                waypoints = self._manager.list()

                if action == "delete" and len(args) == 1:
                    # Also suggest "all" for delete
                    completions = [
                        CommandCompletion("all", "Delete all user waypoints")
                    ]
                else:
                    completions = []

                # Add waypoint IDs
                for wp in waypoints:
                    if wp.is_implicit and action == "delete":
                        continue  # Can't delete w0

                    target = args[1] if len(args) > 1 else ""
                    if wp.id.startswith(target):
                        desc = wp.description[:30] + "..." if len(wp.description) > 30 else wp.description
                        completions.append(
                            CommandCompletion(wp.id, desc)
                        )

                return completions

        return []

    def _execute_waypoint(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the waypoint command.

        Args:
            args: Parsed command arguments.

        Returns:
            Command result as dict.
        """
        self._trace(f"_execute_waypoint: args={args}")

        if not self._ensure_manager():
            self._trace("_execute_waypoint: manager not available")
            return {"error": "Waypoint plugin not available (file_edit plugin not initialized)"}

        # Parse action from args
        action = args.get("action", "list")
        if isinstance(action, str):
            action = action.lower()
        else:
            action = "list"

        target = args.get("target", "")

        self._trace(f"_execute_waypoint: action={action}, target={target}")

        # Handle list action (or no action)
        if action in ("list", "") or action is None:
            return self._list_waypoints()

        # Handle create action
        if action == "create":
            # Target contains the description (with capture_rest)
            description = target.strip('"\'') if target else ""
            return self._create_waypoint(description)

        # Handle restore action
        if action == "restore":
            if not target:
                return {"error": "Usage: waypoint restore <id>"}
            return self._restore_waypoint(target)

        # Handle delete action
        if action == "delete":
            if not target:
                return {"error": "Usage: waypoint delete <id|all>"}
            return self._delete_waypoint(target)

        # Handle info action
        if action == "info":
            if not target:
                return {"error": "Usage: waypoint info <id>"}
            return self._waypoint_info(target)

        return {"error": f"Unknown action: {action}. Try: create, restore, delete, info, list"}

    def _list_waypoints(self, include_ownership: bool = False) -> Dict[str, Any]:
        """List all waypoints.

        Args:
            include_ownership: If True, include owner field in output (for model).
        """
        if not self._manager:
            return {"error": "Manager not initialized"}

        waypoints = self._manager.list()
        current_id = self._manager.current_waypoint

        if not waypoints:
            return {
                "waypoints": [],
                "message": "No waypoints yet. Use 'waypoint create' to mark your first.",
            }

        # Format waypoints for display
        formatted = []
        for wp in waypoints:
            desc = wp.description
            if len(desc) > 40:
                desc = desc[:37] + "..."

            entry = {
                "id": wp.id,
                "description": desc,
                "created_at": wp.created_at.strftime("%H:%M"),
                "messages": wp.message_count,
                "is_implicit": wp.is_implicit,
                "is_current": wp.id == current_id,
            }
            if include_ownership:
                entry["owner"] = wp.owner

            formatted.append(entry)

        return {
            "waypoints": formatted,
            "current": current_id,
        }

    def _create_waypoint(
        self,
        description: str,
        owner: WaypointOwner = "user",
    ) -> Dict[str, Any]:
        """Create a new waypoint.

        Args:
            description: Required description for the waypoint.
            owner: Who is creating this waypoint - "user" or "model".

        Returns:
            Result dict with success/error status.
        """
        if not self._manager:
            return {"error": "Manager not initialized"}

        # Description is required
        if not description:
            return {"error": "Description required. Usage: waypoint create \"your description\""}

        # Get current turn index if available
        turn_index = 0
        if self._get_turn_index:
            turn_index = self._get_turn_index()

        # Get last user message preview
        user_message_preview = None
        if self._get_history:
            history = self._get_history()
            for msg in reversed(history):
                if msg.role.value == "user":
                    text = msg.text or ""
                    user_message_preview = text[:50] + "..." if len(text) > 50 else text
                    break

        waypoint = self._manager.create(
            description=description,
            turn_index=turn_index,
            user_message_preview=user_message_preview,
            owner=owner,
        )

        return {
            "success": True,
            "id": waypoint.id,
            "description": waypoint.description,
            "owner": waypoint.owner,
            "message": f"Waypoint {waypoint.id} created: {waypoint.description}",
        }

    def _restore_waypoint(self, waypoint_id: str) -> Dict[str, Any]:
        """Restore files to a waypoint state."""
        self._trace(f"_restore_waypoint: waypoint_id={waypoint_id}")

        if not self._manager:
            self._trace("_restore_waypoint: manager is None")
            return {"error": "Manager not initialized"}

        self._trace(f"_restore_waypoint: calling manager.restore")
        result = self._manager.restore(waypoint_id)
        self._trace(f"_restore_waypoint: result={result}")

        result_dict = result.to_dict()

        # Store pending notification for prompt enrichment
        # so the model is informed on the next prompt
        if result.success:
            waypoint = self._manager.get(waypoint_id)
            self._pending_restore_notification = {
                "waypoint_id": waypoint_id,
                "description": waypoint.description if waypoint else waypoint_id,
                "files_restored": result.files_restored,
            }
            self._trace(f"_restore_waypoint: stored pending notification: {self._pending_restore_notification}")

        self._trace(f"_restore_waypoint: returning {result_dict}")
        return result_dict

    def _delete_waypoint(self, target: str) -> Dict[str, Any]:
        """Delete waypoint(s)."""
        if not self._manager:
            return {"error": "Manager not initialized"}

        if target.lower() == "all":
            count = self._manager.delete_all()
            return {
                "success": True,
                "deleted": count,
                "message": f"Deleted {count} waypoint(s). Initial state (w0) preserved.",
            }

        success = self._manager.delete(target)
        if success:
            return {
                "success": True,
                "id": target,
                "message": f"Waypoint {target} deleted.",
            }
        else:
            if target == INITIAL_WAYPOINT_ID:
                return {"error": f"Cannot delete initial waypoint {INITIAL_WAYPOINT_ID}"}
            return {"error": f"Waypoint not found: {target}"}

    def _waypoint_info(self, waypoint_id: str) -> Dict[str, Any]:
        """Get detailed info about a waypoint."""
        if not self._manager:
            return {"error": "Manager not initialized"}

        info = self._manager.get_info(waypoint_id)
        if not info:
            return {"error": f"Waypoint not found: {waypoint_id}"}

        return info


    # ==================== Model Tool Executors ====================

    def _execute_list_waypoints(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_waypoints tool for model.

        Returns all waypoints with ownership information.
        """
        if not self._ensure_manager():
            return {"error": "Waypoint plugin not available"}

        return self._list_waypoints(include_ownership=True)

    def _execute_waypoint_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute waypoint_info tool for model.

        Args:
            args: Must contain 'waypoint_id'.
        """
        if not self._ensure_manager():
            return {"error": "Waypoint plugin not available"}

        waypoint_id = args.get("waypoint_id")
        if not waypoint_id:
            return {"error": "waypoint_id is required"}

        return self._waypoint_info(waypoint_id)

    def _execute_create_waypoint(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute create_waypoint tool for model.

        Creates a model-owned waypoint (m-prefix).

        Args:
            args: Must contain 'description'.
        """
        if not self._ensure_manager():
            return {"error": "Waypoint plugin not available"}

        description = args.get("description")
        if not description:
            return {"error": "description is required"}

        return self._create_waypoint(description=description, owner="model")

    def _execute_restore_waypoint(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute restore_waypoint tool for model.

        Model can restore to any waypoint, but restoring to user-owned
        waypoints (w-prefix) should go through permission flow.
        Model-owned waypoints (m-prefix) can be restored freely.

        Args:
            args: Must contain 'waypoint_id'.
        """
        if not self._ensure_manager():
            return {"error": "Waypoint plugin not available"}

        waypoint_id = args.get("waypoint_id")
        if not waypoint_id:
            return {"error": "waypoint_id is required"}

        # Check if waypoint exists
        waypoint = self._manager.get(waypoint_id)
        if not waypoint:
            return {"error": f"Waypoint not found: {waypoint_id}"}

        # Note: Permission for user-owned waypoints is handled by the
        # permission plugin since restore_waypoint is not auto-approved.
        # If we get here, permission was granted (or it's model-owned).

        return self._restore_waypoint(waypoint_id)

    def _execute_delete_waypoint(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute delete_waypoint tool for model.

        Model can only delete its own waypoints (m-prefix).
        User-owned waypoints (w-prefix) cannot be deleted by model.

        Args:
            args: Must contain 'waypoint_id'.
        """
        if not self._ensure_manager():
            return {"error": "Waypoint plugin not available"}

        waypoint_id = args.get("waypoint_id")
        if not waypoint_id:
            return {"error": "waypoint_id is required"}

        # Check if waypoint exists
        waypoint = self._manager.get(waypoint_id)
        if not waypoint:
            return {"error": f"Waypoint not found: {waypoint_id}"}

        # Enforce ownership: model can only delete model-owned waypoints
        if waypoint.owner != "model":
            return {
                "error": (
                    f"Cannot delete user-owned waypoint {waypoint_id}. "
                    "You can only delete waypoints you created (m-prefix)."
                )
            }

        # Proceed with deletion
        success = self._manager.delete(waypoint_id)
        if success:
            return {
                "success": True,
                "id": waypoint_id,
                "message": f"Waypoint {waypoint_id} deleted.",
            }
        else:
            return {"error": f"Failed to delete waypoint: {waypoint_id}"}


def create_plugin() -> WaypointPlugin:
    """Factory function to create the Waypoint plugin instance."""
    return WaypointPlugin()
