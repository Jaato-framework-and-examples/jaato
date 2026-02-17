"""Sandbox Manager plugin for runtime path permission management.

This plugin enables users to dynamically grant or revoke filesystem access
at runtime through a three-tier configuration model:

1. Global (~/.jaato/sandbox_paths.json) - User-wide settings
2. Workspace (<workspace>/.jaato/sandbox.json) - Project-specific settings
3. Session (<workspace>/.jaato/sessions/<id>/sandbox.json) - Runtime overrides

Session configuration has highest precedence and can override paths from
other levels.

User Commands:
    sandbox list   - Show all effective paths from all three levels
    sandbox add    - Grant temporary access for current session
    sandbox remove - Block a path for this session (even if globally allowed)
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..base import UserCommand, CommandCompletion, CommandParameter, HelpLines
from ..model_provider.types import ToolSchema

from shared.path_utils import (
    msys2_to_windows_path,
    normalize_result_path,
    normalized_equals,
)
from shared.trace import trace as _trace_write


# Config file names
GLOBAL_CONFIG_FILE = "sandbox_paths.json"
WORKSPACE_CONFIG_FILE = "sandbox.json"
SESSION_CONFIG_FILE = "sandbox.json"


@dataclass
class SandboxPath:
    """A sandbox path entry with metadata."""
    path: str
    source: str  # "global", "workspace", or "session"
    action: str  # "allow" or "deny"
    access: str = "readwrite"  # "readonly" or "readwrite"
    added_at: Optional[str] = None


@dataclass
class SandboxConfig:
    """Merged sandbox configuration from all tiers."""
    allowed_paths: List[SandboxPath] = field(default_factory=list)
    denied_paths: List[SandboxPath] = field(default_factory=list)


class SandboxManagerPlugin:
    """Plugin for managing sandbox path permissions at runtime.

    This plugin provides user commands (not model tools) for managing
    which paths the model can access during a session. It integrates
    with the PluginRegistry's authorization and denial mechanisms.

    Lifecycle & Pending Paths:
        During plugin initialization, other plugins (e.g., references) may call
        ``add_path_programmatic()`` before this plugin has a workspace or session
        configured.  In that case the path is stored in ``_pending_programmatic_paths``
        and temporarily registered in the registry.  When the workspace/session
        later become available (via ``set_workspace_path`` / ``set_session_id``),
        ``_replay_pending_paths()`` persists the pending entries to the session
        config so they survive config reloads.
    """

    def __init__(self):
        self._registry = None
        self._workspace_path: Optional[str] = None
        self._session_id: Optional[str] = None
        self._config: Optional[SandboxConfig] = None
        self._initialized = False
        # Paths added via add_path_programmatic() before workspace/session
        # were available.  Each entry is (path, access).
        self._pending_programmatic_paths: List[tuple] = []
        # Optional callback invoked when the set of readwrite-allowed paths
        # changes (after ``_sync_to_registry``).  The callback receives a
        # list of absolute paths that currently have readwrite access.
        # Used by the workspace monitor to start/stop watching sandbox dirs.
        self._on_readwrite_paths_changed: Optional[Callable[[List[str]], None]] = None

    @property
    def name(self) -> str:
        return "sandbox_manager"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("SandboxManager", msg)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the sandbox manager plugin.

        Args:
            config: Optional configuration dict containing:
                - session_id: Session identifier for session-level config
        """
        config = config or {}
        self._session_id = config.get("session_id")
        self._initialized = True
        self._trace(f"initialize: session_id={self._session_id}")

        # Load configuration if workspace is already set
        if self._workspace_path:
            self._load_all_configs()

    def shutdown(self) -> None:
        """Shutdown the sandbox manager plugin."""
        self._trace("shutdown")
        # Clear registry state for this plugin
        if self._registry:
            self._registry.clear_authorized_paths(self.name)
            self._registry.clear_denied_paths(self.name)
        self._config = None
        self._pending_programmatic_paths.clear()
        self._initialized = False

    def set_on_readwrite_paths_changed(
        self,
        callback: Optional[Callable[[List[str]], None]],
    ) -> None:
        """Register a callback for when readwrite sandbox paths change.

        The callback receives a list of absolute paths that currently have
        readwrite access.  It is invoked after each ``sandbox add`` or
        ``sandbox remove`` command, as well as after config reloads.

        Used by the session manager to update the workspace monitor's set
        of watched sandbox directories.

        Args:
            callback: Callable receiving ``List[str]`` of readwrite paths,
                or ``None`` to clear.
        """
        self._on_readwrite_paths_changed = callback

    def set_plugin_registry(self, registry) -> None:
        """Receive the plugin registry for authorization management.

        Called by PluginRegistry.expose_tool() during initialization.
        """
        self._registry = registry
        self._trace("set_plugin_registry: registry received")

    def set_workspace_path(self, path: str) -> None:
        """Receive the workspace root path.

        Called by PluginRegistry.set_workspace_path() when workspace changes.
        """
        self._workspace_path = path
        self._trace(f"set_workspace_path: {path}")

        # Reload configuration when workspace changes
        if self._initialized:
            self._load_all_configs()

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for session-level configuration.

        Called by the client when a new session starts.
        """
        self._session_id = session_id
        self._trace(f"set_session_id: {session_id}")

        # Reload configuration when session changes
        if self._initialized and self._workspace_path:
            self._load_all_configs()

    def _get_current_session_id(self) -> Optional[str]:
        """Get the current session ID.

        First checks if set directly via set_session_id(), then queries
        the session plugin via registry for the active session ID.

        Returns:
            Current session ID or None if no session is active.
        """
        # Check if set directly
        if self._session_id:
            return self._session_id

        # Query session plugin via registry
        if self._registry:
            session_plugin = self._registry.get_plugin("session")
            if session_plugin and hasattr(session_plugin, 'get_current_session_id'):
                session_id = session_plugin.get_current_session_id()
                if session_id:
                    self._trace(f"_get_current_session_id: got {session_id} from session plugin")
                    return session_id

        return None

    # ==================== Config Loading ====================

    def _get_global_config_path(self) -> Path:
        """Get path to global config file."""
        return Path.home() / ".jaato" / GLOBAL_CONFIG_FILE

    def _get_workspace_config_path(self) -> Optional[Path]:
        """Get path to workspace config file."""
        if not self._workspace_path:
            return None
        return Path(self._workspace_path) / ".jaato" / WORKSPACE_CONFIG_FILE

    def _get_session_config_path(self) -> Optional[Path]:
        """Get path to session config file."""
        session_id = self._get_current_session_id()
        if not self._workspace_path or not session_id:
            return None
        return (
            Path(self._workspace_path) / ".jaato" / "sessions" /
            session_id / SESSION_CONFIG_FILE
        )

    def _load_config_file(self, path: Path, source: str) -> tuple[List[SandboxPath], List[SandboxPath]]:
        """Load a single config file and return allowed/denied paths.

        Args:
            path: Path to the config file.
            source: Source identifier ("global", "workspace", "session").

        Returns:
            Tuple of (allowed_paths, denied_paths).
        """
        allowed = []
        denied = []

        if not path.exists():
            return allowed, denied

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            self._trace(f"Error loading {source} config from {path}: {e}")
            return allowed, denied

        # Parse allowed_paths
        # Convert MSYS2 paths at the config-loading boundary to handle
        # hand-edited configs or paths saved before the MSYS2 fix.
        for item in data.get("allowed_paths", []):
            if isinstance(item, str):
                allowed.append(SandboxPath(
                    path=msys2_to_windows_path(item),
                    source=source, action="allow",
                ))
            elif isinstance(item, dict) and "path" in item:
                access = item.get("access", "readwrite")
                if access not in ("readonly", "readwrite"):
                    access = "readwrite"
                allowed.append(SandboxPath(
                    path=msys2_to_windows_path(item["path"]),
                    source=source,
                    action="allow",
                    access=access,
                    added_at=item.get("added_at")
                ))

        # Parse denied_paths
        for item in data.get("denied_paths", []):
            if isinstance(item, str):
                denied.append(SandboxPath(
                    path=msys2_to_windows_path(item),
                    source=source, action="deny",
                ))
            elif isinstance(item, dict) and "path" in item:
                denied.append(SandboxPath(
                    path=msys2_to_windows_path(item["path"]),
                    source=source,
                    action="deny",
                    added_at=item.get("added_at")
                ))

        self._trace(f"Loaded {source} config: {len(allowed)} allowed, {len(denied)} denied")
        return allowed, denied

    def _load_all_configs(self) -> None:
        """Load and merge configuration from all three tiers."""
        self._config = SandboxConfig()

        # Load global config
        global_path = self._get_global_config_path()
        allowed, denied = self._load_config_file(global_path, "global")
        self._config.allowed_paths.extend(allowed)
        self._config.denied_paths.extend(denied)

        # Load workspace config
        workspace_path = self._get_workspace_config_path()
        if workspace_path:
            allowed, denied = self._load_config_file(workspace_path, "workspace")
            self._config.allowed_paths.extend(allowed)
            self._config.denied_paths.extend(denied)

        # Load session config (highest precedence)
        session_path = self._get_session_config_path()
        if session_path:
            allowed, denied = self._load_config_file(session_path, "session")
            self._config.allowed_paths.extend(allowed)
            self._config.denied_paths.extend(denied)

        # Sync to registry
        self._sync_to_registry()

        # Replay any paths that were added before workspace/session were ready
        self._replay_pending_paths()

    def _replay_pending_paths(self) -> None:
        """Persist paths that were queued before workspace/session were available.

        When ``add_path_programmatic()`` is called before the plugin has a
        workspace or session, the path is stored in ``_pending_programmatic_paths``
        and only registered in the in-memory registry.  Once the workspace and
        session become available (i.e. ``_load_all_configs`` has run), this
        method persists those paths to the session config so they survive
        future config reloads.

        Paths that already appear in the loaded config (e.g. from a previous
        session) are skipped.  After successful persistence the pending list
        is cleared.
        """
        if not self._pending_programmatic_paths:
            return

        if not self._workspace_path or not self._get_current_session_id():
            # Still can't persist — re-register in registry so they aren't lost
            # after the clear in _sync_to_registry().
            for path, access in self._pending_programmatic_paths:
                if self._registry:
                    self._registry.authorize_external_path(path, self.name, access=access)
            self._trace(f"_replay_pending_paths: still no workspace/session, "
                       f"re-registered {len(self._pending_programmatic_paths)} paths in registry")
            return

        # We can now persist — load current session config and add pending paths
        session_config = self._load_session_config()
        existing_paths = set()
        for item in session_config.get("allowed_paths", []):
            existing_path = item["path"] if isinstance(item, dict) else item
            existing_paths.add(existing_path)

        added = 0
        for path, access in self._pending_programmatic_paths:
            if path not in existing_paths:
                session_config.setdefault("allowed_paths", []).append({
                    "path": path,
                    "access": access,
                    "added_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                })
                added += 1

        if added > 0:
            if self._save_session_config(session_config):
                self._trace(f"_replay_pending_paths: persisted {added} pending paths")
                # Reload so the new paths are in self._config and synced to registry
                self._config = SandboxConfig()
                # Re-load all tiers (global, workspace, session)
                global_path = self._get_global_config_path()
                allowed, denied = self._load_config_file(global_path, "global")
                self._config.allowed_paths.extend(allowed)
                self._config.denied_paths.extend(denied)
                workspace_path = self._get_workspace_config_path()
                if workspace_path:
                    allowed, denied = self._load_config_file(workspace_path, "workspace")
                    self._config.allowed_paths.extend(allowed)
                    self._config.denied_paths.extend(denied)
                session_path = self._get_session_config_path()
                if session_path:
                    allowed, denied = self._load_config_file(session_path, "session")
                    self._config.allowed_paths.extend(allowed)
                    self._config.denied_paths.extend(denied)
                self._sync_to_registry()
            else:
                # Persistence failed — at least keep them in registry
                for path, access in self._pending_programmatic_paths:
                    if self._registry:
                        self._registry.authorize_external_path(path, self.name, access=access)
                self._trace(f"_replay_pending_paths: persistence failed, re-registered in registry")
        else:
            self._trace(f"_replay_pending_paths: all {len(self._pending_programmatic_paths)} paths already in config")

        self._pending_programmatic_paths.clear()

    def _sync_to_registry(self) -> None:
        """Sync current config to the plugin registry.

        After syncing, notifies the readwrite-paths-changed callback
        so that the workspace monitor can update its watched directories.
        """
        if not self._registry or not self._config:
            return

        # Clear previous state from this plugin
        self._registry.clear_authorized_paths(self.name)
        self._registry.clear_denied_paths(self.name)

        # Register allowed paths
        for entry in self._config.allowed_paths:
            # Skip if this path is denied at a higher precedence level
            if not self._is_path_denied_by_higher_precedence(entry.path, entry.source):
                self._registry.authorize_external_path(
                    entry.path, self.name, access=entry.access
                )

        # Register denied paths
        for entry in self._config.denied_paths:
            self._registry.deny_external_path(entry.path, self.name)

        self._trace(f"Synced to registry: {len(self._config.allowed_paths)} allowed, "
                   f"{len(self._config.denied_paths)} denied")

        # Notify the workspace monitor about readwrite path changes
        self._notify_readwrite_paths_changed()

    def get_readwrite_paths(self) -> List[str]:
        """Return the list of currently allowed readwrite paths.

        Only includes paths that are not denied at a higher precedence level.

        Returns:
            List of absolute paths with readwrite access.
        """
        if not self._config:
            return []

        paths = []
        for entry in self._config.allowed_paths:
            if entry.access == "readwrite":
                if not self._is_path_denied_by_higher_precedence(entry.path, entry.source):
                    paths.append(os.path.normpath(os.path.abspath(entry.path)))
        return paths

    def _notify_readwrite_paths_changed(self) -> None:
        """Invoke the readwrite-paths-changed callback if registered."""
        if self._on_readwrite_paths_changed is not None:
            try:
                self._on_readwrite_paths_changed(self.get_readwrite_paths())
            except Exception as e:
                self._trace(f"Error in on_readwrite_paths_changed callback: {e}")

    def _is_path_denied_by_higher_precedence(self, path: str, source: str) -> bool:
        """Check if a path is denied by a higher precedence source.

        Precedence order: session > workspace > global
        """
        if not self._config:
            return False

        precedence = {"global": 0, "workspace": 1, "session": 2}
        source_precedence = precedence.get(source, 0)

        for entry in self._config.denied_paths:
            entry_precedence = precedence.get(entry.source, 0)
            if entry_precedence > source_precedence:
                # Check if the denied path matches
                normalized_path = os.path.normpath(path)
                normalized_denied = os.path.normpath(entry.path)
                if (normalized_path == normalized_denied or
                    normalized_path.startswith(normalized_denied + os.sep)):
                    return True

        return False

    # ==================== Session Config Persistence ====================

    def _ensure_session_config_dir(self) -> Optional[Path]:
        """Ensure session config directory exists and return path."""
        session_path = self._get_session_config_path()
        if not session_path:
            return None

        session_path.parent.mkdir(parents=True, exist_ok=True)
        return session_path

    def _load_session_config(self) -> Dict[str, Any]:
        """Load current session config or return empty structure."""
        session_path = self._get_session_config_path()
        if not session_path or not session_path.exists():
            return {"allowed_paths": [], "denied_paths": []}

        try:
            with open(session_path, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return {"allowed_paths": [], "denied_paths": []}

    def _save_session_config(self, data: Dict[str, Any]) -> bool:
        """Save session config to file."""
        session_path = self._ensure_session_config_dir()
        if not session_path:
            return False

        try:
            with open(session_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except IOError as e:
            self._trace(f"Error saving session config: {e}")
            return False

    # ==================== Programmatic API ====================

    def add_path_programmatic(self, path: str, access: str = "readonly") -> bool:
        """Add a path to the sandbox programmatically.

        Used by other plugins (e.g., references) to grant access to paths
        without going through the user command flow. The path is persisted
        in the session config and synced to the registry.

        If the sandbox has no workspace or session configured, falls back
        to registering directly with the plugin registry for immediate
        (non-persisted) effect.

        Args:
            path: Absolute path to add. Must already be resolved/absolute.
            access: Access mode - "readonly" or "readwrite" (default: "readonly").

        Returns:
            True if the path was added successfully, False on error.
        """
        if access not in ("readonly", "readwrite"):
            self._trace(f"add_path_programmatic: invalid access mode {access!r}")
            return False

        # Normalize the path
        path = os.path.normpath(os.path.abspath(path))

        self._trace(f"add_path_programmatic: path={path}, access={access}")

        # If we can't persist (no workspace or session), fall back to direct
        # registry authorization for immediate effect and queue for later
        # persistence.  When workspace/session become available,
        # _replay_pending_paths() will persist these entries so they survive
        # config reloads (which clear in-memory registry state).
        if not self._workspace_path or not self._get_current_session_id():
            if self._registry:
                self._registry.authorize_external_path(path, self.name, access=access)
                # Store so _replay_pending_paths() can persist later
                if (path, access) not in self._pending_programmatic_paths:
                    self._pending_programmatic_paths.append((path, access))
                self._trace(f"add_path_programmatic: fallback to direct registry auth (queued for persistence)")
                return True
            return False

        # Check if already in session config with same access
        session_config = self._load_session_config()
        for item in session_config.get("allowed_paths", []):
            existing_path = item["path"] if isinstance(item, dict) else item
            if existing_path == path:
                existing_access = item.get("access", "readwrite") if isinstance(item, dict) else "readwrite"
                if existing_access == access:
                    self._trace(f"add_path_programmatic: already allowed with same access")
                    return True
                # Update access mode
                if isinstance(item, dict):
                    item["access"] = access
                    if not self._save_session_config(session_config):
                        return False
                    self._load_all_configs()
                    return True

        # Remove from denied_paths if present
        session_config["denied_paths"] = [
            p for p in session_config.get("denied_paths", [])
            if (p["path"] if isinstance(p, dict) else p) != path
        ]

        # Add to allowed_paths
        session_config.setdefault("allowed_paths", []).append({
            "path": path,
            "access": access,
            "added_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })

        # Save and reload
        if not self._save_session_config(session_config):
            return False

        self._load_all_configs()
        self._trace(f"add_path_programmatic: added successfully")
        return True

    def remove_path_programmatic(self, path: str) -> bool:
        """Remove a path from the sandbox programmatically.

        Used by other plugins (e.g., references) to revoke access to paths
        that were previously added via add_path_programmatic(). If the path
        was added at the session level, it is simply removed from allowed_paths.
        If it was allowed at a higher level, it is added to denied_paths.

        If the sandbox has no workspace or session configured, falls back
        to deauthorizing directly from the plugin registry.

        Args:
            path: Absolute path to remove. Must already be resolved/absolute.

        Returns:
            True if the path was removed successfully, False on error.
        """
        # Normalize the path
        path = os.path.normpath(os.path.abspath(path))

        self._trace(f"remove_path_programmatic: path={path}")

        # If we can't persist, fall back to direct registry deauthorization
        if not self._workspace_path or not self._get_current_session_id():
            if self._registry:
                self._registry.deauthorize_external_path(path, self.name)
                self._trace(f"remove_path_programmatic: fallback to direct registry deauth")
                return True
            return False

        # Load session config
        session_config = self._load_session_config()

        # Check if path was added to session's allowed_paths
        existing_allowed = [
            p["path"] if isinstance(p, dict) else p
            for p in session_config.get("allowed_paths", [])
        ]
        was_session_allowed = path in existing_allowed

        if was_session_allowed:
            # Symmetric undo: just remove from allowed_paths
            session_config["allowed_paths"] = [
                p for p in session_config.get("allowed_paths", [])
                if (p["path"] if isinstance(p, dict) else p) != path
            ]
        else:
            # Not in session allowed - nothing to remove
            self._trace(f"remove_path_programmatic: path not in session allowed_paths")
            return False

        # Save and reload
        if not self._save_session_config(session_config):
            return False

        self._load_all_configs()
        self._trace(f"remove_path_programmatic: removed successfully")
        return True

    # ==================== User Commands ====================

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas - sandbox_manager has no model tools."""
        return []

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executors for user commands."""
        return {
            "sandbox": self._execute_sandbox_command,
        }

    def get_system_instructions(self) -> Optional[str]:
        """No system instructions needed - this is a user-only plugin."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """Return auto-approved tools (user commands don't need approval)."""
        return ["sandbox"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands."""
        return [
            UserCommand(
                name="sandbox",
                description="Manage sandbox path permissions (list|add|remove)",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="subcommand",
                        description="Action: list, add, or remove",
                        required=False,
                    ),
                    CommandParameter(
                        name="path",
                        description="Path to add or remove",
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
        """Provide completions for the sandbox command."""
        if command != "sandbox":
            return []

        # First argument: subcommand
        if len(args) <= 1:
            subcommands = [
                CommandCompletion("list", "Show all effective sandbox paths"),
                CommandCompletion("add", "Allow a path for this session"),
                CommandCompletion("remove", "Block a path for this session"),
                CommandCompletion("help", "Show detailed help for this command"),
            ]
            if args:
                partial = args[0].lower()
                return [s for s in subcommands if s.value.startswith(partial)]
            return subcommands

        # Second argument for "add": access mode
        if len(args) == 2 and args[0] == "add":
            access_modes = [
                CommandCompletion("readonly", "Read-only access (readFile, glob, grep)"),
                CommandCompletion("readwrite", "Full access including writes"),
            ]
            partial = args[1].lower()
            return [m for m in access_modes if m.value.startswith(partial)]

        # For add (after access mode) and remove, could provide path completions in the future
        return []

    def _execute_sandbox_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the sandbox command."""
        subcommand = args.get("subcommand", "list")
        path = args.get("path", "")

        self._trace(f"sandbox command: subcommand={subcommand}, path={path}")

        if subcommand == "list":
            return self._cmd_list()
        elif subcommand == "add":
            if not path:
                return {"error": "Usage: sandbox add <readonly|readwrite> <path>"}
            # Parse mandatory access mode from the beginning of the path argument
            # Syntax: sandbox add <readonly|readwrite> <path>
            parts = path.split(None, 1)  # split into at most 2 parts
            if parts[0] not in ("readonly", "readwrite"):
                return {"error": "Usage: sandbox add <readonly|readwrite> <path>"}
            access = parts[0]
            path = parts[1] if len(parts) > 1 else ""
            if not path:
                return {"error": "Usage: sandbox add <readonly|readwrite> <path>"}
            return self._cmd_add(path, access=access)
        elif subcommand == "remove":
            if not path:
                return {"error": "Path is required for 'sandbox remove'"}
            return self._cmd_remove(path)
        elif subcommand == "help":
            return self._cmd_help()
        else:
            return {"error": f"Unknown subcommand: {subcommand}. Use: list, add, remove, help"}

    def _cmd_help(self) -> HelpLines:
        """Return detailed help text for pager display."""
        return HelpLines(lines=[
            ("Sandbox Command", "bold"),
            ("", ""),
            ("Manage sandbox path permissions. The sandbox restricts which paths the model", ""),
            ("can access for file operations and command execution.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    sandbox [subcommand] [access path]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list              Show all effective sandbox paths from all config levels", "dim"),
            ("                      Displays path, action (allow/deny), source, and timestamp", "dim"),
            ("                      (this is the default when no subcommand is given)", "dim"),
            ("", ""),
            ("    add <access> <path>  Allow a path for the current session", "dim"),
            ("                      access: readonly or readwrite", "dim"),
            ("                      Path is added to session-level allowlist", "dim"),
            ("                      access: readonly or readwrite (default: readwrite)", "dim"),
            ("                      Takes precedence over workspace/global denials", "dim"),
            ("", ""),
            ("    remove <path>     Block a path for the current session", "dim"),
            ("                      Path is added to session-level denylist", "dim"),
            ("                      Takes precedence over workspace/global allowances", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    sandbox                   List all sandbox paths", "dim"),
            ("    sandbox list              Same as above", "dim"),
            ("    sandbox add readwrite /tmp/scratch    Allow readwrite access to /tmp/scratch", "dim"),
            ("    sandbox add readwrite ~/projects      Allow readwrite access to home projects", "dim"),
            ("    sandbox add readonly /docs            Allow read-only access to /docs", "dim"),
            ("    sandbox add readwrite ~/data          Allow full access to ~/data", "dim"),
            ("    sandbox remove /etc                   Block access to /etc", "dim"),
            ("    sandbox remove ~/.ssh                 Block access to SSH keys", "dim"),
            ("", ""),
            ("CONFIGURATION HIERARCHY", "bold"),
            ("    Sandbox paths are configured at three levels (later overrides earlier):", ""),
            ("", ""),
            ("    1. Global Config    ~/.jaato/sandbox.json", "dim"),
            ("                        System-wide defaults for all projects", "dim"),
            ("", ""),
            ("    2. Workspace Config .jaato/sandbox.json", "dim"),
            ("                        Project-specific settings", "dim"),
            ("", ""),
            ("    3. Session Config   .jaato/session/sandbox.json", "dim"),
            ("                        Temporary runtime modifications", "dim"),
            ("", ""),
            ("CONFIGURATION FILE FORMAT", "bold"),
            ('    {', "dim"),
            ('      "allowed_paths": [', "dim"),
            ('        "/path/to/allow",', "dim"),
            ('        {"path": "~/docs", "access": "readonly"},', "dim"),
            ('        {"path": "~/projects", "access": "readwrite"}', "dim"),
            ('      ],', "dim"),
            ('      "denied_paths": ["/sensitive/path"]', "dim"),
            ('    }', "dim"),
            ("", ""),
            ("ACCESS MODES", "bold"),
            ("    readonly     Read-only access (readFile, glob, grep allowed)", "dim"),
            ("    readwrite    Full access including writes (default)", "dim"),
            ("", ""),
            ("PATH FORMATS", "bold"),
            ("    /absolute/path          Absolute path", "dim"),
            ("    ~/relative/to/home      Home directory relative", "dim"),
            ("    ./relative/to/workspace Workspace relative (in workspace config)", "dim"),
            ("", ""),
            ("PRECEDENCE RULES", "bold"),
            ("    - Session rules override workspace rules", "dim"),
            ("    - Workspace rules override global rules", "dim"),
            ("    - Within a level, deny takes precedence over allow", "dim"),
            ("    - Explicit rules override pattern matches", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - Session changes persist until the session ends", "dim"),
            ("    - Use 'sandbox list' to see the effective configuration", "dim"),
            ("    - Blocked paths will cause tool execution to fail", "dim"),
            ("    - Paths are normalized and resolved to absolute paths", "dim"),
        ])

    def _cmd_list(self) -> Dict[str, Any]:
        """Execute 'sandbox list' command."""
        if not self._config:
            self._load_all_configs()

        if not self._config:
            return {"error": "No configuration loaded"}

        # Build effective paths list with precedence handling
        effective_paths = []

        # Collect all paths with their effective status
        all_paths = {}  # path -> (action, source, access, added_at)

        # Process in precedence order (global -> workspace -> session)
        # Later entries override earlier ones
        for entry in self._config.allowed_paths:
            all_paths[entry.path] = ("allow", entry.source, entry.access, entry.added_at)

        for entry in self._config.denied_paths:
            all_paths[entry.path] = ("deny", entry.source, None, entry.added_at)

        # Build output (normalize paths for MSYS2 display)
        for path, (action, source, access, added_at) in sorted(all_paths.items()):
            entry_dict = {
                "path": normalize_result_path(path),
                "action": action,
                "source": source,
                "added_at": added_at,
            }
            if access:
                entry_dict["access"] = access
            effective_paths.append(entry_dict)

        return {
            "effective_paths": effective_paths,
            "summary": {
                "total": len(effective_paths),
                "allowed": sum(1 for p in effective_paths if p["action"] == "allow"),
                "denied": sum(1 for p in effective_paths if p["action"] == "deny"),
            }
        }

    def _cmd_add(self, path: str, access: str = "readwrite") -> Dict[str, Any]:
        """Execute 'sandbox add <path>' command.

        Args:
            path: Path to allow.
            access: Access mode - "readonly" or "readwrite" (default: "readwrite").
        """
        if not self._workspace_path:
            return {"error": "No workspace configured"}

        if not self._get_current_session_id():
            return {"error": "No active session. Start or resume a session first."}

        # Strip @ prefix from file completion
        if path.startswith("@"):
            path = path[1:]

        # Convert MSYS2 drive paths (/c/...) to Windows (C:/...) for Python
        path = msys2_to_windows_path(path)

        # Normalize path (expand ~ and make absolute)
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            # Resolve relative to client workspace, not server CWD
            path = os.path.normpath(os.path.join(self._workspace_path, path))

        # Display path for returning to user (MSYS2-friendly)
        display_path = normalize_result_path(path)

        # Load current session config
        session_config = self._load_session_config()

        # Check if already in allowed_paths (with same or broader access)
        for item in session_config.get("allowed_paths", []):
            existing_path = item["path"] if isinstance(item, dict) else item
            if normalized_equals(existing_path, path):
                existing_access = item.get("access", "readwrite") if isinstance(item, dict) else "readwrite"
                if existing_access == access:
                    return {"status": "already_allowed", "path": display_path, "access": access}
                # Update access mode for existing entry
                if isinstance(item, dict):
                    item["access"] = access
                    if not self._save_session_config(session_config):
                        return {"error": "Failed to save session config"}
                    self._load_all_configs()
                    return {"status": "updated", "path": display_path, "access": access, "source": "session"}

        # Remove from denied_paths if present
        session_config["denied_paths"] = [
            p for p in session_config.get("denied_paths", [])
            if not normalized_equals(
                p["path"] if isinstance(p, dict) else p, path
            )
        ]

        # Add to allowed_paths (store in native Windows format for Python APIs)
        session_config.setdefault("allowed_paths", []).append({
            "path": path,
            "access": access,
            "added_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        })

        # Save
        if not self._save_session_config(session_config):
            return {"error": "Failed to save session config"}

        # Reload and sync
        self._load_all_configs()

        return {"status": "added", "path": display_path, "access": access, "source": "session"}

    def _cmd_remove(self, path: str) -> Dict[str, Any]:
        """Execute 'sandbox remove <path>' command.

        Behavior is symmetric with 'sandbox add':
        - If the path was added to session's allowed_paths, just remove it
        - If the path is allowed at global/workspace level, add to denied_paths
        """
        if not self._workspace_path:
            return {"error": "No workspace configured"}

        if not self._get_current_session_id():
            return {"error": "No active session. Start or resume a session first."}

        # Strip @ prefix from file completion
        if path.startswith("@"):
            path = path[1:]

        # Convert MSYS2 drive paths (/c/...) to Windows (C:/...) for Python
        path = msys2_to_windows_path(path)

        # Normalize path (expand ~ and make absolute)
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            # Resolve relative to client workspace, not server CWD
            path = os.path.normpath(os.path.join(self._workspace_path, path))

        # Display path for returning to user (MSYS2-friendly)
        display_path = normalize_result_path(path)

        # Load current session config
        session_config = self._load_session_config()

        # Check if already in denied_paths
        existing_denied = [
            p["path"] if isinstance(p, dict) else p
            for p in session_config.get("denied_paths", [])
        ]
        if any(normalized_equals(d, path) for d in existing_denied):
            return {"status": "already_denied", "path": display_path}

        # Check if path was added to session's allowed_paths
        existing_allowed = [
            p["path"] if isinstance(p, dict) else p
            for p in session_config.get("allowed_paths", [])
        ]
        was_session_allowed = any(normalized_equals(a, path) for a in existing_allowed)

        if was_session_allowed:
            # Symmetric undo: just remove from allowed_paths, don't add to denied
            session_config["allowed_paths"] = [
                p for p in session_config.get("allowed_paths", [])
                if not normalized_equals(
                    p["path"] if isinstance(p, dict) else p, path
                )
            ]
            status = "removed"
        else:
            # Path is allowed at global/workspace level, add to denied_paths to block it
            session_config.setdefault("denied_paths", []).append({
                "path": path,
                "added_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            })
            status = "denied"

        # Save
        if not self._save_session_config(session_config):
            return {"error": "Failed to save session config"}

        # Reload and sync
        self._load_all_configs()

        return {"status": status, "path": display_path, "source": "session"}


def create_plugin() -> SandboxManagerPlugin:
    """Factory function to create the SandboxManager plugin instance."""
    return SandboxManagerPlugin()
