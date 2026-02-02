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

from ..base import UserCommand, CommandCompletion, CommandParameter
from ..model_provider.types import ToolSchema


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
    """

    def __init__(self):
        self._registry = None
        self._workspace_path: Optional[str] = None
        self._session_id: Optional[str] = None
        self._config: Optional[SandboxConfig] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "sandbox_manager"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] [SandboxManager] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

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
        self._initialized = False

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
        for item in data.get("allowed_paths", []):
            if isinstance(item, str):
                allowed.append(SandboxPath(path=item, source=source, action="allow"))
            elif isinstance(item, dict) and "path" in item:
                allowed.append(SandboxPath(
                    path=item["path"],
                    source=source,
                    action="allow",
                    added_at=item.get("added_at")
                ))

        # Parse denied_paths
        for item in data.get("denied_paths", []):
            if isinstance(item, str):
                denied.append(SandboxPath(path=item, source=source, action="deny"))
            elif isinstance(item, dict) and "path" in item:
                denied.append(SandboxPath(
                    path=item["path"],
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

    def _sync_to_registry(self) -> None:
        """Sync current config to the plugin registry."""
        if not self._registry or not self._config:
            return

        # Clear previous state from this plugin
        self._registry.clear_authorized_paths(self.name)
        self._registry.clear_denied_paths(self.name)

        # Register allowed paths
        for entry in self._config.allowed_paths:
            # Skip if this path is denied at a higher precedence level
            if not self._is_path_denied_by_higher_precedence(entry.path, entry.source):
                self._registry.authorize_external_path(entry.path, self.name)

        # Register denied paths
        for entry in self._config.denied_paths:
            self._registry.deny_external_path(entry.path, self.name)

        self._trace(f"Synced to registry: {len(self._config.allowed_paths)} allowed, "
                   f"{len(self._config.denied_paths)} denied")

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

        # For add/remove, could provide path completions in the future
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
                return {"error": "Path is required for 'sandbox add'"}
            return self._cmd_add(path)
        elif subcommand == "remove":
            if not path:
                return {"error": "Path is required for 'sandbox remove'"}
            return self._cmd_remove(path)
        elif subcommand == "help":
            return self._cmd_help()
        else:
            return {"error": f"Unknown subcommand: {subcommand}. Use: list, add, remove, help"}

    def _cmd_help(self) -> Dict[str, Any]:
        """Return detailed help text."""
        help_text = """Sandbox Command

Manage sandbox path permissions. The sandbox restricts which paths the model
can access for file operations and command execution.

USAGE
    sandbox [subcommand] [path]

SUBCOMMANDS
    list              Show all effective sandbox paths from all config levels
                      Displays path, action (allow/deny), source, and timestamp
                      (this is the default when no subcommand is given)

    add <path>        Allow a path for the current session
                      Path is added to session-level allowlist
                      Takes precedence over workspace/global denials

    remove <path>     Block a path for the current session
                      Path is added to session-level denylist
                      Takes precedence over workspace/global allowances

    help              Show this help message

EXAMPLES
    sandbox                   List all sandbox paths
    sandbox list              Same as above
    sandbox add /tmp/scratch  Allow access to /tmp/scratch
    sandbox add ~/projects    Allow access to home projects
    sandbox remove /etc       Block access to /etc
    sandbox remove ~/.ssh     Block access to SSH keys

CONFIGURATION HIERARCHY
    Sandbox paths are configured at three levels (later overrides earlier):

    1. Global Config    ~/.jaato/sandbox.json
                        System-wide defaults for all projects

    2. Workspace Config .jaato/sandbox.json
                        Project-specific settings

    3. Session Config   .jaato/session/sandbox.json
                        Temporary runtime modifications

CONFIGURATION FILE FORMAT
    {
      "allowed_paths": ["/path/to/allow", "~/another/path"],
      "denied_paths": ["/sensitive/path"]
    }

PATH FORMATS
    /absolute/path          Absolute path
    ~/relative/to/home      Home directory relative
    ./relative/to/workspace Workspace relative (in workspace config)

PRECEDENCE RULES
    - Session rules override workspace rules
    - Workspace rules override global rules
    - Within a level, deny takes precedence over allow
    - Explicit rules override pattern matches

NOTES
    - Session changes persist until the session ends
    - Use 'sandbox list' to see the effective configuration
    - Blocked paths will cause tool execution to fail
    - Paths are normalized and resolved to absolute paths"""
        return {"output": help_text}

    def _cmd_list(self) -> Dict[str, Any]:
        """Execute 'sandbox list' command."""
        if not self._config:
            self._load_all_configs()

        if not self._config:
            return {"error": "No configuration loaded"}

        # Build effective paths list with precedence handling
        effective_paths = []

        # Collect all paths with their effective status
        all_paths = {}  # path -> (action, source, added_at)

        # Process in precedence order (global -> workspace -> session)
        # Later entries override earlier ones
        for entry in self._config.allowed_paths:
            all_paths[entry.path] = ("allow", entry.source, entry.added_at)

        for entry in self._config.denied_paths:
            all_paths[entry.path] = ("deny", entry.source, entry.added_at)

        # Build output
        for path, (action, source, added_at) in sorted(all_paths.items()):
            effective_paths.append({
                "path": path,
                "action": action,
                "source": source,
                "added_at": added_at,
            })

        return {
            "effective_paths": effective_paths,
            "summary": {
                "total": len(effective_paths),
                "allowed": sum(1 for p in effective_paths if p["action"] == "allow"),
                "denied": sum(1 for p in effective_paths if p["action"] == "deny"),
            }
        }

    def _cmd_add(self, path: str) -> Dict[str, Any]:
        """Execute 'sandbox add <path>' command."""
        if not self._workspace_path:
            return {"error": "No workspace configured"}

        if not self._get_current_session_id():
            return {"error": "No active session. Start or resume a session first."}

        # Strip @ prefix from file completion
        if path.startswith("@"):
            path = path[1:]

        # Normalize path (expand ~ and make absolute)
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        # Load current session config
        session_config = self._load_session_config()

        # Check if already in allowed_paths
        existing_paths = [
            p["path"] if isinstance(p, dict) else p
            for p in session_config.get("allowed_paths", [])
        ]
        if path in existing_paths:
            return {"status": "already_allowed", "path": path}

        # Remove from denied_paths if present
        session_config["denied_paths"] = [
            p for p in session_config.get("denied_paths", [])
            if (p["path"] if isinstance(p, dict) else p) != path
        ]

        # Add to allowed_paths
        session_config.setdefault("allowed_paths", []).append({
            "path": path,
            "added_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        })

        # Save
        if not self._save_session_config(session_config):
            return {"error": "Failed to save session config"}

        # Reload and sync
        self._load_all_configs()

        return {"status": "added", "path": path, "source": "session"}

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

        # Normalize path (expand ~ and make absolute)
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        # Load current session config
        session_config = self._load_session_config()

        # Check if already in denied_paths
        existing_denied = [
            p["path"] if isinstance(p, dict) else p
            for p in session_config.get("denied_paths", [])
        ]
        if path in existing_denied:
            return {"status": "already_denied", "path": path}

        # Check if path was added to session's allowed_paths
        existing_allowed = [
            p["path"] if isinstance(p, dict) else p
            for p in session_config.get("allowed_paths", [])
        ]
        was_session_allowed = path in existing_allowed

        if was_session_allowed:
            # Symmetric undo: just remove from allowed_paths, don't add to denied
            session_config["allowed_paths"] = [
                p for p in session_config.get("allowed_paths", [])
                if (p["path"] if isinstance(p, dict) else p) != path
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

        return {"status": status, "path": path, "source": "session"}


def create_plugin() -> SandboxManagerPlugin:
    """Factory function to create the SandboxManager plugin instance."""
    return SandboxManagerPlugin()
