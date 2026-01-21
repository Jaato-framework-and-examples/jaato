
import os
import json
from typing import Any, Dict, List, Optional

# Assuming a base class exists in shared/plugins/base.py
from ..base import BasePlugin, UserCommand, CommandCompletion

# Assuming the registry and utils are available
from ..registry import PluginRegistry
from ..sandbox_utils import check_path_with_jaato_containment

class SandboxManagerPlugin(BasePlugin):
    """
    A plugin to manage sandbox path permissions for the current session.
    """

    def __init__(self, registry: PluginRegistry):
        super().__init__(registry)
        self.session_id: Optional[str] = None
        self.workspace_root: Optional[str] = None
        self.session_sandbox_file: Optional[str] = None

    def set_workspace_path(self, path: str) -> None:
        """Called by the registry when the workspace changes."""
        self.workspace_root = path

    def set_session_id(self, session_id: str) -> None:
        """Called by the registry when a new session starts."""
        self.session_id = session_id
        if self.workspace_root and self.session_id:
            self.session_sandbox_file = os.path.join(
                self.workspace_root, ".jaato", "sessions", self.session_id, "sandbox.json"
            )

    def get_commands(self) -> List[UserCommand]:
        """Exposes the 'sandbox' command to the user."""
        return [
            UserCommand(
                name="sandbox",
                description="Manage sandbox path permissions for the current session.",
                # The callback would be handled by a different mechanism in a real implementation
                # that can parse subcommands. For this placeholder, it's illustrative.
                # callback=self.handle_sandbox_command,
            )
        ]

    def handle_sandbox_command(self, args: List[str]) -> Dict[str, Any]:
        """Handles 'sandbox add', 'sandbox remove', and 'sandbox list'."""
        if not self.session_sandbox_file:
            return {"error": "Session not initialized. Cannot manage sandbox."}

        subcommand = args[0] if args else "list"
        
        if subcommand == "add":
            # Logic for 'sandbox add <path>'
            pass
        elif subcommand == "remove":
            # Logic for 'sandbox remove <path>'
            pass
        elif subcommand == "list":
            # Logic for 'sandbox list'
            pass
        else:
            return {"error": f"Unknown subcommand: {subcommand}"}
        
        return {"status": "ok"} # Placeholder

    def get_command_completions(
        self,
        command: str,
        args: List[str]
    ) -> List[CommandCompletion]:
        """Provide completions for the 'sandbox' command's subcommands."""
        if command != "sandbox":
            return []

        # If the user is typing the first argument (the subcommand)
        if len(args) <= 1:
            subcommands = [
                CommandCompletion("add", "Allow a new path for this session."),
                CommandCompletion("remove", "Block a path for this session."),
                CommandCompletion("list", "List all effective sandbox paths."),
            ]
            # If the user has started typing a subcommand, filter the list
            if args:
                partial = args[0]
                return [s for s in subcommands if s.value.startswith(partial)]
            return subcommands

        # In the future, this could be expanded to provide path completions
        # for the 'add' and 'remove' subcommands.
        
        return []

    # --- Integration with the Plugin Registry for validation ---

    def is_path_denied_by_session(self, path: str) -> bool:
        """
        Checks if the given path is explicitly denied by the current session.
        This method would be called by the central path validator.
        """
        if not self.session_sandbox_file or not os.path.exists(self.session_sandbox_file):
            return False

        try:
            with open(self.session_sandbox_file, "r") as f:
                config = json.load(f)
        except (IOError, json.JSONDecodeError):
            return False # Can't read or parse file, so can't deny
        
        denied_paths = config.get("denied_paths", [])
        
        for item in denied_paths:
            denied_path = item.get("path")
            if denied_path and path.startswith(denied_path):
                return True
        
        return False

def create_plugin(registry: PluginRegistry) -> BasePlugin:
    """The factory function for creating the SandboxManagerPlugin."""
    return SandboxManagerPlugin(registry)

