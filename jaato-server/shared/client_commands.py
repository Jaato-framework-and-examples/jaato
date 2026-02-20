"""Shared client command definitions.

This module defines client-side commands that are handled locally
(not forwarded to the server) in both IPC and direct modes.

Also provides a unified command parser for routing user input to the
appropriate handler (client command, server command, or model message).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


# Client-only commands with descriptions (for help display)
# Completion is handled by DEFAULT_COMMANDS in file_completer.py
# This list is used for command routing (deciding if command is local vs server)
CLIENT_COMMANDS: List[Tuple[str, str]] = [
    ("help", "Show available commands"),
    ("history", "Show conversation history"),
    ("exit", "Exit the client"),
    ("quit", "Exit the client"),
    ("stop", "Stop current model generation"),
    ("clear", "Clear output display"),
    ("context", "Show context window usage"),
    ("reset", "Clear conversation history"),
    ("plugins", "Show available plugins"),
    ("keybindings", "Manage keyboard shortcuts"),
    ("export", "Export session to file"),
]


def get_client_command_names() -> set:
    """Get set of client command names for routing.

    Returns:
        Set of command names that should be handled locally.
    """
    return {cmd[0] for cmd in CLIENT_COMMANDS}


def build_base_help_text() -> List[Tuple[str, str]]:
    """Build the base help text lines for client commands.

    Returns:
        List of (text, style) tuples for display.
    """
    return [
        ("Commands (auto-complete as you type):", "bold"),
        ("  help              - Show this help message", "dim"),
        ("  tools [subcmd]    - Manage tools available to the model", "dim"),
        ("                        tools list          - List all tools with status", "dim"),
        ("                        tools enable <n>    - Enable a tool (or 'all')", "dim"),
        ("                        tools disable <n>   - Disable a tool (or 'all')", "dim"),
        ("  session [subcmd]  - Manage sessions", "dim"),
        ("                        session list        - List all sessions", "dim"),
        ("                        session new         - Create a new session", "dim"),
        ("                        session attach <id> - Attach to a session", "dim"),
        ("                        session delete <id> - Delete a session", "dim"),
        ("  reset             - Clear conversation history", "dim"),
        ("  clear             - Clear output panel", "dim"),
        ("  stop              - Stop current model generation", "dim"),
        ("  quit              - Exit the client", "dim"),
        ("", "dim"),
    ]


def build_permission_help_text() -> List[Tuple[str, str]]:
    """Build help text for permission prompts.

    Returns:
        List of (text, style) tuples for display.
    """
    return [
        ("When the model tries to use a tool, you'll see a permission prompt:", "bold"),
        ("  [y]es     - Allow this execution", "dim"),
        ("  [n]o      - Deny this execution", "dim"),
        ("  [a]lways  - Allow and remember for this session", "dim"),
        ("  [never]   - Deny and block for this session", "dim"),
        ("  [once]    - Allow just this once", "dim"),
        ("", "dim"),
    ]


def build_file_reference_help_text() -> List[Tuple[str, str]]:
    """Build help text for file references.

    Returns:
        List of (text, style) tuples for display.
    """
    return [
        ("File references:", "bold"),
        ("  Use @path/to/file to include file contents in your prompt.", "dim"),
        ("  - @src/main.py      - Reference a file (contents included)", "dim"),
        ("  - @./config.json    - Reference with explicit relative path", "dim"),
        ("  - @~/documents/     - Reference with home directory", "dim"),
        ("  Use @@path to browse sandbox-allowed paths (workspace, external, /tmp).", "dim"),
        ("  - @@                - Show all sandbox-allowed root paths", "dim"),
        ("  - @@/tmp/output/    - Browse files in an allowed path", "dim"),
        ("", "dim"),
    ]



def build_keyboard_shortcuts_help_text() -> List[Tuple[str, str]]:
    """Build help text for keyboard shortcuts.

    Returns:
        List of (text, style) tuples for display.
    """
    return [
        ("Keyboard shortcuts:", "bold"),
        ("  ↑/↓       - Navigate prompt history (or completion menu)", "dim"),
        ("  ←/→       - Move cursor within line", "dim"),
        ("  Ctrl+Y    - Yank (copy) last response to clipboard", "dim"),
        ("  TAB/Enter - Accept selected completion", "dim"),
        ("  Escape    - Dismiss completion menu", "dim"),
        ("  Esc+Esc   - Clear input", "dim"),
        ("  PgUp/PgDn - Scroll output up/down", "dim"),
        ("  Home/End  - Scroll to top/bottom of output", "dim"),
    ]


def build_full_help_text(server_commands: List[dict] = None) -> List[Tuple[str, str]]:
    """Build complete help text including all sections.

    Args:
        server_commands: Optional list of server/plugin commands to include.

    Returns:
        List of (text, style) tuples for display.
    """
    lines = []
    lines.extend(build_base_help_text())

    # Add server/plugin commands if provided
    if server_commands:
        lines.append(("Server/plugin commands:", "bold"))
        for cmd in server_commands:
            name = cmd.get("name", "")
            desc = cmd.get("description", "")
            # Skip session commands (already shown in base help)
            if name.startswith("session "):
                continue
            padding = max(2, 18 - len(name))
            lines.append((f"  {name}{' ' * padding}- {desc}", "dim"))
        lines.append(("", "dim"))

    lines.extend(build_permission_help_text())
    lines.extend(build_file_reference_help_text())
    lines.extend(build_keyboard_shortcuts_help_text())

    return lines


# =============================================================================
# Command Parsing and Routing
# =============================================================================

class CommandAction(Enum):
    """Type of action to take for parsed user input."""
    EXIT = "exit"              # Exit/quit the client
    STOP = "stop"              # Stop model processing
    CLEAR = "clear"            # Clear display (client-only, no-op in headless)
    HELP = "help"              # Show help / request command list
    CONTEXT = "context"        # Show context usage (client-only)
    HISTORY = "history"        # Request conversation history
    RESET = "reset"            # Reset/clear session history
    SERVER_COMMAND = "server"  # Forward command to server
    SEND_MESSAGE = "message"   # Send as message to model


@dataclass
class ParsedCommand:
    """Result of parsing user input.

    Attributes:
        action: What kind of action to take.
        command: Command name for SERVER_COMMAND (e.g., "tools.list", "permissions").
        args: Command arguments for SERVER_COMMAND.
        text: Original text for SEND_MESSAGE.
    """
    action: CommandAction
    command: Optional[str] = None
    args: Optional[List[str]] = None
    text: Optional[str] = None


# Known server/plugin command prefixes
SERVER_COMMAND_PREFIXES = {
    "tools", "session", "permissions", "model", "mcp", "save", "resume",
    "memory", "lsp", "todo", "waypoint", "background", "prompt-library",
    "clarification", "multimodal", "notebook", "references", "sandbox",
    "workspace", "reliability",
}


def parse_user_input(
    text: str,
    server_commands: Optional[List[dict]] = None,
) -> ParsedCommand:
    """Parse user input and determine what action to take.

    This function implements the same routing logic as the TUI, making it
    reusable for command mode and other clients.

    Args:
        text: Raw user input text.
        server_commands: Optional list of server commands for matching.
            Each dict should have "name" key with the command name.

    Returns:
        ParsedCommand indicating what action to take.
    """
    text = text.strip()
    if not text:
        return ParsedCommand(action=CommandAction.SEND_MESSAGE, text="")

    text_lower = text.lower()
    parts = text.split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:] if len(parts) > 1 else []

    # ==================== Client-only commands ====================

    if text_lower in ("exit", "quit", "q"):
        return ParsedCommand(action=CommandAction.EXIT)

    if text_lower == "stop":
        return ParsedCommand(action=CommandAction.STOP)

    if text_lower == "clear":
        return ParsedCommand(action=CommandAction.CLEAR)

    if text_lower == "help":
        return ParsedCommand(action=CommandAction.HELP)

    if text_lower == "context":
        return ParsedCommand(action=CommandAction.CONTEXT)

    if text_lower == "history":
        return ParsedCommand(action=CommandAction.HISTORY)

    if cmd == "reset":
        return ParsedCommand(
            action=CommandAction.SERVER_COMMAND,
            command="reset",
            args=args,
        )

    # ==================== Server commands with subcommands ====================

    if cmd == "tools":
        subcmd = args[0] if args else "list"
        subargs = args[1:] if len(args) > 1 else []
        return ParsedCommand(
            action=CommandAction.SERVER_COMMAND,
            command=f"tools.{subcmd}",
            args=subargs,
        )

    if cmd == "session":
        subcmd = args[0] if args else "list"
        subargs = args[1:] if len(args) > 1 else []
        return ParsedCommand(
            action=CommandAction.SERVER_COMMAND,
            command=f"session.{subcmd}",
            args=subargs,
        )

    # ==================== Known server/plugin commands ====================

    if cmd in SERVER_COMMAND_PREFIXES:
        return ParsedCommand(
            action=CommandAction.SERVER_COMMAND,
            command=cmd,
            args=args,
        )

    # ==================== Match against server command list ====================

    if server_commands:
        for srv_cmd in server_commands:
            cmd_name = srv_cmd.get("name", "").lower()
            cmd_parts = cmd_name.split()
            if not cmd_parts:
                continue

            base_cmd = cmd_parts[0]
            if text_lower == base_cmd or text_lower.startswith(base_cmd + " "):
                # Extract args after base command
                input_parts = text.split()
                command_args = input_parts[1:] if len(input_parts) > 1 else []
                return ParsedCommand(
                    action=CommandAction.SERVER_COMMAND,
                    command=base_cmd,
                    args=command_args,
                )

    # ==================== Default: send as message to model ====================

    return ParsedCommand(action=CommandAction.SEND_MESSAGE, text=text)
