"""Shared client command definitions.

This module defines client-side commands that are handled locally
(not forwarded to the server) in both IPC and direct modes.
"""

from typing import List, Tuple


# Client-only commands with descriptions (for completion and help)
# Note: Commands with subcommands should include both the base command
# and all subcommands for proper progressive completion.
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
    # Keybindings command with subcommands
    ("keybindings", "Manage keyboard shortcuts"),
    ("keybindings list", "Show current keybinding configuration"),
    ("keybindings set", "Set a keybinding (usage: keybindings set <action> <key> [--save])"),
    ("keybindings set submit", "Set submit key (default: enter)"),
    ("keybindings set newline", "Set newline key (default: escape enter)"),
    ("keybindings set clear_input", "Set clear input key (default: escape escape)"),
    ("keybindings set cancel", "Set cancel key (default: c-c)"),
    ("keybindings set exit", "Set exit key (default: c-d)"),
    ("keybindings set scroll_up", "Set scroll up key (default: pageup)"),
    ("keybindings set scroll_down", "Set scroll down key (default: pagedown)"),
    ("keybindings set scroll_top", "Set scroll to top key (default: home)"),
    ("keybindings set scroll_bottom", "Set scroll to bottom key (default: end)"),
    ("keybindings set nav_up", "Set navigation up key (default: up)"),
    ("keybindings set nav_down", "Set navigation down key (default: down)"),
    ("keybindings set pager_quit", "Set pager quit key (default: q)"),
    ("keybindings set pager_next", "Set pager next key (default: space)"),
    ("keybindings set toggle_plan", "Set toggle plan key (default: c-p)"),
    ("keybindings set toggle_tools", "Set toggle tools key (default: c-t)"),
    ("keybindings set cycle_agents", "Set cycle agents key (default: c-a)"),
    ("keybindings set yank", "Set yank/copy key (default: c-y)"),
    ("keybindings set view_full", "Set view full key (default: v)"),
    ("keybindings profile", "Show/switch terminal-specific profiles"),
    ("keybindings reload", "Reload keybindings from config files"),
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
        ("", "dim"),
    ]


def build_slash_command_help_text() -> List[Tuple[str, str]]:
    """Build help text for slash commands.

    Returns:
        List of (text, style) tuples for display.
    """
    return [
        ("Slash commands:", "bold"),
        ("  Use /command_name [args...] to invoke slash commands from .jaato/commands/.", "dim"),
        ("  - Type / to see available commands with descriptions", "dim"),
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
    lines.extend(build_slash_command_help_text())
    lines.extend(build_keyboard_shortcuts_help_text())

    return lines
