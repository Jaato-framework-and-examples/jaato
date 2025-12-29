"""Shared UI formatting utilities.

This module provides common UI formatting functions that can be used
by both IPC mode (client-side) and direct mode (embedded) code.
"""

from typing import Any, Dict, List, Optional, Union


def format_permission_options(
    response_options: List[Union[Dict[str, Any], Any]],
    use_brackets: bool = True
) -> str:
    """Format permission response options for display.

    Works with both dict format (from IPC events) and object format
    (from PermissionResponseOption).

    Args:
        response_options: List of options - either dicts with key/label
            or objects with short/full attributes.
        use_brackets: Whether to wrap keys in brackets like [y]es.

    Returns:
        Formatted options string like "[y]es [n]o [a]lways [once] [never]"
    """
    parts = []
    for opt in response_options:
        # Handle both dict and object formats
        if isinstance(opt, dict):
            key = opt.get('key', opt.get('short', '?'))
            label = opt.get('label', opt.get('full', '?'))
        else:
            key = getattr(opt, 'short', getattr(opt, 'key', '?'))
            label = getattr(opt, 'full', getattr(opt, 'label', '?'))

        # Format: [y]es or [once] depending on whether key differs from label
        if use_brackets:
            if key != label and label.lower().startswith(key.lower()):
                # Key is prefix of label: [y]es, [n]o, [a]lways
                parts.append(f"[{key}]{label[len(key):]}")
            else:
                # Key equals label or is not prefix: [once], [all]
                parts.append(f"[{label}]")
        else:
            parts.append(label)

    return " ".join(parts)


def format_tool_args_summary(tool_args: Dict[str, Any], max_length: int = 60) -> str:
    """Format tool arguments as a truncated summary string.

    Args:
        tool_args: Dictionary of tool arguments.
        max_length: Maximum length before truncation.

    Returns:
        Truncated string representation of args.
    """
    args_str = str(tool_args)
    if len(args_str) > max_length:
        return args_str[:max_length - 3] + "..."
    return args_str


def format_duration(seconds: Optional[float]) -> str:
    """Format a duration in seconds for display.

    Args:
        seconds: Duration in seconds, or None.

    Returns:
        Formatted string like "1.2s" or "".
    """
    if seconds is None:
        return ""
    return f"{seconds:.1f}s"


def format_token_count(count: int) -> str:
    """Format a token count with K/M suffixes for large numbers.

    Args:
        count: Token count.

    Returns:
        Formatted string like "1.2K" or "1.5M".
    """
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def format_percent(value: float, decimals: int = 1) -> str:
    """Format a percentage value.

    Args:
        value: Percentage value (0-100).
        decimals: Number of decimal places.

    Returns:
        Formatted string like "75.5%".
    """
    return f"{value:.{decimals}f}%"


def build_permission_prompt_lines(
    tool_args: Optional[Dict[str, Any]],
    response_options: List[Union[Dict[str, Any], Any]],
    include_tool_name: bool = False,
    tool_name: Optional[str] = None,
) -> List[str]:
    """Build permission prompt lines for display in tool tree.

    This function creates a consistent format for permission prompts
    that works for both IPC mode and direct mode.

    Args:
        tool_args: Arguments passed to the tool.
        response_options: List of valid response options.
        include_tool_name: Whether to include "Tool: <name>" line.
        tool_name: Tool name (required if include_tool_name is True).

    Returns:
        List of prompt lines for display.
    """
    lines = []

    # Optionally include tool name (usually redundant since tool tree shows it)
    if include_tool_name and tool_name:
        lines.append(f"Tool: {tool_name}")

    # Add args summary
    if tool_args:
        lines.append(f"Args: {format_tool_args_summary(tool_args, max_length=100)}")

    # Blank line before options
    lines.append("")

    # Add formatted options
    lines.append(format_permission_options(response_options))

    return lines


def build_clarification_prompt_lines(
    question_text: str,
    question_index: Optional[int] = None,
    total_questions: Optional[int] = None,
) -> List[str]:
    """Build clarification prompt lines for display in tool tree.

    Args:
        question_text: The question text (may contain newlines).
        question_index: Current question number (1-based).
        total_questions: Total number of questions.

    Returns:
        List of prompt lines for display.
    """
    lines = []

    # Add progress header if we have question count
    if question_index is not None and total_questions is not None:
        lines.append(f"Question {question_index}/{total_questions}")
        lines.append("")

    # Split question text into lines
    if question_text:
        lines.extend(question_text.split("\n"))

    return lines


# =============================================================================
# Keybindings Command Handling
# =============================================================================

def handle_keybindings_command(user_input: str, display) -> None:
    """Handle the keybindings command with subcommands.

    Shared implementation used by both direct and IPC modes.

    Subcommands:
        keybindings              - Show current keybindings
        keybindings list         - Show current keybindings
        keybindings reload       - Reload keybindings from config files
        keybindings set <action> <key> [--save]  - Set a keybinding
        keybindings profile      - Show/switch terminal profiles

    Args:
        user_input: The full user input string starting with 'keybindings'.
        display: PTDisplay instance for output and keybinding config access.
    """
    if not display:
        return

    parts = user_input.strip().split()
    subcommand = parts[1].lower() if len(parts) > 1 else "list"

    if subcommand == "list" or subcommand == "keybindings":
        show_keybindings(display)
        return

    if subcommand == "reload":
        reload_keybindings(display)
        return

    if subcommand == "set":
        set_keybinding(parts[2:], display)
        return

    if subcommand == "profile":
        handle_profile_command(parts[2:], display)
        return

    # Unknown subcommand - show help
    display.show_lines([
        (f"[Unknown subcommand: {subcommand}]", "yellow"),
        ("  Available subcommands:", ""),
        ("    keybindings list              - Show current keybindings", "dim"),
        ("    keybindings set <action> <key> [--save]", "dim"),
        ("                                  - Set a keybinding (optionally persist)", "dim"),
        ("    keybindings profile           - Show current terminal profile", "dim"),
        ("    keybindings profile <name>    - Switch to a different profile", "dim"),
        ("    keybindings reload            - Reload keybindings from config", "dim"),
    ])


def show_keybindings(display) -> None:
    """Show current keybinding configuration.

    Args:
        display: PTDisplay instance.
    """
    if not display:
        return

    # Import keybindings module for detect_terminal
    try:
        from keybindings import detect_terminal
    except ImportError:
        detect_terminal = lambda: "unknown"

    config = display._keybinding_config
    bindings = config.to_dict()

    # Show profile info first
    detected = detect_terminal()
    lines = [
        ("Current Keybindings:", "bold"),
        (f"  Profile: {config.profile}", "cyan"),
        (f"  Terminal: {detected}", "dim"),
        (f"  Source: {config.profile_source}", "dim"),
        ("", ""),
    ]

    # Group by category
    categories = {
        "Input": ["submit", "newline", "clear_input"],
        "Exit/Cancel": ["cancel", "exit"],
        "Scrolling": ["scroll_up", "scroll_down", "scroll_top", "scroll_bottom"],
        "Navigation": ["nav_up", "nav_down"],
        "Pager": ["pager_quit", "pager_next"],
        "Features": ["toggle_plan", "toggle_tools", "cycle_agents", "yank", "view_full"],
        "Tool Navigation": ["tool_nav_enter", "tool_expand", "tool_exit", "tool_output_up", "tool_output_down"],
    }

    for category, keys in categories.items():
        lines.append((f"  {category}:", "cyan"))
        for key in keys:
            if key in bindings:
                value = bindings[key]
                if isinstance(value, list):
                    value_str = " ".join(value)
                else:
                    value_str = value
                padding = max(2, 16 - len(key))
                lines.append((f"    {key}{' ' * padding}{value_str}", "dim"))
        lines.append(("", ""))

    lines.extend([
        ("Profile-specific config files:", "bold"),
        (f"  .jaato/keybindings.{detected}.json (terminal-specific)", "dim"),
        ("  .jaato/keybindings.json (base)", "dim"),
        ("  Environment: JAATO_KEYBINDING_PROFILE=<name>", "dim"),
        ("", ""),
        ("Use 'keybindings profile' to view/switch profiles.", "italic"),
    ])

    display.show_lines(lines)


def reload_keybindings(display) -> None:
    """Reload keybindings from configuration files.

    Args:
        display: PTDisplay instance.
    """
    if not display:
        return

    try:
        display.reload_keybindings()
        display.show_lines([
            ("[Keybindings reloaded successfully]", "green"),
            ("New keybindings are now active.", "dim"),
        ])
    except Exception as e:
        display.show_lines([
            (f"[Error reloading keybindings: {e}]", "red"),
        ])


def set_keybinding(args: list, display) -> None:
    """Set a keybinding for an action.

    Args:
        args: List of arguments: <action> <key> [--save]
        display: PTDisplay instance.
    """
    if not display:
        return

    try:
        from keybindings import DEFAULT_KEYBINDINGS
    except ImportError:
        display.show_lines([("[Error: keybindings module not available]", "red")])
        return

    # Parse arguments
    save_to_file = "--save" in args
    args = [a for a in args if a != "--save"]

    if len(args) < 2:
        display.show_lines([
            ("[Error: Missing arguments]", "red"),
            ("  Usage: keybindings set <action> <key> [--save]", "dim"),
            ("", ""),
            ("  Examples:", "bold"),
            ("    keybindings set yank c-shift-y", "dim"),
            ("    keybindings set toggle_plan f1 --save", "dim"),
            ("    keybindings set newline escape enter", "dim"),
            ("", ""),
            ("  Available actions:", "bold"),
        ])
        # Show available actions
        actions = list(DEFAULT_KEYBINDINGS.keys())
        for i in range(0, len(actions), 4):
            chunk = actions[i:i+4]
            display.show_lines([
                ("    " + ", ".join(chunk), "dim"),
            ])
        return

    action = args[0].lower()
    # Key can be multiple words for multi-key sequences (e.g., "escape enter")
    key = " ".join(args[1:])

    # Validate action
    if action not in DEFAULT_KEYBINDINGS:
        display.show_lines([
            (f"[Error: Unknown action '{action}']", "red"),
            ("", ""),
            ("  Available actions:", "bold"),
        ])
        actions = list(DEFAULT_KEYBINDINGS.keys())
        for i in range(0, len(actions), 4):
            chunk = actions[i:i+4]
            display.show_lines([
                ("    " + ", ".join(chunk), "dim"),
            ])
        return

    # Get current config and set the binding
    config = display._keybinding_config
    old_value = getattr(config, action)
    if isinstance(old_value, list):
        old_str = " ".join(old_value)
    else:
        old_str = old_value

    # Set the new binding
    if not config.set_binding(action, key):
        display.show_lines([
            (f"[Error: Failed to set binding for '{action}']", "red"),
        ])
        return

    # Rebuild the app with new keybindings
    display._build_app()

    # Get the normalized key for display
    new_value = getattr(config, action)
    if isinstance(new_value, list):
        new_str = " ".join(new_value)
    else:
        new_str = new_value

    lines = [
        (f"[Keybinding updated: {action}]", "green"),
        (f"  {old_str} → {new_str}", "dim"),
    ]

    # Save to file if requested
    if save_to_file:
        if config.save_to_file():
            lines.append(("  Saved to .jaato/keybindings.json", "cyan"))
        else:
            lines.append(("  [Warning: Failed to save to file]", "yellow"))
    else:
        lines.append(("  (session only - use --save to persist)", "dim italic"))

    display.show_lines(lines)


def handle_profile_command(args: list, display) -> None:
    """Handle the keybindings profile subcommand.

    Args:
        args: List of arguments after 'keybindings profile'
        display: PTDisplay instance.
    """
    if not display:
        return

    try:
        from keybindings import detect_terminal, list_available_profiles, load_keybindings
    except ImportError:
        display.show_lines([("[Error: keybindings module not available]", "red")])
        return

    config = display._keybinding_config
    detected = detect_terminal()
    available = list_available_profiles()

    # No args - show current profile and available profiles
    if not args:
        lines = [
            ("Keybinding Profiles:", "bold"),
            ("", ""),
            (f"  Detected terminal: {detected}", "cyan"),
            (f"  Current profile:   {config.profile}", "green"),
            (f"  Source:            {config.profile_source}", "dim"),
            ("", ""),
            ("  Available profiles:", "bold"),
        ]

        for profile in available:
            marker = " (active)" if profile == config.profile else ""
            marker2 = " (detected)" if profile == detected and profile != config.profile else ""
            lines.append((f"    - {profile}{marker}{marker2}", "dim"))

        lines.extend([
            ("", ""),
            ("  To switch profiles:", "bold"),
            ("    keybindings profile <name>     - Switch to a profile", "dim"),
            ("    JAATO_KEYBINDING_PROFILE=name  - Override via env var", "dim"),
            ("", ""),
            ("  To create a profile:", "bold"),
            (f"    Create .jaato/keybindings.{detected}.json", "dim"),
        ])

        display.show_lines(lines)
        return

    # Switch to specified profile
    new_profile = args[0].lower()

    # Load with new profile
    try:
        new_config = load_keybindings(profile=new_profile)
        display._keybinding_config = new_config
        display._build_app()

        lines = [
            (f"[Switched to profile: {new_profile}]", "green"),
            (f"  Source: {new_config.profile_source}", "dim"),
        ]

        if new_profile not in available:
            lines.append((f"  (no config file found, using defaults)", "yellow"))

        display.show_lines(lines)

    except Exception as e:
        display.show_lines([
            (f"[Error switching profile: {e}]", "red"),
        ])
