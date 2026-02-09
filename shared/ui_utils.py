"""Shared UI formatting utilities.

This module provides common UI formatting functions that can be used
by both IPC mode (client-side) and direct mode (embedded) code.
"""

import os
from typing import Any, Dict, List, Optional, Union

from shared.path_utils import get_display_separator, normalize_path


# =============================================================================
# Path Ellipsization
# =============================================================================


def ellipsize_path(
    path: str,
    max_width: int,
    *,
    keep_first: int = 1,
    keep_last: int = 2,
    ellipsis: str = "...",
) -> str:
    """Ellipsize a file path in the middle to fit within max_width.

    Uses a "middle-ellipsis with smart segmentation" strategy that preserves:
    - First `keep_first` path segments (project/module context)
    - Last `keep_last` path segments (immediate parent + filename)
    - Collapses middle segments with ellipsis

    Examples:
        >>> ellipsize_path("customer-domain-api/src/main/java/com/bank/model/Customer.java", 50)
        'customer-domain-api/.../model/Customer.java'

        >>> ellipsize_path("/home/user/project/src/components/Button.tsx", 40)
        '/home/.../components/Button.tsx'

        >>> ellipsize_path("short/path.txt", 50)  # No ellipsis needed
        'short/path.txt'

    Args:
        path: The file path to ellipsize.
        max_width: Maximum width in characters. Must be positive.
        keep_first: Number of leading path segments to preserve (default: 1).
        keep_last: Number of trailing path segments to preserve (default: 2).
        ellipsis: The ellipsis string to use (default: "...").

    Returns:
        The ellipsized path if it exceeds max_width, otherwise the original path.
        If even the minimal form (first + ellipsis + last) exceeds max_width,
        falls back to showing just the filename with left-ellipsis.
    """
    if not path or max_width <= 0:
        return path

    # Normalize path for MSYS2 display (backslash -> forward slash)
    path = normalize_path(path)

    # Fast path: no truncation needed
    if len(path) <= max_width:
        return path

    # Determine separator for this path
    sep = get_display_separator()
    # Handle both Unix and Windows paths
    if "/" in path and "\\" not in path:
        sep = "/"
    elif "\\" in path:
        sep = "\\"

    # Split into segments, preserving leading separator for absolute paths
    has_leading_sep = path.startswith(sep)
    segments = [s for s in path.split(sep) if s]

    if not segments:
        return path

    # If we have very few segments, can't do middle ellipsis meaningfully
    total_segments = len(segments)
    if total_segments <= keep_first + keep_last:
        # Not enough segments to ellipsize in the middle
        # Fall back to filename-only with left ellipsis if still too long
        filename = segments[-1]
        if len(filename) <= max_width:
            return path if len(path) <= max_width else ellipsis + filename
        # Even filename is too long - truncate it
        return ellipsis + filename[-(max_width - len(ellipsis)):]

    # Build the ellipsized path
    first_segments = segments[:keep_first]
    last_segments = segments[-keep_last:]

    # Construct with middle ellipsis
    first_part = sep.join(first_segments)
    last_part = sep.join(last_segments)

    if has_leading_sep:
        first_part = sep + first_part

    # Try: first/.../ last
    candidate = f"{first_part}{sep}{ellipsis}{sep}{last_part}"

    if len(candidate) <= max_width:
        return candidate

    # Still too long - progressively reduce keep_first
    for reduced_first in range(keep_first - 1, -1, -1):
        if reduced_first > 0:
            first_segments = segments[:reduced_first]
            first_part = sep.join(first_segments)
            if has_leading_sep:
                first_part = sep + first_part
            candidate = f"{first_part}{sep}{ellipsis}{sep}{last_part}"
        else:
            # No first segments - just ellipsis + last
            candidate = f"{ellipsis}{sep}{last_part}"

        if len(candidate) <= max_width:
            return candidate

    # Still too long - progressively reduce keep_last
    for reduced_last in range(keep_last - 1, 0, -1):
        last_segments = segments[-reduced_last:]
        last_part = sep.join(last_segments)
        candidate = f"{ellipsis}{sep}{last_part}"

        if len(candidate) <= max_width:
            return candidate

    # Last resort: just the filename with left ellipsis
    filename = segments[-1]
    if len(ellipsis) + len(filename) <= max_width:
        return ellipsis + filename

    # Even filename is too long - truncate it from the left
    available = max_width - len(ellipsis)
    if available > 0:
        return ellipsis + filename[-available:]

    # Pathological case: max_width is tiny
    return path[:max_width]


def ellipsize_path_pair(
    source: str,
    dest: str,
    max_width: int,
    *,
    separator: str = " -> ",
    keep_first: int = 1,
    keep_last: int = 2,
    ellipsis: str = "...",
) -> str:
    """Ellipsize a source->destination path pair to fit within max_width.

    Useful for move/rename operations. Allocates space proportionally
    between source and destination paths.

    Example:
        >>> ellipsize_path_pair("/long/source/path/file.txt", "/long/dest/path/new.txt", 50)
        '.../path/file.txt -> .../path/new.txt'

    Args:
        source: Source file path.
        dest: Destination file path.
        max_width: Maximum total width.
        separator: String between source and dest (default: " -> ").
        keep_first: Segments to keep at start (default: 1).
        keep_last: Segments to keep at end (default: 2).
        ellipsis: Ellipsis string (default: "...").

    Returns:
        Formatted "source -> dest" string, ellipsized if needed.
    """
    full = f"{source}{separator}{dest}"
    if len(full) <= max_width:
        return full

    # Allocate space: separator is fixed, split remaining between paths
    available = max_width - len(separator)
    if available <= 0:
        return full[:max_width]

    # Give each path half the available space
    per_path = available // 2

    source_ellipsized = ellipsize_path(
        source, per_path, keep_first=keep_first, keep_last=keep_last, ellipsis=ellipsis
    )
    dest_ellipsized = ellipsize_path(
        dest, per_path, keep_first=keep_first, keep_last=keep_last, ellipsis=ellipsis
    )

    return f"{source_ellipsized}{separator}{dest_ellipsized}"


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

    if subcommand == "help":
        display.show_lines([
            ("Keybindings Command", "bold"),
            ("", ""),
            ("Manage keyboard shortcuts and terminal profiles. Keybindings can be", ""),
            ("customized per-terminal and saved to configuration files.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    keybindings [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list              Show all current keybindings with their values", "dim"),
            ("                      (this is the default when no subcommand is given)", "dim"),
            ("", ""),
            ("    set <action> <key> [--save]", "dim"),
            ("                      Set a keybinding for an action", "dim"),
            ("                      Use --save to persist to config file", "dim"),
            ("", ""),
            ("    profile           Show current terminal profile and available profiles", "dim"),
            ("    profile <name>    Switch to a different keybinding profile", "dim"),
            ("", ""),
            ("    reload            Reload keybindings from configuration files", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    keybindings                      Show current keybindings", "dim"),
            ("    keybindings set yank c-shift-y   Set yank to Ctrl+Shift+Y (session)", "dim"),
            ("    keybindings set toggle_plan f1 --save", "dim"),
            ("                                     Set toggle_plan to F1 and save", "dim"),
            ("    keybindings set newline escape enter", "dim"),
            ("                                     Set newline to Escape then Enter", "dim"),
            ("    keybindings profile              Show available profiles", "dim"),
            ("    keybindings profile kitty        Switch to kitty profile", "dim"),
            ("    keybindings reload               Reload from config files", "dim"),
            ("", ""),
            ("KEY SYNTAX (prompt_toolkit format)", "bold"),
            ("    c-x               Ctrl+X", "dim"),
            ("    s-x               Shift+X (where applicable)", "dim"),
            ("    c-s-x             Ctrl+Shift+X", "dim"),
            ("    escape            Escape key", "dim"),
            ("    enter             Enter key", "dim"),
            ("    f1, f2, ...       Function keys", "dim"),
            ("    pageup, pagedown  Page navigation keys", "dim"),
            ("    escape enter      Multi-key sequence (Escape then Enter)", "dim"),
            ("", ""),
            ("CONFIGURATION FILES", "bold"),
            ("    .jaato/keybindings.json           Base config (project)", "dim"),
            ("    .jaato/keybindings.<term>.json    Terminal-specific config", "dim"),
            ("    ~/.jaato/keybindings.json         User-level config", "dim"),
            ("", ""),
            ("ENVIRONMENT", "bold"),
            ("    JAATO_KEYBINDING_PROFILE=<name>   Override profile selection", "dim"),
        ])
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
        ("    keybindings help              - Show detailed help", "dim"),
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
        "Scrolling": ["scroll_up", "scroll_down", "scroll_top", "scroll_bottom", "mouse_scroll_up", "mouse_scroll_down"],
        "Navigation": ["nav_up", "nav_down"],
        "Pager": ["pager_quit", "pager_next"],
        "Features": ["toggle_plan", "toggle_tools", "cycle_agents", "yank", "view_full"],
        "Tool Navigation": ["tool_nav_enter", "tool_expand", "tool_collapse", "tool_exit", "tool_output_up", "tool_output_down"],
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
