"""TUI-local UI formatting utilities.

This module provides UI formatting functions used by the rich client.
Path-related helpers from shared.path_utils are inlined to avoid
depending on the full server package.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# Inlined path_utils helpers (stdlib-only, no external deps)
# =============================================================================

def _is_msys2_environment() -> bool:
    """Detect if running under MSYS2 or Git Bash on Windows."""
    if sys.platform != 'win32':
        return False
    msystem = os.environ.get('MSYSTEM', '')
    if msystem in ('MINGW64', 'MINGW32', 'MSYS', 'UCRT64', 'CLANG64', 'CLANGARM64'):
        return True
    if os.environ.get('TERM_PROGRAM') == 'mintty':
        return True
    return False


def _windows_to_msys2_path(path: str) -> str:
    """Convert C:/foo -> /c/foo for display in MSYS2 shell."""
    import re
    if not path:
        return path
    path = path.replace('\\', '/')
    m = re.match(r'^([a-zA-Z]):[\\/]', path)
    if m:
        drive = m.group(1).lower()
        rest = path[2:]
        return f"/{drive}{rest}"
    return path


def _normalize_path(path: str) -> str:
    """Normalize a Windows path for display under MSYS2."""
    if not path:
        return path
    if _is_msys2_environment():
        return _windows_to_msys2_path(path)
    return path


def _get_display_separator() -> str:
    """Get the path separator to use for display purposes."""
    if _is_msys2_environment():
        return '/'
    return os.sep


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

    Args:
        path: The file path to ellipsize.
        max_width: Maximum width in characters. Must be positive.
        keep_first: Number of leading path segments to preserve (default: 1).
        keep_last: Number of trailing path segments to preserve (default: 2).
        ellipsis: The ellipsis string to use (default: "...").

    Returns:
        The ellipsized path if it exceeds max_width, otherwise the original path.
    """
    if not path or max_width <= 0:
        return path

    path = _normalize_path(path)

    if len(path) <= max_width:
        return path

    sep = _get_display_separator()
    if "/" in path and "\\" not in path:
        sep = "/"
    elif "\\" in path:
        sep = "\\"

    has_leading_sep = path.startswith(sep)
    segments = [s for s in path.split(sep) if s]

    if not segments:
        return path

    total_segments = len(segments)
    if total_segments <= keep_first + keep_last:
        filename = segments[-1]
        if len(filename) <= max_width:
            return path if len(path) <= max_width else ellipsis + filename
        return ellipsis + filename[-(max_width - len(ellipsis)):]

    first_segments = segments[:keep_first]
    last_segments = segments[-keep_last:]

    first_part = sep.join(first_segments)
    last_part = sep.join(last_segments)

    if has_leading_sep:
        first_part = sep + first_part

    candidate = f"{first_part}{sep}{ellipsis}{sep}{last_part}"

    if len(candidate) <= max_width:
        return candidate

    for reduced_first in range(keep_first - 1, -1, -1):
        if reduced_first > 0:
            first_segments = segments[:reduced_first]
            first_part = sep.join(first_segments)
            if has_leading_sep:
                first_part = sep + first_part
            candidate = f"{first_part}{sep}{ellipsis}{sep}{last_part}"
        else:
            candidate = f"{ellipsis}{sep}{last_part}"

        if len(candidate) <= max_width:
            return candidate

    for reduced_last in range(keep_last - 1, 0, -1):
        last_segments = segments[-reduced_last:]
        last_part = sep.join(last_segments)
        candidate = f"{ellipsis}{sep}{last_part}"

        if len(candidate) <= max_width:
            return candidate

    filename = segments[-1]
    if len(ellipsis) + len(filename) <= max_width:
        return ellipsis + filename

    available = max_width - len(ellipsis)
    if available > 0:
        return ellipsis + filename[-available:]

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
    """Ellipsize a source->destination path pair to fit within max_width."""
    full = f"{source}{separator}{dest}"
    if len(full) <= max_width:
        return full

    available = max_width - len(separator)
    if available <= 0:
        return full[:max_width]

    per_path = available // 2

    source_ellipsized = ellipsize_path(
        source, per_path, keep_first=keep_first, keep_last=keep_last, ellipsis=ellipsis
    )
    dest_ellipsized = ellipsize_path(
        dest, per_path, keep_first=keep_first, keep_last=keep_last, ellipsis=ellipsis
    )

    return f"{source_ellipsized}{separator}{dest_ellipsized}"


# =============================================================================
# Name Ellipsization (middle-ellipsis for short labels)
# =============================================================================


def ellipsize_name(name: str, max_width: int, *, ellipsis: str = "…") -> str:
    """Ellipsize a name in the middle, preserving both start and end.

    Unlike ``ellipsize_path`` (which operates on path segments), this function
    works at the *character* level and is intended for short labels such as
    agent IDs or session names where both the prefix **and** the suffix carry
    meaning (e.g. ``"validator-tier3"`` → ``"valid…tier3"``).

    The visible characters are split roughly in half between the head and tail
    of the original string, with a slight bias toward the tail so that
    distinguishing suffixes are preserved.

    Args:
        name: The string to ellipsize.
        max_width: Maximum width in characters.  Must be at least
            ``len(ellipsis) + 2`` for any truncation to occur; if not, the
            name is hard-truncated to *max_width*.
        ellipsis: The ellipsis string to insert (default: ``"…"`` U+2026).

    Returns:
        The original *name* unchanged if it fits within *max_width*, otherwise
        a middle-ellipsized version.

    Examples:
        >>> ellipsize_name("validator-tier3-secondary", 15)
        'valida…econdary'
        >>> ellipsize_name("short", 15)
        'short'
    """
    if not name or max_width <= 0:
        return name

    if len(name) <= max_width:
        return name

    ell_len = len(ellipsis)

    # Need room for at least one char on each side of the ellipsis
    if max_width < ell_len + 2:
        return name[:max_width]

    available = max_width - ell_len
    # Bias toward the tail so distinguishing suffixes are preserved
    tail_len = (available + 1) // 2
    head_len = available - tail_len

    return name[:head_len] + ellipsis + name[-tail_len:]


# =============================================================================
# Permission & Tool Formatting
# =============================================================================


def format_permission_options(
    response_options: List[Union[Dict[str, Any], Any]],
    use_brackets: bool = True,
) -> str:
    """Format permission response options for display."""
    parts = []
    for opt in response_options:
        if isinstance(opt, dict):
            key = opt.get('key', opt.get('short', '?'))
            label = opt.get('label', opt.get('full', '?'))
        else:
            key = getattr(opt, 'short', getattr(opt, 'key', '?'))
            label = getattr(opt, 'full', getattr(opt, 'label', '?'))

        if use_brackets:
            if key != label and label.lower().startswith(key.lower()):
                parts.append(f"[{key}]{label[len(key):]}")
            else:
                parts.append(f"[{label}]")
        else:
            parts.append(label)

    return " ".join(parts)


_PATH_ARG_NAMES = frozenset({
    "file_path", "path", "filepath", "file", "directory", "dir",
    "target", "source", "dest", "destination", "folder", "filename",
    "target_path", "source_path", "dest_path", "working_directory",
    "cwd", "base_path", "root_path", "old_file_path", "new_file_path",
    "old_path", "new_path",
})


def _looks_like_path(value: str) -> bool:
    """Check if a string value looks like a file system path."""
    if not value or len(value) < 2:
        return False
    if value.startswith("/") and "/" in value[1:]:
        return True
    if value.startswith("./") or value.startswith("../"):
        return True
    if value.startswith("~/"):
        return True
    if (
        len(value) >= 3
        and value[0].isalpha()
        and value[1] == ":"
        and value[2] in ("\\", "/")
    ):
        return True
    return False


def format_tool_args_summary(
    tool_args: Dict[str, Any],
    max_length: int = 60,
    *,
    max_path_width: int = 40,
) -> str:
    """Format tool arguments as a summary string with path ellipsization."""
    if not tool_args:
        return ""

    processed: Dict[str, Any] = {}
    for key, value in tool_args.items():
        if isinstance(value, str) and (
            key.lower() in _PATH_ARG_NAMES or _looks_like_path(value)
        ):
            processed[key] = ellipsize_path(value, max_path_width)
        else:
            processed[key] = value

    args_str = str(processed)
    if len(args_str) > max_length:
        return args_str[: max_length - 3] + "..."
    return args_str


def format_tool_arg_value(value: Any, max_width: int) -> str:
    """Format a single tool argument value as a string, truncated to fit."""
    if max_width < 4:
        return "..."[:max_width]

    s = str(value)

    if "\n" in s:
        first_line = s.split("\n", 1)[0]
        s = first_line + "..."

    if isinstance(value, str) and _looks_like_path(value):
        s = ellipsize_path(s, max_width)

    if len(s) > max_width:
        return s[: max_width - 3] + "..."
    return s


def format_duration(seconds: Optional[float]) -> str:
    """Format a duration in seconds for display."""
    if seconds is None:
        return ""
    return f"{seconds:.1f}s"


def format_token_count(count: int) -> str:
    """Format a token count with K/M suffixes for large numbers."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def format_percent(value: float, decimals: int = 1) -> str:
    """Format a percentage value."""
    return f"{value:.{decimals}f}%"


def build_clarification_prompt_lines(
    question_text: str,
    question_index: Optional[int] = None,
    total_questions: Optional[int] = None,
) -> List[str]:
    """Build clarification prompt lines for display in tool tree."""
    lines = []

    if question_index is not None and total_questions is not None:
        lines.append(f"Question {question_index}/{total_questions}")
        lines.append("")

    if question_text:
        lines.extend(question_text.split("\n"))

    return lines


# =============================================================================
# Keybindings Command Handling
# =============================================================================

def handle_keybindings_command(user_input: str, display) -> None:
    """Handle the keybindings command with subcommands.

    Shared implementation used by both direct and IPC modes.

    Args:
        user_input: The full user input string starting with 'keybindings'.
        display: PTDisplay instance for output and keybinding config access.
    """
    if not display:
        return

    parts = user_input.strip().split()
    subcommand = parts[1].lower() if len(parts) > 1 else "list"

    if subcommand == "list" or subcommand == "keybindings":
        _show_keybindings(display)
        return

    if subcommand == "reload":
        _reload_keybindings(display)
        return

    if subcommand == "set":
        _set_keybinding(parts[2:], display)
        return

    if subcommand == "profile":
        _handle_profile_command(parts[2:], display)
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


def _show_keybindings(display) -> None:
    """Show current keybinding configuration."""
    if not display:
        return

    try:
        from keybindings import detect_terminal
    except ImportError:
        detect_terminal = lambda: "unknown"

    config = display._keybinding_config
    bindings = config.to_dict()

    detected = detect_terminal()
    lines = [
        ("Current Keybindings:", "bold"),
        (f"  Profile: {config.profile}", "cyan"),
        (f"  Terminal: {detected}", "dim"),
        (f"  Source: {config.profile_source}", "dim"),
        ("", ""),
    ]

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


def _reload_keybindings(display) -> None:
    """Reload keybindings from configuration files."""
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


def _set_keybinding(args: list, display) -> None:
    """Set a keybinding for an action."""
    if not display:
        return

    try:
        from keybindings import DEFAULT_KEYBINDINGS
    except ImportError:
        display.show_lines([("[Error: keybindings module not available]", "red")])
        return

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
        actions = list(DEFAULT_KEYBINDINGS.keys())
        for i in range(0, len(actions), 4):
            chunk = actions[i:i+4]
            display.show_lines([
                ("    " + ", ".join(chunk), "dim"),
            ])
        return

    action = args[0].lower()
    key = " ".join(args[1:])

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

    config = display._keybinding_config
    old_value = getattr(config, action)
    if isinstance(old_value, list):
        old_str = " ".join(old_value)
    else:
        old_str = old_value

    if not config.set_binding(action, key):
        display.show_lines([
            (f"[Error: Failed to set binding for '{action}']", "red"),
        ])
        return

    display._build_app()

    new_value = getattr(config, action)
    if isinstance(new_value, list):
        new_str = " ".join(new_value)
    else:
        new_str = new_value

    lines = [
        (f"[Keybinding updated: {action}]", "green"),
        (f"  {old_str} \u2192 {new_str}", "dim"),
    ]

    if save_to_file:
        if config.save_to_file():
            lines.append(("  Saved to .jaato/keybindings.json", "cyan"))
        else:
            lines.append(("  [Warning: Failed to save to file]", "yellow"))
    else:
        lines.append(("  (session only - use --save to persist)", "dim italic"))

    display.show_lines(lines)


def _handle_profile_command(args: list, display) -> None:
    """Handle the keybindings profile subcommand."""
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

    new_profile = args[0].lower()

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
