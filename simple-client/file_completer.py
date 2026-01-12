"""File path and command completer for interactive client.

Provides intelligent completion for:
- Commands (help, tools, reset, etc.) when typing at line start
- File/folder paths when user types @path patterns
- Slash commands when user types /command patterns (from .jaato/commands/)

Integrates with prompt_toolkit for rich interactive completion.
"""

import os
from pathlib import Path
from typing import Iterable, Optional, Callable

from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document


# Default commands available in the interactive client
# Note: Session commands (save, resume, sessions) are contributed by the session plugin
DEFAULT_COMMANDS = [
    ("help", "Show help message and available commands"),
    ("tools", "Manage tools available to the model"),
    ("tools list", "List all tools with enabled/disabled status"),
    ("tools enable", "Enable a tool (usage: tools enable <name> or 'all')"),
    ("tools disable", "Disable a tool (usage: tools disable <name> or 'all')"),
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
    ("theme", "Show current theme information"),
    ("theme reload", "Reload theme from config files"),
    ("theme dark", "Switch to dark theme preset"),
    ("theme light", "Switch to light theme preset"),
    ("theme high-contrast", "Switch to high-contrast theme preset"),
    ("plugins", "List available plugins with status"),
    ("reset", "Clear conversation history"),
    ("history", "Show full conversation history"),
    ("context", "Show context window usage"),
    ("export", "Export session to YAML for replay"),
    ("screenshot", "Capture TUI and send hint to model"),
    ("screenshot nosend", "Capture TUI only, no hint to model"),
    ("screenshot format", "Show/set output format"),
    ("screenshot format svg", "SVG format (default, no dependencies)"),
    ("screenshot format png", "PNG format (requires cairosvg)"),
    ("screenshot format html", "HTML format"),
    ("screenshot auto", "Toggle auto-capture on turn end"),
    ("screenshot interval", "Set periodic capture interval (ms) during streaming"),
    ("screenshot help", "Show screenshot command help"),
    ("plan", "Show current plan status"),
    ("quit", "Exit the client"),
    ("exit", "Exit the client"),
]


class CommandCompleter(Completer):
    """Complete commands at the start of input.

    Provides completion for built-in commands like help, tools, reset, etc.
    Only triggers when input appears to be a command (no @ or multi-word input).

    Supports dynamic registration of additional commands from plugins.

    Example usage:
        "he" -> completes to "help"
        "to" -> completes to "tools"
    """

    def __init__(self, commands: Optional[list[tuple[str, str]]] = None):
        """Initialize the command completer.

        Args:
            commands: List of (command_name, description) tuples.
                     Defaults to DEFAULT_COMMANDS if not provided.
        """
        self._builtin_commands = list(commands or DEFAULT_COMMANDS)
        self._plugin_commands: list[tuple[str, str]] = []

    @property
    def commands(self) -> list[tuple[str, str]]:
        """Get all commands (builtin + plugin-contributed)."""
        return self._builtin_commands + self._plugin_commands

    def add_commands(self, commands: list[tuple[str, str]]) -> None:
        """Add commands dynamically (e.g., from plugins).

        Args:
            commands: List of (command_name, description) tuples to add.
        """
        # Avoid duplicates by checking existing names
        existing_names = {cmd[0] for cmd in self.commands}
        for cmd in commands:
            if cmd[0] not in existing_names:
                self._plugin_commands.append(cmd)
                existing_names.add(cmd[0])

    def clear_plugin_commands(self) -> None:
        """Clear all plugin-contributed commands."""
        self._plugin_commands.clear()

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get command completions for the current document.

        Handles progressive completion of multi-word commands:
        - Typing "se" -> shows "session" (base command)
        - Typing "session " -> shows "list", "new", "attach" (subcommands)
        - Typing "session l" -> shows "list"
        """
        raw_text = document.text_before_cursor
        text = raw_text.strip()

        # Skip if contains @ (file reference) or starts with / (slash command)
        if '@' in text or text.startswith('/'):
            return

        # Get the full text being typed (lowercase for matching)
        full_text = text.lower()

        # Check if user has trailing space (indicating they finished a word)
        has_trailing_space = raw_text.endswith(' ') and text

        # Parse what user has typed so far
        parts = full_text.split()

        # Build set of base commands and their subcommands
        base_commands = {}  # base -> {subcommand -> description}
        single_commands = {}  # command -> description

        for cmd_name, cmd_desc in self.commands:
            cmd_parts = cmd_name.split()
            if len(cmd_parts) == 1:
                single_commands[cmd_name.lower()] = (cmd_name, cmd_desc)
            else:
                base = cmd_parts[0].lower()
                rest = ' '.join(cmd_parts[1:])
                if base not in base_commands:
                    base_commands[base] = {}
                base_commands[base][rest.lower()] = (rest, cmd_desc)

        if not parts:
            # Empty input - show all base commands and single commands
            seen = set()
            for base in base_commands:
                if base not in seen:
                    seen.add(base)
                    # Get description from first subcommand or generic
                    desc = f"{base} [subcommand]"
                    yield Completion(base, start_position=0, display=base, display_meta=desc)
            for cmd_lower, (cmd_name, cmd_desc) in single_commands.items():
                if cmd_lower not in seen:
                    seen.add(cmd_lower)
                    yield Completion(cmd_name, start_position=0, display=cmd_name, display_meta=cmd_desc)
            return

        first_word = parts[0]

        if len(parts) == 1 and not has_trailing_space:
            # User is typing the first word - complete base commands and single commands
            seen = set()
            for base in base_commands:
                if base.startswith(first_word) and base not in seen:
                    seen.add(base)
                    desc = f"{base} [subcommand]"
                    yield Completion(base, start_position=-len(text), display=base, display_meta=desc)
            for cmd_lower, (cmd_name, cmd_desc) in single_commands.items():
                if cmd_lower.startswith(first_word) and cmd_lower not in seen:
                    seen.add(cmd_lower)
                    yield Completion(cmd_name, start_position=-len(text), display=cmd_name, display_meta=cmd_desc)
            return

        # User has typed at least one word and space - check for subcommands
        if first_word in base_commands:
            subcommands = base_commands[first_word]

            if len(parts) == 1 and has_trailing_space:
                # User typed base command + space - show first word of each subcommand (deduplicated)
                seen_first_words = set()
                for sub_lower, (sub_name, sub_desc) in subcommands.items():
                    first_sub_word = sub_lower.split()[0]
                    if first_sub_word not in seen_first_words:
                        seen_first_words.add(first_sub_word)
                        # Get description from the single-word entry if it exists
                        if first_sub_word in subcommands:
                            _, desc = subcommands[first_sub_word]
                        else:
                            desc = sub_desc
                        display_word = sub_name.split()[0]
                        yield Completion(display_word, start_position=0, display=display_word, display_meta=desc)

            elif len(parts) == 2 and not has_trailing_space:
                # User is typing a subcommand - filter by first word only (deduplicated)
                partial_sub = parts[1]
                seen_first_words = set()
                for sub_lower, (sub_name, sub_desc) in subcommands.items():
                    first_sub_word = sub_lower.split()[0]
                    if first_sub_word.startswith(partial_sub) and first_sub_word not in seen_first_words:
                        seen_first_words.add(first_sub_word)
                        # Get description from the single-word entry if it exists
                        if first_sub_word in subcommands:
                            _, desc = subcommands[first_sub_word]
                        else:
                            desc = sub_desc
                        display_word = sub_name.split()[0]
                        start_pos = -len(partial_sub)
                        yield Completion(display_word, start_position=start_pos, display=display_word, display_meta=desc)

            elif len(parts) == 2 and has_trailing_space:
                # User typed "base sub " - show 3rd level words if available
                second_word = parts[1]
                for sub_lower, (sub_name, sub_desc) in subcommands.items():
                    sub_parts = sub_lower.split()
                    if len(sub_parts) >= 2 and sub_parts[0] == second_word:
                        # This is a 3rd level entry - show the remaining words
                        third_part = ' '.join(sub_name.split()[1:])
                        yield Completion(third_part, start_position=0, display=third_part, display_meta=sub_desc)

            elif len(parts) >= 3 and not has_trailing_space:
                # User is typing 3rd word - filter matches
                second_word = parts[1]
                partial_third = ' '.join(parts[2:])
                for sub_lower, (sub_name, sub_desc) in subcommands.items():
                    sub_parts = sub_lower.split()
                    if len(sub_parts) >= 2 and sub_parts[0] == second_word:
                        third_part_lower = ' '.join(sub_parts[1:])
                        if third_part_lower.startswith(partial_third):
                            third_part = ' '.join(sub_name.split()[1:])
                            start_pos = -len(partial_third)
                            yield Completion(third_part, start_position=start_pos, display=third_part, display_meta=sub_desc)


class AtFileCompleter(Completer):
    """Complete file and folder paths after @ symbol.

    Triggers completion when user types @, providing:
    - File and folder suggestions from the filesystem
    - Visual dropdown with arrow key navigation
    - Directory indicators (trailing /)
    - Support for relative and absolute paths
    - Home directory expansion (~)

    Example usage:
        "Please review @src/utils.py and @tests/"
        "Load config from @~/projects/config.json"
    """

    def __init__(
        self,
        only_directories: bool = False,
        expanduser: bool = True,
        base_path: Optional[str] = None,
        file_filter: Optional[callable] = None,
    ):
        """Initialize the completer.

        Args:
            only_directories: If True, only suggest directories
            expanduser: If True, expand ~ to home directory
            base_path: Base path for relative completions (default: cwd)
            file_filter: Optional callable(filename) -> bool to filter files
        """
        self.only_directories = only_directories
        self.expanduser = expanduser
        self.base_path = base_path or os.getcwd()
        self.file_filter = file_filter

        # Internal path completer for the heavy lifting
        self._path_completer = PathCompleter(
            only_directories=only_directories,
            expanduser=expanduser,
            file_filter=file_filter,
        )

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get completions for the current document.

        Looks for @ patterns and provides file path completions.
        """
        text = document.text_before_cursor

        # Find the last @ symbol that starts a file reference
        at_pos = self._find_at_position(text)
        if at_pos == -1:
            return

        # Extract the path portion after @
        path_text = text[at_pos + 1:]

        # Skip if there's a space after @ (not a file reference)
        if path_text and path_text[0] == ' ':
            return

        # Create a sub-document for path completion
        path_doc = Document(text=path_text, cursor_position=len(path_text))

        # Get completions from PathCompleter
        for completion in self._path_completer.get_completions(path_doc, complete_event):
            text = completion.text

            # Add metadata for directories and append / to directory completions
            display_meta = completion.display_meta
            full_path = self._resolve_path(path_text, completion.text)
            is_dir = full_path and os.path.isdir(full_path)

            if is_dir:
                # Append / to directory completions for easier navigation
                if not text.endswith('/'):
                    text = text + '/'
                if not display_meta:
                    display_meta = "directory"
            elif not display_meta:
                if full_path and os.path.isfile(full_path):
                    display_meta = self._get_file_type(full_path)

            yield Completion(
                text,
                start_position=completion.start_position,
                display=text,  # Use text as display (includes / for dirs)
                display_meta=display_meta,
            )

    def _find_at_position(self, text: str) -> int:
        """Find the position of @ that starts a file reference.

        Returns -1 if no valid @ reference is found.
        A valid @ is one that:
        - Is at start of string, or preceded by whitespace/punctuation
        - Is not part of an email address pattern
        """
        # Find the last @ in the text
        at_pos = text.rfind('@')
        if at_pos == -1:
            return -1

        # Check if this @ looks like a file reference
        # Valid: "@file", " @file", "(@file", '"@file'
        # Invalid: "user@email" (alphanumeric before @)
        if at_pos > 0:
            prev_char = text[at_pos - 1]
            # If preceded by alphanumeric, dot, underscore, or hyphen -> likely email
            if prev_char.isalnum() or prev_char in '._-':
                return -1

        return at_pos

    def _resolve_path(self, base: str, completion: str) -> Optional[str]:
        """Resolve the full path for a completion."""
        try:
            # Combine base path fragment with completion
            if base:
                dir_part = os.path.dirname(base)
                full_path = os.path.join(dir_part, completion) if dir_part else completion
            else:
                full_path = completion

            # Expand user
            if self.expanduser and full_path.startswith('~'):
                full_path = os.path.expanduser(full_path)

            # Make absolute if relative
            if not os.path.isabs(full_path):
                full_path = os.path.join(self.base_path, full_path)

            return full_path
        except Exception:
            return None

    def _get_file_type(self, path: str) -> str:
        """Get a short description of the file type."""
        ext = os.path.splitext(path)[1].lower()

        type_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text',
            '.sh': 'shell',
            '.bash': 'shell',
            '.env': 'env',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.xml': 'xml',
            '.toml': 'toml',
            '.ini': 'config',
            '.cfg': 'config',
            '.cbl': 'cobol',
            '.cob': 'cobol',
        }

        return type_map.get(ext, 'file')


class SlashCommandCompleter(Completer):
    """Complete slash commands from .jaato/commands/ directory.

    Triggers completion when user types /, providing:
    - Command file suggestions from .jaato/commands/
    - Visual dropdown with arrow key navigation
    - Command descriptions from file contents (first line)

    Example usage:
        "/sum" -> completes to "/summarize" (if summarize file exists)
        "/" -> shows all available commands
    """

    # Default commands directory
    DEFAULT_COMMANDS_DIR = ".jaato/commands"

    def __init__(
        self,
        commands_dir: Optional[str] = None,
        base_path: Optional[str] = None,
    ):
        """Initialize the completer.

        Args:
            commands_dir: Path to commands directory (default: .jaato/commands)
            base_path: Base path for resolving relative commands_dir (default: cwd)
        """
        self.commands_dir = commands_dir or self.DEFAULT_COMMANDS_DIR
        self.base_path = base_path or os.getcwd()
        self._command_descriptions: dict[str, str] = {}

    def _get_commands_path(self) -> Path:
        """Get the absolute path to the commands directory."""
        commands_path = Path(self.commands_dir)
        if not commands_path.is_absolute():
            commands_path = Path(self.base_path) / commands_path
        return commands_path

    def _load_command_descriptions(self) -> dict[str, str]:
        """Load command descriptions from first line of each command file."""
        commands_path = self._get_commands_path()
        if not commands_path.exists():
            return {}

        descriptions = {}
        for item in commands_path.iterdir():
            if item.is_file():
                try:
                    # Read first line as description
                    with open(item, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        # Remove common comment prefixes for cleaner display
                        for prefix in ['#', '//', '/*', '--', '"""', "'''"]:
                            if first_line.startswith(prefix):
                                first_line = first_line[len(prefix):].strip()
                                break
                        # Truncate if too long
                        if len(first_line) > 60:
                            first_line = first_line[:57] + "..."
                        descriptions[item.name] = first_line or "command"
                except Exception:
                    descriptions[item.name] = "command"
        return descriptions

    def list_commands(self) -> list[str]:
        """List available command names."""
        commands_path = self._get_commands_path()
        if not commands_path.exists():
            return []
        return sorted([item.name for item in commands_path.iterdir() if item.is_file()])

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get completions for slash commands.

        Looks for / patterns and provides command completions.
        """
        text = document.text_before_cursor

        # Find the last / that starts a slash command
        slash_pos = self._find_slash_position(text)
        if slash_pos == -1:
            return

        # Extract the command portion after /
        command_text = text[slash_pos + 1:]

        # Skip if there's a space after / (not a slash command)
        if command_text and command_text[0] == ' ':
            return

        # Load command descriptions (cached for this completion session)
        descriptions = self._load_command_descriptions()
        commands = list(descriptions.keys())

        # Filter commands that match the typed text
        command_lower = command_text.lower()
        for cmd_name in commands:
            if cmd_name.lower().startswith(command_lower):
                description = descriptions.get(cmd_name, "command")
                # Complete with the command name (without /)
                yield Completion(
                    cmd_name,
                    start_position=-len(command_text),
                    display=f"/{cmd_name}",
                    display_meta=description,
                )

    def _find_slash_position(self, text: str) -> int:
        """Find the position of / that starts a slash command.

        Returns -1 if no valid / command is found.
        A valid / is one that:
        - Is at start of string, or preceded by whitespace
        - Is not part of a path (like /home/user)
        """
        # Find the last / in the text
        slash_pos = text.rfind('/')
        if slash_pos == -1:
            return -1

        # Check if this / looks like a slash command start
        # Valid: "/cmd", " /cmd"
        # Invalid: "/home/user" (path), "a/b" (middle of word)

        # Must be at start or preceded by whitespace
        if slash_pos > 0:
            prev_char = text[slash_pos - 1]
            if not prev_char.isspace():
                return -1

        # Check if it's followed by more slashes (likely a path)
        remaining = text[slash_pos + 1:]
        if '/' in remaining:
            return -1

        return slash_pos


class FileReferenceProcessor:
    """Process @file references in user input.

    Extracts file paths from @references and optionally loads their contents
    to include in the prompt sent to the model.
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        max_file_size: int = 100_000,  # 100KB default
        include_contents: bool = True,
    ):
        """Initialize the processor.

        Args:
            base_path: Base path for resolving relative references
            max_file_size: Maximum file size to include (bytes)
            include_contents: Whether to include file contents in output
        """
        self.base_path = base_path or os.getcwd()
        self.max_file_size = max_file_size
        self.include_contents = include_contents

    def process(self, text: str) -> tuple[str, list[dict]]:
        """Process text containing @file references.

        Args:
            text: User input potentially containing @path references

        Returns:
            Tuple of (processed_text, file_references)
            - processed_text: Original text with @paths intact
            - file_references: List of dicts with file info and contents
        """
        import re

        # Pattern to match @path references
        # Matches @ followed by a path (letters, numbers, /, ., _, -, ~)
        # Stops at whitespace or end of string
        pattern = r'@([~/\w.\-]+(?:/[~/\w.\-]*)*)'

        references = []

        for match in re.finditer(pattern, text):
            path = match.group(1)
            full_path = self._resolve_path(path)

            if full_path and os.path.exists(full_path):
                ref_info = {
                    'reference': match.group(0),  # @path/to/file
                    'path': path,                  # path/to/file
                    'full_path': full_path,        # /absolute/path/to/file
                    'exists': True,
                    'is_directory': os.path.isdir(full_path),
                }

                if os.path.isfile(full_path) and self.include_contents:
                    ref_info['contents'] = self._read_file(full_path)
                    ref_info['size'] = os.path.getsize(full_path)
                elif os.path.isdir(full_path):
                    ref_info['listing'] = self._list_directory(full_path)

                references.append(ref_info)
            else:
                references.append({
                    'reference': match.group(0),
                    'path': path,
                    'full_path': full_path,
                    'exists': False,
                })

        return text, references

    def expand_references(self, text: str) -> str:
        """Expand @file references to include file contents inline.

        Returns a new prompt with file contents appended in a structured format.
        The @ prefixes are removed from the prompt text since they were only
        used for autocompletion and file resolution.

        Non-existent paths have their @ stripped but are not included in the
        Referenced Files section - the user may be referring to paths they
        want to create.
        """
        processed_text, references = self.process(text)

        if not references:
            return text

        # Remove @ prefixes from the original text
        # Replace each @path with just path (without the @)
        clean_text = text
        for ref in references:
            clean_text = clean_text.replace(ref['reference'], ref['path'])

        # Filter to only existing files/directories for the context section
        existing_refs = [ref for ref in references if ref.get('exists', False)]

        # If no existing references, just return the clean text without @ symbols
        if not existing_refs:
            return clean_text

        # Build expanded prompt with existing file contents
        parts = [clean_text, "\n\n--- Referenced Files ---\n"]

        for ref in existing_refs:
            # Use path (without @) in headers
            if ref['is_directory']:
                parts.append(f"\n[{ref['path']}: Directory]\n")
                if 'listing' in ref:
                    parts.append("Contents:\n")
                    for item in ref['listing'][:50]:  # Limit directory listing
                        parts.append(f"  {item}\n")
                    if len(ref.get('listing', [])) > 50:
                        parts.append(f"  ... and {len(ref['listing']) - 50} more items\n")
            else:
                parts.append(f"\n[{ref['path']}]\n")
                if 'contents' in ref and ref['contents']:
                    parts.append(f"```\n{ref['contents']}\n```\n")
                elif ref.get('size', 0) > self.max_file_size:
                    parts.append(f"[File too large: {ref['size']} bytes]\n")

        return ''.join(parts)

    def _resolve_path(self, path: str) -> Optional[str]:
        """Resolve a path reference to absolute path."""
        try:
            # Expand ~
            if path.startswith('~'):
                path = os.path.expanduser(path)

            # Make absolute if relative
            if not os.path.isabs(path):
                path = os.path.join(self.base_path, path)

            # Normalize
            return os.path.normpath(path)
        except Exception:
            return None

    def _read_file(self, path: str) -> Optional[str]:
        """Read file contents if within size limit."""
        try:
            size = os.path.getsize(path)
            if size > self.max_file_size:
                return None

            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception:
            return None

    def _list_directory(self, path: str) -> list[str]:
        """List directory contents."""
        try:
            items = []
            for item in sorted(os.listdir(path)):
                full_item = os.path.join(path, item)
                suffix = '/' if os.path.isdir(full_item) else ''
                items.append(f"{item}{suffix}")
            return items
        except Exception:
            return []


class PermissionResponseCompleter(Completer):
    """Complete permission response options.

    Provides completion for valid permission responses when a tool is
    awaiting permission approval. This completer is designed to temporarily
    replace normal completions while a permission prompt is active.

    The valid response options are provided dynamically from the permission
    plugin, making the plugin the single source of truth for valid responses.

    Example usage:
        "y" -> completes to "yes" (or accepts "y")
        "a" -> completes to "always" (or accepts "a")
    """

    # Default fallback options in case none are provided
    # Format: (short, full, description)
    DEFAULT_OPTIONS = [
        ("y", "yes", "Allow this tool execution"),
        ("n", "no", "Deny this tool execution"),
        ("a", "always", "Allow and whitelist for session"),
        ("once", "once", "Allow once without remembering"),
        ("never", "never", "Deny and blacklist for session"),
    ]

    def __init__(self):
        """Initialize the permission response completer."""
        # Current options - can be set dynamically from permission plugin
        self._options: Optional[list] = None

    def set_options(self, options: Optional[list]) -> None:
        """Set the valid response options from the permission plugin.

        Args:
            options: List of PermissionResponseOption objects from the plugin,
                    or None to use default fallback options.
                    Each option should have: short, full, description attributes.
        """
        self._options = options

    def clear_options(self) -> None:
        """Clear the current options, reverting to defaults."""
        self._options = None

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get completions for permission responses.

        Uses options from the permission plugin if available,
        otherwise falls back to default options.
        """
        text = document.text_before_cursor.strip().lower()

        # Use plugin-provided options if available, otherwise use defaults
        if self._options:
            for opt in self._options:
                # Handle both dict and object options
                if isinstance(opt, dict):
                    short = opt.get('key', opt.get('short', ''))
                    full = opt.get('label', opt.get('full', ''))
                    description = opt.get('description', '')
                else:
                    short = getattr(opt, 'short', getattr(opt, 'key', ''))
                    full = getattr(opt, 'full', getattr(opt, 'label', ''))
                    description = getattr(opt, 'description', '')
                # Match against both short and full forms
                if not text or short.lower().startswith(text) or full.lower().startswith(text):
                    yield Completion(
                        full,
                        start_position=-len(text) if text else 0,
                        display=f"{short}/{full}",
                        display_meta=description,
                    )
        else:
            # Fallback to defaults
            for short, full, description in self.DEFAULT_OPTIONS:
                if not text or short.startswith(text) or full.startswith(text):
                    yield Completion(
                        full,
                        start_position=-len(text) if text else 0,
                        display=f"{short}/{full}",
                        display_meta=description,
                    )


class SessionIdCompleter(Completer):
    """Complete session IDs for session commands.

    Triggers completion when user types session commands followed by a space:
    - "session attach " -> completes with available session IDs
    - "session delete " -> completes with available session IDs
    - "delete-session " -> completes with available session IDs (legacy)
    - "resume " -> completes with available session IDs

    Supports both full session IDs and numeric indexes.
    """

    # Commands that accept session ID arguments (single-word and multi-word)
    SESSION_COMMANDS = ['delete-session', 'resume', 'session attach', 'session delete']

    def __init__(self, session_provider: Optional[Callable[[], list]] = None):
        """Initialize the session ID completer.

        Args:
            session_provider: Callback that returns list of SessionInfo objects
                            with session_id and description attributes.
        """
        self._session_provider = session_provider

    def set_session_provider(self, provider: Callable[[], list]) -> None:
        """Set the session provider callback.

        Args:
            provider: Callback that returns list of session info objects.
        """
        self._session_provider = provider

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get session ID completions for session commands."""
        # Don't strip - we need to detect the space after the command
        text = document.text_before_cursor

        # Check if input starts with a session command followed by space
        command_match = None
        text_lower = text.lower()
        for cmd in self.SESSION_COMMANDS:
            if text_lower.startswith(cmd + ' '):
                command_match = cmd
                break

        if not command_match:
            return

        # Extract the argument portion after the command (may have leading space)
        arg_text = text[len(command_match) + 1:]  # +1 for the space

        # Get available sessions
        if not self._session_provider:
            return

        try:
            sessions = self._session_provider()
        except Exception:
            return

        if not sessions:
            return

        # Provide completions - show session IDs directly
        for session in sessions:
            session_id = getattr(session, 'session_id', str(session))
            description = getattr(session, 'description', None) or '(unnamed)'

            if not arg_text:
                # No input yet - show all session IDs
                yield Completion(
                    session_id,
                    start_position=0,
                    display=session_id,
                    display_meta=description,
                )
            elif session_id.startswith(arg_text):
                # Filter by prefix
                yield Completion(
                    session_id,
                    start_position=-len(arg_text),
                    display=session_id,
                    display_meta=description,
                )


class PluginCommandCompleter(Completer):
    """Complete user command arguments by querying plugins.

    Dynamically fetches completion options from plugins that implement
    the optional get_command_completions() method. This decouples the
    client from plugin-specific completion logic.

    Example usage:
        "permissions de" -> queries permission plugin for completions
        "permissions default a" -> queries permission plugin for policy options
    """

    def __init__(self, completion_provider: Optional[Callable[[str, list], list]] = None):
        """Initialize the plugin command completer.

        Args:
            completion_provider: Callback that takes (command, args) and returns
                                list of (value, description) tuples from plugins.
        """
        self._completion_provider = completion_provider
        self._command_names: set[str] = set()

    def set_completion_provider(
        self, provider: Callable[[str, list], list]
    ) -> None:
        """Set the completion provider callback.

        Args:
            provider: Callback (command, args) -> [(value, description), ...]
        """
        self._completion_provider = provider

    def set_command_names(self, names: set[str]) -> None:
        """Set the set of command names that have completions.

        Args:
            names: Set of command names (e.g., {"permissions", "sessions"})
        """
        self._command_names = names

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get completions for plugin commands."""
        if not self._completion_provider:
            return

        text = document.text_before_cursor
        text_lower = text.lower()

        # Check if input starts with a known command followed by space
        command_match = None
        for cmd in self._command_names:
            if text_lower.startswith(cmd + ' '):
                command_match = cmd
                break

        if not command_match:
            return

        # Extract arguments after the command
        arg_text = text[len(command_match) + 1:]  # +1 for space
        args = arg_text.split() if arg_text.strip() else []

        # Determine partial text and args to query
        # - Trailing space means we want completions for NEXT argument
        # - No trailing space means we're completing the CURRENT argument
        if arg_text.endswith(' '):
            # Finished current arg, want next arg completions
            # Append empty string to signal "give me options for next position"
            partial = ""
            args_for_query = args + [""]
        elif args:
            # Currently typing an argument
            partial = args[-1]
            args_for_query = args
        else:
            # No args yet
            partial = ""
            args_for_query = []

        # Query plugin for completions
        try:
            completions = self._completion_provider(command_match, args_for_query)
        except Exception:
            return

        # Yield matching completions
        for item in completions:
            # Handle both tuple and object with value/description
            if hasattr(item, 'value'):
                value, description = item.value, item.description
            else:
                value, description = item[0], item[1] if len(item) > 1 else ""

            yield Completion(
                value,
                start_position=-len(partial),
                display=value,
                display_meta=description,
            )


class CombinedCompleter(Completer):
    """Combined completer for commands, file references, slash commands, and plugin commands.

    Merges CommandCompleter, AtFileCompleter, SlashCommandCompleter,
    SessionIdCompleter, and PluginCommandCompleter to provide:
    - Command completion at line start (help, tools, reset, etc.)
    - File path completion after @ symbols
    - Slash command completion after / symbols (from .jaato/commands/)
    - Session ID completion after session commands (delete-session, resume)
    - Plugin command argument completion (via get_command_completions protocol)

    Additionally supports a "permission mode" that temporarily replaces all
    completions with permission response options when a permission prompt is active.

    This allows seamless autocompletion for all use cases.
    """

    def __init__(
        self,
        commands: Optional[list[tuple[str, str]]] = None,
        only_directories: bool = False,
        expanduser: bool = True,
        base_path: Optional[str] = None,
        file_filter: Optional[callable] = None,
        commands_dir: Optional[str] = None,
        session_provider: Optional[Callable[[], list]] = None,
    ):
        """Initialize the combined completer.

        Args:
            commands: List of (command_name, description) tuples for command completion.
            only_directories: If True, only suggest directories for file completion.
            expanduser: If True, expand ~ to home directory.
            base_path: Base path for relative file completions (default: cwd).
            file_filter: Optional callable(filename) -> bool to filter files.
            commands_dir: Path to slash commands directory (default: .jaato/commands).
            session_provider: Callback that returns list of session info objects.
        """
        self._command_completer = CommandCompleter(commands)
        self._file_completer = AtFileCompleter(
            only_directories=only_directories,
            expanduser=expanduser,
            base_path=base_path,
            file_filter=file_filter,
        )
        self._slash_completer = SlashCommandCompleter(
            commands_dir=commands_dir,
            base_path=base_path,
        )
        self._session_completer = SessionIdCompleter(session_provider)
        self._plugin_command_completer = PluginCommandCompleter()
        self._permission_completer = PermissionResponseCompleter()
        self._permission_mode: bool = False

    def set_command_completion_provider(
        self,
        provider: Callable[[str, list], list],
        command_names: set[str]
    ) -> None:
        """Set the plugin command completion provider.

        Args:
            provider: Callback (command, args) -> [(value, description), ...]
            command_names: Set of command names that support completion.
        """
        self._plugin_command_completer.set_completion_provider(provider)
        self._plugin_command_completer.set_command_names(command_names)

    def set_session_provider(self, provider: Callable[[], list]) -> None:
        """Set the session provider callback for session ID completion.

        Args:
            provider: Callback that returns list of session info objects.
        """
        self._session_completer.set_session_provider(provider)

    def add_commands(self, commands: list[tuple[str, str]]) -> None:
        """Add commands dynamically (e.g., from plugins).

        Args:
            commands: List of (command_name, description) tuples to add.
        """
        self._command_completer.add_commands(commands)

    def clear_plugin_commands(self) -> None:
        """Clear all plugin-contributed commands."""
        self._command_completer.clear_plugin_commands()

    def set_permission_mode(self, enabled: bool, options: Optional[list] = None) -> None:
        """Enable or disable permission response completion mode.

        When permission mode is enabled, only permission response options
        are shown in completions. All normal completions (commands, files,
        etc.) are temporarily disabled.

        The valid options are provided by the permission plugin, making it
        the single source of truth for what responses are valid.

        Use this when a permission prompt is active and the user should
        only be able to enter valid permission responses.

        Args:
            enabled: True to enable permission-only completions,
                    False to restore normal completion behavior.
            options: List of PermissionResponseOption objects from the
                    permission plugin. Only used when enabled=True.
                    Each option should have: short, full, description attributes.
        """
        self._permission_mode = enabled
        if enabled and options is not None:
            self._permission_completer.set_options(options)
        elif not enabled:
            self._permission_completer.clear_options()

    @property
    def permission_mode(self) -> bool:
        """Check if permission completion mode is currently active.

        Returns:
            True if permission mode is enabled, False otherwise.
        """
        return self._permission_mode

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get completions from all completers.

        If permission mode is enabled, only yields permission response
        completions. Otherwise yields from all normal completion sources.
        """
        # If permission mode is active, only show permission response options
        if self._permission_mode:
            yield from self._permission_completer.get_completions(document, complete_event)
            return

        # Normal mode: yield completions from all sources
        # CommandCompleter will only yield if appropriate (single word, no @ or /)
        # AtFileCompleter will only yield if @ is present
        # SlashCommandCompleter will only yield if / is present at start of word
        # SessionIdCompleter will only yield after session commands (delete-session, resume)
        # PluginCommandCompleter will yield for plugin commands with completions
        yield from self._command_completer.get_completions(document, complete_event)
        yield from self._file_completer.get_completions(document, complete_event)
        yield from self._slash_completer.get_completions(document, complete_event)
        yield from self._session_completer.get_completions(document, complete_event)
        yield from self._plugin_command_completer.get_completions(document, complete_event)
