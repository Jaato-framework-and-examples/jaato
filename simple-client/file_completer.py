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
DEFAULT_COMMANDS = [
    ("help", "Show help message and available commands"),
    ("tools", "List available tools"),
    ("reset", "Clear conversation history"),
    ("history", "Show full conversation history"),
    ("context", "Show context window usage"),
    ("export", "Export session to YAML for replay"),
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
        """Get command completions for the current document."""
        text = document.text_before_cursor.strip()

        # Only complete if:
        # - Input is a single word (no spaces)
        # - Input doesn't contain @ (file reference)
        # - Input doesn't start with / (slash command)
        # - Input is at the start (no leading content)
        if ' ' in document.text_before_cursor or '@' in text or text.startswith('/'):
            return

        # Get the word being typed
        word = text.lower()

        for cmd_name, cmd_desc in self.commands:
            if cmd_name.startswith(word):
                # Calculate how much to complete
                yield Completion(
                    cmd_name,
                    start_position=-len(text),
                    display=cmd_name,
                    display_meta=cmd_desc,
                )


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
            # Calculate display text
            display = completion.display or completion.text

            # Add metadata for directories
            display_meta = completion.display_meta
            if not display_meta:
                full_path = self._resolve_path(path_text, completion.text)
                if full_path and os.path.isdir(full_path):
                    display_meta = "directory"
                elif full_path and os.path.isfile(full_path):
                    display_meta = self._get_file_type(full_path)

            yield Completion(
                completion.text,
                start_position=completion.start_position,
                display=display,
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
        """
        processed_text, references = self.process(text)

        if not references:
            return text

        # Remove @ prefixes from the original text
        # Replace each @path with just path (without the @)
        clean_text = text
        for ref in references:
            clean_text = clean_text.replace(ref['reference'], ref['path'])

        # Build expanded prompt
        parts = [clean_text, "\n\n--- Referenced Files ---\n"]

        for ref in references:
            # Use path (without @) in headers
            if not ref['exists']:
                parts.append(f"\n[{ref['path']}: File not found]\n")
            elif ref['is_directory']:
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


class CombinedCompleter(Completer):
    """Combined completer for commands, file references, and slash commands.

    Merges CommandCompleter, AtFileCompleter, and SlashCommandCompleter to provide:
    - Command completion at line start (help, tools, reset, etc.)
    - File path completion after @ symbols
    - Slash command completion after / symbols (from .jaato/commands/)

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
    ):
        """Initialize the combined completer.

        Args:
            commands: List of (command_name, description) tuples for command completion.
            only_directories: If True, only suggest directories for file completion.
            expanduser: If True, expand ~ to home directory.
            base_path: Base path for relative file completions (default: cwd).
            file_filter: Optional callable(filename) -> bool to filter files.
            commands_dir: Path to slash commands directory (default: .jaato/commands).
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

    def add_commands(self, commands: list[tuple[str, str]]) -> None:
        """Add commands dynamically (e.g., from plugins).

        Args:
            commands: List of (command_name, description) tuples to add.
        """
        self._command_completer.add_commands(commands)

    def clear_plugin_commands(self) -> None:
        """Clear all plugin-contributed commands."""
        self._command_completer.clear_plugin_commands()

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """Get completions from command, file, and slash command completers."""
        # Yield completions from all sources
        # CommandCompleter will only yield if appropriate (single word, no @ or /)
        # AtFileCompleter will only yield if @ is present
        # SlashCommandCompleter will only yield if / is present at start of word
        yield from self._command_completer.get_completions(document, complete_event)
        yield from self._file_completer.get_completions(document, complete_event)
        yield from self._slash_completer.get_completions(document, complete_event)
