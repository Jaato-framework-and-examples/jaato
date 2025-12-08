"""Slash command plugin for processing /command references.

This plugin enables users to type /command_name arg1 arg2 which gets sent to the model.
The model understands this as a reference to a command file in .jaato/commands/
and calls the processCommand tool to read, substitute parameters, and process the file.

Template syntax in command files:
- {{$1}}, {{$2}}, etc. - Positional parameters (1-indexed)
- {{$1:default}} - Positional parameter with default value
- {{$0}} - All arguments joined with spaces
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from ..model_provider.types import ToolSchema

from ..base import UserCommand


# Default commands directory relative to current working directory
DEFAULT_COMMANDS_DIR = ".jaato/commands"

# Maximum file size to read (100KB)
MAX_COMMAND_FILE_SIZE = 100_000

# Pattern to match template variables: {{$1}}, {{$2:default}}, etc.
TEMPLATE_PATTERN = re.compile(r'\{\{\$(\d+)(?::([^}]*))?\}\}')


class SlashCommandPlugin:
    """Plugin that provides slash command processing capability.

    Users can type /command_name and the model will call processCommand
    to read and process the corresponding file from .jaato/commands/.

    Configuration:
        commands_dir: Path to commands directory (default: .jaato/commands)
    """

    def __init__(self):
        self._commands_dir: str = DEFAULT_COMMANDS_DIR
        self._initialized = False

    @property
    def name(self) -> str:
        return "slash_command"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the slash command plugin.

        Args:
            config: Optional dict with:
                - commands_dir: Path to commands directory (default: .jaato/commands)
        """
        if config:
            if 'commands_dir' in config:
                self._commands_dir = config['commands_dir']
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the slash command plugin."""
        self._initialized = False

    def get_commands_dir(self) -> Path:
        """Get the absolute path to the commands directory."""
        commands_path = Path(self._commands_dir)
        if not commands_path.is_absolute():
            commands_path = Path.cwd() / commands_path
        return commands_path

    def list_available_commands(self) -> List[str]:
        """List available command names (files in commands directory).

        Returns:
            List of command names (filenames without full path)
        """
        commands_path = self.get_commands_dir()
        if not commands_path.exists():
            return []

        commands = []
        for item in commands_path.iterdir():
            if item.is_file():
                commands.append(item.name)
        return sorted(commands)

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the ToolSchema for the processCommand tool."""
        return [ToolSchema(
            name='processCommand',
            description='Process a slash command by reading its file from .jaato/commands/ directory. '
                       'Call this when the user types /command_name [args...] to retrieve and process '
                       'the command file contents with parameter substitution.',
            parameters={
                "type": "object",
                "properties": {
                    "command_name": {
                        "type": "string",
                        "description": "The name of the command file to process (without leading /). "
                                      "For example, if user types '/summarize file.py', pass 'summarize'."
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Positional arguments passed after the command name. "
                                      "For '/summarize file.py --verbose', pass ['file.py', '--verbose']. "
                                      "These are substituted into the command template as {{$1}}, {{$2}}, etc."
                    }
                },
                "required": ["command_name"]
            }
        )]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor mapping."""
        return {'processCommand': self._execute_process_command}

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for slash commands."""
        # Get available commands for the instructions
        available = self.list_available_commands()
        commands_list = ", ".join(f"/{cmd}" for cmd in available) if available else "(no commands available yet)"

        return f"""You have access to slash commands via `processCommand`.

When the user types a message starting with "/" (e.g., "/summarize file.py", "/review src/"),
this is a slash command referencing a command file in .jaato/commands/ directory.

To process a slash command:
1. Extract the command name (the first word after "/")
2. Extract any arguments (words after the command name)
3. Call processCommand(command_name="<name>", args=["arg1", "arg2", ...])
4. Follow the instructions in the returned content

Currently available commands: {commands_list}

Examples:
- User types: "/summarize"
  You call: processCommand(command_name="summarize")

- User types: "/summarize file.py"
  You call: processCommand(command_name="summarize", args=["file.py"])

- User types: "/review src/main.py tests/"
  You call: processCommand(command_name="review", args=["src/main.py", "tests/"])

The command file may contain template variables that get substituted with args:
- {{{{$1}}}} - First argument
- {{{{$2}}}} - Second argument
- {{{{$1:default}}}} - First argument with default value if not provided
- {{{{$0}}}} - All arguments joined with spaces

After processing, the tool returns the command content with parameters substituted.
Execute the instructions contained in the returned content.
If the command file is not found, inform the user and suggest listing available commands."""

    def get_auto_approved_tools(self) -> List[str]:
        """processCommand is read-only, safe to auto-approve."""
        return ['processCommand']

    def get_user_commands(self) -> List[UserCommand]:
        """Slash command plugin provides model tools only, no direct user commands.

        Note: Slash commands go through the model which calls processCommand.
        """
        return []

    def _substitute_parameters(self, content: str, args: List[str]) -> tuple[str, List[str]]:
        """Substitute template variables in command content with provided arguments.

        Template syntax:
        - {{$1}}, {{$2}}, etc. - Positional parameters (1-indexed)
        - {{$1:default}} - Positional parameter with default value
        - {{$0}} - All arguments joined with spaces

        Args:
            content: The command file content with template variables
            args: List of positional arguments

        Returns:
            Tuple of (substituted_content, list of missing required parameters)
        """
        missing_params = []

        def replace_match(match: re.Match) -> str:
            index = int(match.group(1))
            default = match.group(2)  # None if no default specified

            if index == 0:
                # {{$0}} - all arguments joined
                return ' '.join(args) if args else (default if default is not None else '')

            # 1-indexed positional parameter
            if index <= len(args):
                return args[index - 1]
            elif default is not None:
                return default
            else:
                missing_params.append(f'${index}')
                return f'{{{{${index}}}}}'  # Leave unreplaced

        result = TEMPLATE_PATTERN.sub(replace_match, content)
        return result, missing_params

    def _execute_process_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Process a slash command by reading its file and substituting parameters.

        Args:
            args: Dict containing 'command_name' and optionally 'args' (list of strings).

        Returns:
            Dict containing the processed command content or an error.
        """
        command_name = args.get('command_name', '').strip()
        command_args = args.get('args', []) or []

        if not command_name:
            return {
                'error': 'No command name provided',
                'hint': 'Pass the command name without the leading slash'
            }

        # Remove leading slash if accidentally included
        if command_name.startswith('/'):
            command_name = command_name[1:]

        # Validate command name (prevent directory traversal)
        if '..' in command_name or '/' in command_name or '\\' in command_name:
            return {
                'error': 'Invalid command name',
                'hint': 'Command name cannot contain path separators or ..'
            }

        # Build path to command file
        commands_path = self.get_commands_dir()
        command_file = commands_path / command_name

        # Check if commands directory exists
        if not commands_path.exists():
            available = []
            return {
                'error': f'Commands directory not found: {self._commands_dir}',
                'hint': f'Create the directory and add command files to it',
                'available_commands': available
            }

        # Check if command file exists
        if not command_file.exists():
            available = self.list_available_commands()
            return {
                'error': f'Command not found: {command_name}',
                'available_commands': available,
                'hint': f'Available commands: {", ".join(available)}' if available else 'No commands available'
            }

        # Check if it's a file (not a directory)
        if not command_file.is_file():
            return {
                'error': f'{command_name} is not a file',
                'hint': 'Command must reference a file, not a directory'
            }

        # Check file size
        file_size = command_file.stat().st_size
        if file_size > MAX_COMMAND_FILE_SIZE:
            return {
                'error': f'Command file too large: {file_size} bytes',
                'hint': f'Maximum allowed size is {MAX_COMMAND_FILE_SIZE} bytes'
            }

        # Read the command file
        try:
            raw_content = command_file.read_text(encoding='utf-8')

            # Substitute parameters if any args provided
            if command_args:
                content, missing = self._substitute_parameters(raw_content, command_args)
                result = {
                    'command_name': command_name,
                    'content': content,
                    'file_path': str(command_file),
                    'size': file_size,
                    'args_provided': command_args,
                }
                if missing:
                    result['missing_parameters'] = missing
                    result['hint'] = f'Some parameters were not provided: {", ".join(missing)}'
                return result
            else:
                return {
                    'command_name': command_name,
                    'content': raw_content,
                    'file_path': str(command_file),
                    'size': file_size
                }
        except Exception as e:
            return {
                'error': f'Failed to read command file: {e}',
                'command_name': command_name
            }


def create_plugin() -> SlashCommandPlugin:
    """Factory function to create the slash command plugin instance."""
    return SlashCommandPlugin()
