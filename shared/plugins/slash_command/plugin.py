"""Slash command plugin for processing /command references.

This plugin enables users to type /command_name which gets sent to the model.
The model understands this as a reference to a command file in .jaato/commands/
and calls the processCommand tool to read and process the file.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from google.genai import types

from ..base import UserCommand


# Default commands directory relative to current working directory
DEFAULT_COMMANDS_DIR = ".jaato/commands"

# Maximum file size to read (100KB)
MAX_COMMAND_FILE_SIZE = 100_000


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

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Return the FunctionDeclaration for the processCommand tool."""
        return [types.FunctionDeclaration(
            name='processCommand',
            description='Process a slash command by reading its file from .jaato/commands/ directory. '
                       'Call this when the user types /command_name to retrieve the command file contents.',
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "command_name": {
                        "type": "string",
                        "description": "The name of the command file to process (without leading /). "
                                      "For example, if user types '/summarize', pass 'summarize'."
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

When the user types a message starting with "/" (e.g., "/summarize", "/review", "/help"),
this is a slash command referencing a command file in .jaato/commands/ directory.

To process a slash command:
1. Extract the command name (the part after "/")
2. Call processCommand(command_name="<name>") to read the command file
3. Follow the instructions contained in the command file

Currently available commands: {commands_list}

Example:
- User types: "/summarize"
- You call: processCommand(command_name="summarize")
- The tool returns the command file contents with instructions to follow

The command file may contain:
- A prompt template or instructions for how to respond
- Parameters or context for the task
- Specific formatting requirements

After reading the command file, execute the instructions it contains.
If the command file is not found, inform the user and suggest listing available commands."""

    def get_auto_approved_tools(self) -> List[str]:
        """processCommand is read-only, safe to auto-approve."""
        return ['processCommand']

    def get_user_commands(self) -> List[UserCommand]:
        """Slash command plugin provides model tools only, no direct user commands.

        Note: Slash commands go through the model which calls processCommand.
        """
        return []

    def _execute_process_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Process a slash command by reading its file.

        Args:
            args: Dict containing 'command_name'.

        Returns:
            Dict containing the command file contents or an error.
        """
        command_name = args.get('command_name', '').strip()

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
            content = command_file.read_text(encoding='utf-8')
            return {
                'command_name': command_name,
                'content': content,
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
