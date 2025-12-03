"""CLI tool plugin for executing local shell commands."""

import os
import shutil
import shlex
import subprocess
from typing import Dict, List, Any, Callable, Optional
from google.genai import types


class CLIToolPlugin:
    """Plugin that provides CLI command execution capability.

    Configuration:
        extra_paths: List of additional paths to add to PATH when executing commands.
    """

    def __init__(self):
        self._extra_paths: List[str] = []
        self._initialized = False

    @property
    def name(self) -> str:
        return "cli"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the CLI plugin.

        Args:
            config: Optional dict with 'extra_paths' key for additional PATH entries.
        """
        if config and 'extra_paths' in config:
            paths = config['extra_paths']
            if paths:
                self._extra_paths = paths if isinstance(paths, list) else [paths]
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the CLI plugin."""
        self._extra_paths = []
        self._initialized = False

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Return the FunctionDeclaration for the CLI tool."""
        return [types.FunctionDeclaration(
            name='cli_based_tool',
            description='Execute a local CLI command',
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Full command string, e.g. 'gh issue view 123' or 'git status'"
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Argument list for the executable"
                    }
                },
                "required": ["command"]
            }
        )]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor mapping."""
        return {'cli_based_tool': self._execute}

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the CLI tool."""
        return """You have access to `cli_based_tool` which executes local shell commands on the user's machine.

Use it to run commands like `ls`, `cat`, `grep`, `find`, `git`, `gh`, etc.

Example usage:
- List files: cli_based_tool(command="ls -la")
- Read a file: cli_based_tool(command="cat /path/to/file")
- Check git status: cli_based_tool(command="git status")
- Search for text: cli_based_tool(command="grep -r 'pattern' /path")

The tool returns stdout, stderr, and returncode from the executed command."""

    def get_auto_approved_tools(self) -> List[str]:
        """CLI tools require permission - return empty list."""
        return []

    def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a CLI command.

        Exactly one of the following forms should be provided:
        1. command: full shell-like command string (preferred for simplicity).
        2. command + args: command as executable name and args as argument list.

        Args:
            args: Dict containing 'command' and optionally 'args' and 'extra_paths'.

        Returns:
            Dict containing stdout, stderr and returncode; on failure contains error.
        """
        try:
            command = args.get('command')
            arg_list = args.get('args')
            extra_paths = args.get('extra_paths', self._extra_paths)

            argv: List[str] = []
            if command and arg_list:
                # Model passed command as executable name and args separately
                argv = [command] + arg_list
            elif command:
                # Full command string
                argv = shlex.split(command)
            else:
                return {'error': 'cli_based_tool: command must be provided'}

            # Normalize single-string with spaces passed mistakenly as executable
            if len(argv) == 1 and ' ' in argv[0]:
                argv = shlex.split(argv[0])

            # Prepare environment with extended PATH if extra_paths is provided
            env = os.environ.copy()
            if extra_paths:
                path_sep = os.pathsep
                env['PATH'] = env.get('PATH', '') + path_sep + path_sep.join(extra_paths)

            # Resolve executable via PATH (including PATHEXT) for Windows
            # This avoids relying on shell resolution while still finding .exe/.bat
            exe = argv[0]
            resolved = shutil.which(exe, path=env.get('PATH'))
            if resolved:
                argv[0] = resolved
            else:
                # If not found, return a clear error with hint about extra_paths
                return {
                    'error': f"cli_based_tool: executable '{exe}' not found in PATH",
                    'hint': 'Configure extra_paths or provide full path to the executable.'
                }

            # Execute without shell so arguments with spaces/quotes are preserved
            # Passing a list to subprocess.run ensures proper quoting on Windows
            proc = subprocess.run(argv, capture_output=True, text=True, check=False, env=env, shell=False)
            return {'stdout': proc.stdout, 'stderr': proc.stderr, 'returncode': proc.returncode}

        except Exception as exc:
            return {'error': str(exc)}


def create_plugin() -> CLIToolPlugin:
    """Factory function to create the CLI plugin instance."""
    return CLIToolPlugin()
