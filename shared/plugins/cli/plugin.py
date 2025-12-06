"""CLI tool plugin for executing local shell commands."""

import os
import re
import shutil
import shlex
import subprocess
from typing import Dict, List, Any, Callable, Optional
from google.genai import types

from ..base import UserCommand


DEFAULT_MAX_OUTPUT_CHARS = 50000  # ~12k tokens at 4 chars/token

# Shell metacharacters that require shell interpretation
# These cannot be handled by subprocess with shell=False
SHELL_METACHAR_PATTERN = re.compile(
    r'[|<>]'           # Pipes and redirections
    r'|&&|\|\|'        # Command chaining (AND/OR)
    r'|;'              # Command separator
    r'|\$\('           # Command substitution $(...)
    r'|`'              # Backtick command substitution
    r'|&\s*$'          # Background execution (& at end)
)


class CLIToolPlugin:
    """Plugin that provides CLI command execution capability.

    Configuration:
        extra_paths: List of additional paths to add to PATH when executing commands.
        max_output_chars: Maximum characters to return from stdout/stderr (default: 50000).
    """

    def __init__(self):
        self._extra_paths: List[str] = []
        self._max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
        self._initialized = False

    @property
    def name(self) -> str:
        return "cli"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the CLI plugin.

        Args:
            config: Optional dict with:
                - extra_paths: Additional PATH entries
                - max_output_chars: Max characters to return (default: 50000)
        """
        if config:
            if 'extra_paths' in config:
                paths = config['extra_paths']
                if paths:
                    self._extra_paths = paths if isinstance(paths, list) else [paths]
            if 'max_output_chars' in config:
                self._max_output_chars = config['max_output_chars']
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the CLI plugin."""
        self._extra_paths = []
        self._initialized = False

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Return the FunctionDeclaration for the CLI tool."""
        return [types.FunctionDeclaration(
            name='cli_based_tool',
            description=(
                'Execute any shell command on the local machine. This tool provides full access to '
                'the command line, allowing you to: create/delete/move files and directories, '
                'read and write file contents, run scripts and programs, manage git repositories, '
                'install packages, and perform any operation that a user could do in a terminal. '
                'Supports shell features like pipes (|), redirections (>, >>), and command chaining (&&, ||).'
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The shell command to execute. Examples: "
                            "'mkdir -p /path/to/new/folder' (create directories), "
                            "'echo \"content\" > file.txt' (create/write files), "
                            "'cat file.txt' (read files), "
                            "'rm -rf /path/to/delete' (delete files/directories), "
                            "'mv old.txt new.txt' (rename/move files), "
                            "'ls -la' (list directory contents), "
                            "'git status' (check repository status), "
                            "'grep -r \"pattern\" /path' (search in files)"
                        )
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional argument list if passing executable and args separately"
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
        return """You have access to `cli_based_tool` which executes shell commands on the user's machine.

This tool gives you FULL access to the command line. You can perform ANY operation that a user could do in a terminal, including but not limited to:

FILESYSTEM OPERATIONS:
- Create directories: cli_based_tool(command="mkdir -p /path/to/new/folder")
- Create/write files: cli_based_tool(command="echo 'content' > file.txt")
- Append to files: cli_based_tool(command="echo 'more content' >> file.txt")
- Read files: cli_based_tool(command="cat /path/to/file")
- Delete files/directories: cli_based_tool(command="rm -rf /path/to/delete")
- Move/rename files: cli_based_tool(command="mv old.txt new.txt")
- Copy files: cli_based_tool(command="cp source.txt destination.txt")
- List directory contents: cli_based_tool(command="ls -la")
- Check disk usage: cli_based_tool(command="du -sh /path")

SEARCHING AND FILTERING:
- Find files: cli_based_tool(command="find /path -name '*.py'")
- Search file contents: cli_based_tool(command="grep -r 'pattern' /path")
- Filter output: cli_based_tool(command="ls -la | grep '.py'")

VERSION CONTROL:
- Check git status: cli_based_tool(command="git status")
- View git log: cli_based_tool(command="git log --oneline -10")
- Create branches: cli_based_tool(command="git checkout -b new-branch")

RUNNING PROGRAMS:
- Execute scripts: cli_based_tool(command="python script.py")
- Run tests: cli_based_tool(command="pytest tests/")
- Install packages: cli_based_tool(command="pip install package-name")

Shell features like pipes (|), redirections (>, >>), and command chaining (&&, ||) are fully supported.

The tool returns stdout, stderr, and returncode from the executed command.

IMPORTANT: Large outputs are truncated to prevent context overflow. To avoid truncation:
- Use filters (grep, awk) to narrow results
- Use head/tail to limit output lines
- Use -maxdepth with find to limit recursion"""

    def get_auto_approved_tools(self) -> List[str]:
        """CLI tools require permission - return empty list."""
        return []

    def get_user_commands(self) -> List[UserCommand]:
        """CLI plugin provides model tools only, no user commands."""
        return []

    def _requires_shell(self, command: str) -> bool:
        """Check if a command requires shell interpretation.

        Detects shell metacharacters like pipes, redirections, command chaining,
        and command substitution that cannot be handled by subprocess without shell.

        Args:
            command: The command string to check.

        Returns:
            True if the command contains shell metacharacters requiring shell=True.
        """
        return bool(SHELL_METACHAR_PATTERN.search(command))

    def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a CLI command.

        Exactly one of the following forms should be provided:
        1. command: full shell-like command string (preferred for simplicity).
        2. command + args: command as executable name and args as argument list.

        Shell metacharacters (pipes, redirections, command chaining) are auto-detected
        and the command is executed through the shell when required.

        Args:
            args: Dict containing 'command' and optionally 'args' and 'extra_paths'.

        Returns:
            Dict containing stdout, stderr and returncode; on failure contains error.
        """
        try:
            command = args.get('command')
            arg_list = args.get('args')
            extra_paths = args.get('extra_paths', self._extra_paths)

            if not command:
                return {'error': 'cli_based_tool: command must be provided'}

            # Prepare environment with extended PATH if extra_paths is provided
            env = os.environ.copy()
            if extra_paths:
                path_sep = os.pathsep
                env['PATH'] = env.get('PATH', '') + path_sep + path_sep.join(extra_paths)

            # Check if the command requires shell interpretation
            use_shell = self._requires_shell(command)

            if use_shell:
                # Shell mode: pass command string directly to shell
                # Required for pipes, redirections, command chaining, etc.
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env,
                    shell=True
                )
            else:
                # Non-shell mode: parse into argv list for safer execution
                argv: List[str] = []
                if arg_list:
                    # Model passed command as executable name and args separately
                    argv = [command] + arg_list
                else:
                    # Full command string
                    argv = shlex.split(command)

                # Normalize single-string with spaces passed mistakenly as executable
                if len(argv) == 1 and ' ' in argv[0]:
                    argv = shlex.split(argv[0])

                # Resolve executable via PATH (including PATHEXT) for Windows
                exe = argv[0]
                resolved = shutil.which(exe, path=env.get('PATH'))
                if resolved:
                    argv[0] = resolved
                else:
                    return {
                        'error': f"cli_based_tool: executable '{exe}' not found in PATH",
                        'hint': 'Configure extra_paths or provide full path to the executable.'
                    }

                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env,
                    shell=False
                )

            # Truncate large outputs to prevent context window overflow
            stdout = proc.stdout
            stderr = proc.stderr
            truncated = False

            if len(stdout) > self._max_output_chars:
                stdout = stdout[:self._max_output_chars]
                truncated = True

            if len(stderr) > self._max_output_chars:
                stderr = stderr[:self._max_output_chars]
                truncated = True

            result = {'stdout': stdout, 'stderr': stderr, 'returncode': proc.returncode}

            if truncated:
                result['truncated'] = True
                result['truncation_message'] = (
                    f"Output truncated to {self._max_output_chars} chars. "
                    "Consider using more specific commands (e.g., add filters, limits, or pipe to head/tail)."
                )

            return result

        except Exception as exc:
            return {'error': str(exc)}


def create_plugin() -> CLIToolPlugin:
    """Factory function to create the CLI plugin instance."""
    return CLIToolPlugin()
