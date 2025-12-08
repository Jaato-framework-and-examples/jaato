"""CLI tool plugin for executing local shell commands."""

import os
import re
import shutil
import shlex
import subprocess
from typing import Dict, List, Any, Callable, Optional

from ..base import UserCommand
from ..background import BackgroundCapableMixin
from ..model_provider.types import ToolSchema


DEFAULT_MAX_OUTPUT_CHARS = 50000  # ~12k tokens at 4 chars/token

# Default auto-background threshold in seconds
# Commands exceeding this will be automatically backgrounded
DEFAULT_AUTO_BACKGROUND_THRESHOLD = 10.0

# Command patterns that are known to be slow
# Maps pattern to estimated duration in seconds
SLOW_COMMAND_PATTERNS = {
    # Package managers
    'npm install': 30.0,
    'npm ci': 30.0,
    'yarn install': 30.0,
    'pip install': 20.0,
    'pip3 install': 20.0,
    'poetry install': 25.0,
    'cargo build': 60.0,
    'cargo install': 45.0,
    'go build': 30.0,
    'mvn install': 60.0,
    'gradle build': 45.0,
    # Build commands
    'make': 30.0,
    'cmake': 20.0,
    'ninja': 30.0,
    # Test commands
    'pytest': 30.0,
    'npm test': 30.0,
    'yarn test': 30.0,
    'go test': 20.0,
    'cargo test': 30.0,
    'mvn test': 45.0,
    # Other slow operations
    'docker build': 60.0,
    'docker pull': 30.0,
    'git clone': 20.0,
    'wget': 15.0,
    'curl': 10.0,
}

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


class CLIToolPlugin(BackgroundCapableMixin):
    """Plugin that provides CLI command execution capability.

    Supports background execution via BackgroundCapableMixin. Commands that
    exceed the auto-background threshold (default: 10 seconds) will be
    automatically converted to background tasks.

    Configuration:
        extra_paths: List of additional paths to add to PATH when executing commands.
        max_output_chars: Maximum characters to return from stdout/stderr (default: 50000).
        auto_background_threshold: Seconds before auto-backgrounding (default: 10.0).
        background_max_workers: Max concurrent background tasks (default: 4).
    """

    def __init__(self):
        # Initialize BackgroundCapableMixin first
        super().__init__(max_workers=4)

        self._extra_paths: List[str] = []
        self._max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
        self._auto_background_threshold: float = DEFAULT_AUTO_BACKGROUND_THRESHOLD
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
                - auto_background_threshold: Seconds before auto-backgrounding (default: 10.0)
                - background_max_workers: Max concurrent background tasks (default: 4)
        """
        if config:
            if 'extra_paths' in config:
                paths = config['extra_paths']
                if paths:
                    self._extra_paths = paths if isinstance(paths, list) else [paths]
            if 'max_output_chars' in config:
                self._max_output_chars = config['max_output_chars']
            if 'auto_background_threshold' in config:
                self._auto_background_threshold = config['auto_background_threshold']
            if 'background_max_workers' in config:
                self._bg_max_workers = config['background_max_workers']
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the CLI plugin."""
        self._extra_paths = []
        self._initialized = False
        # Cleanup background executor
        self._shutdown_bg_executor()

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the ToolSchema for the CLI tool."""
        return [ToolSchema(
            name='cli_based_tool',
            description=(
                'Execute any shell command on the local machine. This tool provides full access to '
                'the command line, allowing you to: create/delete/move files and directories, '
                'read and write file contents, run scripts and programs, manage git repositories, '
                'install packages, and perform any operation that a user could do in a terminal. '
                'Supports shell features like pipes (|), redirections (>, >>), and command chaining (&&, ||).'
            ),
            parameters={
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

    # --- BackgroundCapable implementation ---

    def supports_background(self, tool_name: str) -> bool:
        """Check if a tool supports background execution.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool can be executed in background.
        """
        # CLI tool supports background execution
        return tool_name == 'cli_based_tool'

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        """Return timeout threshold for automatic backgrounding.

        When a CLI command exceeds this threshold, it's automatically
        converted to a background task and a handle is returned.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            Threshold in seconds, or None to disable auto-background.
        """
        if tool_name == 'cli_based_tool':
            return self._auto_background_threshold
        return None

    def estimate_duration(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate execution duration based on command patterns.

        Analyzes the command to provide duration hints for known slow operations
        like package installations, builds, and tests.

        Args:
            tool_name: Name of the tool.
            arguments: Arguments containing the command.

        Returns:
            Estimated duration in seconds, or None if unknown.
        """
        if tool_name != 'cli_based_tool':
            return None

        command = arguments.get('command', '')
        if not command:
            return None

        # Check against known slow patterns
        command_lower = command.lower()
        for pattern, duration in SLOW_COMMAND_PATTERNS.items():
            if pattern in command_lower:
                return duration

        # Default: unknown duration
        return None

    # --- End BackgroundCapable implementation ---

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
