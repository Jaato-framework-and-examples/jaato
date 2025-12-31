"""CLI tool plugin for executing local shell commands."""

import os
import re
import shutil
import shlex
import subprocess
import tempfile
import threading
from datetime import datetime
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
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Callback for streaming output during execution (tail -f style)
        self._tool_output_callback: Optional[Callable[[str], None]] = None

    @property
    def name(self) -> str:
        return "cli"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [CLI{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

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
            # Extract agent name for trace logging
            self._agent_name = config.get("agent_name")
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
        self._trace(f"initialize: extra_paths={self._extra_paths}, max_output={self._max_output_chars}, auto_bg_threshold={self._auto_background_threshold}")

    def set_tool_output_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set the callback for streaming output during execution.

        When set, the plugin will stream output lines to the callback during
        command execution, enabling live "tail -f" style preview in the UI.

        Args:
            callback: Function that accepts output chunks, or None to disable.
        """
        self._tool_output_callback = callback
        self._trace(f"set_tool_output_callback: callback={'SET' if callback else 'CLEARED'}")

    def shutdown(self) -> None:
        """Shutdown the CLI plugin."""
        self._trace("shutdown: cleaning up")
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

LONG-RUNNING COMMANDS AND AUTO-BACKGROUNDING:
Commands that take longer than 10 seconds are automatically moved to background execution.
When this happens, instead of stdout/stderr, you receive:
{
    "auto_backgrounded": true,
    "task_id": "abc-123",
    "message": "Task exceeded 10.0s threshold, continuing in background..."
}

Known slow commands that will be auto-backgrounded:
- Package managers: npm install, pip install, cargo build, mvn install, gradle build
- Build commands: make, cmake, docker build
- Test suites: pytest, npm test, mvn test, cargo test

When a command is auto-backgrounded, use `getBackgroundTask` to monitor it:

Example workflow for a Maven build:
1. cli_based_tool(command="mvn clean install")
   -> {"auto_backgrounded": true, "task_id": "xyz-789", ...}

2. getBackgroundTask(task_id="xyz-789")
   -> {"status": "running", "stdout": "Downloading...", "stdout_offset": 1024, "has_more": true}

3. getBackgroundTask(task_id="xyz-789", stdout_offset=1024)
   -> {"status": "running", "stdout": "[ERROR] Compilation failed", "stdout_offset": 2048, "has_more": true}
   -> React to errors early! Consider: cancelBackgroundTask(task_id="xyz-789")

4. getBackgroundTask(task_id="xyz-789", stdout_offset=2048)
   -> {"status": "completed", "stdout": "BUILD SUCCESS", "has_more": false, "returncode": 0}

Use the returned stdout_offset for subsequent calls to get only new output.

ERROR HANDLING:
- A non-zero returncode indicates the command failed - always check stderr for details
- "File exists" or "Directory exists" errors mean the goal is already achieved - consider the step successful and continue
- "Permission denied" - try an alternative approach (different path, sudo if appropriate) or report as a blocker
- "Command not found" - check if the required tool is installed, or try an alternative command
- "No such file or directory" - verify the path exists before operating on it
- When a step fails, decide whether to: retry with a workaround, skip if goal is met, or report the blocker

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

    def _get_streaming_executor(
        self,
        tool_name: str
    ) -> Optional[Callable[..., Any]]:
        """Get a streaming executor for CLI commands.

        When running in background mode, this executor uses Popen with
        threading to capture stdout/stderr incrementally.

        Args:
            tool_name: Name of the tool.

        Returns:
            Streaming executor for cli_based_tool, None otherwise.
        """
        if tool_name == 'cli_based_tool':
            return self._execute_streaming
        return None

    def _execute_streaming(
        self,
        args: Dict[str, Any],
        on_stdout: Callable[[bytes], None],
        on_stderr: Callable[[bytes], None],
        on_returncode: Callable[[int], None]
    ) -> Dict[str, Any]:
        """Execute a CLI command with streaming output capture.

        Uses subprocess.Popen with threading to capture stdout/stderr
        incrementally and route them to the provided callbacks.

        Args:
            args: Dict containing 'command' and optionally 'args'.
            on_stdout: Callback for stdout data chunks.
            on_stderr: Callback for stderr data chunks.
            on_returncode: Callback for exit code.

        Returns:
            Dict containing stdout, stderr and returncode.
        """
        try:
            command = args.get('command')
            arg_list = args.get('args')
            extra_paths = args.get('extra_paths', self._extra_paths)

            if not command:
                return {'error': 'cli_based_tool: command must be provided'}

            cmd_preview = command[:100] + "..." if len(command) > 100 else command
            self._trace(f"execute_streaming: {cmd_preview}")

            # Prepare environment
            env = os.environ.copy()
            if extra_paths:
                path_sep = os.pathsep
                env['PATH'] = env.get('PATH', '') + path_sep + path_sep.join(extra_paths)

            # Check if shell interpretation is needed
            use_shell = self._requires_shell(command)

            # Prepare command/argv
            argv: Optional[List[str]] = None
            if not use_shell:
                if arg_list:
                    argv = [command] + arg_list
                else:
                    argv = shlex.split(command)

                if len(argv) == 1 and ' ' in argv[0]:
                    argv = shlex.split(argv[0])

                exe = argv[0]
                resolved = shutil.which(exe, path=env.get('PATH'))
                if resolved:
                    argv[0] = resolved
                else:
                    return {
                        'error': f"cli_based_tool: executable '{exe}' not found in PATH",
                        'hint': 'Configure extra_paths or provide full path to the executable.'
                    }

            # Start process with pipes
            cmd = command if use_shell else argv
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                shell=use_shell
            )

            # Collect output while streaming to callbacks
            stdout_chunks: List[bytes] = []
            stderr_chunks: List[bytes] = []
            stdout_done = threading.Event()
            stderr_done = threading.Event()

            def read_stdout():
                """Read stdout in a thread and call callback."""
                try:
                    # Use line-by-line reading for real-time streaming
                    # read(n) blocks until n bytes are available, which doesn't
                    # work well for slow-producing commands
                    for line in iter(proc.stdout.readline, b''):
                        if not line:
                            break
                        stdout_chunks.append(line)
                        on_stdout(line)
                finally:
                    stdout_done.set()

            def read_stderr():
                """Read stderr in a thread and call callback."""
                try:
                    for line in iter(proc.stderr.readline, b''):
                        if not line:
                            break
                        stderr_chunks.append(line)
                        on_stderr(line)
                finally:
                    stderr_done.set()

            # Start reader threads
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process and readers to complete
            proc.wait()
            stdout_done.wait()
            stderr_done.wait()

            returncode = proc.returncode
            on_returncode(returncode)

            # Combine output for final result
            stdout = b''.join(stdout_chunks).decode('utf-8', errors='replace')
            stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')

            # Truncate for final result (streaming already captured full output)
            truncated = False
            if len(stdout) > self._max_output_chars:
                stdout = stdout[:self._max_output_chars]
                truncated = True
            if len(stderr) > self._max_output_chars:
                stderr = stderr[:self._max_output_chars]
                truncated = True

            result = {'stdout': stdout, 'stderr': stderr, 'returncode': returncode}
            if truncated:
                result['truncated'] = True
                result['truncation_message'] = (
                    f"Output truncated to {self._max_output_chars} chars in final result. "
                    "Full output available via getBackgroundTaskOutput."
                )

            return result

        except Exception as exc:
            return {'error': str(exc)}

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

            # Truncate command for logging (avoid huge commands in trace)
            cmd_preview = command[:100] + "..." if len(command) > 100 else command
            self._trace(f"execute: {cmd_preview}")

            # Prepare environment with extended PATH if extra_paths is provided
            env = os.environ.copy()
            if extra_paths:
                path_sep = os.pathsep
                env['PATH'] = env.get('PATH', '') + path_sep + path_sep.join(extra_paths)

            # Check if the command requires shell interpretation
            use_shell = self._requires_shell(command)

            # Prepare command/argv for execution
            argv: Optional[List[str]] = None
            if not use_shell:
                # Non-shell mode: parse into argv list for safer execution
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

            # Use streaming execution if callback is set
            self._trace(f"execute: streaming={'YES' if self._tool_output_callback else 'NO'}")
            if self._tool_output_callback:
                # Streaming mode with Popen for live output
                cmd = command if use_shell else argv
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    shell=use_shell
                )

                # Read output line by line and stream to callback
                stdout_lines = []
                stderr_lines = []

                # Read stdout with streaming callback
                if proc.stdout:
                    for line in proc.stdout:
                        stdout_lines.append(line)
                        # Call the callback with the line (strip newline for display)
                        self._tool_output_callback(line.rstrip('\n\r'))

                # Read remaining stderr (non-streaming for simplicity)
                if proc.stderr:
                    stderr_lines = proc.stderr.readlines()

                proc.wait()
                stdout = ''.join(stdout_lines)
                stderr = ''.join(stderr_lines)
                returncode = proc.returncode
            else:
                # Non-streaming mode with subprocess.run
                if use_shell:
                    # Shell mode: pass command string directly to shell
                    proc = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        check=False,
                        env=env,
                        shell=True
                    )
                else:
                    proc = subprocess.run(
                        argv,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        check=False,
                        env=env,
                        shell=False
                    )
                stdout = proc.stdout
                stderr = proc.stderr
                returncode = proc.returncode

            # Truncate large outputs to prevent context window overflow
            truncated = False

            if len(stdout) > self._max_output_chars:
                stdout = stdout[:self._max_output_chars]
                truncated = True

            if len(stderr) > self._max_output_chars:
                stderr = stderr[:self._max_output_chars]
                truncated = True

            result = {'stdout': stdout, 'stderr': stderr, 'returncode': returncode}

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
