"""CLI tool plugin for executing local shell commands."""

import logging
import os
from shared.session_context import get_workspace_root, get_config_root
import re
import shutil
import shlex
import subprocess
import tempfile
import threading
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger(__name__)

from jaato_sdk.plugins.base import UserCommand
from ..background import BackgroundCapableMixin
from shared.plugins.runner_forwarding import RunnerForwardingMixin
from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    EditableContent,
    DISCOVERABILITY_DEFERRED,
)
from ..sandbox_utils import check_path_with_jaato_containment, detect_jaato_symlink
from ..workspace_venv import (
    resolve_venv_path, ensure_workspace_venv, apply_venv_to_env, pip_apparmor_rules,
)
from shared.ai_tool_runner import get_current_tool_output_callback, get_current_cancel_token
from jaato_sdk.plugins.model_provider.types import CancelledException
from shared.path_utils import msys2_to_windows_path
from shared.subprocess_runner import run_command, requires_shell, RunResult
from shared.trace import trace as _trace_write
from shared.command_analysis import (
    Segment,
    UnanalyzableCommand,
    WRAPPER_COMMANDS,
    analyze_command,
)


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
    r'|\$\{'           # Variable expansion ${VAR}
    r'|\$[A-Za-z_]'    # Variable expansion $VAR
    r'|`'              # Backtick command substitution
    r'|&\s*$'          # Background execution (& at end)
)

# Commands whose path arguments are write targets.
# For commands with mixed semantics (cp, mv, install), the *last* path argument
# is treated as write; earlier ones are read.  For single-target commands (rm,
# touch, mkdir, etc.) all path arguments are write targets.
_WRITE_ALL_CMDS = frozenset({
    'rm', 'rmdir', 'touch', 'mkdir', 'mkfifo', 'mknod',
    'truncate', 'shred',
})
_WRITE_LAST_CMDS = frozenset({
    'cp', 'mv', 'install', 'rsync', 'scp',
})
_WRITE_OUTPUT_CMDS = frozenset({
    'tee',
})

# Every command name whose presence in a segment implies a write somewhere.
_ALL_WRITE_CMDS = _WRITE_ALL_CMDS | _WRITE_LAST_CMDS | _WRITE_OUTPUT_CMDS


def _path_like(token: str) -> bool:
    """True if a command word should be treated as a filesystem path.

    Mirrors the historical heuristic used by
    :meth:`CLIToolPlugin._extract_path_tokens`: absolute paths, ``..``
    traversal, explicit ``./`` and ``~`` prefixes count; option flags,
    URLs and npm-style ``@scope/package`` names do not.

    Args:
        token: One word from a command, quoting already removed.

    Returns:
        True when the token should be run through the workspace check.
    """
    if not token or token.startswith('-'):
        return False
    if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', token):
        return False
    if token.startswith('@') and '/' in token and not token.startswith('@/'):
        return False
    return (token.startswith('/') or '..' in token or
            token.startswith('./') or token.startswith('~'))


def _arg_path_like(arg: str) -> bool:
    """True if an explicit ``args`` entry should be workspace-checked.

    Looser than :func:`_path_like` on purpose: entries in the separate
    ``args`` list are never shell-parsed, so flag/URL exclusions (which
    exist to avoid mis-reading shell words) must not weaken the check.
    """
    return (arg.startswith('/') or '..' in arg or
            arg.startswith('./') or arg.startswith('~'))


def _effective_command_name(segment: Segment) -> str:
    """Pick the command name that governs a segment's path semantics.

    A segment can name more than one command: ``sudo rm -rf x`` resolves to
    ``['sudo', 'rm']``.  Write semantics win, so the first resolved name in
    :data:`_ALL_WRITE_CMDS` is returned.  When the segment is headed by a
    wrapper (``sudo``, ``env``, ``xargs``, ...) whose argument layout this
    module does not model precisely, every word is scanned for a write
    command -- deliberately over-classifying as write rather than risking a
    write that reads as ``read``.

    Args:
        segment: One analyzed shell segment.

    Returns:
        The governing command basename, or ``''`` when the segment names
        no command (assignments only).
    """
    names = segment.command_names
    for name in names:
        if name in _ALL_WRITE_CMDS:
            return name
    if any(name in WRAPPER_COMMANDS for name in names):
        for word in segment.words:
            base = os.path.basename(word)
            if base in _ALL_WRITE_CMDS:
                return base
    return names[-1] if names else ''


def _classify_word_paths(cmd_name: str, paths: List[str]) -> List[tuple]:
    """Apply the command-name write heuristics to a segment's path words.

    Args:
        cmd_name: The governing command name for the segment.
        paths: Path-looking words, in source order.

    Returns:
        List of ``(path, mode)`` tuples where mode is "read" or "write".
    """
    if cmd_name in _WRITE_ALL_CMDS or cmd_name in _WRITE_OUTPUT_CMDS:
        return [(path, 'write') for path in paths]
    result = [(path, 'read') for path in paths]
    if cmd_name in _WRITE_LAST_CMDS and result:
        result[-1] = (result[-1][0], 'write')
    return result


class CLIToolPlugin(BackgroundCapableMixin, RunnerForwardingMixin):
    """Plugin that provides CLI command execution capability.

    Supports background execution via BackgroundCapableMixin. Commands that
    exceed the auto-background threshold (default: 10 seconds) will be
    automatically converted to background tasks.

    Configuration:
        extra_paths: List of additional paths to add to PATH when executing commands.
        max_output_chars: Maximum characters to return from stdout/stderr (default: 50000).
        auto_background_threshold: Seconds before auto-backgrounding (default: 10.0).
        background_max_workers: Max concurrent background tasks (default: 4).
        workspace_root: Root directory for path sandboxing. Paths outside this
            directory will appear as "No such file or directory" to the model.
    """

    def __init__(self):
        # Initialize BackgroundCapableMixin first
        super().__init__(max_workers=4)

        self._extra_paths: List[str] = []
        # Secrets-broker (feature #10): env-var name globs to strip from the
        # environment handed to model-driven subprocesses.  Empty = off.
        self._scrub_secret_env: List[str] = []
        self._max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
        self._auto_background_threshold: float = DEFAULT_AUTO_BACKGROUND_THRESHOLD
        self._initialized = False

        # Per-session runtime limits installed by ToolExecutor via
        # set_runtime_limits().  None until the executor calls; the
        # Popen branches treat None as "no kernel attach, no app-layer
        # cap override, no wall-clock timeout".
        self._cgroup_attach = None
        self._runtime_limits = None
        # Phase 5 §5.10c: AppArmor child-profile transition callback
        # installed via set_apparmor_child_transition_callback().  When
        # set, the cli plugin's Popen preexec_fn writes
        # ``changeprofile <profile>//child`` to /proc/self/attr/current
        # between fork() and exec() so the spawned subprocess lands in
        # the per-session ``//child`` sub-profile (which drops the
        # escape-vector rules).  None until the executor calls — same
        # contract as _cgroup_attach.  See
        # docs/design/phase5_5_10_apparmor_child_subprofile_audit.md.
        self._apparmor_child_transition: Optional[Callable[[], None]] = None
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Note: tool output callback is managed by BackgroundCapableMixin
        # (self._bg_tool_output_callback) via set_tool_output_callback()
        # Workspace root for path sandboxing (None = no sandboxing)
        self._workspace_root: Optional[str] = None
        # Workspace-scoped venv path for tool subprocesses (None/empty = off).
        # When set, commands run with this venv activated so the model's
        # ``pip install`` persists there and later imports resolve.  See
        # shared/plugins/workspace_venv.py.
        self._workspace_venv: Optional[str] = None
        # Plugin registry for checking authorized external paths
        self._plugin_registry = None

    @property
    def name(self) -> str:
        return "cli"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("CLI", msg)

    def _detect_workspace_root(self) -> Optional[str]:
        """Auto-detect workspace root from environment variables.

        Priority:
        1. JAATO_WORKSPACE_ROOT environment variable
        2. workspaceRoot environment variable (typically from .env file)

        Returns:
            Resolved absolute path to workspace root, or None if not found.
        """
        # Priority 1: JAATO_WORKSPACE_ROOT
        workspace = get_workspace_root()
        if workspace:
            resolved = os.path.realpath(os.path.abspath(workspace))
            self._trace(f"_detect_workspace_root: using JAATO_WORKSPACE_ROOT={resolved}")
            return resolved

        # Priority 2: workspaceRoot (from .env)
        workspace = os.environ.get('workspaceRoot')
        if workspace:
            resolved = os.path.realpath(os.path.abspath(workspace))
            self._trace(f"_detect_workspace_root: using workspaceRoot={resolved}")
            return resolved

        self._trace("_detect_workspace_root: no workspace root found, sandboxing disabled")
        return None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the CLI plugin.

        Args:
            config: Optional dict with:
                - extra_paths: Additional PATH entries
                - max_output_chars: Max characters to return (default: 50000)
                - auto_background_threshold: Seconds before auto-backgrounding (default: 10.0)
                - background_max_workers: Max concurrent background tasks (default: 4)
                - workspace_root: Root directory for path sandboxing. Paths outside
                    this directory will appear as "No such file or directory".
                    If not provided, auto-detects from JAATO_WORKSPACE_ROOT or
                    workspaceRoot environment variables.
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
            if 'workspace_root' in config:
                # Resolve to absolute path and normalize
                workspace = config['workspace_root']
                if workspace:
                    self._workspace_root = os.path.realpath(os.path.abspath(workspace))
            if 'workspace_venv' in config:
                self._workspace_venv = config['workspace_venv']
            if 'scrub_secret_env' in config:
                scrub = config['scrub_secret_env']
                if isinstance(scrub, str):
                    scrub = [scrub]
                if isinstance(scrub, (list, tuple)):
                    self._scrub_secret_env = [str(s) for s in scrub]

        # Auto-detect workspace_root from environment if not explicitly provided
        if not self._workspace_root:
            self._workspace_root = self._detect_workspace_root()

        self._initialized = True
        self._trace(f"initialize: extra_paths={self._extra_paths}, max_output={self._max_output_chars}, auto_bg_threshold={self._auto_background_threshold}, workspace_root={self._workspace_root}")

        # Log .jaato symlink detection for visibility
        if self._workspace_root:
            is_symlink, target = detect_jaato_symlink(self._workspace_root)
            if is_symlink:
                self._trace(f"initialize: .jaato is symlink -> {target} (contained escape enabled)")

    def set_tool_output_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set the callback for streaming output during execution.

        When set, the plugin will stream output lines to the callback during
        command execution, enabling live "tail -f" style preview in the UI.
        Also sets the mixin's callback for background task streaming.

        Args:
            callback: Function that accepts output chunks, or None to disable.
        """
        super().set_tool_output_callback(callback)
        self._trace(f"set_tool_output_callback: callback={'SET' if callback else 'CLEARED'}")

    def _get_effective_output_callback(self) -> Optional[Callable[[str], None]]:
        """Get the effective output callback for the current execution.

        Checks thread-local storage first (for parallel execution),
        then falls back to the instance-level callback.

        Returns:
            The callback to use, or None if not set.
        """
        # Thread-local takes priority (parallel execution)
        thread_callback = get_current_tool_output_callback()
        if thread_callback is not None:
            return thread_callback
        # Fall back to instance-level from mixin (sequential execution)
        return self._bg_tool_output_callback

    def set_workspace_path(self, path: Optional[str]) -> None:
        """Update the workspace root path.

        Called when a client connects with a different working directory.

        Args:
            path: The new workspace root path, or None to disable sandboxing.
        """
        if path:
            self._workspace_root = os.path.realpath(os.path.abspath(path))
        else:
            self._workspace_root = None
        self._trace(f"set_workspace_path: workspace_root={self._workspace_root}")

    def set_plugin_registry(self, registry) -> None:
        """Set the plugin registry for checking authorized external paths.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry
        registry.register_category("system", "Shell commands, environment, system operations")
        self._trace("set_plugin_registry: registry set")

    def set_runtime_limits(self, attach_callback, limits) -> None:
        """Receive per-session cgroup attach + app-layer caps from the executor.

        Forwarded by ``ToolExecutor.set_runtime_limits`` whenever the
        WS server provisions a session's cgroup.  At Popen time:

        * ``attach_callback`` becomes ``preexec_fn`` so the forked child
          joins the session's cgroup before ``exec``, picking up the
          kernel-enforced ``memory.max`` / ``pids.max`` / ``cpu.weight``.
        * ``limits.tool_timeout_seconds`` becomes a wall-clock deadline
          enforced by the Python layer (cgroup v2 has no equivalent).
        * ``limits.max_output_bytes`` overrides the static
          ``_max_output_chars`` for stdout/stderr truncation in the
          final result.

        Both arguments may be ``None`` when no profile-level
        ``runtime_limits`` is configured — in that case Popen falls
        back to the previous behaviour (no preexec_fn, no timeout,
        static output cap).
        """
        self._cgroup_attach = attach_callback
        self._runtime_limits = limits
        self._trace(
            f"set_runtime_limits: attach={attach_callback is not None} "
            f"limits={limits!r}"
        )

    def set_apparmor_child_transition_callback(
        self,
        callback: Optional[Callable[[], None]],
    ) -> None:
        """Install the AppArmor child-profile transition callback
        (Phase 5 §5.10c).

        Forwarded by
        ``ToolExecutor.set_apparmor_child_transition_callback`` at
        runner-side bootstrap.  When set, the cli plugin's
        ``subprocess.Popen`` ``preexec_fn`` composes this callback
        with the cgroup-attach callback: AppArmor transition FIRST,
        then cgroup attach, then exec.  Order matters — the new
        ``//child`` profile must apply during the cgroup write
        (cgroup writes are allowed in ``//child``, but future
        tightening shouldn't surprise us).

        Closes the verified escape at ``apparmor.py:413-449``: a
        process in ``//child`` cannot write ``changeprofile`` to
        ``/proc/self/attr/current`` (kernel rejects with EACCES).
        Model-controlled subprocess content can no longer escape the
        per-session profile.

        Argument may be ``None`` when the runner isn't AppArmor-
        confined (JAATO_RUNNER_DISABLE_CONFINE=1 or daemon-side
        legacy paths) — Popen falls back to cgroup-only preexec_fn.
        """
        self._apparmor_child_transition = callback
        self._trace(
            f"set_apparmor_child_transition_callback: "
            f"transition={callback is not None}"
        )

    def _build_subprocess_preexec_fn(
        self,
    ) -> Optional[Callable[[], None]]:
        """Phase 5 §5.10c: compose the apparmor + cgroup preexec_fn.

        Returns the appropriate callable for ``Popen(preexec_fn=...)``:

        - When BOTH apparmor transition AND cgroup attach are set:
          returns a composite that runs apparmor first, then cgroup.
        - When only one is set: returns just that one.
        - When neither is set: returns ``None`` (Popen with no
          preexec_fn — today's pre-§5.10 behavior).

        Apparmor-first ordering matches §6.1 of the audit doc — the
        new profile applies during the cgroup write.  Both writes
        succeed today on either profile; ordering is defensive
        against future tightening.

        Both callbacks fail-closed: any exception propagates as a
        Popen spawn failure.  A failed apparmor transition would
        leave the child in the parent profile with the escape rules
        intact — exactly the gap §5.10 closes — so spawn failure is
        the correct posture.
        """
        apparmor_cb = self._apparmor_child_transition
        cgroup_cb = self._cgroup_attach
        if apparmor_cb is None and cgroup_cb is None:
            return None
        if apparmor_cb is None:
            return cgroup_cb
        if cgroup_cb is None:
            return apparmor_cb

        def _composite() -> None:
            apparmor_cb()
            cgroup_cb()

        return _composite

    def shutdown(self) -> None:
        """Shutdown the CLI plugin."""
        self._trace("shutdown: cleaning up")
        self._extra_paths = []
        self._workspace_root = None
        self._initialized = False
        # Cleanup background executor
        self._shutdown_bg_executor()

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset — NO-OP for this plugin.

        Phase 1 hotfix (server 0.6.148+): added to satisfy the
        ``ToolPlugin`` / ``EnrichmentPlugin`` protocol's runtime
        ``isinstance`` check.  Per Daniel's litmus test (see
        ``docs/design/runner-cascade-sharing.md`` §4.3), this
        plugin holds no per-session state that the next cascade
        session would benefit from having cleared.  Override in
        future PRs if the litmus test changes.
        """
        pass


    def get_config_schema(self) -> dict:
        """Return JSON Schema for this plugin's configuration."""
        return {
            "type": "object",
            "properties": {
                "extra_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Additional PATH entries to prepend",
                },
                "max_output_chars": {
                    "type": "integer",
                    "default": 50000,
                    "description": "Maximum characters to return from command output",
                },
                "auto_background_threshold": {
                    "type": "number",
                    "default": 10.0,
                    "description": "Seconds before auto-backgrounding",
                },
                "background_max_workers": {
                    "type": "integer",
                    "default": 4,
                    "description": "Maximum concurrent background workers",
                },
                "scrub_secret_env": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": (
                        "Env-var name globs (case-insensitive fnmatch) to strip "
                        "from the environment of commands run by this tool, so a "
                        "model-driven command cannot read raw credentials the "
                        "runner itself holds (e.g. echo $GITHUB_TOKEN). Empty = "
                        "off (default). Recommended starting set: "
                        "['*_API_KEY','*_TOKEN','*_SECRET','ANTHROPIC_AUTH_TOKEN']."
                    ),
                },
                "workspace_venv": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Path to a workspace-scoped venv to activate for "
                        "commands (empty = off). Relative paths resolve "
                        "against the workspace root. Created if absent with "
                        "--system-site-packages; the model's pip installs "
                        "persist there. Recommended: .jaato/tool-venv"
                    ),
                },
            },
        }

    @classmethod
    def get_apparmor_rules(
        cls,
        *,
        workspace_path: str,
        session_id: str,
        config_root: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> List[str]:
        """Contribute pip's AppArmor rules to the profile.

        The cli tool can run ``pip`` (directly or via the model's shell
        commands): the distro/UA OS-id reads (crashes without them under
        confinement) plus, when a ``workspace_venv`` is set, an ``ix`` grant on
        the venv bin so a bare ``pip`` / console script runs.  Scoped to
        sessions that load ``cli`` — least-privilege.  See ``pip_apparmor_rules``.
        """
        return pip_apparmor_rules(plugin_config.get("workspace_venv"), workspace_path)

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
            },
            category="system",
            discoverability=DISCOVERABILITY_DEFERRED,
            editable=EditableContent(
                parameters=["command"],
                format="text",
                template="# Edit the command below. Save and exit to continue.\n",
            ),
        )]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor mapping.

        Phase 3: forwards via runner-RPC when a runner is attached
        (the canonical wave-1 pattern); falls through to in-process
        otherwise.  This collapses Phase 2's hand-rolled
        ``_execute_via_runner`` indirection in ``_execute`` onto the
        shared ``RunnerForwardingMixin`` — same wire path, same
        cancellation contract, less per-plugin duplication.
        """
        return self.wrap_executors_for_runner_forwarding({
            'cli_based_tool': self._execute,
        })

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
Commands that take longer than 10 seconds may be moved to background execution.
This only happens when a background-task reader tool (`getBackgroundTask`) is
loaded in this session — otherwise the command is simply given longer to finish
and you get its real stdout/stderr, because a task_id you cannot read would be
useless. When a command IS backgrounded, instead of stdout/stderr you receive:
{
    "auto_backgrounded": true,
    "task_id": "abc-123",
    "background_reader_available": true,
    "message": "Task exceeded 10.0s threshold, continuing in background.
                Use getBackgroundTask(task_id='abc-123') to check status and output."
}

Read the "message" field and use the tool it names. The task_id is a handle for
that tool only — it is NOT a file path, and nothing on disk corresponds to it.

If a command outlives even the extended wait with no reader loaded, the call
comes back as an ERROR saying so. That is not a command failure: the command is
still running, you just cannot retrieve its output. Do not assume it succeeded.
Either re-run it so the output lands somewhere you can read (redirect to a file,
then read the file), or report that the profile needs the `background` plugin.

Known slow commands that will be auto-backgrounded:
- Package managers: npm install, pip install, cargo build, mvn install, gradle build
- Build commands: make, cmake, docker build
- Test suites: pytest, npm test, mvn test, cargo test

When a command is auto-backgrounded, use `getBackgroundTask` to monitor it
(the "message" field names it; it is available whenever backgrounding happens):

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

NO INTERACTIVITY — this tool runs commands via subprocess, NOT a PTY/TTY.
It captures stdout/stderr and returns them when the process exits.
It CANNOT handle any form of interactive input during execution:
- Password prompts (ssh, sudo, mysql -p) — the process will hang waiting for input
- REPLs (python, node, psql) — no way to send commands after launch
- Wizards or installers that ask questions (npm init, apt install without -y)
- Debuggers (gdb, pdb) — no interactive stepping possible
- Programs that read from /dev/tty directly

If a command requires ANY back-and-forth interaction, use the shell_spawn /
shell_input tools instead — they provide a real PTY session where you can
read output and send input repeatedly.

ENVIRONMENT VARIABLES:
Commands run in a shell that has access to session environment variables.
When a command needs credentials or tokens, use shell variable references
directly — do NOT attempt to echo, print, or retrieve their values:
- GitHub: `$GITHUB_TOKEN` (e.g., `curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/...`)
- Docker: `$DOCKER_HOST`, `$DOCKER_REGISTRY`
- Cloud: `$AWS_ACCESS_KEY_ID`, `$GCP_PROJECT`, `$AZURE_SUBSCRIPTION_ID`
- CI/CD: `$CI_TOKEN`, `$DEPLOY_KEY`
The user manages which env vars are set. If a command fails due to a missing
variable, report which variable is needed so the user can set it.

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

            # Validate paths are within workspace (if sandboxing enabled).
            # Returns a ready result dict on refusal (blocked path or
            # unparseable command), or None when the command is allowed.
            refusal = self._validate_command_paths(command, arg_list)
            if refusal is not None:
                # Call callbacks with the refusal error
                on_stderr(refusal['stderr'].encode('utf-8'))
                on_returncode(refusal['returncode'])
                return refusal

            # Prepare environment
            env = os.environ.copy()
            if extra_paths:
                path_sep = os.pathsep
                env['PATH'] = env.get('PATH', '') + path_sep + path_sep.join(extra_paths)

            # Activate the workspace venv (if configured) so ``pip install``
            # persists to it and later imports resolve.  Prepends the venv
            # bin ahead of extra_paths so the venv's python/pip win.
            venv_path = resolve_venv_path(self._workspace_venv, self._workspace_root)
            if venv_path:
                ensure_workspace_venv(venv_path)
                apply_venv_to_env(env, venv_path)

            # Secrets-broker scrub (feature #10): strip declared secret vars
            # from the subprocess env so a model-driven command can't read raw
            # credentials the runner itself holds.  No-op when unconfigured.
            if self._scrub_secret_env:
                from shared.secret_scrub import scrub_env as _scrub_secret_env
                env = _scrub_secret_env(env, self._scrub_secret_env)

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

            # Start process with pipes.
            # AppArmor confinement (if any) is inherited from the parent
            # thread via fork+exec — see ToolExecutor.set_apparmor_context.
            # Phase 5 §5.10c — preexec_fn composes two callbacks that
            # run between fork() and exec():
            #   1. AppArmor child-profile transition (writes
            #      ``changeprofile <session>//child`` to
            #      /proc/self/attr/current).  The forked child enters
            #      the ``//child`` sub-profile which drops the escape-
            #      vector rules — model-controlled subprocess content
            #      can't write to attr/current anymore.
            #   2. Cgroup attach (writes the forked child's PID to
            #      cgroup.procs), so the new program comes up under
            #      the session's memory.max / pids.max / cpu.weight.
            # Either callback may be None; the composite handles
            # all four (none, apparmor-only, cgroup-only, both).
            cmd = command if use_shell else argv

            # Diagnostic (server 0.6.108+, 2026-05-16): surface the env
            # preconditions immediately before Popen so we can confirm
            # whether ``HOME`` survives the daemon -> template ->
            # pool-slot -> cli inheritance chain.  v100/v101 cascade
            # evidence showed ``cat ~/...`` failing with the tilde
            # unexpanded — symptom of HOME unset/empty in the
            # subprocess.  Grep across the server code base found no
            # explicit HOME mutation, so the loss (if it really is
            # HOME loss) happens via a runtime path that source
            # inspection alone cannot pinpoint.  This log line is the
            # ground truth at the Popen boundary.
            logger.info(
                "CLI_SUBPROCESS_ENV path=streaming shell=%s cwd=%r "
                "home=%r user=%r path_len=%d env_keys=%d cmd_preview=%r",
                use_shell, self._workspace_root,
                env.get("HOME", "<MISSING>"),
                env.get("USER", "<MISSING>"),
                len(env.get("PATH", "")), len(env),
                command[:80] + ("..." if len(command) > 80 else ""),
            )

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                shell=use_shell,
                cwd=self._workspace_root,
                preexec_fn=self._build_subprocess_preexec_fn(),
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

            # Wait for process and readers to complete.
            # When the session's RuntimeLimits sets a wall-clock cap,
            # we honour it here at the Python layer (cgroup v2 has no
            # equivalent knob).  On expiry: SIGTERM, brief grace, then
            # SIGKILL.  The reader threads observe EOF on the now-closed
            # pipes and exit naturally.
            tool_timeout = (
                self._runtime_limits.tool_timeout_seconds
                if self._runtime_limits is not None
                else None
            )
            timed_out = False
            try:
                proc.wait(timeout=tool_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            stdout_done.wait()
            stderr_done.wait()

            returncode = proc.returncode
            on_returncode(returncode)

            # Combine output for final result
            stdout = b''.join(stdout_chunks).decode('utf-8', errors='replace')
            stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')

            # Per-session override of the static max_output_chars.  When
            # the profile sets ``runtime_limits.max_output_bytes`` it
            # takes precedence; otherwise the plugin's configured cap
            # applies.  Bytes vs chars: we truncate the decoded string
            # by character count, accepting that a multi-byte UTF-8
            # tail might be slightly over the byte budget.  The cap is
            # an order-of-magnitude guardrail, not a hard byte ceiling.
            output_cap = self._max_output_chars
            if (
                self._runtime_limits is not None
                and self._runtime_limits.max_output_bytes is not None
            ):
                output_cap = self._runtime_limits.max_output_bytes

            # Truncate for final result (streaming already captured full output)
            truncated = False
            if len(stdout) > output_cap:
                stdout = stdout[:output_cap]
                truncated = True
            if len(stderr) > output_cap:
                stderr = stderr[:output_cap]
                truncated = True

            result = {'stdout': stdout, 'stderr': stderr, 'returncode': returncode}
            if truncated:
                result['truncated'] = True
                result['truncation_message'] = (
                    f"Output truncated to {output_cap} chars in final result. "
                    "Full output available via getBackgroundTaskOutput."
                )
            if timed_out:
                result['timed_out'] = True
                result['timeout_seconds'] = tool_timeout

            return result

        except Exception as exc:
            return {'error': str(exc)}

    # --- End BackgroundCapable implementation ---

    def _requires_shell(self, command: str) -> bool:
        """Check if a command requires shell interpretation.

        Delegates to :func:`shared.subprocess_runner.requires_shell`.
        """
        return requires_shell(command)

    # --- Path sandboxing implementation ---

    def _extract_path_tokens(self, command: str) -> List[str]:
        """Extract tokens that look like filesystem paths from a command.

        Covers every segment the command would run, including the bodies of
        command substitutions, and includes redirection targets.  Mode
        inference is *not* applied here — see :meth:`_classify_path_modes`
        for that.

        Identifies tokens that are likely filesystem paths:
        - Absolute paths starting with /
        - Relative paths with .. traversal
        - Relative paths starting with ./

        Excludes:
        - URLs (http://, https://, ftp://, etc.)
        - Option flags starting with - or --
        - Package names with @ (npm @scope/package)
        - Heredoc delimiters and file-descriptor duplication targets, which
          are not filesystem paths at all

        Args:
            command: The shell command string.

        Returns:
            List of tokens that appear to be filesystem paths, in source
            order.

        Raises:
            UnanalyzableCommand: When the command cannot be modelled the way
                the shell would parse it.  There is no naive-split fallback:
                one parser decides, and when it can't, callers must deny.
        """
        tokens: List[str] = []
        for segment in analyze_command(command):
            tokens.extend(word for word in segment.words if _path_like(word))
            tokens.extend(
                redirect.target for redirect in segment.redirects
                if redirect.mode != 'none' and _path_like(redirect.target)
            )
        return tokens

    def _is_path_within_workspace(self, path: str, mode: str = "write") -> bool:
        """Check if a path is allowed for access.

        A path is allowed if:
        1. No workspace_root is configured (sandboxing disabled)
        2. The path is within the workspace_root
        3. The path is under .jaato and within the .jaato containment boundary
           (see sandbox_utils.py for .jaato contained symlink escape rules)
        4. The path is authorized via the plugin registry (respecting access mode)

        Handles:
        - Absolute paths
        - Relative paths (resolved against workspace_root)
        - Paths with .. traversal
        - Symlinks (resolved to real path, but .jaato gets special handling)
        - ~ home directory expansion

        Args:
            path: The path to check.
            mode: Access mode - "read" or "write" (default: "write").
                 Callers should pass the mode inferred from the path's role
                 in the command. _classify_path_modes() determines this
                 automatically for _validate_command_paths().

        Returns:
            True if the path is allowed, False otherwise.
        """
        if not self._workspace_root:
            # No sandboxing configured
            return True

        try:
            # Convert MSYS2 drive paths (/c/...) to Windows (C:/...) for Python
            path = msys2_to_windows_path(path)

            # Expand ~ to home directory
            expanded = os.path.expanduser(path)

            # Make absolute relative to workspace_root
            if not os.path.isabs(expanded):
                expanded = os.path.join(self._workspace_root, expanded)

            # Use shared sandbox utility with .jaato containment support
            allowed = check_path_with_jaato_containment(
                expanded,
                self._workspace_root,
                self._plugin_registry,
                mode=mode
            )

            if not allowed:
                self._trace(f"_is_path_within_workspace: {path} blocked (outside sandbox, mode={mode})")
            return allowed

        except (OSError, ValueError):
            # If path resolution fails, treat as outside workspace for safety
            return False

    def _classify_segment(self, segment: Segment) -> List[tuple]:
        """Classify the paths of a single shell segment.

        Each segment is judged on its own command name and its own
        redirections, which is what makes compound commands safe to reason
        about: in ``cat README.md && rm -rf notes/`` the ``rm`` segment
        classifies ``notes/`` as write even though the string starts with
        ``cat``.

        Args:
            segment: One segment from :func:`analyze_command`.

        Returns:
            List of ``(path, mode)`` tuples where mode is "read" or "write".
        """
        cmd_name = _effective_command_name(segment)
        word_paths = [word for word in segment.words if _path_like(word)]
        result = _classify_word_paths(cmd_name, word_paths)

        # Redirection targets carry the mode the operator grants, regardless
        # of what the command itself does.
        for redirect in segment.redirects:
            if redirect.mode == 'none' or not _path_like(redirect.target):
                continue
            result.append((redirect.target, redirect.mode))
        return result

    def _classify_path_modes(
        self,
        command: str,
        arg_list: Optional[List[str]] = None,
    ) -> List[tuple]:
        """Classify each path token in a command as "read" or "write".

        The command is first segmented into the simple commands the shell
        would actually run (see :func:`shared.command_analysis.analyze_command`),
        including the bodies of command substitutions.  Each segment is then
        classified independently and the results are unioned, with "write"
        winning over "read" for a path that appears in both roles.

        Per-segment heuristics (in order of priority):
        1. Redirection targets take the mode the operator grants -- the full
           file-descriptor grammar, not just ``>``/``>>`` (so ``2>f``,
           ``&>f``, ``>&f``, ``<>f``, ``>|f`` are all writes, and heredoc
           delimiters are not paths at all).
        2. All path args of commands in ``_WRITE_ALL_CMDS`` are "write".
        3. The last path arg of commands in ``_WRITE_LAST_CMDS`` is "write".
        4. All path args of commands in ``_WRITE_OUTPUT_CMDS`` are "write".
        5. Everything else defaults to "read".

        Args:
            command: The shell command string.
            arg_list: Optional separate argument list.

        Returns:
            List of ``(path, mode)`` tuples, first-seen order, where mode is
            "read" or "write".

        Raises:
            UnanalyzableCommand: When the command cannot be modelled the way
                the shell would parse it.  Callers must refuse it; see
                :meth:`_validate_command_paths`.
        """
        segments = analyze_command(command)

        modes: Dict[str, str] = {}
        order: List[str] = []

        def record(pairs: List[tuple]) -> None:
            for path, mode in pairs:
                if path not in modes:
                    modes[path] = mode
                    order.append(path)
                elif mode == 'write':
                    modes[path] = 'write'

        for segment in segments:
            record(self._classify_segment(segment))

        if arg_list:
            record(self._classify_arg_list(segments, arg_list))

        return [(path, modes[path]) for path in order]

    def _classify_arg_list(
        self,
        segments: List[Segment],
        arg_list: List[str],
    ) -> List[tuple]:
        """Classify paths supplied through the separate ``args`` list.

        The ``args`` form is never shell-parsed, so its entries are checked
        with the looser :func:`_arg_path_like` filter but classified with the
        same command-name heuristics as inline words.

        Args:
            segments: Segments parsed from the ``command`` string (used only
                to resolve the command name).
            arg_list: The explicit argument list.

        Returns:
            List of ``(path, mode)`` tuples.
        """
        head_words = list(segments[0].words) if segments else []
        args = [str(arg) for arg in arg_list]
        synthetic = Segment(words=head_words + args)
        cmd_name = _effective_command_name(synthetic)
        return _classify_word_paths(
            cmd_name, [arg for arg in args if _arg_path_like(arg)]
        )

    def _validate_command_paths(
        self,
        command: str,
        arg_list: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Validate that all paths in a command are within workspace_root.

        Each path is checked with its inferred access mode: paths in write
        positions (redirections, write commands) require "readwrite"
        authorization; all other paths only require "read" access.

        **Fail-closed contract.** The command is parsed by
        :func:`shared.command_analysis.analyze_command`, which models POSIX
        shell word splitting, command chaining, substitution and the full
        redirection grammar. If it *cannot* parse the command (unbalanced
        quotes, dangling escapes, a redirection with no target, an
        unmodelled redirect operator), the structure it would produce no
        longer matches how ``/bin/sh`` will interpret the string — so the
        path extraction can't reason about it safely. Historically the
        helpers degraded to a naive ``str.split()`` here, which parses
        differently than the shell and could let an out-of-workspace path
        slip past the check. Instead we refuse the command outright. This
        is a security boundary: when we can't analyse, we deny.

        Args:
            command: The command string.
            arg_list: Optional separate argument list.

        Returns:
            ``None`` if all paths are valid; otherwise a ready-to-return
            result dict (``stdout``/``stderr``/``returncode``) describing
            why the command was refused — either a blocked path (mimicking
            "not found") or an unparseable command (shell syntax error).
        """
        if not self._workspace_root:
            # No sandboxing configured
            return None

        # Fail closed: a command the analyzer cannot model the way the shell
        # will is refused, not degraded to a looser parse.
        try:
            classified = self._classify_path_modes(command, arg_list)
        except UnanalyzableCommand as exc:
            self._trace(
                f"path_sandbox: refusing unparseable command for validation ({exc})"
            )
            return self._make_unparseable_result(command, exc)

        for path, mode in classified:
            if not self._is_path_within_workspace(path, mode=mode):
                self._trace(f"path_sandbox: blocked access to '{path}' (outside workspace, mode={mode})")
                return self._make_not_found_result(path, command)

        return None

    def _make_not_found_result(self, path: str, command: str) -> Dict[str, Any]:
        """Create a result dict that mimics "file/directory not found".

        Args:
            path: The path that "doesn't exist".
            command: The original command (used to determine error format).

        Returns:
            Dict with stdout, stderr, returncode mimicking not found error.
        """
        # Extract the base command name for realistic error messages
        try:
            cmd_name = shlex.split(command)[0]
            # Get just the executable name without path
            cmd_name = os.path.basename(cmd_name)
        except (ValueError, IndexError):
            cmd_name = "command"

        # Format error message based on common command patterns
        stderr = f"{cmd_name}: {path}: No such file or directory"

        return {
            'stdout': '',
            'stderr': stderr,
            'returncode': 1
        }

    def _make_unparseable_result(
        self, command: str, exc: Exception
    ) -> Dict[str, Any]:
        """Create a result dict for a command that failed sandbox parsing.

        Returned by :meth:`_validate_command_paths` when ``shlex.split``
        cannot tokenise the command (e.g. unbalanced quotes), so the path
        sandbox can't verify it stays within the workspace. The command is
        refused with a shell-style syntax error (returncode 2, the POSIX
        convention for a shell parse failure) rather than executed.

        The message is explicit so the model can self-correct — typically
        by balancing quotes or simplifying the quoting — instead of seeing
        a misleading "file not found".

        Args:
            command: The original command string (unused beyond context;
                kept for symmetry with :meth:`_make_not_found_result`).
            exc: The ``shlex`` parse error, surfaced to aid correction.

        Returns:
            Dict with stdout, stderr, returncode for a refused command.
        """
        stderr = (
            "sh: syntax error: command could not be parsed for sandbox "
            f"validation ({exc}); rewrite it with balanced quotes/escapes."
        )
        return {
            'stdout': '',
            'stderr': stderr,
            'returncode': 2,
        }

    # --- End path sandboxing implementation ---

    def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a CLI command in-process.

        Exactly one of the following forms should be provided:
        1. command: full shell-like command string (preferred for simplicity).
        2. command + args: command as executable name and args as argument list.

        Shell metacharacters (pipes, redirections, command chaining) are auto-detected
        and the command is executed through the shell when required.

        **Phase 3 dispatch.**  When the session has a runner subprocess
        attached, ``get_executors`` wraps this method with the
        ``RunnerForwardingMixin`` forwarder, which routes the call
        through ``tool.execute`` RPC instead of invoking this body.
        The cli ``subprocess.Popen`` then happens inside the kernel-
        confined runner process so the spawned child inherits the
        per-session AppArmor profile.  When no runner is attached,
        the wrapper falls through to this in-process path.

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

            # Validate paths are within workspace (if sandboxing enabled).
            # Returns a ready result dict on refusal (blocked path or
            # unparseable command), or None when the command is allowed.
            refusal = self._validate_command_paths(command, arg_list)
            if refusal is not None:
                return refusal

            # Merge separate arg_list into the command string so the
            # shared runner receives a single command expression.
            if arg_list and not self._requires_shell(command):
                command = ' '.join(
                    [shlex.quote(command)] + [shlex.quote(a) for a in arg_list]
                )

            # Build extra env for PATH extension
            extra_env: Optional[Dict[str, str]] = None
            if extra_paths:
                path_sep = os.pathsep
                extra_env = {
                    'PATH': os.environ.get('PATH', '')
                    + path_sep + path_sep.join(extra_paths)
                }

            # Activate the workspace venv (if configured) on the FOREGROUND path
            # too — run_command starts from os.environ + extra_env, so seed a
            # full env, activate, and carry the venv-touched keys back into
            # extra_env.  Without this, foreground `python`/`pip` resolve to the
            # runner base venv instead of the tool-venv (parity with the
            # streaming path in _execute_streaming).
            venv_path = resolve_venv_path(self._workspace_venv, self._workspace_root)
            if venv_path:
                ensure_workspace_venv(venv_path)
                seed = dict(os.environ)
                if extra_env:
                    seed.update(extra_env)
                apply_venv_to_env(seed, venv_path)
                extra_env = extra_env or {}
                for _k in ('PATH', 'VIRTUAL_ENV', 'PYTHONPATH'):
                    if _k in seed:
                        extra_env[_k] = seed[_k]

            # Resolve streaming callback
            effective_callback = self._get_effective_output_callback()
            self._trace(f"execute: streaming={'YES' if effective_callback else 'NO'}")

            # Per-session runtime limits (from RuntimeLimits) override
            # the static plugin-level caps when present.  Bare attribute
            # reads guard against profiles that set only kernel limits
            # — limits=None is fine, fields default to None.
            limits = self._runtime_limits
            effective_max_output = self._max_output_chars
            effective_timeout: Optional[float] = None
            if limits is not None:
                if limits.max_output_bytes is not None:
                    effective_max_output = limits.max_output_bytes
                if limits.tool_timeout_seconds is not None:
                    effective_timeout = limits.tool_timeout_seconds

            # Delegate to shared subprocess runner
            r: RunResult = run_command(
                command,
                cwd=self._workspace_root,
                timeout=effective_timeout,
                max_output_chars=effective_max_output,
                extra_env=extra_env,
                on_stdout_line=effective_callback,
                check_cancel=True,
                preexec_fn=self._build_subprocess_preexec_fn(),
                scrub_env=self._scrub_secret_env or None,
            )

            # Executable-not-found is surfaced as an error dict so the
            # model sees a clear actionable message rather than a raw
            # returncode=127 result.
            if r.returncode == 127 and "not found in PATH" in r.stderr:
                return {
                    'error': f"cli_based_tool: {r.stderr}",
                    'hint': 'Configure extra_paths or provide full path to the executable.'
                }

            result: Dict[str, Any] = {
                'stdout': r.stdout,
                'stderr': r.stderr,
                'returncode': r.returncode,
            }

            if r.truncated:
                result['truncated'] = True
                result['truncation_message'] = (
                    f"Output truncated to {self._max_output_chars} chars. "
                    "Consider using more specific commands (e.g., add filters, limits, or pipe to head/tail)."
                )

            # _telemetry: Convention-based telemetry: jaato_session forwards
            # these as span attributes on the enclosing tool_span.
            result['_telemetry'] = {
                'jaato.cli.command': command[:200],
                'jaato.cli.returncode': r.returncode,
                'jaato.cli.stdout_bytes': len(r.stdout),
                'jaato.cli.stderr_bytes': len(r.stderr),
                'jaato.cli.shell_mode': requires_shell(command),
                'jaato.cli.cwd': str(self._workspace_root or ''),
            }

            return result

        except Exception as exc:
            return {'error': str(exc)}


def create_plugin() -> CLIToolPlugin:
    """Factory function to create the CLI plugin instance."""
    return CLIToolPlugin()
