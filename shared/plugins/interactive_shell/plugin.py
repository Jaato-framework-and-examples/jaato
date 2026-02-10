"""Interactive shell plugin for driving user-interactive commands.

Provides tools that let the model spawn persistent PTY sessions and
interact with programs that require back-and-forth input: REPLs, password
prompts, wizards, debuggers, SSH sessions, etc.

Design philosophy: the model reads whatever the process outputs and makes
its own decisions about what to type next. No expect patterns required —
the intelligence is in the model, not the tool.
"""

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional

from ..base import UserCommand
from ..model_provider.types import ToolSchema
from .session import ShellSession, _BACKEND, _BACKEND_ERROR, IS_MSYS2
from .ansi import strip_ansi
from shared.ai_tool_runner import get_current_tool_output_callback


# Maximum concurrent interactive sessions
DEFAULT_MAX_SESSIONS = 8

# Session reaper interval (seconds)
REAPER_INTERVAL = 30.0

# Max idle time before a session is reaped (seconds)
DEFAULT_MAX_IDLE = 300  # 5 minutes


class InteractiveShellPlugin:
    """Plugin that provides interactive shell session management.

    Allows the model to spawn long-lived sessions and drive any
    interactive command by reading output and sending input.

    Platform backends (selected at import time by ``session.py``):

    - **pexpect** (Unix / macOS, or MSYS Python with ``pty``): full PTY.
    - **popen_spawn** (MSYS2 with MINGW Python): ``subprocess.Popen`` pipes.
      No real PTY — child ``isatty()`` is ``False``, terminal dimensions
      are ignored, password-prompt echo control is unavailable — but
      timeout/idle detection works reliably.
    - **wexpect** (native Windows): Windows console APIs + named pipes.

    The active backend is exposed as ``session._BACKEND`` and logged in
    the plugin's trace output at ``initialize()`` and ``spawn()`` time.

    Configuration:
        max_sessions: Maximum concurrent sessions (default: 8).
        max_lifetime: Max session lifetime in seconds (default: 600).
        max_idle: Max idle time before reaping in seconds (default: 300).
        idle_timeout: Seconds of silence for output settling (default: 0.5).
        workspace_root: Working directory for spawned processes.
    """

    def __init__(self):
        self._sessions: Dict[str, ShellSession] = {}
        self._session_counter = 0
        self._lock = threading.Lock()

        # Configuration (set during initialize)
        self._max_sessions = DEFAULT_MAX_SESSIONS
        self._max_idle = DEFAULT_MAX_IDLE
        self._max_lifetime = 600
        self._idle_timeout = 0.5
        self._workspace_root: Optional[str] = None
        self._agent_name: Optional[str] = None
        self._initialized = False
        self._tool_output_callback: Optional[Callable[[str], None]] = None

        # Reaper thread
        self._reaper_thread: Optional[threading.Thread] = None
        self._reaper_stop = threading.Event()

    @property
    def name(self) -> str:
        return "interactive_shell"

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
                    f.write(f"[{ts}] [InteractiveShell{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the interactive shell plugin.

        Args:
            config: Optional dict with:
                - max_sessions: Max concurrent sessions (default: 8)
                - max_lifetime: Session lifetime ceiling in seconds (default: 600)
                - max_idle: Max idle seconds before reaping (default: 300)
                - idle_timeout: Output settling time in seconds (default: 0.5)
                - workspace_root: Working directory for spawned processes
                - agent_name: Agent context for trace logging
        """
        if config:
            self._agent_name = config.get('agent_name')
            if 'max_sessions' in config:
                self._max_sessions = config['max_sessions']
            if 'max_lifetime' in config:
                self._max_lifetime = config['max_lifetime']
            if 'max_idle' in config:
                self._max_idle = config['max_idle']
            if 'idle_timeout' in config:
                self._idle_timeout = config['idle_timeout']
            if 'workspace_root' in config:
                workspace = config['workspace_root']
                if workspace:
                    self._workspace_root = os.path.realpath(
                        os.path.abspath(workspace)
                    )

        self._initialized = True
        self._start_reaper()
        self._trace(
            f"initialize: max_sessions={self._max_sessions}, "
            f"max_lifetime={self._max_lifetime}, "
            f"max_idle={self._max_idle}, "
            f"workspace_root={self._workspace_root}, "
            f"backend={_BACKEND or 'NONE'}, "
            f"msys2={IS_MSYS2}"
        )
        if _BACKEND is None and _BACKEND_ERROR:
            self._trace(f"initialize: WARNING - no backend: {_BACKEND_ERROR}")

    def shutdown(self) -> None:
        """Shutdown the plugin, closing all sessions."""
        self._trace("shutdown: closing all sessions")
        self._stop_reaper()

        with self._lock:
            for session_id in list(self._sessions.keys()):
                try:
                    self._sessions[session_id].close()
                except Exception:
                    pass
            self._sessions.clear()

        self._initialized = False

    def set_workspace_path(self, path: Optional[str]) -> None:
        """Update the workspace root path.

        Args:
            path: The new workspace root path, or None.
        """
        if path:
            self._workspace_root = os.path.realpath(os.path.abspath(path))
        else:
            self._workspace_root = None
        self._trace(f"set_workspace_path: {self._workspace_root}")

    # --- Tool schemas ---

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for all interactive shell tools."""
        return [
            ToolSchema(
                name='shell_spawn',
                description=(
                    'Start a new interactive process and return its session_id. '
                    'Use shell_input(session_id=...) for ALL subsequent '
                    'interactions — calling shell_spawn again creates a '
                    'SEPARATE process, it does NOT send input to this one. '
                    'Use for programs that require interactive input: '
                    'REPLs (python, node, psql), password prompts (ssh, sudo), '
                    'wizards (npm init), debuggers (gdb, pdb).'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The command to run interactively. Examples: "
                                "'python3', 'ssh user@host', 'psql -U admin mydb', "
                                "'npm init', 'gdb ./myprogram'"
                            ),
                        },
                        "session_name": {
                            "type": "string",
                            "description": (
                                "Optional human-readable name for this session. "
                                "Auto-generated if omitted (e.g., 'session_0')."
                            ),
                        },
                        "rows": {
                            "type": "integer",
                            "description": "PTY height in rows (default: 24).",
                        },
                        "cols": {
                            "type": "integer",
                            "description": "PTY width in columns (default: 80).",
                        },
                    },
                    "required": ["command"],
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='shell_input',
                description=(
                    'Send text input to a running interactive session (by '
                    'session_id from shell_spawn) and return whatever the '
                    'program outputs next. This is the ONLY tool for '
                    'interacting with an already-running session — do not '
                    'use shell_spawn to send input.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from shell_spawn.",
                        },
                        "input": {
                            "type": "string",
                            "description": (
                                "Text to type into the session. Include \\n at "
                                "the end to press Enter. For example: "
                                "'yes\\n', 'SELECT * FROM users;\\n', "
                                "'my-project-name\\n'"
                            ),
                        },
                    },
                    "required": ["session_id", "input"],
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='shell_read',
                description=(
                    'Read pending output from a session without sending any '
                    'input. Useful for checking on long-running operations or '
                    'reading output that arrived since the last interaction.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from shell_spawn.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": (
                                "Seconds to wait for output (default: 2). "
                                "Use longer values for slow operations."
                            ),
                        },
                    },
                    "required": ["session_id"],
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='shell_control',
                description=(
                    'Send a control key to an interactive session. '
                    'Use for interrupting (Ctrl+C), sending EOF (Ctrl+D), '
                    'suspending (Ctrl+Z), or clearing screen (Ctrl+L).'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from shell_spawn.",
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Control key to send. Options: "
                                "'c-c' (Ctrl+C, interrupt), "
                                "'c-d' (Ctrl+D, EOF/exit), "
                                "'c-z' (Ctrl+Z, suspend), "
                                "'c-\\\\' (Ctrl+\\\\, quit), "
                                "'c-l' (Ctrl+L, clear)"
                            ),
                            "enum": ["c-c", "c-d", "c-z", "c-\\", "c-l"],
                        },
                    },
                    "required": ["session_id", "key"],
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='shell_close',
                description=(
                    'Close an interactive session. Sends EOF, then SIGTERM, '
                    'then SIGKILL if needed. Returns exit status and any '
                    'final output the program produced while shutting down.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from shell_spawn.",
                        },
                    },
                    "required": ["session_id"],
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='shell_list',
                description=(
                    'List all active interactive sessions with their status, '
                    'command, and age. Useful for keeping track of what is '
                    'running, especially in long conversations.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                },
                category="system",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mapping for all tools."""
        return {
            'shell_spawn': self._exec_spawn,
            'shell_input': self._exec_input,
            'shell_read': self._exec_read,
            'shell_control': self._exec_control,
            'shell_close': self._exec_close,
            'shell_list': self._exec_list,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for interactive shell tools."""
        return """INTERACTIVE SHELL SESSIONS (interactive_shell plugin):

STATE MODEL:
Sessions are STATEFUL and PERSISTENT. shell_spawn creates one process
that remains alive across multiple tool calls. You interact with it
using the session_id returned by shell_spawn. The process remembers
everything — previous input, output, state, working directory, variables.

WORKFLOW:
1. shell_spawn(command=...) → session_id + initial output
2. Read the output. Understand what the program is asking.
3. shell_input(session_id=..., input="response\\n") → next output
4. Repeat 2-3 until done.
5. shell_close(session_id=...)

TOOL ROLES:
- shell_spawn: Start a NEW process (once per command). Returns session_id.
- shell_input: Send input to an EXISTING session. Use for ALL interactions after spawn.
- shell_read: Check for new output without sending input.
- shell_control: Send Ctrl+C, Ctrl+D, etc.
- shell_close: Terminate a session.
- shell_list: See running sessions.

WHEN TO USE vs cli_based_tool:
These tools spawn a real PTY session — use them ONLY when the command
requires interactive back-and-forth input during execution. Do NOT use
shell_spawn for commands that simply run and produce output:
- Non-interactive commands (ls, grep, git status, make) → use cli_based_tool
- Commands with -y/--yes flags or piped stdin → use cli_based_tool
- Shell scripts that run to completion without prompts → use cli_based_tool
- Build commands (npm run build, cargo build, pytest) → use cli_based_tool
Use shell_spawn ONLY when the program will ask questions, prompt for
passwords, or present a REPL that requires you to type responses:
- Password prompts (ssh, sudo) → shell_spawn
- REPLs (python, node, psql) → shell_spawn
- Wizards (npm init, interactive installers) → shell_spawn
- Debuggers (gdb, pdb) → shell_spawn
If unsure → try cli_based_tool first; use shell_spawn only if it hangs
waiting for input that cli_based_tool cannot provide

COMMON MISTAKES:
- Calling shell_spawn to send input to a running session. shell_spawn
  starts a NEW process — use shell_input instead.
- Spawning multiple sessions for the same command. Spawn once, then use
  shell_input with the session_id for all subsequent interactions.
- Losing track of the session_id. Note the session_id from shell_spawn
  and reuse it consistently.

IMPORTANT NOTES:
- Always end text input with \\n — that's pressing Enter
- For password prompts, the terminal won't echo what you type — that's normal
- If a process seems stuck, use shell_read() to check for new output
- Use shell_control(key="c-c") to interrupt, shell_control(key="c-d") for EOF
- Sessions are automatically cleaned up after inactivity or timeout
- Always shell_close() sessions when done to free resources"""

    def get_auto_approved_tools(self) -> List[str]:
        """Interactive shell tools require permission — return empty list."""
        # shell_list and shell_read are read-only but we still require
        # permission because they access active session state.
        return ['shell_list']

    def get_user_commands(self) -> List[UserCommand]:
        """No user-facing commands — all tools are model-invoked."""
        return []

    # --- Executors ---

    def _exec_spawn(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_spawn: start a new interactive session.

        Creates a ``ShellSession`` using the platform-appropriate backend
        (see ``session.py`` module docstring).  On MSYS2 with the
        ``popen_spawn`` backend the child process has no PTY — ``isatty()``
        returns ``False`` and terminal dimensions are not forwarded.  Most
        REPLs and debuggers still work; password prompts may not hide input.
        """
        command = args.get('command')
        if not command:
            return {'error': 'shell_spawn: command is required'}

        session_name = args.get('session_name')
        rows = args.get('rows', 24)
        cols = args.get('cols', 80)

        with self._lock:
            # Check session limit
            alive_count = sum(
                1 for s in self._sessions.values() if s.is_alive
            )
            if alive_count >= self._max_sessions:
                return {
                    'error': (
                        f'Maximum concurrent sessions ({self._max_sessions}) '
                        f'reached. Close an existing session first.'
                    ),
                    'active_sessions': [
                        {'session_id': sid, 'command': s.command}
                        for sid, s in self._sessions.items()
                        if s.is_alive
                    ],
                }

            # Generate session ID
            if session_name:
                session_id = session_name
                # Deduplicate
                if session_id in self._sessions:
                    suffix = 1
                    while f"{session_id}_{suffix}" in self._sessions:
                        suffix += 1
                    session_id = f"{session_id}_{suffix}"
            else:
                session_id = f"session_{self._session_counter}"
                self._session_counter += 1

        self._trace(f"spawn: id={session_id}, cmd={command[:80]}, backend={_BACKEND}")

        try:
            session = ShellSession(
                command=command,
                session_id=session_id,
                rows=rows,
                cols=cols,
                idle_timeout=self._idle_timeout,
                max_lifetime=self._max_lifetime,
                cwd=self._workspace_root,
            )

            # Read initial output (program banner, first prompt, etc.)
            initial_output = session.read_initial_output()

            with self._lock:
                self._sessions[session_id] = session

            self._stream_output(initial_output)

            result = {
                'session_id': session_id,
                'output': strip_ansi(initial_output),
                'is_alive': session.is_alive,
            }

            if not session.is_alive:
                result['note'] = (
                    'Process exited immediately. Check the command and output.'
                )

            self._trace(
                f"spawn: id={session_id} output_len={len(initial_output)} "
                f"alive={session.is_alive}"
            )
            if session.is_alive:
                return (result, {"continuation_id": session_id, "show_output": True})
            return result

        except Exception as exc:
            self._trace(f"spawn: FAILED: {exc}")
            return {'error': f'shell_spawn: {exc}'}

    def _exec_input(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_input: send text and return response."""
        session_id = args.get('session_id')
        text = args.get('input', '')

        if not session_id:
            return {'error': 'shell_input: session_id is required'}

        session = self._get_session(session_id)
        if session is None:
            return self._session_not_found(session_id)

        if not session.is_alive:
            return {
                'error': f'Session {session_id!r} has exited.',
                'is_alive': False,
            }

        self._trace(
            f"input: id={session_id} "
            f"text={text[:50]!r}{'...' if len(text) > 50 else ''}"
        )

        try:
            output = session.send_input(text)
            self._stream_output(output)
            result = {
                'output': strip_ansi(output),
                'is_alive': session.is_alive,
            }
            if session.is_alive:
                return (result, {"continuation_id": session_id, "show_output": False})
            return result
        except Exception as exc:
            self._trace(f"input: id={session_id} FAILED: {exc}")
            return {'error': f'shell_input: {exc}', 'is_alive': session.is_alive}

    def _exec_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_read: read pending output without sending."""
        session_id = args.get('session_id')
        timeout = args.get('timeout', 2.0)

        if not session_id:
            return {'error': 'shell_read: session_id is required'}

        session = self._get_session(session_id)
        if session is None:
            return self._session_not_found(session_id)

        self._trace(f"read: id={session_id} timeout={timeout}")

        try:
            output = session.read_output(timeout=timeout)
            self._stream_output(output)
            result = {
                'output': strip_ansi(output),
                'is_alive': session.is_alive,
            }
            if session.is_alive:
                return (result, {"continuation_id": session_id, "show_output": False, "show_popup": False})
            return result
        except Exception as exc:
            return {'error': f'shell_read: {exc}', 'is_alive': session.is_alive}

    def _exec_control(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_control: send a control key."""
        session_id = args.get('session_id')
        key = args.get('key')

        if not session_id:
            return {'error': 'shell_control: session_id is required'}
        if not key:
            return {'error': 'shell_control: key is required'}

        session = self._get_session(session_id)
        if session is None:
            return self._session_not_found(session_id)

        if not session.is_alive:
            return {
                'error': f'Session {session_id!r} has exited.',
                'is_alive': False,
            }

        self._trace(f"control: id={session_id} key={key}")

        try:
            output = session.send_control(key)
            self._stream_output(output)
            result = {
                'output': strip_ansi(output),
                'is_alive': session.is_alive,
            }
            if session.is_alive:
                return (result, {"continuation_id": session_id, "show_output": False})
            return result
        except ValueError as exc:
            return {'error': str(exc)}
        except Exception as exc:
            return {'error': f'shell_control: {exc}', 'is_alive': session.is_alive}

    def _exec_close(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_close: terminate a session."""
        session_id = args.get('session_id')

        if not session_id:
            return {'error': 'shell_close: session_id is required'}

        session = self._get_session(session_id)
        if session is None:
            return self._session_not_found(session_id)

        self._trace(f"close: id={session_id}")

        try:
            result = session.close()

            with self._lock:
                self._sessions.pop(session_id, None)

            self._trace(
                f"close: id={session_id} exit_status={result.get('exit_status')}"
            )
            return (result, {"show_output": True})

        except Exception as exc:
            # Clean up even on error
            with self._lock:
                self._sessions.pop(session_id, None)
            return {'error': f'shell_close: {exc}'}

    def _exec_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell_list: list active sessions."""
        with self._lock:
            sessions = []
            for sid, session in self._sessions.items():
                sessions.append({
                    'session_id': sid,
                    'command': session.command,
                    'is_alive': session.is_alive,
                    'age_seconds': round(session.age_seconds, 1),
                    'idle_seconds': round(session.idle_seconds, 1),
                })

        return {'sessions': sessions, 'count': len(sessions)}

    # --- Helpers ---

    def set_tool_output_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set the callback for streaming output during tool execution.

        Called by ToolExecutor before/after each tool call. Also supports
        thread-local fallback for parallel execution.
        """
        self._tool_output_callback = callback

    def _stream_output(self, text: str) -> None:
        """Forward raw output text to the tool output callback if available.

        Splits on \\n only (not \\r) to preserve carriage returns and ANSI
        escape sequences within lines for proper terminal emulation by pyte.
        """
        callback = get_current_tool_output_callback() or self._tool_output_callback
        if callback and text:
            parts = text.split('\n')
            # Remove trailing empty from trailing \n (avoids extra blank line)
            if parts and not parts[-1]:
                parts.pop()
            for chunk in parts:
                callback(chunk)

    def _get_session(self, session_id: str) -> Optional[ShellSession]:
        """Look up a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def _session_not_found(self, session_id: str) -> Dict[str, Any]:
        """Return a helpful error for missing sessions."""
        with self._lock:
            available = [
                sid for sid, s in self._sessions.items() if s.is_alive
            ]
        result: Dict[str, Any] = {
            'error': f'No session with id {session_id!r}.',
        }
        if available:
            result['available_sessions'] = available
            result['hint'] = 'Use shell_list to see all sessions.'
        else:
            result['hint'] = 'No active sessions. Use shell_spawn to start one.'
        return result

    # --- Reaper thread ---

    def _start_reaper(self) -> None:
        """Start the background reaper thread."""
        if self._reaper_thread is not None:
            return

        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            daemon=True,
            name="interactive-shell-reaper",
        )
        self._reaper_thread.start()
        self._trace("reaper: started")

    def _stop_reaper(self) -> None:
        """Stop the background reaper thread."""
        self._reaper_stop.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=5.0)
            self._reaper_thread = None
        self._trace("reaper: stopped")

    def _reaper_loop(self) -> None:
        """Periodically check for expired or dead sessions."""
        while not self._reaper_stop.wait(timeout=REAPER_INTERVAL):
            self._reap_sessions()

    def _reap_sessions(self) -> None:
        """Close sessions that are expired, idle, or dead."""
        to_reap = []

        with self._lock:
            for sid, session in self._sessions.items():
                reason = None
                if not session.is_alive:
                    reason = "process exited"
                elif session.is_expired:
                    reason = f"lifetime exceeded ({self._max_lifetime}s)"
                elif session.idle_seconds > self._max_idle:
                    reason = f"idle too long ({self._max_idle}s)"

                if reason:
                    to_reap.append((sid, session, reason))

        for sid, session, reason in to_reap:
            self._trace(f"reaper: closing {sid} ({reason})")
            try:
                session.close()
            except Exception:
                pass
            with self._lock:
                self._sessions.pop(sid, None)


def create_plugin() -> InteractiveShellPlugin:
    """Factory function to create the interactive shell plugin instance."""
    return InteractiveShellPlugin()
