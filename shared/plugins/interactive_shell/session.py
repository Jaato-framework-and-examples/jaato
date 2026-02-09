"""Shell session wrapper around pexpect/wexpect with idle-based output detection.

Instead of requiring the caller to specify expect patterns, this module
uses idle detection: it reads until the process stops producing output
for a configurable period. This lets the calling model read whatever
appeared and make its own decisions about what to send next.

On Unix/macOS this uses pexpect (PTY-based). On Windows this uses wexpect,
which provides a similar API backed by Windows console and named pipes.
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, Any

from .ansi import strip_ansi

IS_WINDOWS = sys.platform == "win32"

# Backend function/exception references.  On Windows these are populated
# lazily so the module can be imported (and the plugin registered) even
# when wexpect is not yet installed.
_spawn = None
_TIMEOUT = None
_EOF = None
_BACKEND_ERROR: Optional[str] = None

if IS_WINDOWS:
    # wexpect has a bare `import pkg_resources` at module level (no
    # try/except guard).  On Python 3.12+ pkg_resources is only available
    # when setuptools is installed, and even then it may be missing in
    # stripped-down venvs.  Inject a minimal stub into sys.modules so the
    # import succeeds — wexpect only uses it for version detection and
    # already has its own fallback for that.
    try:
        import pkg_resources  # noqa: F401 — test if it's available
    except ImportError:
        import types as _types
        _stub = _types.ModuleType("pkg_resources")
        _stub.require = lambda *a, **kw: (_ for _ in ()).throw(  # type: ignore[attr-defined]
            Exception("pkg_resources stub")
        )
        sys.modules["pkg_resources"] = _stub

    try:
        import wexpect
        _spawn = wexpect.spawn
        _TIMEOUT = wexpect.TIMEOUT
        _EOF = wexpect.EOF
    except ImportError as _exc:
        if "pkg_resources" in str(_exc):
            _BACKEND_ERROR = (
                "wexpect failed to import because pkg_resources is missing. "
                "On Python 3.12+ pkg_resources is no longer bundled by default. "
                "Install it with: pip install setuptools wexpect"
            )
        else:
            _BACKEND_ERROR = (
                "wexpect is required for interactive shell sessions on Windows. "
                "Install it with: pip install wexpect"
            )
else:
    import pexpect
    _spawn = pexpect.spawn
    _TIMEOUT = pexpect.TIMEOUT
    _EOF = pexpect.EOF


# Default PTY dimensions
DEFAULT_ROWS = 24
DEFAULT_COLS = 80

# Default idle detection: how long (seconds) the process must be silent
# before we consider its output "settled" and return it to the caller.
DEFAULT_IDLE_TIMEOUT = 0.5

# Hard ceiling on how long to wait for output in a single read call.
DEFAULT_MAX_WAIT = 30.0

# Maximum output buffer size per session (bytes).
DEFAULT_MAX_BUFFER = 64 * 1024

# Default session lifetime ceiling (seconds).
DEFAULT_MAX_LIFETIME = 600  # 10 minutes


class ShellSession:
    """Wraps a single pexpect/wexpect-spawned process with idle-based I/O.

    The key insight: instead of expect(pattern), we use read_until_idle()
    which returns all output the process produced until it goes quiet.
    The model then reads that output and decides what to do.

    On Unix, uses pexpect with real PTY sessions.
    On Windows, uses wexpect with Windows console pipes.
    """

    # Polling interval for wexpect's non-blocking reads (seconds).
    _POLL_INTERVAL = 0.05

    def __init__(
        self,
        command: str,
        session_id: str,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        max_wait: float = DEFAULT_MAX_WAIT,
        max_lifetime: float = DEFAULT_MAX_LIFETIME,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        if _spawn is None:
            raise ImportError(_BACKEND_ERROR or "No PTY backend available")

        self.session_id = session_id
        self.command = command
        self.idle_timeout = idle_timeout
        self.max_wait = max_wait
        self.max_lifetime = max_lifetime
        self.created_at = time.time()
        self.last_interaction = time.time()

        # Merge extra env vars with current environment
        spawn_env = os.environ.copy()
        # Disable pager programs that would block
        spawn_env['PAGER'] = 'cat'
        spawn_env['GIT_PAGER'] = 'cat'
        # Force dumb terminal to reduce escape sequences
        spawn_env['TERM'] = 'dumb'
        # Prevent spawned shells from writing to the user's history file
        spawn_env['HISTFILE'] = ''
        spawn_env['HISTSIZE'] = '0'
        spawn_env['SAVEHIST'] = '0'
        spawn_env['fish_history'] = ''
        if env:
            spawn_env.update(env)

        if IS_WINDOWS:
            # wexpect uses codepage instead of encoding, and doesn't
            # support the dimensions parameter.  codepage=65001 → UTF-8.
            self._process = _spawn(
                command,
                timeout=max_wait,
                env=spawn_env,
                cwd=cwd,
                codepage=65001,
            )
        else:
            self._process = _spawn(
                command,
                encoding='utf-8',
                timeout=max_wait,
                dimensions=(rows, cols),
                env=spawn_env,
                cwd=cwd,
            )

        # Lock for thread-safe access to the process
        self._lock = threading.Lock()

    @property
    def is_alive(self) -> bool:
        """Check if the underlying process is still running."""
        return self._process.isalive()

    @property
    def age_seconds(self) -> float:
        """Seconds since this session was created."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Seconds since the last interaction (send or read)."""
        return time.time() - self.last_interaction

    @property
    def is_expired(self) -> bool:
        """Whether this session has exceeded its max lifetime."""
        return self.age_seconds > self.max_lifetime

    def read_initial_output(self) -> str:
        """Read initial output after spawning, using idle detection.

        Called once right after spawn to capture the program's first
        output (banner, prompt, etc.).

        Returns:
            Clean text output from the process.
        """
        return self._read_until_idle()

    def send_input(self, text: str) -> str:
        """Send text input to the process and return the response.

        Sends the text (which should include \\n if Enter is intended),
        then waits for the process to settle via idle detection.

        Args:
            text: Text to send to the process. Include \\n for Enter.

        Returns:
            Clean text output that appeared after sending the input.
        """
        with self._lock:
            self.last_interaction = time.time()
            self._process.send(text)
            return self._read_until_idle()

    def send_control(self, key: str) -> str:
        """Send a control character to the process.

        Args:
            key: Control key identifier. Supported:
                "c-c" or "c": Ctrl+C (SIGINT)
                "c-d" or "d": Ctrl+D (EOF)
                "c-z" or "z": Ctrl+Z (SIGTSTP)
                "c-\\\\" or "\\\\": Ctrl+\\ (SIGQUIT)
                "c-l" or "l": Ctrl+L (clear screen)

        Returns:
            Clean text output that appeared after the control signal.
        """
        control_map = {
            'c-c': '\x03',
            'c': '\x03',
            'c-d': '\x04',
            'd': '\x04',
            'c-z': '\x1a',
            'z': '\x1a',
            'c-\\': '\x1c',
            '\\': '\x1c',
            'c-l': '\x0c',
            'l': '\x0c',
        }

        char = control_map.get(key)
        if char is None:
            raise ValueError(
                f"Unknown control key: {key!r}. "
                f"Supported: {', '.join(sorted(control_map.keys()))}"
            )

        with self._lock:
            self.last_interaction = time.time()
            self._process.send(char)
            return self._read_until_idle()

    def read_output(self, timeout: Optional[float] = None) -> str:
        """Non-blocking read of any pending output.

        Useful for checking on long-running operations without sending
        input, or reading output that arrived between tool calls.

        Args:
            timeout: How long to wait for output. Defaults to idle_timeout.

        Returns:
            Clean text output, possibly empty if nothing new.
        """
        with self._lock:
            self.last_interaction = time.time()
            return self._read_until_idle(
                idle_timeout=timeout or self.idle_timeout
            )

    def close(self) -> Dict[str, Any]:
        """Gracefully terminate the session.

        On Unix: sends EOF, waits, then escalates to SIGTERM and SIGKILL.
        On Windows: sends EOF, waits, then calls terminate().

        Returns:
            Dict with exit_status and final_output.
        """
        final_output = ""

        with self._lock:
            if self._process.isalive():
                try:
                    # Try graceful EOF first
                    self._process.sendeof()
                    # Read any final output
                    final_output = self._read_until_idle(idle_timeout=1.0)
                except (_EOF, OSError):
                    pass

            if self._process.isalive():
                try:
                    self._process.terminate(force=False)
                    self._process.wait()
                except Exception:
                    pass

            if self._process.isalive():
                try:
                    self._process.terminate(force=True)
                except Exception:
                    pass

        exit_status = self._process.exitstatus
        # signalstatus is Unix-only (not available in wexpect)
        signal_status = getattr(self._process, 'signalstatus', None)

        return {
            'exit_status': exit_status if exit_status is not None else signal_status,
            'final_output': strip_ansi(final_output),
        }

    def _read_until_idle(
        self,
        idle_timeout: Optional[float] = None,
        max_wait: Optional[float] = None,
    ) -> str:
        """Read output until the process stops producing it.

        Uses adaptive idle detection: reads in short bursts and considers
        output "settled" when no new data arrives for idle_timeout seconds.

        On Unix (pexpect), ``read_nonblocking(size, timeout)`` handles the
        idle wait internally.  On Windows (wexpect), ``read_nonblocking``
        is truly non-blocking (no *timeout* parameter), so we poll with
        a short sleep and track the idle interval ourselves.

        Args:
            idle_timeout: Seconds of silence before output is considered settled.
                Defaults to self.idle_timeout.
            max_wait: Hard ceiling on total wait time.
                Defaults to self.max_wait.

        Returns:
            Raw text output since last read (may contain ANSI escape sequences).
            Callers should use strip_ansi() when preparing output for the model.
        """
        idle_timeout = idle_timeout if idle_timeout is not None else self.idle_timeout
        max_wait = max_wait if max_wait is not None else self.max_wait
        deadline = time.time() + max_wait
        chunks: list[str] = []
        total_bytes = 0

        if IS_WINDOWS:
            return self._read_until_idle_windows(
                idle_timeout, deadline, chunks, total_bytes,
            )
        else:
            return self._read_until_idle_unix(
                idle_timeout, deadline, chunks, total_bytes,
            )

    def _read_until_idle_unix(
        self,
        idle_timeout: float,
        deadline: float,
        chunks: list[str],
        total_bytes: int,
    ) -> str:
        """pexpect path: read_nonblocking handles the idle wait internally."""
        while time.time() < deadline:
            try:
                chunk = self._process.read_nonblocking(
                    size=4096,
                    timeout=idle_timeout,
                )
                if chunk:
                    chunks.append(chunk)
                    total_bytes += len(chunk)
                    # Safety: cap buffer to prevent runaway accumulation
                    if total_bytes > DEFAULT_MAX_BUFFER:
                        break
            except _TIMEOUT:
                # No data for idle_timeout — output has settled
                break
            except _EOF:
                # Process exited
                break

        return ''.join(chunks)

    def _read_until_idle_windows(
        self,
        idle_timeout: float,
        deadline: float,
        chunks: list[str],
        total_bytes: int,
    ) -> str:
        """wexpect path: read_nonblocking is instant, so we poll manually."""
        last_data_time = time.time()

        while time.time() < deadline:
            try:
                chunk = self._process.read_nonblocking(size=4096)
                if chunk:
                    # Ensure we always have str (wexpect may return bytes
                    # depending on version/codepage configuration)
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode('utf-8', errors='replace')
                    chunks.append(chunk)
                    total_bytes += len(chunk)
                    last_data_time = time.time()
                    if total_bytes > DEFAULT_MAX_BUFFER:
                        break
                else:
                    # Empty read — check idle elapsed
                    if time.time() - last_data_time >= idle_timeout:
                        break
                    time.sleep(self._POLL_INTERVAL)
            except _TIMEOUT:
                # No data available right now — check idle elapsed
                if time.time() - last_data_time >= idle_timeout:
                    break
                time.sleep(self._POLL_INTERVAL)
            except _EOF:
                break

        return ''.join(chunks)
