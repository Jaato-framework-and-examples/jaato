"""Shell session wrapper around pexpect with idle-based output detection.

Instead of requiring the caller to specify expect patterns, this module
uses idle detection: it reads until the process stops producing output
for a configurable period. This lets the calling model read whatever
appeared and make its own decisions about what to send next.
"""

import os
import time
import threading
from typing import Optional, Dict, Any

import pexpect

from .ansi import strip_ansi


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
    """Wraps a single pexpect-spawned process with idle-based I/O.

    The key insight: instead of expect(pattern), we use read_until_idle()
    which returns all output the process produced until it goes quiet.
    The model then reads that output and decides what to do.
    """

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
        if env:
            spawn_env.update(env)

        self._process = pexpect.spawn(
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

        Sends EOF, waits briefly, then escalates to SIGTERM and SIGKILL.

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
                except (pexpect.EOF, OSError):
                    pass

            if self._process.isalive():
                try:
                    self._process.terminate(force=False)  # SIGTERM
                    self._process.wait()
                except Exception:
                    pass

            if self._process.isalive():
                try:
                    self._process.terminate(force=True)  # SIGKILL
                except Exception:
                    pass

        exit_status = self._process.exitstatus
        signal_status = self._process.signalstatus

        return {
            'exit_status': exit_status if exit_status is not None else signal_status,
            'final_output': final_output,
        }

    def _read_until_idle(
        self,
        idle_timeout: Optional[float] = None,
        max_wait: Optional[float] = None,
    ) -> str:
        """Read output until the process stops producing it.

        Uses adaptive idle detection: reads in short bursts and considers
        output "settled" when no new data arrives for idle_timeout seconds.

        Args:
            idle_timeout: Seconds of silence before output is considered settled.
                Defaults to self.idle_timeout.
            max_wait: Hard ceiling on total wait time.
                Defaults to self.max_wait.

        Returns:
            Cleaned text (ANSI stripped) of all output since last read.
        """
        idle_timeout = idle_timeout if idle_timeout is not None else self.idle_timeout
        max_wait = max_wait if max_wait is not None else self.max_wait
        deadline = time.time() + max_wait
        chunks = []
        total_bytes = 0

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
            except pexpect.TIMEOUT:
                # No data for idle_timeout — output has settled
                break
            except pexpect.EOF:
                # Process exited
                break

        raw = ''.join(chunks)
        return strip_ansi(raw)
