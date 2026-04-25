"""Shared subprocess execution infrastructure.

Provides a low-level ``run_command()`` function used by any plugin that
needs to execute shell commands.  The CLI plugin and the prompt library
plugin both delegate here so the following concerns are handled once:

* **Shell detection** — ``SHELL_METACHAR_PATTERN`` decides when
  ``shell=True`` is required.
* **Cancel token** — the running process is killed when the session's
  ``CancelToken`` fires.
* **Output truncation** — large stdout/stderr is capped to keep the
  model's context window healthy.
* **Encoding** — always UTF-8 with ``errors='replace'``.
* **AppArmor** — confinement is inherited via fork+exec; this module
  does *not* apply profiles, but it documents the contract.

Plugin-specific policies (workspace path validation, streaming
callbacks, background execution, timeout choices) stay in the calling
plugin — this module is deliberately minimal.
"""

import os
import re
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from shared.ai_tool_runner import get_current_cancel_token
from jaato_sdk.plugins.model_provider.types import CancelledException

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default cap on captured stdout/stderr (chars).  ~12k tokens at 4
#: chars/token — enough to be useful without flooding the context window.
DEFAULT_MAX_OUTPUT_CHARS = 50_000

#: Shell metacharacters that force ``shell=True``.
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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a ``run_command()`` invocation.

    Attributes:
        stdout: Captured standard output (may be truncated).
        stderr: Captured standard error (may be truncated).
        returncode: Process exit code.
        truncated: ``True`` when either stream was capped at
                   *max_output_chars*.
        timed_out: ``True`` when the process was killed due to timeout.
        cancelled: ``True`` when the process was killed because the
                   session's ``CancelToken`` fired.
    """
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def requires_shell(command: str) -> bool:
    """Return ``True`` when *command* contains shell metacharacters.

    When metacharacters are present the command must be executed via the
    system shell (``shell=True``); otherwise an argv list is safer.
    """
    return bool(SHELL_METACHAR_PATTERN.search(command))


def _resolve_argv(
    command: str,
    env: Dict[str, str],
) -> Optional[List[str]]:
    """Parse *command* into an argv list and resolve the executable.

    Returns ``None`` if the executable cannot be found on ``PATH``.
    """
    argv = shlex.split(command)

    # Normalise single-string with spaces passed as one token.
    if len(argv) == 1 and ' ' in argv[0]:
        argv = shlex.split(argv[0])

    exe = argv[0]
    resolved = shutil.which(exe, path=env.get('PATH'))
    if resolved is None:
        return None
    argv[0] = resolved
    return argv


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_command(
    command: str,
    *,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    extra_env: Optional[Dict[str, str]] = None,
    on_stdout_line: Optional[Callable[[str], None]] = None,
    check_cancel: bool = True,
    preexec_fn: Optional[Callable[[], None]] = None,
) -> RunResult:
    """Execute *command* and return a :class:`RunResult`.

    This is the single process-execution entry point shared by all
    plugins.  It handles shell detection, cancel-token checking,
    output truncation, and encoding.

    Args:
        command: Shell command string.  Pipes, redirections, and
                 variable expansions are auto-detected — when present
                 the command runs through the system shell.
        cwd: Working directory for the child process.
        timeout: Maximum wall-clock seconds.  ``None`` means no limit.
        max_output_chars: Cap on captured stdout **and** stderr.  Set to
                         ``0`` to disable truncation.
        extra_env: Additional environment variables merged on top of
                   ``os.environ``.
        on_stdout_line: Optional callback invoked for each line of
                        stdout as it arrives (streaming mode).  When
                        ``None`` stdout is read in 4 KiB chunks instead.
        check_cancel: When ``True`` (the default), the current thread's
                      ``CancelToken`` is checked between reads and the
                      process is killed if it fires.
        preexec_fn: Optional zero-arg callable run between fork() and
                    exec() in the child.  Used by the cgroups runtime
                    to attach the child to a per-session cgroup before
                    the new program starts.  ``None`` means no
                    preexec.  POSIX-only (ignored on Windows by
                    ``subprocess`` itself).

    Returns:
        A :class:`RunResult` with captured output and status flags.

    Raises:
        CancelledException: If *check_cancel* is ``True`` and the
                            cancel token fires during execution.
    """

    # ---- environment ----
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # ---- shell vs argv ----
    use_shell = requires_shell(command)

    cmd: object  # str | List[str]
    if use_shell:
        cmd = command
    else:
        argv = _resolve_argv(command, env)
        if argv is None:
            exe = shlex.split(command)[0]
            return RunResult(
                stderr=f"executable '{exe}' not found in PATH",
                returncode=127,
            )
        cmd = argv

    # ---- cancel token ----
    cancel_token = get_current_cancel_token() if check_cancel else None

    # ---- spawn ----
    # AppArmor confinement (if any) is inherited from the parent thread
    # via fork+exec.  Cgroup attach (if any) runs as preexec_fn — moves
    # the forked child into the session's cgroup before exec, so the
    # kernel-enforced memory.max / pids.max / cpu.weight apply from the
    # first instruction of the new program.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
        shell=use_shell,
        cwd=cwd,
        preexec_fn=preexec_fn,
    )

    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    timed_out = False

    # When a timeout is set, a watchdog timer kills the process if the
    # deadline fires — this works regardless of whether stdout.read()
    # is blocking.
    watchdog: Optional[threading.Timer] = None
    if timeout is not None:
        def _kill_on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            try:
                proc.kill()
            except OSError:
                pass  # already dead
        watchdog = threading.Timer(timeout, _kill_on_timeout)
        watchdog.start()

    try:
        if on_stdout_line is not None:
            # Streaming: line-by-line
            if proc.stdout:
                for line in proc.stdout:
                    if timed_out:
                        break
                    if cancel_token is not None and cancel_token.is_cancelled:
                        proc.kill()
                        proc.wait()
                        raise CancelledException(
                            "Cancelled during subprocess output"
                        )
                    stdout_parts.append(line)
                    on_stdout_line(line.rstrip('\n\r'))
        else:
            # Blocking: 4 KiB chunks with cancel checks
            if proc.stdout:
                while True:
                    if timed_out:
                        break
                    if cancel_token is not None and cancel_token.is_cancelled:
                        proc.kill()
                        proc.wait()
                        raise CancelledException(
                            "Cancelled during subprocess execution"
                        )
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    stdout_parts.append(chunk)

        # Read remaining stderr (non-blocking after stdout is done)
        if proc.stderr and not timed_out:
            stderr_parts = proc.stderr.readlines()

        proc.wait()

    finally:
        if watchdog is not None:
            watchdog.cancel()

    stdout = ''.join(stdout_parts)
    stderr = ''.join(stderr_parts)
    returncode = proc.returncode if proc.returncode is not None else -1

    # ---- truncation ----
    truncated = False
    if max_output_chars and len(stdout) > max_output_chars:
        stdout = stdout[:max_output_chars]
        truncated = True
    if max_output_chars and len(stderr) > max_output_chars:
        stderr = stderr[:max_output_chars]
        truncated = True

    return RunResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        truncated=truncated,
        timed_out=timed_out,
    )
