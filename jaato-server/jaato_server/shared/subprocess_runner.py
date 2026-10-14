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

import functools
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from jaato_server.shared.ai_tool_runner import get_current_cancel_token
from jaato_sdk.plugins.model_provider.types import CancelledException

logger = logging.getLogger(__name__)

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


def resolve_executable(
    exe: str,
    search_path: Optional[str],
    cwd: Optional[str],
) -> Optional[str]:
    """Resolve the executable word of an argv-mode command (#1652).

    A bare name is looked up on *search_path*, exactly as before.  A
    path-shaped word (one with a directory component) is resolved against
    *cwd*, the directory the command will run in: ``shutil.which`` ignores
    ``path`` for such a word and tests it relative to the CALLING process's
    cwd, which is the runner's, not the command's.  Containment
    (``command_containment.executable_on_search_path``) already judges a
    relative executable against the subprocess cwd, so resolving it the
    same way is what makes judging and running agree (#1202's rule).

    Args:
        exe: The word in the executable position.
        search_path: The PATH the subprocess runs under.
        cwd: The subprocess working directory; ``None`` keeps the calling
            process's cwd, which is what the subprocess then inherits.

    Returns:
        The path to execute, or ``None`` when nothing executable is found.
    """
    has_dir = os.sep in exe or (os.altsep is not None and os.altsep in exe)
    if has_dir and cwd and not os.path.isabs(exe):
        exe = os.path.join(cwd, exe)
    return shutil.which(exe, path=search_path)


def _resolve_argv(
    command: str,
    env: Dict[str, str],
    cwd: Optional[str] = None,
) -> Optional[List[str]]:
    """Parse *command* into an argv list and resolve the executable.

    Returns ``None`` if the executable cannot be found on ``PATH`` (or, for
    a path-shaped word, relative to *cwd*; see :func:`resolve_executable`).
    """
    argv = shlex.split(command)

    # Normalise single-string with spaces passed as one token.
    if len(argv) == 1 and ' ' in argv[0]:
        argv = shlex.split(argv[0])

    exe = argv[0]
    resolved = resolve_executable(exe, env.get('PATH'), cwd)
    if resolved is None:
        return None
    argv[0] = resolved
    return argv


# ---------------------------------------------------------------------------
# Bounded spawn (#1673)
# ---------------------------------------------------------------------------

#: Seconds a forked child may take to reach ``exec`` (#1673).  A healthy
#: ``preexec_fn`` chain (capability drop, LSM ``//child`` transition,
#: seccomp, cgroup attach) is a handful of syscalls and takes microseconds;
#: the deadline only has to be long enough that a loaded host never trips
#: it.  It is NOT the tool's run-time bound (``tool_timeout_seconds`` is):
#: it bounds the window in which ``subprocess.Popen`` itself blocks.
EXEC_DEADLINE_SECONDS = 30.0


#: What a model-driven subprocess's child runs between ``fork`` and
#: ``exec``, in order (#1673): ``(step, where it is built)``.  ``cli``,
#: ``interactive_shell`` and the notebook kernel receive the first five as
#: one callable (``server.runner.session._child_preexec``); each plugin
#: appends its own last step.  ``jaato-scaffold explain runtime`` renders
#: this table, and the #1673 guard checks every ``where`` still resolves.
#: Rule for every step: no Python-level lock, no logging, no import, no
#: symbol lookup, no builtin ``open`` -- only syscalls on arguments built
#: in the runner, because another runner thread may hold any lock at fork.
CHILD_PREEXEC_STEPS = (
    ("CPython post-fork hooks (os.register_at_fork after_in_child, every "
     "library's)", "subprocess.Popen(preexec_fn=...)"),
    ("drop the capability bounding set (PR_CAPBSET_DROP), while still in "
     "the runner's profile", "capability_drop.CapabilityDrop.before_transition"),
    ("enter the LSM child: AppArmor 'changeprofile <profile>//child' to "
     "/proc/self/attr/current, or SELinux setexeccon",
     "server.runner.lsm_confine.child_transition_callback"),
    ("clear ambient / permitted / effective / inheritable capabilities, "
     "PR_SET_NO_NEW_PRIVS", "capability_drop.CapabilityDrop.after_transition"),
    ("PR_SET_NO_NEW_PRIVS + the seccomp filter",
     "seccomp_filter.CompiledFilter.install"),
    ("cli: cgroup attach ('0' to cgroup.procs); notebook: PR_SET_PDEATHSIG",
     "server.cgroups.procs_attach_callback"),
    ("exec the command; a child not there within the exec deadline is "
     "killed and the call fails", "subprocess_runner.popen_bounded"),
)


class ChildNeverExecd(OSError):
    """A forked child did not reach ``exec`` within the deadline (#1673).

    ``subprocess.Popen`` returns only once the child has exec'd or failed:
    the parent reads the child's exec-status pipe until it closes.  A child
    stuck between ``fork`` and ``exec`` (a lock inherited from another
    thread of a multi-threaded runner, taken by Python code that runs in
    the child before the target program starts) therefore blocks
    ``Popen`` itself, before any ``communicate(timeout=)`` or watchdog has
    started.  :func:`popen_bounded` kills such a child and raises this.

    Attributes:
        pid: The killed child's pid.
        deadline: The deadline it missed, in seconds.
    """

    def __init__(self, pid: int, deadline: float) -> None:
        self.pid = pid
        self.deadline = deadline
        super().__init__(
            f"the child process (pid {pid}) did not reach exec within "
            f"{deadline:g}s after fork and was killed; nothing ran.  It "
            "stopped in the runner's post-fork window (the //child "
            "transition, capability drop and seccomp filter run there), "
            "which is a runner-side fault, not the command's (#1673)")


@functools.lru_cache(maxsize=None)
def _watched_class(popen_cls: type) -> type:
    """A subclass of *popen_cls* that publishes itself before it forks.

    ``Popen._execute_child`` sets ``self.pid`` right after ``fork`` and
    only then blocks reading the exec-status pipe, so the object appended
    to ``box`` here is how :func:`popen_bounded`'s timer learns the pid of
    a child that ``Popen`` has not returned yet.  Built from the
    ``subprocess.Popen`` current at call time, so a test that replaced it
    with a subclass still gets its own class.
    """

    def __init__(self, box, *args, **kwargs):
        box.append(self)
        popen_cls.__init__(self, *args, **kwargs)

    return type("_WatchedPopen", (popen_cls,), {"__init__": __init__})


def popen_bounded(
    args,
    *,
    exec_deadline: Optional[float] = EXEC_DEADLINE_SECONDS,
    **kwargs,
) -> "subprocess.Popen":
    """``subprocess.Popen(args, **kwargs)``, killing a child that never execs.

    A timer armed before the spawn SIGKILLs the child if ``Popen`` has not
    returned when it fires.  The kill closes the child's end of the
    exec-status pipe, ``Popen`` returns, the child is reaped and
    :class:`ChildNeverExecd` is raised, so the caller reports a failed
    call instead of blocking forever.  Only ``preexec_fn`` spawns can stop
    there in Python code, but every spawn is bounded the same way.

    Stated limit: a kill landing between the child's ``exec`` and
    ``Popen`` returning (a few bytecodes) would be reported as a child
    that never exec'd.  The deadline makes that window irrelevant in
    practice, and the alternative is no bound.

    Args:
        args: Passed to ``Popen``.
        exec_deadline: Seconds the child may take to exec.  ``None`` or a
            non-positive value disables the bound (a plain ``Popen``).
        **kwargs: Passed to ``Popen``.

    Raises:
        ChildNeverExecd: The child was killed before it exec'd.
    """
    popen_cls = subprocess.Popen
    if not exec_deadline or exec_deadline <= 0 or not isinstance(popen_cls, type):
        # No bound asked for, or Popen is a test's stand-in function.
        return popen_cls(args, **kwargs)
    box: List[subprocess.Popen] = []
    guard = threading.Lock()
    state = {"returned": False, "killed": None}

    def _kill_if_still_spawning() -> None:
        with guard:
            if state["returned"] or not box:
                return
            pid = box[0].pid
            if pid is None:
                return  # still before fork(): nothing to kill
            state["killed"] = pid
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    timer = threading.Timer(exec_deadline, _kill_if_still_spawning)
    timer.daemon = True
    timer.start()
    try:
        proc = _watched_class(popen_cls)(box, args, **kwargs)
    finally:
        with guard:
            state["returned"] = True
        timer.cancel()
    if state["killed"] is not None:
        _discard(proc)
        logger.error("SPAWN_NEVER_EXECD pid=%s deadline=%ss args=%r",
                     state["killed"], exec_deadline, args)
        raise ChildNeverExecd(state["killed"], exec_deadline)
    return proc


def _discard(proc: "subprocess.Popen") -> None:
    """Reap a killed child and close the pipes ``Popen`` opened for it."""
    proc.wait()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


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
    scrub_env: Optional[Sequence[str]] = None,
    scrub_keep: Optional[Iterable[str]] = None,
    exec_deadline: Optional[float] = EXEC_DEADLINE_SECONDS,
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
        scrub_env: Secret env-var name globs removed from the child's env
                   (see :mod:`shared.secret_scrub`).  ``None`` = no scrub.
        scrub_keep: Exact, case-sensitive names that survive *scrub_env*
                    (the daemon's ``app://`` grants).  ``None`` = keep
                    nothing; the caller decides, not this runner.
        exec_deadline: Seconds the child may take between ``fork`` and
                       ``exec`` before it is killed (:func:`popen_bounded`,
                       #1673).  ``None`` disables the bound.

    Returns:
        A :class:`RunResult` with captured output and status flags.

    Raises:
        CancelledException: If *check_cancel* is ``True`` and the
                            cancel token fires during execution.
        ChildNeverExecd: The child did not reach ``exec`` within
                         *exec_deadline* and was killed (#1673).
    """

    # ---- environment ----
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # Secrets-broker scrub (feature #10): remove declared secret vars from the
    # env handed to this model-driven subprocess, so a shell command can't read
    # raw credentials the runner itself legitimately holds.  No-op when unset.
    if scrub_env:
        from jaato_server.shared.secret_scrub import scrub_env as _scrub_secret_env
        env = _scrub_secret_env(env, scrub_env, keep=scrub_keep)

    # ---- shell vs argv ----
    use_shell = requires_shell(command)

    cmd: object  # str | List[str]
    if use_shell:
        cmd = command
    else:
        argv = _resolve_argv(command, env, cwd)
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
    # Diagnostic (server 0.6.108+, 2026-05-16): see comment in
    # cli/plugin.py:_execute_streaming for the v100/v101 HOME-unset
    # investigation that motivated this log line.  Two Popen sites
    # share this diagnostic — ``path=`` distinguishes them.
    logger.info(
        "CLI_SUBPROCESS_ENV path=run_command shell=%s cwd=%r "
        "home=%r user=%r path_len=%d env_keys=%d cmd_preview=%r",
        use_shell, cwd,
        env.get("HOME", "<MISSING>"),
        env.get("USER", "<MISSING>"),
        len(env.get("PATH", "")), len(env),
        command[:80] + ("..." if len(command) > 80 else ""),
    )
    proc = popen_bounded(
        cmd,
        exec_deadline=exec_deadline,
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
