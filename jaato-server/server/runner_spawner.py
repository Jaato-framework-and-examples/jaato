"""Daemon-side fork+exec of the per-session runner subprocess.

Implements the spawn half of §4.6 ("Spawn:") in
``docs/design/per_session_confined_runner.md``.

The session manager calls :meth:`RunnerSpawner.spawn` AFTER
``_run_pre_initialize_hooks`` provisions the AppArmor profile and
BEFORE ``JaatoServer.initialize()`` configures plugins.  By that
point the kernel has the per-session profile loaded, so the runner's
``aa_change_profile`` (in :mod:`server.runner.bootstrap`) finds it.

The spawner enforces the §6.1 "fork-inherits-apparmor" risk
mitigation: it reads ``/proc/self/attr/current`` immediately before
``os.fork()`` and refuses to spawn if the daemon thread is confined
to anything other than ``unconfined`` (Phase 5 default; Phase 6 may
swap in a daemon-side narrow profile that grants
``change_profile -> jaato-ws-*`` for every loaded session profile,
in which case this check broadens to "must permit transition to
jaato-ws-*").
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


_PROC_ATTR_PATH = "/proc/self/attr/current"


class DaemonConfinementError(RuntimeError):
    """Raised when the daemon is confined at the moment of fork.

    Phase 2 task 2.1 removed all daemon-side confinement, so reaching
    this error means a regression somewhere — almost certainly a
    re-introduced ``set_apparmor_confinement(...)`` or a
    ``apparmor_confine`` context manager wrapping fork-spawn-and-confine
    (see §4.6 daemon apparmor-state constraint).  Loud-fail rather
    than silently spawning a runner that can't self-confine.
    """


@dataclass
class SpawnedRunner:
    """Daemon-side handle for a spawned runner subprocess.

    Owned by the session's :class:`server.core.JaatoServer`; closed
    via :meth:`server.runner_rpc_client.RunnerRPCClient.close` at session
    end (which closes the parent socket end, waits for the runner
    to exit, then SIGTERM/SIGKILL escalates per §4.6 "Death — daemon
    shutdown").
    """

    pid: int
    parent_socket: socket.socket
    profile_name: str
    session_id: str


class RunnerSpawner:
    """Forks ``python -m server.runner`` once per top-level session.

    Phase 2 design:
    - One ``RunnerSpawner`` per daemon (instantiated in
      ``server/__main__.py`` at startup).
    - One ``SpawnedRunner`` per top-level session.  Subagent sessions
      sharing the parent's runner per §4.3 default do NOT trigger
      another spawn — that's enforced at the call site
      (``SessionManager.create_session``), not here.
    """

    def __init__(self) -> None:
        self._python_executable = sys.executable

    def spawn(
        self,
        *,
        profile_name: str,
        session_id: str,
        workspace_path: Optional[str],
        log_path: Optional[str] = None,
        max_output_chars: Optional[int] = None,
        tool_timeout_seconds: Optional[float] = None,
        disable_confine: bool = False,
    ) -> SpawnedRunner:
        """Fork+exec a runner; return the daemon-side handle.

        Args:
            profile_name: AppArmor profile the runner self-confines
                to.  Must already be loaded in the kernel.  Required
                unless *disable_confine* is set.
            session_id: For env propagation + log attribution.
            workspace_path: Working directory for cli subprocesses
                spawned inside the runner.
            log_path: If set, the runner's stdout/stderr are
                redirected to this file (per plan §5.1).  When
                ``None`` the runner inherits the daemon's stdout/
                stderr; the runner module's logging.basicConfig
                falls back to inherited stderr.
            max_output_chars: cli output cap (env passthrough).
            tool_timeout_seconds: cli wall-clock cap (env passthrough).
            disable_confine: developer escape hatch matching the
                runner's ``JAATO_RUNNER_DISABLE_CONFINE`` env;
                spec §5 — NOT a supported deployment.

        Raises:
            DaemonConfinementError: daemon thread is confined at the
                moment of fork (§6.1 mitigation tripped).
            ValueError: ``profile_name`` is empty and confinement
                isn't disabled.
            OSError: socketpair / fork / exec failed.
        """
        if not profile_name and not disable_confine:
            raise ValueError(
                "RunnerSpawner.spawn: profile_name required (set "
                "disable_confine=True for the developer escape hatch)"
            )

        self._assert_daemon_unconfined()

        parent_sock, child_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM,
        )

        env = self._build_env(
            profile_name=profile_name,
            session_id=session_id,
            workspace_path=workspace_path,
            log_path=log_path,
            max_output_chars=max_output_chars,
            tool_timeout_seconds=tool_timeout_seconds,
            disable_confine=disable_confine,
        )

        pid = os.fork()
        if pid == 0:
            # ----- child -----
            try:
                self._exec_runner(child_sock, parent_sock, log_path, env)
            except BaseException:  # noqa: BLE001 — child must never return
                # Any failure pre-exec lands us here.  os._exit(127) so
                # the parent sees a non-zero status it can attribute.
                os._exit(127)
            # exec replaces the process; reaching here is impossible.
            os._exit(127)

        # ----- parent -----
        child_sock.close()
        logger.info(
            "RunnerSpawner: spawned pid=%d for session %s (profile=%s)",
            pid, session_id, profile_name,
        )
        return SpawnedRunner(
            pid=pid,
            parent_socket=parent_sock,
            profile_name=profile_name,
            session_id=session_id,
        )

    # ----------------------------- helpers ------------------------------

    def _assert_daemon_unconfined(self) -> None:
        """Refuse to spawn if the daemon is confined to a per-session
        AppArmor profile (the §6.1 silent-failure trap).

        Per §4.6 daemon apparmor-state constraint, the daemon must
        permit the runner's ``aa_change_profile`` transition.  Any
        per-session ``jaato-ws-*`` profile lacks
        ``change_profile -> jaato-ws-*`` rules for OTHER sessions, so
        a daemon thread confined to one session's profile cannot fork
        a runner that confines to a different session.

        The check is **specifically** for the ``jaato-ws-*`` prefix
        (the only known silent-failure pattern after Phase 2 task
        2.1) rather than "must equal unconfined" — kernel-default
        placeholders like ``kernel\\x00`` are observed on hosts
        without an apparmor policy, and a future Phase 6 narrow
        daemon profile may legitimately confine the daemon to a
        non-``unconfined`` value that DOES grant the transition.

        The runner's own bootstrap step 3
        (:func:`server.runner.bootstrap.confine_to_profile`) is the
        load-bearing assertion — ANY post-fork transition failure
        surfaces there with a clear
        :class:`ConfinementMismatchError`.  This pre-fork check is
        defense in depth for the one case we know presents silently.
        """
        try:
            with open(_PROC_ATTR_PATH, "r") as f:
                current = f.read().rstrip("\n").rstrip("\x00")
        except OSError:
            # Non-Linux or apparmor-less host: nothing to check.
            return

        if current.startswith("jaato-ws-"):
            raise DaemonConfinementError(
                f"daemon thread is confined to {current!r} at the moment "
                f"of RunnerSpawner.spawn; per design §4.6 daemon "
                f"apparmor-state constraint, the daemon must NOT be "
                f"confined to a per-session profile so the runner can "
                f"transition to ANY per-session profile.  Likely cause: "
                f"a re-introduced daemon-side set_apparmor_confinement / "
                f"apparmor_confine call.  See Phase 2 task 2.1."
            )

    def _build_env(
        self,
        *,
        profile_name: str,
        session_id: str,
        workspace_path: Optional[str],
        log_path: Optional[str],
        max_output_chars: Optional[int],
        tool_timeout_seconds: Optional[float],
        disable_confine: bool,
    ) -> Dict[str, str]:
        """Compose the env dict the runner reads at startup."""
        env = os.environ.copy()
        env["JAATO_RUNNER_PROFILE"] = profile_name
        env["JAATO_RUNNER_SESSION_ID"] = session_id
        if workspace_path:
            env["JAATO_RUNNER_WORKSPACE"] = workspace_path
        if log_path:
            env["JAATO_RUNNER_LOG_PATH"] = log_path
        if max_output_chars is not None:
            env["JAATO_RUNNER_MAX_OUTPUT_CHARS"] = str(max_output_chars)
        if tool_timeout_seconds is not None:
            env["JAATO_RUNNER_TOOL_TIMEOUT_SECONDS"] = str(tool_timeout_seconds)
        if disable_confine:
            env["JAATO_RUNNER_DISABLE_CONFINE"] = "1"
        return env

    def _exec_runner(
        self,
        child_sock: socket.socket,
        parent_sock: socket.socket,
        log_path: Optional[str],
        env: Dict[str, str],
    ) -> None:
        """Child-side: dup socket → fd 3, optionally redirect 1+2 to log,
        close inherited fds, exec the runner.

        Never returns — exec replaces the process.  Any pre-exec
        failure raises and the caller (``spawn``'s child branch)
        ``os._exit(127)``s.
        """
        # Close the parent end first so an exec failure doesn't leak it.
        parent_sock.close()

        # dup the child socket end onto fd 3.  ``os.dup2`` clears
        # CLOEXEC on the target, so the runner inherits fd 3 across
        # exec.  ``set_inheritable`` is belt-and-braces.
        os.dup2(child_sock.fileno(), 3)
        os.set_inheritable(3, True)
        # Don't close the original fd via the socket object — we've
        # already moved it to fd 3 and the original may be > 3.

        # Optional fd 1 / fd 2 redirection (per plan §5.1).
        if log_path:
            try:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                log_fd = os.open(
                    log_path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o600,
                )
                os.dup2(log_fd, 1)
                os.dup2(log_fd, 2)
                if log_fd > 2:
                    os.close(log_fd)
            except OSError:
                # Logging redirect is best-effort; fall through to
                # inherited stderr/stdout.  The runner's log says
                # which mode it's in via _setup_logging.
                pass

        # Close every other inherited fd (§6.2 fd-inheritance risk).
        # The runner only needs fd 0 (stdin, normally /dev/null), 1,
        # 2, and 3.  ``os.closerange`` is fast and stops at the soft
        # rlimit, which is fine.
        try:
            soft_limit = os.sysconf("SC_OPEN_MAX")
        except (AttributeError, OSError, ValueError):
            soft_limit = 1024
        os.closerange(4, max(int(soft_limit), 4))

        os.execvpe(
            self._python_executable,
            [self._python_executable, "-m", "server.runner"],
            env,
        )
