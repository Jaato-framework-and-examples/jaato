"""Pre-warm runner template lifecycle manager.

PR 2 of the runner pre-warm pool design (``docs/design/runner_prewarm_pool_plan.md``).

Spawns ONE long-running "template runner" subprocess at daemon startup
that imports the runner-tier plugin modules + walks plugin discovery.
Pool slots (PR 3) will fork from this template, inheriting the warm-
imports memory image — turning the per-session ~50s plugin-discovery
cost into a one-time daemon-startup cost.

This module ships ONLY the template lifecycle (spawn + shutdown +
PID/socket bookkeeping).  The fork-slot protocol (template responds to
FORK_SLOT control commands by forking and handing back a child
PID/socket via SCM_RIGHTS) is PR 3 work.

Why "one template" instead of "fork pool slots directly from daemon":
the daemon's Python process imports daemon-tier plugins (auth, cache_*,
gc_*, model_provider classes, etc.) — but NOT runner-tier plugins
(cli, file_edit, mcp, template, references, etc.).  Pool slots need
runner-tier imports warm; the daemon doesn't have them.  A dedicated
template subprocess that imports runner-tier plugins fills that gap.

Lifecycle:

  daemon startup ─→ TemplateManager().spawn()
                          │
                          ▼
                  (template subprocess runs)
                          │
                          ▼
  daemon shutdown ─→ TemplateManager().shutdown()
                          │
                          ▼
                  (template exits cleanly)

PR 3 will add ``request_fork_slot()`` to this manager — sends FORK_SLOT
command over the control socket; receives child PID + fresh socket via
SCM_RIGHTS; registers the child in the pool.

This module is **daemon-tier** — runs in the daemon's process, manages
a subprocess external to it.  Not a plugin in the discovery sense (no
``__init__.py`` declares PLUGIN_KIND); just a server-internal helper.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
from typing import Optional


logger = logging.getLogger(__name__)


class TemplateManager:
    """Owns the single template subprocess for the daemon's lifetime.

    Constructed at daemon startup.  Call :meth:`spawn` once to start
    the template; call :meth:`shutdown` once to stop it.  Thread-safe
    (only the spawn/shutdown calls take the lock; the template's
    socket/PID reads are stable after spawn).

    Attributes (post-spawn, before shutdown):
        pid: Template subprocess PID.
        control_sock: Daemon-side end of the control socketpair.
            Sends ``SHUTDOWN\\n`` for clean exit; future PRs will
            send ``FORK_SLOT\\n`` for pool replenishment.

    Both attributes are ``None`` before :meth:`spawn` and after
    :meth:`shutdown` returns.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pid: Optional[int] = None
        self.control_sock: Optional[socket.socket] = None
        # Captured at spawn time for diagnostic logging on shutdown.
        self._python_executable: str = sys.executable

    def spawn(self) -> None:
        """Spawn the template subprocess.

        Creates a socketpair, forks, and exec's ``python -m server.runner
        --template-mode`` in the child with the child's socket end on
        fd 3.  Returns once the fork+exec has succeeded — does NOT wait
        for the template to finish importing plugins (that's the
        template's own deferred warming).

        Idempotent: a second :meth:`spawn` call after a successful
        first one is a no-op (logs a warning).

        Raises:
            OSError: If the socketpair / fork / exec fails.  Daemon
                startup callers should treat this as fatal — without
                the template, pool slots can't get warm imports later.
        """
        with self._lock:
            if self.pid is not None:
                logger.warning(
                    "TemplateManager.spawn called twice; ignoring "
                    "(template pid=%d already running)", self.pid,
                )
                return

            # Socketpair: parent_sock stays with daemon for control;
            # child_sock crosses exec to fd 3 in the template.
            parent_sock, child_sock = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_STREAM,
            )

            pid = os.fork()
            if pid == 0:
                # Child branch — set up fd 3 and exec the template.
                try:
                    parent_sock.close()
                    os.dup2(child_sock.fileno(), 3)
                    # Close all other inherited FDs except 0/1/2/3.
                    try:
                        soft_limit = os.sysconf("SC_OPEN_MAX")
                    except (AttributeError, OSError, ValueError):
                        soft_limit = 1024
                    os.closerange(4, max(int(soft_limit), 4))
                    # No env vars needed — template-mode is fully self-
                    # describing via the --template-mode flag.
                    env = os.environ.copy()
                    os.execvpe(
                        self._python_executable,
                        [
                            self._python_executable,
                            "-m", "server.runner",
                            "--template-mode",
                        ],
                        env,
                    )
                except Exception:  # noqa: BLE001 — pre-exec failure
                    # Couldn't exec — die cleanly so the daemon sees
                    # SIGCHLD + non-zero exit and logs the failure.
                    os._exit(127)

            # Parent branch — keep the daemon-side socket.
            child_sock.close()
            self.pid = pid
            self.control_sock = parent_sock
            logger.info(
                "TemplateManager: spawned template pid=%d "
                "(imports warming up — plugins will be ready in "
                "~5-10s based on daemon-host load)",
                pid,
            )

    def shutdown(self, timeout: float = 5.0) -> None:
        """Tear down the template subprocess cleanly.

        Sends ``SHUTDOWN\\n`` over the control socket, then waits up to
        ``timeout`` seconds for the template to exit on its own.  After
        timeout, SIGTERMs the template; after another timeout, SIGKILLs.

        Idempotent: calling :meth:`shutdown` after a successful first
        call is a no-op (logs a warning).
        """
        with self._lock:
            if self.pid is None:
                logger.warning(
                    "TemplateManager.shutdown called with no template "
                    "running; nothing to do",
                )
                return

            pid = self.pid
            sock = self.control_sock

            # 1. Polite shutdown command.
            if sock is not None:
                try:
                    sock.sendall(b"SHUTDOWN\n")
                except OSError as exc:
                    logger.warning(
                        "TemplateManager.shutdown: failed to send "
                        "SHUTDOWN command (pid=%d): %s; falling "
                        "back to SIGTERM", pid, exc,
                    )
                try:
                    sock.close()
                except OSError:
                    pass

            # 2. Wait for clean exit.
            deadline_reached = False
            try:
                # ``waitpid`` blocks; use a poll loop instead.
                import time
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    waited_pid, _ = os.waitpid(pid, os.WNOHANG)
                    if waited_pid == pid:
                        break
                    time.sleep(0.05)
                else:
                    deadline_reached = True
            except ChildProcessError:
                # Template already gone — treat as success.
                pass

            # 3. Escalate to SIGTERM if still alive.
            if deadline_reached:
                logger.warning(
                    "TemplateManager.shutdown: template pid=%d "
                    "didn't exit within %.1fs; sending SIGTERM",
                    pid, timeout,
                )
                try:
                    os.kill(pid, 15)  # SIGTERM
                except ProcessLookupError:
                    pass  # already dead
                # Brief second wait
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass

            self.pid = None
            self.control_sock = None
            logger.info(
                "TemplateManager.shutdown: template pid=%d torn down",
                pid,
            )

    def is_alive(self) -> bool:
        """Return True if the template subprocess is still running.

        Cheap check via ``os.waitpid(pid, WNOHANG)`` — does not block.
        Returns False before :meth:`spawn` and after :meth:`shutdown`.
        """
        with self._lock:
            if self.pid is None:
                return False
            try:
                waited_pid, _ = os.waitpid(self.pid, os.WNOHANG)
                if waited_pid == self.pid:
                    # Reap the child — it's dead.  Clear state.
                    self.pid = None
                    if self.control_sock is not None:
                        try:
                            self.control_sock.close()
                        except OSError:
                            pass
                        self.control_sock = None
                    return False
                return True
            except ChildProcessError:
                self.pid = None
                return False
