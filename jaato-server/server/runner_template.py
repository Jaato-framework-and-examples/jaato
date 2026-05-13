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


# Pool PR 5c: wall-clock cap on waiting for the template's READY
# signal post-spawn.  Plugin discovery typically takes ~1.5-2s on
# fast workstations and up to ~10s on slow hosts; 30s is generous
# enough to cover CI / cold-host edge cases without leaving the
# daemon hung indefinitely on a stuck template.
_DEFAULT_READY_TIMEOUT = 30.0


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
                "(imports warming up — waiting for READY signal "
                "before declaring template usable)", pid,
            )

        # Pool PR 5c: wait for the template to send "READY\n" over the
        # control socket once plugin discovery completes.  Replaces
        # the daemon's previous 2s ``time.sleep``.  Held OUTSIDE the
        # lock above so we don't pin _lock for the full ready-wait
        # window — other accesses to is_alive() / shutdown() should
        # not block on template warm-up.
        self._wait_for_ready(timeout=_DEFAULT_READY_TIMEOUT)

    def _wait_for_ready(self, timeout: float) -> None:
        """Block until template sends ``READY\\n`` over the control
        socket OR the timeout fires.

        Pool PR 5c: replaces the daemon's previous 2s ``time.sleep``
        after template spawn.  On timeout, the daemon logs a warning
        but does NOT fail — the template may still come up later
        (next FORK_SLOT will hang or fail gracefully) and the
        watchdog (PR 5b) will eventually respawn if needed.

        Reads exactly the "READY\\n" prefix.  Any additional bytes
        in the recv buffer (e.g. an unexpected message) are
        non-fatal: the control loop will deal with them.
        """
        with self._lock:
            sock = self.control_sock
            pid = self.pid

        if sock is None or pid is None:
            return  # Defensive: spawn() didn't actually populate state.

        sock.settimeout(timeout)
        try:
            data = sock.recv(64)
        except socket.timeout:
            logger.warning(
                "TemplateManager: template pid=%d did not send READY "
                "within %.1fs.  Pool may be slow to fill; watchdog "
                "will retry on subsequent loop iterations.  Possible "
                "causes: plugin discovery stuck, very slow host, or "
                "template subprocess crashed before sending signal.",
                pid, timeout,
            )
            return
        except OSError as exc:
            logger.warning(
                "TemplateManager: recv READY failed (pid=%d): %s; "
                "treating as best-effort.  Pool may be empty until "
                "watchdog respawn cycle.", pid, exc,
            )
            return
        finally:
            sock.settimeout(None)

        if not data:
            logger.warning(
                "TemplateManager: template pid=%d closed control "
                "socket before sending READY (likely crashed during "
                "plugin imports).  Watchdog will respawn.", pid,
            )
            return

        decoded = data.decode("utf-8", errors="replace").strip()
        if decoded.split("\n", 1)[0] == "READY":
            logger.info(
                "TemplateManager: template pid=%d READY (plugin "
                "discovery complete; pool slots can fork now)", pid,
            )
        else:
            logger.warning(
                "TemplateManager: template pid=%d sent unexpected "
                "first message %r (expected 'READY\\n'); proceeding "
                "anyway but the template may be in a degraded state",
                pid, decoded[:60],
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

    def request_fork_slot(
        self, timeout: float = 5.0,
    ) -> "Optional[tuple[int, socket.socket]]":
        """Ask the template to fork a new pool slot (pool PR 3).

        Protocol:

        1. Create a socketpair (daemon_end, slot_end)
        2. Send ``FORK_SLOT\\n`` + ``slot_end`` FD via SCM_RIGHTS over
           the template's control pipe
        3. Close daemon's copy of ``slot_end`` (the kernel keeps the
           refcount alive in the template's address space)
        4. Wait for the template's ``FORKED:<pid>\\n`` reply
        5. Return ``(slot_pid, daemon_end)`` — the daemon-side socket
           is the only way to talk to that specific slot

        Thread-safe: serializes concurrent fork-slot requests via
        the manager's lock so the template's single-threaded control
        loop sees one command at a time.

        Args:
            timeout: Wall-clock cap on the FORKED reply.  Default 5s;
                template fork should be sub-100ms in practice.

        Returns:
            ``(slot_pid, daemon_side_socket)`` on success.  ``None``
            on failure (template dead, fork failed, timeout).  Caller
            tracks the handle for later session bootstrap (PR 4) +
            shutdown.

        Raises:
            None — all failures return ``None`` with a warning log.
            Pool manager treats ``None`` as "couldn't grow the pool;
            sessions fall back to cold-spawn session-mode".
        """
        import struct

        with self._lock:
            if self.control_sock is None or self.pid is None:
                logger.warning(
                    "TemplateManager.request_fork_slot: no template "
                    "running; cannot fork-slot",
                )
                return None

            # Step 1: create the new slot's socketpair.
            try:
                daemon_end, slot_end = socket.socketpair(
                    socket.AF_UNIX, socket.SOCK_STREAM,
                )
            except OSError as exc:
                logger.warning(
                    "TemplateManager.request_fork_slot: socketpair "
                    "failed: %s", exc,
                )
                return None

            # Step 2: send FORK_SLOT + slot_end FD via SCM_RIGHTS.
            try:
                self.control_sock.sendmsg(
                    [b"FORK_SLOT\n"],
                    [(
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        struct.pack("i", slot_end.fileno()),
                    )],
                )
            except OSError as exc:
                logger.warning(
                    "TemplateManager.request_fork_slot: sendmsg "
                    "failed: %s", exc,
                )
                daemon_end.close()
                slot_end.close()
                return None

            # Step 3: close daemon's slot_end (template's kernel-side
            # dup keeps the FD alive in the child after fork).
            slot_end.close()

            # Step 4: wait for FORKED:<pid> reply.
            self.control_sock.settimeout(timeout)
            try:
                reply_bytes = self.control_sock.recv(256)
            except socket.timeout:
                logger.warning(
                    "TemplateManager.request_fork_slot: timed out "
                    "after %.1fs waiting for FORKED reply", timeout,
                )
                daemon_end.close()
                return None
            except OSError as exc:
                logger.warning(
                    "TemplateManager.request_fork_slot: recv failed: %s",
                    exc,
                )
                daemon_end.close()
                return None
            finally:
                self.control_sock.settimeout(None)

            if not reply_bytes:
                logger.warning(
                    "TemplateManager.request_fork_slot: template "
                    "closed control pipe before replying",
                )
                daemon_end.close()
                return None

            reply = reply_bytes.decode("utf-8", errors="replace").strip()
            if reply.startswith("FORKED:"):
                try:
                    slot_pid = int(reply[len("FORKED:"):])
                except ValueError:
                    logger.warning(
                        "TemplateManager.request_fork_slot: malformed "
                        "FORKED reply %r", reply,
                    )
                    daemon_end.close()
                    return None
                logger.info(
                    "TemplateManager: forked pool slot pid=%d (template "
                    "pid=%d)", slot_pid, self.pid,
                )
                return (slot_pid, daemon_end)

            if reply == "FORK_FAILED":
                logger.warning(
                    "TemplateManager.request_fork_slot: template "
                    "reported FORK_FAILED",
                )
                daemon_end.close()
                return None

            logger.warning(
                "TemplateManager.request_fork_slot: unexpected reply %r",
                reply,
            )
            daemon_end.close()
            return None

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
