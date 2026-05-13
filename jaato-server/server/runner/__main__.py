"""Runner subprocess entry point: ``python -m server.runner``.

Two modes:

- **Session mode** (default): spawned by
  :class:`server.runner_spawner.RunnerSpawner` for a single session.
  Self-confines to its AppArmor profile, adopts the RPC socketpair on
  fd 3, executes session work until the daemon closes the socket.

- **Template mode** (``--template-mode``): spawned by
  :class:`server.runner_template.TemplateManager` ONCE at daemon
  startup.  Imports runner-tier plugin modules + walks the plugin
  discovery directory so the resulting Python process has all the
  imports + class definitions warm in memory.  Then sits idle on a
  control pipe (fd 3) waiting for fork-slot requests (PR 3) or
  shutdown.  Per the pre-warm runner pool design at
  ``docs/design/runner_prewarm_pool_plan.md``, pool slots fork from
  this template and inherit the warm-imports memory image — turning
  the per-session ~50s plugin-discovery + plugin-import cost into a
  one-time daemon-startup cost.

Spawned (session mode) with the RPC socketpair inherited as fd 3 and
per-session config in env:

- ``JAATO_RUNNER_PROFILE`` — AppArmor profile to self-confine to (REQUIRED).
- ``JAATO_RUNNER_SESSION_ID`` — session id (informational; logging only).
- ``JAATO_RUNNER_WORKSPACE`` — workspace root for cli's ``cwd``.
- ``JAATO_RUNNER_MAX_OUTPUT_CHARS`` — cli output cap (optional).
- ``JAATO_RUNNER_TOOL_TIMEOUT_SECONDS`` — cli wall-clock cap (optional).
- ``JAATO_RUNNER_LOG_PATH`` — optional log-file path (else fd 1/2 left
  inherited from the daemon — see plan §5.1 for the per-workspace
  log default that the daemon's spawner sets).
- (no env vars consumed by this entry point — the runner-side
  session bootstrap is driven by the daemon's
  ``session.bootstrap`` RPC; the daemon dispatches that
  unconditionally as of Phase 3 §7c step 1.  The historical
  ``JAATO_RUNNER_HOSTS_SESSION`` review-aid flag was removed in
  §7c step 1 — see ``server/__main__.py`` for the
  always-bootstrap call site.)

Bootstrap order (§4.6):
1. Read profile from env.
2. ``aa_change_profile(profile)`` via ctypes.
3. Verify ``/proc/self/attr/current`` matches.  Hard-exit with code 2
   on any failure — the daemon detects via socket EOF + non-zero
   exit and emits ``SessionFailedEvent``.  No fallback to unconfined.
4. THEN import plugin code, build the executor, serve RPC on fd 3.

Per the spec's "DO NOT add try/except around aa_change_profile" rule,
errors surface verbatim with the kernel errno preserved.  Errors land
on stderr (which the spawner redirects to the per-session log file)
plus a single ``RUNNER_FATAL:`` line so the daemon's log scraping has
a stable marker for failure attribution.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from typing import Optional


_FATAL_PREFIX = "RUNNER_FATAL:"  # Stable log marker for daemon scraping.


def _fatal(message: str) -> "None":
    """Print a fatal-error line to stderr and exit with code 2."""
    sys.stderr.write(f"{_FATAL_PREFIX} {message}\n")
    sys.stderr.flush()
    os._exit(2)


def _setup_logging(log_path: Optional[str]) -> None:
    """Wire stdlib logging to a file or fall through to inherited stderr.

    The spawner is expected to redirect fd 1/2 to a per-session log
    file BEFORE exec; this function is for the case where that
    redirection didn't happen and the runner falls back to the
    daemon's stderr (testing / one-off invocations).
    """
    handlers = []
    if log_path:
        try:
            handlers.append(logging.FileHandler(log_path, mode="a"))
        except OSError as exc:
            sys.stderr.write(
                f"runner: failed to open log file {log_path!r}: {exc}; "
                f"falling back to stderr\n"
            )
    if not handlers:
        handlers.append(logging.StreamHandler(sys.stderr))

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def _parse_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        sys.stderr.write(
            f"runner: ignoring non-int env {name}={raw!r}\n"
        )
        return None


def _parse_float_env(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        sys.stderr.write(
            f"runner: ignoring non-float env {name}={raw!r}\n"
        )
        return None


def _run_template_mode() -> None:
    """Template-mode entry point — pre-warm runner pool design (pool PR 2).

    Imports the runner-tier plugin modules + walks plugin discovery so
    the resulting Python process has all the imports + class definitions
    warm in memory.  Then sits idle on fd 3 (the control pipe from the
    daemon) until shutdown.

    Pool slots (PR 3) will fork from this template, inheriting the warm
    imports — eliminating the per-session 50s plugin-discovery cost
    observed on v62 step 6 (see ``docs/design/runner_prewarm_pool_plan.md``
    §2 for the cost breakdown).

    PR 2 ships ONLY the template subprocess.  The fork-slot protocol
    (template responds to FORK_SLOT messages by forking + handing back
    a child PID/socket via SCM_RIGHTS) is PR 3 work.  In PR 2 the
    template just imports + sleeps.

    Three things make this safe to run unconfined (no AppArmor) inside
    the daemon-host:

    - The template never serves user-facing requests.  Only the daemon
      writes to its control pipe.
    - The template never reads workspace files.  Plugin discovery walks
      ``shared/plugins/*/`` — daemon-host code paths only.
    - The template has no FS writes beyond stdlib's compiled-bytecode
      caches.

    Pool slots forked from the template DO self-confine to a per-session
    AppArmor profile before running session work (mirrors today's
    runner self-confine pattern).

    Lifecycle exits:
      - Daemon closes fd 3 → ``os.read(3, ...)`` returns empty bytes
        → clean exit with code 0.
      - Daemon sends ``SHUTDOWN\\n`` line → clean exit with code 0.
      - Any uncaught exception during plugin import → fatal exit code
        2 (matching session-mode's _fatal contract so daemon detection
        is uniform).
    """
    _setup_logging(log_path=None)
    log = logging.getLogger("server.runner.template")
    log.info("runner template starting (warming runner-tier plugin imports)")

    try:
        from shared.plugins.registry import PluginRegistry
    except Exception as exc:  # noqa: BLE001 — boundary surface
        _fatal(
            f"template-mode: failed to import PluginRegistry: "
            f"{type(exc).__name__}: {exc}"
        )

    # Walk runner-tier plugins.  Matches the call site in
    # ``server/runner/session.py:_configure_runtime_plugins`` (line
    # 187) so the template inherits exactly the import set sessions
    # consume.  Daemon-tier plugins (auth, gc_*, cache_*, etc.) are
    # excluded — they live on the daemon side and aren't part of the
    # runner's per-session bootstrap cost.
    t0 = __import__("time").perf_counter()
    try:
        registry = PluginRegistry()
        discovered = registry.discover(tier_filter="runner")
    except Exception as exc:  # noqa: BLE001 — boundary surface
        _fatal(
            f"template-mode: plugin discovery crashed: "
            f"{type(exc).__name__}: {exc}"
        )
    discover_ms = (__import__("time").perf_counter() - t0) * 1000
    log.info(
        "runner template ready: %d plugins discovered in %.1fms — "
        "pool slots will fork from this process and inherit warm "
        "imports.  Note: discover() walks __init__.py files; some "
        "plugins defer heavy imports to initialize() which only fires "
        "post-bootstrap.  PR 2 measures actual savings will be confirmed "
        "when PR 4 routes sessions through the pool.",
        len(discovered),
        discover_ms,
    )

    # Sit idle on fd 3 waiting for the daemon's shutdown signal.
    # PR 3 will extend this loop to dispatch FORK_SLOT requests.
    try:
        control_sock = socket.socket(fileno=3)
        control_sock.setblocking(True)
    except OSError as exc:
        _fatal(
            f"template-mode: could not adopt fd 3 as control socket: "
            f"{exc}; daemon's TemplateManager.spawn() is supposed to "
            f"dup the control socketpair to fd 3 before exec"
        )

    log.info("runner template idle on fd 3; awaiting daemon control commands")
    try:
        _template_control_loop(control_sock, log)
    finally:
        try:
            control_sock.close()
        except OSError:
            pass


def _template_control_loop(control_sock: "socket.socket", log) -> None:
    """Template's main loop — read commands from daemon on fd 3.

    Two commands supported (PR 3):

    - ``SHUTDOWN\\n`` — clean exit.
    - ``FORK_SLOT\\n`` with a socket FD attached via ``SCM_RIGHTS``
      — fork; in child, become a pool slot listening on the
      received socket; in parent (template), reply
      ``FORKED:<pid>\\n`` to the daemon over the control socket.

    Each command must arrive as a complete line via a single
    ``recvmsg`` call (no buffering across calls).  This works
    because the daemon-side ``TemplateManager`` sends each command
    atomically via one ``sendmsg``, and the kernel's Unix-socket
    datagram-like framing for ``SOCK_STREAM`` preserves the boundary
    for short payloads — but we still defensively read until we
    have a complete line.

    Unknown commands log a warning and are dropped.
    """
    import struct

    while True:
        # ``recvmsg`` lets us receive both bytes AND ancillary data
        # (SCM_RIGHTS FDs).  CMSG_SPACE(struct.calcsize("i")) sizes
        # the ancillary buffer for a single passed FD.
        try:
            data, ancdata, _flags, _addr = control_sock.recvmsg(
                4096,
                socket.CMSG_SPACE(struct.calcsize("i")),
            )
        except OSError as exc:
            log.warning("runner template: recvmsg failed: %s; exiting", exc)
            return

        if not data:
            # Daemon closed the control pipe — clean shutdown path.
            log.info("runner template: control pipe EOF; shutting down")
            return

        # Extract any SCM_RIGHTS FD that came alongside this message.
        received_fd: Optional[int] = None
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                # CMSG_RIGHTS can carry an array of FDs; take the first.
                received_fd = struct.unpack("i", cmsg_data[:struct.calcsize("i")])[0]
                break

        # Parse the command line (strip newline + whitespace).
        cmd = data.decode("utf-8", errors="replace").strip()

        if cmd == "SHUTDOWN":
            log.info("runner template: SHUTDOWN command received")
            if received_fd is not None:
                # SHUTDOWN shouldn't carry an FD, but defend against it.
                try:
                    os.close(received_fd)
                except OSError:
                    pass
            return

        if cmd == "FORK_SLOT":
            if received_fd is None:
                log.error(
                    "runner template: FORK_SLOT command arrived without "
                    "SCM_RIGHTS FD; dropping"
                )
                continue
            _handle_fork_slot(control_sock, received_fd, log)
            continue

        # Unknown command — log + drop any attached FD.
        log.warning("runner template: unknown control command %r", cmd)
        if received_fd is not None:
            try:
                os.close(received_fd)
            except OSError:
                pass


def _handle_fork_slot(
    control_sock: "socket.socket", slot_fd: int, log,
) -> None:
    """Handle a FORK_SLOT command (template side).

    Fork the template process.  In the child, close the template's
    control socket (the child doesn't talk to the daemon over that
    pipe; it has its own slot socket on the received FD) and enter
    slot-mode.  In the parent (template), close the received FD
    (only the child needs it) and reply ``FORKED:<child_pid>\\n``
    to the daemon.

    Both branches log + handle errors uniformly:

    - Fork failure: log + reply ``FORK_FAILED\\n`` to daemon.
    - Child-side slot-mode exit: ``sys.exit(0)`` ends the child
      cleanly — the daemon sees the slot's slot-socket closure +
      the SIGCHLD signal as the slot's lifecycle terminating.
    """
    try:
        child_pid = os.fork()
    except OSError as exc:
        log.error("runner template: fork failed: %s", exc)
        try:
            os.close(slot_fd)
        except OSError:
            pass
        try:
            control_sock.sendall(b"FORK_FAILED\n")
        except OSError:
            pass
        return

    if child_pid == 0:
        # CHILD: become a pool slot.  Drop the template's control
        # socket (child doesn't talk to the daemon over that pipe;
        # it has its own slot socket) and enter slot-mode.
        try:
            control_sock.close()
        except OSError:
            pass
        _run_slot_mode(slot_fd, log)
        # _run_slot_mode calls sys.exit; this return is a safety net.
        os._exit(0)

    # PARENT (template): close received FD (only the child uses it)
    # and acknowledge to daemon with the slot's PID.
    try:
        os.close(slot_fd)
    except OSError:
        pass
    try:
        control_sock.sendall(f"FORKED:{child_pid}\n".encode("utf-8"))
        log.info(
            "runner template: forked slot pid=%d (warm imports inherited)",
            child_pid,
        )
    except OSError as exc:
        log.warning(
            "runner template: failed to acknowledge fork-slot to daemon "
            "(slot pid=%d already forked): %s", child_pid, exc,
        )


def _run_slot_mode(slot_fd: int, log) -> None:
    """Slot-mode entry — runs AFTER fork from template.

    The slot inherits the template's warm-imported plugin modules
    (the whole point of the pool design).  Sits idle on the slot
    socket waiting for a bootstrap envelope (PR 4) or shutdown.

    PR 3 ships ONLY the idle-wait behavior — slots wait for a
    SHUTDOWN command or pipe EOF and exit.  PR 4 will add
    envelope-handling that bootstraps a real session inside the
    slot (reusing the existing ``bootstrap_session`` code path that
    today's session-mode runners take).
    """
    # Dup the slot socket FD to fd 3 to match the existing runner
    # convention (session-mode also adopts fd 3 as its RPC socket).
    # PR 4 will route the bootstrap envelope here; for PR 3 the slot
    # just listens for SHUTDOWN.
    try:
        os.dup2(slot_fd, 3)
    except OSError as exc:
        sys.stderr.write(
            f"runner slot: failed to dup slot_fd to fd 3: {exc}\n"
        )
        os._exit(2)
    try:
        os.close(slot_fd)
    except OSError:
        pass

    try:
        slot_sock = socket.socket(fileno=3)
        slot_sock.setblocking(True)
    except OSError as exc:
        sys.stderr.write(
            f"runner slot: failed to adopt fd 3 as socket: {exc}\n"
        )
        os._exit(2)

    slot_log = logging.getLogger("server.runner.slot")
    slot_log.info(
        "runner slot ready: pid=%d (warm imports inherited from "
        "template); waiting for envelope or shutdown",
        os.getpid(),
    )

    try:
        buf = b""
        while True:
            chunk = slot_sock.recv(4096)
            if not chunk:
                slot_log.info(
                    "runner slot pid=%d: daemon closed slot pipe; exiting",
                    os.getpid(),
                )
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.decode("utf-8", errors="replace").strip()
                if cmd == "SHUTDOWN":
                    slot_log.info(
                        "runner slot pid=%d: SHUTDOWN command received",
                        os.getpid(),
                    )
                    sys.exit(0)
                # PR 4 will add envelope handling here — receive
                # the SessionInitEnvelope, self-confine to the
                # session's AppArmor profile, run bootstrap_session,
                # serve the session.
                slot_log.warning(
                    "runner slot pid=%d: ignoring unknown command %r "
                    "(PR 4 will add envelope handling)",
                    os.getpid(), cmd,
                )
    finally:
        try:
            slot_sock.close()
        except OSError:
            pass
    sys.exit(0)


def main() -> None:
    """Run bootstrap + serve.  Never returns under normal operation —
    serve exits cleanly on peer EOF, after which we ``sys.exit(0)``."""
    # ----- 0. Mode dispatch -----
    # ``--template-mode`` selects the pre-warm template subprocess
    # entry point (pool PR 2).  Daemon spawns one template at startup
    # via :class:`server.runner_template.TemplateManager`.
    if "--template-mode" in sys.argv:
        _run_template_mode()
        sys.exit(0)

    # ----- 1. Read env -----
    profile_name = os.environ.get("JAATO_RUNNER_PROFILE", "").strip()
    session_id = os.environ.get("JAATO_RUNNER_SESSION_ID", "").strip()
    workspace_root = os.environ.get("JAATO_RUNNER_WORKSPACE", "").strip() or None
    log_path = os.environ.get("JAATO_RUNNER_LOG_PATH", "").strip() or None
    max_output_chars = _parse_int_env("JAATO_RUNNER_MAX_OUTPUT_CHARS")
    tool_timeout_seconds = _parse_float_env("JAATO_RUNNER_TOOL_TIMEOUT_SECONDS")

    # JAATO_RUNNER_DISABLE_CONFINE is the developer pdb-attach escape
    # hatch (see spec §5).  Documented as unsupported; ships ONLY for
    # the local-debug loop and is gated to a clear "this is not a
    # supported deployment" warning.
    disable_confine = (
        os.environ.get("JAATO_RUNNER_DISABLE_CONFINE", "").lower()
        in ("1", "true", "yes")
    )

    _setup_logging(log_path)
    log = logging.getLogger("server.runner")
    log.info(
        "runner starting: session_id=%s profile=%s workspace=%s "
        "max_output_chars=%s tool_timeout_seconds=%s disable_confine=%s",
        session_id, profile_name, workspace_root,
        max_output_chars, tool_timeout_seconds, disable_confine,
    )

    if not profile_name and not disable_confine:
        _fatal(
            "JAATO_RUNNER_PROFILE not set; the runner refuses to start "
            "without an AppArmor profile to self-confine to.  Set "
            "JAATO_RUNNER_DISABLE_CONFINE=1 to bypass for local pdb "
            "debugging only — this is NOT a supported deployment."
        )

    # ----- 2-3. Self-confine + verify -----
    if disable_confine:
        log.warning(
            "JAATO_RUNNER_DISABLE_CONFINE is set — running unconfined.  "
            "This is NOT a supported deployment; AppArmor isolation is "
            "disabled for this runner."
        )
    else:
        # Imported here, NOT at module top, so a bug in bootstrap
        # surfaces before any unrelated code runs.
        from .bootstrap import (
            ConfinementMismatchError,
            confine_to_profile,
        )
        try:
            confine_to_profile(profile_name)
        except ConfinementMismatchError as exc:
            _fatal(
                f"AppArmor confinement mismatch — kernel reports "
                f"{exc.actual!r} but we requested {exc.expected!r}.  "
                f"Likely cause: parent profile lacks "
                f"'change_profile -> {exc.expected}' rule.  See design "
                f"§4.6 daemon apparmor-state constraint."
            )
        except RuntimeError as exc:
            _fatal(f"AppArmor self-confine failed: {exc}")

    # ----- 4. Now import plugin code -----
    # Per §4.6 step 4 the import happens AFTER confinement so plugin
    # module-load runs under the per-session profile, not unconfined.
    from .rpc import RunnerRPC
    from .tool_executor import ToolExecutor

    # ----- 5. Adopt fd 3 as the RPC socketpair -----
    try:
        sock = socket.socket(fileno=3)
        # Switch to blocking mode — runner-side framing uses
        # ``sock.recv`` with explicit byte counts.
        sock.setblocking(True)
    except OSError as exc:
        _fatal(
            f"could not adopt fd 3 as a socket: {exc}; the runner "
            f"expects the daemon to dup the socketpair to fd 3 before "
            f"exec()"
        )

    # ----- 6. Build the executor + dispatcher -----
    executor = ToolExecutor(
        workspace_root=workspace_root,
        # When env didn't override, ToolExecutor falls back to its
        # cli-runner default cap.
        **({"max_output_chars": max_output_chars} if max_output_chars is not None else {}),
        **({"tool_timeout_seconds": tool_timeout_seconds} if tool_timeout_seconds is not None else {}),
    )
    # workspace_root is forwarded for §3.1 traceback sanitization —
    # captured tracebacks have tenant-specific paths redacted to
    # ``<WORKSPACE>/...`` before crossing the RPC boundary.
    rpc = RunnerRPC(sock, executor.execute, workspace_root=workspace_root)

    log.info("runner ready; serving RPC on fd 3")
    try:
        rpc.serve()
    except KeyboardInterrupt:
        log.info("runner: SIGINT — shutting down")
    finally:
        try:
            sock.close()
        except OSError:
            pass

    log.info("runner exiting cleanly")
    sys.exit(0)


if __name__ == "__main__":
    main()
