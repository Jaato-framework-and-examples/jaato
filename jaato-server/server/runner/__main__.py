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
        registry.discover(tier_filter="runner")
    except Exception as exc:  # noqa: BLE001 — boundary surface
        _fatal(
            f"template-mode: plugin discovery crashed: "
            f"{type(exc).__name__}: {exc}"
        )
    discover_ms = (__import__("time").perf_counter() - t0) * 1000
    log.info(
        "runner template ready: %d plugins discovered in %.1fms — "
        "pool slots will fork from this process",
        len(list(registry.list_exposed())) if hasattr(registry, "list_exposed") else 0,
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
        buf = b""
        while True:
            chunk = control_sock.recv(256)
            if not chunk:
                # Daemon closed the control pipe — clean shutdown path.
                log.info("runner template: control pipe EOF; shutting down")
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.decode("utf-8", errors="replace").strip()
                if cmd == "SHUTDOWN":
                    log.info("runner template: SHUTDOWN command received")
                    return
                # PR 3 will add FORK_SLOT handling here.
                log.warning(
                    "runner template: unknown control command %r "
                    "(PR 3 will add FORK_SLOT)", cmd,
                )
    finally:
        try:
            control_sock.close()
        except OSError:
            pass


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
