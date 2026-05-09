"""Runner subprocess entry point: ``python -m server.runner``.

Spawned by :class:`server.runner_spawner.RunnerSpawner` with the RPC
socketpair inherited as fd 3 and per-session config in env:

- ``JAATO_RUNNER_PROFILE`` — AppArmor profile to self-confine to (REQUIRED).
- ``JAATO_RUNNER_SESSION_ID`` — session id (informational; logging only).
- ``JAATO_RUNNER_WORKSPACE`` — workspace root for cli's ``cwd``.
- ``JAATO_RUNNER_MAX_OUTPUT_CHARS`` — cli output cap (optional).
- ``JAATO_RUNNER_TOOL_TIMEOUT_SECONDS`` — cli wall-clock cap (optional).
- ``JAATO_RUNNER_LOG_PATH`` — optional log-file path (else fd 1/2 left
  inherited from the daemon — see plan §5.1 for the per-workspace
  log default that the daemon's spawner sets).
- ``JAATO_RUNNER_HOSTS_SESSION`` — Phase 3 §3.3b transitional
  review-aid flag.  When ``"1"`` / ``"true"``, the runner is
  authoritative for ``JaatoSession`` hosting; when unset, the
  daemon side instantiates and the runner is the cli-only Phase 2
  surface.  This flag is consumed by §3.3c's ``session.bootstrap``
  RPC handler (not yet wired here); §3.3b ships only the bootstrap
  function (``server.runner.session.bootstrap_session``) + its unit
  tests.  NOT a feature flag in the parent §5 sense — bounded to
  the §3.3b → §3.3c PR window.

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


def main() -> None:
    """Run bootstrap + serve.  Never returns under normal operation —
    serve exits cleanly on peer EOF, after which we ``sys.exit(0)``."""
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
