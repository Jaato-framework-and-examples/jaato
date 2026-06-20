"""Client-side preflight diagnostic for jaato SDK clients.

Run ``python -m jaato_sdk.doctor`` *before* writing or debugging an SDK
client.  It answers, deterministically and from the client's own
vantage, the questions that otherwise cost an hour of trial-and-error:

- Is the ``server`` package importable, so autostart's
  ``python -m server`` will work?
- Is a daemon listening on the socket, or is the socket file **stale**
  (present but dead) — the state that silently blocks autostart?
- **Which** daemon am I attached to, and **what HOME does it run with?**
  Daemon-side ``pass://`` secret resolution reads the *daemon's* HOME, not
  your shell's.  A HOME mismatch is the single most expensive jaato-client
  failure (``pass show ... is not in the password store`` even though
  ``pass`` works in your shell).  The doctor reads the daemon's HOME from
  ``/proc/<pid>/environ`` and compares it to yours.
- Does a given ``pass://`` secret actually resolve — from the daemon's
  HOME *and* from yours?
- Where will profiles be discovered and session logs be written?
- Is ``connect()``'s default timeout adequate for a cold daemon start?

Every check prints ``PASS`` / ``WARN`` / ``FAIL`` with a concrete,
actionable detail line.  The process exits non-zero if any check is
``FAIL`` — so the doctor is usable as a CI / harness gate.

Design rules (matching the framework's): **no hardcoded fallbacks** —
the socket and pidfile paths come from the SDK's own
``DEFAULT_SOCKET_PATH`` / ``DEFAULT_PID_FILE`` constants so the doctor
can never drift from the client.  Platform capabilities (``/proc``,
``pass``) are *stated*, never silently assumed.  Secrets are never
printed; only their resolvability (a subprocess return code) is reported.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Import the SDK's own path constants so the doctor diagnoses exactly the
# files the real client uses — never a re-declared copy that could drift.
from jaato_sdk.client.ipc import DEFAULT_SOCKET_PATH, DEFAULT_PID_FILE

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_GLYPH = {PASS: "✔", WARN: "▲", FAIL: "✘"}


@dataclass
class Check:
    """One diagnostic result.

    Attributes:
        name: Short label shown to the user (e.g. ``"daemon HOME"``).
        status: One of ``PASS`` / ``WARN`` / ``FAIL``.
        detail: A concrete, actionable one-or-more-line explanation.  For
            ``FAIL`` this should name the exact fix.
    """

    name: str
    status: str
    detail: str


@dataclass
class DaemonInfo:
    """What the doctor could learn about the daemon behind a socket.

    Populated by :func:`probe_daemon`.  ``listening`` is the only field
    that is always trustworthy; ``pid`` / ``home`` are best-effort and may
    be ``None`` when the pidfile is absent (e.g. a non-default socket) or
    ``/proc`` is unavailable (non-Linux).
    """

    socket_path: str
    socket_exists: bool
    listening: bool
    pid: Optional[int] = None
    pid_alive: Optional[bool] = None
    home: Optional[str] = None
    user: Optional[str] = None
    password_store_dir: Optional[str] = None
    env: Optional[dict] = None           # the daemon's full /proc/<pid>/environ


# --------------------------------------------------------------------------
# Probes (side-effect-free observation; no autostart, no mutation)
# --------------------------------------------------------------------------

def _socket_listening(path: str) -> bool:
    """Return True iff something accepts a connection on the Unix socket.

    A plain ``connect_ex`` — deliberately does NOT trigger the SDK's
    autostart, so the doctor observes the *current* state rather than
    changing it.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        return s.connect_ex(path) == 0
    finally:
        s.close()


def _read_pid(pidfile: str) -> Optional[int]:
    """Read an integer PID from ``pidfile``; ``None`` if absent/unparseable."""
    try:
        return int(Path(pidfile).read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    """Return True iff ``pid`` names a live process (signal-0 probe)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user — still "alive" for our purpose.
        return True


def _proc_environ(pid: int) -> Optional[dict]:
    """Parse ``/proc/<pid>/environ`` into a dict; ``None`` if unavailable.

    The NUL-separated environ block is how the doctor recovers the
    daemon's HOME — there is no SDK/handshake field that exposes it.
    Returns ``None`` on non-Linux (no ``/proc``) or permission denial.
    """
    p = Path(f"/proc/{pid}/environ")
    try:
        raw = p.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    env: dict = {}
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        key, _, value = entry.partition(b"=")
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def probe_daemon(socket_path: str, pidfile: str) -> DaemonInfo:
    """Observe the daemon behind ``socket_path`` without mutating anything.

    Reads listen-state, then (best-effort) the pidfile → liveness →
    ``/proc/<pid>/environ`` for HOME/USER/PASSWORD_STORE_DIR.
    """
    info = DaemonInfo(
        socket_path=socket_path,
        socket_exists=Path(socket_path).exists(),
        listening=_socket_listening(socket_path),
    )
    info.pid = _read_pid(pidfile)
    if info.pid is not None:
        info.pid_alive = _pid_alive(info.pid)
        if info.pid_alive:
            env = _proc_environ(info.pid)
            if env is not None:
                info.env = env
                info.home = env.get("HOME")
                info.user = env.get("USER")
                info.password_store_dir = env.get("PASSWORD_STORE_DIR")
    return info


# --------------------------------------------------------------------------
# Checks (turn probes into PASS/WARN/FAIL verdicts)
# --------------------------------------------------------------------------

def check_python_env() -> List[Check]:
    """Verify the interpreter can import what an SDK client needs.

    Autostart launches the daemon via ``python -m server`` using the
    *current* ``sys.executable`` — so ``server`` must be importable on
    this interpreter, not just ``jaato_sdk``.
    """
    checks = [Check("python", PASS, f"{sys.executable}")]

    import importlib.util as _u
    for mod, hint in (
        ("jaato_sdk", "pip install -e jaato-sdk/."),
        ("server", "pip install -e 'jaato-server/.[all]' — required for "
                    "autostart (python -m server)"),
    ):
        if _u.find_spec(mod) is not None:
            checks.append(Check(f"import {mod}", PASS, "importable"))
        else:
            checks.append(Check(
                f"import {mod}", FAIL,
                f"{mod!r} not importable on this interpreter — {hint}",
            ))
    return checks


def check_socket(info: DaemonInfo, *, auto_start: bool) -> List[Check]:
    """Classify the socket state, distinguishing the stale-socket trap.

    The dangerous state is *file present, nothing listening*: with the
    pidfile also pointing at a dead PID, autostart can be blocked.  This
    check is the reproduction signal for that class of failure.
    """
    if info.listening:
        return [Check("socket", PASS,
                      f"{info.socket_path} — daemon listening")]

    if not info.socket_exists:
        if auto_start:
            return [Check("socket", WARN,
                          f"{info.socket_path} absent — no daemon yet; "
                          f"autostart will launch one (cold start ~30-60s; "
                          f"call connect(timeout>=120)).")]
        return [Check("socket", FAIL,
                      f"{info.socket_path} absent and auto_start=False — "
                      f"start a daemon or set auto_start=True.")]

    # File exists but nothing is listening → STALE.
    stale_pid = (
        info.pid is not None and info.pid_alive is False
    )
    detail = (
        f"{info.socket_path} exists but nothing is listening — STALE socket "
        f"(daemon crashed/killed)."
    )
    if stale_pid:
        detail += (
            f"  Pidfile {DEFAULT_PID_FILE} → PID {info.pid} is DEAD: this is "
            f"the autostart-block state.  Clear it: "
            f"`rm -f {info.socket_path} {DEFAULT_PID_FILE}`."
        )
    elif info.pid is None:
        detail += (
            f"  No pidfile at {DEFAULT_PID_FILE}.  Clear the socket: "
            f"`rm -f {info.socket_path}`."
        )
    return [Check("socket", FAIL, detail)]


def check_daemon_identity(info: DaemonInfo) -> List[Check]:
    """Report the attached daemon's PID and introspection availability."""
    if not info.listening:
        return [Check("daemon identity", WARN,
                      "no daemon listening — identity checks skipped.")]
    if info.pid is None:
        return [Check("daemon identity", WARN,
                      f"daemon listening but no pidfile at {DEFAULT_PID_FILE} "
                      f"(non-default socket?) — cannot read its HOME.")]
    if info.home is None:
        return [Check("daemon identity", WARN,
                      f"PID {info.pid} alive but /proc/{info.pid}/environ "
                      f"unreadable (non-Linux or permission) — cannot verify "
                      f"the daemon's HOME for secret resolution.")]
    return [Check("daemon identity", PASS,
                  f"PID {info.pid}  HOME={info.home}  "
                  f"USER={info.user or '?'}")]


def check_home_match(info: DaemonInfo) -> List[Check]:
    """Compare the daemon's HOME to the caller's — the #1 pass:// trap.

    ``pass://`` resolves daemon-side, so a secret store is found under
    *the daemon's* ``$HOME/.password-store``.  When the daemon was
    autostarted by a process with a different HOME (a sandboxed harness,
    a sudo context), secrets in your shell's store are invisible to it.
    """
    if info.home is None:
        return []  # already surfaced by check_daemon_identity
    my_home = os.environ.get("HOME", "")
    if info.home == my_home:
        return [Check("HOME match", PASS,
                      f"daemon HOME == your HOME ({my_home}) — pass:// "
                      f"secrets resolve from the same store.")]
    return [Check("HOME match", FAIL,
                  f"MISMATCH: daemon HOME={info.home} but your HOME={my_home}. "
                  f"Daemon-side pass:// resolves from "
                  f"{info.home}/.password-store, NOT your "
                  f"{my_home}/.password-store.  Fix: stop targeting that "
                  f"daemon; let the client autostart its own on a fresh "
                  f"socket so it inherits your HOME.")]


def load_known_env_vars() -> Optional[Dict[str, str]]:
    """Map ``{env_var_name: tier}`` the framework actually reads, or ``None``.

    Soft-imports the server-side introspector (``shared.scaffold.introspect``)
    — the SAME scan that powers ``jaato-scaffold explain env`` — so the doctor
    reports the daemon's environ against the REAL read-set (``JAATO_*`` AND
    ``OTEL_*`` / ``AI_*`` / provider vendor vars / …), not a hardcoded prefix
    list.  Returns ``None`` when the introspector isn't importable (a
    client-only install); the daemon-env check then WARNs rather than guessing.
    """
    try:
        from shared.scaffold.introspect import env_vars  # type: ignore
    except Exception:
        return None
    return {n: v.tier for n, v in env_vars().items()}


def check_daemon_env(
    info: DaemonInfo, known: Optional[Dict[str, str]],
) -> List[Check]:
    """Report the framework env vars the running daemon was fired with.

    The daemon's full environ is already read for the HOME check; this surfaces
    every var the framework code actually reads (per :func:`load_known_env_vars`
    — any tier, any namespace) that the daemon was started with, each tagged
    with the tier of its reader, so a stale/overriding launch knob (a
    ``JAATO_GC_THRESHOLD``, an ``OTEL_EXPORTER_OTLP_ENDPOINT``, a provider key)
    is visible rather than silent.  Secret-valued vars (name contains
    KEY/TOKEN/SECRET/PASSWORD) are shown set-but-redacted, never printed.
    """
    if not info.listening or info.env is None:
        return [Check("daemon env", WARN,
                      "daemon environ unavailable (not listening / non-Linux / "
                      "no pidfile) — cannot show how it was fired.")]
    if known is None:
        return [Check("daemon env", WARN,
                      "jaato-server's introspector is not importable here — "
                      "cannot map the daemon's environ to the framework's "
                      "read-set (install jaato-server / run from its env).")]
    secret = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    # HOME / USER / PASSWORD_STORE_DIR are already shown by the identity check.
    shown_elsewhere = ("HOME", "USER", "PASSWORD_STORE_DIR")
    items = []
    for k in sorted(info.env):
        if k not in known or k in shown_elsewhere:
            continue   # not a framework-read var, or already in the identity line
        v = info.env[k]
        if any(s in k.upper() for s in secret):
            v = "<set, redacted>" if v else "<empty>"
        items.append(f"{k}={v} [{known[k]}]")
    if not items:
        return [Check("daemon env", PASS,
                      "none of the framework's env vars are set — daemon fired "
                      "with defaults.")]
    return [Check("daemon env", PASS,
                  f"{len(items)} framework env var(s) at launch "
                  f"(name=value [tier]): " + ", ".join(items))]


def _pass_resolves(secret_path: str, home: str) -> Optional[bool]:
    """Return True/False if ``pass show <path>`` succeeds under ``home``.

    ``None`` when the ``pass`` binary is absent.  Never prints or returns
    the secret value — only the subprocess success flag.
    """
    env = dict(os.environ)
    env["HOME"] = home
    # PASSWORD_STORE_DIR, if the daemon set one, would override HOME-based
    # lookup; we honor the caller's current env except for HOME, mirroring
    # how the daemon's own environ drives resolution.
    try:
        proc = subprocess.run(
            ["pass", "show", secret_path],
            env=env, capture_output=True, timeout=15,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def check_secret(info: DaemonInfo, secret: Optional[str]) -> List[Check]:
    """Test whether a ``pass://`` secret resolves from daemon + shell HOME.

    ``secret`` may be a ``pass://path`` URI or a bare ``path``.  Resolves
    from the daemon's HOME (the one that actually matters at session time)
    and from the caller's HOME (to localize a mismatch).
    """
    if not secret:
        return [Check("secret resolution", WARN,
                      "no --secret given — pass `--secret pass://jaato/"
                      "<provider>/api-key` to test daemon-side resolution.")]
    path = secret[len("pass://"):] if secret.startswith("pass://") else secret

    my_home = os.environ.get("HOME", "")
    mine = _pass_resolves(path, my_home)
    if mine is None:
        return [Check("secret resolution", WARN,
                      "`pass` binary not found on PATH — cannot test "
                      f"resolution of {path!r}.")]

    results = []
    if info.home and info.home != my_home:
        daemon = _pass_resolves(path, info.home)
        if daemon:
            results.append(Check("secret resolution", PASS,
                                 f"{path} resolves from daemon HOME "
                                 f"{info.home}."))
        else:
            results.append(Check("secret resolution", FAIL,
                                 f"{path} does NOT resolve from daemon HOME "
                                 f"{info.home} (resolves from your HOME "
                                 f"{my_home}: {mine}).  The daemon will raise "
                                 f"SecretResolutionError at session time."))
    else:
        status = PASS if mine else FAIL
        verb = "resolves" if mine else "does NOT resolve"
        results.append(Check("secret resolution", status,
                             f"{path} {verb} from HOME {my_home}."))
    return results


def check_env_file(env_file: Optional[str], workspace: str) -> List[Check]:
    """Verify the env_file is a real path — None/absent breaks the handshake.

    The IPC handshake serializes the env_file path; ``env_file=None``
    raises an opaque ``os.PathLike`` ``TypeError`` during connect.  A
    missing file is also worth flagging before the run.
    """
    if env_file is None:
        return [Check("env_file", FAIL,
                      "env_file is None — the IPC handshake needs a real "
                      "path (passing None raises an opaque os.PathLike "
                      "TypeError).  Point it at a .env, even a minimal one.")]
    p = Path(env_file)
    if not p.is_absolute():
        p = Path(workspace) / env_file
    if p.exists():
        return [Check("env_file", PASS, f"{p}")]
    return [Check("env_file", WARN,
                  f"{p} does not exist yet — create it (JAATO_PROVIDER, "
                  f"MODEL_NAME, and the provider key) before connecting.")]


def check_workspace(workspace: str, config_root: Optional[str]) -> List[Check]:
    """Report where profiles are discovered and session logs are written.

    Both are anchored to the client's workspace (config_root overrides the
    profile search root), with ``~/.jaato`` as the profile fallback tier.
    Knowing these paths up front answers "where are my session logs?".
    """
    ws = Path(workspace).resolve()
    profiles_root = Path(config_root).resolve() if config_root else ws / ".jaato"
    log_dir = os.environ.get("JAATO_SESSION_LOG_DIR", ".jaato/logs")
    log_path = Path(log_dir) if Path(log_dir).is_absolute() else ws / log_dir
    return [
        Check("workspace", PASS, str(ws)),
        Check("profiles", PASS,
              f"{profiles_root}/profiles/  (fallback: ~/.jaato/profiles/)"),
        Check("session logs", PASS,
              f"{log_path}/session_<id>_client_<n>.log"),
    ]


# --------------------------------------------------------------------------
# Orchestration + CLI
# --------------------------------------------------------------------------

def run_checks(
    *,
    socket_path: str,
    pidfile: str,
    workspace: str,
    config_root: Optional[str],
    env_file: Optional[str],
    secret: Optional[str],
    auto_start: bool,
) -> List[Check]:
    """Run every check and return the flat result list (in display order)."""
    info = probe_daemon(socket_path, pidfile)
    checks: List[Check] = []
    checks += check_python_env()
    checks += check_socket(info, auto_start=auto_start)
    checks += check_daemon_identity(info)
    checks += check_home_match(info)
    checks += check_daemon_env(info, load_known_env_vars())
    checks += check_secret(info, secret)
    checks += check_env_file(env_file, workspace)
    checks += check_workspace(workspace, config_root)
    return checks


def _print(checks: List[Check]) -> int:
    """Print results; return a process exit code (1 if any FAIL)."""
    width = max((len(c.name) for c in checks), default=0)
    n_fail = n_warn = 0
    print("jaato SDK doctor\n" + "─" * 60)
    for c in checks:
        if c.status == FAIL:
            n_fail += 1
        elif c.status == WARN:
            n_warn += 1
        print(f"{_GLYPH[c.status]} {c.status}  {c.name.ljust(width)}  {c.detail}")
    print("─" * 60)
    verdict = "FAIL" if n_fail else ("WARN" if n_warn else "OK")
    print(f"{verdict}: {n_fail} fail, {n_warn} warn, {len(checks)} checks")
    return 1 if n_fail else 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``python -m jaato_sdk.doctor [options]``."""
    ap = argparse.ArgumentParser(
        prog="jaato-doctor",
        description="Preflight diagnostic for jaato SDK clients — verifies "
                    "daemon reachability, daemon HOME vs yours (pass:// "
                    "resolution), socket staleness, env_file, and config "
                    "paths.",
    )
    ap.add_argument("--socket", default=DEFAULT_SOCKET_PATH,
                    help=f"IPC socket path (default: {DEFAULT_SOCKET_PATH})")
    ap.add_argument("--pidfile", default=DEFAULT_PID_FILE,
                    help=f"daemon pidfile (default: {DEFAULT_PID_FILE})")
    ap.add_argument("--workspace", default=os.getcwd(),
                    help="client workspace_path (default: cwd)")
    ap.add_argument("--config-root", default=None,
                    help="config_root override (default: <workspace>/.jaato)")
    ap.add_argument("--env-file", default=".env",
                    help="env_file the client will pass (default: .env)")
    ap.add_argument("--secret", default=None,
                    help="a pass:// secret to test resolution for, e.g. "
                         "pass://jaato/nebius/api-key")
    ap.add_argument("--no-auto-start", action="store_true",
                    help="diagnose as if the client has auto_start=False")
    args = ap.parse_args(argv)

    checks = run_checks(
        socket_path=args.socket,
        pidfile=args.pidfile,
        workspace=args.workspace,
        config_root=args.config_root,
        env_file=args.env_file,
        secret=args.secret,
        auto_start=not args.no_auto_start,
    )
    return _print(checks)


if __name__ == "__main__":
    raise SystemExit(main())
