#!/usr/bin/env python3
"""Run one shell command where a Jaato model-driven payload runs, and report.

The command is executed by the ``cli`` tool inside a confined session — the
same path (``//child`` on AppArmor, ``jaato_child_t`` on SELinux, plus the
cgroup and, with #1503, the seccomp filter) a model's tool call takes.  No
model and no API key: the ``echo`` provider is scripted to make exactly one
``cli_based_tool`` call with the command (``jaato_sdk.conformance.daemon.
echo_workspace``, the mechanism ``selinux/tools/live_session.py`` uses).

    python confined_exec.py --command 'grep Seccomp /proc/self/status'
    python confined_exec.py --command 'unshare -U true; echo rc=$?' --json
    python confined_exec.py --command '...' --seccomp off
    python confined_exec.py --command '...' --fragments git,python --unconfined

It starts a private daemon (own socket under --root, never /tmp/jaato.sock),
runs one session per invocation, and stops the daemon unless --keep-daemon
(then pass the same --root to reuse it; --stop-daemon stops it).

Output: the command's combined output, then a trailer with the session's
sandbox mode, the confinement lines the daemon sent, and whether the tool
call succeeded.  --json prints one JSON object instead (for an adapter).
Exit status: 0 if the tool call ran (whatever the command's own status),
2 if the session was refused or reported unconfined while confinement was
asked for, 3 on timeout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

PROFILE_NAME = "confinedexec"
OUT_FILE = ".confined-exec.out"
RC_FILE = ".confined-exec.rc"
SCRIPT_FILE = ".confined-exec.sh"


def daemon_paths(root: Path) -> dict:
    return {"sock": root / "jaato.sock", "pid": root / "jaato.pid"}


def daemon_up(sock: Path) -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(str(sock))
        return True
    except OSError:
        return False
    finally:
        s.close()


def start_daemon(root: Path, pool: bool) -> None:
    p = daemon_paths(root)
    if daemon_up(p["sock"]):
        return
    env = dict(os.environ)
    env["JAATO_RUNNER_POOL_ENABLED"] = "true" if pool else "false"
    subprocess.run([sys.executable, "-m", "jaato_server", "--ipc-socket", str(p["sock"]),
                    "--pid-file", str(p["pid"]), "--log-file", str(root / "daemon.log"),
                    "--daemon"], env=env, check=True,
                   cwd=str(root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline:
        if daemon_up(p["sock"]):
            return
        time.sleep(0.25)
    raise RuntimeError(f"daemon on {p['sock']} never accepted a connection")


def stop_daemon(root: Path) -> None:
    p = daemon_paths(root)
    subprocess.run([sys.executable, "-m", "jaato_server", "--stop", "--ipc-socket",
                    str(p["sock"]), "--pid-file", str(p["pid"])], cwd=str(root),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_workspace(ws: Path, command: str, args: argparse.Namespace) -> None:
    from jaato_sdk.conformance.daemon import echo_workspace

    if ws.exists() and not args.keep_workspace:
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    for f in ws.glob(".confined-exec.*"):
        f.unlink()
    for src in args.stage or []:
        # Files the payload needs (a probe binary, a script), copied into the
        # workspace so the command can name them by a relative path.
        sp = Path(src)
        dst = ws / sp.name
        shutil.copy2(sp, dst)
    limits: dict = {"unload_grace_seconds": 0, "tool_timeout_seconds": args.timeout}
    if args.seccomp:
        limits["seccomp"] = args.seccomp
    if args.seccomp_allow:
        limits["seccomp_allow"] = [x for x in args.seccomp_allow.split(",") if x]
    # The cli tool's own output is not on the wire for a short command, so the
    # payload writes its output and exit status into the workspace (which
    # //child may write) and the result is read back from there.
    # The command goes in a script file inside the workspace: the cli plugin's
    # app-layer path check refuses an argv naming a path outside the
    # workspace (e.g. /proc/self/status) before anything runs, which would
    # test that heuristic instead of the kernel boundary.
    (ws / SCRIPT_FILE).write_text(command + "\n")
    wrapped = f"sh ./{SCRIPT_FILE} > {OUT_FILE} 2>&1; echo $? > {RC_FILE}"
    echo_workspace(ws, tool_call={"name": "cli_based_tool", "args": {"command": wrapped}},
                   response="done", plugins=["cli"], name=PROFILE_NAME,
                   runtime_limits=limits,
                   plugin_configs={"permission": {"policy": {"defaultPolicy": "allow"}}})
    if args.fragments is not None:
        # Per-stage exec scoping: a declared list, even empty, makes //child
        # exec authority fragment-only (template v18 / v34, #1251).
        prof = ws / ".jaato" / "profiles" / f"{PROFILE_NAME}.json"
        if prof.exists():
            data = json.loads(prof.read_text())
            data["apparmor_fragments"] = [x for x in args.fragments.split(",") if x]
            prof.write_text(json.dumps(data, indent=2))
        else:
            raise SystemExit(f"--fragments: expected {prof} (echo_workspace layout changed?)")


async def run(sock: Path, ws: Path, confine: bool, timeout: float) -> dict:
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import (ClientType, ErrorEvent, PermissionRequestedEvent,
                                  SessionInfoEvent, SystemMessageEvent, ToolCallEndEvent,
                                  ToolOutputEvent, TurnCompletedEvent)

    out: dict = {"output": "", "tool_end": None, "errors": [], "notes": [], "sessions": []}
    c = IPCClient(socket_path=str(sock), client_type=ClientType.API, workspace_path=str(ws),
                  auto_start=False, apparmor=confine)
    if not await c.connect(timeout=60):
        raise RuntimeError("connect failed")
    done = asyncio.Event()

    async def collect() -> None:
        async for ev in c.events():
            if isinstance(ev, PermissionRequestedEvent):
                await c.respond_to_permission(ev.request_id, "a")
            elif isinstance(ev, ToolOutputEvent):
                out["output"] += ev.chunk or ""
            elif isinstance(ev, ToolCallEndEvent):
                out["tool_end"] = {"tool": ev.tool_name, "success": ev.success,
                                   "error": ev.error_message}
            elif isinstance(ev, SystemMessageEvent):
                msg = getattr(ev, "message", "") or ""
                if any(k in msg.lower() for k in ("apparmor", "selinux", "confine", "seccomp")):
                    out["notes"].append(msg.strip())
            elif isinstance(ev, SessionInfoEvent):
                out["sessions"] = [s for s in (ev.sessions or []) if isinstance(s, dict)]
            elif isinstance(ev, ErrorEvent):
                out["errors"].append(str(getattr(ev, "error", ev)))
                done.set()
            elif isinstance(ev, TurnCompletedEvent):
                done.set()

    t0 = time.perf_counter()
    try:
        try:
            sid = await c.create_session(profile=PROFILE_NAME, timeout=180)
        except Exception as exc:  # noqa: BLE001 -- a refusal is a result
            out["refused"] = f"{type(exc).__name__}: {exc}"
            return out
        out["session_id"] = sid
        task = asyncio.create_task(collect())
        await asyncio.sleep(0.3)
        await c.send_message("go")
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout + 60)
        except asyncio.TimeoutError:
            out["errors"].append("timeout")
        await asyncio.sleep(0.5)
        task.cancel()
    finally:
        out["elapsed_s"] = round(time.perf_counter() - t0, 3)
        try:
            await c.disconnect()
        except Exception:  # noqa: BLE001
            pass
    me = next((s for s in out["sessions"] if s.get("session_id") == out.get("session_id")
               or s.get("id") == out.get("session_id")), {})
    out["sandbox_mode"] = me.get("sandbox_mode")
    out["seccomp"] = me.get("seccomp")
    return out


def tool_result_from_history(ws: Path, sid) -> dict:
    """The cli tool's own result from the persisted session, if any."""
    f = ws / ".jaato" / "sessions" / f"{sid}.json"
    if not sid or not f.exists():
        return {}
    try:
        hist = json.loads(f.read_text()).get("history", [])
    except (OSError, ValueError):
        return {}
    for msg in hist:
        for part in msg.get("parts", []) or []:
            r = part.get("result") if isinstance(part, dict) else None
            if isinstance(r, dict) and ("stdout" in r or "stderr" in r):
                return {k: r.get(k) for k in ("stdout", "stderr", "returncode", "exit_code") if k in r}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--command",
                    help="the shell command to run in the payload (required unless --stop-daemon)")
    ap.add_argument("--root", default=str(Path.home() / "jaato-confined-exec"))
    ap.add_argument("--unconfined", action="store_true", help="do not ask for confinement")
    ap.add_argument("--pool", action="store_true", help="daemon with the runner pool on")
    ap.add_argument("--seccomp", choices=("default", "off"), help="runtime_limits.seccomp (#1503)")
    ap.add_argument("--seccomp-allow", help="comma list of families to allow back (#1503)")
    ap.add_argument("--fragments", help="comma list -> apparmor_fragments (scoped stage); '' = none")
    ap.add_argument("--stage", action="append", help="copy this file into the workspace first (repeatable)")
    ap.add_argument("--keep-workspace", action="store_true",
                    help="reuse the workspace under --root instead of starting from an empty one")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--keep-daemon", action="store_true")
    ap.add_argument("--stop-daemon", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.command is None and not args.stop_daemon:
        ap.error("--command is required (except with --stop-daemon)")

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stop_daemon:
        stop_daemon(root)
        return 0
    ws = root / "ws"
    prepare_workspace(ws, args.command, args)
    start_daemon(root, args.pool)
    try:
        res = asyncio.run(run(daemon_paths(root)["sock"], ws, not args.unconfined, args.timeout))
    finally:
        if not args.keep_daemon:
            stop_daemon(root)
    res["command"] = args.command
    out_f, rc_f = ws / OUT_FILE, ws / RC_FILE
    res["output"] = out_f.read_text(errors="replace") if out_f.exists() else res["output"]
    if not out_f.exists():
        res["tool_result"] = tool_result_from_history(ws, res.get("session_id"))
    res["exit_status"] = int(rc_f.read_text().strip()) if rc_f.exists() and rc_f.read_text().strip().isdigit() else None
    res["confinement_requested"] = not args.unconfined

    status = 0
    if "refused" in res:
        status = 2
    elif not args.unconfined and res.get("sandbox_mode") not in ("apparmor", "selinux"):
        # Enforcing kernel boundary only: "soft", "apparmor-complain" and
        # "selinux-permissive" are NOT a boundary (#1014), and a missing mode
        # is not evidence of one (#1253 / #1299).
        res["errors"].append(
            f"confinement requested but sandbox_mode={res.get('sandbox_mode')!r}")
        status = 2
    elif "timeout" in res["errors"]:
        status = 3

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(res["output"], end="" if res["output"].endswith("\n") else "\n")
        print("----")
        for k in ("exit_status", "tool_result", "sandbox_mode", "seccomp", "tool_end", "elapsed_s", "refused", "errors", "notes"):
            if res.get(k) not in (None, [], ""):
                print(f"{k}: {res[k]}")
    return status


if __name__ == "__main__":
    sys.exit(main())
