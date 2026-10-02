#!/usr/bin/env python3
"""Run one confined jaato session on an SELinux host (phase 2b).

Run as root, with the jaato venv's interpreter, on a host where the module
is loaded (docs/design/selinux-phase2b-handoff.md).  It starts a jaato
daemon of its OWN (private socket, PID file and log; never the default
``/tmp/jaato.sock``), opens one IPC session that asks for kernel
confinement, drives one ``echo``-provider turn whose tool call runs
``id -Z`` and writes into the workspace, and checks what the kernel says:

* the runner process runs in ``jaato_runner_t`` at a per-workspace level;
* the command the model ran runs in ``jaato_child_t`` at the same level;
* the workspace is labelled at that level, authored config read-only;
* the session record says ``sandbox_mode: selinux``.

    /opt/jaato/venv/bin/python live_session.py --root /srv/jaato-2b

Prints one PASS / FAIL line per check, then the daemon's SELinux log lines
and the AVCs the run produced.  Writes ``live-session.json`` beside it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

RESULTS: List[dict] = []
OUTPUT: List[str] = []
RUNNER_CONTEXTS: Dict[int, str] = {}
COMMAND = "id -Z; echo probe > selinux-probe.txt && echo WSOK"


def record(name: str, ok: Optional[bool], observed: str) -> None:
    state = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append({"check": name, "result": state, "observed": observed})
    print(f"{state:4}  {name}\n      {observed.strip()[:300]}")


def context_of(pid: int) -> Optional[str]:
    try:
        return open(f"/proc/{pid}/attr/current").read().strip("\0\n ")
    except OSError:
        return None


def sample_runners(daemon_pid: int, stop: threading.Event) -> None:
    """Record the context of every runner the daemon starts."""
    while not stop.is_set():
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
            except OSError:
                continue
            if b"server.runner" in cmd and b"template-mode" not in cmd:
                ctx = context_of(pid)
                if ctx:
                    RUNNER_CONTEXTS[pid] = ctx
        stop.wait(0.2)


async def run_session(sock: Path, ws: Path) -> dict:
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import (
        ClientType, ErrorEvent, PermissionRequestedEvent, SessionInfoEvent,
        ToolCallEndEvent, ToolOutputEvent, TurnCompletedEvent,
    )

    c = IPCClient(socket_path=str(sock), client_type=ClientType.API,
                  workspace_path=str(ws), auto_start=False, apparmor=True)
    assert await c.connect(timeout=60), "connect failed"
    out: dict = {"ends": [], "errors": [], "sessions": []}
    done = asyncio.Event()

    async def collect() -> None:
        async for ev in c.events():
            if isinstance(ev, PermissionRequestedEvent):
                await c.respond_to_permission(ev.request_id, "a")
            elif isinstance(ev, ToolCallEndEvent):
                out["ends"].append({"tool": ev.tool_name, "success": ev.success})
            elif isinstance(ev, ToolOutputEvent):
                OUTPUT.append(ev.chunk or "")
            elif isinstance(ev, SessionInfoEvent):
                out["sessions"] = [s for s in (ev.sessions or []) if isinstance(s, dict)]
            elif isinstance(ev, ErrorEvent):
                out["errors"].append(str(getattr(ev, "error", ev)))
                done.set()
            elif isinstance(ev, TurnCompletedEvent):
                done.set()

    try:
        await c.create_session(profile="selinux2b")
        task = asyncio.create_task(collect())
        await asyncio.sleep(0.5)
        await c.send_message("go")
        try:
            await asyncio.wait_for(done.wait(), timeout=180)
        except asyncio.TimeoutError:
            out["errors"].append("timeout waiting for the turn")
        await asyncio.sleep(1.0)
        task.cancel()
    finally:
        await c.disconnect()
    return out


def level_of(context: Optional[str]) -> str:
    return (context or "").split(":", 3)[3] if (context or "").count(":") >= 3 else ""


def type_of(context: Optional[str]) -> str:
    parts = (context or "").split(":")
    return parts[2] if len(parts) > 2 else ""


def _prepare(root: Path) -> Path:
    """A fresh echo-provider workspace whose one tool call runs COMMAND."""
    from jaato_sdk.conformance.daemon import echo_workspace

    shutil.rmtree(root, ignore_errors=True)
    ws = root / "ws"
    ws.mkdir(parents=True)
    echo_workspace(ws, tool_call={"name": "cli_based_tool", "args": {"command": COMMAND}},
                   response="done", plugins=["cli"], name="selinux2b")
    return ws


def _drive(root: Path, ws: Path, sock: Path, log: Path) -> Optional[dict]:
    """Start a private daemon, run the session, stop the daemon by group.

    ``None`` when the daemon exits before its socket appears (recorded).
    """
    env = dict(os.environ, JAATO_CONFINEMENT="selinux", JAATO_REQUIRE_CONFINEMENT="1")
    daemon = subprocess.Popen(
        [sys.executable, "-m", "jaato_server", "--ipc-socket", str(sock),
         "--pid-file", str(root / "d.pid"), "--log-file", str(log)],
        cwd=str(ws), env=env, stdout=open(root / "d.out", "wb"),
        stderr=subprocess.STDOUT, start_new_session=True)
    stop = threading.Event()
    threading.Thread(target=sample_runners, args=(daemon.pid, stop), daemon=True).start()
    try:
        for _ in range(600):
            if sock.exists() or daemon.poll() is not None:
                break
            time.sleep(0.1)
        if daemon.poll() is not None:
            record("the daemon starts with JAATO_CONFINEMENT=selinux", False,
                   (root / "d.out").read_text(errors="replace")[-1500:])
            return None
        time.sleep(1.5)
        return asyncio.run(run_session(sock, ws))
    finally:
        stop.set()
        try:
            os.killpg(daemon.pid, signal.SIGTERM)
            daemon.wait(timeout=20)
        except Exception:
            os.killpg(daemon.pid, signal.SIGKILL)


def _label(path: Path) -> str:
    return subprocess.run(["ls", "-Zd", str(path)], capture_output=True, text=True).stdout.strip()


def _check_processes(result: dict) -> str:
    """The runner's and the child's domains, the turn and the record."""
    runner_ctxs = sorted(set(RUNNER_CONTEXTS.values()))
    runner = next((c for c in runner_ctxs if type_of(c) == "jaato_runner_t"), None)
    level = level_of(runner)
    output = "".join(OUTPUT)
    child = next((l.strip() for l in output.splitlines() if "jaato_" in l), "")
    record("the runner runs in jaato_runner_t at a workspace level",
           bool(runner) and level.startswith("s0:c"), f"runner contexts seen: {runner_ctxs}")
    record("the model's command runs in jaato_child_t at the same level",
           type_of(child) == "jaato_child_t" and level_of(child) == level,
           f"id -Z said: {child!r}; output: {output!r}")
    record("the command wrote the workspace", "WSOK" in output, repr(output))
    record("the turn completed without errors", not result["errors"]
           and any(e["success"] for e in result["ends"]), json.dumps(result)[:600])
    modes = {s.get("sandbox_mode") for s in result["sessions"]}
    record("the session record says sandbox_mode: selinux", "selinux" in modes,
           f"sandbox_mode values: {sorted(m for m in modes if m)}")
    return level


def _check_labels(ws: Path, level: str) -> Dict[str, str]:
    """The workspace, its authored config and a file the child created."""
    labels = {str(p): _label(p) for p in (
        ws, ws / ".jaato" / "profiles", ws / "selinux-probe.txt")}
    at_level = bool(level)
    record("the workspace is labelled at the runner's level",
           at_level and f"jaato_workspace_t:{level}" in labels[str(ws)], labels[str(ws)])
    authored = labels[str(ws / ".jaato" / "profiles")]
    record("authored config is jaato_authored_t", "jaato_authored_t" in authored, authored)
    created = labels[str(ws / "selinux-probe.txt")]
    record("a file the child created carries the level",
           at_level and f":{level}" in created, created)
    return labels


def _report(root: Path, log: Path, marker: str, labels: Dict[str, str]) -> int:
    """Print the daemon's SELinux lines and the AVCs; write the JSON."""
    dmesg = subprocess.run(["dmesg"], capture_output=True, text=True).stdout.splitlines()
    idx = max((i for i, l in enumerate(dmesg) if marker in l), default=-1)
    avcs = [l for l in dmesg[idx + 1:] if "avc:" in l and "jaato_" in l]
    log_lines = [l for l in log.read_text(errors="replace").splitlines()
                 if "selinux" in l.lower() or "confine" in l.lower()]
    print("\n--- daemon log (SELinux / confinement lines) ---")
    print("\n".join(log_lines[-40:]))
    print(f"\n--- AVCs naming a jaato type since the marker: {len(avcs)} ---")
    print("\n".join(avcs[-60:]))
    json.dump({"results": RESULTS, "runner_contexts": sorted(set(RUNNER_CONTEXTS.values())),
               "output": "".join(OUTPUT), "labels": labels, "avcs": avcs,
               "log": log_lines[-80:]},
              open(root / "live-session.json", "w"), indent=2)
    fails = [r for r in RESULTS if r["result"] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)} of {len(RESULTS)} not failed; "
          f"results in {root / 'live-session.json'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/srv/jaato-2b")
    args = ap.parse_args()
    if os.geteuid() != 0:
        print("run as root", file=sys.stderr)
        return 2
    root = Path(args.root)
    ws = _prepare(root)
    rundir = Path("/run/jaato-2b")
    rundir.mkdir(parents=True, exist_ok=True)
    log = root / "d.log"
    marker = f"jaato-2b-live start pid={os.getpid()} {time.time():.0f}"
    open("/dev/kmsg", "w").write(marker + "\n")
    result = _drive(root, ws, rundir / "d.sock", log)
    if result is None:
        return _report(root, log, marker, {})
    level = _check_processes(result)
    return _report(root, log, marker, _check_labels(ws, level))


if __name__ == "__main__":
    sys.exit(main())
