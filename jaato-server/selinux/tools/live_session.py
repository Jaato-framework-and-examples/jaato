#!/usr/bin/env python3
"""Run one confined jaato session on an SELinux host (phases 2b and 3).

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

    /opt/jaato/venv/bin/python live_session.py --root /srv/jaato-2b \
        [--as-uid 1000] [--trace-subprocess]

``--isolated rw|ro`` (phase 3) makes the model's one tool call a
``spawn_subagent`` with ``agent_params.isolated``, from a second echo
profile whose own tool call writes ``iso-probe.txt`` into the parent's
workspace; ``ro`` adds ``isolated_read_only_workspace``.  It then checks
that a sub-runner ran in ``jaato_isolated_t`` (or ``jaato_isolated_ro_t``)
at the parent's level, and that the file was written (``rw``) or refused
(``ro``).

``--as-uid`` gives the workspace to that uid and runs the daemon with
``--runner-uid-policy workspace-owner`` (#1168), so the runner and its
children are that user, not root.  ``--trace-subprocess`` drops a
``sitecustomize.py`` into the venv for the run that writes a stack for every
subprocess the runner starts, and for every listing of the runner's
``~/.jaato``, into its log (``<ws>/.jaato/logs/runner-*.log``), to name the
code behind an ``execute`` or ``read`` AVC; it is removed afterwards.

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
ISO_FILE = "iso-probe.txt"
CHILD_PROFILE = "selinux3child"


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


def _wait_for_isolated(ws: Path, mode: str) -> None:
    """Hold the session open while the isolated sub-runner works.

    Its turn is asynchronous to the parent's; disconnecting (or stopping
    the daemon) first would tear it down before it ran.  Waits for the
    probe file, or (``ro``, where the file must not appear) for the
    sub-runner's domain to be seen and a few seconds more.
    """
    want = "jaato_isolated_ro_t" if mode == "ro" else "jaato_isolated_t"
    probe = ws / ISO_FILE
    for _ in range(120):
        if probe.exists() or (mode == "ro" and any(
                type_of(c) == want for c in RUNNER_CONTEXTS.values())):
            break
        time.sleep(0.5)
    time.sleep(5.0 if mode == "ro" else 1.0)


async def run_session(sock: Path, ws: Path, isolated: Optional[str] = None) -> dict:
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
                out["ends"].append({"tool": ev.tool_name, "success": ev.success,
                                    "error": ev.error_message})
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
        try:
            await c.create_session(profile="selinux2b")
        except Exception as exc:  # SessionRefused: record it, keep reporting
            out["refused"] = f"{type(exc).__name__}: {exc}"
            return out
        task = asyncio.create_task(collect())
        await asyncio.sleep(0.5)
        await c.send_message("go")
        try:
            await asyncio.wait_for(done.wait(), timeout=180)
        except asyncio.TimeoutError:
            out["errors"].append("timeout waiting for the turn")
        if isolated:
            await asyncio.to_thread(_wait_for_isolated, ws, isolated)
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


def _prepare(root: Path, as_uid: Optional[int], isolated: Optional[str]) -> Path:
    """A fresh echo-provider workspace whose one tool call runs COMMAND, or
    (``isolated``) spawns an isolated subagent that writes ISO_FILE."""
    from jaato_sdk.conformance.daemon import echo_workspace

    shutil.rmtree(root, ignore_errors=True)
    ws = root / "ws"
    ws.mkdir(parents=True)
    if isolated:
        params = {"isolated": True}
        if isolated == "ro":
            params["isolated_read_only_workspace"] = True
        echo_workspace(ws, tool_call={"name": "spawn_subagent", "args": {
            "task": "write the probe file", "profile": CHILD_PROFILE,
            "agent_params": params}},
            response="spawned", plugins=["subagent"], name="selinux2b")
        echo_workspace(ws, tool_call={"name": "writeNewFile", "args": {
            "path": ISO_FILE, "content": "isolated\n"}},
            response="done", plugins=["file_edit"], name=CHILD_PROFILE,
            plugin_configs={"permission": {"policy": {"defaultPolicy": "allow"}}})
    else:
        echo_workspace(ws, tool_call={"name": "cli_based_tool", "args": {"command": COMMAND}},
                       response="done", plugins=["cli"], name="selinux2b")
    if as_uid is not None:
        for dirpath, dirnames, filenames in os.walk(ws):
            for name in [dirpath] + [os.path.join(dirpath, n) for n in dirnames + filenames]:
                os.chown(name, as_uid, as_uid)
    return ws


TRACE_HOOK = '''# jaato-2b live_session.py --trace-subprocess (removed after the run)
import os, sys, traceback
def _jaato_2b_trace(event, args):
    if event == "subprocess.Popen":
        sys.stderr.write("JAATO-2B-SUBPROCESS %r\\n%s" % (args[1], "".join(traceback.format_stack()[-12:])))
    elif event in ("os.scandir", "os.listdir") and str(args[0]).rstrip("/") == os.path.join(os.path.expanduser("~"), ".jaato"):
        sys.stderr.write("JAATO-2B-LISTDIR %r\\n%s" % (args[0], "".join(traceback.format_stack()[-12:])))
sys.addaudithook(_jaato_2b_trace)
'''


def _trace_hook_path() -> Path:
    import sysconfig
    return Path(sysconfig.get_paths()["purelib"]) / "sitecustomize.py"


def _install_trace_hook() -> Path:
    """Write the subprocess trace hook into this venv; refuse to overwrite."""
    path = _trace_hook_path()
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite it for --trace-subprocess")
    path.write_text(TRACE_HOOK)
    subprocess.run(["restorecon", str(path)], check=False)
    return path


def _drive(root: Path, ws: Path, sock: Path, log: Path,
           uid_policy: str, isolated: Optional[str] = None) -> Optional[dict]:
    """Start a private daemon, run the session, stop the daemon by group.

    The daemon runs in the foreground, where ``--log-file`` does not apply,
    so its log is its stdout, *log*.  ``None`` when the daemon exits before
    its socket appears (recorded).  ``OLDPWD`` is dropped: a child shell
    validates it at startup, and one inherited from the operator's shell
    produced ``dac_*`` AVCs nothing in jaato caused (phase 2b run).
    """
    env = dict(os.environ, JAATO_CONFINEMENT="selinux", JAATO_REQUIRE_CONFINEMENT="1")
    env.pop("OLDPWD", None)
    daemon = subprocess.Popen(
        [sys.executable, "-m", "jaato_server", "--ipc-socket", str(sock),
         "--pid-file", str(root / "d.pid"), "--runner-uid-policy", uid_policy],
        cwd=str(ws), env=env, stdout=open(log, "wb"),
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
                   log.read_text(errors="replace")[-1500:])
            return None
        time.sleep(1.5)
        return asyncio.run(run_session(sock, ws, isolated))
    finally:
        stop.set()
        try:
            os.killpg(daemon.pid, signal.SIGTERM)
            daemon.wait(timeout=20)
        except Exception:
            os.killpg(daemon.pid, signal.SIGKILL)


def _label(path: Path) -> str:
    return subprocess.run(["ls", "-Zd", str(path)], capture_output=True, text=True).stdout.strip()


def _check_spawn(result: dict) -> None:
    """The parent's spawn_subagent call, with the daemon's reason if refused."""
    spawns = [e for e in result["ends"] if e["tool"] == "spawn_subagent"]
    record("spawn_subagent returned", any(e["success"] for e in spawns),
           "; ".join(e["error"] or "ok" for e in spawns)
           or json.dumps(result)[:600])


def _check_processes(result: dict) -> str:
    """The runner's and the child's domains, the turn and the record."""
    record("the session was created", "refused" not in result,
           result.get("refused", "created"))
    runner_ctxs = sorted(set(RUNNER_CONTEXTS.values()))
    runner = next((c for c in runner_ctxs if type_of(c) == "jaato_runner_t"), None)
    level = level_of(runner)
    output = "\n".join(OUTPUT)
    child = next((l.strip() for l in output.splitlines() if "jaato_" in l), "")
    record("the runner runs in jaato_runner_t at a workspace level",
           bool(runner) and level.startswith("s0:c"), f"runner contexts seen: {runner_ctxs}")
    if result.get("isolated"):
        _check_spawn(result)
        return level
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


def _avcs_since(marker: str) -> List[str]:
    """The kernel's AVC lines naming a jaato type, after the run's marker."""
    dmesg = subprocess.run(["dmesg"], capture_output=True, text=True).stdout.splitlines()
    idx = max((i for i, l in enumerate(dmesg) if marker in l), default=-1)
    return [l for l in dmesg[idx + 1:] if "avc:" in l and "jaato_" in l]


def _check_isolated(ws: Path, level: str, mode: str, marker: str) -> None:
    """Phase 3: the sub-runner's domain, its log and what it could write."""
    want = "jaato_isolated_ro_t" if mode == "ro" else "jaato_isolated_t"
    ctxs = sorted(set(RUNNER_CONTEXTS.values()))
    iso = next((c for c in ctxs if type_of(c) == want), None)
    record(f"a sub-runner runs in {want} at the parent's level",
           bool(iso) and level_of(iso) == level, f"runner contexts seen: {ctxs}")
    _check_sub_runner_log(ws)
    if mode == "ro":
        _check_read_only_refused(ws, marker)
    else:
        probe = ws / ISO_FILE
        label = _label(probe) if probe.exists() else "(not written)"
        record("the sub-runner wrote its parent's workspace at the level",
               probe.exists() and f":{level}" in label, label)


def _check_sub_runner_log(ws: Path) -> None:
    """The log the daemon labelled jaato_runner_log_t, appended to (v3)."""
    logs = sorted((ws / ".jaato" / "logs").glob("runner-*__sub_*.log"))
    label = _label(logs[0]) if logs else "(no sub-runner log)"
    size = logs[0].stat().st_size if logs else 0
    record("the sub-runner wrote its own log (jaato_runner_log_t, not empty)",
           bool(logs) and "jaato_runner_log_t" in label and size > 0,
           f"{label}; bytes: {size}")


def _check_read_only_refused(ws: Path, marker: str) -> None:
    """The read-only sub-runner tried to write and the kernel refused it.

    An absent file alone is also what a sub-runner that never ran leaves,
    so the kernel's word is required. Creating a file is refused on its
    DIRECTORY (write / add_name), so the AVC names the directory.
    """
    probe = ws / ISO_FILE
    refused = [l for l in _avcs_since(marker)
               if "jaato_isolated_ro_t" in l and "permissive=0" in l
               and "tclass=dir" in l
               and ("{ write }" in l or "add_name" in l)]
    record("the read-only sub-runner tried to write the workspace and was refused",
           not probe.exists() and bool(refused),
           f"{ISO_FILE} exists: {probe.exists()}; refusals: {refused[:2]}")


def _check_labels(ws: Path, level: str, isolated: Optional[str] = None) -> Dict[str, str]:
    """The workspace, its authored config and a file the child created.

    In isolated mode no child runs a command, so the created-file check
    is ``_check_isolated``'s, on ``ISO_FILE``.
    """
    labels = {str(p): _label(p) for p in (
        ws, ws / ".jaato" / "profiles", ws / "selinux-probe.txt")}
    at_level = bool(level)
    record("the workspace is labelled at the runner's level",
           at_level and f"jaato_workspace_t:{level}" in labels[str(ws)], labels[str(ws)])
    authored = labels[str(ws / ".jaato" / "profiles")]
    record("the persona/profile config is jaato_agent_config_t",
           "jaato_agent_config_t" in authored, authored)
    if not isolated:
        created = labels[str(ws / "selinux-probe.txt")]
        record("a file the child created carries the level",
               at_level and f":{level}" in created, created)
    return labels


def _report(root: Path, log: Path, marker: str, labels: Dict[str, str]) -> int:
    """Print the daemon's SELinux lines and the AVCs; write the JSON."""
    avcs = _avcs_since(marker)
    log_lines = [l for l in log.read_text(errors="replace").splitlines()
                 if "selinux" in l.lower() or "confine" in l.lower()]
    print("\n--- daemon log (SELinux / confinement lines) ---")
    print("\n".join(log_lines[-40:]))
    print(f"\n--- AVCs naming a jaato type since the marker: {len(avcs)} ---")
    print("\n".join(avcs[-60:]))
    json.dump({"results": RESULTS, "runner_contexts": sorted(set(RUNNER_CONTEXTS.values())),
               "output": "\n".join(OUTPUT), "labels": labels, "avcs": avcs,
               "log": log_lines[-80:]},
              open(root / "live-session.json", "w"), indent=2)
    fails = [r for r in RESULTS if r["result"] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)} of {len(RESULTS)} not failed; "
          f"results in {root / 'live-session.json'}")
    return 1 if fails else 0


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/srv/jaato-2b")
    ap.add_argument("--as-uid", type=int,
                    help="give the workspace to this uid and drop the runner to it (#1168)")
    ap.add_argument("--trace-subprocess", action="store_true",
                    help="log a stack for every subprocess the runner starts")
    ap.add_argument("--isolated", choices=("rw", "ro"),
                    help="phase 3: spawn an isolated subagent instead of a cli command")
    return ap.parse_args()


def _ratelimit(value: Optional[str]) -> str:
    """Set ``kernel.printk_ratelimit`` (when *value*), return the old one."""
    path = "/proc/sys/kernel/printk_ratelimit"
    old = open(path).read().strip()
    if value is not None:
        open(path, "w").write(value)
    return old


def main() -> int:
    args = _parse()
    if os.geteuid() != 0:
        print("run as root", file=sys.stderr)
        return 2
    root = Path(args.root)
    ws = _prepare(root, args.as_uid, args.isolated)
    rundir = Path("/run/jaato-2b")
    rundir.mkdir(parents=True, exist_ok=True)
    log = root / "d.out"
    hook = _install_trace_hook() if args.trace_subprocess else None
    marker = f"jaato-2b-live start pid={os.getpid()} {time.time():.0f}"
    old_rate = _ratelimit("0")
    open("/dev/kmsg", "w").write(marker + "\n")
    try:
        result = _drive(root, ws, rundir / "d.sock", log,
                        "workspace-owner" if args.as_uid is not None else "daemon",
                        args.isolated)
    finally:
        _ratelimit(old_rate)
        if hook is not None:
            hook.unlink()
    if result is None:
        return _report(root, log, marker, {})
    result["isolated"] = args.isolated
    level = _check_processes(result)
    if args.isolated:
        _check_isolated(ws, level, args.isolated, marker)
    return _report(root, log, marker, _check_labels(ws, level, args.isolated))


if __name__ == "__main__":
    sys.exit(main())
