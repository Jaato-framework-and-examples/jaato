#!/usr/bin/env python3
"""Measure what a kernel boundary costs a Jaato session, next to a container.

Run on a Linux host where Jaato's confinement works (AppArmor loaded with the
jaato sudoers rule, or SELinux enforcing with the jaato module installed):

    python confinement_bench.py                 # 10 rounds per case
    python confinement_bench.py --rounds 20 --docker

It starts its OWN daemon on a private socket (twice: runner pool off, then on),
so nothing you already run is touched.  Every case drives the credential-free
``echo`` provider, so no API key is needed and no model latency is measured.

Per round it times:
  create  connect-to-``session.new`` answered (runner forked, confined, ready)
  first   first ``ask`` answered (the runner actually serving a turn)

and records the confinement line the daemon sends, so an "apparmor" row that
was not really confined cannot pass silently.  With --docker it also times
``docker run --rm alpine true`` (image pulled once first, not timed).

Output: a table on stdout and confinement_bench.json beside this script.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROFILE = {"model": "echo", "provider": "echo", "plugins": []}


def _ver(dist: str) -> str:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:  # noqa: BLE001
        return "?"


def lsm() -> str:
    try:
        return Path("/sys/kernel/security/lsm").read_text().strip()
    except OSError:
        return "unknown"


class Daemon:
    def __init__(self, root: Path, pool: bool) -> None:
        self.root, self.pool = root, pool
        self.socket = str(root / "jaato.sock")
        self.pidfile = str(root / "jaato.pid")
        self.log = str(root / "daemon.log")

    def start(self) -> None:
        env = dict(os.environ)
        env["JAATO_RUNNER_POOL_ENABLED"] = "true" if self.pool else "false"
        subprocess.run([sys.executable, "-m", "jaato_server", "--ipc-socket", self.socket,
                        "--pid-file", self.pidfile, "--log-file", self.log, "--daemon"],
                       env=env, check=True,
                       cwd=str(self.root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 120
        while time.time() < deadline:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.settimeout(0.5)
                s.connect(self.socket)
                return
            except OSError:
                time.sleep(0.25)
            finally:
                s.close()
        raise RuntimeError(f"daemon on {self.socket} never accepted a connection")

    def stop(self) -> None:
        subprocess.run([sys.executable, "-m", "jaato_server", "--stop", "--ipc-socket",
                        self.socket, "--pid-file", self.pidfile], cwd=str(self.root),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def one_round(sock: str, workspace: Path, confine: bool) -> dict:
    from jaato_sdk.client.convenience import Session
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType, EventType

    notes: list[str] = []
    c = IPCClient(socket_path=sock, client_type=ClientType.API, auto_start=False,
                  workspace_path=str(workspace), apparmor=confine)
    if not await c.connect(timeout=15):
        raise RuntimeError("could not connect")

    def on_output(ev):
        text = getattr(ev, "text", "") or getattr(ev, "message", "") or ""
        low = text.lower()
        if any(k in low for k in ("apparmor", "selinux", "confine", "sandbox")):
            notes.append(text.strip()[:200])

    c.subscribe(EventType.AGENT_OUTPUT, on_output)
    c.subscribe(EventType.SYSTEM_MESSAGE, on_output)
    t0 = time.perf_counter()
    sid = await c.create_session(profile=dict(PROFILE), timeout=180)
    t1 = time.perf_counter()
    s = Session(c, sid, on_permission=lambda ev: "y")
    await s.ask("hi", timeout=180)
    t2 = time.perf_counter()
    await asyncio.sleep(0.2)
    try:
        await c.end_session()
    except Exception:  # noqa: BLE001
        pass
    try:
        await c.disconnect()
    except Exception:  # noqa: BLE001
        pass
    return {"create_s": t1 - t0, "first_answer_s": t2 - t0, "confinement_lines": notes}


def summarise(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "min": xs[0], "median": statistics.median(xs),
            "p90": xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))], "max": xs[-1]}


def run_case(daemon: Daemon, workspace: Path, confine: bool, rounds: int) -> dict:
    rows = []
    for i in range(rounds + 1):  # round 0 is a warm-up, not counted
        r = asyncio.run(one_round(daemon.socket, workspace, confine))
        if i:
            rows.append(r)
        time.sleep(0.5)
    lines = sorted({l for r in rows for l in r["confinement_lines"]})
    return {"create": summarise([r["create_s"] for r in rows]),
            "first_answer": summarise([r["first_answer_s"] for r in rows]),
            "confinement_lines": lines}


def docker_case(rounds: int) -> dict | None:
    if not shutil.which("docker"):
        return None
    subprocess.run(["docker", "pull", "-q", "alpine"], check=True, stdout=subprocess.DEVNULL)
    xs = []
    for i in range(rounds + 1):
        t0 = time.perf_counter()
        subprocess.run(["docker", "run", "--rm", "alpine", "true"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if i:
            xs.append(time.perf_counter() - t0)
    return {"docker_run_rm_alpine_true": summarise(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--docker", action="store_true", help="also time docker run --rm alpine true")
    ap.add_argument("--no-pool-run", action="store_true", help="skip the pool-enabled daemon")
    ap.add_argument("--root", default=str(Path.home() / "jaato-confinement-bench"),
                    help="where the per-daemon roots and workspaces go (default ~/jaato-confinement-bench). "
                         "Not under /tmp: on SELinux a confined runner may not search a user_tmp_t "
                         "parent, and since #1522 the daemon refuses such a workspace")
    args = ap.parse_args()

    import jaato_server
    import jaato_sdk
    meta = {"host": platform.node(), "kernel": platform.release(), "lsm": lsm(),
            "python": platform.python_version(),
            "jaato_server": _ver("jaato-server"),
            "jaato_sdk": _ver("jaato-sdk"),
            "confinement_env": os.environ.get("JAATO_CONFINEMENT", "auto"),
            "apparmor_profile_grace_s": os.environ.get("JAATO_APPARMOR_PROFILE_GRACE_SECONDS", "default"),
            "rounds": args.rounds, "when": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    results: dict = {"meta": meta, "cases": {}}

    pools = [False] if args.no_pool_run else [False, True]
    for pool in pools:
        base = Path(args.root).resolve()
        base.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="jaato-bench-", dir=str(base)))
        ws = root / "workspace"
        ws.mkdir()
        d = Daemon(root, pool)
        d.start()
        try:
            for confine in (False, True):
                name = f"{'pool' if pool else 'cold'}/{'confined' if confine else 'unconfined'}"
                print(f"… {name}", file=sys.stderr, flush=True)
                results["cases"][name] = run_case(d, ws, confine, args.rounds)
        finally:
            d.stop()
        try:
            lines = [l.strip() for l in Path(d.log).read_text(errors="replace").splitlines()
                     if "provision timings" in l]
        except OSError:
            lines = []
        results.setdefault("provision_timings", {})["pool" if pool else "cold"] = lines

    if args.docker:
        results["docker"] = docker_case(args.rounds)

    out = Path(__file__).with_name("confinement_bench.json")
    out.write_text(json.dumps(results, indent=2))

    print(json.dumps(meta, indent=2))
    print(f"\n{'case':<22}{'create median':>15}{'p90':>9}{'first answer median':>22}{'p90':>9}")
    for name, c in results["cases"].items():
        print(f"{name:<22}{c['create']['median']:>14.3f}s{c['create']['p90']:>8.3f}s"
              f"{c['first_answer']['median']:>21.3f}s{c['first_answer']['p90']:>8.3f}s")
    for name, c in results["cases"].items():
        print(f"\n{name} confinement lines: {c['confinement_lines'] or 'NONE SEEN'}")
    if results.get("docker"):
        d = results["docker"]["docker_run_rm_alpine_true"]
        print(f"\ndocker run --rm alpine true: median {d['median']:.3f}s  p90 {d['p90']:.3f}s")
    for name, lines in results.get("provision_timings", {}).items():
        print(f"\n{name} daemon, AppArmor provision timings (#1501): {len(lines)} line(s)")
        for l in lines[-4:]:
            print("   ", l[-200:])
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
