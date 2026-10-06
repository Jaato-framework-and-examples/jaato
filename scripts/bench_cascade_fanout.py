#!/usr/bin/env python3
"""Measure a cascade fan-out: K parallel stages in ONE workspace.

The question this answers is how the runner pool serves stages that run
at the same time and share every part of the slot key (cascade, config
root, workspace, profile).  Sequential stages already reuse one warm
slot; parallel ones each need a runner of their own today.  Before
changing that, the cost has to be measured: how many cold spawns a
fan-out pays, how long until every stage is running, and how much
memory the runners hold while the stages mostly wait on the model.

The model is the ``echo`` provider with simulated latency
(``plugin_configs.echo.delay_ms`` / ``jitter_ms`` / ``seed``), so every
turn waits like a real inference call and a run costs nothing.

The script starts its OWN daemon (private socket, private ``HOME``,
temporary workspace) for each configuration, so runs do not share pool
state, and stops it afterwards.

Per configuration it prints one row:

    k            parallel stages
    jitter_ms    the latency spread used
    open_p50/max time from asking to a confirmed session (s)
    all_open     time until the LAST stage was confirmed (s)
    turn_p50/max one turn's latency, as the stage saw it (s)
    wall         time until every stage finished (s)
    misses       pool acquire misses during the run (cold spawns)
    runners_max  most runner processes alive at once (template excluded)
    priv_mb_max  most Private_Dirty summed over those runners (MB)

Usage:
    .venv/bin/python scripts/bench_cascade_fanout.py --k 2 4 8 \\
        --turns 3 --delay-ms 3000 --jitter-ms 0 2000 --pool-size 2

Linux only (reads ``/proc``).  Not a test: nothing asserts, it reports.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import ClientType


# ---------------------------------------------------------------- processes


def _ppid(pid: int) -> Optional[int]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[1])


def _runner_pids(daemon_pid: int) -> List[int]:
    """Runner processes of one daemon, the pre-warm template excluded.

    Pool slots are forked from the template and keep its command line
    (``--template-mode``), so the template is told apart by its parent:
    it is the daemon's child, while slots are the template's children.
    A cold-spawned runner is the daemon's child with no such flag.
    """
    out = subprocess.run(["pgrep", "-f", "jaato_server.server.runner"],
                         capture_output=True, text=True).stdout.split()
    pids = []
    for raw in out:
        pid = int(raw)
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        parent = _ppid(pid)
        if parent == daemon_pid and b"--template-mode" in cmd:
            continue  # the template itself
        # keep only this daemon's descendants (child or grandchild)
        if parent != daemon_pid and _ppid(parent or 0) != daemon_pid:
            continue
        pids.append(pid)
    return pids


def _private_dirty_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Private_Dirty:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


@dataclass
class Sampler:
    """Samples runner count and private memory every ``interval`` seconds."""

    daemon_pid: int = 0
    interval: float = 0.25
    runners_max: int = 0
    priv_kb_max: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            pids = _runner_pids(self.daemon_pid)
            self.runners_max = max(self.runners_max, len(pids))
            self.priv_kb_max = max(
                self.priv_kb_max, sum(_private_dirty_kb(p) for p in pids))
            self._stop.wait(self.interval)


# ------------------------------------------------------------------- daemon


class Daemon:
    """A private daemon: own socket, own HOME, own workspace."""

    def __init__(self, pool_size: int, pool_max: Optional[int],
                 python: str) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="jaato-fanout-"))
        self.home = self.root / "home"
        self.ws = self.root / "ws"
        self.sock = self.root / "j.sock"
        self.log = self.root / "daemon.log"
        self.home.mkdir()
        (self.ws / ".jaato").mkdir(parents=True)
        (self.ws / ".env").write_text("JAATO_PROVIDER=echo\nMODEL_NAME=echo\n")
        env = dict(os.environ, HOME=str(self.home),
                   JAATO_RUNNER_POOL_SIZE=str(pool_size))
        if pool_max is not None:
            env["JAATO_RUNNER_POOL_MAX_SIZE"] = str(pool_max)
        self.env = env
        self._log_fh = open(self.log, "w")
        self.proc = subprocess.Popen(
            [python, "-m", "jaato_server", "--ipc-socket", str(self.sock)],
            env=env, stdout=self._log_fh, stderr=subprocess.STDOUT)
        self.pool_size = pool_size

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Wait until the socket exists and the pool has filled."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"daemon exited; see {self.log}")
            if self.sock.exists() and (
                    self.pool_size == 0
                    or len(_runner_pids(self.proc.pid)) >= self.pool_size):
                time.sleep(1.0)
                return
            time.sleep(0.25)
        raise TimeoutError(f"daemon not ready; see {self.log}")

    def stop(self, keep: bool) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        for pid in _runner_pids(self.proc.pid):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        self._log_fh.close()
        if not keep:
            shutil.rmtree(self.root, ignore_errors=True)


# -------------------------------------------------------------------- stage


@dataclass
class StageResult:
    open_s: float
    opened_at: float
    turns_s: List[float]
    error: str = ""


async def _stage(daemon: Daemon, index: int, cascade: str, turns: int,
                 echo: Dict, plugins: List[str], t0: float,
                 session_timeout: float) -> StageResult:
    """One cascade stage: open a session, take ``turns`` turns, close."""
    profile = {"model": "echo", "provider": "echo", "plugins": plugins,
               "plugin_configs": {"echo": echo}}
    start = time.monotonic()
    turns_s: List[float] = []
    try:
        async with IPCClient.session(
                socket_path=str(daemon.sock), workspace_path=str(daemon.ws),
                env_file=str(daemon.ws / ".env"), auto_start=False,
                client_type=ClientType.API, profile=profile,
                cascade_driver_id=cascade,
                session_timeout=session_timeout) as s:
            opened = time.monotonic()
            for turn in range(turns):
                ts = time.monotonic()
                await s.ask(f"stage {index} turn {turn}",
                            timeout=session_timeout)
                turns_s.append(time.monotonic() - ts)
        return StageResult(opened - start, opened - t0, turns_s)
    except Exception as exc:  # report, never abort the whole run
        return StageResult(time.monotonic() - start, time.monotonic() - t0,
                           turns_s, error=f"{type(exc).__name__}: {exc}")


async def _pool_misses(daemon: Daemon) -> Optional[int]:
    """The pool's acquire-miss counter, or None if it cannot be read."""
    client = IPCClient(str(daemon.sock), auto_start=False,
                       workspace_path=str(daemon.ws),
                       client_type=ClientType.API)
    try:
        await client.connect()
        status = await client.pool_status()
        return int((status.telemetry or {}).get("pool_acquire_miss_total", 0))
    except Exception:
        return None
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass


async def _run_fanout(daemon: Daemon, k: int, turns: int, echo: Dict,
                      plugins: List[str], session_timeout: float) -> Dict:
    misses_before = await _pool_misses(daemon)
    sampler = Sampler(daemon_pid=daemon.proc.pid)
    sampler.start()
    cascade = uuid.uuid4().hex
    t0 = time.monotonic()
    results = await asyncio.gather(*(
        _stage(daemon, i, cascade, turns, echo, plugins, t0,
               session_timeout)
        for i in range(k)))
    wall = time.monotonic() - t0
    sampler.stop()
    misses_after = await _pool_misses(daemon)
    opens = [r.open_s for r in results]
    turns_all = [t for r in results for t in r.turns_s] or [0.0]
    misses = (None if misses_before is None or misses_after is None
              else misses_after - misses_before)
    return {
        "k": k,
        "open_p50": statistics.median(opens),
        "open_max": max(opens),
        "all_open": max(r.opened_at for r in results),
        "turn_p50": statistics.median(turns_all),
        "turn_max": max(turns_all),
        "wall": wall,
        "misses": misses,
        "runners_max": sampler.runners_max,
        "priv_mb_max": sampler.priv_kb_max / 1024.0,
        "errors": [r.error for r in results if r.error],
    }


# --------------------------------------------------------------------- main


def _row(jitter: float, r: Dict) -> str:
    misses = "?" if r["misses"] is None else str(r["misses"])
    return (f"{r['k']:>3} {jitter:>9.0f} {r['open_p50']:>8.2f} "
            f"{r['open_max']:>8.2f} {r['all_open']:>8.2f} "
            f"{r['turn_p50']:>8.2f} {r['turn_max']:>8.2f} "
            f"{r['wall']:>7.2f} {misses:>6} {r['runners_max']:>11} "
            f"{r['priv_mb_max']:>11.1f}")


HEADER = ("  k jitter_ms open_p50 open_max all_open turn_p50 turn_max "
          "   wall misses runners_max priv_mb_max")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--delay-ms", type=float, default=3000)
    ap.add_argument("--jitter-ms", type=float, nargs="+", default=[0, 2000])
    ap.add_argument("--seed", default="fanout")
    ap.add_argument("--pool-size", type=int, default=2)
    ap.add_argument("--pool-max", type=int, default=None)
    ap.add_argument("--session-timeout", type=float, default=180)
    ap.add_argument("--plugins", nargs="*",
                    default=["cli", "file_edit", "todo"],
                    help="plugins each stage's profile enables")
    ap.add_argument("--keep", action="store_true",
                    help="keep each daemon's directory (log, workspace)")
    args = ap.parse_args(argv)

    print(f"# turns={args.turns} delay_ms={args.delay_ms:.0f} "
          f"pool_size={args.pool_size} pool_max={args.pool_max} "
          f"plugins={args.plugins}")
    print(HEADER, flush=True)
    for jitter in args.jitter_ms:
        for k in args.k:
            daemon = Daemon(args.pool_size, args.pool_max, sys.executable)
            try:
                daemon.wait_ready()
                echo = {"delay_ms": args.delay_ms, "jitter_ms": jitter,
                        "seed": args.seed, "response": "ok"}
                result = asyncio.run(_run_fanout(
                    daemon, k, args.turns, echo, args.plugins,
                    args.session_timeout))
                print(_row(jitter, result), flush=True)
                for err in result["errors"]:
                    print(f"#   error: {err}", flush=True)
            finally:
                daemon.stop(keep=args.keep)
                if args.keep:
                    print(f"#   kept {daemon.root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
