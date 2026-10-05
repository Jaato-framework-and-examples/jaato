#!/usr/bin/env python3
"""What #1503's filter costs a spawn, and a syscall-heavy payload.

Run with the jaato venv's interpreter on the test branch (no daemon needed):

    python seccomp_microbench.py            # 1000 spawns per case
    python seccomp_microbench.py --n 3000

Compares, for ``subprocess.run(["/bin/true"])``:

  none     no preexec_fn (posix_spawn/vfork path)
  noop     a no-op preexec_fn (forces fork+exec, as Jaato's LSM transition
           callback already does for every model-driven subprocess)
  filter   preexec_fn = the compiled #1503 filter's install()

The cost of the filter is ``filter - noop``: Jaato already pays ``noop -
none`` for the LSM transition.  It also times a syscall-heavy loop
(getppid x N) in a child with and without the filter, since a BPF filter is
evaluated on every syscall for the life of the process, not only at install.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time

from jaato_server.shared.seccomp_filter import compile_filter

LOOP = "import os,time\nt=time.perf_counter()\nfor _ in range({n}): os.getppid()\nprint(time.perf_counter()-t)"


def spawn_ms(pre, n):
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        subprocess.run(["/bin/true"], preexec_fn=pre, check=True)
        xs.append(time.perf_counter() - t)
    return statistics.median(xs) * 1e3


def loop_s(pre, n):
    r = subprocess.run([sys.executable, "-c", LOOP.format(n=n)], preexec_fn=pre,
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--loop", type=int, default=2_000_000)
    a = ap.parse_args()
    cf = compile_filter()
    noop = lambda: None  # noqa: E731
    res = {"spawn_median_ms": {}, "syscall_loop_s": {}}
    for case, pre in (("none", None), ("noop", noop), ("filter", cf.install)):
        res["spawn_median_ms"][case] = round(spawn_ms(pre, a.n), 4)
    for case, pre in (("noop", noop), ("filter", cf.install)):
        res["syscall_loop_s"][case] = round(min(loop_s(pre, a.loop) for _ in range(3)), 4)
    s = res["spawn_median_ms"]
    l = res["syscall_loop_s"]
    res["filter_cost_per_spawn_ms"] = round(s["filter"] - s["noop"], 4)
    res["filter_cost_per_syscall_ns"] = round((l["filter"] - l["noop"]) / a.loop * 1e9, 2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
