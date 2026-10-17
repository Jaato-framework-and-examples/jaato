#!/usr/bin/env python3
"""Measure how the references plugin scales with the size of the catalog.

Phase 0 of docs/design/references-storage-scaling.md. It generates a
workspace catalog of N references, then drives the REAL references plugin
in-process (``create_plugin`` + ``initialize``, the tool executors, the
refresh check) and reports, per size:

- ``init_s`` / ``rss_mb``: ``initialize()`` wall time and the process RSS
  after it (each size runs in a fresh child process so RSS is its own);
- ``refresh_noop_ms``: the read-path refresh check when nothing changed
  (median of 20), and ``refresh_one_file_ms`` after one reference file is
  rewritten in place (the check plus the reload it triggers);
- ``list_ms`` / ``list_bytes`` / ``list_tokens_est``: ``listReferences``
  with no arguments, and the size of what the model would receive
  (tokens estimated at 4 bytes per token, the framework's own rule);
- ``select_*``: ``selectReferences`` on one id with transitive expansion,
  unbounded and with ``max_transitive_references: 25``;
- ``link_index_ms``: one ``LinkIndex`` build over the whole catalog, the
  cost every link-reading call pays;
- ``enrich_ms``: ``enrich_prompt`` on a prompt naming three ids.

What it does NOT measure (stated, not implied): vector search and
reconcile (no embedding provider is loaded), a daemon-side promotion, and
anything under AppArmor confinement. Those need a running daemon and are a
separate harness.

Generated catalog shape: references split into bundles of ``--per-bundle``
(each a directory with ``bundle.json``) under ``<ws>/.jaato/references``;
each reference is a LOCAL markdown body of ``--body-kb`` KB that mentions
``--mentions`` other ids by name, and declares 1..``--max-links`` typed
links. All mtimes are aged an hour into the past, so the 2 s "racily
clean" window of ``catalog_watch.settle`` does not force reloads.

Usage::

    .venv/bin/python scripts/bench_references_scale.py \\
        --root /path/to/scratch --sizes 1000 10000 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

AGE_SECONDS = 3600


def _rss_mb() -> float:
    with open("/proc/self/status", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def _ref_id(i: int) -> str:
    return f"ref-{i:06d}"


def generate(ws: Path, n: int, *, per_bundle: int, mentions: int,
             max_links: int, body_kb: int, seed: int) -> Dict[str, Any]:
    """Write a catalog of ``n`` references under ``ws``; return its shape."""
    rng = random.Random(seed)
    refs_root = ws / ".jaato" / "references"
    docs = ws / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    refs_root.mkdir(parents=True, exist_ok=True)
    filler = ("lorem ipsum dolor sit amet consectetur adipiscing elit " * 40)
    started = time.perf_counter()
    bundles = max(1, (n + per_bundle - 1) // per_bundle)
    for b in range(bundles):
        bdir = refs_root / f"bundle-{b:04d}"
        bdir.mkdir(exist_ok=True)
        (bdir / "bundle.json").write_text(
            json.dumps({"name": f"bundle-{b:04d}"}), encoding="utf-8")
    for i in range(n):
        rid = _ref_id(i)
        others = [_ref_id(rng.randrange(n)) for _ in range(mentions)]
        body = [f"# {rid}\n"]
        size = 0
        while size < body_kb * 1024:
            body.append(filler)
            size += len(filler)
        body.append("\nSee also " + ", ".join(others) + ".\n")
        doc = docs / f"{rid}.md"
        doc.write_text("".join(body), encoding="utf-8")
        links = [{"to": _ref_id(rng.randrange(n)),
                  "rel": rng.choice(["depends-on", "elaborates", "see-also"])}
                 for _ in range(rng.randint(1, max_links))]
        entry = {
            "id": rid, "name": f"Reference {i}", "description": f"about topic {i % 97}",
            "type": "local", "path": str(doc), "mode": "selectable",
            "tags": [f"t{i % 50}", f"area{i % 7}"], "links": links,
        }
        (refs_root / f"bundle-{i // per_bundle:04d}" / f"{rid}.json").write_text(
            json.dumps(entry), encoding="utf-8")
    gen_s = time.perf_counter() - started
    past = time.time() - AGE_SECONDS
    for dirpath, dirnames, filenames in os.walk(ws):
        for name in filenames + dirnames:
            os.utime(os.path.join(dirpath, name), (past, past))
        os.utime(dirpath, (past, past))
    return {"n": n, "bundles": bundles, "generate_s": round(gen_s, 2)}


def _timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t)


def measure(ws: Path) -> Dict[str, Any]:
    """Drive the real plugin against ``ws``; runs in a fresh process."""
    from jaato_server.shared.plugins.references.links import LinkIndex
    from jaato_server.shared.plugins.references.plugin import create_plugin
    from jaato_server.shared.session_context import isolated_current_session

    base_rss = _rss_mb()

    def session(**extra: Any):
        plugin = create_plugin()
        plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "tags_only", **extra})
        plugin.set_workspace_path(str(ws))
        return plugin

    out: Dict[str, Any] = {}
    plugin, dt = _timed(session)
    out["init_s"] = round(dt, 3)
    out["rss_mb"] = round(_rss_mb(), 1)
    out["rss_delta_mb"] = round(out["rss_mb"] - base_rss, 1)
    n = len(plugin._sources)
    out["loaded_sources"] = n

    noop = []
    for _ in range(20):
        _, dt = _timed(plugin._refresh_catalog_if_changed)
        noop.append(dt)
    out["refresh_noop_ms"] = round(statistics.median(noop) * 1000, 2)
    out["watched_paths"] = len(plugin._snapshot.watch)

    execs = plugin.get_executors()
    with isolated_current_session():
        listed, dt = _timed(execs["listReferences"], {})
    blob = json.dumps(listed, default=str)
    out["list_ms"] = round(dt * 1000, 1)
    out["list_bytes"] = len(blob)
    out["list_tokens_est"] = len(blob) // 4

    _, dt = _timed(LinkIndex, plugin._sources)
    out["link_index_ms"] = round(dt * 1000, 1)

    for label, extra in (("unbounded", {}), ("cap25", {"max_transitive_references": 25})):
        p = session(**extra)
        with isolated_current_session():
            res, dt = _timed(p.get_executors()["selectReferences"], {"ids": [_ref_id(0)]})
        out[f"select_{label}_ms"] = round(dt * 1000, 1)
        out[f"select_{label}_count"] = len(res.get("sources", res.get("selected", [])) or [])
        out[f"select_{label}_bytes"] = len(json.dumps(res, default=str))
        if label == "unbounded":
            out["rss_after_two_sessions_mb"] = round(_rss_mb(), 1)

    prompt = f"Please look at @{_ref_id(1)} and @{_ref_id(2)} and {_ref_id(3)}."
    with isolated_current_session():
        _, dt = _timed(plugin.enrich_prompt, prompt)
    out["enrich_ms"] = round(dt * 1000, 1)
    # One reference rewritten in place: the check plus the reload it triggers.
    some = next(p for p in (ws / ".jaato" / "references").rglob("ref-*.json"))
    data = json.loads(some.read_text(encoding="utf-8"))
    data["description"] += " (edited)"
    some.write_text(json.dumps(data), encoding="utf-8")
    _, dt = _timed(plugin._refresh_catalog_if_changed)
    out["refresh_one_file_ms"] = round(dt * 1000, 1)

    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="scratch directory for workspaces")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    ap.add_argument("--per-bundle", type=int, default=1000)
    ap.add_argument("--mentions", type=int, default=3)
    ap.add_argument("--max-links", type=int, default=3)
    ap.add_argument("--body-kb", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1145)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        print(json.dumps(measure(args.child)))
        return 0

    rows: List[Dict[str, Any]] = []
    for n in args.sizes:
        ws = args.root / f"ws-{n}"
        home = args.root / f"home-{n}"
        home.mkdir(parents=True, exist_ok=True)
        if not (ws / ".jaato" / "references").exists():
            shape = generate(ws, n, per_bundle=args.per_bundle, mentions=args.mentions,
                             max_links=args.max_links, body_kb=args.body_kb, seed=args.seed)
        else:
            shape = {"n": n, "generate_s": None}
        env = {**os.environ, "HOME": str(home)}
        env.pop("REFERENCES_CONFIG_PATH", None)
        proc = subprocess.run([sys.executable, __file__, "--root", str(args.root),
                               "--child", str(ws)], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
            return proc.returncode
        row = {**shape, **json.loads(proc.stdout.strip().splitlines()[-1])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
