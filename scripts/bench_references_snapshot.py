#!/usr/bin/env python3
"""Measure the ReferenceStore / CatalogSnapshot / WorkingSet design before building it.

Companion to ``bench_references_scale.py`` (phase 0). The design under test
(docs/design/references-storage-scaling.md, "The seam: ReferenceStore,
CatalogSnapshot, WorkingSet") is not implemented; this script prototypes the
two steps it would take and measures them against the real plugin, on the
catalogs ``bench_references_scale.py`` generates. Nothing here changes the
plugin.

Part ``snapshot`` (step 1, the seam with a full reload), per size, in a fresh
process:

- ``load_*``: the real plugin's full load, split into the parse (the
  store's ``reload()``; ``_reload_catalog`` + ``_discover_and_load_bundles``
  before the store existed) and the whole ``_reload_from_disk``;
- ``index_*``: building a snapshot's indexes over the loaded list (by-id
  dict, per-bundle index, ``LinkIndex``), and the memory they add
  (``tracemalloc``, over objects the list already holds);
- ``lookup_*``: one by-id lookup, as the plugin does it today
  (``next(s for s in self._sources if s.id == x)``) and through the dict.

Part ``derive`` (step 2, the next generation built from the previous one):

- ``detect_ms``: the stat of every watched path (today's refresh check);
- ``derive_<case>_ms``: (median of 5) building generation G+1 from G after the case
  (edit 1 / 10 / 100 files in place, add 1, delete 1): re-parse only the
  changed files, copy the by-id dict and replace their entries, rebuild the
  changed bundles' indexes and the whole ``LinkIndex``, and rebase a
  working set of 25 selected ids (drop the ids that left, note whether a
  selected reference's links changed);
- ``derive_<case>_mem_kb``: what G+1 allocates beyond G (shared entries
  are not copied);
- ``full_reload_<case>_s``: the plugin's ``_reload_from_disk`` on the same
  change, for comparison;
- ``derive_<case>_equal``: G+1 has the same ids, and the same parsed
  entries, as a full reload of the same disk;
- ``store_derive_<case>_ms`` / ``store_derive_<case>_equal``: the same, for
  the store's own derivation (``ReferenceStore._derive``, median of 5 from
  the same G), once it was built.

The prototype's per-file parse mirrors the body of the loop in
``config_loader.discover_references`` (``ReferenceSource.from_dict`` and the
LOCAL path resolution); the equality check is what keeps it honest.

Part ``daemon`` (who holds the snapshot): a private daemon
(``ConformanceDaemon``, own socket and pid file, killed by process group),
``--sessions`` sessions created one after another with a profile that lists
``references`` and as many with one that does not, the daemon process's RSS
after each. A daemon-side mirror registry initializes every discovered plugin
(#1566), so this shows whether each session costs the daemon a catalog copy.

Usage::

    .venv/bin/python scripts/bench_references_snapshot.py snapshot \\
        --root /path/to/scratch --sizes 1000 10000 100000 --out snap.json
    .venv/bin/python scripts/bench_references_snapshot.py daemon \\
        --root /path/to/scratch --size 10000 --sessions 3 --out daemon.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_references_scale import AGE_SECONDS, generate  # noqa: E402

CASES = (("edit1", 1), ("edit10", 10), ("edit100", 100), ("add1", 1), ("delete1", 1))


def _timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def _rss_mb(pid: str = "self") -> float:
    with open(f"/proc/{pid}/status", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


# ---------------------------------------------------------------- prototype


class Snapshot:
    """One immutable generation: entries by id, files by path, indexes."""

    def __init__(self, by_id: Dict[str, Any], file_ids: Dict[str, str],
                 bundle_of: Dict[str, str], link_index: Any):
        self.by_id = by_id            # id -> ReferenceSource
        self.file_ids = file_ids      # json path -> id
        self.bundle_of = bundle_of    # json path -> bundle name
        self.by_bundle: Dict[str, List[str]] = {}
        for path, rid in file_ids.items():
            self.by_bundle.setdefault(bundle_of[path], []).append(rid)
        self.links = link_index


def _parse_one(plugin, path: Path, bundle_name: str, project_root: str):
    """The loop body of ``discover_references`` for one bundle file.

    A bundle reference's path is not resolved for the session at load
    (only the root's are: ``store._append_bundle_sources``), so neither
    does this.
    """
    from jaato_server.shared.path_utils import normalize_path
    from jaato_server.shared.plugins.references.models import ReferenceSource, SourceType

    data = json.loads(path.read_text(encoding="utf-8"))
    source = ReferenceSource.from_dict(data)
    if source.type == SourceType.LOCAL and source.path:
        sp = Path(source.path)
        absolute = sp.resolve() if sp.is_absolute() else (path.parent / sp).resolve()
        source.resolved_path = normalize_path(os.path.relpath(absolute, project_root))
    source.bundle_name = bundle_name
    return source


def _bundle_dirs(ws: Path) -> Dict[str, str]:
    """Bundle directory -> bundle name, for the generated catalog shape."""
    refs = ws / ".jaato" / "references"
    return {str(d): d.name for d in sorted(refs.iterdir()) if (d / "bundle.json").is_file()}


def build_snapshot(plugin, ws: Path) -> Snapshot:
    from jaato_server.shared.plugins.references.links import LinkIndex

    by_id, file_ids, bundle_of = {}, {}, {}
    for d, name in _bundle_dirs(ws).items():
        for f in sorted(Path(d).glob("*.json")):
            if f.name == "bundle.json":
                continue
            src = _parse_one(plugin, f, name, plugin._project_root)
            by_id[src.id] = src
            file_ids[str(f)] = src.id
            bundle_of[str(f)] = name
    return Snapshot(by_id, file_ids, bundle_of, LinkIndex(by_id.values()))


def derive(plugin, prev: Snapshot, stamps_before: Dict[str, Any],
           stamps_now: Dict[str, Any], dirs: Dict[str, str],
           selected: List[str]) -> Tuple[Snapshot, Dict[str, Any]]:
    """Build G+1 from G: re-parse what moved, share the rest."""
    from jaato_server.shared.plugins.references.links import LinkIndex

    moved = {p for p, s in stamps_now.items() if stamps_before.get(p) != s}
    by_id = dict(prev.by_id)
    file_ids = dict(prev.file_ids)
    bundle_of = dict(prev.bundle_of)
    touched_bundles = set()
    added = set()
    for d in (p for p in moved if p in dirs):            # a listing changed
        present = {str(f) for f in Path(d).glob("*.json") if f.name != "bundle.json"}
        known = {p for p, b in prev.bundle_of.items() if b == dirs[d]}
        for gone in known - present:
            by_id.pop(file_ids.pop(gone), None)
            bundle_of.pop(gone)
        added |= present - known
        touched_bundles.add(dirs[d])
    changed_files = {p for p in moved if p in file_ids} | added
    for p in changed_files:
        bundle = prev.bundle_of.get(p) or dirs[str(Path(p).parent)]
        old = file_ids.get(p)
        src = _parse_one(plugin, Path(p), bundle, plugin._project_root)
        if old and old != src.id:
            by_id.pop(old, None)
        by_id[src.id] = src
        file_ids[p] = src.id
        bundle_of[p] = bundle
        touched_bundles.add(bundle)
    nxt = Snapshot(by_id, file_ids, bundle_of, LinkIndex(by_id.values()))
    kept = [i for i in selected if i in by_id]
    relink = [i for i in kept if prev.by_id[i].links != by_id[i].links]
    return nxt, {"reparsed": len(changed_files), "bundles": len(touched_bundles),
                 "dropped_selected": len(selected) - len(kept), "relinked_selected": len(relink)}


# ---------------------------------------------------------------- part 1+2


def _apply(ws: Path, case: str, count: int, step: int) -> None:
    refs = ws / ".jaato" / "references"
    files = sorted(p for p in refs.rglob("ref-*.json"))
    if case.startswith("edit"):
        stride = max(1, len(files) // count)
        for f in (files[(step * 211 + k * stride) % len(files)] for k in range(count)):
            data = json.loads(f.read_text(encoding="utf-8"))
            data["description"] += f" (edit {step})"
            f.write_text(json.dumps(data), encoding="utf-8")
    elif case == "add1":
        src = json.loads(files[0].read_text(encoding="utf-8"))
        src["id"] = f"added-{step}"
        (files[0].parent / f"added-{step}.json").write_text(json.dumps(src), encoding="utf-8")
    elif case == "delete1":
        files[-1 - step].unlink()


def measure(ws: Path) -> Dict[str, Any]:
    from jaato_server.shared.plugins.references import catalog_watch
    from jaato_server.shared.plugins.references.links import LinkIndex
    from jaato_server.shared.plugins.references.plugin import create_plugin

    out: Dict[str, Any] = {}
    plugin = create_plugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "tags_only"})
    plugin.set_workspace_path(str(ws))
    sources = plugin._sources
    n = len(sources)
    out["loaded_sources"] = n

    def parse_half():
        plugin._store.reload()
    _, dt = _timed(parse_half)
    out["load_parse_s"] = round(dt, 2)
    _, dt = _timed(plugin._reload_from_disk)
    out["load_full_reload_s"] = round(dt, 2)
    sources = plugin._sources

    # Indexes over the list the plugin already holds.
    def indexes():
        by_id = {s.id: s for s in sources}
        by_bundle: Dict[str, List[str]] = {}
        for s in sources:
            by_bundle.setdefault(s.bundle_name, []).append(s.id)
        return by_id, by_bundle, LinkIndex(sources)
    (by_id, _, _), dt = _timed(indexes)
    out["index_build_ms"] = round(dt * 1000, 1)
    tracemalloc.start()
    held = indexes()
    out["index_mem_mb"] = round(tracemalloc.get_traced_memory()[0] / 2**20, 1)
    tracemalloc.stop()
    del held

    target = sources[n // 2].id
    lin = [_timed(lambda: next(s for s in sources if s.id == target))[1] for _ in range(20)]
    out["lookup_linear_us"] = round(statistics.median(lin) * 1e6, 1)
    dct = [_timed(by_id.get, target)[1] for _ in range(20)]
    out["lookup_dict_us"] = round(statistics.median(dct) * 1e6, 2)

    # Step 2.
    snap, dt = _timed(build_snapshot, plugin, ws)
    out["prototype_load_s"] = round(dt, 2)
    out["prototype_ids_match"] = set(snap.by_id) == {s.id for s in sources}
    dirs = _bundle_dirs(ws)
    selected = sorted(snap.by_id)[:: max(1, len(snap.by_id) // 25)][:25]
    for step, (case, count) in enumerate(CASES):
        g = plugin._store.current(refresh=False)
        watched = list(plugin._snapshot.watch)
        before = catalog_watch.stat_paths(watched)
        _apply(ws, case, count, step)
        now, dt = _timed(catalog_watch.stat_paths, watched)
        out["detect_ms"] = round(dt * 1000, 1)
        runs = [_timed(derive, plugin, snap, before, now, dirs, selected) for _ in range(5)]
        nxt, info = runs[-1][0]
        out[f"derive_{case}_ms"] = round(statistics.median(r[1] for r in runs) * 1000, 1)
        out[f"derive_{case}_info"] = info
        tracemalloc.start()
        again, _ = derive(plugin, snap, before, now, dirs, selected)
        out[f"derive_{case}_mem_kb"] = round(tracemalloc.get_traced_memory()[0] / 1024)
        tracemalloc.stop()
        del again
        store_runs = [_timed(plugin._store._derive, g) for _ in range(5)]
        stored = store_runs[-1][0]
        out[f"store_derive_{case}_ms"] = round(statistics.median(r[1] for r in store_runs) * 1000, 1)
        del store_runs
        _, dt = _timed(plugin._reload_from_disk)
        out[f"full_reload_{case}_s"] = round(dt, 2)
        full = {s.id: s.to_dict() for s in plugin._sources}
        out[f"derive_{case}_equal"] = full == {i: s.to_dict() for i, s in nxt.by_id.items()}
        out[f"store_derive_{case}_equal"] = full == {i: s.to_dict() for i, s in stored.by_id.items()}
        del stored
        snap = nxt
    return out


# ---------------------------------------------------------------- part 3


PROFILE = """\
name: {name}
description: references snapshot benchmark session
model: echo
provider: echo
plugins: {plugins}
plugin_configs:
  echo:
    usage: {{prompt_tokens: 10, output_tokens: 5, total_tokens: 15}}
  references:
    lookup_strategy: tags_only
"""


async def daemon_part(ws: Path, sessions: int) -> Dict[str, Any]:
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.conformance.daemon import ConformanceDaemon
    from jaato_sdk.events import ClientType

    out: Dict[str, Any] = {}
    with ConformanceDaemon(ws) as daemon:
        pid = str(daemon._proc.pid)
        client = IPCClient(daemon.socket_path, client_type=ClientType.API, auto_start=False,
                           workspace_path=str(ws), env_file=str(ws / ".env"))
        if not await client.connect(timeout=60):
            raise RuntimeError("could not connect to the private daemon")
        try:
            await asyncio.sleep(5)
            out["daemon_rss_idle_mb"] = round(_rss_mb(pid), 1)
            for profile in ("snap-norefs", "snap-refs"):
                rows = []
                for _ in range(sessions):
                    t = time.perf_counter()
                    await client.create_session(profile=profile, timeout=600)
                    await asyncio.sleep(2)
                    rows.append({"session_new_s": round(time.perf_counter() - t, 2),
                                 "daemon_rss_mb": round(_rss_mb(pid), 1)})
                out[profile] = rows
        finally:
            await client.disconnect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=["snapshot", "daemon"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    ap.add_argument("--size", type=int, default=10000)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        print(json.dumps(measure(args.child)))
        return 0

    if args.part == "daemon":
        ws = args.root / f"snapws-{args.size}"
        home = args.root / f"snaphome-{args.size}"
        home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(home)
        os.environ.pop("REFERENCES_CONFIG_PATH", None)
        os.environ.setdefault("JAATO_RUNNER_POOL_SIZE", "1")
        row: Dict[str, Any] = {"n": args.size}
        if not (ws / ".jaato" / "references").exists():
            row.update(generate(ws, args.size, per_bundle=1000, mentions=3,
                                max_links=3, body_kb=3, seed=1145))
        (ws / ".env").touch()
        prof = ws / ".jaato" / "profiles"
        prof.mkdir(parents=True, exist_ok=True)
        (prof / "snap-refs.yaml").write_text(
            PROFILE.format(name="snap-refs", plugins="[references]"), encoding="utf-8")
        (prof / "snap-norefs.yaml").write_text(
            PROFILE.format(name="snap-norefs", plugins="[todo]"), encoding="utf-8")
        past = time.time() - AGE_SECONDS
        for dirpath, dirnames, filenames in os.walk(ws):
            for name in filenames + dirnames:
                os.utime(os.path.join(dirpath, name), (past, past))
        row.update(asyncio.run(daemon_part(ws, args.sessions)))
        print(json.dumps(row, indent=2))
        if args.out:
            args.out.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return 0

    rows = []
    for n in args.sizes:
        # A copy per run: the derive cases edit, add and delete files.
        src = args.root / f"ws-{n}"
        ws = args.root / f"snapws-{n}"
        if not ws.exists():
            if src.exists():
                subprocess.run(["cp", "-a", str(src), str(ws)], check=True)
            else:
                generate(ws, n, per_bundle=1000, mentions=3, max_links=3,
                         body_kb=3, seed=1145)
        home = args.root / f"snaphome-{n}"
        home.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HOME": str(home)}
        env.pop("REFERENCES_CONFIG_PATH", None)
        proc = subprocess.run([sys.executable, __file__, "snapshot", "--root", str(args.root),
                               "--child", str(ws)], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
            return proc.returncode
        row = {"n": n, **json.loads(proc.stdout.strip().splitlines()[-1])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
